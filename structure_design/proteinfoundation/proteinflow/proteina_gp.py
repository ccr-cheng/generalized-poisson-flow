"""GPFlow training, validation, and protein generation."""

import os
import random
from contextlib import contextmanager
from functools import partial
from math import prod
from typing import Dict, Literal

import lightning as L
import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.spatial.transform import Rotation
from transformers import AutoTokenizer, EsmForProteinFolding

from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.metrics.designability import proteinmpnn_device_index, scRMSD
from proteinfoundation.nn.motif_factory import SingleMotifFactory
from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3
from proteinfoundation.utils.coors_utils import ang_to_nm, trans_nm_to_atom37
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

tokenizer = None
esm_model = None


def sample_uniform_rotation(
        shape=tuple(), dtype=None, device=None
) -> torch.Tensor:
    """
    Samples rotations distributed uniformly.

    Args:
        shape: tuple (if empty then samples single rotation)
        dtype: used for samples
        device: torch.device

    Returns:
        Uniformly samples rotation matrices [*shape, 3, 3]
    """
    return torch.tensor(
        Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


class ProteinaGP(L.LightningModule):
    """Implement GPFlow losses, training, validation, and generation."""

    def __init__(self, cfg_exp, store_dir=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_exp = cfg_exp
        self.store_dir = store_dir if store_dir is not None else "./tmp"
        self.nflops = 0
        self.nsamples_processed = 0
        self.inf_cfg = None
        self.nn_ag = None

        # Define flow matcher
        self.motif_conditioning = cfg_exp.training.get("motif_conditioning", False)
        self.fm = R3NFlowMatcher(zero_com=not self.motif_conditioning, scale_ref=1.0)  # Work in nm
        if self.motif_conditioning:
            self.motif_conditioning_sequence_rep = cfg_exp.training.get("motif_conditioning_sequence_rep", False)
            if self.motif_conditioning_sequence_rep:
                if "motif_sequence_mask" not in cfg_exp.model.nn.feats_init_seq:
                    cfg_exp.model.nn.feats_init_seq.append("motif_sequence_mask")
                if "motif_x1" not in cfg_exp.model.nn.feats_init_seq:
                    cfg_exp.model.nn.feats_init_seq.append("motif_x1")
                if "motif_aa1" not in cfg_exp.model.nn.feats_init_seq:
                    cfg_exp.model.nn.feats_init_seq.append("motif_aa1")

            self.motif_conditioning_pair_rep = cfg_exp.training.get("motif_conditioning_pair_rep", False)
            if self.motif_conditioning_pair_rep:
                if "motif_structure_mask" not in cfg_exp.model.nn.feats_pair_repr:
                    cfg_exp.model.nn.feats_pair_repr.append("motif_structure_mask")
                if "motif_x1_pair_dists" not in cfg_exp.model.nn.feats_pair_repr:
                    cfg_exp.model.nn.feats_pair_repr.append("motif_x1_pair_dists")
            self.motif_factory = SingleMotifFactory(motif_prob=cfg_exp.training.get("motif_prob", 1.0))

        # Neural network
        self.nn = ProteinTransformerAF3(**cfg_exp.model.nn)

        self.nparams = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=self.cfg_exp.opt.lr
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except:
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

    def extract_clean_sample(self, batch):
        """
        Extracts clean sample, mask, batch size, protein length n, and dtype from the batch.
        Applies augmentations if those are required.

        Args:
            batch: batch from dataloader.

        Returns:
            Tuple (x_1, mask, batch_shape, n, dtype)
        """
        x_1 = batch["coords"][:, :, 1, :]  # [b, n, 3]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        if self.cfg_exp.model.augmentation.global_rotation:
            # CAREFUL: If naug_rot is > 1 this increases "batch size"
            x_1, mask = self.apply_random_rotation(
                x_1, mask, naug=self.cfg_exp.model.augmentation.naug_rot
            )
        batch_shape = x_1.shape[:-2]
        n = x_1.shape[-2]
        return (
            ang_to_nm(x_1),
            mask,
            batch_shape,
            n,
            x_1.dtype,
        )  # Since we work in nm throughout

    def apply_random_rotation(self, x, mask, naug=1):
        """
        Applies random rotation augmentation. Each sample in the batch may receive more than one augmentation,
        specified by the parameters naug. If naug > 1 this is basically increaseing the batch size from b to
        naug * b. This should likely be implemented in the dataloaders.

        Args:
            - x: Data batch, shape [b, n, 3]
            - mask: Binary, shape [b, n]
            - naug: Number of augmentations to apply to each sample, effectively increasing batch size if >1.

        Returns:
            Augmented samples and mask, shapes [b * naug, n, 3] and [B * naug, n].
        """
        assert (
                x.ndim == 3
        ), f"Augmetations can only be used for simple (x_1) batches [b, n, 3], current shape is {x.shape}"
        assert (
                mask.ndim == 2
        ), f"Augmetations can only be used for simple (mask) batches [b, n], current shape is {mask.shape}"
        assert naug >= 1, f"Number of augmentations (int) should >= 1, currently {naug}"

        # Repeat for multiple augmentations per sample
        x = x.repeat([naug, 1, 1])  # [naug * b, n, 3]
        mask = mask.repeat([naug, 1])  # [naug * b, n]

        # Sample and apply rotations
        rots = sample_uniform_rotation(
            shape=x.shape[:-2], dtype=x.dtype, device=x.device
        )  # [naug * b, 3, 3]
        x_rot = torch.matmul(x, rots)
        return self.fm._mask_and_zero_com(x_rot, mask), mask

    def sample_t(self, t_cfg, shape):
        if t_cfg.name == "uniform":
            t_max = t_cfg.p2
            return torch.rand(shape, device=self.device) * t_max  # [*]
        elif t_cfg.name == "square":
            t_min = t_cfg.p1
            t_max = t_cfg.p2
            return torch.rand(shape, device=self.device).square() * (t_max - t_min) + t_min  # [*]
        elif t_cfg.name == "sqrt":
            t_min = t_cfg.p1
            t_max = t_cfg.p2
            return torch.rand(shape, device=self.device).sqrt() * (t_max - t_min) + t_min  # [*]
        elif t_cfg.name == "logit-normal":
            mean = t_cfg.p1
            std = t_cfg.p2
            noise = torch.randn(shape, device=self.device) * std + mean  # [*]
            return torch.nn.functional.sigmoid(noise)  # [*]
        elif t_cfg.name == "beta":
            p1 = t_cfg.p1
            p2 = t_cfg.p2
            dist = torch.distributions.beta.Beta(p1, p2)
            return dist.sample(shape).to(self.device)
        elif t_cfg.name == "mix_up02_beta":
            p1 = t_cfg.p1
            p2 = t_cfg.p2
            dist = torch.distributions.beta.Beta(p1, p2)
            samples_beta = dist.sample(shape).to(self.device)
            samples_uniform = torch.rand(shape, device=self.device)
            u = torch.rand(shape, device=self.device)
            return torch.where(u < 0.02, samples_uniform, samples_beta)
        elif t_cfg.name == "uniform_max":
            t_max = t_cfg.p2
            u = (torch.rand(shape, device=self.device) / t_max).clamp(max=1)
            return u
        elif t_cfg.name == "sqrt_max":
            t_min = t_cfg.p1
            t_max = t_cfg.p2
            u = (torch.rand(shape, device=self.device) / t_max).sqrt().clamp(max=1, min=t_min)
            return u
        else:
            raise NotImplementedError(
                f"Sampling mode for t {t_cfg.name} not implemented"
            )

    def t_weight(self, t_cfg, t):
        if t_cfg.name == "square":
            return 2 * t.sqrt() / (1 - t)
        elif t_cfg.name == "sqrt_max":
            t_max = t_cfg.p2
            return 1 / (1 - t).clamp(min=1e-3) / t / t_max / 2
        elif t_cfg.name == "uniform_max":
            t_max = t_cfg.p2
            return 1 / (1 - t).clamp(min=1e-3) / t_max
        return 1 / (1 - t)

    def _nn_out_to_x_clean(self, nn_out, batch):
        """
        Transforms the output of the nn to a clean sample prediction. The transformation depends on the
        parameterization used. For now we admit x_1 or v.

        Args:
            nn_out: Dictionary, nerual network output
                - "coords_pred": Tensor of shape [b, n, 3], could be the clean sample or the velocity
                - "pair_pred" (Optional): Tensor of shape [b, n, n, num_buckets_predict_pair], could be the clean sample or the velocity
            batch: Dictionary, batch of data

        Returns:
            Clean sample prediction, tensor of shape [b, n, 3].
        """
        nn_pred = nn_out["coors_pred"]
        t = batch["t"]  # [*]
        t_ext = t[..., None, None]  # [*, 1, 1]
        x_t = batch["x_t"]  # [*, n, 3]
        if self.cfg_exp.model.target_pred == "x_1":
            x_1_pred = nn_pred
        elif self.cfg_exp.model.target_pred == "v":
            x_1_pred = x_t + (1.0 - t_ext) * nn_pred
        else:
            raise IOError(
                f"Wrong parameterization chosen: {self.cfg_exp.model.target_pred}"
            )
        return x_1_pred

    def predict_clean(
            self,
            batch: Dict,
    ):
        """
        Predicts clean samples given noisy ones and time.

        Args:
            batch: a batch of data with some additions, including
                - "x_t": Type depends on the mode (see beluw, "returns" part)
                - "t": Time, shape [*]
                - "mask": Binary mask of shape [*, n]
                - "x_sc" (optional): Prediction for self-conditioning
                - Other features from the dataloader.

        Returns:
            Predicted clean sample, depends on the "modality" we're in.
                - For frameflow it returns a dictionary with keys "trans" and "rot", and values
                tensors of shape [*, n, 3] and [*, n, 3, 3] respectively,
                - For CAflow it returns a tensor of shape [*, n, 3].
            Other things predicted by nn (pair_pred for distogram loss)
        """
        nn_out = self.nn(batch)  # [*, n, 3]
        return self._nn_out_to_x_clean(nn_out, batch), nn_out  # [*, n, 3]

    def predict_clean_n_v_w_guidance(
            self,
            batch: Dict,
            guidance_weight: float = 1.0,
            return_nn_out: bool = False,
    ):
        """Predict coordinates and their vector field, optionally with autoguidance.

        The auxiliary network must use the same coordinate parameterization as
        the main model. Predictions combine as w * main + (1 - w) * auxiliary.
        """
        if self.motif_conditioning and ("fixed_structure_mask" not in batch or "x_motif" not in batch):
            # for generation we have to pass conditioning info in. But for validation do the same as training
            batch.update(self.motif_factory(batch, zeroes=True))

        nn_out = self.nn(batch)
        x_pred = self._nn_out_to_x_clean(nn_out, batch)

        if guidance_weight != 1.0:
            if self.nn_ag is None:
                raise ValueError("Autoguidance requires an auxiliary network when guidance_weight != 1.0.")
            nn_out_ag = self.nn_ag(batch)
            x_pred_ag = self._nn_out_to_x_clean(nn_out_ag, batch)
            x_pred = guidance_weight * x_pred + (1 - guidance_weight) * x_pred_ag

        v = self.fm.xt_dot(x_pred, batch["x_t"], batch["t"], batch["mask"])
        if return_nn_out:
            return x_pred, v, nn_out
        return x_pred, v

    def compute_fm_loss(
            self,
            x_1: torch.Tensor,
            x_1_pred: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor,
            mask: torch.Tensor,
            log_prefix: str,
    ) -> torch.Tensor:
        """
        Computes and logs flow matching loss.

        Args:
            x_1: True clean sample, shape [*, n, 3].
            x_1_pred: Predicted clean sample, shape [*, n, 3].
            x_t: Sample at interpolation time t (used as input to predict clean sample), shape [*, n, 3].
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, nres].

        Returns:
            Flow matching loss.
        """
        nres = torch.sum(mask, dim=-1).clamp(min=1) * 3  # [*]

        err = (x_1 - x_1_pred) * mask[..., None]  # [*, n, 3]
        loss = torch.sum(err ** 2, dim=(-1, -2)) / nres  # [*]

        total_loss_w = 1.0 / ((1.0 - t) ** 2 + 1e-5)

        loss = loss * total_loss_w  # [*]
        if log_prefix:
            self.log(
                f"{log_prefix}/trans_loss",
                torch.mean(loss),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )
        return loss

    def compute_poisson_loss(
            self, ins_rate, ins_coord, ins_mask, x_t, t, mask, log_prefix=""
    ):
        """
        Computes the generalized Poisson loss.
        :param ins_rate: Insertion rate predicted by the model, [B, L + 1].
        :param ins_coord: Insertion coordinates predicted by the model, [B, L + 1, 3].
        :param ins_mask: Insertion mask indicating tokens deleted from x_t, [B, L].
        :param x_t: Noisy coordinates at time t, [B, L, 3].
        :param t: Timestep, [B,].
        :param mask: Sequence mask, [B, L].
        :param log_prefix: Prefix for logging.
        :return: Insertion rate loss and insertion coordinate loss.
        """
        B, L = ins_mask.shape
        device = ins_mask.device
        ins_mask = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.bool), ins_mask], dim=1)  # [B, L + 1]
        keep_mask = ~ins_mask  # [B, L + 1]
        keep_mask[:, 1:] &= mask  # only keep valid tokens
        seq_lens = mask.sum(-1).clamp(min=1).float()  # [B,]
        idx = keep_mask.cummax(-1).indices
        ins_rate_expand = torch.gather(ins_rate, 1, idx)  # [B, L + 1]
        weight = self.t_weight(self.cfg_exp.loss.t_edit_distribution, t).unsqueeze(-1)  # [B, 1]
        ins_loss = (
                (ins_rate * keep_mask.float()).sum(-1)
                - (ins_rate_expand.clamp(min=1e-8).log() * ins_mask * weight).sum(-1)
        )  # (B,)
        # gt_rate = torch.zeros_like(ins_rate).scatter_add_(1, idx, ins_mask.float()) * weight  # [B, L + 1]
        # ins_loss = ((ins_rate - gt_rate) * keep_mask.float()).square().sum(-1)
        norm = seq_lens if self.cfg_exp.loss.norm_type == 'length' else self.cfg_exp.loss.norm_const
        ins_loss = ins_loss / norm
        ins_loss = ins_loss.mean() * self.cfg_exp.loss.ins_rate_loss_weight

        idx_expand = idx.unsqueeze(-1).expand_as(ins_coord)  # [B, L + 1, 3]
        ins_coord_expand = torch.gather(ins_coord, 1, idx_expand)  # [B, L + 1, 3]
        # The value here does not matter since ins_mask is always 0 on the first position
        x_t = torch.cat([torch.zeros(B, 1, 3, device=device, dtype=x_t.dtype), x_t], dim=1)  # [B, L + 1, 3]
        ins_coord_loss = (ins_coord_expand - x_t).square().mean(-1) * ins_mask * weight  # [B, L + 1]
        ins_coord_loss = ins_coord_loss.sum(-1) / norm  # (B,)
        ins_coord_loss = ins_coord_loss.mean() * self.cfg_exp.loss.ins_coord_loss_weight
        if log_prefix:
            log_config = {
                "on_step": True,
                "on_epoch": True,
                "prog_bar": True,
                "logger": True,
                "batch_size": B,
                "sync_dist": True,
                "add_dataloader_idx": False,
            }
            self.log(
                f"{log_prefix}/ins_rate",
                ins_rate[keep_mask].mean(),
                **log_config
            )
            self.log(
                f"{log_prefix}/ins_rate_loss",
                ins_loss,
                **log_config
            )
            self.log(
                f"{log_prefix}/ins_coord_loss",
                ins_coord_loss,
                **log_config
            )
        return ins_loss, ins_coord_loss

    def training_step(self, batch, batch_idx):
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batches.
        """
        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"

        # Extract inputs from batch (our dataloader)
        # This may apply augmentations, if requested in the config file
        x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
        batch["aa_1"] = batch["residue_type"]  # ground truth amino acid types

        # Center and mask input
        x_1 = self.fm._mask_and_zero_com(x_1, mask)

        # Sample time, reference and align reference to target
        t = self.sample_t(self.cfg_exp.loss.t_distribution, batch_shape)
        x_0 = self.fm.sample_reference(
            n=n, shape=batch_shape, device=self.device, dtype=dtype, mask=mask
        )

        if self.motif_conditioning:
            # Use the augmented coordinates and let the factory inpaint the reference.
            batch.update({"x_0": x_0, "x_1": x_1, "mask": mask})
            batch.update(self.motif_factory(batch, inpaint_x0=True))
            x_0, x_1 = batch["x_0"], batch["x_1"]

        # Interpolation
        x_t = self.fm.interpolate(x_0, x_1, t)

        # Preserve the unknown-label input for legacy fold embeddings.
        batch.pop("cath_code", None)

        B, L = x_1.size()[:2]
        t_edit = self.sample_t(self.cfg_exp.loss.t_edit_distribution, batch_shape)
        # tokens to be deleted from x_1
        ins_mask = (torch.rand(B, L, dtype=t_edit.dtype, device=t_edit.device) > t_edit.unsqueeze(-1)) & mask
        if self.motif_conditioning:
            ins_mask &= ~batch["fixed_sequence_mask"]
        new_mask = mask & ~ins_mask  # tokens kept in x_t
        # Add a few things to batch, needed for nn
        batch.update({
            "x_t": x_t,
            "mask": new_mask,
            "t": t,
            "t_edit": t_edit,
        })
        # Prediction for self-conditioning
        if random.random() > 0.5 and self.cfg_exp.training.self_cond:
            x_pred_sc, _ = self.predict_clean(batch)
            batch["x_sc"] = x_pred_sc.detach()

        x_1_pred, nn_out = self.predict_clean(batch)
        ins_rate, ins_coord = nn_out["ins_rate"], nn_out["ins_coord"]

        # Compute losses
        fm_loss = self.compute_fm_loss(
            x_1, x_1_pred, x_t, t, new_mask, log_prefix=log_prefix
        ).mean()  # [*]
        ins_loss, ins_coord_loss = self.compute_poisson_loss(
            ins_rate=ins_rate, ins_coord=ins_coord, ins_mask=ins_mask, x_t=x_t, t=t_edit, mask=mask,
            log_prefix=log_prefix,
        )
        train_loss = fm_loss + ins_loss + ins_coord_loss

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )

        # Don't log if validation step (indicated by batch_id)
        if not val_step:
            self.log(
                f"train_loss",
                train_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )

            # For scaling laws
            b, n = mask.shape
            nflops_step = None
            log_config = {
                "on_step": True,
                "on_epoch": False,
                "prog_bar": False,
                "logger": True,
                "batch_size": 1,
                "sync_dist": True,
            }
            if nflops_step is not None:
                self.nflops = (
                        self.nflops + nflops_step * self.trainer.world_size
                )  # Times number of processes so it logs sum across devices
                self.log(
                    "scaling/nflops",
                    self.nflops * 1.0,
                    **log_config
                )

            self.nsamples_processed = (
                    self.nsamples_processed + b * self.trainer.world_size
            )
            self.log(
                "scaling/nsamples_processed",
                self.nsamples_processed * 1.0,
                **log_config
            )

            self.log(
                "scaling/nparams",
                self.nparams * 1.0,
                **log_config
            )
            # Constant line but ok, easy to compare # params

        return train_loss

    def validation_step(self, batch, batch_idx):
        """Evaluate and log the training losses on a validation batch."""
        with torch.no_grad():
            self.training_step(batch, batch_idx=-1)

    def on_validation_epoch_end(self):
        """Shard a global sample budget and log sample-weighted metrics across ranks."""
        if self.trainer.sanity_checking:
            return
        nsamples = self.cfg_exp.get("validation", {}).get("nsamples", 16)
        if isinstance(nsamples, bool) or not isinstance(nsamples, int) or nsamples < 0:
            raise ValueError("validation.nsamples must be a non-negative integer")
        if nsamples == 0:
            return

        sample_ids = range(self.global_rank, nsamples, self.trainer.world_size)
        local_count = len(sample_ids)
        # Empty ranks must still log each metric and participate in its reduction.
        # Float counts avoid integer truncation when ranks have unequal workloads.
        metric_sums = torch.zeros(5, device=self.device, dtype=torch.float64)
        if local_count:
            with self._validation_rng(), torch.no_grad():
                x, lengths = self._predict(batch_nsamples=local_count)
                predictions = self.samples_to_atom37(x).detach().cpu().numpy()
                lengths = lengths.detach().cpu().tolist()
                sample_root_path = os.path.join(self.store_dir, f"step{self.global_step}_samples")
                os.makedirs(sample_root_path, exist_ok=True)
                eval_tokenizer, eval_model = self._load_esm()
                device_idx = self._proteinmpnn_device_index()
                plddts, rmsds, paes = [], [], []
                for local_idx, sample_id in enumerate(sample_ids):
                    pdb_path = os.path.join(sample_root_path, f"{sample_id}.pdb")
                    # ProteinMPNN/ESMFold use fixed artifact names within tmp_path.
                    eval_path = os.path.join(sample_root_path, str(sample_id))
                    try:
                        os.makedirs(eval_path, exist_ok=True)
                        write_prot_to_pdb(
                            predictions[local_idx, :lengths[local_idx]],
                            pdb_path,
                            overwrite=True,
                            no_indexing=True,
                        )
                        res_rmsd, res_plddt, res_pae = scRMSD(
                            pdb_path, ret_min=False, tmp_path=eval_path,
                            num_seq_per_target=self.inf_cfg.designability_seqs_per_struct,
                            tokenizer=eval_tokenizer, esm_model=eval_model, device_idx=device_idx,
                        )
                        # Compute all three before appending so failures keep equal counts.
                        plddt, rmsd, pae = max(res_plddt) * 100, min(res_rmsd), min(res_pae)
                    except Exception as e:
                        logger.warning(
                            f"Error evaluating validation sample {sample_id} on rank {self.global_rank}: {e}"
                        )
                        plddt, rmsd, pae = 0., 30., 30.
                    plddts.append(plddt)
                    rmsds.append(rmsd)
                    paes.append(pae)
                metric_sums = torch.tensor(
                    [local_count, sum(lengths), sum(plddts), sum(rmsds), sum(paes)],
                    device=self.device, dtype=torch.float64,
                )

        # Reduce sums and counts explicitly: Lightning's mean reduction of integer
        # batch-size counters can truncate uneven shard sizes (including to zero).
        metric_sums = self.trainer.strategy.reduce(metric_sums, reduce_op="sum")
        global_means = metric_sums[1:] / metric_sums[0]
        for name, value in zip(("length", "plddt", "rmsd", "pae"), global_means):
            self.log(
                f"sample/{name}", value,
                on_step=False, on_epoch=True, batch_size=1,
                prog_bar=False, logger=True, sync_dist=True,
            )

    @contextmanager
    def _validation_rng(self):
        """Give ranks distinct sampling RNGs without changing training randomness."""
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cuda_devices = [self.device.index] if self.device.type == "cuda" else []
        seed = (
            self.cfg_exp.get("seed", 42) + self.global_step * self.trainer.world_size + self.global_rank
        ) % (2 ** 32)
        with torch.random.fork_rng(devices=cuda_devices):
            try:
                random.seed(seed)
                np.random.seed(seed)
                torch.random.default_generator.manual_seed(seed)
                if cuda_devices:
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(seed)
                yield
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

    def _proteinmpnn_device_index(self):
        """Select this rank's GPU for a subprocess whose only device is cuda:0."""
        return proteinmpnn_device_index(self.device)

    def configure_inference(self, inf_cfg, nn_ag):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        if inf_cfg.get("guidance_weight", 1.0) != 1.0 and nn_ag is None:
            raise ValueError("Autoguidance requires an auxiliary network when guidance_weight != 1.0.")
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag

    def _predict(
            self, batch_nsamples=None, x_motif=None, aa_motif=None,
            fixed_sequence_mask=None, fixed_structure_mask=None,
            contig_types=None, contig_lengths=None
    ):
        sampling_args = self.inf_cfg.sampling_caflow
        guidance_weight = self.inf_cfg.get("guidance_weight", 1.0)
        batch_nsamples = batch_nsamples or self.inf_cfg.nsamples

        return self.generate(
            nsamples=batch_nsamples,
            max_length=self.inf_cfg.max_length,
            dt=self.inf_cfg.dt,
            self_cond=self.inf_cfg.self_cond,
            guidance_weight=guidance_weight,
            dtype=torch.float32,
            schedule_mode=self.inf_cfg.schedule.schedule_mode,
            schedule_p=self.inf_cfg.schedule.schedule_p,
            schedule_edit_mode=self.inf_cfg.schedule.schedule_edit_mode,
            schedule_edit_p=self.inf_cfg.schedule.schedule_edit_p,
            sampling_mode=sampling_args["sampling_mode"],
            sc_scale_noise=sampling_args["sc_scale_noise"],
            sc_scale_score=sampling_args["sc_scale_score"],
            gt_mode=sampling_args["gt_mode"],
            gt_p=sampling_args["gt_p"],
            gt_clamp_val=sampling_args["gt_clamp_val"],
            scale_rate=sampling_args.get("scale_rate", 1.0),
            tau_leap=sampling_args.get("tau_leap", False),
            x_motif=x_motif,
            aa_motif=aa_motif,
            fixed_sequence_mask=fixed_sequence_mask,
            fixed_structure_mask=fixed_structure_mask,
            contig_types=contig_types,
            contig_lengths=contig_lengths,
        )

    def generate(
            self,
            nsamples: int,
            max_length: int,
            dt: float,
            self_cond: bool,
            guidance_weight: float = 1.0,
            dtype: torch.dtype = None,
            schedule_mode: str = "uniform",
            schedule_p: float = 1.0,
            schedule_edit_mode: str = "uniform",
            schedule_edit_p: float = 1.0,
            sampling_mode: str = "sc",
            sc_scale_noise: float = "1.0",
            sc_scale_score: float = "1.0",
            gt_mode: Literal["us", "tan"] = "us",
            gt_p: float = 1.0,
            gt_clamp_val: float = None,
            scale_rate: float = 1.0,
            x_motif=None,
            aa_motif=None,
            fixed_sequence_mask=None,
            fixed_structure_mask=None,
            contig_types=None,
            contig_lengths=None,
            tau_leap: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate variable-length samples, optionally using Poisson tau-leaping.
        """
        predict_clean_n_v_w_guidance = partial(
            self.predict_clean_n_v_w_guidance,
            guidance_weight=guidance_weight,
            return_nn_out=True,
        )
        return self.fm.full_simulation_gp(
            predict_clean_n_v_w_guidance,
            dt=dt,
            nsamples=nsamples,
            max_length=max_length,
            self_cond=self_cond,
            device=self.device,
            dtype=dtype,
            schedule_mode=schedule_mode,
            schedule_p=schedule_p,
            schedule_edit_mode=schedule_edit_mode,
            schedule_edit_p=schedule_edit_p,
            sampling_mode=sampling_mode,
            sc_scale_noise=sc_scale_noise,
            sc_scale_score=sc_scale_score,
            gt_mode=gt_mode,
            gt_p=gt_p,
            gt_clamp_val=gt_clamp_val,
            scale_rate=scale_rate,
            x_motif=x_motif,
            aa_motif=aa_motif,
            fixed_sequence_mask=fixed_sequence_mask,
            fixed_structure_mask=fixed_structure_mask,
            contig_types=contig_types,
            contig_lengths=contig_lengths,
            tau_leap=tau_leap,
        )

    def samples_to_atom37(self, samples):
        """
        Transforms samples to atom37 representation.

        Args:
            samples: Tensor of shape [b, n, 3]

        Returns:
            Samples in atom37 representation, shape [b, n, 37, 3].
        """
        return trans_nm_to_atom37(samples)  # [b, n, 37, 3]

    def on_predict_start(self) -> None:
        seed = self.inf_cfg.seed + self.trainer.global_rank
        L.seed_everything(seed)
        logger.info(f"Seeding rank {self.trainer.global_rank} with seed {seed}")

    def _load_esm(self):
        global tokenizer, esm_model

        is_cluster_run = os.environ.get("SLURM_JOB_ID") is not None
        cache_dir = None
        if is_cluster_run:
            cache_dir = os.environ.get("CACHE_DIR")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "facebook/esmfold_v1", cache_dir=cache_dir
            )
        if esm_model is None:
            logger.info("Loading ESMFold model...")
            esm_model = EsmForProteinFolding.from_pretrained(
                "facebook/esmfold_v1", cache_dir=cache_dir
            )
            esm_model = esm_model.to(self.device)
        return tokenizer, esm_model

    def predict_step(self, batch, batch_idx):
        if batch.get("motif_structure") is not None:
            nsamples = batch['nsamples']
            fixed_sequence_mask = batch['motif_seq_mask'].to(self.device)
            fixed_structure_mask = fixed_sequence_mask[:, :, None] * fixed_sequence_mask[:, None, :]

            x_motif = batch['motif_structure'].to(self.device)
            contig_types = batch['contig_types']
            contig_lengths = batch['contig_lengths']
            aa_motif = batch['aa_motif']
            motif_input = {
                'batch_nsamples': nsamples,
                'x_motif': x_motif,
                'fixed_sequence_mask': fixed_sequence_mask,
                'fixed_structure_mask': fixed_structure_mask,
                'contig_types': contig_types,
                'contig_lengths': contig_lengths,
                'aa_motif': aa_motif,
            }

            x, lengths, contig_types, contig_lengths = self._predict(**motif_input)
            predictions = self.samples_to_atom37(x).detach().cpu()  # [b, n, 37, 3]
            predictions = [
                predictions[i, :lengths[i]].numpy() for i in range(predictions.shape[0])
            ]  # list of [n, 37, 3]
            return predictions, None, lengths, contig_types, contig_lengths

        # Unconditional generation
        x, lengths = self._predict()
        predictions = self.samples_to_atom37(x).detach().cpu()  # [b, n, 37, 3]
        predictions = [predictions[i, :lengths[i]] for i in range(predictions.shape[0])]  # list of [n, 37, 3]

        cfg = self.inf_cfg
        root_path = os.path.join("./inference", cfg.run_name)
        # Code for designability and
        # Store samples generated as pdbs and also scRMSD

        # Add some columns to store per-sample results
        columns = ["pdb_path", "L"]
        if cfg.compute_designability:
            columns += [
                "scRMSD", "scRMSD_all",
                "pLDDT", "pLDDT_all",
                "pAE", "pAE_all",
            ]
        results = []
        # Save each generation as a pdb file
        for i, p in enumerate(predictions):
            # Create directory where everything related to this sample will be stored
            n = p.size(0)
            dir_name = f"b{batch['index'][0]}_i{i}"
            sample_root_path = os.path.join(
                root_path, dir_name
            )  # ./inference/conf_{}/b{b}_i{i}
            os.makedirs(sample_root_path, exist_ok=True)

            # Save generated structure as pdb
            fname = dir_name + ".pdb"
            pdb_path = os.path.join(sample_root_path, fname)
            write_prot_to_pdb(
                p.numpy(),
                pdb_path,
                overwrite=True,
                no_indexing=True,
            )
            res_row = [pdb_path, n]

            # If needed run designability, storing all intermediate values generated in sample_root_path
            if cfg.compute_designability:
                tokenizer, esm_model = self._load_esm()
                res_rmsd, res_plddt, res_pae = scRMSD(
                    pdb_path, ret_min=False, tmp_path=sample_root_path,
                    num_seq_per_target=cfg.designability_seqs_per_struct,
                    tokenizer=tokenizer, esm_model=esm_model, device_idx=self._proteinmpnn_device_index()
                )

                res_rmsd_str = '[' + ','.join([f"{v:.3f}" for v in res_rmsd]) + ']'
                res_plddt_str = '[' + ','.join([f"{v * 100:.2f}" for v in res_plddt]) + ']'
                res_pae_str = '[' + ','.join([f"{v:.3f}" for v in res_pae]) + ']'
                res_row += [min(res_rmsd), res_rmsd_str]
                res_row += [max(res_plddt) * 100, res_plddt_str]
                res_row += [min(res_pae), res_pae_str]

                logger.info(f'scRMSD: {res_rmsd_str}')
                logger.info(f'pLDDT: {res_plddt_str}')
                logger.info(f'pAE: {res_pae_str}')

            results.append(res_row)

        # Create the dataframe with results
        df = pd.DataFrame(results, columns=columns)
        if cfg.compute_designability:
            designability = (df['scRMSD'] <= 2.0).mean() * 100
            logger.info(f'Designability: {designability:.2f}%')

        return df
