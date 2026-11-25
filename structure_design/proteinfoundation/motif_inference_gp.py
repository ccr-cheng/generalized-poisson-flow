import time
import os
import sys
from typing import Optional

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)  # Import this checkout even when launched from another directory.

import argparse

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from proteinfoundation.proteinflow.proteina_gp import ProteinaGP
from proteinfoundation.utils.checkpoint_utils import resolve_checkpoint_path
from proteinfoundation.datasets.constants import AA_TOKEN_2_ID, UNK_TOKEN_ID
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.nn.motif_factory import (
    motif_extract,
    parse_contig_from_pdb,
    contig_string_to_array,
    save_contig_info_csv,
)
from openfold.np.residue_constants import restype_3to1


class GenMotifDataset(Dataset):
    """Batch motif coordinates and contigs for variable-length scaffolding.

    Only motif residues are present initially. Linker segments start empty and
    grow during sampling; the dataset never chooses an output backbone length.
    """

    def __init__(
            self,
            nsamples: int = 1,
            max_nsamples_per_batch: int = 1,
            contig_string: Optional[str] = None,
            motif_pdb_path: Optional[str] = None,
            motif_only: bool = False,
    ):
        super().__init__()
        if nsamples < 1 or max_nsamples_per_batch < 1:
            raise ValueError("Sample count and maximum batch size must be positive.")
        if motif_pdb_path is None:
            raise ValueError("Motif scaffolding requires a motif PDB file.")
        if contig_string is None:
            with open(motif_pdb_path, 'r') as f:
                contig_string, _, _ = parse_contig_from_pdb(f.read())
        self.contig_string = contig_string
        contig_array = contig_string_to_array(contig_string)
        if not any(seg['type'] == 'motif' for seg in contig_array):
            raise ValueError("The contig must contain at least one motif segment.")

        motif = motif_extract(contig_string, str(motif_pdb_path), motif_only=motif_only)
        self.motif_structure = torch.as_tensor(motif.coord, dtype=torch.float32) / 10
        self.aa_motif = torch.tensor([
            AA_TOKEN_2_ID.get(restype_3to1.get(name, 'X'), UNK_TOKEN_ID)
            for name in motif.res_name
        ], dtype=torch.long)

        contig_types, contig_lengths = [], []
        for seg in contig_array:
            is_motif = seg['type'] == 'motif'
            contig_types.append(int(is_motif))
            contig_lengths.append(seg['length'] if is_motif else 0)
        if contig_types[0] == 1:
            contig_types.insert(0, 0)
            contig_lengths.insert(0, 0)
        if contig_types[-1] == 1:
            contig_types.append(0)
            contig_lengths.append(0)
        if sum(contig_lengths) != len(motif):
            raise ValueError("Contig motif lengths must match the extracted motif residues.")
        self.contig_types = np.asarray(contig_types, dtype=np.int64)
        self.contig_lengths = np.asarray(contig_lengths, dtype=np.int64)
        self.batch_sizes = [
            min(max_nsamples_per_batch, nsamples - start)
            for start in range(0, nsamples, max_nsamples_per_batch)
        ]
        logger.info(
            f"Generating {nsamples} variable-length scaffolds in {len(self)} batches "
            f"around {len(motif)} fixed motif residues."
        )

    def __len__(self):
        return len(self.batch_sizes)

    def __getitem__(self, index: int):
        batch_size = self.batch_sizes[index]
        motif_length = self.motif_structure.shape[0]
        return {
            "nsamples": batch_size,
            "motif_seq_mask": torch.ones(batch_size, motif_length, dtype=torch.bool),
            "motif_structure": self.motif_structure.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            "aa_motif": self.aa_motif,
            "contig_types": np.tile(self.contig_types, (batch_size, 1)),
            "contig_lengths": np.tile(self.contig_lengths, (batch_size, 1)),
        }

    @staticmethod
    def collate_fn(batch):
        return batch[0]


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_motif",
        help="Name of the config yaml file.",
    )
    parser.add_argument(
        "--config_number", type=int, default=-1, help="Number of the config yaml file."
    )
    parser.add_argument(
        "--config_subdir",
        type=str,
        help="(Optional) Name of directory with config files, if not included uses base inference config.\
            Likely only used when submitting to the cluster with script.",
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=0,
        help="Leave as 0.",
    )
    args, hydra_overrides = parser.parse_known_args()
    logger.info(" ".join(sys.argv))

    assert (
        torch.cuda.is_available()
    ), "CUDA not available"  # Needed for ESMfold and designability
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )  # Send to stdout

    # Inference config
    # If config_subdir is None then use base inference config
    # Otherwise use config_subdir/some_config
    config_path = "../configs/experiment_config"

    with hydra.initialize(config_path, version_base=hydra.__version__):
        # If number provided use it, otherwise name
        if args.config_number != -1:
            config_name = f"inf_{args.config_number}"
        else:
            config_name = args.config_name
        cfg = hydra.compose(
            config_name=config_name,
            overrides=hydra_overrides,
        )
        logger.info(f"Inference config {cfg}")
        run_name = cfg.run_name

    assert (
            not cfg.compute_designability or not cfg.compute_fid
    ), "Designability cannot be computed together with FID"

    # Set root path for this inference run
    root_path = f"./inference/{cfg.run_name}"
    os.makedirs(root_path, exist_ok=True)

    dataset = GenMotifDataset(
        nsamples=cfg.nsamples,
        max_nsamples_per_batch=cfg.max_nsamples_per_batch,
        contig_string=cfg.contig_string,
        motif_pdb_path=cfg.motif_pdb_path,
        motif_only=cfg.motif_only,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=GenMotifDataset.collate_fn)
    # Load model from checkpoint
    ckpt_file = resolve_checkpoint_path(cfg.ckpt_path, cfg.get("ckpt_name"))
    logger.info(f"Using checkpoint {ckpt_file}")
    model = ProteinaGP.load_from_checkpoint(ckpt_file)

    # Set seed
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    # Set inference variables and potentially load autoguidance
    nn_ag = None
    if cfg.get("guidance_weight", 1.0) != 1.0:
        ckpt_ag_file = cfg.get("autoguidance_ckpt_path")
        if ckpt_ag_file is None:
            raise ValueError("Non-unit guidance_weight requires autoguidance_ckpt_path.")
        ckpt_ag_file = resolve_checkpoint_path(ckpt_ag_file)
        model_ag = ProteinaGP.load_from_checkpoint(ckpt_ag_file)
        nn_ag = model_ag.nn

    model.configure_inference(cfg, nn_ag=nn_ag)

    # The dataloader batch size stays 1: each item already contains a batch of
    # motif structures, masks, amino acids, and contig metadata.

    # Flatten config and use it to initialize results dataframes columns
    flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    flat_dict = pd.json_normalize(flat_cfg, sep="_").to_dict(orient="records")[0]
    flat_dict = {k: str(v) for k, v in flat_dict.items()}
    columns = list(flat_dict.keys())

    # Sample the model
    trainer = L.Trainer(accelerator="gpu", devices=1)
    predictions = trainer.predict(model, dataloader)  # list of df

    structures, lengths, contig_types, contig_lengths = [], [], [], []
    for batch_idx, (preds, _, ls, types, lens) in enumerate(predictions):
        structures.extend(preds)
        lengths.extend(ls)
        contig_types.extend(types)
        contig_lengths.extend(lens)

    save_contig_info_csv(cfg.motif_pdb_path, cfg.motif_task_name, contig_types, contig_lengths, outpath=root_path)

    for idx, struct in enumerate(structures):
        pdb_path = os.path.join(root_path, f"{idx}.pdb")
        write_prot_to_pdb(struct, pdb_path, overwrite=True, no_indexing=True)

    time.sleep(3)  # Make sure all files are written before moving on
