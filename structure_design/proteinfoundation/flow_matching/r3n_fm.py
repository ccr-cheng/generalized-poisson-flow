from typing import Callable, Literal, Optional, Tuple

import math
import numpy as np
import torch
from loguru import logger
from torch import Dict, Tensor
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from tqdm import tqdm

from proteinfoundation.datasets.constants import UNK_TOKEN_ID
from proteinfoundation.utils.align_utils.align_utils import mean_w_mask
from proteinfoundation.nn.motif_factory import contig_info_to_insert_mask


@rank_zero_only
def log_info(msg):
    logger.info(msg)


def check_mask_contiguous(mask: torch.Tensor):
    mask_int = mask.int()
    diff = mask_int[:, 1:] - mask_int[:, :-1]
    assert not (diff == 1).any(), "False -> True transition found"


def find_linker_for_gap(i, contig_type, contig_start):
    """
    contig_start[k] = start position of segment k
    contig_type[k] = 0 for linker, 1 for motif
    """
    seg_idx = None
    for k in range(len(contig_type)):
        if contig_type[k] == 0 and contig_start[k] <= i:
            seg_idx = k
    assert seg_idx is not None
    return seg_idx


def count_insertions_in_segments(ins_count_row, contig_type_row, contig_length_row):
    """Add insertion counts to their linker segments, leaving motif lengths fixed."""
    seg_starts = []
    pos = 0
    for l in contig_length_row:
        seg_starts.append(pos)
        pos += l

    assert len(seg_starts) == len(contig_length_row) == len(contig_type_row)

    new_contig_length_row = contig_length_row.copy()
    for idx, inserted in enumerate(ins_count_row.tolist()):
        if inserted:
            seg_idx = find_linker_for_gap(idx, contig_type_row, seg_starts)
            new_contig_length_row[seg_idx] += int(inserted)
    return new_contig_length_row


class R3NFlowMatcher:
    """
    Flow matching on (R^3)^n, where n is for the number of elements
    per sample (e.g. number of residues).

    We include the option of using (R_0^3)^n (centered translations).
    """

    def __init__(
            self,
            zero_com: bool = False,
            scale_ref: float = 1.0,
    ):
        self.dim = 3
        self.scale_ref = scale_ref
        self.zero_com = zero_com

    def _force_zero_com(
            self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Centers tensor over n dimension.

        Args:
            x: Tensor of shape [*, n, 3]
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Centered x = x - mean(x, dim=-2), shape [*, n, 3].
        """
        if mask is None:
            x = x - torch.mean(x, dim=-2, keepdim=True)
        else:
            x = (x - mean_w_mask(x, mask, keepdim=True)) * mask[..., None]
        return x

    def _apply_mask(
            self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Applies mask to x. Sets masked elements to zero.

        Args:
            x: Tensor of shape [*, n, 3]
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Masked x of shape [*, n, 3]
        """
        if mask is None:
            return x
        return x * mask[..., None]  # [*, n, 3]

    def _mask_and_zero_com(
            self, x, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Applies mask to and centers x if needed (if zero_com=True).

        Args:
            x: Batch of samples, batch shape *
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Masked (and possibly center) samples.
        """
        x = self._apply_mask(x, mask)
        if self.zero_com:
            x = self._force_zero_com(x, mask)
        return x

    def _extend_t(self, n: int, t: torch.Tensor) -> torch.Tensor:
        """
        Extends t shape with n. Needed to use flow matching utils.

        Args:
            n (int): Number of elements per sample (e.g. number of residues)
            t: Float vector, shape [*]

        Returns:
            Extended t vector of shape [*, n] compatible with flow matching utils.
        """
        return t[..., None].expand(t.shape + (n,))

    def interpolate(
            self,
            x_0: torch.Tensor,
            x_1: torch.Tensor,
            t: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Interpolates between rigids x_0 (base) and x_1 (data) using t.

        Args:
            x_0: Tensor sampled from reference, shape [*, n, 3]
            x_1: Tensor sampled from target, shape [*, n, 3]
            t: Interpolation times, shape [*]
            mask (optional): Binary mask, shape [*, n]

        Returns:
            x_t: Interpolated tensor, shape [*, n, 3]
        """
        x_0, x_1 = map(
            lambda args: self._mask_and_zero_com(*args), ((x_0, mask), (x_1, mask))
        )
        # x_0 should already be masked (reference), x_1 depends on dataloader
        # x_0 should be centered (reference), x_1 depends on dataloader

        n = x_0.shape[-2]
        t = self._extend_t(n, t)  # [*, n]
        t = t[..., None]  # [*, n, 1]
        trans_t = (1.0 - t) * x_0 + t * x_1
        return trans_t  # Masking nor centering necessary since x_0 and x_1 are

    def log_snr(
            self, t: torch.Tensor, eps: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns log-SNR, and time derivative, of interpolation scheme.
        With our interpolation, given by

        x_t = (1 - t) x_0 + t x_1, (in euclidean space)

        we get log-SNR(t) = log (t/(1-t))^2, and the time derivative is
        d/dt log-SNR(t) = 2 / (t*(1-t)).

        For now we only support this interpolation scheme.

        Args:
            t: interpolation time, shape [*]

        Returns:
            Tuple with two elements, log-SNR(t) and d/dt log-SNR(t),
            each one with shape [*].
        """
        t = t.clamp(min=eps, max=1.0 - eps)  # For safety
        log_snr = 2.0 * torch.log(t / (1.0 - t))
        dlog_snr = 2.0 / (t * (1.0 - t))
        return log_snr, dlog_snr

    def xt_dot(
            self,
            x_1: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Computes \dot{x_t} for the interpolation scheme defined
        above. This is the target used in flow matching loss.

        Args:
            x_1: Sample tensor from target, shape [*, n, 3]
            x_t: Interpolated tensor, shape [*, n, 3]
            t: Interpolation times, shape [*]
            mask (optional): Binary mask of shape [*, n]

        Returns:
            dx_t / dt, with shapes [*, n, 3].
        """
        x_1, x_t = map(
            lambda args: self._mask_and_zero_com(*args), ((x_1, mask), (x_t, mask))
        )
        # x_t should be masked (interp or sampling), x_1 depdnds on dataloader and pred network
        # x_t should already be centered, x_1 not necessarily (data or pred)

        n = x_1.shape[-2]
        t = self._extend_t(n, t)  # [*, n]
        t = t[..., None]  # [*, n, 1]
        x_t_dot = (x_1 - x_t) / (1.0 - t).clamp(min=1e-4)
        return x_t_dot
        # Masking not necessary since both x_1 and x_t masked

    def simulation_step(
            self,
            x_t: torch.Tensor,
            v: torch.Tensor,
            t: torch.Tensor,
            dt: float,
            gt: float,
            sampling_mode: Literal["vf", "sc"],
            sc_scale_noise: float,
            sc_scale_score: float,
            mask: Optional[torch.Tensor] = None,
            parallel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single integration step of ODE \dot{x_t} = v(x_t, t) using Euler integration scheme.

        Args:
            x_t: Current values, shape [*, n, 3]
            v: Vector field of shape [*, n, 3]
            t: Current time, shape [*]
            dt: Step-size, float
            sampling_mode: "vf" of "sc", standing for vector field (normal flow matching, eq. (1)) and
                score (introduces score, eq. (2)).
            sc_scale_noise: scale applied to the noise when simulating eq. (2).
            sc_scale_score: scale applied to the score when simulating eq. (2).
            sc_g: constant g(t) from eq. (2), if zero should reduce to eq. (1) (modulo numerics).
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Updated x_t after Euler integration step, shape [*, n, 3]
            Updated time [*]
        """
        v = self._apply_mask(v, mask)
        n = x_t.shape[-2]

        # Euler step
        t_ext = self._extend_t(n, t)  # [*, n]

        x_t_updated, _ = (self.step_euler_parallel if parallel else self.step_euler)(
            x_t=x_t,
            v=v,
            t=t_ext,
            dt=dt,
            gt=gt,
            sampling_mode=sampling_mode,
            sc_scale_noise=sc_scale_noise,
            sc_scale_score=sc_scale_score,
        )

        return (
            self._mask_and_zero_com(
                x_t_updated, mask
            ),  # Equivalent to centering the update vector since x_t is centered
            t + dt,
        )

    def step_euler(
            self,
            x_t: torch.Tensor,
            v: torch.Tensor,
            t: torch.Tensor,
            dt: float,
            gt: float,
            sampling_mode: Literal["vf", "sc"],
            sc_scale_noise: float,
            sc_scale_score: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single integration step of ODE

        eq. (1): d x_t = v(x_t, t) dt

        or SDE

        eq. (2): d x_t = [v(x_t, t) + g(t) s(x_t, t)] dt + \sqrt{2g(t)} dw_t

        using Euler integration scheme.

        For our interpolation scheme (i.e. stochastic interpolant) we can obtain
        the score as a function of the vector field from

        v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

        or equivalently,

        s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

        We add a few additional parameters to the SDE to control noise/score scale and
        perform stochastic and low temperature sampling:

        eq. (3): d x_t = [v(x_t, t) + g(t) * sc_score_scale * s(x_t, t) * sc_score_scale] dt + \sqrt{2 * g(t) * sc_noise_scale} dw_t,

        where g(t) = sc_g * min(5, (1-t)/t).

        At the moment we do not scale the vector field v.

        Args:
            x_t: Current value, shape [*, n, 3]
            v: Vector field, shape [*, n, 3]
            t: Current time, shape [*, n]
            dt: Step-size, float
            sampling_mode: "vf" of "sc", standing for vector field (normal flow matching, eq. (1)) and
                score (introduces score, eq. (2)).
            sc_scale_noise: scale applied to the noise when simulating eq. (2),
            sc_scale_score: scale applied to the score when simulating eq. (2).
            sc_g: constant g(t) from eq. (2), if zero should reduce to eq. (1) (modulo numerics).

        Returns:
            Updated values for x_t after an Euler integration step, shape [*, n, 3].
            Updated time [*, n]
        """
        assert sampling_mode in [
            "vf",
            "sc",
        ], f"Invalid sampling mode {sampling_mode}, should be `vf` or `sc`"
        assert (
                sc_scale_noise >= 0
        ), f"Scale noise for sampling should be >= 0, got {sc_scale_noise}"
        assert (
                sc_scale_score >= 0
        ), f"Scale score for sampling should be >= 0, got {sc_scale_score}"
        assert gt >= 0, f"gt for sampling should be >= 0, got {gt}"
        t_element = t.flatten()[0]
        assert torch.all(
            t_element == t
        ), "Sampling only implemented for same time for all samples"
        # The last few steps are always taken with eq. (1).

        if (
                sampling_mode == "vf" or t_element > 1.0
        ):
            return x_t + v * dt, t + dt

        if sampling_mode == "sc":
            score = self.vf_to_score(x_t, v, t)  # get score from v, [*, dim]
            eps = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)  # [*, dim]
            std_eps = torch.sqrt(2 * gt * sc_scale_noise * dt)
            delta_x = (v + gt * score) * dt + std_eps * eps
            return x_t + delta_x, t + dt

    def step_euler_parallel(
            self,
            x_t: torch.Tensor,
            v: torch.Tensor,
            t: torch.Tensor,
            dt: torch.Tensor,
            gt: torch.Tensor,
            sampling_mode: Literal["vf", "sc"],
            sc_scale_noise: float,
            sc_scale_score: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt_expand = dt[:, None, None]
        if sampling_mode == "vf":
            return x_t + v * dt_expand, t + dt.unsqueeze(-1)

        if sampling_mode == "sc":
            score = self.vf_to_score(x_t, v, t.clamp(max=0.999))  # get score from v, [*, dim]
            eps = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)  # [*, dim]
            std_eps = torch.sqrt(2 * gt * sc_scale_noise * dt)[:, None, None]
            delta_x = (v + gt[:, None, None] * score) * dt_expand + std_eps * eps
            return x_t + delta_x, t + dt.unsqueeze(-1)

    def vf_to_score(
            self,
            x_t: torch.Tensor,
            v: torch.Tensor,
            t: torch.Tensor,
    ):
        """
        Compute score of noisy density given the vector field learned by flow matching. With
        our interpolation scheme these are related by

        v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

        or equivalently,

        s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

        Args:
            x_t: Noisy sample, shape [*, dim]
            v: Vector field, shape [*, dim]
            t: Interpolation time, shape [*]

        Returns:
            Score of intermediate density, shape [*, dim].
        """
        assert torch.all(t < 1.0), "vf_to_score requires t < 1 (strict)"
        num = t[..., None] * v - x_t  # [*, n, 3]
        den = (1.0 - t)[..., None] * self.scale_ref ** 2  # [*, n, 1]
        score = num / den
        return score  # [*, dim]

    def sample_reference(
            self,
            n: int,
            shape: Tuple = tuple(),
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Samples reference distribution std Gaussian (possibly centered).

        Args:
            n: number of frames in a single sample, int
            shape: tuple (if empty then single sample)
            dtype (optional): torch.dtype used
            device (optional): torch device used
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Samples from refenrece [N(0, I_3)]^n shape [*shape, n, 3]
        """
        x = (
                torch.randn(
                    shape
                    + (
                        n,
                        self.dim,
                    ),
                    device=device,
                    dtype=dtype,
                )
                * self.scale_ref
        )
        return self._mask_and_zero_com(x, mask)

    def get_gt(
            self,
            t: torch.Tensor,
            mode: str,
            param: float,
            clamp_val: Optional[float] = None,
            eps: float = 1e-2,
    ) -> torch.Tensor:
        """
        Computes gt for different modes.

        Args:
            t: times where we'll evaluate, covers [0, 1), shape [nsteps]
            mode: "us" or "tan"
            param: parameterized transformation
            clamp_val: value to clamp gt, no clamping if None
            eps: small value leave as it is

        Returns
        """

        # Function to get variants for some gt mode
        def transform_gt(gt, f_pow=1.0):
            # 1.0 means no transformation
            if f_pow == 1.0:
                return gt

            # First we somewhat normalize between 0 and 1
            log_gt = torch.log(gt)
            mean_log_gt = torch.mean(log_gt)
            log_gt_centered = log_gt - mean_log_gt
            normalized = torch.nn.functional.sigmoid(log_gt_centered)
            # Transformation here
            normalized = normalized ** f_pow
            # Undo normalization with the transformed variable
            log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
            log_gt_rec = log_gt_centered_rec + mean_log_gt
            gt_rec = torch.exp(log_gt_rec)
            return gt_rec

        # Numerical reasons for some schedule
        t = torch.clamp(t, 0, 1 - 1e-5)

        if mode == "us":
            num = 1.0 - t
            den = t
            gt = num / (den + eps)
        elif mode == "tan":
            num = torch.sin((1.0 - t) * torch.pi / 2.0)
            den = torch.cos((1.0 - t) * torch.pi / 2.0)
            gt = (torch.pi / 2.0) * num / (den + eps)
        elif mode == "1/t":
            num = 1.0
            den = t
            gt = num / (den + eps)
        else:
            raise NotImplementedError(f"gt not implemented {mode}")
        gt = transform_gt(gt, f_pow=param)
        gt = torch.clamp(gt, 0, clamp_val)  # If None no clamping
        return gt  # [s]

    def get_schedule(self, mode: str, nsteps: int, *, p1: float = None, eps=1e-5):
        # Useful for schedules defined in terms of SNR
        def snr_to_us_t(snr):
            snr = torch.clamp(snr, 1e-10, 1e10)
            return torch.sqrt(snr) / (1 + torch.sqrt(snr))

        # In case we want to use EDM schedule
        def snr_edm(n, rho):
            step = torch.arange(n)
            s_min, s_max = 0.002, 80.0
            r = rho
            ir = 1.0 / rho
            sigma = (s_max ** ir + (step / (n - 1)) * (s_min ** ir - s_max ** ir)) ** r
            snr = 1.0 / sigma ** 2
            return snr

        if mode == "uniform":
            t = torch.linspace(0, 1, nsteps + 1)
            return t
        elif mode == "early_uniform":
            assert p1 is not None, "p1 cannot be none for the early uniform schedule"
            return torch.linspace(0, 1, nsteps + 1).div(p1).clamp(max=1.0)
        elif mode == "early_sqrt":
            assert p1 is not None, "p1 cannot be none for the early square root schedule"
            return torch.linspace(0, 1, nsteps + 1).div(p1).sqrt().clamp(max=1.0)
        elif mode == "early_square":
            assert p1 is not None, "p1 cannot be none for the early square schedule"
            return torch.linspace(0, 1, nsteps + 1).div(p1).square().clamp(max=1.0)
        elif mode == "power":
            assert p1 is not None, "p1 cannot be none for the power schedule"
            t = torch.linspace(0, 1, nsteps + 1)
            t = t ** p1
            return t
        elif mode == "cos_sch_v_snr":
            assert p1 is not None, "p1 cannot be none for the cos_sch_v_snr schedule"
            t = torch.linspace(0, 1, nsteps + 1)
            num_snr = torch.cos(torch.pi * (1 - t) / 2) + eps
            den_snr = torch.sin(torch.pi * (1 - t) / 2) + eps
            snr = num_snr ** p1 / den_snr ** p1
            t = snr_to_us_t(snr)
            t = t - torch.min(t)
            t = t / torch.max(t)
            return t
        elif mode == "loglinear":
            t = snr_to_us_t(torch.logspace(-6, 6, nsteps + 1))
            t = t - torch.min(t)
            t = t / torch.max(t)
            return t
        elif mode == "edm":
            assert p1 is not None, "p1 cannot be none for the edm schedule"
            t = snr_to_us_t(snr_edm(nsteps + 1, p1))
            t = t - torch.min(t)
            t = t / torch.max(t)
            return t
        elif mode == "log":
            assert p1 is not None, "p1 cannot be none for the log schedule"
            assert p1 > 0, f"p1 must be >0 for the log schedule, got {p1}"
            t = 1.0 - torch.logspace(-p1, 0, nsteps + 1).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
            return t
        else:
            # Should not get here
            raise IOError(f"Schedule mode not recognized {mode}")

    def full_simulation_gp(
            self,
            predict_clean_n_v: Callable,
            dt: float,
            nsamples: int,
            max_length: int,
            self_cond: bool,
            device: torch.device,
            schedule_mode: Literal["uniform", "power", "cos_sch_v_snr", "loglinear", "edm", "log"],
            schedule_p: float,
            schedule_edit_mode: Literal["uniform", "power", "cos_sch_v_snr", "loglinear", "edm", "log"],
            schedule_edit_p: float,
            sampling_mode: str,
            sc_scale_noise: float,
            sc_scale_score: float,
            gt_mode: Literal["us", "tan"],
            gt_p: float,
            gt_clamp_val: float,
            scale_rate: float,
            x_motif=None,
            aa_motif=None,
            fixed_sequence_mask=None,
            fixed_structure_mask=None,
            dtype: Optional[torch.dtype] = None,
            contig_types=None,
            contig_lengths=None,
            tau_leap: bool = False,
    ) -> tuple:
        """
        Generate variable-length samples from t=0 to t=1, optionally around a motif.

        Args:
            predict_clean_n_v: A function that predicts clean sample and vector field
                takes as input a dictionary with keys:
                    - "x_t"
                    - "t"
                    - "mask"
                    - "x_sc" (optional, self-conditioning)
                with values the corresponding tensors
            dt: step-size, float
            nsamples: number of samples to generate, int
            max_length: stop inserting into samples that have reached this length;
                simultaneous insertions can overshoot the threshold.
            self_cond: whether to use self conditioning or not
            schedule_mode: "uniform", "power", "log", "inv_smooth_step"
            schedule_p: parameter of the schedule for the times covering [0, 1]
                uniform: this is ignored, points uniformly spaced
                power: uniform ** schedule_p
                log: ts = (1.0 - np.logspace(schedule_p, 0, num=nsteps)[::-1])
                     ts = ts / ts[-1]  # to make sure it goes exactly from 0 to 1
            dtype (optional): dtype used for the simulation
            x_motif, aa_motif, fixed_sequence_mask, fixed_structure_mask,
                contig_types, contig_lengths: required together for motif inpainting.
                Insertions are restricted to the linker segments in the contig.
            tau_leap: sample Poisson insertion counts per gap instead of at most
                one insertion. Tau-leaping skips insertions on the final step.

        Returns:
            Padded coordinates and sample lengths; motif runs also return updated
            contig types and lengths. Motif coordinates remain fixed in both modes.
        """
        motif_inputs = (x_motif, aa_motif, fixed_sequence_mask, fixed_structure_mask,
                        contig_types, contig_lengths)
        if any(value is not None for value in motif_inputs) and any(value is None for value in motif_inputs):
            raise ValueError(
                "Motif inpainting requires coordinates, amino acids, sequence and structure masks, "
                "and contig types and lengths together."
            )

        # Get discretization
        nsteps = math.ceil(
            1.0 / dt
        )  # below it uses nsteps + 1 since we need to include limits 0 and 1, but we never evaluate at 1
        log_info(
            f"Sampling: nsteps={nsteps}, schedule={schedule_mode}, param={schedule_p}, gt={gt_mode}, gt_p={gt_p}, gt_clamp={gt_clamp_val}, temp={sc_scale_noise}"
        )

        ts = self.get_schedule(
            mode=schedule_mode,
            nsteps=nsteps,
            p1=schedule_p,
        )
        ts_edit = self.get_schedule(
            mode=schedule_edit_mode,
            nsteps=nsteps,
            p1=schedule_edit_p,
        )
        # [nsteps + 1], first element is 0, last element is 1
        # we eval the learned vector field on the values ts[:-1], ie we
        # do not evaluate at 1. So this defines nsteps steps.
        # Get gt
        t_eval = ts[:-1]  # [nsteps], last one is 1 not used to eval but to define dt
        gt = self.get_gt(
            t=t_eval,
            mode=gt_mode,
            param=gt_p,
            clamp_val=gt_clamp_val,
        )
        if aa_motif is not None:
            aa_motif_expanded = aa_motif.unsqueeze(0).expand(nsamples, -1).to(device)

        with torch.no_grad():
            x = torch.empty(nsamples, 0, 3, dtype=dtype, device=device)
            mask = torch.empty(nsamples, 0, dtype=torch.bool, device=device)

            if fixed_sequence_mask is not None:
                x_motif = (x_motif - mean_w_mask(x_motif, fixed_sequence_mask, keepdim=True)) * \
                          fixed_sequence_mask[..., None]
                x = x_motif.clone()
                mask = fixed_sequence_mask.clone()
                motif_batch_idx, motif_seq_idx = fixed_sequence_mask.nonzero(as_tuple=True)
                motif_coords = x_motif[motif_batch_idx, motif_seq_idx]

            cur_lens = mask.sum(-1)
            layout_changed = True
            batch_ones = torch.ones(nsamples, device=device)
            pbar = tqdm(range(nsteps))
            for step in pbar:
                # These inputs depend on residue positions, not on coordinates or time.
                # Reuse them until an insertion changes the layout.
                if layout_changed:
                    full_length_mask = cur_lens >= max_length
                    valid_gaps = torch.cat([
                        torch.ones(nsamples, 1, dtype=torch.bool, device=device), mask
                    ], dim=-1)
                    if fixed_sequence_mask is not None:
                        aa_1 = torch.full((nsamples, x.size(1)), UNK_TOKEN_ID, dtype=torch.long, device=device)
                        aa_1.masked_scatter_(fixed_sequence_mask, aa_motif_expanded)
                        valid_gaps &= torch.as_tensor(
                            contig_info_to_insert_mask(contig_types, contig_lengths),
                            dtype=torch.bool, device=device,
                        )
                    layout_changed = False

                t = ts[step] * batch_ones  # [nsamples]
                t_edit = ts_edit[step] * batch_ones  # [nsamples]
                gt_step = gt[step]  # float

                nn_in = {
                    "x_t": x,
                    "t": t,
                    "t_edit": t_edit,
                    "mask": mask,
                }
                if fixed_structure_mask is not None:
                    nn_in.update({
                        "motif_mask": fixed_sequence_mask,
                        "fixed_structure_mask": fixed_structure_mask,
                        "x_motif": x_motif,
                        "aa_1": aa_1,
                    })

                if step > 0 and self_cond:
                    sc_pred = predict_clean_n_v(nn_in)[0]
                    nn_in["x_sc"] = sc_pred.detach()  # Self-conditioning

                x_1_pred, v, nn_out = predict_clean_n_v(nn_in)
                ins_rate, ins_coord = nn_out['ins_rate'], nn_out['ins_coord']
                # Avoid synchronizing diagnostic reductions with the CPU every step.
                if step % 10 == 0 or step == nsteps - 1:
                    total_rate = (ins_rate[:, 1:] * mask).sum(-1) + ins_rate[:, 0]
                    lengths = cur_lens.float()
                    rate_avg, total_avg, length_avg, length_std = torch.stack([
                        total_rate.sum() / (cur_lens.sum() + nsamples), total_rate.mean(),
                        lengths.mean(), lengths.std(unbiased=False),
                    ]).tolist()
                    pbar.set_description(
                        f"Step {step + 1}/{nsteps}, t={ts[step]:.3f}, t_edit={ts_edit[step]:.3f}, "
                        f"ins_rate={rate_avg:.3f}, total_rate={total_avg:.3f}, "
                        f"len={length_avg:.2f}±{length_std:.2f}"
                    )

                # Accomodate last few steps
                if ts[step] > 0.99:
                    sampling_mode = "vf"
                if schedule_mode in ["cos_sch_v_snr", "edm"]:
                    if ts[step] > 0.985:
                        sampling_mode = "vf"

                if x.size(1) > 0:
                    x, _ = self.simulation_step(
                        x_t=x,
                        v=v,
                        t=t,
                        dt=ts[step + 1] - ts[step],
                        gt=gt_step,
                        sampling_mode=sampling_mode,
                        sc_scale_noise=sc_scale_noise,
                        sc_scale_score=sc_scale_score,
                        mask=mask,
                    )

                # Integration can move the motif even when no insertions occur.
                if fixed_sequence_mask is not None:
                    if x.dtype == motif_coords.dtype:
                        x[motif_batch_idx, motif_seq_idx] = motif_coords
                    else:
                        # Preserve type promotion when motif and sampling dtypes differ.
                        x = torch.where(fixed_sequence_mask[..., None], x_motif, x)

                # Apply insertion
                ins_prob = ins_rate * (ts_edit[step + 1] - ts_edit[step]) * scale_rate  # (B, L + 1)
                # For sequences that are already at max length, zero out the insert probabilities
                ins_prob.masked_fill_(full_length_mask[:, None], 0.0)
                if tau_leap:
                    ins_count = torch.poisson(ins_prob)
                else:
                    ins_count = torch.rand_like(ins_prob) < ins_prob
                ins_count *= valid_gaps

                # Keep the random draws above even when the edit schedule has stopped:
                # skipping them would change subsequent SDE noise for the same seed.
                if not ins_count.any():
                    continue  # reuse all layout tensors; no expansion or remapping
                if tau_leap and step == nsteps - 1:
                    continue  # repeated final-step insertions would remain coincident
                ins_count = ins_count.long()

                # Interleave insertion sites and existing residues, then expand counts.
                batch_size, length = mask.shape
                combined_count = torch.zeros(batch_size, 2 * length + 1, device=device, dtype=torch.long)
                combined_count[:, ::2] = ins_count
                combined_count[:, 1::2] = mask
                new_lengths = combined_count.sum(-1)
                new_max_length = new_lengths.max().item()

                combined_x = torch.zeros(batch_size, 2 * length + 1, 3, device=device, dtype=dtype)
                combined_x[:, ::2] = ins_coord
                combined_x[:, 1::2] = x
                expanded_x = torch.repeat_interleave(
                    combined_x.reshape(-1, 3), combined_count.flatten(), dim=0
                )
                new_mask = torch.arange(new_max_length, device=device)[None, :] < new_lengths[:, None]
                new_x = torch.zeros(nsamples, new_max_length, 3, device=device, dtype=dtype)
                new_x[new_mask] = expanded_x

                # Move motif conditioning with the original residues, across all insertions.
                if fixed_sequence_mask is not None:
                    motif_seq_idx = motif_seq_idx + ins_count.cumsum(-1)[motif_batch_idx, motif_seq_idx]
                    new_motif_mask = torch.zeros(nsamples, new_max_length, device=device, dtype=torch.bool)
                    new_x_motif = torch.zeros(nsamples, new_max_length, 3, device=device, dtype=x_motif.dtype)
                    new_x_motif[motif_batch_idx, motif_seq_idx] = motif_coords
                    new_motif_mask[motif_batch_idx, motif_seq_idx] = True

                    x_motif = new_x_motif
                    fixed_sequence_mask = new_motif_mask
                    fixed_structure_mask = fixed_sequence_mask[:, :, None] & fixed_sequence_mask[:, None, :]

                    # Transfer counts once for the entire batch, rather than per sample.
                    ins_count_cpu = ins_count.cpu().numpy()
                    new_contig_lengths = []
                    for b in range(nsamples):
                        new_contig_lengths.append(
                            count_insertions_in_segments(ins_count_cpu[b], contig_types[b], contig_lengths[b]))
                    new_contig_lengths = np.array(new_contig_lengths)
                    contig_lengths = new_contig_lengths

                x = new_x
                mask = new_mask
                cur_lens = new_lengths
                layout_changed = True

        if fixed_sequence_mask is not None:
            return x, cur_lens, contig_types, contig_lengths
        else:
            return x, cur_lens
