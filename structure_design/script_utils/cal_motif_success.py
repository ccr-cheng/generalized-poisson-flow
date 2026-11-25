"""Evaluate motif-scaffolding success with ProteinMPNN and ESMFold.

Run from the repository root::

    python script_utils/cal_motif_success.py /path/to/samples 1bcf --n_sample 1000
    python script_utils/cal_motif_success.py /path/to/samples 1bcf --motif_dir /path/to/motifs

Expects samples_dir/task/{idx}.pdb and motif_info.csv, with one contig per
sample. Reference motifs are motif_dir/task.pdb; motif_dir defaults to the
repository's motif_data/design25. Designs eight sequences per scaffold,
always fixing supplied motif amino acids during ProteinMPNN sampling.
Unknown amino acids remain designable. Success requires one sequence with
scRMSD <= 2 A, motif RMSD <= 1 A, pAE <= 5 A and pLDDT >= 0.7.
Writes results_<rank>.csv and per-sample artifacts inside samples_dir/task.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from loguru import logger
from transformers import AutoTokenizer, EsmForProteinFolding
from Bio.PDB import PDBParser, PDBIO

# Add root to path to ensure project imports work
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from proteinfoundation.nn.motif_factory import parse_contig_from_pdb, motif_extract
from proteinfoundation.metrics.designability import rmsd_metric, scRMSD
from openfold.np.residue_constants import restype_3to1

ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def extract_motif_info(ref_pdb_path):
    # Extract structure
    with open(ref_pdb_path) as f:
        ref_contig_string, _, _ = parse_contig_from_pdb(f.read())
    ref_motif = motif_extract(ref_contig_string, ref_pdb_path, motif_only=True, atom_part='backbone')
    ref_motif = torch.from_numpy(np.array([x.coord for x in ref_motif])).view(-1, 4, 3)
    ref_motif = torch.cat(
        [ref_motif, torch.zeros((ref_motif.size(0), 33, 3), dtype=ref_motif.dtype)], dim=1
    )

    # Extract sequence
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure('structure', ref_pdb_path)
    res_id, res_type = [], []
    for residue in structure.get_residues():
        res_id.append(residue.get_id()[1])  # Get residue number
        res_type.append(residue.get_resname())

    motif_idx = 0
    fix_info = []
    for config in ref_contig_string.split('/'):
        if config[0] in ALPHABET:
            fix_pos = []
            fix_res = []
            start, end = map(int, config[1:].split('-'))
            for _ in range(end - start + 1):
                rid, rtype = res_id[motif_idx], res_type[motif_idx]
                assert start <= rid <= end, f"Residue {rid} not in range {start}-{end}!"
                # ProteinMPNN maps unknown residue names (including UNK) to X;
                # only supplied, standard amino acids are fixed during design.
                if rtype in restype_3to1:
                    fix_pos.append(rid - start)  # Convert to relative position
                    fix_res.append(rtype)
                motif_idx += 1
            fix_info.append({"pos": fix_pos, "res": fix_res})
    logger.info(f"Extracted motif of {ref_motif.size()}, contig: {ref_contig_string}, fix_info: {fix_info}")
    return ref_motif, fix_info


def fix_pdb_file(pdb_path, fix_pdb_path, fix_json_path, fix_info, contig_str):
    # Write json file
    motif_idx = 0
    fix_res = []
    fix_type = []
    for config in contig_str.split('/'):
        if config[0] in ALPHABET:
            if '-' in config:
                start = int(config[1:].split('-')[0])
            else:
                start = int(config[1:])
            d = fix_info[motif_idx]
            fix_res.extend([start + idx for idx in d["pos"]])
            fix_type.extend(d["res"])
            motif_idx += 1
    with open(fix_json_path, 'w') as f:
        json.dump({os.path.basename(fix_pdb_path)[:-4]: {"A": fix_res}}, f)
    logger.info(f"Saved fix info to {fix_json_path}")

    # Modify PDB file
    res_map = {res_id: res_type for res_id, res_type in zip(fix_res, fix_type)}
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure('structure', pdb_path)
    for model in structure:
        for chain in model:
            chain.id = 'A'  # Change chain ID to A
    for residue in structure.get_residues():
        res_id = residue.get_id()[1]  # Get residue number
        if res_id in res_map:
            residue.resname = res_map[res_id]
    io = PDBIO()
    io.set_structure(structure)
    io.save(fix_pdb_path)
    logger.info(f"Saved modified PDB to {fix_pdb_path}")


def normalize_contig_str(contig):
    parts = contig.split('/')
    return '/'.join(
        ('A' + p[1:]) if p and p[0] in ALPHABET else p
        for p in parts
    )


def compute_motif_rmsd(contig_str, pdb_path, ref_motif):
    contig_str = normalize_contig_str(contig_str)
    gen_coors = motif_extract(contig_str, pdb_path, motif_only=False, atom_part='backbone')
    gen_coors = torch.from_numpy(np.array([x.coord for x in gen_coors])).view(-1, 4, 3)
    gen_coors = torch.cat([gen_coors, torch.zeros((gen_coors.size(0), 33, 3), dtype=gen_coors.dtype)], dim=1)
    motif_rmsd = rmsd_metric(gen_coors, ref_motif, mode='bb')  # Use backbone RMSD for motif
    return motif_rmsd


class DesignabilityDataset(Dataset):
    """Load numbered scaffolds and their contigs for one motif target."""

    def __init__(self, root_dir, task, n_sample=1000, motif_dir=None):
        root_dir = Path(root_dir).expanduser().resolve()
        self.sample_path = os.path.join(root_dir, task)
        self.n_sample = n_sample

        contig_df = pd.read_csv(os.path.join(root_dir, task, "motif_info.csv"))
        self.contig_strs = contig_df['contig'].tolist()
        assert len(self.contig_strs) == self.n_sample, \
            f"Got {self.n_sample} pdbs with {len(self.contig_strs)} contigs"

        motif_dir = (
            Path(motif_dir).expanduser()
            if motif_dir is not None else Path(root) / 'motif_data' / 'design25'
        )
        ref_pdb_path = str(motif_dir / f'{task}.pdb')
        self.ref_motif, self.fix_info = extract_motif_info(ref_pdb_path)

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        return (
            os.path.join(self.sample_path, f'{idx}.pdb'),
            self.contig_strs[idx],
            self.ref_motif,
            self.fix_info
        )


class DesignabilityModule(pl.LightningModule):
    """Evaluate motif success with supplied motif amino acids fixed in ProteinMPNN."""

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.esm_model = None

    def setup(self, stage=None):
        """
        Load ESM model and tokenizer once during setup.
        """
        if self.tokenizer is None:
            logger.info("Loading ESMFold Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

        if self.esm_model is None:
            logger.info("Loading ESMFold Model...")
            self.esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            self.esm_model.eval()
            # Note: Lightning will handle moving self.esm_model to the correct device

    def success(self, rmsd, motifrmsd, plddt, pae):
        for _rmsd, _motifrmsd, _plddt, _pae in zip(rmsd, motifrmsd, plddt, pae):
            if _rmsd <= 2.0 and _motifrmsd <= 1.0 and _pae <= 5.0 and _plddt >= 0.7:
                return True
        return False

    def predict_step(self, batch, batch_idx):
        """
        Calculates structure-only designability (scRMSD) for a batch of PDBs.
        """
        pdb_path, contig_str, ref_motif, fix_info = batch[0]
        last_contig = contig_str.split('/')[-1]
        L = int(last_contig.split('-')[-1]) if '-' in last_contig else int(last_contig)
        ref_motif = ref_motif.cpu()

        # Temporary path for scRMSD artifacts (same dir as the pdb)
        sample_root_path = pdb_path[:-4]  # strip .pdb
        os.makedirs(sample_root_path, exist_ok=True)
        n_seq = 8

        try:
            # Calculate scRMSD
            # using self.device to ensure tensors created inside are on the correct GPU

            # Write fix_info to JSON for scRMSD to use
            fix_pdb_path = os.path.join(sample_root_path, f'fixed_{os.path.basename(pdb_path)}')
            fix_json_path = os.path.join(sample_root_path, "fix_info.json")
            fix_pdb_file(pdb_path, fix_pdb_path, fix_json_path, fix_info, contig_str)
            res_rmsd, res_plddt, res_pae = scRMSD(
                fix_pdb_path,
                ret_min=False,
                tmp_path=sample_root_path,
                num_seq_per_target=n_seq,
                tokenizer=self.tokenizer,
                esm_model=self.esm_model,
                device_idx=self.trainer.global_rank if self.trainer else 0,
                fix_mode='pre',
                fix_json_path=fix_json_path
            )

            motif_rmsds = []
            for idx in range(n_seq):
                out_esm = os.path.join(sample_root_path, f'esm_{idx + 1}.pdb')
                motif_rmsds.append(compute_motif_rmsd(contig_str, out_esm, ref_motif))

            # Aggregate results
            scrmsd_best = min(res_rmsd)
            motifrmsd_best = min(motif_rmsds)
            plddt_best = max(res_plddt) * 100
            pae_best = min(res_pae)

            # Determine success based on criteria
            is_success = self.success(res_rmsd, motif_rmsds, res_plddt, res_pae)

            # Create string representations for the CSV
            res_rmsd_str = '[' + ','.join([f"{v:.3f}" for v in res_rmsd]) + ']'
            res_motifrmsd_str = '[' + ','.join([f"{v:.3f}" for v in motif_rmsds]) + ']'
            res_plddt_str = '[' + ','.join([f"{v * 100:.2f}" for v in res_plddt]) + ']'
            res_pae_str = '[' + ','.join([f"{v:.3f}" for v in res_pae]) + ']'

            logger.info(
                f"{os.path.basename(pdb_path)}: "
                f"scRMSD={scrmsd_best:.3f}, motifRMSD={motifrmsd_best:.3f}, "
                f"pLDDT={plddt_best:.2f}, pAE={pae_best:.3f}, Success={is_success}"
            )

            return {
                "pdb_path": pdb_path,
                "L": L,
                "scRMSD": scrmsd_best,
                "motifRMSD": motifrmsd_best,
                "pAE": pae_best,
                "pLDDT": plddt_best,
                "success": is_success,
                "scRMSD_all": res_rmsd_str,
                "motifRMSD_all": res_motifrmsd_str,
                "pAE_all": res_pae_str,
                "pLDDT_all": res_plddt_str,
            }

        except Exception as e:
            logger.error(f"Error calculating designability for {pdb_path}: {e}")
            return {
                "pdb_path": pdb_path, "L": L,
                "scRMSD": 100.0, "motifRMSD": 100.0, "pAE": 100.0, "pLDDT": 0.0, "success": False,
                "scRMSD_all": "[]", "motifRMSD_all": "[]", "pAE_all": "[]", "pLDDT_all": "[]",
            }


def cal_success_task(samples_dir, task, n_sample=1000, motif_dir=None):
    """Evaluate one target under samples_dir, optionally overriding reference motifs."""
    samples_dir = str(Path(samples_dir).expanduser().resolve())
    # Setup Dataset
    dataset = DesignabilityDataset(samples_dir, task, n_sample=n_sample, motif_dir=motif_dir)
    if len(dataset) == 0:
        logger.error("No PDB files found. Exiting.")
        return

    # Setup DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda b: b)

    # Setup Module
    model = DesignabilityModule()

    # Setup Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        logger=False,  # We handle logging manually
        enable_checkpointing=False
    )

    logger.info(f"Starting designability calculation for {task}...")
    predictions = trainer.predict(model, dataloader)

    # Predictions is a list of dictionaries (since batch_size=1)
    # If batch_size > 1, we would need to flatten the list
    results_df = pd.DataFrame(predictions)
    csv_file = os.path.join(samples_dir, task, f"results_{trainer.global_rank}.csv")
    results_df.to_csv(csv_file, index=False)
    logger.info(f"Results saved to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "samples_dir",
        type=str,
        help="Directory containing the generated motif-task subdirectories."
    )
    parser.add_argument(
        "task",
        type=str,
        help="Task name corresponding to the subdirectory in samples_dir (e.g., 1bcf)"
    )
    parser.add_argument(
        "--motif_dir",
        type=str,
        default=None,
        help="Reference motif directory containing <task>.pdb (default: repo/motif_data/design25)."
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=1000,
        help="Number of samples to evaluate (default: 1000)"
    )
    args = parser.parse_args()
    cal_success_task(args.samples_dir, args.task, n_sample=args.n_sample, motif_dir=args.motif_dir)
