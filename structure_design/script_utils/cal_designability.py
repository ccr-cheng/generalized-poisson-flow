"""Evaluate designability of unconditional generations with ProteinMPNN/ESMFold.

Run from the repository root::

    python script_utils/cal_designability.py /path/to/samples

Expects samples_dir/b{batch_idx}_i{idx}/b{batch_idx}_i{idx}.pdb;
--pdb_glob overrides this layout with a pattern relative to samples_dir.
Writes results_<rank>.csv in samples_dir and refolding artifacts beside each
input PDB. A sample is designable when its best scRMSD is <= 2 Angstrom.
The inference config controls designability_seqs_per_struct (default: 8).
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from transformers import AutoTokenizer, EsmForProteinFolding
import hydra

# Add root to path to ensure project imports work
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from proteinfoundation.metrics.designability import proteinmpnn_device_index, scRMSD


class DesignabilityDataset(Dataset):
    """
    Dataset to glob target PDB files for designability calculation.
    Expects structure: root_dir/b{batch_idx}_i{idx}/b{batch_idx}_i{idx}.pdb
    """

    def __init__(self, root_dir, pdb_glob="b*_i*/b*_i*.pdb"):
        root_dir = Path(root_dir).expanduser().resolve()
        # Match generated backbones without including ESMFold evaluation outputs.
        pattern = os.path.join(root_dir, pdb_glob)
        self.pdb_files = sorted(path for path in glob.glob(pattern) if os.path.isfile(path))

        if not self.pdb_files:
            raise RuntimeError(f"No files found with pattern {pattern}!")

        logger.info(f"Found {len(self.pdb_files)} PDB files in {root_dir}")

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        return self.pdb_files[idx]


class DesignabilityModule(pl.LightningModule):
    """
    Lightning Module that loads ESM and calculates scRMSD in predict_step.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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

    def predict_step(self, batch, batch_idx):
        """
        Calculates structure-only designability (scRMSD) for a batch of PDBs.
        """
        # Batch comes from DesignabilityDataset, so it is a list of PDB paths
        # Batch size is typically 1 here to manage ESM memory usage
        pdb_path = batch[0]

        # Temporary path for scRMSD artifacts (same dir as the pdb)
        sample_root_path = os.path.dirname(pdb_path)

        try:
            # Calculate scRMSD
            # using self.device to ensure tensors created inside are on the correct GPU
            res_designability, res_plddt, _ = scRMSD(
                pdb_path,
                ret_min=False,
                tmp_path=sample_root_path,
                num_seq_per_target=self.cfg.designability_seqs_per_struct,
                tokenizer=self.tokenizer,
                esm_model=self.esm_model,
                device_idx=proteinmpnn_device_index(self.device)
            )

            # Aggregate results
            sc_rmsd_best = min(res_designability)
            plddt_best = max(res_plddt) * 100

            # Create string representations for the CSV
            res_designability_str = '[' + ','.join([f"{v:.3f}" for v in res_designability]) + ']'
            res_plddt_str = '[' + ','.join([f"{v * 100:.2f}" for v in res_plddt]) + ']'

            logger.info(f"{os.path.basename(pdb_path)}: scRMSD={sc_rmsd_best:.3f}, pLDDT={plddt_best:.2f}")

            return {
                "pdb_path": pdb_path,
                "scRMSD": sc_rmsd_best,
                "scRMSD_all": res_designability_str,
                "pLDDT": plddt_best,
                "pLDDT_all": res_plddt_str
            }

        except Exception as e:
            logger.error(f"Error calculating designability for {pdb_path}: {e}")
            return {
                "pdb_path": pdb_path,
                "scRMSD": 100.0,  # High value for failure
                "scRMSD_all": "[]",
                "pLDDT": 0.0,
                "pLDDT_all": "[]"
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "samples_dir",
        type=str,
        help="Directory containing the generated sample subdirectories."
    )
    parser.add_argument(
        "--pdb_glob",
        default="b*_i*/b*_i*.pdb",
        help="Input PDB glob relative to samples_dir (default: %(default)s)."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_uncond",
        help="Name of the config yaml file (for scRMSD params).",
    )
    parser.add_argument(
        "--config_subdir",
        type=str,
        default=None,
        help="Subdirectory for config files if needed.",
    )

    args = parser.parse_args()
    args.samples_dir = str(Path(args.samples_dir).expanduser().resolve())

    # Load Configuration using Hydra
    if args.config_subdir is None:
        config_path = "../configs/experiment_config"
    else:
        config_path = f"../configs/experiment_config/{args.config_subdir}"

    logger.info(f"Loading config from {config_path} / {args.config_name}")

    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=args.config_name)

    # Setup Dataset
    dataset = DesignabilityDataset(args.samples_dir, pdb_glob=args.pdb_glob)
    if len(dataset) == 0:
        logger.error("No PDB files found. Exiting.")
        sys.exit(1)

    # Setup DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Setup Module
    model = DesignabilityModule(cfg)

    # Setup Trainer
    # We use basic settings suitable for inference
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        logger=False,  # We handle logging manually
        enable_checkpointing=False
    )

    logger.info("Starting designability calculation...")
    predictions = trainer.predict(model, dataloader)

    # Predictions is a list of dictionaries (since batch_size=1)
    # If batch_size > 1, we would need to flatten the list
    results_df = pd.DataFrame(predictions)

    # Calculate summary statistics
    designability_rate = (results_df['scRMSD'] <= 2.0).mean() * 100
    avg_scrmsd = results_df['scRMSD'].mean()
    avg_plddt = results_df['pLDDT'].mean()

    logger.info("-" * 40)
    logger.info(f"Total Samples: {len(results_df)}")
    logger.info(f"Designability (scRMSD <= 2.0): {designability_rate:.2f}%")
    logger.info(f"Average scRMSD: {avg_scrmsd:.3f}")
    logger.info(f"Average pLDDT: {avg_plddt:.2f}")
    logger.info("-" * 40)

    # Save to CSV
    csv_file = os.path.join(args.samples_dir, f"results_{trainer.global_rank}.csv")
    results_df.to_csv(csv_file, index=False)
    logger.info(f"Results saved to {csv_file}")
