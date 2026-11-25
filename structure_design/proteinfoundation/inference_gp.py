import os
import sys
import argparse

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)  # Import this checkout even when launched from another directory.

import hydra
import lightning as L
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
from loguru import logger

from proteinfoundation.proteinflow.proteina_gp import ProteinaGP
from proteinfoundation.utils.checkpoint_utils import resolve_checkpoint_path


class GenDataset(Dataset):
    """Schedule batches of variable-length samples without a target length."""

    def __init__(self, nbatch):
        self.nbatch = nbatch

    def __len__(self):
        return self.nbatch

    def __getitem__(self, index):
        return {'index': index}


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_uncond",
        help="Name of the config yaml file.",
    )
    parser.add_argument(
        "--config_number", type=int, default=-1, help="Number of the config yaml file."
    )
    parser.add_argument(
        "--codesign", action="store_true", help="Force codesign (for loading older checkpoints)"
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
    if args.config_subdir is None:
        config_path = "../configs/experiment_config"
    else:
        config_path = f"../configs/experiment_config/{args.config_subdir}"

    with hydra.initialize(config_path, version_base=hydra.__version__):
        # If number provided use it, otherwise name
        if args.config_number != -1:
            config_name = f"inf_{args.config_number}"
        else:
            config_name = args.config_name
        cfg = hydra.compose(config_name=config_name, overrides=hydra_overrides)
        logger.info(f"Inference config {cfg}")
        run_name = cfg.run_name

    assert (
            not cfg.compute_designability or not cfg.compute_fid
    ), "Designability cannot be computed together with FID"

    # Set root path for this inference run
    root_path = os.path.join("./inference", run_name)
    os.makedirs(root_path, exist_ok=True)

    # Load model from checkpoint
    ckpt_file = resolve_checkpoint_path(cfg.ckpt_path, cfg.get("ckpt_name"))
    logger.info(f"Using checkpoint {ckpt_file}")

    model = ProteinaGP.load_from_checkpoint(ckpt_file, map_location="cpu", codesign=args.codesign)

    # Set seed
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    # Set inference variables and potentially load autoguidance
    nn_ag = None
    if cfg.get("guidance_weight", 1.0) != 1.0:
        if not cfg.get("autoguidance_ckpt_path"):
            raise ValueError("guidance_weight != 1.0 requires autoguidance_ckpt_path.")
        ckpt_ag_file = resolve_checkpoint_path(cfg.autoguidance_ckpt_path)
        model_ag = ProteinaGP.load_from_checkpoint(ckpt_ag_file)
        nn_ag = model_ag.nn
    model.configure_inference(cfg, nn_ag=nn_ag)
    dataloader = DataLoader(GenDataset(cfg.nbatch), batch_size=1)

    # Sample the model
    trainer = L.Trainer(accelerator="gpu", devices='auto')
    predictions = trainer.predict(model, dataloader)  # list of df
    df = pd.concat(predictions, ignore_index=True)
    csv_file = os.path.join(root_path, f"results_{trainer.global_rank}.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"Wrote results to {csv_file}")

    # run commands like
    # >>> import glob, pandas as pd
    # >>> run_name = 'infer_pdb'
    # >>> df = pd.concat([pd.read_csv(fn) for fn in glob.glob(f'inference/{run_name}/results_*.csv')])
    # >>> df.to_csv(f'inference/{run_name}/results.csv', index=False)
