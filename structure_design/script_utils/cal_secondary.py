"""Calculate mean secondary-structure fractions for either generation task.

Usage::

    python script_utils/cal_secondary.py /path/to/unconditional/samples
    python script_utils/cal_secondary.py /path/to/scaffolds/1bcf

Reads samples_dir/b{batch_idx}_i{idx}/b{batch_idx}_i{idx}.pdb for unconditional
generation or samples_dir/{idx}.pdb for one motif target. Use --pdb_glob for
another layout. Writes alpha-helix, beta-sheet, and coil fractions to
samples_dir/ss_stats.json, excluding sequence-design/refolding artifacts.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm
import numpy as np

# Support direct invocation from outside the repository directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def calc_sample_ss(sample_dir, pdb_glob=None):
    """Average per-structure SS fractions over the selected generated PDBs."""
    sample_dir = Path(sample_dir).expanduser().resolve()
    if pdb_glob is not None:
        pdbs = sample_dir.glob(pdb_glob)
    else:
        pdbs = [
            *sample_dir.glob('b*_i*/b*_i*.pdb'),
            *(path for path in sample_dir.glob('*.pdb') if path.stem.isdigit()),
        ]
    pdbs = sorted({path for path in pdbs if path.is_file()})
    if not pdbs:
        raise FileNotFoundError(
            f'No generated PDBs found in {sample_dir}. Expected batch subdirectories '
            'or numbered motif PDBs; use --pdb_glob for a different layout.'
        )

    from proteinfoundation.metrics.secondary import calc_secondary_structure

    alpha, beta, coil = [], [], []
    for path in tqdm(pdbs):
        a, b, c = calc_secondary_structure(str(path))
        alpha.append(a)
        beta.append(b)
        coil.append(c)
    return float(np.mean(alpha)), float(np.mean(beta)), float(np.mean(coil))


def main(samples_dir, pdb_glob=None):
    sample_dir = Path(samples_dir).expanduser().resolve()
    a, b, c = calc_sample_ss(sample_dir, pdb_glob)
    print(f'{sample_dir}: alpha={a:.3f}, beta={b:.3f}, coil={c:.3f}')
    with open(sample_dir / 'ss_stats.json', 'w') as f:
        json.dump({'alpha': a, 'beta': b, 'coil': c}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('samples_dir', help='Generated samples directory, or one motif target directory.')
    parser.add_argument('--pdb_glob', help='PDB glob relative to samples_dir; quote patterns to prevent shell expansion.')
    args = parser.parse_args()
    main(args.samples_dir, args.pdb_glob)
