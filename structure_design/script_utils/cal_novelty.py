"""Measure structural novelty for unconditional designs or motif scaffolds.

Usage::

    python script_utils/cal_novelty.py /path/to/samples --db_path /path/to/foldseek/pdb
    python script_utils/cal_novelty.py /path/to/scaffolds/1bcf --db_path /path/to/foldseek/afdb --db afdb

Reads samples_dir/b{batch_idx}_i{idx}/b{batch_idx}_i{idx}.pdb or numbered
samples_dir/{idx}.pdb motif scaffolds. Use --pdb_glob for another layout.
The database prefix is supplied explicitly; FOLDSEEK_PATH or --foldseek_path
selects the executable. Writes out_<db>.tsv and log_novelty_<db>.txt in
samples_dir. The reported score is the mean best-hit TM-score (lower is more novel).
"""

import argparse
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd


def find_pdbs(sample_dir, pdb_glob=None):
    """Find generated structures without descending into evaluation artifacts."""
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
    return pdbs


def main(samples_dir, db_path, db='pdb', pdb_glob=None, foldseek_path=None):
    """Search generated structures against an explicitly selected Foldseek DB."""
    sample_dir = Path(samples_dir).expanduser().resolve()
    db_path = Path(db_path).expanduser().resolve()
    if not db or Path(db).name != db or db in {'.', '..'}:
        raise ValueError('db must be a filename label, not a path; use db_path for the database')
    pdbs = find_pdbs(sample_dir, pdb_glob)
    print(f'{sample_dir}: Number of pdbs: {len(pdbs)}')
    if len({path.name for path in pdbs}) != len(pdbs):
        raise ValueError('PDB filenames must be unique; evaluate motif targets separately')

    foldseek_path = os.path.expanduser(foldseek_path or os.environ.get('FOLDSEEK_PATH', 'foldseek'))
    out_path = sample_dir / f'out_{db}.tsv'
    log_path = sample_dir / f'log_novelty_{db}.txt'
    # A fresh query directory prevents a changed selection from reusing stale PDBs.
    with tempfile.TemporaryDirectory(prefix='.novelty_pdbs_', dir=sample_dir) as pdb_dir:
        for path in pdbs:
            shutil.copy2(path, pdb_dir)
        cmd = [
            foldseek_path, 'easy-search', pdb_dir, str(db_path),
            str(out_path), str(sample_dir / 'temp2'),
            '--alignment-type', '1', '--exhaustive-search',
            '--tmscore-threshold', '0.0', '--max-seqs', '10000000000',
            '--format-output', 'query,target,alntmscore,lddt',
        ]
        print(f'Running command: {shlex.join(cmd)}')
        start = time.time()
        with open(log_path, 'w') as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
    print(f'Success! Took {time.time() - start:.2f}s. Log saved to {log_path}')

    df = pd.read_csv(out_path, sep='\t', header=None, names=['query', 'target', 'tmscore', 'lddt'])
    novelty = df.groupby('query')['tmscore'].max().mean()
    print(f'{sample_dir}, {db}: novelty (mean best-hit TM-score): {novelty:.4f}')
    return novelty


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('samples_dir', help='Generated samples directory, or one motif target directory.')
    parser.add_argument('--db_path', required=True, help='Foldseek database prefix (not its containing directory).')
    parser.add_argument('--db', default='pdb', help='Database label used in output filenames (default: pdb).')
    parser.add_argument('--pdb_glob', help='PDB glob relative to samples_dir; quote patterns to prevent shell expansion.')
    parser.add_argument('--foldseek_path', help='Foldseek executable (default: FOLDSEEK_PATH or foldseek on PATH).')
    args = parser.parse_args()
    main(args.samples_dir, args.db_path, args.db, args.pdb_glob, args.foldseek_path)
