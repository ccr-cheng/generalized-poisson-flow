"""Cluster designable unconditional samples or successful motif scaffolds.

Run after cal_designability.py (scRMSD <= 2 A) or cal_motif_success.py
(success == True); the input results_*.csv must contain pdb_path pointing to
the original generated PDBs. Examples::

    python script_utils/cal_diversity.py /path/to/run --mode unconditional --recopy
    python script_utils/cal_diversity.py /path/to/scaffolds --tasks 1bcf --recopy

Motif mode (the default) reads <samples_dir>/<task>/results_*.csv; unconditional
mode reads <samples_dir>/results_*.csv. Foldseek uses TM-score thresholds 0.5
(unconditional) or 0.6 (motif), writing artifacts alongside the input CSVs and
motif counts to clusters.json. Set FOLDSEEK_PATH if foldseek is not on PATH;
no reference database is needed. Use --recopy when the selected samples change.
"""

import os
import argparse
import json
import shlex
import shutil
import subprocess
import time
from glob import glob

import pandas as pd


def cal_unconditional_diversity(sample_dir, recopy=False):
    """Cluster unconditional designs from an explicit generation directory."""
    sample_dir = os.path.abspath(os.path.expanduser(sample_dir))
    df = pd.concat([pd.read_csv(fn) for fn in glob(os.path.join(sample_dir, 'results_*.csv'))])
    design_dir = os.path.join(sample_dir, 'dpdbs')
    if recopy and os.path.exists(design_dir):
        shutil.rmtree(design_dir)
    os.makedirs(design_dir, exist_ok=True)
    design_paths = df[df.scRMSD <= 2.0].pdb_path
    print(
        f'{sample_dir}: Number of designable: {len(design_paths)}, '
        f'designability: {len(design_paths) / len(df) if len(df) else 0.0:.4f}'
    )
    if len(design_paths) == 0:
        print('No designable samples. Skipping clustering.')
        return 0
    print(f'Copying files to {design_dir}')
    for fn in design_paths:
        dst_fn = os.path.join(design_dir, os.path.basename(fn))
        if not os.path.exists(dst_fn):
            shutil.copy2(fn, design_dir)

    foldseek_path = os.environ.get('FOLDSEEK_PATH', 'foldseek')
    cmd = [
        foldseek_path, 'easy-cluster', design_dir,
        os.path.join(sample_dir, 'res'), os.path.join(sample_dir, 'temp'),
        '--alignment-type', '1', '--cov-mode', '0', '--min-seq-id', '0',
        '--tmscore-threshold', '0.5',
    ]
    print(f'Running command: {shlex.join(cmd)}')
    start = time.time()
    log_path = os.path.join(sample_dir, 'log_diversity.txt')
    with open(log_path, 'w') as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f'Foldseek failed with exit code {ret}; see {log_path}')
    print(f'Success! Took {time.time() - start:.2f}s. Log saved to {log_path}')

    cnt = 0
    with open(os.path.join(sample_dir, 'res_rep_seq.fasta')) as f:
        for line in f:
            if line.startswith('>'):
                cnt += 1
    print(f'{sample_dir}: Number of clusters: {cnt}, diversity: {cnt / len(design_paths):.4f}')
    return cnt


def cal_motif_task_diversity(sample_dir, task, recopy=False):
    """Count unique successful motif scaffolds for one task below sample_dir."""
    sample_dir = os.path.abspath(os.path.expanduser(sample_dir))
    task_dir = os.path.join(sample_dir, task)
    df = pd.concat([pd.read_csv(fn) for fn in glob(os.path.join(task_dir, 'results_*.csv'))])
    df = df[df.success]

    if len(df) == 0:
        print(f'{task}: Number of designable: 0. Skipping clustering.')
        return 0

    design_dir = os.path.join(task_dir, 'dpdbs')
    if recopy:
        # Remove existing copies so only the current successful designs are clustered.
        if os.path.exists(design_dir):
            shutil.rmtree(design_dir)
    os.makedirs(design_dir, exist_ok=True)
    print(f'{task}: Number of designable: {len(df)}')
    print(f'Copying files to {design_dir}')
    for fn in df.pdb_path:
        dst_fn = os.path.join(design_dir, os.path.basename(fn))
        if not os.path.exists(dst_fn) or recopy:
            shutil.copy2(fn, design_dir)

    foldseek_path = os.environ.get('FOLDSEEK_PATH', 'foldseek')
    cmd = [
        foldseek_path, 'easy-cluster', design_dir,
        os.path.join(task_dir, 'res'), os.path.join(task_dir, 'temp'),
        '--alignment-type', '1', '--cov-mode', '0', '--min-seq-id', '0',
        '--tmscore-threshold', '0.6',
    ]
    print(f'Running command: {shlex.join(cmd)}')
    start = time.time()
    log_path = os.path.join(task_dir, 'log_diversity.txt')
    with open(log_path, 'w') as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f'Foldseek failed with exit code {ret}; see {log_path}')
    print(f'Success! Took {time.time() - start:.2f}s. Log saved to {log_path}')

    cnt = 0
    with open(os.path.join(task_dir, 'res_rep_seq.fasta')) as f:
        for line in f:
            if line.startswith('>'):
                cnt += 1
    print(f'{task}: Number of clusters: {cnt}')
    return cnt


def cal_motif_diversity(sample_dir, tasks=None, out_path='clusters.json', recopy=False):
    """Write motif-scaffolding counts, resolving relative out_path below sample_dir."""
    sample_dir = os.path.abspath(os.path.expanduser(sample_dir))
    if tasks is None or len(tasks) == 0:
        tasks = [entry.name for entry in os.scandir(sample_dir)
                 if entry.is_dir() and glob(os.path.join(entry.path, 'results_*.csv'))]
    tasks = sorted(tasks)
    if not tasks:
        raise ValueError(f'No task directories with results_*.csv found in {sample_dir}')
    print(f'Calculating diversity for: {tasks}')
    all_diversity = {task: cal_motif_task_diversity(sample_dir, task, recopy) for task in tasks}
    out_path = os.path.join(sample_dir, out_path)
    with open(out_path, 'w') as f:
        json.dump(all_diversity, f)
    print(f'Diversity results saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "samples_dir",
        type=str,
        help="Generation directory (unconditional) or parent of motif task directories."
    )
    parser.add_argument(
        "--mode", choices=["unconditional", "motif"], default="motif",
        help="Evaluation type (default: motif).",
    )
    parser.add_argument(
        "--tasks",
        nargs='*',
        type=str,
        default=[],
        help="Motif task names; defaults to task directories containing results_*.csv."
    )
    parser.add_argument(
        "--recopy",
        action='store_true',
        help="Whether to force recopy pdb files."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default='clusters.json',
        help="Motif counts JSON: absolute path or relative to samples_dir (default: clusters.json)."
    )
    args = parser.parse_args()
    if args.mode == 'unconditional':
        if args.tasks:
            parser.error('--tasks is only supported in motif mode')
        cal_unconditional_diversity(args.samples_dir, recopy=args.recopy)
    else:
        cal_motif_diversity(args.samples_dir, args.tasks, args.out_path, args.recopy)
