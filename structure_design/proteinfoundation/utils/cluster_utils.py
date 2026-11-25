# inspired from https://github.com/a-r-j/graphein/blob/master/graphein/ml/datasets/pdb_data.py

import pathlib
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from typing import Dict, List, Literal, Tuple

import pandas as pd
import torch
import torch_geometric
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from torch.utils.data import DistributedSampler


@rank_zero_only
def log_info(msg):
    logger.info(msg)


class ClusterSampler(DistributedSampler):
    """Shard clusters once, then choose one sequence from each assigned cluster.

    Inheriting DistributedSampler prevents Lightning from wrapping and sharding
    this sampler a second time, and lets Lightning advance its epoch on resume.
    """

    def __init__(
            self,
            dataset: torch_geometric.data.Dataset,
            clusterid_to_seqid_mapping: Dict[str, List[str]],
            sampling_mode: Literal["cluster-random", "cluster-reps"],
            shuffle: bool = True,
            drop_last: bool = False,
            *,
            seed: int = 0,
            num_replicas: int = None,
            rank: int = None,
    ):
        if sampling_mode not in ("cluster-random", "cluster-reps"):
            raise ValueError(f"Unknown cluster sampling mode: {sampling_mode}")
        self.sequence_dataset = dataset
        self.clusterid_to_seqid_mapping = clusterid_to_seqid_mapping
        self.cluster_names = sorted(clusterid_to_seqid_mapping)
        self.sampling_mode = sampling_mode
        if dataset.database in ("pdb", "scop"):
            self.sequence_id_to_idx = {
                fname.split(".")[0]: i for i, fname in enumerate(dataset.file_names)
            }
        elif dataset.database == "pinder":
            self.sequence_id_to_idx = dataset.pinder_id_to_idx
        else:
            self.sequence_id_to_idx = dataset.protein_to_idx
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size() if distributed else 1
        if rank is None:
            rank = torch.distributed.get_rank() if distributed else 0
        # The distributed sampler partitions cluster indices, not sequence indices.
        super().__init__(
            self.cluster_names, num_replicas=num_replicas, rank=rank,
            shuffle=shuffle, seed=seed, drop_last=drop_last,
        )

    def __iter__(self):
        for position, cluster_index in enumerate(super().__iter__()):
            cluster_name = self.cluster_names[cluster_index]
            if self.sampling_mode == "cluster-reps":
                sequence_id = cluster_name
            else:
                # A private, per-cluster epoch seed is independent of global RNG
                # state and gives a padded cluster the same member on every rank.
                rng = random.Random((self.seed + self.epoch) * len(self.cluster_names) + cluster_index)
                sequence_id = rng.choice(self.clusterid_to_seqid_mapping[cluster_name])
            if position == 0:
                logger.info(
                    f"First cluster sampling: sampling {sequence_id} from cluster {cluster_name}, rank {self.rank}"
                )
            yield self.sequence_id_to_idx[sequence_id]


def split_dataframe(
        df: pd.DataFrame,
        splits: List[str],
        ratios: List[float],
        leftover_split: int = 0,
        seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Split a DataFrame into multiple parts based on specified split ratios.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        splits (List[str]): Names of the resulting splits.
        ratios (List[float]): Ratios to split df into. Must sum to 1.0.
        leftover_split (int): Index of split to assign leftover rows to.
            Defaults to 0.
        seed (int): Random seed for shuffling. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping split names to
            DataFrame splits.

    Raises:
        AssertionError: If len(splits) != len(ratios) or sum(ratios) != 1.
    """
    assert len(splits) == len(ratios), "Number of splits must equal number of ratios"
    assert sum(ratios) == 1, "Split ratios must sum to 1"

    # Calculate size of each split
    split_sizes = [int(len(df) * ratio) for ratio in ratios]

    # Assign leftover rows to specified split
    split_sizes[leftover_split] += len(df) - sum(split_sizes)

    # Shuffle DataFrame rows
    df = df.sample(frac=1, random_state=seed)

    # Split DataFrame into parts
    split_dfs = {}
    start = 0
    for split, size in zip(splits, split_sizes):
        split_dfs[split] = df.iloc[start: start + size]
        start += size

    return split_dfs


def merge_dataframe_splits(
        df1: pd.DataFrame, df2: pd.DataFrame, list_columns: List[str]
) -> pd.DataFrame:
    """
    Merge two DataFrame splits on all columns except 'split'.

    Args:
        df1 (pd.DataFrame): First DataFrame split to merge.
        df2 (pd.DataFrame): Second DataFrame split to merge.
        list_columns (List[str]): Columns containing lists to convert to tuples.

    Returns:
        pd.DataFrame: Merged DataFrame containing rows in both splits.
    """
    # Convert list columns to tuples for merging
    for df in [df1, df2]:
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(tuple)

    # Merge the two DataFrames
    merge_cols = [c for c in df1.columns if c != "split"]
    merged_df = pd.merge(df1, df2, on=merge_cols, how="inner")

    # Convert tuple columns back to lists
    for df in [df1, df2]:
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(list)

    return merged_df


def cluster_sequences(
        fasta_input_filepath: str,
        cluster_output_filepath: str = None,
        min_seq_id: float = 0.3,
        coverage: float = 0.8,
        overwrite: bool = False,
        silence_mmseqs_output: bool = True,
        efficient_linclust: bool = False,
) -> None:
    """
    Cluster protein sequences in a DataFrame using MMseqs2.

    Args:
        fasta_input_file (str): Fasta File path containing protein sequences.
        cluster_output_filepath (str): Path to write clustering results. If None, defaults to
            "cluster_rep_seq_id_{min_seq_id}_c_{coverage}.fasta".
        min_seq_id (float): Minimum sequence identity for clustering. Defaults to 0.3.
        coverage (float): Minimum coverage for clustering. Defaults to 0.8.
        overwrite (bool): Whether to overwrite existing cluster file. Defaults to False.
        silence_mmseqs_output (bool): Whether to silence MMseqs2 output. Defaults to True.
        efficient_linclust (bool): Whether to use efficient linclust for clustering for large datasets. Defaults to False.
    """
    if cluster_output_filepath is None:
        cluster_output_filepath = f"cluster_rep_seq_id_{min_seq_id}_c_{coverage}.fasta"

    fasta_input_filepath = pathlib.Path(fasta_input_filepath).resolve()
    cluster_fasta_path = pathlib.Path(cluster_output_filepath).resolve()
    cluster_tsv_path = cluster_fasta_path.with_suffix(".tsv")
    if not overwrite and cluster_fasta_path.exists() and cluster_tsv_path.exists():
        try:
            validate_sequence_clusters(fasta_input_filepath, cluster_fasta_path, cluster_tsv_path)
            return
        except ValueError as error:
            logger.warning(f"Rebuilding invalid sequence clusters: {error}")

    mmseqs = shutil.which("mmseqs")
    if mmseqs is None:
        raise FileNotFoundError("MMseqs2 not found on PATH; add .tools/mmseqs2/bin or install MMseqs2.")
    cluster_fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mmseqs-", dir=cluster_fasta_path.parent) as directory:
        workdir = pathlib.Path(directory)
        prefix = workdir / "clusters"
        command = [
            mmseqs, "easy-linclust" if efficient_linclust else "easy-cluster",
            str(fasta_input_filepath), str(prefix), str(workdir / "tmp"),
            "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "1",
        ]
        try:
            subprocess.run(
                command, cwd=workdir, check=True, text=True,
                stdout=subprocess.PIPE if silence_mmseqs_output else None,
                stderr=subprocess.STDOUT if silence_mmseqs_output else None,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"MMseqs2 failed with exit code {error.returncode}: {command}. "
                f"{(error.stdout or '')[-4000:]}"
            ) from error
        generated_fasta = pathlib.Path(f"{prefix}_rep_seq.fasta")
        generated_tsv = pathlib.Path(f"{prefix}_cluster.tsv")
        validate_sequence_clusters(fasta_input_filepath, generated_fasta, generated_tsv)
        generated_fasta.replace(cluster_fasta_path)
        generated_tsv.replace(cluster_tsv_path)


def validate_sequence_clusters(input_fasta, representative_fasta, cluster_tsv):
    """Require a complete partition of input IDs and consistent representatives."""
    paths = [pathlib.Path(path) for path in (input_fasta, representative_fasta, cluster_tsv)]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty clustering artifact: {path}")
    inputs = fasta_to_df(paths[0])
    representatives = fasta_to_df(paths[1])
    for label, frame in (("input", inputs), ("representative", representatives)):
        if frame.empty or frame["id"].duplicated().any() or frame["sequence"].eq("").any():
            raise ValueError(f"Invalid {label} FASTA: IDs must be unique and sequences nonempty")
    mapping = read_cluster_tsv(paths[2])
    representative_ids = set(representatives["id"])
    if set(mapping) != representative_ids:
        raise ValueError("Cluster TSV representative IDs do not match representative FASTA")
    members = [member for values in mapping.values() for member in values]
    counts = Counter(members)
    expected_ids = set(inputs["id"])
    if set(counts) != expected_ids or any(count != 1 for count in counts.values()):
        missing = sorted(expected_ids - set(counts))[:10]
        extra = sorted(set(counts) - expected_ids)[:10]
        duplicates = [key for key, count in counts.items() if count != 1][:10]
        raise ValueError(f"Invalid cluster membership: missing={missing}, extra={extra}, duplicates={duplicates}")
    input_sequences = inputs.set_index("id")["sequence"].to_dict()
    for row in representatives.itertuples():
        if row.id not in mapping[row.id] or input_sequences.get(row.id) != row.sequence:
            raise ValueError(f"Representative {row.id} is absent from its cluster or has an incorrect sequence")
    return mapping


def split_sequence_clusters(
        df, splits, ratios, leftover_split=0, seed=42
) -> Dict[str, pd.DataFrame]:
    """
    Split clustered sequences into train/val/test sets.

    Args:
        df (pd.DataFrame): DataFrame with clustered sequences.
        splits (List[str]): Names of splits, e.g. ["train", "val", "test"].
        ratios (List[float]): Ratios for each split. Must sum to 1.0.
        leftover_split (int): Index of split to assign leftover sequences.
            Defaults to 0.
        seed (int): Random seed. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping split names to DataFrames that contain randomly-split representative sequences.
    """
    # Split clusters into subsets
    cluster_splits = split_dataframe(df, splits, ratios, leftover_split, seed)
    # Get representative sequences for each split
    split_dfs = {}
    for split, cluster_df in cluster_splits.items():
        rep_seqs = cluster_df.representative_sequences()
        split_dfs[split] = rep_seqs

    return split_dfs


def expand_cluster_splits(
        cluster_rep_splits: Dict[str, pd.DataFrame],
        clusterid_to_seqid_mapping: Dict[str, List[str]],
) -> Dict[str, pd.DataFrame]:
    """
    Expand the cluster representative splits to full cluster splits based on the provided cluster dictionary.

    Args:
        cluster_rep_splits: A dictionary containing DataFrames for each split (e.g., 'train', 'val', 'test').
            Each DataFrame should have an 'id' column representing the cluster representative IDs.
        clusterid_to_seqid_mapping: A dictionary mapping cluster representative IDs to their corresponding cluster member IDs.

    Returns:
        A new dictionary of DataFrames with expanded 'id' columns based on the cluster dictionary.
        The 'id' column in the original DataFrames is replaced with the corresponding cluster member IDs.
        If df_sequences is provided, the additional columns from df_sequences are added to the resulting DataFrames.

    """
    full_cluster_splits = {}
    split_clusterid_to_seqid_mapping = {}

    for split_name, split_df in cluster_rep_splits.items():
        # Create a dictionary to store the cluster members for the current split
        split_cluster_members = {}

        for rep_id in split_df["id"]:
            if rep_id in clusterid_to_seqid_mapping:
                split_cluster_members[rep_id] = clusterid_to_seqid_mapping[rep_id]
            else:
                raise ValueError(
                    f"ID {rep_id} is a representative in the splits, but not in the cluster_dicts"
                )

        # Create a DataFrame with the cluster representative IDs and their corresponding cluster member IDs for the current split
        split_cluster_members_df = pd.DataFrame(
            [
                (rep_id, member_id)
                for rep_id, member_ids in split_cluster_members.items()
                for member_id in member_ids
            ],
            columns=["cluster_id", "id"],
        )
        # Split the 'id' column into 'pdb' and 'chain' columns
        if len(split_cluster_members_df) > 0:
            split_cluster_members_df[["pdb", "chain"]] = split_cluster_members_df[
                "id"
            ].str.split("_", n=1, expand=True)
        else:
            split_cluster_members_df["pdb"] = pd.Series(dtype=str)
            split_cluster_members_df["chain"] = pd.Series(dtype=str)
        # Add the expanded DataFrame to the dictionary
        full_cluster_splits[split_name] = split_cluster_members_df
        # Add the split-specific cluster_dict to the dictionary
        split_clusterid_to_seqid_mapping[split_name] = split_cluster_members
    return full_cluster_splits, split_clusterid_to_seqid_mapping


def read_cluster_tsv(cluster_tsv_filepath: pathlib.Path) -> Dict[str, List[str]]:
    """
    Read the cluster TSV file that is output from mmseqs2 and construct a dictionary mapping cluster representatives to sequence IDs.

    Args:
        cluster_tsv_filepath (pathlib.Path): The path to the cluster TSV file.

    Returns:
        Dict[str, List[str]]: A dictionary mapping cluster representatives to lists of sequence IDs.
    """
    cluster_dict = {}
    with open(cluster_tsv_filepath, "r") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.strip().split("\t")
            if len(fields) != 2 or not all(fields):
                raise ValueError(f"Invalid cluster TSV row {line_number} in {cluster_tsv_filepath}")
            cluster_name, sequence_name = fields
            cluster_dict.setdefault(cluster_name, []).append(sequence_name)
    return cluster_dict


def setup_clustering_file_paths(
        data_dir: str,
        file_identifier: str,
        split_sequence_similarity: float,
) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Set up file paths for the fasta file, cluster file, and cluster TSV file.

    Args:
        data_dir (str): The directory where the files will be stored.
        file_identifier (str): The identifier used to name the files.
        split_sequence_similarity (float): The sequence similarity threshold for splitting.

    Returns:
        Tuple[pathlib.Path, pathlib.Path, pathlib.Path]: A tuple containing the file paths for
            the input fasta file, cluster file, and cluster TSV file.
    """
    input_fasta_filepath = pathlib.Path(data_dir) / f"seq_{file_identifier}.fasta"
    cluster_filepath = (
            pathlib.Path(data_dir)
            / f"cluster_seqid_{split_sequence_similarity}_{file_identifier}.fasta"
    )
    cluster_tsv_filepath = cluster_filepath.with_suffix(".tsv")
    return input_fasta_filepath, cluster_filepath, cluster_tsv_filepath


def df_to_fasta(df: pd.DataFrame, output_file: str) -> None:
    """
    Convert a pandas DataFrame to a FASTA file.

    Args:
        df (pd.DataFrame): DataFrame containing 'id' and 'sequence' columns.
        output_file (str): Path to the output FASTA file.

    Returns:
        None
    """
    output_file = pathlib.Path(output_file)
    with tempfile.NamedTemporaryFile(mode="w", dir=output_file.parent, suffix=".fasta.tmp", delete=False) as handle:
        temporary_path = pathlib.Path(handle.name)
        try:
            for _, row in df.iterrows():
                handle.write(f">{row['id']}\n{row['sequence']}\n")
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(output_file)


def fasta_to_df(fasta_input_file: str) -> pd.DataFrame:
    """
    Convert a FASTA file to a pandas DataFrame.

    Args:
        fasta_input_file (str): Path to the input FASTA file.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'sequence' columns.
    """
    data = []
    with open(fasta_input_file, "r") as file:
        sequence_id = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id is not None:
                    data.append([sequence_id, "".join(sequence)])
                sequence_id = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if sequence_id is not None:
            data.append([sequence_id, "".join(sequence)])

        df = pd.DataFrame(data, columns=["id", "sequence"])
    return df
