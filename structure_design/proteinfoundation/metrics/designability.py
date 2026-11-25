import os
import json
from typing import List, Optional

import einops
import torch
from loguru import logger
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers import logging as hf_logging
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb
from Bio.PDB import PDBParser
import biotite.structure.io as bsio

from proteinfoundation.utils.align_utils.align_utils import kabsch_align_ind
from proteinfoundation.utils.ff_utils.pdb_utils import from_pdb_string
from openfold.np.residue_constants import restype_3to1

hf_logging.set_verbosity_error()


def pdb_name_from_path(pdb_file_path):
    return pdb_file_path.strip(os.sep).split(os.sep)[-1][
        :-4
    ]  # Name of the pdb file without ".pdb" extension


# ProteinMPNN
## ## ## ## ## ## ## ## ## ## ## ##


def proteinmpnn_device_index(device: torch.device) -> Optional[int | str]:
    """Map a local CUDA device to the GPU selector for a ProteinMPNN subprocess."""
    if device.type != "cuda":
        return None
    local_device = device.index if device.index is not None else torch.cuda.current_device()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        return visible_devices.split(",")[local_device].strip()
    return local_device


def extract_gen_seqs(path_to_file: str) -> List[str]:
    """
    Extracts sequences from ProteinMPNN generation files.

    Args:
        path_to_file: Path to file with pmpnn output.

    Returns:
        List of sequences produced by pmpnn.
    """
    seqs = []
    with open(path_to_file, "r") as f:
        first = True  # Assuming first sequence is not a generation
        for line in f:
            if not line.startswith(">"):
                if first:
                    first = False
                    continue
                else:
                    seqs.append(line.strip())
    return seqs


def run_proteinmpnn(
        pdb_file_path: str,
        out_dir_root: str,
        sampling_temp: float = 0.1,
        num_seq_per_target: int = 8,
        seed: Optional[int] = None,
        ca_only: bool = True,
        verbose: bool = False,
        device_idx: Optional[int] = None,
        fix_mode: str = 'none',
        fix_json_path: Optional[str] = None,
) -> List[str]:
    """
    Just an interfact to ProteinMPNN.

    Args:
        pdb_file_path: path to PDB file
        out_dir_root: Path used to store produced sequences
        sampling_temp: Sampling temperature for ProteinMPNN
        num_seq_per_target: Number of sequences produced per target provided
        seed: Random seed used for sampling
        ca_only: Whether to only use alpha carbons
        verbose: Print stuff or not
        device_idx: Device index to use.

    Returns:
        List of sequences (strings)
    """
    name = pdb_name_from_path(pdb_file_path)

    python_exec = os.environ.get("PYTHON_EXEC")
    if python_exec is None:
        python_exec = "python"

    if device_idx is not None:
        python_exec = f"CUDA_VISIBLE_DEVICES={device_idx} {python_exec}"

    command = f"""
    {python_exec} ./ProteinMPNN/protein_mpnn_run.py \
        --pdb_path {pdb_file_path} \
        --pdb_path_chains A \
        --out_folder {out_dir_root} \
        --num_seq_per_target {num_seq_per_target} \
        --sampling_temp {sampling_temp} \
        --batch_size {num_seq_per_target} \
        --suppress_print {0 if verbose else 1} \
    """

    if ca_only:
        command += " --ca_only "
    if seed is not None:
        command += f" --seed {seed} "
    if not verbose:
        command += f" > /dev/null 2>&1"
    if fix_mode == 'pre':
        assert fix_json_path is not None, "fix_json_path must be provided if fix_mode is `pre`"
        command += f" --fixed_positions_jsonl {fix_json_path}"

    status = os.system(command)
    if status != 0:
        raise RuntimeError(f"Command failed with exit code: {status}")
    gen_seqs = extract_gen_seqs(os.path.join(out_dir_root, "seqs", name + ".fa"))
    if fix_mode == 'post':
        assert fix_json_path is not None, "fix_json_path must be provided if fix_mode is `post`"
        pdb_parser = PDBParser(QUIET=True)
        structure = pdb_parser.get_structure('structure', pdb_file_path)
        res_dict = {
            r.get_id()[1]: restype_3to1[r.get_resname()] for r in structure.get_residues()
        }
        with open(fix_json_path) as f:
            fix_idx = json.load(f)[os.path.basename(pdb_file_path)[:-4]]['A']
        fixed_seqs = []
        for seq in gen_seqs:
            s = list(seq)
            for idx in fix_idx:
                s[idx - 1] = res_dict[idx]  # Residue numbering starts at 1 in PDB files
            fixed_seqs.append("".join(s))
        gen_seqs = fixed_seqs
    return gen_seqs


## ## ## ## ## ## ## ## ## ## ## ##


# ESMFold
## ## ## ## ## ## ## ## ## ## ## ##


# I got this function from hugging face's ESM notebook example
def convert_outputs_to_pdb(outputs) -> List[str]:
    """Takes ESMFold outputs and converts them to a list of PDBs (as strings)."""
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def run_and_store_esm(
        name: str,
        seqs: List[str],
        path_to_esmfold_out: str,
        tokenizer=None,
        esm_model=None,
        return_esm_scores: bool = False,
) -> List[str]:
    """
    Runs ESMFold and stores results as PDB files.

    For now, runs with a single GPU, though not a big deal if we parallelie jobs (easily
    done with our inference pipeline).

    Args:
        name: name to use when storing
        seqs: List of sequences (strings)
        path_to_esmfold_out: Root directory to store outputs of ESMFold as PDBs

    Returns:
        List of paths (list of str) to PDB files
    """
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
        esm_model = esm_model.cuda()

    # Run ESMFold
    len(seqs)
    max_nres = max([len(x) for x in seqs])
    list_of_strings_pdb = []
    batch_size = 8
    num_batches = 1

    plddt_scores = []
    pae_scores = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        inputs = tokenizer(
            seqs[start_idx:end_idx],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        inputs = {k: inputs[k].cuda() for k in inputs}
        same_length = (len(set([len(x) for x in seqs[start_idx:end_idx]])) == 1)

        with torch.no_grad():
            _outputs = esm_model(**inputs)
            if return_esm_scores and same_length:
                plddt = _outputs["plddt"][:, :, 1].mean(dim=(-1)).cpu().numpy()
                pae = _outputs["predicted_aligned_error"].mean(dim=(-1, -2)).cpu().numpy()
                plddt_scores.extend(plddt.tolist())
                pae_scores.extend(pae.tolist())

        _list_of_strings_pdb = convert_outputs_to_pdb(_outputs)
        list_of_strings_pdb.extend(_list_of_strings_pdb)

    # Create out directory if not there
    if not os.path.exists(path_to_esmfold_out):
        os.makedirs(path_to_esmfold_out)

    # Store generations for each sequence
    out_esm_paths = []
    for i, pdb in enumerate(list_of_strings_pdb):
        fname = f"esm_{i + 1}.pdb"
        fdir = os.path.join(path_to_esmfold_out, fname)
        with open(fdir, "w") as f:
            f.write(pdb)
            out_esm_paths.append(fdir)
    if return_esm_scores:
        return out_esm_paths, plddt_scores, pae_scores
    return out_esm_paths


## ## ## ## ## ## ## ## ## ## ## ##


def load_pdb(fname: str) -> str:
    """Returns pdb stored in input file as string."""
    with open(fname, "r") as f:
        return from_pdb_string(f.read())


def rmsd_metric(
        coors_1_atom37: torch.Tensor,
        coors_2_atom37: torch.Tensor,
        mask_1_atom_37: Optional[torch.Tensor] = None,
        mask_2_atom_37: Optional[torch.Tensor] = None,
        mode: str = "ca",
        incl_ox: bool = False,
        align: bool = True,
) -> torch.Tensor:
    """
    Computes RMSD between two protein structures in the Atom37 represnetation.
    For now we only use mask to check whether we have all required atoms.

    Args:
        coors_1_atom37: First structure, shape [n, 37, 3]
        coors_2_atom37: Second structure, shape [n, 37, 3]
        mask_1_atom37: Binary mask of first structure, shape [n, 37]
        mask_2_atom37: Binary mask of first structure, shape [n, 37]
        mode: Modality to use, for now single option "ca", referring to only alpha
            carbon.
        incl_ox: Wehther to include oxygen atom, should be left to False
        align: Whether to align pointclouds before computing RMSD.

    Returns:
        RMSD value, as a Torch (float) tensor with a single element
    """
    assert coors_1_atom37.shape == coors_2_atom37.shape
    assert coors_1_atom37.shape[-1] == 3
    assert coors_1_atom37.shape[-2] == 37
    if mask_1_atom_37 is not None:
        assert mask_1_atom_37.shape == coors_1_atom37.shape[1:]
    if mask_2_atom_37 is not None:
        assert mask_2_atom_37.shape == coors_2_atom37.shape[1:]
    if mode == 'ca':
        idx_select = [1]  # [CA]
    elif mode == 'bb':
        idx_select = [0, 1, 2, 3]  # [CA]
    else:
        raise ValueError(f'Unknown mode {mode}')

    coors_1 = coors_1_atom37[:, idx_select, :]  # [n, natoms_sel, 3]
    coors_2 = coors_2_atom37[:, idx_select, :]  # [n, natoms_sel, 3]

    # Check all atoms actually present if we have mask
    for mask_atom_37 in [mask_1_atom_37, mask_2_atom_37]:
        if mask_atom_37 is not None:
            mask = mask_atom_37[:, idx_select]
            assert mask.sum() == mask.numel()

    # Compute RMSD (potentially) aligning structures
    coors_1 = einops.rearrange(coors_1, "n s t -> (n s) t")  # [n * natoms_sel, 3]
    coors_2 = einops.rearrange(coors_2, "n s t -> (n s) t")  # [n * natoms_sel, 3]

    if align:
        coors_1, coors_2 = kabsch_align_ind(coors_1, coors_2, ret_both=True)

    sq_err = (coors_1 - coors_2) ** 2
    return sq_err.sum(dim=-1).mean().sqrt().item()


def scRMSD(
        pdb_file_path: str,
        tmp_path: str = "./tmp/metrics/",
        num_seq_per_target: int = 8,
        pmpnn_sampling_temp: float = 0.1,
        ret_min=True,
        tokenizer=None,
        esm_model=None,
        device_idx: Optional[int] = None,
        fix_mode: str = 'none',
        fix_json_path: Optional[str] = None,
):
    """
    Evaluates self-consistency RMSD metrics for given pdb.

    Args:
        pdb_file_path: Path to PDB file.
        tmp_path: Path to store files produced by ProteinMPNN and ESMFold.
        num_seq_per_target: Number of sequences generated by ProteinMPNN per structure.
        pmpnn_sampling_temp: ProteinMPNN sampling temperature.
        ret_min: Whether to return min RMSD or a list of all values.

    Returns:
        Either best RMSD (scRMSD) or a list of all values for all generations, depending on
        the ret_min argument.
    """
    name = pdb_name_from_path(pdb_file_path)

    logger.info(f"Running ProteinMPNN for {name}")
    mpnn_gen_seqs = run_proteinmpnn(  # For now do not use keep ca_only=False
        pdb_file_path,
        tmp_path,
        num_seq_per_target=num_seq_per_target,
        sampling_temp=pmpnn_sampling_temp,
        device_idx=device_idx,
        fix_mode=fix_mode,
        fix_json_path=fix_json_path,
    )  # List of sequences

    logger.info(f"Running ESMFold for {name}")
    out_esm_paths, plddts, paes = run_and_store_esm(
        name, mpnn_gen_seqs, tmp_path, tokenizer=tokenizer, esm_model=esm_model, return_esm_scores=True
    )
    # List of paths to PDBs

    # Compute RMSDs
    rmsds = []

    # Load generated
    gen_prot = load_pdb(pdb_file_path)

    # Load ESMs
    for out_esm in out_esm_paths:
        rec_prot_esm = load_pdb(out_esm)
        gen_coors = torch.Tensor(gen_prot.atom_positions)
        rec_coors = torch.Tensor(rec_prot_esm.atom_positions)
        rmsds.append(rmsd_metric(gen_coors, rec_coors))  # rmsd_ca

    if ret_min:
        return min(rmsds), max(plddts), min(paes)
    return rmsds, plddts, paes


def codesignability_score(esm_model, aa, pdb_path, esm_path):
    """
    Computes the codesignability score based on ESMFold pLDDT and RMSD.
    :param esm_model: ESMFold model
    :param aa: input amino acid sequence tensor, should only contain one sequence
    :param pdb_path: path to the generated structure pdb file
    :param esm_path: path to store the esmfold predicted structure pdb file
    :return: tuple of (plddt, rmsd)
    """
    try:
        with torch.no_grad():
            output = esm_model(input_ids=aa.unsqueeze(0))
        pdb_str = convert_outputs_to_pdb(output)[0]
        with open(esm_path, "w") as f:
            f.write(pdb_str)
        struct = bsio.load_structure(esm_path, extra_fields=['b_factor'])
        plddt = struct.b_factor.mean()
        gen_prot = load_pdb(pdb_path)
        esm_prot = load_pdb(esm_path)
        gen_coors = torch.Tensor(gen_prot.atom_positions)
        esm_coors = torch.Tensor(esm_prot.atom_positions)
        rmsd = rmsd_metric(gen_coors, esm_coors)
    except Exception as e:
        logger.info(f"Error generating ESMFold structure for sample {esm_path.split('/')[-1]}: {e}")
        plddt = 0
        rmsd = 30.0
    return plddt, rmsd
