import os
import torch
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import random

import biotite.structure.io as strucio
from proteinfoundation.utils.align_utils.align_utils import mean_w_mask
from proteinfoundation.utils.coors_utils import ang_to_nm


def parse_contig_from_pdb(
        pdb_text: str,
        motif_chain_map: Dict[Tuple[str, int, int], str] = None,
):
    """
    Parse contig_string, contig_array, min_length, max_length
    from PDB text with REMARK 999 INPUT lines.

    Args:
        pdb_text: full pdb file content
        motif_chain_map:
            optional remapping:
            {(orig_chain, start, end): new_chain}
            e.g. {("A", 92, 99): "A"}

    Returns:
        contig_string: str
        contig_array: list of dict
        min_length: int
        max_length: int
    """

    contig_array = []
    min_length = None
    max_length = None

    for line in pdb_text.splitlines():
        if line.startswith("REMARK 999 INPUT"):
            tokens = line.split()

            # linker: INPUT 8 15
            if len(tokens) == 5:
                min_length, max_length = map(int, tokens[-2:])
                contig_array.append({
                    "type": "linker",
                    "min": min_length,
                    "max": max_length,
                })

            # motif: INPUT A 92 99
            elif len(tokens) == 6:
                chain = tokens[-3]
                start, end = map(int, tokens[-2:])

                if motif_chain_map and (chain, start, end) in motif_chain_map:
                    chain = motif_chain_map[(chain, start, end)]

                contig_array.append({
                    "type": "motif",
                    "chain": chain,
                    "start": start,
                    "end": end,
                })

        elif line.startswith("REMARK 999 MINIMUM TOTAL LENGTH"):
            min_length = int(line.split()[-1])

        elif line.startswith("REMARK 999 MAXIMUM TOTAL LENGTH"):
            max_length = int(line.split()[-1])

    # build contig_string
    contig_parts = []
    for seg in contig_array:
        if seg["type"] == "linker":
            contig_parts.append(f"{seg['min']}-{seg['max']}")
        else:
            contig_parts.append(
                f"{seg['chain']}{seg['start']}-{seg['end']}"
            )

    contig_string = "/".join(contig_parts)

    return contig_string, min_length, max_length


def contig_string_to_array(contig_string: str) -> List[Dict]:
    contig_array = []
    for part in contig_string.split('/'):
        if not part:
            continue

        if part[0].isalpha():
            chain = part[0]
            span = part[1:]
            if '-' in span:
                start, end = map(int, span.split('-'))
                length = end - start + 1
            else:
                length = 1
            contig_array.append({
                'type': 'motif',
                'length': length,
            })
        else:
            span = part
            if '-' in span:
                min_length, max_length = map(int, span.split('-'))
            else:
                min_length, max_length = int(span), int(span)
            contig_array.append({
                'type': 'linker',
                'min_length': min_length,
                'max_length': max_length,
            })
    return contig_array


def contig_info_to_string(contig_types: list, contig_lengths: list) -> str:
    seg_start = 1
    seg_strs = []
    for contig_type, contig_length in zip(contig_types, contig_lengths):
        seg_end = seg_start + contig_length - 1
        if contig_type == 0:
            seg_strs.append(f"{seg_start}-{seg_end}")
        else:
            seg_strs.append(f"A{seg_start}-{seg_end}")
        seg_start = seg_end + 1
    seg_str = '/'.join(seg_strs)
    return seg_str


def build_insert_mask(contig_type, contig_length):
    # contig_info: shape (m, 2) -> [seg_type, seg_len]
    total_len = int(contig_length.sum())
    ins_mask = np.zeros(total_len + 1, dtype=bool)  # disable insertion by default

    assert contig_type[0] == 0 and contig_type[-1] == 0, "The first and last segment must be linkers"

    pos = 0  # pointer to the insertion position and the start of the current residue
    for seg_type, seg_len in zip(contig_type, contig_length):
        seg_len = int(seg_len)

        if seg_type == 0:
            # Include both linker boundaries, including a zero-length linker.
            # Adjacent motif segments without a linker cannot receive insertions.
            ins_mask[pos:pos + seg_len + 1] = True

        # move to the next segment (pos points to the insertion position after the current segment)
        pos += seg_len

    # the last insertion position is allowed
    ins_mask[pos] = True
    return ins_mask


def contig_info_to_insert_mask(contig_types: list, contig_lengths: list) -> np.ndarray:
    B = len(contig_types)
    max_length = max(sum(contig_lengths[i]) for i in range(B))
    insert_mask = np.zeros((B, max_length + 1), dtype=bool)
    for i, (contig_type, contig_length) in enumerate(zip(contig_types, contig_lengths)):
        insert_mask_i = build_insert_mask(contig_type, contig_length)
        insert_mask[i, :len(insert_mask_i)] = insert_mask_i
    return insert_mask


def save_contig_info_csv(pdb_path, motif_task_name, contig_types, contig_lengths, outpath=None, segment_order='A'):
    pdb_name = pdb_path.split('/')[-1].split('.')[0]

    # Create a list of dictionaries to be converted into a DataFrame
    # Each dictionary represents a row in the CSV file
    contigs = []
    for contig_type, contig_length in zip(contig_types, contig_lengths):
        contigs.append(contig_info_to_string(contig_type, contig_length))

    data = [
        {
            'pdb_name': pdb_name,
            'sample_num': index,
            'contig': value,
            'redesign_positions': ' ',
            'segment_order': segment_order
        }
        for index, value in enumerate(contigs)
    ]

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    if outpath is None:
        outpath = f"./motif_info.csv"
    else:
        outpath = os.path.join(outpath, f"motif_info.csv")

    # Save the DataFrame to a CSV file
    df.to_csv(outpath, index=False)


def motif_extract(
        position: str,
        structure_path,
        atom_part: Optional[str] = "CA",
        split_char: str = "/",
        motif_only=False,
):
    """Extracting motif positions from input protein structure.

    Args:
        position (str): Motif region of input protein. DEMO: "A1-7/A28-79" corresponds defines res1-7 and res28-79 in chain A to be motif.
        structure_path (Union[str, None]): Input protein structure, can either be a path or an AtomArray.
        atom_part (str, optional): _description_. Defaults to "all".
        split_char (str): Spliting character between discontinuous motifs. Defaults to "/".

    Returns:
        motif (biotite.structure.AtomArray): The motif positions extracted by user-specified way (all-atom / CA / backbone)
    """

    position = position.split(split_char)
    ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    if isinstance(structure_path, str):
        array = strucio.load_structure(structure_path, model=1)
    else:
        array = structure_path
    motif_array = []

    if motif_only:
        seen = set()
        for i in position:
            chain_id = i[0]
            if chain_id not in ALPHABET or chain_id in seen:
                continue
            seen.add(chain_id)  # used for 1QJG multiple A chains
            if atom_part == "all-atom":
                motif_array.append(array[(array.chain_id == chain_id) & (array.hetero == False)])
            elif atom_part == "CA":
                motif_array.append(
                    array[(array.chain_id == chain_id) & (array.hetero == False) & (array.atom_name == "CA")])
            elif atom_part == "backbone":
                motif_array.append(array[(array.chain_id == chain_id) & (array.hetero == False) & (
                        (array.atom_name == "N") | (array.atom_name == "CA") | (array.atom_name == "C") | (
                        array.atom_name == "O"))])
    else:
        for i in position:
            chain_id = i[0]
            if chain_id not in ALPHABET:
                continue
            i = i.replace(chain_id, "")
            if "-" not in i:  # Single-residue motif
                start = end = int(i)
            else:
                start, end = i.split("-")
                start, end = int(start), int(end)

            if atom_part == "all-atom":
                motif_array.append(array[(array.chain_id == chain_id) & (array.res_id <= end) & (
                        array.res_id >= start) & (array.hetero == False)])
            elif atom_part == "CA":
                motif_array.append(array[(array.chain_id == chain_id) & (array.res_id <= end) & (
                        array.res_id >= start) & (array.hetero == False) & (array.atom_name == "CA")])
            elif atom_part == "backbone":
                motif_array.append(array[(array.chain_id == chain_id) & (array.res_id <= end) & (
                        array.res_id >= start) & (array.hetero == False) & (
                                                 (array.atom_name == "N") | (array.atom_name == "CA") | (
                                                 array.atom_name == "C") | (array.atom_name == "O"))])

    motif = motif_array[0]
    for i in range(len(motif_array) - 1):
        motif += motif_array[i + 1]

    return motif


class SingleMotifFactory:
    def __init__(self,
                 motif_prob: float = 1.0,  # 0.8,
                 motif_min_pct_res: float = 0.05,
                 motif_max_pct_res: float = 0.5,
                 motif_min_n_seg: int = 1,
                 motif_max_n_seg: int = 4):
        """
        Initialize SingleMotifFactory with default parameters.

        Args:
            motif_prob (float, optional): Motif probability. Defaults to 0.8.
            motif_min_pct_res (float, optional): Minimum percentage of residues in motif. Defaults to 0.05.
            motif_max_pct_res (float, optional): Maximum percentage of residues in motif. Defaults to 0.5.
            motif_min_n_seg (int, optional): Minimum number of segments in motif. Defaults to 1.
            motif_max_n_seg (int, optional): Maximum number of segments in motif. Defaults to 4.
        """
        self.motif_prob = motif_prob
        self.motif_min_pct_res = motif_min_pct_res
        self.motif_max_pct_res = motif_max_pct_res
        self.motif_min_n_seg = motif_min_n_seg
        self.motif_max_n_seg = motif_max_n_seg

    def __call__(self, batch, zeroes=False, inpaint_x0=False):
        return self.create_batch_motif(batch, zeroes, inpaint_x0)

    def create_batch_motif(self, batch, zeroes=False, inpaint_x0=False):
        result = {}
        if "mask" in batch:
            mask = batch["mask"]
        else:
            mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        batch_size, num_residues = mask.shape
        if zeroes or random.random() > self.motif_prob:
            motif_sequence_mask = torch.zeros((batch_size, num_residues), dtype=torch.bool)
            motif_structure_mask = torch.zeros((batch_size, num_residues, num_residues), dtype=torch.bool)
            result['fixed_sequence_mask'] = motif_sequence_mask.to(mask.device)
            result['fixed_structure_mask'] = motif_structure_mask.to(mask.device)
            result['x_motif'] = torch.zeros((batch_size, num_residues, 3)).to(mask.device)
            return result

        if "x_1" in batch:
            x_1 = batch["x_1"]  # [b, n, 3]
        else:
            x_1 = batch["coords"][:, :, 1, :]  # [b, n, 3]
            x_1 = ang_to_nm(x_1)

        batch_num_residues = mask.sum(-1).cpu().numpy()

        motif_n_res_batch = (
                np.random.rand(len(batch_num_residues)) * (
                batch_num_residues * self.motif_max_pct_res - batch_num_residues * self.motif_min_pct_res
        ) + batch_num_residues * self.motif_min_pct_res
        ).astype(np.int64)

        motif_n_seg_batch = (
                np.random.rand(len(motif_n_res_batch)) * (
                np.minimum(motif_n_res_batch, np.full_like(motif_n_res_batch, self.motif_max_n_seg))
                - self.motif_min_n_seg + 1
        ) + self.motif_min_n_seg
        ).astype(np.int64)

        indices_batch = []
        motif_seg_lens_batch = []
        for i in range(len(motif_n_res_batch)):
            indices = np.sort(np.random.choice(motif_n_res_batch[i] - 1, motif_n_seg_batch[i] - 1, replace=False) + 1)
            indices = np.concatenate([[0], indices, [motif_n_res_batch[i]]])
            indices_batch.append(indices)

            # Calculate motif segment lengths
            motif_seg_lens = indices[1:] - indices[:-1]
            motif_seg_lens_batch.append(motif_seg_lens)

        motif_sequence_masks = []
        # motif_structure_masks = []

        for i in range(len(motif_seg_lens_batch)):
            segs = [''.join(['1'] * l) for l in motif_seg_lens_batch[i]]
            segs.extend(['0'] * (batch_num_residues[i] - motif_n_res_batch[i]).astype(np.int64))
            random.shuffle(segs)
            motif_sequence_mask = torch.tensor([int(elt) for elt in ''.join(segs)], dtype=torch.bool)
            motif_sequence_masks.append(motif_sequence_mask)

        motif_sequence_masks = torch.nn.utils.rnn.pad_sequence(
            motif_sequence_masks, batch_first=True, padding_value=False
        )
        motif_structure_masks = motif_sequence_masks[:, :, None] * motif_sequence_masks[:, None, :]
        result['fixed_sequence_mask'] = motif_sequence_masks.to(mask.device)
        result['fixed_structure_mask'] = motif_structure_masks.to(mask.device)
        result['x_motif'] = x_1.clone()
        # ! Center the conditional Motif
        result['x_motif'] = (result['x_motif'] - mean_w_mask(result['x_motif'], result['fixed_sequence_mask'],
                                                             keepdim=True)) * result['fixed_sequence_mask'][..., None]
        # ! Translate x_1 so that the motif is in the center
        batch["x_1"] = (x_1 - mean_w_mask(x_1, result['fixed_sequence_mask'], keepdim=True)) * mask[..., None]
        # Optionally inpaint x_0 with the centered motif so the reference keeps the motif fixed
        if inpaint_x0 and "x_0" in batch:
            x_0 = batch["x_0"]
            x_0 = x_0 * (~result['fixed_sequence_mask'])[..., None] + result['x_motif']
            batch["x_0"] = x_0
        return result

    def create_motif(self, num_residues: int) -> dict:
        """
        Create a single motif and update the input features.

        Args:
            np_features (dict): Input features.

        Returns:
            dict: Updated features with motif information.
        """
        result = {}
        if random.random() > self.motif_prob:
            motif_sequence_mask = torch.zeros((num_residues))
            motif_structure_mask = torch.zeros((num_residues, num_residues))
            result['fixed_sequence_mask'] = motif_sequence_mask
            result['fixed_structure_mask'] = motif_structure_mask
            return result

        # Sample number of motif residues
        motif_n_res = np.random.randint(
            np.floor(num_residues * self.motif_min_pct_res),
            np.ceil(num_residues * self.motif_max_pct_res)
        )

        # Sample number of motif segments
        motif_n_seg = np.random.randint(
            self.motif_min_n_seg,
            min(self.motif_max_n_seg, motif_n_res) + 1
        )

        # Sample motif segments
        indices = sorted(np.random.choice(motif_n_res - 1, motif_n_seg - 1, replace=False) + 1)
        indices = [0] + indices + [motif_n_res]
        motif_seg_lens = [indices[i + 1] - indices[i] for i in range(motif_n_seg)]

        # Generate motif mask
        segs = [''.join(['1'] * l) for l in motif_seg_lens]
        segs.extend(['0'] * (num_residues - motif_n_res))
        random.shuffle(segs)
        motif_sequence_mask = torch.tensor([int(elt) for elt in ''.join(segs)], dtype=torch.bool)
        motif_structure_mask = motif_sequence_mask[:, None] * motif_sequence_mask[None, :]
        motif_structure_mask = motif_structure_mask.bool()

        # Update
        result['fixed_sequence_mask'] = motif_sequence_mask
        result['fixed_structure_mask'] = motif_structure_mask

        return result


if __name__ == "__main__":
    contig_str = '0-14/A15-15/16-44/A45-45/46-52/A53-53/54-70'
    pdb_path = 'inference/motif_scaffolding/0/0.pdb'
    motif = motif_extract(contig_str, pdb_path, motif_only=False)
    print(motif.shape)
    print(motif.coord)

    coords1 = np.array([x.coord for x in motif])
    # coords1 = [N, 3] -> [N, 37, 3]
    coords1_37 = np.zeros((coords1.shape[0], 37, 3))
    coords1_37[:, 1, :] = coords1
    coords1_37 = torch.Tensor(coords1_37)

    contig_str = '10-20/B38/15-30/A14/15-30/C99/10-20'
    pdb_path = 'motif_data/design25/1qjg.pdb'
    motif = motif_extract(contig_str, pdb_path, motif_only=True)
    print(motif.shape)
    print(motif.coord)

    coords2 = np.array([x.coord for x in motif])
    coords2_37 = np.zeros((coords2.shape[0], 37, 3))
    coords2_37[:, 1, :] = coords2
    coords2_37 = torch.Tensor(coords2_37)

    from proteinfoundation.metrics.designability import rmsd_metric

    rmsd = rmsd_metric(coords1_37, coords2_37)
    print(rmsd)
