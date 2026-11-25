from openfold.np.residue_constants import restypes

UNK_TOKEN = 'X'
PAD_TOKEN = '<pad>'
MASK_TOKEN = '<mask>'
AA_VOCAB = [
    *restypes,  # Standard amino acids
    UNK_TOKEN,  # Unknown amino acid
    PAD_TOKEN,  # Padding token
    MASK_TOKEN,  # Mask token
]  # consistent with openfold
AA_TOKEN_2_ID = {token: idx for idx, token in enumerate(AA_VOCAB)}
ID_2_AA_TOKEN = {idx: token for idx, token in enumerate(AA_VOCAB)}
UNK_TOKEN_ID = AA_TOKEN_2_ID[UNK_TOKEN]
PAD_TOKEN_ID = AA_TOKEN_2_ID[PAD_TOKEN]
MASK_TOKEN_ID = AA_TOKEN_2_ID[MASK_TOKEN]
