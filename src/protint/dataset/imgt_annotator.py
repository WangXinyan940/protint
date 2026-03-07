"""IMGT annotation tool for antibody sequences using ANARCI.

This module provides tools to annotate antibody sequences with IMGT region labels
and chain type information using the ANARCI package.
"""

from typing import Tuple, Optional, List
import torch


# IMGT CDR definitions (numbering scheme)
# CDR1: 27-38, CDR2: 56-65, CDR3: 105-117
CDR1_START, CDR1_END = 27, 38
CDR2_START, CDR2_END = 56, 65
CDR3_START, CDR3_END = 105, 117

# Region labels for one-hot encoding
REGION_LABELS = ['FR1', 'FR2', 'FR3', 'FR4', 'CDR1', 'CDR2', 'CDR3']
REGION_LABEL_TO_IDX = {label: idx for idx, label in enumerate(REGION_LABELS)}

# Chain type labels for one-hot encoding
CHAIN_TYPE_LABELS = ['light', 'heavy', 'non_antibody']
CHAIN_TYPE_LABEL_TO_IDX = {label: idx for idx, label in enumerate(CHAIN_TYPE_LABELS)}


def get_region_from_imgt_number(imgt_number: int) -> str:
    """Convert IMGT numbering to region label.

    Args:
        imgt_number: IMGT position number (integer)

    Returns:
        Region label: FR1, FR2, FR3, FR4, CDR1, CDR2, or CDR3
    """
    if imgt_number < CDR1_START:
        return 'FR1'
    elif CDR1_START <= imgt_number <= CDR1_END:
        return 'CDR1'
    elif CDR1_END < imgt_number < CDR2_START:
        return 'FR2'
    elif CDR2_START <= imgt_number <= CDR2_END:
        return 'CDR2'
    elif CDR2_END < imgt_number < CDR3_START:
        return 'FR3'
    elif CDR3_START <= imgt_number <= CDR3_END:
        return 'CDR3'
    else:  # imgt_number > CDR3_END
        return 'FR4'


def annotate_sequence(
    sequence: str
) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[str]]:
    """Annotate an antibody sequence using ANARCI.

    Uses ANARCI to predict IMGT numbering and chain type for an antibody sequence.

    Args:
        sequence: Amino acid sequence to annotate

    Returns:
        Tuple of:
        - imgt_numbers: List of IMGT position numbers for each residue (None if failed)
        - region_labels: List of region indices (0-6) for each residue (None if failed)
        - chain_type: Chain type string 'light', 'heavy', or None if failed
    """
    try:
        from anarci import anarci
    except ImportError:
        raise ImportError("ANARCI is required for IMGT annotation. Install with: pip install anarci")

    # Run ANARCI
    result = anarci([(sequence, sequence)], scheme="imgt", output=False)
    numbered, alignment_details, hit_tables = result

    # Check if annotation was successful
    if not numbered[0] or numbered[0][0] is None:
        return None, None, None

    numbered_seq, start, end = numbered[0][0]

    # Extract IMGT numbers and region labels for aligned residues
    imgt_numbers = []
    region_labels = []

    for item in numbered_seq:
        if item[1] != "-":  # Skip gaps
            imgt_number = item[0][0]  # IMGT position number
            imgt_numbers.append(imgt_number)
            region = get_region_from_imgt_number(imgt_number)
            region_labels.append(REGION_LABEL_TO_IDX[region])

    # Handle residues before start and after end
    # Fill with boundary values
    if start > 1:
        # Residues before the alignment start
        imgt_numbers = [imgt_numbers[0] - 1] * (start - 1) + imgt_numbers
        region_labels = [region_labels[0]] * (start - 1) + region_labels

    if end < len(sequence):
        # Residues after the alignment end
        imgt_numbers = imgt_numbers + [imgt_numbers[-1] + 1] * (len(sequence) - end)
        region_labels = region_labels + [region_labels[-1]] * (len(sequence) - end)

    # Get chain type
    chain_type = None
    if alignment_details and alignment_details[0]:
        chain_type_raw = alignment_details[0][0].get('chain_type')
        if chain_type_raw == 'L':
            chain_type = 'light'
        elif chain_type_raw == 'H':
            chain_type = 'heavy'

    return imgt_numbers, region_labels, chain_type


def create_region_one_hot(region_labels: Optional[List[int]], sequence_length: int) -> torch.Tensor:
    """Create one-hot encoding for IMGT regions.

    Args:
        region_labels: List of region indices (0-6), or None for non-antibody
        sequence_length: Length of the sequence

    Returns:
        One-hot tensor of shape (sequence_length, 7)
        Returns all zeros if region_labels is None
    """
    one_hot = torch.zeros(sequence_length, 7)

    if region_labels is not None:
        for i, label_idx in enumerate(region_labels):
            if i < sequence_length:
                one_hot[i, label_idx] = 1.0

    return one_hot


def create_chain_type_one_hot(chain_type: Optional[str], sequence_length: int) -> torch.Tensor:
    """Create one-hot encoding for chain type.

    Args:
        chain_type: Chain type string ('light', 'heavy', or None for non-antibody)
        sequence_length: Length of the sequence

    Returns:
        One-hot tensor of shape (sequence_length, 3)
        Returns [0, 0, 1] (non_antibody) repeated if chain_type is None or unrecognized
    """
    one_hot = torch.zeros(sequence_length, 3)

    if chain_type == 'light':
        one_hot[:, 0] = 1.0
    elif chain_type == 'heavy':
        one_hot[:, 1] = 1.0
    else:
        # non_antibody or unrecognized
        one_hot[:, 2] = 1.0

    return one_hot


def create_imgt_features(
    sequence: str,
    is_antibody: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """...
    Returns:
        - region_one_hot: (L, 7) tensor
        - imgt_numbers: (L,) tensor
        - chain_type_one_hot: (L, 3) tensor  # 修改：从 (3,) 改为 (L, 3)
    """
    if not is_antibody:
        return (
            torch.zeros(len(sequence), 7),
            torch.zeros(len(sequence), dtype=torch.long),
            torch.zeros(len(sequence), 3)  # 修改：所有残基都是 non_antibody
        )

    # Run annotation
    imgt_numbers, region_labels, chain_type = annotate_sequence(sequence)

    if imgt_numbers is None:
        # ANARCI failed to annotate - return all zeros for region, chain type = non_antibody
        return (
            torch.zeros(len(sequence), 7),
            torch.zeros(len(sequence), dtype=torch.long),
            torch.zeros(len(sequence), 3)  # non_antibody
        )

    # Convert to tensors
    region_one_hot = create_region_one_hot(region_labels, len(sequence))
    imgt_numbers_tensor = torch.tensor(imgt_numbers, dtype=torch.long)
    chain_type_one_hot = create_chain_type_one_hot(chain_type, len(sequence))

    return region_one_hot, imgt_numbers_tensor, chain_type_one_hot
