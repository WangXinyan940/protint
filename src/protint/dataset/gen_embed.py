from ..model.submodules import run_protein_mpnn_forward, load_protein_mpnn, run_esmc_embed, load_esm_c_model
from .parse import parse_pdb
from .imgt_annotator import create_imgt_features
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers
import json
import torch


def parse_pdb_file(
    pdb_file: str,
    esm_c,
    protein_mpnn,
    is_antibody: bool = True
) -> dict:
    """Parse a PDB file and generate embeddings with IMGT features.

    Args:
        pdb_file: Path to the PDB file
        esm_c: Loaded ESM-C model
        protein_mpnn: Loaded ProteinMPNN model
        is_antibody: Whether this PDB file is an antibody (True) or antigen (False).

    Returns:
        Dictionary containing:
        - node_features: (L, 960 + 128 + 7 + 3) = (L, 1098) tensor
        - edge_features: (L, N_neighbor, 128) tensor
        - edge_indices: (L, N_neighbor) tensor
    """
    parsed_data = parse_pdb(pdb_file)

    sequence_features = []
    imgt_region_features = []
    imgt_number_features = []
    imgt_chain_type_features = []
    chain_order = []

    for chainid in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        chain_key = f"seq_chain_{chainid}"
        if chain_key in parsed_data:
            sequence = parsed_data[chain_key]
            seq_embed = run_esmc_embed(sequence, esm_c)
            sequence_features.append(seq_embed)
            chain_order.append(chainid)

            # Generate IMGT features
            region_one_hot, imgt_numbers, chain_type_one_hot = create_imgt_features(
                sequence, is_antibody=is_antibody
            )
            imgt_region_features.append(region_one_hot)
            imgt_number_features.append(imgt_numbers)
            imgt_chain_type_features.append(chain_type_one_hot)

    sequence_features = torch.cat(sequence_features, dim=1)[0]  # (L_total, 960)
    imgt_region_cat = torch.cat(imgt_region_features, dim=0)  # (L_total, 7)
    imgt_numbers_cat = torch.cat(imgt_number_features, dim=0)  # (L_total,)
    imgt_chain_type_cat = torch.cat(imgt_chain_type_features, dim=0)  # (L_total, 3)

    graph_embed = run_protein_mpnn_forward(
        [json.dumps(parsed_data)],
        protein_mpnn,
    )
    node_features = graph_embed[0]['node_embeddings'][0]  # (L_total, D)
    edge_features = graph_embed[0]['edge_embeddings'][0]  # (L_total, N_neighbor, D)
    edge_indices = graph_embed[0]['edge_indices'][0]  # (L_total, N_neighbor)

    # Concatenate all node features: ESM-C (960) + ProteinMPNN (128) + IMGT region (7) + chain type (3)
    sum_node_feat = torch.cat([
        sequence_features.cpu(),      # (L_total, 960)
        node_features.cpu(),          # (L_total, 128)
        imgt_region_cat.cpu(),        # (L_total, 7)
        imgt_chain_type_cat.cpu(),    # (L_total, 3)
    ], dim=1)  # (L_total, 1098)

    return {
        'node_features': sum_node_feat.cpu(),
        'edge_features': edge_features.cpu(),
        'edge_indices': edge_indices.cpu(),
        'imgt_numbers': imgt_numbers_cat.cpu(),  # (L_total,) IMGT sequence labels
        'chain_order': chain_order,  # List of chain IDs in order
    }