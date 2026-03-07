import os
from typing import Dict, List, Optional

import torch

from .protein_mpnn_utils import (
    ProteinMPNN,
    StructureDataset,
    tied_featurize,
)


def load_protein_mpnn(checkpoint_path: str) -> ProteinMPNN:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden_dim = 128
    num_layers = 3
    ca_only = False
    use_soluble_model = False
    model_name = "v_48_020"
    backbone_noise = 0.0
    max_length = 200000

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ProteinMPNN(
        ca_only=ca_only,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=backbone_noise,
        k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_protein_mpnn_forward(
    json_lines: list[str],
    model: ProteinMPNN,
) -> List[Dict[str, torch.Tensor]]:
    """Load ProteinMPNN, read jsonl input, and run a forward pass.

    Defaults follow protein_mpnn_run.py. Returns list of dicts with
    keys: name, log_probs, mask, S.
    """
    max_length = 200000
    ca_only = False
    use_soluble_model = False
    device = next(model.parameters()).device
    dataset_valid = StructureDataset(json_lines, truncate=None, max_length=max_length, verbose=False)

    outputs: List[Dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for protein in dataset_valid:
            batch_clones = [protein]
            (
                X,
                S,
                mask,
                _lengths,
                chain_M,
                chain_encoding_all,
                _chain_list_list,
                _visible_list_list,
                _masked_list_list,
                _masked_chain_length_list_list,
                chain_M_pos,
                _omit_AA_mask,
                residue_idx,
                _dihedral_mask,
                _tied_pos_list_of_lists_list,
                _pssm_coef,
                _pssm_bias,
                _pssm_log_odds_all,
                _bias_by_res_all,
                _tied_beta,
            ) = tied_featurize(
                batch_clones,
                device,
                None,
                None,
                None,
                None,
                None,
                None,
                ca_only=ca_only,
            )

            randn_1 = torch.randn(chain_M.shape, device=X.device)
            ret = model(
                X,
                S,
                mask,
                chain_M * chain_M_pos,
                residue_idx,
                chain_encoding_all,
                randn_1,
            )
            outputs.append(
                ret
            )

    return outputs