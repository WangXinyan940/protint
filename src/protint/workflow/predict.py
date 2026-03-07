"""Prediction workflow for trained antigen-antibody model."""

import torch
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..model.model import AntigenAntibodyModel


def load_model(
    checkpoint_path: str,
    node_input_dim: Optional[int] = None,
    edge_input_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
    num_layers: Optional[int] = None,
    device: Optional[str] = None,
) -> Any:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to PyTorch Lightning checkpoint file
        node_input_dim: Input node feature dimension (auto-detected from checkpoint if not provided)
        edge_input_dim: Input edge feature dimension (auto-detected from checkpoint if not provided)
        hidden_dim: Hidden dimension (auto-detected from checkpoint if not provided)
        num_heads: Number of attention heads (auto-detected from checkpoint if not provided)
        num_layers: Number of Graph Transformer layers (auto-detected from checkpoint if not provided)
        device: Device to load model onto (auto-detected if not provided)

    Returns:
        Loaded LightningModule
    """
    from .train import AntigenAntibodyLitModule

    # Auto-detect device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    # Auto-detect hyperparameters from checkpoint if not provided
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint.get('hyper_parameters', {})

    # Use provided parameters or fall back to saved hyperparameters
    final_params = {
        'node_input_dim': node_input_dim if node_input_dim is not None else hparams.get('node_input_dim', 1088),
        'edge_input_dim': edge_input_dim if edge_input_dim is not None else hparams.get('edge_input_dim', 128),
        'hidden_dim': hidden_dim if hidden_dim is not None else hparams.get('hidden_dim', 128),
        'num_heads': num_heads if num_heads is not None else hparams.get('num_heads', 8),
        'num_layers': num_layers if num_layers is not None else hparams.get('num_layers', 3),
    }

    lit_model = AntigenAntibodyLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        node_input_dim=final_params['node_input_dim'],
        edge_input_dim=final_params['edge_input_dim'],
        hidden_dim=final_params['hidden_dim'],
        num_heads=final_params['num_heads'],
        num_layers=final_params['num_layers'],
    )
    lit_model.eval()
    return lit_model


def predict_single(
    model: Any,
    antigen_data: Dict[str, torch.Tensor],
    antibody_data: Dict[str, torch.Tensor],
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Run prediction on a single antigen-antibody pair.

    Args:
        model: Trained LightningModule
        antigen_data: Antigen features with keys:
            - node_features: (L_ag, D)
            - edge_features: (L_ag, N_neighbor, D_edge)
            - edge_indices: (L_ag, N_neighbor)
        antibody_data: Antibody features with same keys
        device: Device to run inference on

    Returns:
        Dictionary with:
            - classification_prob: Probability of binding [0, 1]
            - antigen_vec: Antigen representation vector
            - antibody_vec: Antibody representation vector
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    # Convert edge indices to PyG format (2, E)
    def convert_edge_indices(edge_idx: torch.Tensor, device: str) -> torch.Tensor:
        if edge_idx.dim() == 2 and edge_idx.shape[0] > 2:
            # (L, N_neighbor) -> (2, E)
            L, N = edge_idx.shape
            src = torch.arange(L, device=device).unsqueeze(-1).expand(-1, N).flatten()
            dst = edge_idx.flatten().to(device)
            return torch.stack([src, dst], dim=0)
        return edge_idx.to(device)

    # Prepare inputs
    antigen_nodes = antigen_data['node_features'].to(device)
    antigen_edges = convert_edge_indices(antigen_data['edge_indices'], device)
    antigen_edge_feat = antigen_data['edge_features'].flatten(0, 1).to(device)

    antibody_nodes = antibody_data['node_features'].to(device)
    antibody_edges = convert_edge_indices(antibody_data['edge_indices'], device)
    antibody_edge_feat = antibody_data['edge_features'].flatten(0, 1).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(
            antigen_nodes=antigen_nodes,
            antigen_edge_indices=antigen_edges,
            antigen_edge_features=antigen_edge_feat,
            antibody_nodes=antibody_nodes,
            antibody_edge_indices=antibody_edges,
            antibody_edge_features=antibody_edge_feat,
        )

    result = {
        'classification_prob': outputs['classification_prob'].item(),
        'antigen_vec': outputs['antigen_vec'].cpu().numpy(),
        'antibody_vec': outputs['antibody_vec'].cpu().numpy(),
    }

    return result


def predict_pkl(
    model: Any,
    pkl_path: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run prediction on a pkl file containing antigen-antibody pair.

    Args:
        model: Trained LightningModule
        pkl_path: Path to pkl file with antigen and antibody data
        device: Device for inference

    Returns:
        Prediction results with filename
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    result = predict_single(
        model=model,
        antigen_data=data['antigen'],
        antibody_data=data['antibody'],
        device=device,
    )
    result['filename'] = str(pkl_path)

    return result


def predict_directory(
    model: Any,
    data_dir: str,
    output_path: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run prediction on all pkl files in a directory.

    Note: This function expects paired pkl files. For dataset directory
    with pairs.csv, use predict_pairs_directory instead.

    Args:
        model: Trained LightningModule
        data_dir: Directory containing pkl files (each should contain antigen-antibody pair)
        output_path: Optional path to save results (pkl format)
        device: Device for inference

    Returns:
        List of prediction results
    """
    from pathlib import Path

    pkl_files = list(Path(data_dir).glob('*.pkl'))

    results = []
    for pkl_file in pkl_files:
        result = predict_pkl(model, str(pkl_file), device)
        results.append(result)
        print(f"Processed: {pkl_file.name}")

    # Save results if output path specified
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to: {output_path}")

    return results


def predict_pairs_directory(
    model: Any,
    data_dir: str,
    output_path: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run prediction on all pairs in a dataset directory with pairs.csv.

    Args:
        model: Trained LightningModule
        data_dir: Directory containing pkl files and pairs.csv
        output_path: Optional path to save results (pkl format)
        device: Device for inference

    Returns:
        List of prediction results with pair identifiers
    """
    import pandas as pd

    pairs_csv = Path(data_dir) / "pairs.csv"
    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs.csv not found in {data_dir}")

    pairs_df = pd.read_csv(pairs_csv)

    results = []
    for _, row in pairs_df.iterrows():
        antibody_pkl = row['antibody_pkl']
        antigen_pkl = row['antigen_pkl']
        pair_id = f"{antibody_pkl}_{antigen_pkl}"

        # Load antibody and antigen embeddings
        with open(Path(data_dir) / antibody_pkl, 'rb') as f:
            antibody_data = pickle.load(f)
        with open(Path(data_dir) / antigen_pkl, 'rb') as f:
            antigen_data = pickle.load(f)

        result = predict_single(
            model=model,
            antigen_data=antigen_data,
            antibody_data=antibody_data,
            device=device,
        )
        result['pair_id'] = pair_id
        result['antibody'] = antibody_pkl.replace('.pkl', '')
        result['antigen'] = antigen_pkl.replace('.pkl', '')
        results.append(result)
        print(f"Processed: {pair_id}")

    # Save results if output path specified
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to: {output_path}")

    return results


# CLI entry point
def main():
    """CLI for prediction."""
    import argparse

    parser = argparse.ArgumentParser(description='Predict antigen-antibody interactions')
    parser.add_argument('-c', '--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('-i', '--input', required=True, help='Input pkl file or directory')
    parser.add_argument('-o', '--output', help='Output file for results (pkl format)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device for inference')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint)

    # Run prediction
    import os
    if os.path.isdir(args.input):
        print(f"Predicting on directory: {args.input}")
        results = predict_directory(model, args.input, args.output, args.device)
        print(f"Processed {len(results)} samples")
    else:
        print(f"Predicting file: {args.input}")
        result = predict_pkl(model, args.input, args.device)
        print(f"Classification probability: {result['classification_prob']:.4f}")

        if args.output:
            with open(args.output, 'wb') as f:
                pickle.dump([result], f)
            print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
