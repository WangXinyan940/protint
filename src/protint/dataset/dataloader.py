"""Dataset and DataLoader for antigen-antibody interaction prediction."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
import pickle
import os
import pandas as pd


class AntigenAntibodyDataset(Dataset):
    """Dataset for loading antigen-antibody interaction data from pkl files.

    Each pkl file should contain:
    - antigen: dict with 'node_features', 'edge_features', 'edge_indices'
    - antibody: dict with 'node_features', 'edge_features', 'edge_indices'
    - label (optional): dict with 'binding' (0/1)

    Args:
        data_dir: Directory containing pkl files
        file_list (optional): List of filenames to load. If None, load all .pkl files
    """

    def __init__(
        self,
        data_dir: str,
        file_list: Optional[List[str]] = None,
    ):
        self.data_dir = data_dir

        if file_list is None:
            # Load all pkl files in directory
            self.files = [
                f for f in os.listdir(data_dir)
                if f.endswith('.pkl')
            ]
        else:
            self.files = file_list

        self.files.sort()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single sample.

        Returns:
            Dictionary containing:
                - antigen: dict with node_features, edge_features, edge_indices
                - antibody: dict with node_features, edge_features, edge_indices
                - label (if available): dict with binding
                - filename: name of the pkl file
        """
        filename = self.files[idx]
        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        result = {
            'antigen': {
                'node_features': torch.as_tensor(data['antigen']['node_features'], dtype=torch.float32),
                'edge_features': torch.as_tensor(data['antigen']['edge_features'], dtype=torch.float32),
                'edge_indices': self._convert_edge_indices(data['antigen']['edge_indices']),
            },
            'antibody': {
                'node_features': torch.as_tensor(data['antibody']['node_features'], dtype=torch.float32),
                'edge_features': torch.as_tensor(data['antibody']['edge_features'], dtype=torch.float32),
                'edge_indices': self._convert_edge_indices(data['antibody']['edge_indices']),
            },
            'filename': filename,
        }

        # Load labels if available
        if 'label' in data:
            result['label'] = {}
            if 'binding' in data['label']:
                result['label']['binding'] = torch.as_tensor(data['label']['binding'], dtype=torch.float32)

        return result

    def _convert_edge_indices(self, edge_indices: torch.Tensor) -> torch.Tensor:
        """Convert edge indices from (L, N_neighbor) to (2, E) format for PyTorch Geometric.

        The gen_embed.py outputs edge_indices as (L, N_neighbor) where each row
        contains the neighbor indices for that node.

        We need to convert this to (2, E) format where:
        - First row: source nodes
        - Second row: target nodes
        """
        if isinstance(edge_indices, torch.Tensor):
            edge_indices = edge_indices.clone()
        else:
            edge_indices = torch.tensor(edge_indices)

        L, N_neighbor = edge_indices.shape

        # Create source indices (each node appears once as source)
        src = torch.arange(L).unsqueeze(-1).expand(-1, N_neighbor).flatten()

        # Target indices are the neighbor indices
        dst = edge_indices.flatten()

        # Stack to (2, E)
        edge_index = torch.stack([src, dst], dim=0)

        return edge_index


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader.

    For batched training, we need to:
    1. Create batched graphs using PyG's batch mechanism
    2. Handle the fact that antigen and antibody are separate graphs

    For simplicity, this implementation returns a list of samples
    and the model can process them one at a time or implement
    proper graph batching.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched data dictionary
    """
    if len(batch) == 1:
        # Single sample - no batching needed
        return batch[0]

    # For multi-sample batch, we'll create a batched representation
    # This is a simplified version - for production you may want proper graph batching

    result = {
        'samples': batch,
        'batch_size': len(batch),
    }

    return result


def create_dataloader(
    data_dir: str,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    file_list: Optional[List[str]] = None,
) -> DataLoader:
    """Create a DataLoader for antigen-antibody dataset.

    Args:
        data_dir: Directory containing pkl files
        batch_size: Batch size (default: 1, as graphs may have different sizes)
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        file_list: Optional list of files to load

    Returns:
        PyTorch DataLoader
    """
    dataset = AntigenAntibodyDataset(data_dir, file_list)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader


class PairDataset(Dataset):
    """Dataset for loading antigen-antibody pairs from pkl files and pairs.csv.

    Each sample is a pair of antigen and antibody with optional labels.
    Rows with classification_label=-1 are filtered out (not included in the dataset).

    Args:
        data_dir: Directory containing pkl files and pairs.csv
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.pairs_csv = os.path.join(data_dir, "pairs.csv")

        if not os.path.exists(self.pairs_csv):
            raise FileNotFoundError(f"pairs.csv not found in {data_dir}")

        self.pairs_df = pd.read_csv(self.pairs_csv)
        # Filter out rows with classification_label=-1 (no label available)
        self.pairs_df = self.pairs_df[self.pairs_df['classification_label'] != -1].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.pairs_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single antigen-antibody pair.

        Returns:
            Dictionary containing:
                - antigen: dict with node_features, edge_features, edge_indices
                - antibody: dict with node_features, edge_features, edge_indices
                - label (if available): dict with 'binding'
                    (only included if label != -1)
                - pair_id: string identifier for the pair
        """
        row = self.pairs_df.iloc[idx]

        antibody_pkl = row['antibody_pkl']
        antigen_pkl = row['antigen_pkl']
        cls_label = row['classification_label']

        # Load antibody embedding
        with open(os.path.join(self.data_dir, antibody_pkl), 'rb') as f:
            antibody_data = pickle.load(f)

        # Load antigen embedding
        with open(os.path.join(self.data_dir, antigen_pkl), 'rb') as f:
            antigen_data = pickle.load(f)

        result = {
            'antigen': {
                'node_features': torch.as_tensor(antigen_data['node_features'], dtype=torch.float32),
                'edge_features': torch.as_tensor(antigen_data['edge_features'], dtype=torch.float32),
                'edge_indices': self._convert_edge_indices(antigen_data['edge_indices']),
            },
            'antibody': {
                'node_features': torch.as_tensor(antibody_data['node_features'], dtype=torch.float32),
                'edge_features': torch.as_tensor(antibody_data['edge_features'], dtype=torch.float32),
                'edge_indices': self._convert_edge_indices(antibody_data['edge_indices']),
            },
            'pair_id': f"{antibody_pkl}_{antigen_pkl}",
        }

        # Handle labels: -1 means the task is not applicable
        label_dict = {}
        if cls_label != -1:
            label_dict['binding'] = torch.as_tensor(cls_label, dtype=torch.float32)

        if label_dict:
            result['label'] = label_dict

        return result

    def _convert_edge_indices(self, edge_indices: torch.Tensor) -> torch.Tensor:
        """Convert edge indices from (L, N_neighbor) to (2, E) format."""
        if isinstance(edge_indices, torch.Tensor):
            edge_indices = edge_indices.clone()
        else:
            edge_indices = torch.tensor(edge_indices)

        L, N_neighbor = edge_indices.shape
        src = torch.arange(L).unsqueeze(-1).expand(-1, N_neighbor).flatten()
        dst = edge_indices.flatten()
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index


def create_pair_dataloader(
    data_dir: str,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for antigen-antibody pairs dataset.

    Args:
        data_dir: Directory containing pkl files and pairs.csv
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers

    Returns:
        PyTorch DataLoader
    """
    dataset = PairDataset(data_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader
