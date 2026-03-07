"""Antigen-Antibody interaction prediction model."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Dict

from .layers import GraphTransformerLayer, TargetAttentionLayer


class AntigenAntibodyModel(nn.Module):
    """Main model for antigen-antibody interaction prediction.

    Architecture:
    1. Graph Transformer encoder (3 layers) for antigen and antibody
    2. Antigen branch: sum pooling -> antigen vector (128 dim)
    3. Antibody branch: target attention (with antigen vector as K,V) -> sum pooling -> antibody vector (128 dim)
    4. Prediction head: classification head

    Args:
        node_input_dim: Input node feature dimension (ESM-C 960 + ProteinMPNN 128 + IMGT region 7 + chain type 3 = 1098)
        edge_input_dim: Input edge feature dimension
        hidden_dim: Hidden dimension for Graph Transformer (default: 128)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of Graph Transformer layers (default: 3)
        dropout: Dropout probability
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Input projection
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim)

        # Graph Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            GraphTransformerLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_heads=num_heads,
                concat=True,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Layer norm after each encoder layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Antigen pooling projection
        self.antigen_pool_proj = nn.Linear(hidden_dim, hidden_dim)

        # Target attention for antibody
        self.target_attention = TargetAttentionLayer(
            node_dim=hidden_dim,
            antigen_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Antibody pooling projection
        self.antibody_pool_proj = nn.Linear(hidden_dim, hidden_dim)

        # Prediction head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_antigen(
        self,
        node_features: Tensor,
        edge_indices: Tensor,
        edge_features: Tensor,
        antigen_lengths: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """Encode antigen graph and pool to vector.

        Args:
            node_features: Node features (L, D_in)
            edge_indices: Edge indices in Graph Transformer format (2, E)
            edge_features: Edge features (E, D_edge)
            antigen_lengths: Length of each antigen in batch (optional for single sample)

        Returns:
            antigen_node_embeddings: Encoded node embeddings (L, hidden_dim)
            antigen_vec: Pooled antigen vector (hidden_dim,)
        """
        # Project input features
        x = self.node_proj(node_features)  # (L, hidden_dim)
        edge_attr = self.edge_proj(edge_features)  # (E, hidden_dim)

        # Graph Transformer encoder
        for i, (encoder_layer, layer_norm) in enumerate(zip(self.encoder_layers, self.layer_norms)):
            x_new = encoder_layer(x, edge_indices, edge_attr)
            x = layer_norm(x + x_new)  # Residual connection + layer norm

        # Sum pooling
        antigen_vec = torch.sum(x, dim=0)  # (hidden_dim,)
        antigen_vec = self.antigen_pool_proj(antigen_vec)  # (hidden_dim,)

        return x, antigen_vec

    def encode_antibody(
        self,
        node_features: Tensor,
        edge_indices: Tensor,
        edge_features: Tensor,
        antigen_vec: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Encode antibody graph with target attention to antigen.

        Args:
            node_features: Node features (L, D_in)
            edge_indices: Edge indices in Graph Transformer format (2, E)
            edge_features: Edge features (E, D_edge)
            antigen_vec: Antigen pooled vector (hidden_dim,)

        Returns:
            antibody_node_embeddings: Encoded node embeddings (L, hidden_dim)
            antibody_vec: Pooled antibody vector (hidden_dim,)
        """
        # Project input features
        x = self.node_proj(node_features)  # (L, hidden_dim)
        edge_attr = self.edge_proj(edge_features)  # (E, hidden_dim)

        # Graph Transformer encoder
        for i, (encoder_layer, layer_norm) in enumerate(zip(self.encoder_layers, self.layer_norms)):
            x_new = encoder_layer(x, edge_indices, edge_attr)
            x = layer_norm(x + x_new)  # Residual connection + layer norm

        # Target attention: antibody nodes attend to antigen vector
        attended = self.target_attention(x, antigen_vec)  # (L, hidden_dim)

        # Sum pooling
        antibody_vec = torch.sum(attended, dim=0)  # (hidden_dim,)
        antibody_vec = self.antibody_pool_proj(antibody_vec)  # (hidden_dim,)

        return x, antibody_vec

    def forward(
        self,
        antigen_nodes: Tensor,
        antigen_edge_indices: Tensor,
        antigen_edge_features: Tensor,
        antibody_nodes: Tensor,
        antibody_edge_indices: Tensor,
        antibody_edge_features: Tensor,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            antigen_nodes: Antigen node features (L_ag, D_in)
            antigen_edge_indices: Antigen edge indices (2, E_ag)
            antigen_edge_features: Antigen edge features (E_ag, D_edge)
            antibody_nodes: Antibody node features (L_ab, D_in)
            antibody_edge_indices: Antibody edge indices (2, E_ab)
            antibody_edge_features: Antibody edge features (E_ab, D_edge)

        Returns:
            Dictionary containing:
                - antigen_vec: Antigen pooled vector (hidden_dim,)
                - antibody_vec: Antibody pooled vector (hidden_dim,)
                - classification_logits: Classification output (1,)
                - classification_prob: Classification probability (1,)
        """
        # Encode antigen
        antigen_node_emb, antigen_vec = self.encode_antigen(
            antigen_nodes, antigen_edge_indices, antigen_edge_features
        )

        # Encode antibody with target attention to antigen
        antibody_node_emb, antibody_vec = self.encode_antibody(
            antibody_nodes, antibody_edge_indices, antibody_edge_features, antigen_vec
        )

        # Concatenate antigen and antibody vectors
        combined = torch.cat([antigen_vec, antibody_vec], dim=-1)  # (hidden_dim * 2,)

        # Classification head
        classification_logits = self.classification_head(combined)  # (1,)

        # Apply sigmoid for classification probability
        classification_prob = torch.sigmoid(classification_logits)

        return {
            "antigen_vec": antigen_vec,
            "antibody_vec": antibody_vec,
            "classification_logits": classification_logits,
            "classification_prob": classification_prob,
        }
