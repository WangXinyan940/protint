"""Custom GNN layers for antigen-antibody interaction prediction."""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import TransformerConv


class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer based on TransformerConv.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        num_heads: Number of attention heads
        concat: If True, concatenate heads; if False, average them
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        concat: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat

        # TransformerConv uses edge features to compute attention
        # We use a linear layer to project edge features for the attention mechanism
        self.transformer_conv = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels // num_heads if concat else out_channels,
            heads=num_heads,
            concat=concat,
            dropout=dropout,
            edge_dim=in_channels,  # Edge features have same dim as node features
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Node features (L, D)
            edge_index: Edge indices (2, E) - note: TransformerConv expects (2, E) format
            edge_attr: Edge features (E, D_edge)

        Returns:
            Updated node features (L, D_out)
        """
        return self.transformer_conv(x, edge_index, edge_attr=edge_attr)


class TargetAttentionLayer(nn.Module):
    """Target attention layer for antibody-antigen interaction.

    The antibody nodes attend to the antigen representation using
    standard attention mechanism where:
    - Q comes from antibody node features
    - K, V come from antigen pooled representation

    Args:
        node_dim: Node feature dimension
        antigen_dim: Antigen representation dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        node_dim: int,
        antigen_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.antigen_dim = antigen_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        # Query projection from antibody node features
        self.q_proj = nn.Linear(node_dim, node_dim)
        # Key and Value projections from antigen representation
        self.k_proj = nn.Linear(antigen_dim, node_dim)
        self.v_proj = nn.Linear(antigen_dim, node_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(node_dim, node_dim)

    def forward(
        self,
        antibody_nodes: Tensor,
        antigen_vec: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            antibody_nodes: Antibody node features (L_ab, D)
            antigen_vec: Pooled antigen representation (batch, D_antigen) or (D_antigen,)
            mask: Optional attention mask

        Returns:
            Attended antibody features (L_ab, D)
        """
        # Handle single sample case
        if antigen_vec.dim() == 1:
            antigen_vec = antigen_vec.unsqueeze(0)  # (1, D_antigen)

        batch_size = antigen_vec.shape[0]
        L_ab = antibody_nodes.shape[0]

        # For single sample, expand to batch
        if antibody_nodes.dim() == 2:
            # Single sample: (L_ab, D) -> (batch, L_ab, D)
            antibody_nodes = antibody_nodes.unsqueeze(0).expand(batch_size, -1, -1)

        # Project to Q, K, V
        Q = self.q_proj(antibody_nodes)  # (batch, L_ab, D)
        K = self.k_proj(antigen_vec)  # (batch, 1, D) - antigen is single vector
        V = self.v_proj(antigen_vec)  # (batch, 1, D)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, L_ab, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, L_ab, head_dim)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, 1, head_dim)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, 1, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, heads, L_ab, 1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)  # (batch, heads, L_ab, 1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # (batch, heads, L_ab, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, L_ab, self.node_dim)  # (batch, L_ab, D)

        # Output projection
        output = self.out_proj(attended)  # (batch, L_ab, D)

        # Squeeze back to single sample if input was single sample
        if output.shape[0] == 1 and antibody_nodes.dim() == 3:
            output = output.squeeze(0)  # (L_ab, D)

        return output
