"""PyTorch Lightning training workflow for antigen-antibody prediction model."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Dict, Any

from ..model.model import AntigenAntibodyModel


class AntigenAntibodyLitModule(pl.LightningModule):
    """LightningModule wrapper for AntigenAntibodyModel.

    Args:
        node_input_dim: Input node feature dimension
        edge_input_dim: Input edge feature dimension
        hidden_dim: Hidden dimension for Graph Transformer
        num_heads: Number of attention heads
        num_layers: Number of Graph Transformer layers
        dropout: Dropout probability
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        classification_weight: Weight for classification loss
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        classification_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AntigenAntibodyModel(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.classification_weight = classification_weight

        # Loss functions
        self.classification_loss_fn = nn.BCEWithLogitsLoss()

        # Metrics tracking
        self.train_metrics = {"loss": [], "cls_loss": []}
        self.val_metrics = {"loss": [], "cls_loss": []}

    def forward(
        self,
        antigen_nodes: torch.Tensor,
        antigen_edge_indices: torch.Tensor,
        antigen_edge_features: torch.Tensor,
        antibody_nodes: torch.Tensor,
        antibody_edge_indices: torch.Tensor,
        antibody_edge_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.model(
            antigen_nodes=antigen_nodes,
            antigen_edge_indices=antigen_edge_indices,
            antigen_edge_features=antigen_edge_features,
            antibody_nodes=antibody_nodes,
            antibody_edge_indices=antibody_edge_indices,
            antibody_edge_features=antibody_edge_features,
        )

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute classification loss.

        Args:
            outputs: Model output dictionary
            labels: Label dictionary with 'binding'

        Returns:
            Dictionary with total_loss, classification_loss
        """
        loss_dict = {}

        # Classification loss
        if 'binding' in labels:
            cls_loss = self.classification_loss_fn(
                outputs['classification_logits'].squeeze(),
                labels['binding']
            )
            loss_dict['classification_loss'] = cls_loss
        else:
            # No label available - this should not happen (should be filtered in dataloader)
            raise ValueError("Missing 'binding' label for loss computation")

        # Total loss
        total_loss = self.classification_weight * cls_loss
        loss_dict['total_loss'] = total_loss

        return loss_dict

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Single training step."""
        # Handle batched data
        if 'samples' in batch:
            # Multi-sample batch - process first sample for now
            # TODO: Implement proper graph batching
            sample = batch['samples'][0]
        else:
            sample = batch

        # Forward pass
        outputs = self(
            antigen_nodes=sample['antigen']['node_features'],
            antigen_edge_indices=sample['antigen']['edge_indices'],
            antigen_edge_features=sample['antigen']['edge_features'],
            antibody_nodes=sample['antibody']['node_features'],
            antibody_edge_indices=sample['antibody']['edge_indices'],
            antibody_edge_features=sample['antibody']['edge_features'],
        )

        # Compute loss
        labels = sample.get('label', {})
        loss_dict = self._compute_loss(outputs, labels)

        # Log metrics
        self.log('train_loss', loss_dict['total_loss'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_cls_loss', loss_dict['classification_loss'], on_step=True, on_epoch=True)

        return loss_dict['total_loss']

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Single validation step."""
        # Handle batched data
        if 'samples' in batch:
            sample = batch['samples'][0]
        else:
            sample = batch

        # Forward pass
        outputs = self(
            antigen_nodes=sample['antigen']['node_features'],
            antigen_edge_indices=sample['antigen']['edge_indices'],
            antigen_edge_features=sample['antigen']['edge_features'],
            antibody_nodes=sample['antibody']['node_features'],
            antibody_edge_indices=sample['antibody']['edge_indices'],
            antibody_edge_features=sample['antibody']['edge_features'],
        )

        # Compute loss
        labels = sample.get('label', {})
        loss_dict = self._compute_loss(outputs, labels)

        # Log metrics
        self.log('val_loss', loss_dict['total_loss'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_cls_loss', loss_dict['classification_loss'], on_step=True, on_epoch=True)

        return loss_dict['total_loss']

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Single test step."""
        # Handle batched data
        if 'samples' in batch:
            sample = batch['samples'][0]
        else:
            sample = batch

        # Forward pass
        outputs = self(
            antigen_nodes=sample['antigen']['node_features'],
            antigen_edge_indices=sample['antigen']['edge_indices'],
            antigen_edge_features=sample['antigen']['edge_features'],
            antibody_nodes=sample['antibody']['node_features'],
            antibody_edge_indices=sample['antibody']['edge_indices'],
            antibody_edge_features=sample['antibody']['edge_features'],
        )

        # Compute loss if labels available
        labels = sample.get('label', {})
        if 'binding' in labels:
            loss_dict = self._compute_loss(outputs, labels)
        else:
            # No label available - raise error
            raise ValueError("Missing 'binding' label for loss computation")

        return {
            'outputs': outputs,
            'labels': labels,
            'filename': sample.get('filename', ''),
            **loss_dict,
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss',
        }


def train(
    model: AntigenAntibodyModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    classification_weight: float = 1.0,
    accelerator: str = 'auto',
    devices: int = 1,
    checkpoint_dir: str = 'checkpoints',
) -> AntigenAntibodyLitModule:
    """Train the model.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        classification_weight: Weight for classification loss
        accelerator: Accelerator type ('cpu', 'gpu', 'auto')
        devices: Number of devices
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Trained LightningModule
    """
    lit_model = AntigenAntibodyLitModule(
        node_input_dim=model.node_input_dim,
        edge_input_dim=model.edge_input_dim,
        hidden_dim=model.hidden_dim,
        num_heads=model.num_heads,
        num_layers=model.num_layers,
        dropout=0.1,
        lr=learning_rate,
        weight_decay=weight_decay,
        classification_weight=classification_weight,
    )

    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='train_loss',
            mode='min',
            save_top_k=3,
        ),
        pl.callbacks.EarlyStopping(
            monitor='train_loss',
            patience=20,
            mode='min',
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate every epoch
    )

    # Train
    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Load best checkpoint
    best_checkpoint = callbacks[0].best_model_path
    if best_checkpoint:
        lit_model = AntigenAntibodyLitModule.load_from_checkpoint(best_checkpoint)

    return lit_model
