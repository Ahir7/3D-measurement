"""
Fine-tuning support for depth estimation models.

Provides training infrastructure for adapting pre-trained depth models
to specific domains (e.g., indoor box measurement).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 10
    warmup_epochs: int = 1

    # Fine-tuning strategy
    head_only: bool = True  # Only train prediction head
    freeze_encoder_epochs: int = 5  # Epochs before unfreezing encoder
    encoder_lr_multiplier: float = 0.1  # Lower LR for encoder

    # Loss function
    loss_type: str = "scale_invariant"  # scale_invariant | l1 | berhu

    # Validation
    val_frequency: int = 1  # Validate every N epochs
    early_stopping_patience: int = 5

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_best_only: bool = True

    # Hardware
    device: str = "cuda:0"
    mixed_precision: bool = True
    num_workers: int = 4


@dataclass
class TrainingMetrics:
    """Training metrics container."""

    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


class DepthDataset(Dataset):
    """
    Dataset for depth fine-tuning.

    Expected data format:
    - images: RGB images [H, W, 3]
    - depths: Ground truth depth maps [H, W]
    - masks: Optional validity masks [H, W]
    """

    def __init__(
        self,
        image_paths: List[Path],
        depth_paths: List[Path],
        mask_paths: Optional[List[Path]] = None,
        transform: Optional[Callable] = None,
        depth_scale: float = 1.0
    ):
        """
        Initialize dataset.

        Args:
            image_paths: Paths to RGB images
            depth_paths: Paths to depth maps
            mask_paths: Optional paths to validity masks
            transform: Optional transform function
            depth_scale: Scale factor for depth values
        """
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.depth_scale = depth_scale

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        import cv2
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load depth
        depth = np.load(str(self.depth_paths[idx]))
        depth = depth * self.depth_scale

        # Load mask if available
        mask = None
        if self.mask_paths is not None:
            mask = np.load(str(self.mask_paths[idx]))

        # Apply transforms
        if self.transform is not None:
            image, depth, mask = self.transform(image, depth, mask)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        depth_tensor = torch.from_numpy(depth).float()

        result = {
            'image': image_tensor,
            'depth': depth_tensor
        }

        if mask is not None:
            result['mask'] = torch.from_numpy(mask).float()

        return result


class FineTuningTrainer:
    """
    Trainer for fine-tuning depth estimation models.

    Supports:
    - Head-only fine-tuning (memory efficient)
    - Full model fine-tuning
    - Gradual unfreezing
    - Mixed precision training
    """

    def __init__(self, config: FineTuningConfig):
        """
        Initialize trainer.

        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Setup checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Fine-tuning trainer initialized on {config.device}")

    def train_head_only(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None
    ) -> nn.Module:
        """
        Fine-tune only the prediction head.

        Freezes encoder and trains only the decoder/head layers.

        Args:
            model: Pre-trained depth model
            train_data: Training data loader
            val_data: Optional validation data loader

        Returns:
            Fine-tuned model
        """
        logger.info("Starting head-only fine-tuning")

        # Freeze encoder
        self._freeze_encoder(model)

        # Get trainable parameters (head only)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Training {len(trainable_params)} parameter groups (head only)")

        # Setup optimizer
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training loop
        model = self._training_loop(model, train_data, val_data, optimizer)

        return model

    def train_full_model(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None
    ) -> nn.Module:
        """
        Fine-tune the entire model with gradual unfreezing.

        Starts with frozen encoder, then gradually unfreezes.

        Args:
            model: Pre-trained depth model
            train_data: Training data loader
            val_data: Optional validation data loader

        Returns:
            Fine-tuned model
        """
        logger.info("Starting full model fine-tuning with gradual unfreezing")

        # Initially freeze encoder
        self._freeze_encoder(model)

        # Setup optimizer with different LR for encoder
        param_groups = self._get_param_groups(model)
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )

        # Training loop with unfreezing
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Unfreeze encoder after specified epochs
            if epoch == self.config.freeze_encoder_epochs:
                logger.info(f"Epoch {epoch}: Unfreezing encoder")
                self._unfreeze_encoder(model)
                # Update optimizer with new parameters
                param_groups = self._get_param_groups(model)
                optimizer = optim.AdamW(
                    param_groups,
                    weight_decay=self.config.weight_decay
                )

            # Train one epoch
            train_loss = self._train_epoch(model, train_data, optimizer)

            # Validation
            val_loss = None
            if val_data is not None and epoch % self.config.val_frequency == 0:
                val_loss = self._validate(model, val_data)

            # Log metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=optimizer.param_groups[0]['lr']
            )
            self._log_metrics(metrics)

            # Early stopping
            if val_loss is not None:
                if self._check_early_stopping(val_loss, model):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        return model

    def _training_loop(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: Optional[DataLoader],
        optimizer: optim.Optimizer
    ) -> nn.Module:
        """Run the main training loop."""
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs
        )

        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self._train_epoch(model, train_data, optimizer, scaler)

            # Validate
            val_loss = None
            if val_data is not None and epoch % self.config.val_frequency == 0:
                val_loss = self._validate(model, val_data)

            # Log
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=scheduler.get_last_lr()[0]
            )
            self._log_metrics(metrics)

            # LR schedule
            scheduler.step()

            # Early stopping
            if val_loss is not None:
                if self._check_early_stopping(val_loss, model):
                    break

        return model

    def _train_epoch(
        self,
        model: nn.Module,
        train_data: DataLoader,
        optimizer: optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_data):
            images = batch['image'].to(self.device)
            depths = batch['depth'].to(self.device)
            masks = batch.get('mask')
            if masks is not None:
                masks = masks.to(self.device)

            optimizer.zero_grad()

            # Mixed precision forward
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                pred_depths = model(images)
                if hasattr(pred_depths, 'predicted_depth'):
                    pred_depths = pred_depths.predicted_depth
                if pred_depths.dim() == 4:
                    pred_depths = pred_depths.squeeze(1)

                loss = self._compute_loss(pred_depths, depths, masks)

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_data)

    def _validate(
        self,
        model: nn.Module,
        val_data: DataLoader
    ) -> float:
        """Validate model."""
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_data:
                images = batch['image'].to(self.device)
                depths = batch['depth'].to(self.device)
                masks = batch.get('mask')
                if masks is not None:
                    masks = masks.to(self.device)

                pred_depths = model(images)
                if hasattr(pred_depths, 'predicted_depth'):
                    pred_depths = pred_depths.predicted_depth
                if pred_depths.dim() == 4:
                    pred_depths = pred_depths.squeeze(1)

                loss = self._compute_loss(pred_depths, depths, masks)
                total_loss += loss.item()

        return total_loss / len(val_data)

    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute depth loss."""
        if self.config.loss_type == "scale_invariant":
            return self._scale_invariant_loss(pred, target, mask)
        elif self.config.loss_type == "l1":
            return self._l1_loss(pred, target, mask)
        elif self.config.loss_type == "berhu":
            return self._berhu_loss(pred, target, mask)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

    def _scale_invariant_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Scale-invariant log loss (Eigen et al.)."""
        # Add small epsilon to avoid log(0)
        pred_log = torch.log(pred + 1e-6)
        target_log = torch.log(target + 1e-6)
        diff = pred_log - target_log

        if mask is not None:
            diff = diff[mask > 0]

        # Scale-invariant loss
        n = diff.numel()
        loss = (diff ** 2).mean() - 0.5 * (diff.sum() ** 2) / (n ** 2)

        return loss

    def _l1_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """L1 loss."""
        diff = torch.abs(pred - target)
        if mask is not None:
            diff = diff[mask > 0]
        return diff.mean()

    def _berhu_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        threshold: float = 0.2
    ) -> torch.Tensor:
        """Reverse Huber (BerHu) loss."""
        diff = torch.abs(pred - target)
        if mask is not None:
            diff = diff[mask > 0]

        c = threshold * diff.max().detach()

        l1_mask = diff <= c
        l2_mask = diff > c

        loss = torch.zeros_like(diff)
        loss[l1_mask] = diff[l1_mask]
        loss[l2_mask] = (diff[l2_mask] ** 2 + c ** 2) / (2 * c)

        return loss.mean()

    def _freeze_encoder(self, model: nn.Module) -> None:
        """Freeze encoder parameters."""
        for name, param in model.named_parameters():
            if 'encoder' in name.lower() or 'backbone' in name.lower():
                param.requires_grad = False

    def _unfreeze_encoder(self, model: nn.Module) -> None:
        """Unfreeze encoder parameters."""
        for name, param in model.named_parameters():
            param.requires_grad = True

    def _get_param_groups(self, model: nn.Module) -> List[Dict]:
        """Get parameter groups with different learning rates."""
        encoder_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if 'encoder' in name.lower() or 'backbone' in name.lower():
                encoder_params.append(param)
            else:
                head_params.append(param)

        return [
            {'params': head_params, 'lr': self.config.learning_rate},
            {'params': encoder_params,
             'lr': self.config.learning_rate * self.config.encoder_lr_multiplier}
        ]

    def _check_early_stopping(
        self,
        val_loss: float,
        model: nn.Module
    ) -> bool:
        """Check early stopping criterion."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0

            # Save best model
            if self.config.save_best_only:
                self._save_checkpoint(model, "best_model.pt")

            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience

    def _save_checkpoint(self, model: nn.Module, filename: str) -> None:
        """Save model checkpoint."""
        path = self.config.checkpoint_dir / filename
        torch.save(model.state_dict(), path)
        logger.info(f"Saved checkpoint: {path}")

    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics."""
        msg = f"Epoch {metrics.epoch}: train_loss={metrics.train_loss:.4f}"
        if metrics.val_loss is not None:
            msg += f", val_loss={metrics.val_loss:.4f}"
        msg += f", lr={metrics.learning_rate:.6f}"
        logger.info(msg)
