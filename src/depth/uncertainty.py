"""
Uncertainty estimation for depth predictions.

Provides multiple uncertainty estimation methods including:
- Monte Carlo Dropout for epistemic uncertainty
- Flip consistency for stability assessment
- Ensemble-based uncertainty (when multiple models available)
- Uncertainty fusion strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

from ..core.config import UncertaintyConfig

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """
    Container for uncertainty estimation results.

    Attributes:
        combined_uncertainty: Fused uncertainty map [H, W]
        mc_variance: Variance from MC Dropout [H, W]
        flip_consistency: Uncertainty from flip consistency [H, W]
        ensemble_variance: Variance across ensemble models [H, W]
        sources: Dictionary of all uncertainty sources
        calibration_factor: Calibration multiplier for uncertainty
    """
    combined_uncertainty: torch.Tensor  # [H, W]
    mc_variance: Optional[torch.Tensor] = None
    flip_consistency: Optional[torch.Tensor] = None
    ensemble_variance: Optional[torch.Tensor] = None
    sources: Dict[str, torch.Tensor] = field(default_factory=dict)
    calibration_factor: float = 1.0

    def get_confidence(self) -> torch.Tensor:
        """
        Convert uncertainty to confidence scores.

        Returns:
            Confidence map [H, W] in range [0, 1]
        """
        # Higher uncertainty = lower confidence
        return torch.exp(-self.combined_uncertainty * self.calibration_factor)

    def filter_by_uncertainty(
        self,
        points: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter points by uncertainty threshold.

        Args:
            points: Point cloud [N, 3] or coordinates [H, W, 3]
            threshold: Maximum allowed uncertainty (0-1)

        Returns:
            Tuple of (filtered_points, mask)
        """
        mask = self.combined_uncertainty < threshold
        if points.dim() == 3:  # [H, W, 3]
            filtered = points[mask]
        else:  # [N, 3]
            flat_mask = mask.flatten()[:len(points)]
            filtered = points[flat_mask]
        return filtered, mask


class MCDropoutEstimator:
    """
    Monte Carlo Dropout for epistemic uncertainty estimation.

    Runs multiple forward passes with dropout enabled at inference time
    and measures the variance in predictions.
    """

    def __init__(
        self,
        n_passes: int = 10,
        dropout_rate: float = 0.1
    ):
        """
        Initialize MC Dropout estimator.

        Args:
            n_passes: Number of forward passes (default 10 for 6GB GPUs)
            dropout_rate: Dropout probability
        """
        self.n_passes = n_passes
        self.dropout_rate = dropout_rate

    def estimate(
        self,
        images: torch.Tensor,
        model: nn.Module,
        inference_fn: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate epistemic uncertainty using MC Dropout.

        Args:
            images: Input images [B, C, H, W]
            model: Depth estimation model
            inference_fn: Optional custom inference function

        Returns:
            Tuple of (mean_depth, variance) where each is [B, H, W]
        """
        logger.debug(f"Running MC Dropout with {self.n_passes} passes")

        # Enable dropout for all dropout layers
        self._enable_dropout(model)

        predictions = []
        with torch.no_grad():
            for i in range(self.n_passes):
                if inference_fn is not None:
                    depth = inference_fn(images)
                else:
                    depth = model(images)

                # Handle different output formats
                if hasattr(depth, 'predicted_depth'):
                    depth = depth.predicted_depth
                if depth.dim() == 4 and depth.shape[1] == 1:
                    depth = depth.squeeze(1)

                predictions.append(depth)

        # Restore eval mode
        model.eval()

        # Stack predictions and compute statistics
        predictions = torch.stack(predictions, dim=0)  # [N, B, H, W]

        mean_depth = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        logger.debug(f"MC Dropout variance: mean={variance.mean():.4f}, max={variance.max():.4f}")

        return mean_depth, variance

    def _enable_dropout(self, model: nn.Module) -> None:
        """Enable dropout layers while keeping rest in eval mode."""
        model.eval()  # First set everything to eval
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                module.p = self.dropout_rate


class FlipConsistencyEstimator:
    """
    Flip consistency for uncertainty estimation.

    Measures prediction stability by comparing depth of original image
    with flipped depth of horizontally flipped image.
    """

    def estimate(
        self,
        images: torch.Tensor,
        model: nn.Module,
        inference_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Estimate uncertainty from flip consistency.

        Args:
            images: Input images [B, C, H, W]
            model: Depth estimation model
            inference_fn: Optional custom inference function

        Returns:
            Inconsistency map [B, H, W] (higher = more uncertain)
        """
        logger.debug("Computing flip consistency...")

        with torch.no_grad():
            # Original prediction
            if inference_fn is not None:
                depth_original = inference_fn(images)
            else:
                depth_original = model(images)

            if hasattr(depth_original, 'predicted_depth'):
                depth_original = depth_original.predicted_depth
            if depth_original.dim() == 4:
                depth_original = depth_original.squeeze(1)

            # Flipped prediction
            images_flipped = torch.flip(images, dims=[3])  # Horizontal flip

            if inference_fn is not None:
                depth_flipped = inference_fn(images_flipped)
            else:
                depth_flipped = model(images_flipped)

            if hasattr(depth_flipped, 'predicted_depth'):
                depth_flipped = depth_flipped.predicted_depth
            if depth_flipped.dim() == 4:
                depth_flipped = depth_flipped.squeeze(1)

            # Flip back for comparison
            depth_flipped_back = torch.flip(depth_flipped, dims=[2])

            # Compute absolute difference (normalized)
            depth_range = depth_original.max() - depth_original.min() + 1e-6
            inconsistency = torch.abs(depth_original - depth_flipped_back) / depth_range

        logger.debug(f"Flip consistency: mean={inconsistency.mean():.4f}")

        return inconsistency


class EnsembleEstimator:
    """
    Ensemble-based uncertainty estimation.

    Uses variance across multiple models as uncertainty measure.
    """

    def estimate(
        self,
        images: torch.Tensor,
        models: List[nn.Module],
        inference_fns: Optional[List[Callable]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty from model ensemble.

        Args:
            images: Input images [B, C, H, W]
            models: List of depth estimation models
            inference_fns: Optional list of custom inference functions

        Returns:
            Tuple of (mean_depth, variance) where each is [B, H, W]
        """
        if len(models) < 2:
            logger.warning("Ensemble requires at least 2 models")
            return None, None

        logger.debug(f"Running ensemble with {len(models)} models")

        predictions = []
        with torch.no_grad():
            for i, model in enumerate(models):
                if inference_fns and i < len(inference_fns):
                    depth = inference_fns[i](images)
                else:
                    depth = model(images)

                if hasattr(depth, 'predicted_depth'):
                    depth = depth.predicted_depth
                if depth.dim() == 4:
                    depth = depth.squeeze(1)

                predictions.append(depth)

        predictions = torch.stack(predictions, dim=0)
        mean_depth = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        return mean_depth, variance


class UncertaintyFusion:
    """
    Fuses multiple uncertainty sources into a single uncertainty map.
    """

    def __init__(self, method: str = "weighted_average"):
        """
        Initialize fusion strategy.

        Args:
            method: Fusion method ('weighted_average', 'max', 'learned')
        """
        self.method = method

        # Default weights for different sources
        self.weights = {
            'mc_variance': 0.4,
            'flip_consistency': 0.3,
            'ensemble_variance': 0.3,
            'local_variance': 0.2
        }

    def fuse(
        self,
        uncertainties: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Fuse multiple uncertainty sources.

        Args:
            uncertainties: Dictionary mapping source names to uncertainty maps
            weights: Optional custom weights for each source

        Returns:
            Fused uncertainty map [H, W]
        """
        if not uncertainties:
            raise ValueError("No uncertainty sources provided")

        weights = weights or self.weights

        if self.method == "weighted_average":
            return self._weighted_average(uncertainties, weights)
        elif self.method == "max":
            return self._max_fusion(uncertainties)
        elif self.method == "learned":
            return self._learned_fusion(uncertainties)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

    def _weighted_average(
        self,
        uncertainties: Dict[str, torch.Tensor],
        weights: Dict[str, float]
    ) -> torch.Tensor:
        """Weighted average of uncertainty sources."""
        result = None
        total_weight = 0.0

        for name, uncertainty in uncertainties.items():
            w = weights.get(name, 0.25)  # Default weight

            # Normalize uncertainty to [0, 1]
            u_min = uncertainty.min()
            u_max = uncertainty.max()
            uncertainty_norm = (uncertainty - u_min) / (u_max - u_min + 1e-6)

            if result is None:
                result = w * uncertainty_norm
            else:
                result = result + w * uncertainty_norm
            total_weight += w

        if total_weight > 0:
            result = result / total_weight

        return result

    def _max_fusion(
        self,
        uncertainties: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Take maximum uncertainty across sources."""
        stacked = []
        for uncertainty in uncertainties.values():
            # Normalize
            u_min = uncertainty.min()
            u_max = uncertainty.max()
            uncertainty_norm = (uncertainty - u_min) / (u_max - u_min + 1e-6)
            stacked.append(uncertainty_norm)

        stacked = torch.stack(stacked, dim=0)
        return stacked.max(dim=0)[0]

    def _learned_fusion(
        self,
        uncertainties: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Learned fusion (placeholder for future neural network-based fusion).
        Currently falls back to weighted average.
        """
        logger.warning("Learned fusion not yet implemented, using weighted average")
        return self._weighted_average(uncertainties, self.weights)


class DepthUncertaintyEstimator:
    """
    High-level uncertainty estimator combining multiple methods.
    """

    def __init__(self, config: UncertaintyConfig):
        """
        Initialize uncertainty estimator.

        Args:
            config: Uncertainty estimation configuration
        """
        self.config = config

        # Initialize sub-estimators
        self.mc_estimator = MCDropoutEstimator(
            n_passes=config.mc_dropout_passes,
            dropout_rate=config.dropout_rate
        ) if config.enable_mc_dropout else None

        self.flip_estimator = FlipConsistencyEstimator() if config.enable_flip_consistency else None

        self.ensemble_estimator = EnsembleEstimator() if config.enable_ensemble else None

        self.fusion = UncertaintyFusion(method=config.fusion_method)

        logger.info(
            f"Initialized uncertainty estimator: "
            f"MC={config.enable_mc_dropout}, "
            f"Flip={config.enable_flip_consistency}, "
            f"Ensemble={config.enable_ensemble}"
        )

    @torch.amp.autocast(device_type='cuda', enabled=True)
    def estimate(
        self,
        images: torch.Tensor,
        model: nn.Module,
        depth_maps: Optional[torch.Tensor] = None,
        ensemble_models: Optional[List[nn.Module]] = None,
        inference_fn: Optional[Callable] = None
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty for depth predictions.

        Args:
            images: Input images [B, C, H, W]
            model: Primary depth estimation model
            depth_maps: Optional pre-computed depth maps
            ensemble_models: Optional list of models for ensemble
            inference_fn: Optional custom inference function

        Returns:
            UncertaintyEstimate with all computed uncertainties
        """
        uncertainties = {}
        mc_variance = None
        flip_consistency = None
        ensemble_variance = None

        # MC Dropout
        if self.mc_estimator is not None:
            try:
                mean_depth, mc_variance = self.mc_estimator.estimate(
                    images, model, inference_fn
                )
                uncertainties['mc_variance'] = mc_variance
                logger.debug("MC Dropout uncertainty computed")
            except Exception as e:
                logger.warning(f"MC Dropout failed: {e}")

        # Flip consistency
        if self.flip_estimator is not None:
            try:
                flip_consistency = self.flip_estimator.estimate(
                    images, model, inference_fn
                )
                uncertainties['flip_consistency'] = flip_consistency
                logger.debug("Flip consistency computed")
            except Exception as e:
                logger.warning(f"Flip consistency failed: {e}")

        # Ensemble
        if self.ensemble_estimator is not None and ensemble_models:
            try:
                _, ensemble_variance = self.ensemble_estimator.estimate(
                    images, ensemble_models
                )
                if ensemble_variance is not None:
                    uncertainties['ensemble_variance'] = ensemble_variance
                    logger.debug("Ensemble uncertainty computed")
            except Exception as e:
                logger.warning(f"Ensemble uncertainty failed: {e}")

        # Local variance from depth maps
        if depth_maps is not None:
            local_variance = self._compute_local_variance(depth_maps)
            uncertainties['local_variance'] = local_variance

        # Fuse uncertainties
        if uncertainties:
            combined = self.fusion.fuse(uncertainties)
        else:
            # Fallback: uniform uncertainty
            if depth_maps is not None:
                combined = torch.ones_like(depth_maps[0]) * 0.5
            else:
                combined = torch.ones(images.shape[2], images.shape[3], device=images.device) * 0.5

        return UncertaintyEstimate(
            combined_uncertainty=combined,
            mc_variance=mc_variance,
            flip_consistency=flip_consistency,
            ensemble_variance=ensemble_variance,
            sources=uncertainties
        )

    def _compute_local_variance(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Compute local variance as additional uncertainty source."""
        if depth_maps.dim() == 3:
            depth = depth_maps[0]
        else:
            depth = depth_maps

        kernel_size = 5
        padding = kernel_size // 2

        depth_unfold = F.unfold(
            depth.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            padding=padding
        )

        variance = depth_unfold.var(dim=1)
        return variance.view(depth.shape)
