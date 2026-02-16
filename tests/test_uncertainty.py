"""
Unit tests for uncertainty estimation module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.depth.uncertainty import (
    UncertaintyEstimate,
    MCDropoutEstimator,
    FlipConsistencyEstimator,
    EnsembleEstimator,
    UncertaintyFusion,
    DepthUncertaintyEstimator
)
from src.core.config import UncertaintyConfig


class TestUncertaintyEstimate:
    """Tests for UncertaintyEstimate dataclass."""

    def test_creation(self):
        """Test basic creation."""
        combined = torch.rand(100, 100)
        estimate = UncertaintyEstimate(combined_uncertainty=combined)

        assert estimate.combined_uncertainty.shape == (100, 100)
        assert estimate.mc_variance is None
        assert estimate.flip_consistency is None

    def test_get_confidence(self):
        """Test confidence conversion."""
        # Low uncertainty should give high confidence
        low_uncertainty = torch.zeros(100, 100)
        estimate = UncertaintyEstimate(combined_uncertainty=low_uncertainty)

        confidence = estimate.get_confidence()
        assert torch.all(confidence > 0.9)

        # High uncertainty should give low confidence
        high_uncertainty = torch.ones(100, 100) * 2
        estimate2 = UncertaintyEstimate(combined_uncertainty=high_uncertainty)

        confidence2 = estimate2.get_confidence()
        assert torch.all(confidence2 < 0.5)

    def test_filter_by_uncertainty(self):
        """Test point filtering by uncertainty."""
        uncertainty = torch.tensor([
            [0.1, 0.2, 0.8],
            [0.3, 0.9, 0.1],
            [0.7, 0.4, 0.2]
        ])
        estimate = UncertaintyEstimate(combined_uncertainty=uncertainty)

        # Create points matching uncertainty shape
        points = torch.rand(3, 3, 3)  # [H, W, 3]

        filtered, mask = estimate.filter_by_uncertainty(points, threshold=0.5)

        # Should filter out high uncertainty points
        assert mask.sum() < 9  # Some points filtered


class TestMCDropoutEstimator:
    """Tests for MC Dropout estimator."""

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = MCDropoutEstimator(n_passes=5, dropout_rate=0.2)

        assert estimator.n_passes == 5
        assert estimator.dropout_rate == 0.2

    def test_estimate_with_mock_model(self):
        """Test estimation with mock model."""
        estimator = MCDropoutEstimator(n_passes=3)

        # Create mock model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(16, 1, 1)
        )

        images = torch.randn(1, 3, 64, 64)

        mean_depth, variance = estimator.estimate(images, model)

        assert mean_depth.shape == (1, 64, 64)
        assert variance.shape == (1, 64, 64)
        assert torch.all(variance >= 0)  # Variance is non-negative


class TestFlipConsistencyEstimator:
    """Tests for flip consistency estimator."""

    def test_estimate_with_mock_model(self):
        """Test flip consistency estimation."""
        estimator = FlipConsistencyEstimator()

        # Simple model that just returns input mean
        class SimpleModel(nn.Module):
            def forward(self, x):
                return x.mean(dim=1, keepdim=True)

        model = SimpleModel()
        images = torch.randn(1, 3, 64, 64)

        inconsistency = estimator.estimate(images, model)

        assert inconsistency.shape == (1, 64, 64)
        # Inconsistency should be bounded
        assert inconsistency.min() >= 0


class TestUncertaintyFusion:
    """Tests for uncertainty fusion."""

    def test_weighted_average(self):
        """Test weighted average fusion."""
        fusion = UncertaintyFusion(method="weighted_average")

        uncertainties = {
            'source1': torch.ones(100, 100) * 0.3,
            'source2': torch.ones(100, 100) * 0.5
        }

        result = fusion.fuse(uncertainties)

        assert result.shape == (100, 100)
        # Result should be between input uncertainties (after normalization)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_max_fusion(self):
        """Test max fusion."""
        fusion = UncertaintyFusion(method="max")

        uncertainties = {
            'low': torch.zeros(100, 100),
            'high': torch.ones(100, 100)
        }

        result = fusion.fuse(uncertainties)

        assert result.shape == (100, 100)
        # Max fusion should be dominated by high uncertainty source

    def test_empty_uncertainties_raises(self):
        """Test that empty uncertainties raises error."""
        fusion = UncertaintyFusion()

        with pytest.raises(ValueError):
            fusion.fuse({})


class TestDepthUncertaintyEstimator:
    """Tests for high-level uncertainty estimator."""

    def test_initialization(self):
        """Test estimator initialization."""
        config = UncertaintyConfig(
            enable_mc_dropout=True,
            enable_flip_consistency=True,
            enable_ensemble=False
        )

        estimator = DepthUncertaintyEstimator(config)

        assert estimator.mc_estimator is not None
        assert estimator.flip_estimator is not None
        assert estimator.ensemble_estimator is None

    def test_initialization_all_disabled(self):
        """Test with all methods disabled."""
        config = UncertaintyConfig(
            enable_mc_dropout=False,
            enable_flip_consistency=False,
            enable_ensemble=False
        )

        estimator = DepthUncertaintyEstimator(config)

        assert estimator.mc_estimator is None
        assert estimator.flip_estimator is None

    def test_local_variance_computation(self):
        """Test local variance computation."""
        config = UncertaintyConfig()
        estimator = DepthUncertaintyEstimator(config)

        depth_maps = torch.randn(2, 100, 100)
        variance = estimator._compute_local_variance(depth_maps)

        assert variance.shape == (100, 100)
        assert torch.all(variance >= 0)
