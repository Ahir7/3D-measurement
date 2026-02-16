import numpy as np

from src.core.config import ScaleRecoveryConfig
from src.scale.scale_optimizer import ScaleOptimizer


def test_fuse_view_scales_rejects_outlier_view():
    optimizer = ScaleOptimizer(ScaleRecoveryConfig(), device='cpu')

    scales = [0.98, 1.01, 1.00, 3.50]
    weights = [1.0, 1.0, 1.0, 0.2]

    fused = optimizer._fuse_view_scales(scales, weights)
    assert fused is not None

    scale, diagnostics, scales_filtered, weights_filtered = fused

    assert diagnostics['views_before_filter'] == 4
    assert diagnostics['views_after_filter'] >= 3
    assert 0.9 <= scale <= 1.1
    assert np.isclose(weights_filtered.sum(), 1.0)
    assert diagnostics['scale_dispersion'] < 0.2


def test_fuse_view_scales_single_view_passthrough():
    optimizer = ScaleOptimizer(ScaleRecoveryConfig(), device='cpu')

    fused = optimizer._fuse_view_scales([1.23], [0.5])
    assert fused is not None

    scale, diagnostics, scales_filtered, weights_filtered = fused
    assert np.isclose(scale, 1.23)
    assert diagnostics['views_after_filter'] == 1
    assert scales_filtered.shape[0] == 1
    assert np.isclose(weights_filtered[0], 1.0)


def test_weighted_median_prefers_high_confidence_cluster():
    optimizer = ScaleOptimizer(ScaleRecoveryConfig(), device='cpu')

    values = np.array([0.95, 1.00, 1.02, 1.75], dtype=np.float64)
    weights = np.array([0.2, 1.0, 1.0, 0.05], dtype=np.float64)

    weighted_median = optimizer._weighted_median(values, weights)
    assert 0.98 <= weighted_median <= 1.05
