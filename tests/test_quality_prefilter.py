import numpy as np

from src.core.config import SystemConfig
from src.core.measurement_system_gpu import MeasurementSystemGPU


def test_prefilter_no_paths_returns_original():
    system = MeasurementSystemGPU.__new__(MeasurementSystemGPU)
    system.config = SystemConfig(enable_capture_quality_filter=True)

    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)]
    filtered_images, filtered_paths, info = system._prefilter_low_quality_images(images, None)

    assert filtered_images is images
    assert filtered_paths is None
    assert info is None


def test_prefilter_skip_when_disabled():
    system = MeasurementSystemGPU.__new__(MeasurementSystemGPU)
    system.config = SystemConfig(enable_capture_quality_filter=False)

    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)]
    paths = []
    filtered_images, filtered_paths, info = system._prefilter_low_quality_images(images, paths)

    assert filtered_images is images
    assert filtered_paths is paths
    assert info is None


def test_adaptive_drop_fraction_increases_on_low_overlap():
    system = MeasurementSystemGPU.__new__(MeasurementSystemGPU)
    system.config = SystemConfig(
        quality_drop_fraction=0.20,
        enable_adaptive_quality_drop=True,
        adaptive_quality_drop_min=0.10,
        adaptive_quality_drop_max=0.35,
        capture_quality_threshold=0.45,
    )

    summary = {
        'quality_score': 0.20,
        'overlap_median': 0.04,
        'overlap_std': 0.08,
    }
    value = system._adaptive_quality_drop_fraction(summary)
    assert 0.20 <= value <= 0.35


def test_adaptive_drop_fraction_respects_min_when_disabled():
    system = MeasurementSystemGPU.__new__(MeasurementSystemGPU)
    system.config = SystemConfig(
        quality_drop_fraction=0.18,
        enable_adaptive_quality_drop=False,
        adaptive_quality_drop_min=0.10,
        adaptive_quality_drop_max=0.35,
    )

    summary = {
        'quality_score': 0.10,
        'overlap_median': 0.01,
        'overlap_std': 0.20,
    }
    value = system._adaptive_quality_drop_fraction(summary)
    assert np.isclose(value, 0.18)
