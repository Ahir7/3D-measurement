import pytest

from src.core.config import SystemConfig, GPUConfig


def test_invalid_depth_normalization_mode_rejected(monkeypatch):
    monkeypatch.setattr(GPUConfig, 'validate', lambda self: True)
    config = SystemConfig()
    config.metric3d.depth_normalization_mode = 'invalid_mode'

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_percentile_bounds_rejected(monkeypatch):
    monkeypatch.setattr(GPUConfig, 'validate', lambda self: True)
    config = SystemConfig()
    config.metric3d.percentile_low = 0.95
    config.metric3d.percentile_high = 0.90

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_capture_quality_threshold_rejected(monkeypatch):
    monkeypatch.setattr(GPUConfig, 'validate', lambda self: True)
    config = SystemConfig()
    config.capture_quality_threshold = 1.5

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_quality_drop_fraction_rejected(monkeypatch):
    monkeypatch.setattr(GPUConfig, 'validate', lambda self: True)
    config = SystemConfig()
    config.quality_drop_fraction = 0.9

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_adaptive_quality_bounds_rejected(monkeypatch):
    monkeypatch.setattr(GPUConfig, 'validate', lambda self: True)
    config = SystemConfig()
    config.adaptive_quality_drop_min = 0.4
    config.adaptive_quality_drop_max = 0.2

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_depth_confidence_min_rejected(monkeypatch):
    monkeypatch.setattr(GPUConfig, 'validate', lambda self: True)
    config = SystemConfig()
    config.scale_recovery.depth_confidence_min = 1.2

    with pytest.raises(ValueError):
        config.validate()
