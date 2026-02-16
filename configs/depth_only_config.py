"""Depth-only configuration compatible with current config API."""

from src.core.config import SystemConfig, ScaleRecoveryConfig, GPUConfig, Metric3DConfig


def get_config() -> SystemConfig:
    """Return a depth-only configuration tuned for limited-VRAM GPUs."""
    config = SystemConfig()

    config.scale_recovery = ScaleRecoveryConfig(
        marker_weight=0.0,
        imu_weight=0.0,
        depth_weight=1.0,
        object_weight=0.0,
        marker_types=[],
        marker_size_mm=100.0,
        min_confidence=0.0,
        min_methods_required=1,
        depth_confidence_min=0.4,
        depth_confidence_weight_power=1.3,
        depth_only_calibration=1.0,
    )

    config.gpu = GPUConfig(
        device="cuda:0",
        mixed_precision=True,
        num_streams=2,
        memory_fraction=0.9,
        allow_tf32=True,
    )

    config.metric3d = Metric3DConfig(
        model_name="metric3d_vit_large",
        input_size=(518, 518),
        max_input_size=(1024, 1024),
        use_mixed_precision=True,
        compile_model=False,
        use_tensorrt=False,
        depth_scale_factor=1.0,
        min_depth=0.1,
        max_depth=100.0,
        near_depth=1.0,
        far_depth=8.0,
    )

    config.batch_size = 2
    config.max_image_size = 1024
    config.min_images = 15
    config.max_images = 25

    return config


config = get_config()

