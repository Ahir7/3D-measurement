"""
Enhanced accuracy configuration for depth-only 3D measurement.

This configuration enables all accuracy enhancement features:
- Uncertainty estimation (MC Dropout + Flip Consistency)
- Geometric priors (Plane detection + Prism fitting)
- Multi-model fallback support

Optimized for RTX 2060 6GB while targeting:
- Dimension MAPE: < 2% (width/height), < 3% (depth)
- Volume MAPE: < 5%
- Confidence calibration ECE: < 0.05
"""

from pathlib import Path
from src.core.config import (
    SystemConfig,
    GPUConfig,
    COLMAPConfig,
    Metric3DConfig,
    ScaleRecoveryConfig,
    ModelSelectionConfig,
    UncertaintyConfig,
    GeometricPriorsConfig
)


def get_enhanced_config() -> SystemConfig:
    """
    Get fully enhanced accuracy configuration.

    Enables all accuracy improvement features while staying
    within 6GB GPU memory budget.
    """

    # GPU Configuration - optimized for RTX 2060 6GB
    gpu_config = GPUConfig(
        device="cuda:0",
        mixed_precision=True,
        num_streams=4,
        memory_fraction=0.92,
        allow_tf32=True
    )

    # COLMAP Configuration - increased features for better reconstruction
    colmap_config = COLMAPConfig(
        num_features=32768,
        use_gpu=True,
        gpu_index="0",
        matching_method="exhaustive",
        max_num_matches=65536,
        ba_refine_focal_length=True,
        ba_refine_principal_point=True,
        ba_refine_extra_params=True,
        min_num_matches=15,
        min_track_length=2
    )

    # Model Selection - DPT-Large primary with fallback
    model_selection = ModelSelectionConfig(
        primary_model="dpt_large",
        fallback_model=None,  # No fallback to conserve memory
        model_weights_dir=Path("models/depth"),
        enable_fine_tuning=False,
        fine_tune_head_only=True,
        depth_anything_variant="vitl"
    )

    # Uncertainty Configuration - balanced for 6GB GPU
    uncertainty_config = UncertaintyConfig(
        enable_mc_dropout=True,
        mc_dropout_passes=10,  # Conservative for memory
        dropout_rate=0.1,
        enable_ensemble=False,  # Disabled - too memory intensive
        ensemble_size=3,
        enable_flip_consistency=True,
        fusion_method="weighted_average",
        uncertainty_threshold=0.5,
        enable_uncertainty_weighting=True,
        enable_uncertainty_calibration=False
    )

    # Metric3D Configuration with enhanced settings
    metric3d_config = Metric3DConfig(
        model_name="metric3d_vit_large",
        input_size=(518, 518),
        max_input_size=(2160, 2880),
        use_mixed_precision=True,
        compile_model=False,
        use_tensorrt=False,
        depth_scale_factor=1.0,
        min_depth=0.1,
        max_depth=100.0,
        depth_normalization_mode="global_percentile",
        percentile_low=0.01,
        percentile_high=0.99,
        near_depth=1.0,
        far_depth=8.0,
        model_selection=model_selection,
        uncertainty=uncertainty_config
    )

    # Scale Recovery - depth-only mode
    scale_recovery_config = ScaleRecoveryConfig(
        marker_weight=0.0,
        imu_weight=0.0,
        depth_weight=1.0,
        object_weight=0.0,
        min_confidence=0.0,
        min_methods_required=1,
        low_confidence_threshold=0.35,
        depth_aligned_min_views=3,
        depth_confidence_min=0.35,
        depth_confidence_weight_power=1.25,
        depth_only_calibration=1.0
    )

    # Geometric Priors - all enabled
    geometric_priors_config = GeometricPriorsConfig(
        enable_prism_fitting=True,
        prism_fitting_iterations=100,
        prism_inlier_threshold=0.02,
        enable_plane_detection=True,
        ransac_iterations=1000,
        ransac_threshold=0.01,
        min_plane_points=50,
        max_planes=6,
        enable_box_topology=True,
        orthogonality_tolerance_degrees=5.0,
        parallelism_tolerance_degrees=5.0,
        enable_epipolar_check=True,
        epipolar_threshold=2.0,
        min_epipolar_inliers=50,
        enable_geometric_refinement=True,
        refinement_iterations=50,
        refinement_learning_rate=0.01
    )

    # System Configuration
    config = SystemConfig(
        gpu=gpu_config,
        colmap=colmap_config,
        metric3d=metric3d_config,
        scale_recovery=scale_recovery_config,
        geometric_priors=geometric_priors_config,
        batch_size=3,
        max_image_size=2048,
        min_images=3,
        max_images=50,
        enable_capture_quality_filter=True,
        capture_quality_threshold=0.45,
        quality_drop_fraction=0.20,
        enable_adaptive_quality_drop=True,
        adaptive_quality_drop_min=0.10,
        adaptive_quality_drop_max=0.35,
        min_images_after_quality_filter=10,
        output_dir=Path("output"),
        save_pointcloud=True,
        save_depth_maps=True,
        save_camera_poses=True,
        enable_profiling=False,
        log_level="INFO"
    )

    return config


def get_fast_config() -> SystemConfig:
    """
    Get fast configuration with minimal accuracy features.

    Useful for quick testing or when speed is more important
    than maximum accuracy.
    """
    config = get_enhanced_config()

    # Disable expensive features
    config.metric3d.uncertainty.enable_mc_dropout = False
    config.metric3d.uncertainty.mc_dropout_passes = 1
    config.metric3d.uncertainty.enable_flip_consistency = False

    config.geometric_priors.enable_prism_fitting = False
    config.geometric_priors.enable_plane_detection = False
    config.geometric_priors.enable_epipolar_check = False
    config.geometric_priors.enable_geometric_refinement = False

    config.enable_capture_quality_filter = False

    return config


def get_memory_constrained_config() -> SystemConfig:
    """
    Get configuration for GPUs with less than 6GB VRAM.

    Reduces memory usage while keeping key accuracy features.
    """
    config = get_enhanced_config()

    # Reduce batch size and image size
    config.batch_size = 2
    config.max_image_size = 1536

    # Reduce MC Dropout passes
    config.metric3d.uncertainty.mc_dropout_passes = 5

    # Simplify geometric priors
    config.geometric_priors.ransac_iterations = 500
    config.geometric_priors.prism_fitting_iterations = 50

    # Reduce COLMAP features
    config.colmap.num_features = 16384
    config.colmap.max_num_matches = 32768

    return config


# Default export
DEFAULT_CONFIG = get_enhanced_config()


if __name__ == "__main__":
    # Print configuration summary
    config = get_enhanced_config()

    print("Enhanced Accuracy Configuration")
    print("=" * 50)
    print(f"Primary Model: {config.metric3d.model_selection.primary_model}")
    print(f"MC Dropout: {config.metric3d.uncertainty.enable_mc_dropout} "
          f"({config.metric3d.uncertainty.mc_dropout_passes} passes)")
    print(f"Flip Consistency: {config.metric3d.uncertainty.enable_flip_consistency}")
    print(f"Plane Detection: {config.geometric_priors.enable_plane_detection}")
    print(f"Prism Fitting: {config.geometric_priors.enable_prism_fitting}")
    print(f"Epipolar Check: {config.geometric_priors.enable_epipolar_check}")
    print(f"Geometric Refinement: {config.geometric_priors.enable_geometric_refinement}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Max Image Size: {config.max_image_size}")
    print("=" * 50)
