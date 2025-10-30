"""
Optimized configuration for NVIDIA RTX 2060 (6GB VRAM).

This config maximizes quality and performance for 6GB VRAM GPUs.
Significantly improved over GTX 1650 Mobile (4GB) settings.
"""

from src.core.config import (
    SystemConfig,
    GPUConfig,
    COLMAPConfig,
    Metric3DConfig,
    ScaleRecoveryConfig
)


def get_rtx2060_config() -> SystemConfig:
    """
    Get optimized configuration for RTX 2060 Desktop 6GB.
    
    Improvements over GTX 1650 Mobile:
    - 2x more SIFT features (16K → 32K)
    - Larger batch sizes for depth estimation
    - Higher resolution support
    - More aggressive GPU memory usage
    - AprilTag support enabled
    
    Returns:
        SystemConfig optimized for 6GB VRAM
    """
    config = SystemConfig(
        gpu=GPUConfig(
            device="cuda:0",
            mixed_precision=True,
            num_streams=4,  # Increased from 2
            memory_fraction=0.92,  # More aggressive than 0.85
            allow_tf32=True
        ),
        
        colmap=COLMAPConfig(
            num_features=32768,  # 2x increase from GTX 1650 (16384)
            use_gpu=True,
            gpu_index="0",
            matching_method="exhaustive",
            max_num_matches=65536,  # 2x increase from 32768
            ba_refine_focal_length=True,
            ba_refine_principal_point=True,
            ba_refine_extra_params=True,  # Re-enabled for better accuracy
            min_num_matches=15,
            min_track_length=2
        ),
        
        metric3d=Metric3DConfig(
            model_name="metric3d_vit_large",
            input_size=(518, 518),
            max_input_size=(2160, 2880),  # 6MP max (was 1024x1024 for GTX 1650)
            use_mixed_precision=True,
            compile_model=False,  # Disabled due to 10+ min compilation time
            use_tensorrt=False,
            depth_scale_factor=1.0,
            min_depth=0.1,
            max_depth=100.0,
            near_depth=1.0,  # 1m for typical indoor close objects
            far_depth=8.0    # 8m for typical indoor far objects
        ),
        
        scale_recovery=ScaleRecoveryConfig(
            marker_weight=0.0,
            imu_weight=0.0,
            depth_weight=1.0,
            object_weight=0.0,
            marker_types=[],  # depth-only
            marker_size_mm=100.0,
            imu_sampling_rate=100.0,
            imu_gravity=(0.0, 0.0, -9.81),
            min_confidence=0.0,
            min_methods_required=1,
            depth_only_calibration=1.0
        ),
        
        # Processing settings optimized for 6GB
        batch_size=3,  # 3x increase from GTX 1650 (was 1)
        max_image_size=2048,  # 2x increase (was 1024)
        min_images=15,
        max_images=50,  # 2x increase (was 25)
        
        # Output settings
        output_dir="output",
        save_pointcloud=True,
        save_depth_maps=False,  # Still disabled to save disk space
        save_camera_poses=True,
        
        # Performance
        enable_profiling=False,
        log_level="INFO"
    )
    
    return config


# Usage example
if __name__ == "__main__":
    import torch
    
    config = get_rtx2060_config()
    
    print("\n" + "="*60)
    print("RTX 2060 Configuration")
    print("="*60)
    print(f"GPU Device: {config.gpu.device}")
    print(f"GPU Memory Fraction: {config.gpu.memory_fraction}")
    print(f"Mixed Precision: {config.gpu.mixed_precision}")
    print()
    print(f"COLMAP Features: {config.colmap.num_features}")
    print(f"Max Matches: {config.colmap.max_num_matches}")
    print(f"Bundle Adjustment Extra Params: {config.colmap.ba_refine_extra_params}")
    print()
    print(f"Depth Batch Size: {config.batch_size}")
    print(f"Max Image Size: {config.max_image_size}px")
    print(f"Max Images: {config.max_images}")
    print()
    print(f"Marker Types: {', '.join(config.scale_recovery.marker_types)}")
    print("="*60)
    
    # Verify GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✓ GPU Detected: {gpu_name}")
        print(f"✓ Total VRAM: {gpu_mem:.2f} GB")
        print(f"✓ Usable VRAM: {gpu_mem * config.gpu.memory_fraction:.2f} GB")
    else:
        print("\n✗ No GPU detected!")

