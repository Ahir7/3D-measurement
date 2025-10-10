"""
Optimized configuration for NVIDIA GTX 1650 (4GB VRAM).

This config is specifically tuned for GPUs with limited memory.
"""

from src.core.config import (
    SystemConfig,
    GPUConfig,
    COLMAPConfig,
    Metric3DConfig,
    ScaleRecoveryConfig
)


def get_gtx1650_config() -> SystemConfig:
    """
    Get optimized configuration for GTX 1650.
    
    Optimizations:
    - Reduced image size (1024 max)
    - Smaller batch sizes
    - Memory-efficient settings
    - Conservative GPU memory fraction
    
    Returns:
        SystemConfig optimized for 4GB VRAM
    """
    config = SystemConfig(
        gpu=GPUConfig(
            device="cuda:0",
            mixed_precision=True,  # Use FP16 to save memory
            num_streams=2,  # Reduce streams to save memory
            memory_fraction=0.85,  # Leave some VRAM for OS
            allow_tf32=True  # Faster computation
        ),
        
        colmap=COLMAPConfig(
            num_features=8192,  # Reduced from 16384
            use_gpu=True,
            gpu_index="0",
            matching_method="exhaustive",
            max_num_matches=16384,  # Reduced from 32768
            ba_refine_focal_length=True,
            ba_refine_principal_point=True,
            ba_refine_extra_params=False,  # Disable to save memory
            min_num_matches=15,
            min_track_length=2
        ),
        
        metric3d=Metric3DConfig(
            model_name="metric3d_vit_large",
            input_size=(518, 518),  # Standard size
            max_input_size=(1024, 1024),  # Reduced from 4K
            use_mixed_precision=True,
            compile_model=False,  # Disable compilation to save memory
            use_tensorrt=False,
            depth_scale_factor=1.0,
            min_depth=0.1,
            max_depth=100.0
        ),
        
        scale_recovery=ScaleRecoveryConfig(
            marker_weight=0.40,
            imu_weight=0.25,
            depth_weight=0.20,
            object_weight=0.15,
            marker_types=["aruco", "qr"],  # Removed apriltag to save memory
            marker_size_mm=100.0,
            imu_sampling_rate=100.0,
            imu_gravity=(0.0, 0.0, -9.81),
            min_confidence=0.5,
            min_methods_required=2
        ),
        
        # Processing settings optimized for 4GB
        batch_size=1,  # Process one at a time
        max_image_size=1024,  # Reduced from 2048
        min_images=15,  # Research-based: minimum for good accuracy
        max_images=25,  # Optimal for GTX 1650: balance of accuracy (±2-3%) and memory
        
        # Output settings
        output_dir="output",
        save_pointcloud=True,
        save_depth_maps=False,  # Disable to save memory
        save_camera_poses=True,
        
        # Performance
        enable_profiling=False,
        log_level="INFO"
    )
    
    return config


# Usage example
if __name__ == "__main__":
    config = get_gtx1650_config()
    config.validate()
    print("GTX 1650 configuration validated successfully!")
    print(f"Max image size: {config.max_image_size}px")
    print(f"Max images: {config.max_images}")
    print(f"Memory fraction: {config.gpu.memory_fraction}")

