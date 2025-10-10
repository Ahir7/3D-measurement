"""
Depth-Only Configuration

This configuration allows measurements using only depth estimation,
without requiring markers or other scale sources.

Note: Results will be less accurate without a physical reference.
Use the calibrate_scale.py tool to improve accuracy.
"""

from src.core.config import SystemConfig, ScaleRecoveryConfig, GPUConfig, ProcessingConfig

# Create base config
config = SystemConfig()

# Scale recovery: depth-only with reduced requirements
config.scale_recovery = ScaleRecoveryConfig(
    marker_types=['aruco'],      # ArUco markers if available
    marker_size_mm=100,          # Default marker size
    marker_weight=1.0,           # High weight if markers found
    depth_weight=1.0,            # Use depth estimation
    object_weight=0.3,           # Low weight for object detection
    imu_weight=0.0,              # No IMU data
    min_methods_required=1,      # ✅ REDUCED: Allow single method
    optimization_iterations=50
)

# GPU settings for GTX 1650 (4GB)
config.gpu = GPUConfig(
    device="cuda:0",
    mixed_precision=True,
    num_streams=4,
    memory_fraction=0.9,
    allow_tf32=True
)

# Processing settings optimized for accuracy
config.processing = ProcessingConfig(
    min_images=15,
    max_images=25,
    target_image_size=(768, 1024),
    batch_size=2,              # Process 2 images at a time
    compile_model=False        # Disabled on Windows
)

print("[INFO] Loaded depth-only configuration")
print("       - Minimum methods required: 1 (reduced from 2)")
print("       - Will use depth estimation for scale")
print("       - Use calibrate_scale.py to improve accuracy")

