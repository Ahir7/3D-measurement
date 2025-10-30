# Calibrated configuration
# Use this for depth-only measurements

from src.core.config import SystemConfig, ScaleRecoveryConfig

def get_config():
    config = SystemConfig()
    config.scale_recovery = ScaleRecoveryConfig(
        marker_weight=0.0,
        depth_weight=1.0,
        object_weight=0.0,
        imu_weight=0.0,
        min_confidence=0.1,
        min_methods_required=1
    )
    return config

config = get_config()
