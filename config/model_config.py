"""
Model configuration and parameters
"""

from pathlib import Path
from typing import Dict, Any

class ModelConfig:
    # Model Files
    DUST3R_MODEL_NAME = "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    MAST3R_MODEL_NAME = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    
    # Model URLs
    DUST3R_MODEL_URL = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    MAST3R_MODEL_URL = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    
    # Processing Parameters
    IMAGE_SIZE = 512
    BATCH_SIZE = 1
    
    # Scale Recovery Parameters
    SCALE_RECOVERY_METHODS = {
        'imu': {
            'enabled': True,
            'weight': 0.35,
            'min_motion': 0.05,  # meters
            'max_motion': 5.0,   # meters
            'sampling_rate': 100  # Hz
        },
        'metadata': {
            'enabled': True,
            'weight': 0.25,
            'sensor_widths': {
                'iPhone': 4.8,    # mm
                'Samsung': 5.6,   # mm
                'Google': 5.0,    # mm
                'OnePlus': 5.4,   # mm
                'Xiaomi': 5.2     # mm
            },
            'default_sensor_width': 5.0  # mm
        },
        'geometric': {
            'enabled': True,
            'weight': 0.20,
            'typical_motion_range': [0.3, 0.8],  # meters
            'baseline_consistency_threshold': 0.1
        },
        'ml_depth': {
            'enabled': True,
            'weight': 0.15,
            'model_name': 'MiDaS',  # or 'DPT'
            'confidence_threshold': 0.3
        },
        'motion_parallax': {
            'enabled': True,
            'weight': 0.05,
            'typical_speed': 0.3  # m/s
        }
    }
    
    # Quality Assessment
    QUALITY_THRESHOLDS = {
        'min_points': 1000,
        'min_cameras': 2,
        'max_reprojection_error': 2.0,  # pixels
        'min_baseline': 0.01,  # relative to scene size
        'min_triangulation_angle': 5.0  # degrees
    }
    
    # Optimization Parameters
    GLOBAL_ALIGNMENT = {
        'mode': 'PointCloudOptimizer',
        'max_iterations': 100,
        'convergence_threshold': 1e-6,
        'regularization': 1e-3
    }
    
    # Output Configuration
    OUTPUT_FORMATS = {
        'json': True,
        'ply': False,  # Point cloud
        'obj': False,  # Mesh
        'csv': False   # Tabular data
    }
    
    @classmethod
    def get_model_path(cls, model_name: str, models_dir: Path) -> Path:
        """Get full path to model file"""
        if model_name == "dust3r":
            return models_dir / cls.DUST3R_MODEL_NAME
        elif model_name == "mast3r":
            return models_dir / cls.MAST3R_MODEL_NAME
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @classmethod
    def get_scale_method_config(cls, method_name: str) -> Dict[str, Any]:
        """Get configuration for specific scale recovery method"""
        return cls.SCALE_RECOVERY_METHODS.get(method_name, {})
    
    @classmethod
    def is_method_enabled(cls, method_name: str) -> bool:
        """Check if scale recovery method is enabled"""
        method_config = cls.get_scale_method_config(method_name)
        return method_config.get('enabled', False)
    
    @classmethod
    def get_method_weight(cls, method_name: str) -> float:
        """Get weight for scale recovery method"""
        method_config = cls.get_scale_method_config(method_name)
        return method_config.get('weight', 0.0)

# Create singleton instance
model_config = ModelConfig()
