"""
GPU-only configuration system for 3D measurement.

This module provides configuration management with validation
for GPU-accelerated 3D reconstruction and measurement.
"""

import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """GPU-specific configuration settings."""
    
    device: str = "cuda:0"
    mixed_precision: bool = True
    num_streams: int = 4
    memory_fraction: float = 0.92  # Optimized for 6GB+ GPUs
    allow_tf32: bool = True
    
    def validate(self) -> bool:
        """
        Validate GPU configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            RuntimeError: If GPU is not available
        """
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for this system. No CUDA device found.")
        
        device_id = int(self.device.split(':')[1]) if ':' in self.device else 0
        if device_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU device {device_id} not available. "
                f"Found {torch.cuda.device_count()} devices."
            )
        
        logger.info(f"GPU validated: {torch.cuda.get_device_name(device_id)}")
        return True


@dataclass
class COLMAPConfig:
    """COLMAP reconstruction configuration."""
    
    # Feature extraction - Optimized for RTX 2060 6GB
    num_features: int = 32768  # 2x increase for 6GB GPUs
    use_gpu: bool = True
    gpu_index: str = "0"
    
    # Matching
    matching_method: str = "exhaustive"
    max_num_matches: int = 65536  # 2x increase for 6GB GPUs
    
    # Bundle adjustment
    ba_refine_focal_length: bool = True
    ba_refine_principal_point: bool = True
    ba_refine_extra_params: bool = True
    
    # Quality thresholds
    min_num_matches: int = 15
    min_track_length: int = 2


@dataclass
class Metric3DConfig:
    """Metric3D depth estimation configuration."""
    
    model_name: str = "metric3d_vit_large"
    input_size: Tuple[int, int] = (518, 518)
    max_input_size: Tuple[int, int] = (2160, 2880)  # 6MP max for 6GB GPUs
    
    # Inference settings
    use_mixed_precision: bool = True
    compile_model: bool = False  # Disabled: 10+ min first-time compilation
    use_tensorrt: bool = False
    
    # Depth processing
    depth_scale_factor: float = 0.2214  # Keep 1.0; apply post-scale calibration instead
    min_depth: float = 0.1  # meters
    max_depth: float = 100.0  # meters
    
    # Depth normalization range (for indoor scenes)
    near_depth: float = 1.0  # meters - closest expected objects
    far_depth: float = 8.0   # meters - farthest expected objects


@dataclass
class ScaleRecoveryConfig:
    """Multi-source scale recovery configuration."""
    
    # Method weights - Depth-only (no markers/IMU/object)
    marker_weight: float = 0.0
    imu_weight: float = 0.0
    depth_weight: float = 1.0
    object_weight: float = 0.0
    
    # Marker detection (disabled in depth-only mode)
    marker_types: List[str] = field(default_factory=list)
    marker_size_mm: float = 100.0
    
    # IMU settings
    imu_sampling_rate: float = 100.0  # Hz
    imu_gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    
    # Confidence thresholds - depth-only accepts any estimate
    min_confidence: float = 0.0
    min_methods_required: int = 1  # Allow single method if confidence > threshold
    
    # Depth-only mode calibration
    # This corrects the absolute scale after depth-based scale recovery
    # Calibrate by measuring a known object and using: true_size / measured_size
    depth_only_calibration: float = 1.0  # 1.0 = no correction
    
    # Known object detection
    object_database: Optional[str] = None


@dataclass
class SystemConfig:
    """Complete system configuration with validation."""
    
    # Sub-configurations
    gpu: GPUConfig = field(default_factory=GPUConfig)
    colmap: COLMAPConfig = field(default_factory=COLMAPConfig)
    metric3d: Metric3DConfig = field(default_factory=Metric3DConfig)
    scale_recovery: ScaleRecoveryConfig = field(default_factory=ScaleRecoveryConfig)
    
    # Processing settings - Optimized for RTX 2060 6GB
    batch_size: int = 3  # 3x increase for 6GB GPUs
    max_image_size: int = 2048
    min_images: int = 3
    max_images: int = 50
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("output"))
    save_pointcloud: bool = True
    save_depth_maps: bool = False
    save_camera_poses: bool = True
    
    # Performance settings
    enable_profiling: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize and validate configuration after creation."""
        # Convert string paths to Path objects
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def validate(self) -> bool:
        """
        Validate complete system configuration.
        
        Returns:
            True if all configurations are valid
            
        Raises:
            RuntimeError: If any configuration is invalid
            ValueError: If parameter values are out of range
        """
        # Validate GPU
        self.gpu.validate()
        
        # Validate image limits
        if self.min_images < 2:
            raise ValueError("min_images must be at least 2")
        
        if self.max_images < self.min_images:
            raise ValueError("max_images must be >= min_images")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        
        if self.max_image_size < 512:
            raise ValueError("max_image_size must be at least 512")
        
        # Validate scale recovery weights
        total_weight = (
            self.scale_recovery.marker_weight +
            self.scale_recovery.imu_weight +
            self.scale_recovery.depth_weight +
            self.scale_recovery.object_weight
        )
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"Scale recovery weights must sum to 1.0, got {total_weight}"
            )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Configuration validated successfully")
        return True
    
    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'gpu': {
                'device': self.gpu.device,
                'mixed_precision': self.gpu.mixed_precision,
                'num_streams': self.gpu.num_streams
            },
            'colmap': {
                'num_features': self.colmap.num_features,
                'matching_method': self.colmap.matching_method
            },
            'metric3d': {
                'model_name': self.metric3d.model_name,
                'input_size': self.metric3d.input_size
            },
            'scale_recovery': {
                'marker_weight': self.scale_recovery.marker_weight,
                'imu_weight': self.scale_recovery.imu_weight,
                'depth_weight': self.scale_recovery.depth_weight,
                'object_weight': self.scale_recovery.object_weight
            },
            'processing': {
                'batch_size': self.batch_size,
                'max_image_size': self.max_image_size,
                'min_images': self.min_images,
                'max_images': self.max_images
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SystemConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            SystemConfig instance
        """
        gpu_config = GPUConfig(**config_dict.get('gpu', {}))
        colmap_config = COLMAPConfig(**config_dict.get('colmap', {}))
        metric3d_config = Metric3DConfig(**config_dict.get('metric3d', {}))
        scale_config = ScaleRecoveryConfig(**config_dict.get('scale_recovery', {}))
        
        processing = config_dict.get('processing', {})
        
        return cls(
            gpu=gpu_config,
            colmap=colmap_config,
            metric3d=metric3d_config,
            scale_recovery=scale_config,
            batch_size=processing.get('batch_size', 1),
            max_image_size=processing.get('max_image_size', 2048),
            min_images=processing.get('min_images', 3),
            max_images=processing.get('max_images', 50)
        )


# GPU optimization functions
def setup_gpu_optimizations(config: GPUConfig) -> None:
    """
    Setup GPU optimizations based on configuration.
    
    Args:
        config: GPU configuration object
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required but not available")
    
    # Set device
    torch.cuda.set_device(config.device)
    
    # Enable TF32 for faster computation on Ampere+ GPUs
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for matmul and cuDNN")
    
    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True
    
    # Set memory fraction
    if config.memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(
            config.memory_fraction,
            device=config.device
        )
        logger.info(f"GPU memory fraction set to {config.memory_fraction}")
    
    logger.info(f"GPU optimizations configured for {torch.cuda.get_device_name()}")


def get_gpu_info() -> Dict[str, any]:
    """
    Get current GPU information and statistics.
    
    Returns:
        Dictionary with GPU information
    """
    if not torch.cuda.is_available():
        return {'available': False}
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    return {
        'available': True,
        'device_id': device,
        'name': torch.cuda.get_device_name(device),
        'total_memory_gb': props.total_memory / 1e9,
        'allocated_memory_gb': torch.cuda.memory_allocated(device) / 1e9,
        'reserved_memory_gb': torch.cuda.memory_reserved(device) / 1e9,
        'cuda_version': torch.version.cuda,
        'compute_capability': f"{props.major}.{props.minor}",
        'multi_processor_count': props.multi_processor_count
    }

