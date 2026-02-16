"""
GPU-only configuration system for 3D measurement.

This module provides configuration management with validation
for GPU-accelerated 3D reconstruction and measurement.
"""

import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from enum import Enum

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


class DepthModelType(Enum):
    """Supported depth estimation models."""
    DPT_LARGE = "dpt_large"
    DEPTH_PRO = "depth_pro"
    DEPTH_ANYTHING_V2 = "depth_anything_v2"
    MIDAS_V3 = "midas_v3"


class UncertaintyFusionMethod(Enum):
    """Methods for fusing multiple uncertainty sources."""
    WEIGHTED_AVERAGE = "weighted_average"
    MAX = "max"
    LEARNED = "learned"


@dataclass
class ModelSelectionConfig:
    """Configuration for multi-model depth estimation."""

    # Primary model selection
    primary_model: str = "dpt_large"  # dpt_large | depth_pro | depth_anything_v2 | midas_v3
    fallback_model: Optional[str] = None

    # Model weights directory
    model_weights_dir: Path = field(default_factory=lambda: Path("models/depth"))

    # Fine-tuning options
    enable_fine_tuning: bool = False
    fine_tune_head_only: bool = True  # Only fine-tune prediction head
    fine_tune_checkpoint: Optional[Path] = None

    # Model-specific settings
    depth_anything_variant: str = "vitl"  # vits | vitb | vitl | vitg
    midas_variant: str = "dpt_beit_large_512"  # dpt_beit_large_512 | dpt_swin2_large_384

    def __post_init__(self):
        if isinstance(self.model_weights_dir, str):
            self.model_weights_dir = Path(self.model_weights_dir)
        if self.fine_tune_checkpoint and isinstance(self.fine_tune_checkpoint, str):
            self.fine_tune_checkpoint = Path(self.fine_tune_checkpoint)


@dataclass
class UncertaintyConfig:
    """Configuration for depth uncertainty estimation."""

    # Monte Carlo Dropout
    enable_mc_dropout: bool = True
    mc_dropout_passes: int = 10  # Conservative for 6GB GPUs
    dropout_rate: float = 0.1

    # Ensemble (disabled by default - memory intensive)
    enable_ensemble: bool = False
    ensemble_size: int = 5
    ensemble_models: List[str] = field(default_factory=list)

    # Flip consistency
    enable_flip_consistency: bool = True

    # Uncertainty fusion
    fusion_method: str = "weighted_average"  # weighted_average | max | learned

    # Uncertainty-based filtering
    uncertainty_threshold: float = 0.5  # Reject points with uncertainty > threshold
    enable_uncertainty_weighting: bool = True  # Weight points by inverse uncertainty

    # Calibration
    enable_uncertainty_calibration: bool = False
    calibration_data_path: Optional[Path] = None

    def __post_init__(self):
        if self.calibration_data_path and isinstance(self.calibration_data_path, str):
            self.calibration_data_path = Path(self.calibration_data_path)


@dataclass
class GeometricPriorsConfig:
    """Configuration for geometric prior constraints."""

    # Rectangular prism fitting
    enable_prism_fitting: bool = True
    prism_fitting_iterations: int = 100
    prism_inlier_threshold: float = 0.02  # meters

    # Plane detection (RANSAC)
    enable_plane_detection: bool = True
    ransac_iterations: int = 1000
    ransac_threshold: float = 0.01  # meters
    min_plane_points: int = 50
    max_planes: int = 6  # Maximum number of planes to detect

    # Box topology validation
    enable_box_topology: bool = True
    orthogonality_tolerance_degrees: float = 5.0
    parallelism_tolerance_degrees: float = 5.0

    # Epipolar consistency (multi-view)
    enable_epipolar_check: bool = True
    epipolar_threshold: float = 2.0  # pixels
    min_epipolar_inliers: int = 50

    # Refinement options
    enable_geometric_refinement: bool = True
    refinement_iterations: int = 50
    refinement_learning_rate: float = 0.01


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
    depth_scale_factor: float = 1.0  # Keep at 1.0; use depth_only_calibration for post-scale correction
    min_depth: float = 0.1  # meters
    max_depth: float = 100.0  # meters
    depth_normalization_mode: str = "global_percentile"  # global_percentile | per_image_percentile | none
    percentile_low: float = 0.01
    percentile_high: float = 0.99

    # Depth normalization range (for indoor scenes)
    near_depth: float = 1.0  # meters - closest expected objects
    far_depth: float = 8.0   # meters - farthest expected objects

    # Model selection and uncertainty (new for accuracy enhancement)
    model_selection: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)


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
    low_confidence_threshold: float = 0.35

    # Depth-aligned fusion robustness
    depth_aligned_min_views: int = 3
    depth_confidence_min: float = 0.35
    depth_confidence_weight_power: float = 1.25
    
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
    geometric_priors: GeometricPriorsConfig = field(default_factory=GeometricPriorsConfig)
    
    # Processing settings - Optimized for RTX 2060 6GB
    batch_size: int = 3  # 3x increase for 6GB GPUs
    max_image_size: int = 2048
    min_images: int = 3
    max_images: int = 50

    # Accuracy settings (image-only quality prefilter)
    enable_capture_quality_filter: bool = True
    capture_quality_threshold: float = 0.45
    quality_drop_fraction: float = 0.20
    enable_adaptive_quality_drop: bool = True
    adaptive_quality_drop_min: float = 0.10
    adaptive_quality_drop_max: float = 0.35
    min_images_after_quality_filter: int = 10
    
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

        # Validate quality prefilter settings
        if not (0.0 <= self.capture_quality_threshold <= 1.0):
            raise ValueError("capture_quality_threshold must be in [0, 1]")
        if not (0.0 <= self.quality_drop_fraction <= 0.7):
            raise ValueError("quality_drop_fraction must be in [0, 0.7]")
        if not (0.0 <= self.adaptive_quality_drop_min <= self.adaptive_quality_drop_max <= 0.7):
            raise ValueError("adaptive_quality_drop_min/max must satisfy 0 <= min <= max <= 0.7")
        if self.min_images_after_quality_filter < self.min_images:
            raise ValueError("min_images_after_quality_filter must be >= min_images")

        # Validate depth normalization settings
        valid_norm_modes = {"global_percentile", "per_image_percentile", "none"}
        if self.metric3d.depth_normalization_mode not in valid_norm_modes:
            raise ValueError(
                f"Invalid depth_normalization_mode: {self.metric3d.depth_normalization_mode}. "
                f"Expected one of {sorted(valid_norm_modes)}"
            )
        if not (0.0 <= self.metric3d.percentile_low < self.metric3d.percentile_high <= 1.0):
            raise ValueError(
                "percentile_low/percentile_high must satisfy 0 <= low < high <= 1"
            )
        
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

        if not (0.0 <= self.scale_recovery.depth_confidence_min <= 1.0):
            raise ValueError("depth_confidence_min must be in [0, 1]")
        if not (0.5 <= self.scale_recovery.depth_confidence_weight_power <= 3.0):
            raise ValueError("depth_confidence_weight_power must be in [0.5, 3.0]")

        # Validate uncertainty configuration
        if self.metric3d.uncertainty.mc_dropout_passes < 1:
            raise ValueError("mc_dropout_passes must be at least 1")
        if not (0.0 <= self.metric3d.uncertainty.dropout_rate <= 0.5):
            raise ValueError("dropout_rate must be in [0, 0.5]")
        if self.metric3d.uncertainty.fusion_method not in {"weighted_average", "max", "learned"}:
            raise ValueError("fusion_method must be 'weighted_average', 'max', or 'learned'")

        # Validate geometric priors configuration
        if self.geometric_priors.ransac_iterations < 10:
            raise ValueError("ransac_iterations must be at least 10")
        if not (0.0 < self.geometric_priors.ransac_threshold <= 0.1):
            raise ValueError("ransac_threshold must be in (0, 0.1]")
        if not (0.0 < self.geometric_priors.orthogonality_tolerance_degrees <= 15.0):
            raise ValueError("orthogonality_tolerance_degrees must be in (0, 15]")
        
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
                'input_size': self.metric3d.input_size,
                'model_selection': {
                    'primary_model': self.metric3d.model_selection.primary_model,
                    'fallback_model': self.metric3d.model_selection.fallback_model,
                    'enable_fine_tuning': self.metric3d.model_selection.enable_fine_tuning
                },
                'uncertainty': {
                    'enable_mc_dropout': self.metric3d.uncertainty.enable_mc_dropout,
                    'mc_dropout_passes': self.metric3d.uncertainty.mc_dropout_passes,
                    'enable_flip_consistency': self.metric3d.uncertainty.enable_flip_consistency,
                    'fusion_method': self.metric3d.uncertainty.fusion_method
                }
            },
            'scale_recovery': {
                'marker_weight': self.scale_recovery.marker_weight,
                'imu_weight': self.scale_recovery.imu_weight,
                'depth_weight': self.scale_recovery.depth_weight,
                'object_weight': self.scale_recovery.object_weight
            },
            'geometric_priors': {
                'enable_prism_fitting': self.geometric_priors.enable_prism_fitting,
                'enable_plane_detection': self.geometric_priors.enable_plane_detection,
                'enable_epipolar_check': self.geometric_priors.enable_epipolar_check,
                'enable_geometric_refinement': self.geometric_priors.enable_geometric_refinement
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

        # Handle nested metric3d config
        metric3d_dict = config_dict.get('metric3d', {})
        model_selection_dict = metric3d_dict.pop('model_selection', {})
        uncertainty_dict = metric3d_dict.pop('uncertainty', {})

        model_selection_config = ModelSelectionConfig(**model_selection_dict)
        uncertainty_config = UncertaintyConfig(**uncertainty_dict)

        metric3d_config = Metric3DConfig(
            **metric3d_dict,
            model_selection=model_selection_config,
            uncertainty=uncertainty_config
        )

        scale_config = ScaleRecoveryConfig(**config_dict.get('scale_recovery', {}))
        geometric_priors_config = GeometricPriorsConfig(**config_dict.get('geometric_priors', {}))

        processing = config_dict.get('processing', {})

        return cls(
            gpu=gpu_config,
            colmap=colmap_config,
            metric3d=metric3d_config,
            scale_recovery=scale_config,
            geometric_priors=geometric_priors_config,
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

