"""
GPU-only 3D measurement system.

Main pipeline combining COLMAP, Metric3D, and multi-source scale recovery.
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .config import SystemConfig, setup_gpu_optimizations, get_gpu_info
from .calibration import CameraCalibrator, CameraIntrinsics
from ..reconstruction.colmap_gpu import COLMAPReconstructor, Reconstruction3D
from ..depth.metric3d_gpu import Metric3DEstimator, DepthEstimation
from ..scale.scale_optimizer import ScaleOptimizer, ScaleResult
from ..utils.geometry import (
    remove_outliers,
    compute_oriented_bbox,
    estimate_measurement_errors,
    compute_point_cloud_quality
)

logger = logging.getLogger(__name__)


@dataclass
class MeasurementResult:
    """Complete measurement result with GPU metrics."""
    
    measurements: Dict[str, float]
    confidence: float
    gpu_time: float
    total_time: float
    scale_result: ScaleResult
    reconstruction: Reconstruction3D
    depth_estimations: Optional[List[DepthEstimation]] = None
    pointcloud_path: Optional[str] = None
    error_bounds: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'success': True,
            'measurements': self.measurements,
            'confidence': self.confidence,
            'processing_times': {
                'gpu_time': self.gpu_time,
                'total_time': self.total_time
            },
            'scale_recovery': self.scale_result.to_dict(),
            'reconstruction_stats': self.reconstruction.to_dict(),
            'pointcloud_path': self.pointcloud_path
        }
        
        if self.error_bounds:
            result['error_bounds'] = self.error_bounds
        
        return result


class MeasurementSystemGPU:
    """GPU-only 3D measurement system."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize GPU measurement system.
        
        Args:
            config: System configuration object
            
        Raises:
            RuntimeError: If GPU is not available
        """
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for this system. No CUDA device found.")
        
        self.config = config or SystemConfig()
        self.config.validate()
        
        # Set device
        self.device = torch.device(self.config.gpu.device)
        
        # Setup GPU optimizations
        setup_gpu_optimizations(self.config.gpu)
        
        # Initialize components
        self._init_components()
        self._init_streams()
        self._preallocate_memory()
        
        # Log GPU info
        gpu_info = get_gpu_info()
        logger.info(f"Initialized on {gpu_info['name']}")
        logger.info(f"Total GPU memory: {gpu_info['total_memory_gb']:.2f} GB")
    
    def _init_components(self):
        """Initialize all processing components."""
        logger.info("Initializing processing components...")
        
        # Calibration
        self.calibrator = CameraCalibrator(self.config.gpu.device)
        
        # 3D Reconstruction
        self.reconstructor = COLMAPReconstructor(
            self.config.colmap,
            self.config.gpu.device
        )
        
        # Depth estimation
        self.depth_estimator = Metric3DEstimator(
            self.config.metric3d,
            self.config.gpu.device
        )
        
        # Scale recovery
        self.scale_optimizer = ScaleOptimizer(
            self.config.scale_recovery,
            self.config.gpu.device
        )
        
        logger.info("All components initialized")
    
    def _init_streams(self):
        """Initialize CUDA streams for parallel processing."""
        self.streams = [
            torch.cuda.Stream() for _ in range(self.config.gpu.num_streams)
        ]
        logger.debug(f"Initialized {len(self.streams)} CUDA streams")
    
    def _preallocate_memory(self):
        """Pre-allocate GPU memory buffers."""
        self.buffers = {
            'images': torch.empty(
                (self.config.batch_size, 3, 
                 self.config.max_image_size, 
                 self.config.max_image_size),
                device=self.config.gpu.device,
                dtype=torch.float16 if self.config.gpu.mixed_precision else torch.float32
            )
        }
        logger.debug("Pre-allocated GPU memory buffers")
    
    @torch.amp.autocast(device_type='cuda', enabled=True)
    def measure(
        self,
        images: List[np.ndarray],
        image_paths: Optional[List[Path]] = None,
        imu_data: Optional[List[Dict]] = None,
        metadata: Optional[List[Dict]] = None,
        known_intrinsics: Optional[CameraIntrinsics] = None
    ) -> MeasurementResult:
        """
        Measure dimensions from images.
        
        Args:
            images: List of input images as numpy arrays [H, W, 3]
            image_paths: Optional paths to image files
            imu_data: Optional IMU sensor data
            metadata: Optional image metadata (EXIF)
            known_intrinsics: Optional known camera calibration
            
        Returns:
            MeasurementResult with dimensions and metrics
            
        Raises:
            ValueError: If insufficient images provided
            RuntimeError: If measurement fails
        """
        # Validate inputs
        if len(images) < self.config.min_images:
            raise ValueError(
                f"At least {self.config.min_images} images required, got {len(images)}"
            )
        
        if len(images) > self.config.max_images:
            raise ValueError(
                f"Maximum {self.config.max_images} images allowed, got {len(images)}"
            )
        
        logger.info(f"Starting measurement with {len(images)} images")
        
        # Start timing
        total_start = time.time()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        try:
            # Transfer images to GPU
            images_gpu = self._transfer_to_gpu(images)
            logger.debug(f"Transferred {len(images)} images to GPU")
            
            # Parallel processing using streams
            with torch.cuda.stream(self.streams[0]):
                # 3D Reconstruction
                logger.info("Running 3D reconstruction...")
                reconstruction = self.reconstructor.reconstruct(
                    images_gpu,
                    image_paths=image_paths,
                    known_intrinsics=known_intrinsics
                )
            
            # Clear GPU memory after COLMAP before running Metric3D (4GB GPU constraint)
            torch.cuda.synchronize()
            logger.info(f"GPU memory before cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Move images back to CPU temporarily to free GPU memory
            images_cpu = images_gpu.cpu()
            del images_gpu
            torch.cuda.empty_cache()
            
            logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"Free GPU memory: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
            
            # Move images back to GPU for depth estimation
            images_gpu = images_cpu.to(self.device, non_blocking=True)
            del images_cpu
            
            with torch.cuda.stream(self.streams[1]):
                # Depth estimation
                logger.info("Estimating depth maps...")
                depth_estimations = self.depth_estimator.estimate_depth(
                    images_gpu,
                    return_confidence=True
                )
            
            # Synchronize streams
            torch.cuda.synchronize()
            
            # Convert depth estimations to tensor
            depth_maps = torch.stack([d.depth_map for d in depth_estimations])
            
            # Scale recovery
            logger.info("Recovering metric scale...")
            reconstruction_dict = {
                'points': reconstruction.points,
                'camera_poses': reconstruction.camera_poses,
                'camera_intrinsics': reconstruction.camera_intrinsics,
                'image_names': reconstruction.image_names
            }
            
            scale_result = self.scale_optimizer.recover_scale(
                images_gpu,
                reconstruction_dict,
                depth_maps=depth_maps,
                imu_data=imu_data,
                metadata=metadata
            )
            
            # Apply scale to point cloud
            scaled_points = reconstruction.points * scale_result.scale_factor
            
            # Apply depth-only calibration if configured
            if self.config.scale_recovery.depth_only_calibration != 1.0:
                calibration = self.config.scale_recovery.depth_only_calibration
                scaled_points = scaled_points * calibration
                logger.info(f"Applied depth-only calibration factor: {calibration:.6f}")
            
            # Compute dimensions
            logger.info("Computing final measurements...")
            measurements = self._compute_dimensions(scaled_points)
            
            # Estimate error bounds
            error_bounds = estimate_measurement_errors(
                measurements,
                scale_result.confidence,
                method='detailed'
            )
            logger.info(f"Estimated error: ±{error_bounds['relative_error_percent']:.1f}%")
            
            # Save outputs if configured
            pointcloud_path = None
            if self.config.save_pointcloud:
                pointcloud_path = self.config.output_dir / "pointcloud.ply"
                reconstruction.points = scaled_points  # Update with scaled points
                self.reconstructor.save_reconstruction(
                    reconstruction,
                    pointcloud_path,
                    format='ply'
                )
            
            # Record timing
            end_event.record()
            torch.cuda.synchronize()
            gpu_time = start_event.elapsed_time(end_event) / 1000.0
            total_time = time.time() - total_start
            
            # Create result
            result = MeasurementResult(
                measurements=measurements,
                confidence=scale_result.confidence,
                gpu_time=gpu_time,
                total_time=total_time,
                scale_result=scale_result,
                reconstruction=reconstruction,
                depth_estimations=depth_estimations,
                pointcloud_path=str(pointcloud_path) if pointcloud_path else None,
                error_bounds=error_bounds
            )
            
            logger.info(f"Measurement complete in {total_time:.2f}s")
            logger.info(f"Dimensions: W={measurements['width']:.1f} x "
                       f"H={measurements['height']:.1f} x "
                       f"D={measurements['depth']:.1f} cm")
            
            return result
            
        except Exception as e:
            logger.error(f"Measurement failed: {e}")
            raise RuntimeError(f"Measurement failed: {e}")
        
        finally:
            # Cleanup
            torch.cuda.empty_cache()
            if self.config.enable_profiling:
                self._log_gpu_stats()
    
    def _transfer_to_gpu(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Transfer images to GPU efficiently.
        
        Args:
            images: List of numpy arrays
            
        Returns:
            GPU tensor [N, H, W, 3]
        """
        # Resize images if needed
        processed_images = []
        for img in images:
            if img.shape[0] > self.config.max_image_size or img.shape[1] > self.config.max_image_size:
                import cv2
                scale = self.config.max_image_size / max(img.shape[:2])
                new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
            
            processed_images.append(img)
        
        # Stack and transfer
        images_np = np.stack(processed_images)
        
        # Normalize to [0, 1]
        if images_np.dtype == np.uint8:
            images_np = images_np.astype(np.float32) / 255.0
        
        # Create pinned memory tensor for faster transfer
        images_tensor = torch.from_numpy(images_np).pin_memory()
        
        # Transfer to GPU non-blocking
        images_gpu = images_tensor.to(self.config.gpu.device, non_blocking=True)
        
        return images_gpu
    
    def _compute_dimensions(self, points: torch.Tensor) -> Dict[str, float]:
        """
        Compute bounding box dimensions from point cloud with outlier removal.
        
        Args:
            points: 3D points tensor [N, 3] in meters
            
        Returns:
            Dictionary with measurements in centimeters
        """
        # Convert to numpy for geometry processing
        if isinstance(points, torch.Tensor):
            points_np = points.cpu().numpy()
        else:
            points_np = points
        
        # Remove outliers using both methods
        # NOTE: Keeping STRICT filtering (eps=0.05) because with 707 sparse points,
        # we need aggressive filtering to remove background/noise points.
        # Less strict filtering includes too many outliers, making bbox way too large.
        logger.info(f"Processing {len(points_np)} points...")
        points_clean = remove_outliers(points_np, method='both', eps=0.05, min_samples=10)
        logger.info(f"After outlier removal: {len(points_clean)} points")
        
        # Compute oriented bounding box for better accuracy
        try:
            bbox = compute_oriented_bbox(points_clean)
            
            # Convert from meters to centimeters
            width_cm = bbox.width * 100
            height_cm = bbox.height * 100
            depth_cm = bbox.depth * 100
            volume_cm3 = bbox.volume * 1e6  # m³ to cm³
            
            # Compute center
            center_cm = bbox.center * 100
            
            logger.info(f"Oriented bounding box computed: {width_cm:.2f} x {height_cm:.2f} x {depth_cm:.2f} cm")
            
        except Exception as e:
            logger.warning(f"Failed to compute oriented bbox, using axis-aligned: {e}")
            # Fallback to axis-aligned
            min_coords = points_clean.min(axis=0)
            max_coords = points_clean.max(axis=0)
            dimensions = (max_coords - min_coords) * 100  # Convert to cm
            
            # Sort dimensions
            sorted_dims = sorted(dimensions, reverse=True)
            width_cm, height_cm, depth_cm = sorted_dims
            volume_cm3 = width_cm * height_cm * depth_cm
            center_cm = (min_coords + max_coords) / 2 * 100
        
        # Estimate surface area (approximate as box)
        surface_area_cm2 = 2 * (width_cm*height_cm + height_cm*depth_cm + depth_cm*width_cm)
        
        # Compute point cloud quality
        quality = compute_point_cloud_quality(points_clean)
        
        measurements = {
            'width': float(width_cm),
            'height': float(height_cm),
            'depth': float(depth_cm),
            'volume_cm3': float(volume_cm3),
            'surface_area_cm2': float(surface_area_cm2),
            'center_x': float(center_cm[0]),
            'center_y': float(center_cm[1]),
            'center_z': float(center_cm[2]),
            'num_points': len(points_clean),
            'point_cloud_quality': quality['overall_quality']
        }
        
        return measurements
    
    def _remove_outliers(
        self,
        points: torch.Tensor,
        std_threshold: float = 2.0
    ) -> torch.Tensor:
        """
        Remove outlier points using statistical method.
        
        Args:
            points: Input points [N, 3]
            std_threshold: Number of standard deviations for outlier threshold
            
        Returns:
            Filtered points
        """
        # Compute center and distances
        center = torch.mean(points, dim=0)
        distances = torch.norm(points - center, dim=1)
        
        # Compute threshold
        mean_dist = torch.mean(distances)
        std_dist = torch.std(distances)
        threshold = mean_dist + std_threshold * std_dist
        
        # Filter
        mask = distances < threshold
        points_filtered = points[mask]
        
        num_removed = len(points) - len(points_filtered)
        if num_removed > 0:
            logger.debug(f"Removed {num_removed} outlier points")
        
        return points_filtered
    
    def _log_gpu_stats(self):
        """Log GPU memory and performance statistics."""
        gpu_info = get_gpu_info()
        logger.info("=" * 50)
        logger.info("GPU Statistics:")
        logger.info(f"  Allocated: {gpu_info['allocated_memory_gb']:.2f} GB")
        logger.info(f"  Reserved: {gpu_info['reserved_memory_gb']:.2f} GB")
        logger.info(f"  Total: {gpu_info['total_memory_gb']:.2f} GB")
        logger.info("=" * 50)
    
    def benchmark(
        self,
        num_images: int = 5,
        image_size: Tuple[int, int] = (1024, 1024),
        num_runs: int = 3
    ) -> Dict[str, float]:
        """
        Benchmark system performance.
        
        Args:
            num_images: Number of test images
            image_size: Size of test images (H, W)
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Running benchmark with {num_images} images...")
        
        # Generate random test images
        test_images = [
            np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            for _ in range(num_images)
        ]
        
        times = []
        
        # Warmup
        logger.info("Warming up...")
        try:
            self.measure(test_images)
        except:
            pass  # Warmup may fail, that's okay
        
        # Benchmark runs
        for i in range(num_runs):
            logger.info(f"Benchmark run {i+1}/{num_runs}")
            start = time.time()
            
            try:
                result = self.measure(test_images)
                elapsed = time.time() - start
                times.append(elapsed)
                logger.info(f"Run {i+1} completed in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"Benchmark run {i+1} failed: {e}")
        
        if not times:
            return {'error': 'All benchmark runs failed'}
        
        metrics = {
            'num_images': num_images,
            'image_size': image_size,
            'num_runs': len(times),
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': num_images / np.mean(times)
        }
        
        logger.info(f"Benchmark complete: {metrics['mean_time']:.2f}s ± {metrics['std_time']:.2f}s")
        
        return metrics

