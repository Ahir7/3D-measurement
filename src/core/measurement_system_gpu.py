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
from ..utils.capture_quality import analyze_capture_quality

logger = logging.getLogger(__name__)


@dataclass
class MeasurementResult:
    """Complete measurement result with GPU metrics and uncertainty."""

    measurements: Dict[str, float]
    confidence: float
    gpu_time: float
    total_time: float
    scale_result: ScaleResult
    reconstruction: Reconstruction3D
    depth_estimations: Optional[List[DepthEstimation]] = None
    pointcloud_path: Optional[str] = None
    error_bounds: Optional[Dict[str, float]] = None

    # New fields for accuracy enhancement
    uncertainty_bounds: Optional[Dict[str, float]] = None
    geometric_fit_score: Optional[float] = None
    plane_detections: Optional[List[Dict]] = None
    model_used: Optional[str] = None

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

        # Add new accuracy enhancement fields
        if self.uncertainty_bounds:
            result['uncertainty_bounds'] = self.uncertainty_bounds

        if self.geometric_fit_score is not None:
            result['geometric_fit_score'] = self.geometric_fit_score

        if self.plane_detections:
            result['plane_detections'] = self.plane_detections

        if self.model_used:
            result['model_used'] = self.model_used

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

        # Cache cleanup policy
        self._measure_calls = 0
        self._cache_cleanup_interval = 5
        
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
        self._measure_calls += 1

        # Accuracy stage: prefilter low-quality captures when paths are available
        quality_info = None
        images, image_paths, quality_info = self._prefilter_low_quality_images(images, image_paths)
        
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
            confidence_maps = torch.stack([
                d.confidence_map if d.confidence_map is not None else torch.ones_like(d.depth_map)
                for d in depth_estimations
            ])
            
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
                confidence_maps=confidence_maps,
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

            if quality_info is not None:
                error_bounds['capture_quality_score'] = quality_info.get('quality_score_before')
                error_bounds['capture_quality_score_after'] = quality_info.get('quality_score_after')
                error_bounds['capture_images_removed'] = quality_info.get('removed_images', 0)
                error_bounds['capture_images_used'] = quality_info.get('used_images', len(images))

            # Add confidence gating and diagnostics for depth-only mode
            low_conf_threshold = self.config.scale_recovery.low_confidence_threshold
            if scale_result.confidence < low_conf_threshold:
                logger.warning(
                    f"Low scale confidence ({scale_result.confidence:.2f} < {low_conf_threshold:.2f}); "
                    "measurement reliability is limited"
                )
                error_bounds['low_confidence_warning'] = True
                error_bounds['relative_error_percent'] = max(
                    float(error_bounds.get('relative_error_percent', 0.0)),
                    25.0
                )
                error_bounds['quality'] = 'poor'

            aligned_estimate = next(
                (e for e in scale_result.individual_estimates if e.method == 'depth_aligned'),
                None
            )
            if aligned_estimate and aligned_estimate.metadata:
                error_bounds['depth_aligned_views_used'] = aligned_estimate.metadata.get('views_used')
                error_bounds['depth_aligned_scale_dispersion'] = aligned_estimate.metadata.get('scale_dispersion')

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
            # Periodic cleanup to avoid allocator thrash
            if self._measure_calls % self._cache_cleanup_interval == 0:
                torch.cuda.empty_cache()
            if self.config.enable_profiling:
                self._log_gpu_stats()

    def _prefilter_low_quality_images(
        self,
        images: List[np.ndarray],
        image_paths: Optional[List[Path]]
    ) -> Tuple[List[np.ndarray], Optional[List[Path]], Optional[Dict[str, float]]]:
        """Drop lowest-quality images for more stable reconstruction when quality is poor."""
        if not self.config.enable_capture_quality_filter:
            return images, image_paths, None
        if not image_paths or len(images) != len(image_paths):
            return images, image_paths, None
        if len(images) <= self.config.min_images_after_quality_filter:
            return images, image_paths, None

        try:
            paths = [Path(path) for path in image_paths]
            quality_report = analyze_capture_quality(paths)
            summary = quality_report['summary']
            quality_before = float(summary['quality_score'])

            info = {
                'quality_score_before': quality_before,
                'quality_score_after': quality_before,
                'removed_images': 0,
                'used_images': len(images)
            }

            if quality_before >= self.config.capture_quality_threshold:
                logger.info(
                    f"Capture quality acceptable ({quality_before:.3f}); no prefiltering applied"
                )
                return images, image_paths, info

            max_removable = len(images) - self.config.min_images_after_quality_filter
            drop_fraction = self._adaptive_quality_drop_fraction(summary)
            desired_remove = int(len(images) * drop_fraction)
            remove_count = min(max(desired_remove, 1), max_removable)

            if remove_count <= 0:
                logger.info("Capture quality low but prefiltering skipped due to min image constraints")
                return images, image_paths, info

            scored = []
            for idx, item in enumerate(quality_report['images']):
                blur = float(item['blur_score'])
                exposure_mean = float(item['exposure_mean'])
                under = float(item['underexposed_ratio'])
                over = float(item['overexposed_ratio'])

                blur_norm = float(np.clip(np.log1p(blur) / np.log1p(350.0), 0.0, 1.0))
                exposure_balance = float(np.clip(1.0 - abs(exposure_mean - 0.5) / 0.5, 0.0, 1.0))
                clipping_penalty = float(np.clip(1.0 - (under + over), 0.0, 1.0))
                score = 0.50 * blur_norm + 0.25 * exposure_balance + 0.25 * clipping_penalty
                scored.append((idx, score))

            scored.sort(key=lambda pair: pair[1], reverse=True)
            keep_indices = sorted(idx for idx, _ in scored[:len(images) - remove_count])

            filtered_images = [images[idx] for idx in keep_indices]
            filtered_paths = [image_paths[idx] for idx in keep_indices]

            filtered_report = analyze_capture_quality([Path(path) for path in filtered_paths])
            quality_after = float(filtered_report['summary']['quality_score'])

            info.update({
                'quality_score_after': quality_after,
                'removed_images': remove_count,
                'used_images': len(filtered_images),
                'quality_drop_fraction_applied': drop_fraction
            })

            logger.warning(
                f"Low capture quality ({quality_before:.3f} < {self.config.capture_quality_threshold:.3f}); "
                f"drop_fraction={drop_fraction:.3f}, removed {remove_count} low-quality images, "
                f"quality {quality_before:.3f} -> {quality_after:.3f}"
            )

            return filtered_images, filtered_paths, info

        except Exception as error:
            logger.warning(f"Capture quality prefilter failed; using original images: {error}")
            return images, image_paths, None

    def _adaptive_quality_drop_fraction(self, quality_summary: Dict[str, float]) -> float:
        """Adapt pruning strength using quality gap and overlap stability."""
        base_fraction = float(self.config.quality_drop_fraction)
        if not self.config.enable_adaptive_quality_drop:
            return base_fraction

        quality_score = float(quality_summary.get('quality_score', 0.0))
        overlap_median = float(quality_summary.get('overlap_median', 0.0))
        overlap_std = float(quality_summary.get('overlap_std', 0.0))

        threshold = float(self.config.capture_quality_threshold)
        quality_gap = np.clip((threshold - quality_score) / max(threshold, 1e-6), 0.0, 1.0)

        target_overlap = 0.15
        overlap_penalty = np.clip((target_overlap - overlap_median) / target_overlap, 0.0, 1.0)
        overlap_dispersion = overlap_std / max(overlap_median, 1e-6)
        dispersion_penalty = np.clip(overlap_dispersion / 0.8, 0.0, 1.0)

        adaptive_multiplier = 1.0 + 0.6 * quality_gap + 0.4 * overlap_penalty + 0.3 * dispersion_penalty
        adaptive_fraction = base_fraction * adaptive_multiplier
        adaptive_fraction = float(np.clip(
            adaptive_fraction,
            self.config.adaptive_quality_drop_min,
            self.config.adaptive_quality_drop_max,
        ))

        return adaptive_fraction
    
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
        
        # Remove outliers with adaptive DBSCAN parameters
        logger.info(f"Processing {len(points_np)} points...")
        points_clean = self._adaptive_outlier_filter(points_np)

        # Fallback when filtering is too aggressive
        minimum_points = max(30, int(0.03 * len(points_np)))
        if len(points_clean) < minimum_points:
            logger.warning(
                f"Adaptive filtering kept too few points ({len(points_clean)}<{minimum_points}); "
                "falling back to statistical filtering"
            )
            points_clean = remove_outliers(points_np, method='statistical', std_ratio=2.5)

        if len(points_clean) < 10:
            logger.warning("Too few points after filtering; reverting to unfiltered points")
            points_clean = points_np

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
            'num_points_raw': len(points_np),
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

    def _adaptive_outlier_filter(self, points_np: np.ndarray) -> np.ndarray:
        """Adaptive two-stage outlier filtering based on point spacing."""
        if len(points_np) < 40:
            return points_np

        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(points_np)
            distances, _ = tree.query(points_np, k=2)
            nn_distances = distances[:, 1]

            base_spacing = float(np.median(nn_distances))
            eps = float(np.clip(base_spacing * 3.0, 0.01, 0.20))
            min_samples = int(np.clip(len(points_np) * 0.01, 8, 20))

            logger.info(
                f"Adaptive outlier filter: spacing={base_spacing:.4f}, eps={eps:.4f}, "
                f"min_samples={min_samples}"
            )
            return remove_outliers(points_np, method='both', eps=eps, min_samples=min_samples)

        except Exception as error:
            logger.warning(f"Adaptive outlier filtering failed, using default: {error}")
            return remove_outliers(points_np, method='both', eps=0.05, min_samples=10)
    
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
        except Exception as error:
            logger.warning(f"Warmup failed (continuing benchmark): {error}")
        
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

    def _apply_geometric_validation(
        self,
        points: np.ndarray,
        depth_maps: torch.Tensor,
        camera_poses: List,
        camera_intrinsics: List,
        bbox
    ) -> Tuple[Optional[object], Dict]:
        """
        Apply geometric validation to refine measurements.

        Uses plane detection, prism fitting, and epipolar consistency
        to validate and potentially refine the bounding box.

        Args:
            points: Point cloud [N, 3]
            depth_maps: Depth maps [V, H, W]
            camera_poses: List of camera pose matrices
            camera_intrinsics: List of camera intrinsics
            bbox: Initial bounding box

        Returns:
            Tuple of (refined_bbox or None, diagnostics)
        """
        diagnostics = {}

        if not self.config.geometric_priors.enable_geometric_refinement:
            return None, diagnostics

        try:
            from ..geometry.geometric_validator import GeometricValidator

            validator = GeometricValidator(self.config.geometric_priors)

            result = validator.validate_and_refine(
                points,
                depth_maps=depth_maps,
                camera_poses=camera_poses,
                camera_intrinsics=camera_intrinsics,
                initial_bbox=bbox
            )

            diagnostics['geometric_validation'] = result.diagnostics
            diagnostics['geometric_confidence'] = result.confidence_score
            diagnostics['is_valid_geometry'] = result.is_valid

            if result.prism_fit:
                diagnostics['prism_dimensions'] = result.prism_fit.dimensions.tolist()
                diagnostics['prism_residual'] = result.prism_fit.residual

            if result.plane_detections:
                diagnostics['num_planes'] = len(result.plane_detections)

            logger.info(
                f"Geometric validation: valid={result.is_valid}, "
                f"confidence={result.confidence_score:.2f}"
            )

            return result.refined_bbox, diagnostics

        except ImportError as e:
            logger.debug(f"Geometric validation not available: {e}")
            return None, {'error': 'not_available'}
        except Exception as e:
            logger.warning(f"Geometric validation failed: {e}")
            return None, {'error': str(e)}

    def _estimate_uncertainty_bounds(
        self,
        depth_estimations: List[DepthEstimation],
        measurements: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Estimate uncertainty bounds from depth estimation uncertainty.

        Args:
            depth_estimations: List of depth estimations with uncertainty
            measurements: Current measurements

        Returns:
            Dictionary with uncertainty bounds for each measurement
        """
        bounds = {
            'width_uncertainty': 0.0,
            'height_uncertainty': 0.0,
            'depth_uncertainty': 0.0,
            'volume_uncertainty': 0.0
        }

        if not depth_estimations:
            return bounds

        # Collect uncertainty statistics
        uncertainties = []
        for est in depth_estimations:
            if est.uncertainty_map is not None:
                mean_unc = float(est.uncertainty_map.mean())
                uncertainties.append(mean_unc)
            elif est.mc_variance is not None:
                mean_var = float(est.mc_variance.mean())
                uncertainties.append(np.sqrt(mean_var))

        if not uncertainties:
            return bounds

        # Average uncertainty across views
        avg_uncertainty = np.mean(uncertainties)

        # Scale uncertainty to measurement units
        # Uncertainty in depth translates to uncertainty in measurements
        scale_factor = avg_uncertainty * 100  # Convert to cm

        bounds['width_uncertainty'] = measurements.get('width', 0) * avg_uncertainty
        bounds['height_uncertainty'] = measurements.get('height', 0) * avg_uncertainty * 1.1
        bounds['depth_uncertainty'] = measurements.get('depth', 0) * avg_uncertainty * 1.2

        # Volume uncertainty (propagate through product)
        if 'volume_cm3' in measurements:
            vol = measurements['volume_cm3']
            rel_unc = 3 * avg_uncertainty  # Approximate for product
            bounds['volume_uncertainty'] = vol * rel_unc

        bounds['mean_depth_uncertainty'] = float(avg_uncertainty)

        return bounds

    def measure_with_uncertainty(
        self,
        images: List[np.ndarray],
        image_paths: Optional[List[Path]] = None,
        imu_data: Optional[List[Dict]] = None,
        metadata: Optional[List[Dict]] = None,
        known_intrinsics: Optional[CameraIntrinsics] = None
    ) -> MeasurementResult:
        """
        Measure dimensions with full uncertainty quantification.

        Extended version of measure() that computes uncertainty bounds
        and applies geometric validation.

        Args:
            images: List of input images
            image_paths: Optional paths to image files
            imu_data: Optional IMU sensor data
            metadata: Optional image metadata
            known_intrinsics: Optional known camera calibration

        Returns:
            MeasurementResult with uncertainty bounds
        """
        # Run base measurement
        result = self.measure(
            images,
            image_paths=image_paths,
            imu_data=imu_data,
            metadata=metadata,
            known_intrinsics=known_intrinsics
        )

        # Add uncertainty bounds from depth estimation
        if result.depth_estimations:
            uncertainty_bounds = self._estimate_uncertainty_bounds(
                result.depth_estimations,
                result.measurements
            )
            result.uncertainty_bounds = uncertainty_bounds

            # Get model name from first estimation
            if result.depth_estimations[0].model_name:
                result.model_used = result.depth_estimations[0].model_name

        # Apply geometric validation if configured
        if (self.config.geometric_priors.enable_geometric_refinement and
            result.reconstruction.points is not None):

            points_np = result.reconstruction.points
            if isinstance(points_np, torch.Tensor):
                points_np = points_np.cpu().numpy()

            depth_maps = None
            if result.depth_estimations:
                depth_maps = torch.stack([d.depth_map for d in result.depth_estimations])

            refined_bbox, geo_diag = self._apply_geometric_validation(
                points_np,
                depth_maps,
                result.reconstruction.camera_poses or [],
                result.reconstruction.camera_intrinsics or [],
                None
            )

            if geo_diag:
                if result.error_bounds is None:
                    result.error_bounds = {}
                result.error_bounds.update(geo_diag)

            if geo_diag.get('geometric_confidence'):
                result.geometric_fit_score = geo_diag['geometric_confidence']

        return result

