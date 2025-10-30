"""
Multi-source scale recovery and optimization.

Combines multiple scale estimation methods for robust metric measurements.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize

from .marker_detection import DetectedMarker, MarkerDetector
from ..core.config import ScaleRecoveryConfig

logger = logging.getLogger(__name__)


@dataclass
class ScaleEstimate:
    """Scale estimation from a single method."""
    
    method: str
    scale_factor: float
    confidence: float
    metadata: Dict = None


@dataclass
class ScaleResult:
    """Final scale recovery result."""
    
    scale_factor: float
    confidence: float
    methods_used: List[str]
    individual_estimates: List[ScaleEstimate]
    optimization_iterations: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'scale_factor': self.scale_factor,
            'confidence': self.confidence,
            'methods_used': self.methods_used,
            'num_estimates': len(self.individual_estimates),
            'optimization_iterations': self.optimization_iterations
        }


class ScaleOptimizer:
    """Multi-source scale recovery optimizer."""
    
    def __init__(self, config: ScaleRecoveryConfig, device: str = 'cuda:0'):
        """
        Initialize scale optimizer.
        
        Args:
            config: Scale recovery configuration
            device: GPU device identifier
        """
        self.config = config
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        
        self.marker_detector = MarkerDetector(device)
        
        logger.info(f"Scale optimizer initialized on {device}")
    
    def recover_scale(
        self,
        images: torch.Tensor,
        reconstruction: Dict,
        depth_maps: Optional[torch.Tensor] = None,
        imu_data: Optional[List[Dict]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> ScaleResult:
        """
        Recover metric scale from multiple sources.
        
        Args:
            images: Input images [N, H, W, 3]
            reconstruction: 3D reconstruction with points and poses
            depth_maps: Optional depth maps from Metric3D
            imu_data: Optional IMU sensor data
            metadata: Optional image metadata
            
        Returns:
            ScaleResult with optimized scale factor
        """
        logger.info("Starting multi-source scale recovery")
        
        estimates = []
        
        # Method 1: Marker-based scale
        if self.config.marker_weight > 0:
            marker_estimate = self._estimate_from_markers(images, reconstruction)
            if marker_estimate:
                estimates.append(marker_estimate)
        
        # Method 2: IMU-based scale
        if self.config.imu_weight > 0 and imu_data:
            imu_estimate = self._estimate_from_imu(imu_data, reconstruction)
            if imu_estimate:
                estimates.append(imu_estimate)
        
        # Method 3: Depth-based scale
        if self.config.depth_weight > 0 and depth_maps is not None:
            aligned_estimate = None
            if reconstruction.get('camera_intrinsics'):
                aligned_estimate = self._estimate_from_depth_aligned(
                    depth_maps,
                    reconstruction
                )
            if aligned_estimate:
                estimates.append(aligned_estimate)
            else:
                depth_estimate = self._estimate_from_depth(depth_maps, reconstruction)
                if depth_estimate:
                    estimates.append(depth_estimate)
        
        # Method 4: Object-based scale
        if self.config.object_weight > 0:
            object_estimate = self._estimate_from_objects(images)
            if object_estimate:
                estimates.append(object_estimate)
        
        # Check if we have enough estimates
        if len(estimates) < self.config.min_methods_required:
            logger.warning(
                f"Insufficient scale estimates: {len(estimates)} < "
                f"{self.config.min_methods_required}, using default scale"
            )
            return ScaleResult(
                scale_factor=1.0,
                confidence=0.0,
                methods_used=[],
                individual_estimates=[]
            )
        
        # Optimize scale
        scale_factor, confidence, iterations = self._optimize_scale(estimates)
        
        logger.info(
            f"Scale recovery complete: scale={scale_factor:.4f}, "
            f"confidence={confidence:.2f}, methods={len(estimates)}"
        )
        
        return ScaleResult(
            scale_factor=scale_factor,
            confidence=confidence,
            methods_used=[e.method for e in estimates],
            individual_estimates=estimates,
            optimization_iterations=iterations
        )
    
    def _estimate_from_markers(
        self,
        images: torch.Tensor,
        reconstruction: Dict
    ) -> Optional[ScaleEstimate]:
        """Estimate scale from detected markers."""
        try:
            # Detect markers in all images
            marker_types = [getattr(__import__('src.scale.marker_detection').scale.marker_detection, 'MarkerType')[mt.upper()] 
                          for mt in self.config.marker_types]
            
            known_sizes = {i: self.config.marker_size_mm for i in range(100)}
            
            all_markers = self.marker_detector.batch_detect(
                list(images),
                marker_types,
                known_sizes
            )
            
            # Collect scale estimates from all detected markers
            scales = []
            confidences = []
            
            for markers in all_markers:
                for marker in markers:
                    scale, conf = self.marker_detector.estimate_scale_from_marker(marker)
                    scales.append(scale)
                    confidences.append(conf)
            
            if not scales:
                logger.warning("No markers detected")
                return None
            
            # Weighted average
            scales = np.array(scales)
            confidences = np.array(confidences)
            
            # Remove outliers (>2 std from median)
            median_scale = np.median(scales)
            std_scale = np.std(scales)
            mask = np.abs(scales - median_scale) < 2 * std_scale
            
            scales = scales[mask]
            confidences = confidences[mask]
            
            if len(scales) == 0:
                return None
            
            avg_scale = np.average(scales, weights=confidences)
            avg_confidence = np.mean(confidences) * self.config.marker_weight
            
            # Convert from mm/px to meters/px for compatibility with measurement system
            avg_scale_meters = avg_scale / 1000.0
            
            logger.info(f"Marker-based scale: {avg_scale:.4f} mm/px ({avg_scale_meters:.6f} m/px) from {len(scales)} markers")
            
            return ScaleEstimate(
                method="marker",
                scale_factor=float(avg_scale_meters),  # Use meters, not millimeters
                confidence=float(avg_confidence),
                metadata={'num_markers': len(scales)}
            )
            
        except Exception as e:
            logger.error(f"Marker-based scale estimation failed: {e}")
            return None
    
    def _estimate_from_imu(
        self,
        imu_data: List[Dict],
        reconstruction: Dict
    ) -> Optional[ScaleEstimate]:
        """Estimate scale from IMU data."""
        try:
            if len(imu_data) < 2:
                return None
            
            # Integrate IMU motion
            camera_poses = reconstruction.get('camera_poses', [])
            if len(camera_poses) < 2:
                return None
            
            # Calculate real-world motion from IMU
            total_motion_real = 0.0
            gravity = np.array(self.config.imu_gravity)
            
            for i in range(1, len(imu_data)):
                dt = (imu_data[i].get('timestamp', 0) - 
                      imu_data[i-1].get('timestamp', 0)) / 1000.0
                
                if dt <= 0:
                    continue
                
                accel = imu_data[i].get('accelerometer', [])
                if accel:
                    accel_data = np.array([accel.get('x', 0), accel.get('y', 0), accel.get('z', 0)])
                    linear_accel = accel_data - gravity
                    motion = np.linalg.norm(linear_accel) * dt * dt * 0.5
                    total_motion_real += motion
            
            # Calculate motion from camera poses
            total_motion_recon = 0.0
            for i in range(1, len(camera_poses)):
                if isinstance(camera_poses[i], torch.Tensor):
                    pose_i = camera_poses[i].cpu().numpy()
                    pose_prev = camera_poses[i-1].cpu().numpy()
                else:
                    pose_i = camera_poses[i]
                    pose_prev = camera_poses[i-1]
                
                motion = np.linalg.norm(pose_i[:3, 3] - pose_prev[:3, 3])
                total_motion_recon += motion
            
            if total_motion_recon < 1e-6:
                return None
            
            # Scale = real_world_motion / reconstruction_motion
            scale = total_motion_real / total_motion_recon
            
            # Confidence based on motion magnitude
            if total_motion_real < 0.05:  # Less than 5cm
                confidence = 0.3
            elif total_motion_real > 5.0:  # More than 5m
                confidence = 0.5
            else:
                confidence = 0.8
            
            confidence *= self.config.imu_weight
            
            logger.info(f"IMU-based scale: {scale:.4f} from {total_motion_real:.3f}m motion")
            
            return ScaleEstimate(
                method="imu",
                scale_factor=float(scale),
                confidence=float(confidence),
                metadata={'motion_real': total_motion_real, 'motion_recon': total_motion_recon}
            )
            
        except Exception as e:
            logger.error(f"IMU-based scale estimation failed: {e}")
            return None
    
    def _estimate_from_depth(
        self,
        depth_maps: torch.Tensor,
        reconstruction: Dict
    ) -> Optional[ScaleEstimate]:
        """
        Estimate scale from depth maps with improved confidence calculation.
        
        Uses multi-factor confidence based on:
        1. Depth consistency (lower variance = better)
        2. Reconstruction quality (more points = better)
        3. Coverage (more images = better)
        """
        try:
            points = reconstruction.get('points')
            if points is None:
                return None
            
            # Calculate depth statistics
            valid_depth = depth_maps[depth_maps > 0]
            median_depth = torch.median(valid_depth).item()
            depth_std = torch.std(valid_depth).item()
            
            # Calculate reconstruction statistics
            if isinstance(points, torch.Tensor):
                points_np = points.cpu().numpy()
            else:
                points_np = points
            
            distances = np.linalg.norm(points_np, axis=1)
            median_distance = np.median(distances)
            
            if median_distance < 1e-6:
                return None
            
            # Scale estimation: depth_real / depth_reconstruction
            scale = median_depth / median_distance
            
            # Multi-factor confidence calculation
            
            # Factor 1: Depth consistency (coefficient of variation)
            depth_cv = depth_std / (median_depth + 1e-6)
            consistency_score = np.exp(-depth_cv)  # Range [0, 1], higher is better
            
            # Factor 2: Reconstruction quality (number of points)
            num_points = len(points_np)
            point_score = min(num_points / 500.0, 1.0)  # Saturates at 500 points
            
            # Factor 3: Depth coverage (number of images)
            num_images = depth_maps.shape[0]
            coverage_score = min(num_images / 15.0, 1.0)  # Saturates at 15 images
            
            # Combined confidence (weighted average)
            confidence_raw = (
                consistency_score * 0.5 +  # Consistency is most important
                point_score * 0.3 +        # Quality matters
                coverage_score * 0.2       # Coverage helps
            )
            
            # Apply depth weight (1.0 in depth-only mode)
            confidence = confidence_raw * self.config.depth_weight
            
            logger.info(
                f"Depth-based scale: {scale:.4f} "
                f"(median_depth={median_depth:.3f}m, "
                f"median_dist={median_distance:.3f}, "
                f"consistency={consistency_score:.2f}, "
                f"points={num_points}, images={num_images}, "
                f"confidence={confidence:.2f})"
            )
            
            return ScaleEstimate(
                method="depth",
                scale_factor=float(scale),
                confidence=float(confidence),
                metadata={
                    'median_depth': median_depth,
                    'depth_std': depth_std,
                    'depth_cv': depth_cv,
                    'consistency_score': consistency_score,
                    'num_points': num_points,
                    'num_images': num_images,
                    'point_score': point_score,
                    'coverage_score': coverage_score
                }
            )
            
        except Exception as e:
            logger.error(f"Depth-based scale estimation failed: {e}")
            return None
    
    def _estimate_from_depth_aligned(
        self,
        depth_maps: torch.Tensor,
        reconstruction: Dict
    ) -> Optional[ScaleEstimate]:
        """Estimate scale by aligning COLMAP points with depth maps."""
        try:
            points = reconstruction.get('points')
            camera_poses = reconstruction.get('camera_poses')
            camera_intrinsics = reconstruction.get('camera_intrinsics')
            image_names = reconstruction.get('image_names')

            if points is None or camera_poses is None or camera_intrinsics is None:
                return None

            if isinstance(points, torch.Tensor):
                points_np = points.detach().cpu().numpy()
            else:
                points_np = points

            if points_np.shape[0] == 0:
                return None

            # Limit number of points for efficiency
            max_points = 8000
            if points_np.shape[0] > max_points:
                rng = np.random.default_rng(seed=42)
                indices = rng.choice(points_np.shape[0], size=max_points, replace=False)
                points_np = points_np[indices]

            per_view_scales = []
            per_view_weights = []

            num_depth_maps = depth_maps.shape[0]

            # Align COLMAP views with input image order based on file names
            ordered_poses: List[Optional[torch.Tensor]] = [None] * num_depth_maps
            ordered_intrinsics: List[Optional[object]] = [None] * num_depth_maps

            if image_names and len(image_names) == len(camera_poses):
                for pose, intrinsics, name in zip(camera_poses, camera_intrinsics, image_names):
                    index = None
                    if isinstance(name, str):
                        base = name.split('.')[0]
                        if '_' in base:
                            suffix = base.split('_')[-1]
                            if suffix.isdigit():
                                index = int(suffix)
                    if index is not None and 0 <= index < num_depth_maps:
                        ordered_poses[index] = pose
                        ordered_intrinsics[index] = intrinsics

            # Fallback: if alignment failed, use sequential order
            for i in range(num_depth_maps):
                if ordered_poses[i] is None and i < len(camera_poses):
                    ordered_poses[i] = camera_poses[i]
                if ordered_intrinsics[i] is None and i < len(camera_intrinsics):
                    ordered_intrinsics[i] = camera_intrinsics[i]

            for view_idx in range(num_depth_maps):
                pose = ordered_poses[view_idx]
                intrinsics = ordered_intrinsics[view_idx]

                if pose is None or intrinsics is None:
                    continue

                if isinstance(pose, torch.Tensor):
                    pose_np = pose.detach().cpu().numpy()
                else:
                    pose_np = pose

                R = pose_np[:3, :3]
                t = pose_np[:3, 3]

                depth_map = depth_maps[view_idx].detach().cpu().numpy()
                H, W = depth_map.shape

                # Project points into current camera
                points_cam = (R @ points_np.T + t.reshape(3, 1)).T

                positive_z = points_cam[:, 2] > 1e-6
                if positive_z.sum() < 100:
                    continue

                points_cam = points_cam[positive_z]

                u = points_cam[:, 0] / points_cam[:, 2]
                v = points_cam[:, 1] / points_cam[:, 2]

                u = u * intrinsics.fx + intrinsics.cx
                v = v * intrinsics.fy + intrinsics.cy

                # Valid pixel locations (allow room for bilinear sampling)
                valid_u = (u >= 1) & (u < W - 2)
                valid_v = (v >= 1) & (v < H - 2)
                valid = valid_u & valid_v

                if valid.sum() < 100:
                    continue

                u = u[valid]
                v = v[valid]
                depths_colmap = points_cam[valid, 2]

                # Bilinear interpolation from depth map
                u0 = np.floor(u).astype(np.int32)
                v0 = np.floor(v).astype(np.int32)
                du = u - u0
                dv = v - v0

                depth_samples = (
                    (1 - du) * (1 - dv) * depth_map[v0, u0] +
                    du * (1 - dv) * depth_map[v0, u0 + 1] +
                    (1 - du) * dv * depth_map[v0 + 1, u0] +
                    du * dv * depth_map[v0 + 1, u0 + 1]
                )

                valid_depths = depth_samples > 0.05
                if valid_depths.sum() < 50:
                    continue

                ratios = depth_samples[valid_depths] / (depths_colmap[valid_depths] + 1e-6)

                if ratios.size < 50:
                    continue

                # Robust outlier removal using MAD
                median_ratio = np.median(ratios)
                mad = np.median(np.abs(ratios - median_ratio)) + 1e-6
                inliers = np.abs(ratios - median_ratio) <= (3.0 * mad)

                if inliers.sum() < 30:
                    continue

                ratios_inliers = ratios[inliers]
                scale_view = np.median(ratios_inliers)
                dispersion = np.std(ratios_inliers)

                inlier_ratio = inliers.sum() / len(ratios)
                consistency = np.exp(-dispersion / (abs(scale_view) + 1e-6))

                per_view_scales.append(scale_view)
                per_view_weights.append(max(inlier_ratio * consistency, 1e-3))

            if not per_view_scales:
                return None

            per_view_scales = np.array(per_view_scales)
            per_view_weights = np.array(per_view_weights)
            per_view_weights = per_view_weights / per_view_weights.sum()

            scale = float(np.sum(per_view_scales * per_view_weights))

            # Confidence combines view consistency and coverage
            if len(per_view_scales) > 1:
                scale_std = np.std(per_view_scales)
                scale_mean = np.mean(per_view_scales)
                view_consistency = np.exp(-scale_std / (abs(scale_mean) + 1e-6))
            else:
                view_consistency = 0.7

            coverage = min(len(per_view_scales) / 10.0, 1.0)
            confidence = (view_consistency * 0.6 + coverage * 0.4) * self.config.depth_weight

            logger.info(
                f"Depth-aligned scale: {scale:.4f} from {len(per_view_scales)} views, "
                f"consistency={view_consistency:.2f}, coverage={coverage:.2f}"
            )

            return ScaleEstimate(
                method="depth_aligned",
                scale_factor=float(scale),
                confidence=float(confidence),
                metadata={
                    'views_used': len(per_view_scales),
                    'weights': per_view_weights.tolist(),
                    'view_scales': per_view_scales.tolist()
                }
            )

        except Exception as e:
            logger.error(f"Depth-aligned scale estimation failed: {e}")
            return None

    def _estimate_from_objects(self, images: torch.Tensor) -> Optional[ScaleEstimate]:
        """Estimate scale from known objects (placeholder)."""
        # This would use object detection to find known objects
        # and estimate scale based on their known sizes
        # For now, return None
        logger.debug("Object-based scale estimation not yet implemented")
        return None
    
    def _optimize_scale(
        self,
        estimates: List[ScaleEstimate]
    ) -> Tuple[float, float, int]:
        """
        Optimize scale factor from multiple estimates.
        
        Args:
            estimates: List of scale estimates
            
        Returns:
            Tuple of (optimized_scale, confidence, iterations)
        """
        if not estimates:
            return 1.0, 0.0, 0
        
        # Filter by confidence (allow all if threshold <= 0)
        if self.config.min_confidence > 0.0:
            valid_estimates = [e for e in estimates if e.confidence >= self.config.min_confidence]
        else:
            valid_estimates = estimates[:]
        
        if not valid_estimates:
            logger.warning("No scale estimates available")
            return 1.0, 0.0, 0
        
        # Simple weighted average
        scales = np.array([e.scale_factor for e in valid_estimates])
        weights = np.array([e.confidence for e in valid_estimates])
        
        # Normalize weights (fallback to uniform if all zeros)
        wsum = weights.sum()
        if wsum <= 1e-8:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / wsum
        
        # Weighted average
        optimized_scale = np.average(scales, weights=weights)
        
        # Calculate confidence as weighted average of individual confidences
        confidence = float(np.sum(weights * np.array([e.confidence for e in valid_estimates])))
        
        # Penalize if we have disagreement between methods
        scale_std = np.std(scales)
        scale_mean = np.mean(scales)
        if scale_mean > 0:
            disagreement_penalty = min(scale_std / scale_mean, 0.5)
            confidence *= (1.0 - disagreement_penalty)
        
        logger.debug(
            f"Optimized scale: {optimized_scale:.4f}, "
            f"confidence: {confidence:.2f}, "
            f"from {len(valid_estimates)} estimates"
        )
        
        return float(optimized_scale), float(confidence), 0
    
    def refine_scale_iterative(
        self,
        initial_scale: float,
        estimates: List[ScaleEstimate],
        max_iterations: int = 10,
        tolerance: float = 1e-4
    ) -> Tuple[float, int]:
        """
        Refine scale using iterative optimization.
        
        Args:
            initial_scale: Initial scale guess
            estimates: List of scale estimates
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (refined_scale, num_iterations)
        """
        def objective(scale):
            """Objective function to minimize."""
            error = 0.0
            for estimate in estimates:
                diff = scale - estimate.scale_factor
                error += estimate.confidence * diff ** 2
            return error
        
        # Optimize
        result = minimize(
            objective,
            x0=[initial_scale],
            method='BFGS',
            options={'maxiter': max_iterations, 'gtol': tolerance}
        )
        
        return float(result.x[0]), int(result.nit)

