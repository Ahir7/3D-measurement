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
            
            logger.info(f"Marker-based scale: {avg_scale:.4f} mm/px from {len(scales)} markers")
            
            return ScaleEstimate(
                method="marker",
                scale_factor=float(avg_scale),
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
        """Estimate scale from depth maps."""
        try:
            points = reconstruction.get('points')
            if points is None:
                return None
            
            # Calculate median depth from depth maps
            median_depth = torch.median(depth_maps[depth_maps > 0]).item()
            
            # Calculate median distance from reconstruction
            if isinstance(points, torch.Tensor):
                points_np = points.cpu().numpy()
            else:
                points_np = points
            
            distances = np.linalg.norm(points_np, axis=1)
            median_distance = np.median(distances)
            
            if median_distance < 1e-6:
                return None
            
            # Scale = depth_real / depth_reconstruction
            scale = median_depth / median_distance
            
            # Confidence based on depth map quality
            depth_std = torch.std(depth_maps[depth_maps > 0]).item()
            depth_consistency = 1.0 - min(depth_std / median_depth, 1.0)
            confidence = depth_consistency * self.config.depth_weight
            
            logger.info(f"Depth-based scale: {scale:.4f} from median depth {median_depth:.3f}m")
            
            return ScaleEstimate(
                method="depth",
                scale_factor=float(scale),
                confidence=float(confidence),
                metadata={'median_depth': median_depth, 'consistency': depth_consistency}
            )
            
        except Exception as e:
            logger.error(f"Depth-based scale estimation failed: {e}")
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
        
        # Filter by confidence
        valid_estimates = [
            e for e in estimates 
            if e.confidence >= self.config.min_confidence
        ]
        
        if not valid_estimates:
            logger.warning("No estimates meet minimum confidence threshold")
            return 1.0, 0.0, 0
        
        # Simple weighted average
        scales = np.array([e.scale_factor for e in valid_estimates])
        weights = np.array([e.confidence for e in valid_estimates])
        
        # Normalize weights
        weights = weights / weights.sum()
        
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

