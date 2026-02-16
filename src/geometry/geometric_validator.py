"""
Geometric validation for 3D measurements.

Combines plane detection, prism fitting, and epipolar consistency
to validate and refine 3D measurements.
"""

import torch
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core.config import GeometricPriorsConfig
from ..utils.geometry import BoundingBox
from .plane_detection import (
    PlaneEstimate,
    MultiPlaneRANSAC,
    validate_box_topology
)
from .prism_fitting import (
    PrismFit,
    RectangularPrismFitter,
    BoxConstraints
)
from .epipolar_consistency import (
    EpipolarConsistencyChecker,
    EpipolarResult,
    CameraIntrinsics
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of geometric validation.

    Attributes:
        is_valid: Whether geometry passes validation
        confidence_score: Overall validation confidence [0, 1]
        refined_bbox: Refined bounding box (if refinement enabled)
        plane_detections: List of detected planes
        prism_fit: Fitted rectangular prism
        epipolar_result: Epipolar consistency result
        diagnostics: Detailed diagnostic information
    """
    is_valid: bool
    confidence_score: float
    refined_bbox: Optional[BoundingBox] = None
    plane_detections: Optional[List[PlaneEstimate]] = None
    prism_fit: Optional[PrismFit] = None
    epipolar_result: Optional[EpipolarResult] = None
    diagnostics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            'is_valid': self.is_valid,
            'confidence_score': self.confidence_score,
            'diagnostics': self.diagnostics
        }

        if self.refined_bbox:
            result['refined_dimensions'] = {
                'width': self.refined_bbox.width,
                'height': self.refined_bbox.height,
                'depth': self.refined_bbox.depth
            }

        if self.plane_detections:
            result['num_planes_detected'] = len(self.plane_detections)

        if self.prism_fit:
            result['prism_fit'] = {
                'dimensions': self.prism_fit.dimensions.tolist(),
                'residual': self.prism_fit.residual,
                'inlier_ratio': self.prism_fit.inlier_ratio
            }

        if self.epipolar_result:
            result['epipolar'] = {
                'mean_error': self.epipolar_result.mean_reprojection_error,
                'inlier_ratio': self.epipolar_result.inlier_ratio
            }

        return result


class GeometricValidator:
    """
    Validates and refines 3D measurements using geometric priors.

    Integrates multiple validation methods:
    1. Plane detection for box-like structures
    2. Rectangular prism fitting
    3. Epipolar consistency across views
    4. Box topology validation
    """

    def __init__(self, config: GeometricPriorsConfig):
        """
        Initialize geometric validator.

        Args:
            config: Geometric priors configuration
        """
        self.config = config

        # Initialize sub-components
        if config.enable_plane_detection:
            self.plane_detector = MultiPlaneRANSAC(
                n_iterations=config.ransac_iterations,
                distance_threshold=config.ransac_threshold,
                min_inliers=config.min_plane_points,
                max_planes=config.max_planes
            )
        else:
            self.plane_detector = None

        if config.enable_prism_fitting:
            self.prism_fitter = RectangularPrismFitter(
                max_iterations=config.prism_fitting_iterations,
                inlier_threshold=config.prism_inlier_threshold
            )
        else:
            self.prism_fitter = None

        if config.enable_epipolar_check:
            self.epipolar_checker = EpipolarConsistencyChecker(
                reprojection_threshold=config.epipolar_threshold,
                min_valid_views=config.min_epipolar_inliers // 10  # Scale down
            )
        else:
            self.epipolar_checker = None

        if config.enable_box_topology:
            self.box_constraints = BoxConstraints(
                orthogonality_tolerance=config.orthogonality_tolerance_degrees,
                parallelism_tolerance=config.parallelism_tolerance_degrees
            )
        else:
            self.box_constraints = None

        logger.info(
            f"GeometricValidator initialized: "
            f"planes={config.enable_plane_detection}, "
            f"prism={config.enable_prism_fitting}, "
            f"epipolar={config.enable_epipolar_check}"
        )

    def validate_and_refine(
        self,
        points: np.ndarray,
        depth_maps: Optional[torch.Tensor] = None,
        camera_poses: Optional[List[torch.Tensor]] = None,
        camera_intrinsics: Optional[List] = None,
        initial_bbox: Optional[BoundingBox] = None
    ) -> ValidationResult:
        """
        Validate and optionally refine 3D measurements.

        Args:
            points: Point cloud [N, 3]
            depth_maps: Optional depth maps [V, H, W]
            camera_poses: Optional list of camera poses
            camera_intrinsics: Optional list of camera intrinsics
            initial_bbox: Optional initial bounding box

        Returns:
            ValidationResult with validation status and refinements
        """
        diagnostics = {
            'num_points': len(points),
            'stages_completed': []
        }

        planes = None
        prism = None
        epipolar_result = None
        confidence_scores = []

        # Stage 1: Plane Detection
        if self.plane_detector is not None:
            try:
                planes = self.plane_detector.detect_all(points)
                diagnostics['num_planes'] = len(planes)
                diagnostics['stages_completed'].append('plane_detection')

                if planes:
                    # Validate box topology
                    is_box, topology_score, topo_diag = validate_box_topology(
                        planes,
                        self.config.orthogonality_tolerance_degrees,
                        self.config.parallelism_tolerance_degrees
                    )
                    diagnostics['box_topology'] = topo_diag
                    diagnostics['is_box_topology'] = is_box
                    confidence_scores.append(topology_score)

                    # Enforce constraints if enabled
                    if self.box_constraints is not None and is_box:
                        planes = self.box_constraints.enforce_orthogonality(planes)
                        planes = self.box_constraints.enforce_parallelism(planes)
                        diagnostics['constraints_enforced'] = True

                logger.debug(f"Plane detection: {len(planes)} planes found")

            except Exception as e:
                logger.warning(f"Plane detection failed: {e}")
                diagnostics['plane_detection_error'] = str(e)

        # Stage 2: Prism Fitting
        if self.prism_fitter is not None:
            try:
                prism = self.prism_fitter.fit(points, initial_planes=planes)
                diagnostics['prism_residual'] = prism.residual
                diagnostics['prism_inlier_ratio'] = prism.inlier_ratio
                diagnostics['stages_completed'].append('prism_fitting')

                # Prism fit quality contributes to confidence
                prism_confidence = prism.inlier_ratio * np.exp(-prism.residual * 10)
                confidence_scores.append(prism_confidence)

                logger.debug(
                    f"Prism fit: dims={prism.dimensions}, "
                    f"residual={prism.residual:.4f}"
                )

            except Exception as e:
                logger.warning(f"Prism fitting failed: {e}")
                diagnostics['prism_fitting_error'] = str(e)

        # Stage 3: Epipolar Consistency
        if (self.epipolar_checker is not None and
            depth_maps is not None and
            camera_poses is not None and
            camera_intrinsics is not None):

            try:
                # Convert intrinsics if needed
                intrinsics_list = []
                for intr in camera_intrinsics:
                    if isinstance(intr, CameraIntrinsics):
                        intrinsics_list.append(intr)
                    else:
                        # Assume it has fx, fy, cx, cy attributes
                        intrinsics_list.append(CameraIntrinsics(
                            fx=intr.fx, fy=intr.fy,
                            cx=intr.cx, cy=intr.cy
                        ))

                epipolar_result = self.epipolar_checker.check(
                    depth_maps, camera_poses, intrinsics_list
                )
                diagnostics['epipolar_error'] = epipolar_result.mean_reprojection_error
                diagnostics['epipolar_inlier_ratio'] = epipolar_result.inlier_ratio
                diagnostics['stages_completed'].append('epipolar_check')

                confidence_scores.append(epipolar_result.inlier_ratio)

                logger.debug(
                    f"Epipolar check: error={epipolar_result.mean_reprojection_error:.4f}, "
                    f"inliers={epipolar_result.inlier_ratio:.2%}"
                )

            except Exception as e:
                logger.warning(f"Epipolar check failed: {e}")
                diagnostics['epipolar_error_msg'] = str(e)

        # Compute overall confidence
        if confidence_scores:
            overall_confidence = float(np.mean(confidence_scores))
        else:
            overall_confidence = 0.5  # Unknown

        diagnostics['confidence_components'] = confidence_scores

        # Determine validity
        is_valid = overall_confidence > 0.5

        # Refine bounding box if prism fit available
        refined_bbox = None
        if prism is not None and self.config.enable_geometric_refinement:
            refined_bbox = self._prism_to_bbox(prism)

        logger.info(
            f"Geometric validation: valid={is_valid}, "
            f"confidence={overall_confidence:.2f}, "
            f"stages={diagnostics['stages_completed']}"
        )

        return ValidationResult(
            is_valid=is_valid,
            confidence_score=overall_confidence,
            refined_bbox=refined_bbox,
            plane_detections=planes,
            prism_fit=prism,
            epipolar_result=epipolar_result,
            diagnostics=diagnostics
        )

    def _prism_to_bbox(self, prism: PrismFit) -> BoundingBox:
        """Convert PrismFit to BoundingBox."""
        # Sort dimensions for consistency
        sorted_dims = np.sort(prism.dimensions)[::-1]

        return BoundingBox(
            width=float(sorted_dims[0]),
            height=float(sorted_dims[1]),
            depth=float(sorted_dims[2]),
            volume=float(np.prod(prism.dimensions)),
            center=prism.center,
            orientation=prism.rotation,
            corners=prism.corners
        )

    def get_dimension_corrections(
        self,
        initial_measurements: Dict[str, float],
        validation_result: ValidationResult
    ) -> Dict[str, float]:
        """
        Compute corrections to apply to initial measurements.

        Args:
            initial_measurements: Initial width, height, depth measurements
            validation_result: Result from validate_and_refine

        Returns:
            Dictionary with correction factors for each dimension
        """
        corrections = {
            'width_factor': 1.0,
            'height_factor': 1.0,
            'depth_factor': 1.0
        }

        if validation_result.refined_bbox is None:
            return corrections

        # Compare refined to initial
        refined = validation_result.refined_bbox
        initial_w = initial_measurements.get('width', refined.width * 100)
        initial_h = initial_measurements.get('height', refined.height * 100)
        initial_d = initial_measurements.get('depth', refined.depth * 100)

        # Refined dimensions are in meters, convert to cm
        refined_w = refined.width * 100
        refined_h = refined.height * 100
        refined_d = refined.depth * 100

        # Compute correction factors
        if initial_w > 0:
            corrections['width_factor'] = refined_w / initial_w
        if initial_h > 0:
            corrections['height_factor'] = refined_h / initial_h
        if initial_d > 0:
            corrections['depth_factor'] = refined_d / initial_d

        # Limit correction magnitude
        for key in corrections:
            corrections[key] = float(np.clip(corrections[key], 0.8, 1.2))

        return corrections

    def estimate_geometric_uncertainty(
        self,
        validation_result: ValidationResult
    ) -> Dict[str, float]:
        """
        Estimate uncertainty based on geometric validation.

        Args:
            validation_result: Result from validation

        Returns:
            Dictionary with uncertainty estimates
        """
        uncertainty = {
            'overall': 0.1,  # Base 10% uncertainty
            'width': 0.05,
            'height': 0.05,
            'depth': 0.10  # Depth typically more uncertain
        }

        # Adjust based on validation confidence
        conf = validation_result.confidence_score
        scale = 1.0 / (conf + 0.5)  # Lower confidence = higher uncertainty

        for key in uncertainty:
            uncertainty[key] *= scale

        # Specific adjustments from diagnostics
        diag = validation_result.diagnostics

        # Prism fit quality
        if 'prism_residual' in diag:
            residual_factor = 1.0 + diag['prism_residual'] * 10
            uncertainty['overall'] *= residual_factor

        # Box topology
        if diag.get('is_box_topology', False):
            # Good topology reduces uncertainty
            for key in ['width', 'height', 'depth']:
                uncertainty[key] *= 0.8

        # Epipolar consistency
        if 'epipolar_inlier_ratio' in diag:
            inlier_ratio = diag['epipolar_inlier_ratio']
            if inlier_ratio > 0.8:
                uncertainty['overall'] *= 0.9
            elif inlier_ratio < 0.5:
                uncertainty['overall'] *= 1.5

        return uncertainty
