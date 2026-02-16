"""
Geometry modules for 3D measurement.

Includes plane detection, prism fitting, epipolar consistency checking,
and geometric validation for accuracy enhancement.
"""

from .plane_detection import (
    PlaneEstimate,
    RANSACPlaneDetector,
    MultiPlaneRANSAC,
    compute_plane_orthogonality,
    compute_plane_parallelism,
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
    compute_reprojection_error,
    compute_essential_matrix,
    compute_fundamental_matrix
)
from .geometric_validator import GeometricValidator, ValidationResult

__all__ = [
    # Plane detection
    'PlaneEstimate',
    'RANSACPlaneDetector',
    'MultiPlaneRANSAC',
    'compute_plane_orthogonality',
    'compute_plane_parallelism',
    'validate_box_topology',
    # Prism fitting
    'PrismFit',
    'RectangularPrismFitter',
    'BoxConstraints',
    # Epipolar consistency
    'EpipolarConsistencyChecker',
    'EpipolarResult',
    'compute_reprojection_error',
    'compute_essential_matrix',
    'compute_fundamental_matrix',
    # Geometric validator
    'GeometricValidator',
    'ValidationResult',
]
