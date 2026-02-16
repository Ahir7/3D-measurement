"""Utility modules for 3D measurement system."""

from .geometry import (
    BoundingBox,
    remove_outliers,
    remove_outliers_statistical,
    remove_outliers_dbscan,
    compute_oriented_bbox,
    compute_axis_aligned_bbox,
    estimate_measurement_errors,
    format_measurement_with_error,
    compute_point_cloud_quality,
    fit_rectangular_prism,
    detect_planes_ransac,
    validate_box_topology,
    compute_geometric_confidence,
    refine_bbox_with_prism
)

__all__ = [
    'BoundingBox',
    'remove_outliers',
    'remove_outliers_statistical',
    'remove_outliers_dbscan',
    'compute_oriented_bbox',
    'compute_axis_aligned_bbox',
    'estimate_measurement_errors',
    'format_measurement_with_error',
    'compute_point_cloud_quality',
    'fit_rectangular_prism',
    'detect_planes_ransac',
    'validate_box_topology',
    'compute_geometric_confidence',
    'refine_bbox_with_prism',
]
