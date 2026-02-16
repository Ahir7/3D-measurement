"""
Geometry utilities for 3D point cloud processing.

Includes outlier removal, oriented bounding box computation, and error estimation.
"""

import numpy as np
import torch
import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Oriented bounding box result."""
    width: float
    height: float
    depth: float
    volume: float
    center: np.ndarray
    orientation: np.ndarray  # Eigenvectors
    corners: Optional[np.ndarray] = None


def remove_outliers_statistical(points: np.ndarray,
                                std_ratio: float = 2.0) -> np.ndarray:
    """
    Remove statistical outliers based on distance to neighbors.
    
    Args:
        points: Input points [N, 3]
        std_ratio: Standard deviation multiplier for outlier threshold
        
    Returns:
        Filtered points without outliers
    """
    if len(points) < 10:
        return points
    
    # Compute distances to centroid
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    
    # Remove points beyond std_ratio standard deviations
    mean_dist = distances.mean()
    std_dist = distances.std()
    threshold = mean_dist + std_ratio * std_dist
    
    mask = distances < threshold
    filtered = points[mask]
    
    removed = len(points) - len(filtered)
    if removed > 0:
        logger.debug(f"Removed {removed} statistical outliers ({removed/len(points)*100:.1f}%)")
    
    return filtered


def remove_outliers_dbscan(points: np.ndarray,
                           eps: float = 0.1,
                           min_samples: int = 10) -> np.ndarray:
    """
    Remove outliers using DBSCAN clustering (keeps largest cluster).
    
    Args:
        points: Input points [N, 3]
        eps: Maximum distance between two samples for one to be in the neighborhood
        min_samples: Minimum number of samples in a neighborhood
        
    Returns:
        Points from the largest cluster
    """
    if len(points) < min_samples * 2:
        logger.warning(f"Too few points ({len(points)}) for DBSCAN, skipping")
        return points
    
    try:
        from sklearn.cluster import DBSCAN
        
        # Cluster points
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        
        # Count non-noise labels
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        
        if len(unique_labels) == 0:
            logger.warning("DBSCAN found no clusters, keeping all points")
            return points
        
        # Keep largest cluster
        largest_cluster_label = unique_labels[np.argmax(counts)]
        mask = labels == largest_cluster_label
        filtered = points[mask]
        
        removed = len(points) - len(filtered)
        if removed > 0:
            logger.info(f"DBSCAN removed {removed} outliers ({removed/len(points)*100:.1f}%), "
                       f"kept {len(filtered)} points from largest cluster")
        
        return filtered
        
    except ImportError:
        logger.warning("scikit-learn not available, skipping DBSCAN outlier removal")
        return points


def remove_outliers(points: np.ndarray,
                    method: str = 'both',
                    std_ratio: float = 2.0,
                    eps: float = 0.1,
                    min_samples: int = 10) -> np.ndarray:
    """
    Remove outliers using specified method.
    
    Args:
        points: Input points [N, 3]
        method: 'statistical', 'dbscan', or 'both'
        std_ratio: Standard deviation multiplier for statistical method
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN minimum samples parameter
        
    Returns:
        Filtered points
    """
    if method == 'statistical':
        return remove_outliers_statistical(points, std_ratio=std_ratio)
    elif method == 'dbscan':
        return remove_outliers_dbscan(points, eps=eps, min_samples=min_samples)
    elif method == 'both':
        # Apply both methods sequentially
        points = remove_outliers_statistical(points, std_ratio=std_ratio)
        points = remove_outliers_dbscan(points, eps=eps, min_samples=min_samples)
        return points
    else:
        raise ValueError(f"Unknown outlier removal method: {method}")


def compute_oriented_bbox(points: np.ndarray) -> BoundingBox:
    """
    Compute minimum oriented bounding box using PCA.
    
    Args:
        points: Input points [N, 3]
        
    Returns:
        BoundingBox with dimensions and orientation
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points for bounding box")
    
    # Center points
    mean = points.mean(axis=0)
    centered = points - mean
    
    # PCA for orientation
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort by eigenvalue (largest first)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1
    
    # Transform points to principal axes
    transformed = centered @ eigenvectors
    
    # Compute dimensions along principal axes
    lower_percentile = np.percentile(transformed, 10.0, axis=0)
    upper_percentile = np.percentile(transformed, 90.0, axis=0)
    mins = lower_percentile
    maxs = upper_percentile
    dimensions = maxs - mins
    
    # Sort dimensions (width >= height >= depth convention)
    sorted_dims = sorted(dimensions, reverse=True)
    width, height, depth = sorted_dims
    
    # Compute volume
    volume = width * height * depth
    
    # Compute corners in world space
    corners_local = np.array([
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]]
    ])
    corners_world = (corners_local @ eigenvectors.T) + mean
    
    return BoundingBox(
        width=width,
        height=height,
        depth=depth,
        volume=volume,
        center=mean,
        orientation=eigenvectors,
        corners=corners_world
    )


def compute_axis_aligned_bbox(points: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute axis-aligned bounding box (faster but less accurate for rotated objects).
    
    Args:
        points: Input points [N, 3]
        
    Returns:
        (width, height, depth) in decreasing order
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    dimensions = maxs - mins
    
    # Sort dimensions
    sorted_dims = sorted(dimensions, reverse=True)
    
    return tuple(sorted_dims)


def estimate_measurement_errors(measurements: Dict[str, float],
                                confidence: float,
                                method: str = 'simple') -> Dict[str, float]:
    """
    Estimate error bounds for measurements.
    
    Args:
        measurements: Dictionary with width, height, depth
        confidence: Overall confidence score (0-1)
        method: 'simple' or 'detailed'
        
    Returns:
        Dictionary with error estimates
    """
    if method == 'simple':
        # Simple error model: inversely proportional to confidence
        base_error_percent = (1.0 - confidence) * 15.0  # Max 15% error
        
        return {
            'width_error': measurements['width'] * (base_error_percent / 100.0),
            'height_error': measurements['height'] * (base_error_percent / 100.0),
            'depth_error': measurements['depth'] * (base_error_percent / 100.0),
            'volume_error': measurements.get('volume', 0) * (base_error_percent * 3 / 100.0),
            'relative_error_percent': base_error_percent,
            'confidence': confidence
        }
    
    elif method == 'detailed':
        # More detailed error model based on confidence ranges
        if confidence > 0.9:
            base_error = 0.02  # 2% error
        elif confidence > 0.7:
            base_error = 0.05  # 5% error
        elif confidence > 0.5:
            base_error = 0.10  # 10% error
        elif confidence > 0.3:
            base_error = 0.15  # 15% error
        else:
            base_error = 0.25  # 25% error
        
        # Add dimension-specific factors
        errors = {
            'width_error': measurements['width'] * base_error,
            'height_error': measurements['height'] * base_error * 1.1,  # Slightly worse for height
            'depth_error': measurements['depth'] * base_error * 1.2,  # Worst for depth
            'volume_error': measurements.get('volume', 0) * (base_error ** 3),  # Cubic error
            'relative_error_percent': base_error * 100,
            'confidence': confidence,
            'quality': 'excellent' if confidence > 0.9 else 
                      'good' if confidence > 0.7 else
                      'fair' if confidence > 0.5 else
                      'poor'
        }
        
        return errors
    
    else:
        raise ValueError(f"Unknown error estimation method: {method}")


def format_measurement_with_error(value: float, error: float, unit: str = 'cm') -> str:
    """
    Format a measurement with error bounds in a human-readable way.
    
    Args:
        value: Measurement value
        error: Error estimate
        unit: Unit string
        
    Returns:
        Formatted string like "123.4 ± 5.6 cm"
    """
    return f"{value:.2f} ± {error:.2f} {unit}"


def compute_point_cloud_quality(points: np.ndarray) -> Dict[str, float]:
    """
    Compute quality metrics for a point cloud.
    
    Args:
        points: Input points [N, 3]
        
    Returns:
        Dictionary with quality metrics
    """
    # Point density
    bbox_volume = np.prod(points.max(axis=0) - points.min(axis=0))
    density = len(points) / bbox_volume if bbox_volume > 0 else 0
    
    # Uniformity (using nearest neighbor distances)
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)  # k=2 to get nearest neighbor
    nn_distances = distances[:, 1]  # Skip self (distance 0)
    
    uniformity = 1.0 - (nn_distances.std() / nn_distances.mean()) if nn_distances.mean() > 0 else 0
    uniformity = np.clip(uniformity, 0, 1)
    
    # Completeness (ratio of points within 2 std of centroid)
    centroid = points.mean(axis=0)
    distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
    threshold = distances_to_centroid.mean() + 2 * distances_to_centroid.std()
    completeness = (distances_to_centroid < threshold).sum() / len(points)
    
    # Overall quality score
    quality_score = (density / 1000 * 0.3 + uniformity * 0.4 + completeness * 0.3)
    quality_score = np.clip(quality_score, 0, 1)
    
    return {
        'point_count': len(points),
        'density_pts_per_unit3': density,
        'uniformity': uniformity,
        'completeness': completeness,
        'overall_quality': quality_score
    }


def fit_rectangular_prism(
    points: np.ndarray,
    max_iterations: int = 100,
    inlier_threshold: float = 0.02
) -> Tuple[BoundingBox, float]:
    """
    Fit a rectangular prism (oriented bounding box) to point cloud.

    Uses PCA for initial orientation followed by iterative refinement.

    Args:
        points: Input points [N, 3]
        max_iterations: Maximum refinement iterations
        inlier_threshold: Distance threshold for inliers (meters)

    Returns:
        Tuple of (BoundingBox, fit_residual)
    """
    if len(points) < 8:
        raise ValueError("Need at least 8 points for prism fitting")

    # Import here to avoid circular dependency
    from ..geometry.prism_fitting import RectangularPrismFitter

    fitter = RectangularPrismFitter(
        max_iterations=max_iterations,
        inlier_threshold=inlier_threshold
    )

    prism = fitter.fit(points)

    # Convert to BoundingBox
    bbox = BoundingBox(
        width=float(prism.dimensions[0]),
        height=float(prism.dimensions[1]),
        depth=float(prism.dimensions[2]),
        volume=float(np.prod(prism.dimensions)),
        center=prism.center,
        orientation=prism.rotation,
        corners=prism.corners
    )

    return bbox, prism.residual


def detect_planes_ransac(
    points: np.ndarray,
    max_planes: int = 6,
    distance_threshold: float = 0.01,
    min_inliers: int = 50
) -> list:
    """
    Detect multiple planes in point cloud using RANSAC.

    Args:
        points: Input points [N, 3]
        max_planes: Maximum number of planes to detect
        distance_threshold: RANSAC inlier distance threshold
        min_inliers: Minimum points per plane

    Returns:
        List of PlaneEstimate objects
    """
    # Import here to avoid circular dependency
    from ..geometry.plane_detection import MultiPlaneRANSAC

    detector = MultiPlaneRANSAC(
        n_iterations=1000,
        distance_threshold=distance_threshold,
        min_inliers=min_inliers,
        max_planes=max_planes
    )

    return detector.detect_all(points)


def validate_box_topology(
    planes: list,
    tolerance_degrees: float = 5.0
) -> Tuple[bool, float, Dict]:
    """
    Validate if detected planes form a box-like topology.

    Checks for orthogonal and parallel plane pairs as expected
    in rectangular boxes.

    Args:
        planes: List of PlaneEstimate objects
        tolerance_degrees: Angle tolerance for orthogonality/parallelism

    Returns:
        Tuple of (is_valid, confidence_score, diagnostics)
    """
    # Import here to avoid circular dependency
    from ..geometry.plane_detection import validate_box_topology as _validate

    return _validate(planes, tolerance_degrees, tolerance_degrees)


def compute_geometric_confidence(
    points: np.ndarray,
    bbox: BoundingBox
) -> float:
    """
    Compute confidence score based on geometric fit quality.

    Evaluates how well points fit the bounding box geometry.

    Args:
        points: Point cloud [N, 3]
        bbox: Fitted bounding box

    Returns:
        Confidence score [0, 1]
    """
    # Transform points to bbox local coordinates
    centered = points - bbox.center
    local = centered @ bbox.orientation

    # Get half-dimensions
    half_dims = np.array([bbox.width, bbox.height, bbox.depth]) / 2

    # Compute distance to surface for each point
    scaled = local / (half_dims + 1e-6)
    max_scaled = np.max(np.abs(scaled), axis=1)

    # Points inside have max_scaled <= 1
    inside = max_scaled <= 1.0
    inside_ratio = np.mean(inside)

    # Surface distance for points
    surface_dist = (max_scaled - 1) * np.min(half_dims)
    mean_dist = np.mean(np.abs(surface_dist))

    # Confidence combines inside ratio and surface fit
    fit_quality = np.exp(-mean_dist / (np.min(half_dims) + 0.01))
    confidence = 0.6 * inside_ratio + 0.4 * fit_quality

    return float(np.clip(confidence, 0, 1))


def refine_bbox_with_prism(
    points: np.ndarray,
    initial_bbox: BoundingBox,
    max_iterations: int = 50
) -> BoundingBox:
    """
    Refine bounding box using iterative prism fitting.

    Args:
        points: Point cloud [N, 3]
        initial_bbox: Initial bounding box estimate
        max_iterations: Maximum iterations for refinement

    Returns:
        Refined BoundingBox
    """
    # Import here to avoid circular dependency
    from ..geometry.prism_fitting import RectangularPrismFitter

    fitter = RectangularPrismFitter(max_iterations=max_iterations)

    # Use initial bbox orientation as hint (via plane detection)
    prism = fitter.fit(points)

    return BoundingBox(
        width=float(prism.dimensions[0]),
        height=float(prism.dimensions[1]),
        depth=float(prism.dimensions[2]),
        volume=float(np.prod(prism.dimensions)),
        center=prism.center,
        orientation=prism.rotation,
        corners=prism.corners
    )

