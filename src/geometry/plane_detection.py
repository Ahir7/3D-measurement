"""
RANSAC-based plane detection for 3D point clouds.

Detects planar surfaces in point clouds using RANSAC algorithm,
supporting multi-plane detection and box topology validation.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PlaneEstimate:
    """
    Estimated plane parameters and metadata.

    Attributes:
        normal: Unit normal vector [3]
        distance: Distance from origin (plane equation: n·x = d)
        inliers: Indices of inlier points
        confidence: Confidence score for plane detection
        centroid: Centroid of inlier points
        extent: Approximate plane extent [width, height]
    """
    normal: np.ndarray  # [3] unit normal
    distance: float  # distance from origin
    inliers: np.ndarray  # [N] indices of inlier points
    confidence: float
    centroid: Optional[np.ndarray] = None  # [3] centroid of inliers
    extent: Optional[Tuple[float, float]] = None  # (width, height)

    def get_plane_equation(self) -> Tuple[float, float, float, float]:
        """Return plane equation coefficients (a, b, c, d) where ax + by + cz = d."""
        return (self.normal[0], self.normal[1], self.normal[2], self.distance)

    def point_to_plane_distance(self, points: np.ndarray) -> np.ndarray:
        """
        Compute signed distance from points to plane.

        Args:
            points: Points array [N, 3]

        Returns:
            Signed distances [N]
        """
        return np.dot(points, self.normal) - self.distance


class RANSACPlaneDetector:
    """
    RANSAC-based single plane detector.

    Implements standard RANSAC algorithm for robust plane fitting
    with configurable parameters.
    """

    def __init__(
        self,
        n_iterations: int = 1000,
        distance_threshold: float = 0.01,
        min_inliers: int = 50,
        confidence: float = 0.99
    ):
        """
        Initialize RANSAC plane detector.

        Args:
            n_iterations: Maximum RANSAC iterations
            distance_threshold: Maximum distance for inliers (meters)
            min_inliers: Minimum required inliers for valid plane
            confidence: Target confidence level for adaptive stopping
        """
        self.n_iterations = n_iterations
        self.distance_threshold = distance_threshold
        self.min_inliers = min_inliers
        self.confidence = confidence

    def detect(self, points: np.ndarray) -> Optional[PlaneEstimate]:
        """
        Detect best plane in point cloud.

        Args:
            points: Point cloud [N, 3]

        Returns:
            PlaneEstimate or None if no valid plane found
        """
        if len(points) < self.min_inliers:
            logger.warning(f"Insufficient points for plane detection: {len(points)}")
            return None

        n_points = len(points)
        best_inliers = None
        best_inlier_count = 0
        best_plane = None

        # Adaptive iteration count based on inlier ratio
        max_iterations = self.n_iterations
        iterations = 0

        while iterations < max_iterations:
            # Random sample 3 points
            indices = np.random.choice(n_points, 3, replace=False)
            sample_points = points[indices]

            # Compute plane from 3 points
            plane = self._fit_plane_from_points(sample_points)
            if plane is None:
                iterations += 1
                continue

            normal, distance = plane

            # Count inliers
            distances = np.abs(np.dot(points, normal) - distance)
            inlier_mask = distances < self.distance_threshold
            inlier_count = np.sum(inlier_mask)

            # Update best if improved
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = np.where(inlier_mask)[0]
                best_plane = (normal, distance)

                # Adaptive stopping
                inlier_ratio = inlier_count / n_points
                if inlier_ratio > 0.1:  # If reasonable inlier ratio
                    # Update max iterations based on current best
                    if inlier_ratio > 0:
                        k = np.log(1 - self.confidence) / np.log(1 - inlier_ratio**3 + 1e-10)
                        max_iterations = min(max_iterations, int(k) + 1)

            iterations += 1

        if best_plane is None or best_inlier_count < self.min_inliers:
            logger.debug(f"No valid plane found (best had {best_inlier_count} inliers)")
            return None

        # Refine plane using all inliers
        inlier_points = points[best_inliers]
        refined_normal, refined_distance = self._fit_plane_svd(inlier_points)

        # Compute confidence based on inlier ratio and fit quality
        inlier_ratio = best_inlier_count / n_points
        residuals = np.abs(np.dot(inlier_points, refined_normal) - refined_distance)
        fit_quality = np.exp(-np.mean(residuals) / self.distance_threshold)
        confidence = inlier_ratio * fit_quality

        # Compute centroid and extent
        centroid = np.mean(inlier_points, axis=0)
        extent = self._compute_plane_extent(inlier_points, refined_normal)

        logger.debug(
            f"Plane detected: {best_inlier_count}/{n_points} inliers, "
            f"confidence={confidence:.3f}"
        )

        return PlaneEstimate(
            normal=refined_normal,
            distance=refined_distance,
            inliers=best_inliers,
            confidence=confidence,
            centroid=centroid,
            extent=extent
        )

    def _fit_plane_from_points(
        self,
        points: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Fit plane from exactly 3 points using cross product."""
        if len(points) != 3:
            return None

        v1 = points[1] - points[0]
        v2 = points[2] - points[0]

        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)

        if norm < 1e-10:  # Degenerate case (collinear points)
            return None

        normal = normal / norm
        distance = np.dot(normal, points[0])

        # Ensure consistent normal direction (pointing towards positive distance)
        if distance < 0:
            normal = -normal
            distance = -distance

        return normal, distance

    def _fit_plane_svd(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Fit plane using SVD (best fit for multiple points)."""
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # SVD
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]  # Smallest singular value corresponds to normal

        # Ensure consistent direction
        normal = normal / np.linalg.norm(normal)
        distance = np.dot(normal, centroid)

        if distance < 0:
            normal = -normal
            distance = -distance

        return normal, distance

    def _compute_plane_extent(
        self,
        points: np.ndarray,
        normal: np.ndarray
    ) -> Tuple[float, float]:
        """Compute approximate extent of plane in local coordinates."""
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # Create local coordinate system on plane
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, [0, 0, 1])
        else:
            u = np.cross(normal, [1, 0, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        # Project points to local 2D
        proj_u = np.dot(centered, u)
        proj_v = np.dot(centered, v)

        width = proj_u.max() - proj_u.min()
        height = proj_v.max() - proj_v.min()

        return (width, height)


class MultiPlaneRANSAC:
    """
    Multi-plane RANSAC detector.

    Sequentially detects multiple planes by removing inliers after
    each successful detection.
    """

    def __init__(
        self,
        n_iterations: int = 1000,
        distance_threshold: float = 0.01,
        min_inliers: int = 50,
        max_planes: int = 6
    ):
        """
        Initialize multi-plane detector.

        Args:
            n_iterations: RANSAC iterations per plane
            distance_threshold: Inlier distance threshold
            min_inliers: Minimum inliers per plane
            max_planes: Maximum planes to detect
        """
        self.detector = RANSACPlaneDetector(
            n_iterations=n_iterations,
            distance_threshold=distance_threshold,
            min_inliers=min_inliers
        )
        self.max_planes = max_planes

    def detect_all(
        self,
        points: np.ndarray,
        min_remaining: int = 100
    ) -> List[PlaneEstimate]:
        """
        Detect all significant planes in point cloud.

        Args:
            points: Point cloud [N, 3]
            min_remaining: Stop when fewer points remain

        Returns:
            List of PlaneEstimate objects
        """
        planes = []
        remaining_points = points.copy()
        remaining_indices = np.arange(len(points))

        while len(remaining_points) > min_remaining and len(planes) < self.max_planes:
            # Detect plane in remaining points
            plane = self.detector.detect(remaining_points)

            if plane is None:
                break

            # Map inlier indices back to original
            original_inliers = remaining_indices[plane.inliers]
            plane.inliers = original_inliers

            planes.append(plane)

            # Remove inliers for next iteration
            mask = np.ones(len(remaining_points), dtype=bool)
            mask[plane.inliers] = False
            mask_original = np.isin(np.arange(len(remaining_points)),
                                    np.where(~np.isin(remaining_indices, original_inliers))[0],
                                    invert=True)

            # Actually remove the points that were detected as inliers
            keep_mask = ~np.isin(remaining_indices, original_inliers)
            remaining_points = remaining_points[keep_mask]
            remaining_indices = remaining_indices[keep_mask]

            logger.debug(f"Plane {len(planes)} detected, {len(remaining_points)} points remaining")

        logger.info(f"Detected {len(planes)} planes from {len(points)} points")
        return planes


def compute_plane_orthogonality(
    planes: List[PlaneEstimate],
    tolerance_degrees: float = 5.0
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Check if planes are mutually orthogonal (box topology).

    For a rectangular box, we expect 3 pairs of parallel planes
    with each pair orthogonal to the other two pairs.

    Args:
        planes: List of detected planes
        tolerance_degrees: Angle tolerance for orthogonality

    Returns:
        Tuple of (orthogonality_score, list of (i, j, angle) for orthogonal pairs)
    """
    if len(planes) < 2:
        return 0.0, []

    orthogonal_pairs = []
    tolerance_rad = np.radians(tolerance_degrees)

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            n1 = planes[i].normal
            n2 = planes[j].normal

            # Compute angle between normals
            dot = np.abs(np.dot(n1, n2))
            angle = np.arccos(np.clip(dot, 0, 1))

            # Check for orthogonality (angle ~= 90 degrees)
            angle_from_90 = np.abs(angle - np.pi / 2)

            if angle_from_90 < tolerance_rad:
                orthogonal_pairs.append((i, j, np.degrees(angle)))

    # Score based on number of orthogonal pairs
    # For 6 planes (box), we expect up to 12 orthogonal pairs
    # For 3 planes (corner), we expect 3 orthogonal pairs
    expected_pairs = min(len(planes) * (len(planes) - 1) // 4, 12)
    score = len(orthogonal_pairs) / max(expected_pairs, 1)

    return float(np.clip(score, 0, 1)), orthogonal_pairs


def compute_plane_parallelism(
    planes: List[PlaneEstimate],
    tolerance_degrees: float = 5.0
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Check for parallel plane pairs (opposite faces of box).

    Args:
        planes: List of detected planes
        tolerance_degrees: Angle tolerance for parallelism

    Returns:
        Tuple of (parallelism_score, list of (i, j, angle) for parallel pairs)
    """
    if len(planes) < 2:
        return 0.0, []

    parallel_pairs = []
    tolerance_rad = np.radians(tolerance_degrees)

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            n1 = planes[i].normal
            n2 = planes[j].normal

            # Compute angle between normals
            # Parallel planes have normals that are either same or opposite direction
            dot = np.abs(np.dot(n1, n2))
            angle = np.arccos(np.clip(dot, 0, 1))

            # Check for parallelism (angle ~= 0 or 180 degrees)
            if angle < tolerance_rad or angle > np.pi - tolerance_rad:
                parallel_pairs.append((i, j, np.degrees(angle)))

    # For a box, we expect 3 pairs of parallel planes
    expected_pairs = min(len(planes) // 2, 3)
    score = len(parallel_pairs) / max(expected_pairs, 1)

    return float(np.clip(score, 0, 1)), parallel_pairs


def validate_box_topology(
    planes: List[PlaneEstimate],
    orthogonality_tolerance: float = 5.0,
    parallelism_tolerance: float = 5.0
) -> Tuple[bool, float, dict]:
    """
    Validate if detected planes form a box topology.

    A valid box should have:
    - At least 3 mutually orthogonal planes (corner case)
    - Or 6 planes with 3 pairs of parallel planes

    Args:
        planes: List of detected planes
        orthogonality_tolerance: Angle tolerance for orthogonality (degrees)
        parallelism_tolerance: Angle tolerance for parallelism (degrees)

    Returns:
        Tuple of (is_valid, confidence_score, diagnostics_dict)
    """
    diagnostics = {
        'num_planes': len(planes),
        'orthogonal_pairs': [],
        'parallel_pairs': [],
        'orthogonality_score': 0.0,
        'parallelism_score': 0.0
    }

    if len(planes) < 3:
        return False, 0.0, diagnostics

    # Check orthogonality
    orth_score, orth_pairs = compute_plane_orthogonality(planes, orthogonality_tolerance)
    diagnostics['orthogonality_score'] = orth_score
    diagnostics['orthogonal_pairs'] = orth_pairs

    # Check parallelism
    para_score, para_pairs = compute_plane_parallelism(planes, parallelism_tolerance)
    diagnostics['parallelism_score'] = para_score
    diagnostics['parallel_pairs'] = para_pairs

    # Combined score
    if len(planes) >= 6:
        # Full box: need both orthogonality and parallelism
        combined_score = 0.5 * orth_score + 0.5 * para_score
        is_valid = combined_score > 0.6 and len(para_pairs) >= 2
    else:
        # Partial box: focus on orthogonality
        combined_score = orth_score
        is_valid = len(orth_pairs) >= 2

    diagnostics['combined_score'] = combined_score

    return is_valid, combined_score, diagnostics
