"""
Rectangular prism fitting for 3D point clouds.

Fits oriented bounding boxes and rectangular prisms to point clouds
with geometric constraint enforcement.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .plane_detection import PlaneEstimate

logger = logging.getLogger(__name__)


@dataclass
class PrismFit:
    """
    Rectangular prism fit result.

    Attributes:
        center: Center point of prism [3]
        dimensions: Dimensions [width, height, depth] in meters
        rotation: Rotation matrix [3, 3]
        residual: Average fitting residual
        inlier_ratio: Ratio of points within prism tolerance
        corners: 8 corner points of the prism [8, 3]
    """
    center: np.ndarray  # [3]
    dimensions: np.ndarray  # [width, height, depth]
    rotation: np.ndarray  # [3, 3] rotation matrix
    residual: float
    inlier_ratio: float
    corners: Optional[np.ndarray] = None  # [8, 3]

    def __post_init__(self):
        """Compute corners if not provided."""
        if self.corners is None:
            self.corners = self._compute_corners()

    def _compute_corners(self) -> np.ndarray:
        """Compute 8 corner points of the prism."""
        w, h, d = self.dimensions / 2

        # Local corners
        local_corners = np.array([
            [-w, -h, -d],
            [+w, -h, -d],
            [+w, +h, -d],
            [-w, +h, -d],
            [-w, -h, +d],
            [+w, -h, +d],
            [+w, +h, +d],
            [-w, +h, +d]
        ])

        # Transform to world coordinates
        world_corners = (local_corners @ self.rotation.T) + self.center

        return world_corners

    def get_volume(self) -> float:
        """Return volume of prism in cubic meters."""
        return float(np.prod(self.dimensions))

    def get_surface_area(self) -> float:
        """Return surface area of prism in square meters."""
        w, h, d = self.dimensions
        return 2 * (w*h + h*d + d*w)

    def contains_point(self, point: np.ndarray, tolerance: float = 0.0) -> bool:
        """Check if point is inside prism (with optional tolerance)."""
        # Transform to local coordinates
        local = (point - self.center) @ self.rotation
        half_dims = self.dimensions / 2 + tolerance
        return np.all(np.abs(local) <= half_dims)

    def distance_to_surface(self, points: np.ndarray) -> np.ndarray:
        """Compute distance from points to nearest prism surface."""
        # Transform to local coordinates
        local = (points - self.center) @ self.rotation
        half_dims = self.dimensions / 2

        # Distance to each face
        distances = np.abs(local) - half_dims

        # For points inside, distance is negative
        # For points outside, distance is the max positive component
        inside_mask = np.all(distances < 0, axis=1)

        result = np.zeros(len(points))
        result[inside_mask] = -np.max(-distances[inside_mask], axis=1)
        result[~inside_mask] = np.max(np.maximum(distances[~inside_mask], 0), axis=1)

        return result


class RectangularPrismFitter:
    """
    Fits rectangular prisms to 3D point clouds.

    Uses PCA for initial orientation estimation followed by
    iterative refinement with optional plane constraints.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        inlier_threshold: float = 0.02,
        refine_rotation: bool = True
    ):
        """
        Initialize prism fitter.

        Args:
            max_iterations: Maximum optimization iterations
            inlier_threshold: Distance threshold for inliers (meters)
            refine_rotation: Whether to refine rotation iteratively
        """
        self.max_iterations = max_iterations
        self.inlier_threshold = inlier_threshold
        self.refine_rotation = refine_rotation

    def fit(
        self,
        points: np.ndarray,
        initial_planes: Optional[List[PlaneEstimate]] = None
    ) -> PrismFit:
        """
        Fit rectangular prism to point cloud.

        Args:
            points: Point cloud [N, 3]
            initial_planes: Optional detected planes for initialization

        Returns:
            PrismFit with fitted prism parameters
        """
        if len(points) < 8:
            raise ValueError("Need at least 8 points for prism fitting")

        logger.debug(f"Fitting prism to {len(points)} points")

        # Initial estimate using PCA
        center, rotation, dimensions = self._pca_initial_fit(points)

        # If planes provided, use them to refine rotation
        if initial_planes and len(initial_planes) >= 3:
            rotation = self._refine_rotation_from_planes(rotation, initial_planes)

        # Refine iteratively
        if self.refine_rotation:
            center, rotation, dimensions = self._iterative_refinement(
                points, center, rotation, dimensions
            )

        # Compute fit quality
        residual, inlier_ratio = self._compute_fit_quality(
            points, center, rotation, dimensions
        )

        # Ensure dimensions are sorted (width >= height >= depth)
        sorted_indices = np.argsort(dimensions)[::-1]
        dimensions = dimensions[sorted_indices]
        rotation = rotation[:, sorted_indices]

        prism = PrismFit(
            center=center,
            dimensions=dimensions,
            rotation=rotation,
            residual=residual,
            inlier_ratio=inlier_ratio
        )

        logger.debug(
            f"Prism fit: dims={dimensions}, "
            f"residual={residual:.4f}, inlier_ratio={inlier_ratio:.2f}"
        )

        return prism

    def _pca_initial_fit(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get initial prism estimate using PCA."""
        center = np.mean(points, axis=0)
        centered = points - center

        # PCA for orientation
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        rotation = eigenvectors[:, idx]

        # Ensure right-handed coordinate system
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1

        # Project to get dimensions
        projected = centered @ rotation
        dimensions = np.ptp(projected, axis=0)  # Range along each axis

        return center, rotation, dimensions

    def _refine_rotation_from_planes(
        self,
        initial_rotation: np.ndarray,
        planes: List[PlaneEstimate]
    ) -> np.ndarray:
        """Refine rotation using detected plane normals."""
        if len(planes) < 3:
            return initial_rotation

        # Find 3 most orthogonal planes
        best_triple = None
        best_orthogonality = -1

        for i in range(len(planes)):
            for j in range(i+1, len(planes)):
                for k in range(j+1, len(planes)):
                    n1, n2, n3 = planes[i].normal, planes[j].normal, planes[k].normal

                    # Check orthogonality
                    dot12 = np.abs(np.dot(n1, n2))
                    dot13 = np.abs(np.dot(n1, n3))
                    dot23 = np.abs(np.dot(n2, n3))

                    orthogonality = 3 - (dot12 + dot13 + dot23)

                    if orthogonality > best_orthogonality:
                        best_orthogonality = orthogonality
                        best_triple = (planes[i], planes[j], planes[k])

        if best_triple is None:
            return initial_rotation

        # Build rotation from plane normals
        n1 = best_triple[0].normal
        n2 = best_triple[1].normal

        # Orthogonalize
        n2_orth = n2 - np.dot(n2, n1) * n1
        n2_orth = n2_orth / np.linalg.norm(n2_orth)
        n3 = np.cross(n1, n2_orth)

        rotation = np.column_stack([n1, n2_orth, n3])

        # Ensure right-handed
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1

        return rotation

    def _iterative_refinement(
        self,
        points: np.ndarray,
        center: np.ndarray,
        rotation: np.ndarray,
        dimensions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Iteratively refine prism parameters."""
        # Convert rotation to axis-angle for optimization
        r = Rotation.from_matrix(rotation)
        rotvec = r.as_rotvec()

        # Initial parameters: [cx, cy, cz, rx, ry, rz, w, h, d]
        x0 = np.concatenate([center, rotvec, dimensions])

        def objective(params):
            c = params[:3]
            rv = params[3:6]
            dims = np.abs(params[6:9])  # Ensure positive

            R = Rotation.from_rotvec(rv).as_matrix()

            # Transform points to local coordinates
            local = (points - c) @ R
            half_dims = dims / 2

            # Distance to surface (for box, use L-infinity distance concept)
            scaled = local / (half_dims + 1e-6)
            dist_to_surface = np.max(np.abs(scaled), axis=1) - 1

            # Penalize points outside
            outside = np.maximum(dist_to_surface, 0)

            return np.sum(outside ** 2)

        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': self.max_iterations}
        )

        # Extract refined parameters
        center_refined = result.x[:3]
        rotation_refined = Rotation.from_rotvec(result.x[3:6]).as_matrix()
        dimensions_refined = np.abs(result.x[6:9])

        return center_refined, rotation_refined, dimensions_refined

    def _compute_fit_quality(
        self,
        points: np.ndarray,
        center: np.ndarray,
        rotation: np.ndarray,
        dimensions: np.ndarray
    ) -> Tuple[float, float]:
        """Compute fitting residual and inlier ratio."""
        # Transform to local coordinates
        local = (points - center) @ rotation
        half_dims = dimensions / 2

        # Compute distance to surface
        scaled = local / (half_dims + 1e-6)
        max_scaled = np.max(np.abs(scaled), axis=1)

        # Points inside have max_scaled <= 1
        distances = (max_scaled - 1) * np.min(half_dims)

        # Residual is mean absolute distance
        residual = float(np.mean(np.abs(distances)))

        # Inlier ratio
        inliers = np.abs(distances) < self.inlier_threshold
        inlier_ratio = float(np.sum(inliers) / len(points))

        return residual, inlier_ratio


class BoxConstraints:
    """
    Enforces geometric constraints for box-like structures.

    Constraints include orthogonality, parallelism, and planarity.
    """

    def __init__(
        self,
        orthogonality_tolerance: float = 5.0,
        parallelism_tolerance: float = 5.0
    ):
        """
        Initialize constraint enforcer.

        Args:
            orthogonality_tolerance: Tolerance for 90-degree angles (degrees)
            parallelism_tolerance: Tolerance for parallel faces (degrees)
        """
        self.orthogonality_tolerance = np.radians(orthogonality_tolerance)
        self.parallelism_tolerance = np.radians(parallelism_tolerance)

    def enforce_orthogonality(
        self,
        planes: List[PlaneEstimate]
    ) -> List[PlaneEstimate]:
        """
        Adjust plane normals to be mutually orthogonal.

        Uses Gram-Schmidt-like process to orthogonalize normals
        while minimizing deviation from original normals.

        Args:
            planes: List of detected planes

        Returns:
            List of planes with adjusted normals
        """
        if len(planes) < 2:
            return planes

        # Group planes by approximate normal direction
        groups = self._group_by_normal(planes)

        if len(groups) < 3:
            return planes

        # Take dominant plane from each group
        dominant_normals = []
        for group in groups[:3]:
            # Weighted average normal
            weights = np.array([p.confidence for p in group])
            normals = np.array([p.normal for p in group])
            avg_normal = np.average(normals, axis=0, weights=weights)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            dominant_normals.append(avg_normal)

        # Orthogonalize using Gram-Schmidt
        n1 = dominant_normals[0]
        n2 = dominant_normals[1] - np.dot(dominant_normals[1], n1) * n1
        n2 = n2 / np.linalg.norm(n2)
        n3 = np.cross(n1, n2)

        orthogonal_normals = [n1, n2, n3]

        # Assign each plane to nearest orthogonal normal
        adjusted_planes = []
        for plane in planes:
            dots = [np.abs(np.dot(plane.normal, on)) for on in orthogonal_normals]
            best_idx = np.argmax(dots)
            new_normal = orthogonal_normals[best_idx]

            # Check if normal should be flipped
            if np.dot(plane.normal, new_normal) < 0:
                new_normal = -new_normal

            # Create adjusted plane
            adjusted = PlaneEstimate(
                normal=new_normal,
                distance=np.dot(new_normal, plane.centroid) if plane.centroid is not None
                         else plane.distance,
                inliers=plane.inliers,
                confidence=plane.confidence,
                centroid=plane.centroid,
                extent=plane.extent
            )
            adjusted_planes.append(adjusted)

        return adjusted_planes

    def enforce_parallelism(
        self,
        planes: List[PlaneEstimate]
    ) -> List[PlaneEstimate]:
        """
        Adjust opposite faces to be exactly parallel.

        Args:
            planes: List of detected planes (preferably after orthogonality enforcement)

        Returns:
            List of planes with parallel pairs aligned
        """
        if len(planes) < 2:
            return planes

        # Group by normal direction (parallel planes have same or opposite normals)
        groups = self._group_by_normal(planes, tolerance=self.parallelism_tolerance)

        adjusted_planes = []
        for group in groups:
            if len(group) == 1:
                adjusted_planes.append(group[0])
                continue

            # Average normal for group
            normals = np.array([p.normal for p in group])
            weights = np.array([p.confidence for p in group])

            # Account for opposite directions
            reference = normals[0]
            aligned_normals = []
            for n in normals:
                if np.dot(n, reference) < 0:
                    aligned_normals.append(-n)
                else:
                    aligned_normals.append(n)

            avg_normal = np.average(aligned_normals, axis=0, weights=weights)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)

            # Adjust each plane in group
            for plane in group:
                sign = 1 if np.dot(plane.normal, avg_normal) >= 0 else -1
                new_normal = sign * avg_normal

                adjusted = PlaneEstimate(
                    normal=new_normal,
                    distance=plane.distance,
                    inliers=plane.inliers,
                    confidence=plane.confidence,
                    centroid=plane.centroid,
                    extent=plane.extent
                )
                adjusted_planes.append(adjusted)

        return adjusted_planes

    def _group_by_normal(
        self,
        planes: List[PlaneEstimate],
        tolerance: Optional[float] = None
    ) -> List[List[PlaneEstimate]]:
        """Group planes by similar normal direction."""
        if tolerance is None:
            tolerance = self.orthogonality_tolerance

        groups = []
        used = [False] * len(planes)

        for i, plane in enumerate(planes):
            if used[i]:
                continue

            group = [plane]
            used[i] = True

            for j, other in enumerate(planes[i+1:], i+1):
                if used[j]:
                    continue

                # Check if normals are similar (same or opposite direction)
                dot = np.abs(np.dot(plane.normal, other.normal))
                angle = np.arccos(np.clip(dot, 0, 1))

                if angle < tolerance or angle > np.pi - tolerance:
                    group.append(other)
                    used[j] = True

            groups.append(group)

        return groups
