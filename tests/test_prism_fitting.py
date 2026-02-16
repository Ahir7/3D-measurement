"""
Unit tests for prism fitting module.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation

from src.geometry.prism_fitting import (
    PrismFit,
    RectangularPrismFitter,
    BoxConstraints
)
from src.geometry.plane_detection import PlaneEstimate


class TestPrismFit:
    """Tests for PrismFit dataclass."""

    def test_creation(self):
        """Test basic prism creation."""
        prism = PrismFit(
            center=np.array([0, 0, 0]),
            dimensions=np.array([1, 2, 3]),
            rotation=np.eye(3),
            residual=0.01,
            inlier_ratio=0.95
        )

        assert np.allclose(prism.center, [0, 0, 0])
        assert np.allclose(prism.dimensions, [1, 2, 3])
        assert prism.residual == 0.01
        assert prism.inlier_ratio == 0.95

    def test_corners_computed(self):
        """Test that corners are computed automatically."""
        prism = PrismFit(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 2, 2]),
            rotation=np.eye(3),
            residual=0.0,
            inlier_ratio=1.0
        )

        assert prism.corners is not None
        assert prism.corners.shape == (8, 3)

        # Check corner positions for unit cube centered at origin
        expected_max = 1.0
        assert np.allclose(np.abs(prism.corners).max(axis=0), [1, 1, 1])

    def test_get_volume(self):
        """Test volume computation."""
        prism = PrismFit(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 3, 4]),
            rotation=np.eye(3),
            residual=0.0,
            inlier_ratio=1.0
        )

        assert prism.get_volume() == 24.0

    def test_get_surface_area(self):
        """Test surface area computation."""
        # 2x3x4 box
        prism = PrismFit(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 3, 4]),
            rotation=np.eye(3),
            residual=0.0,
            inlier_ratio=1.0
        )

        # Surface area = 2*(2*3 + 3*4 + 4*2) = 2*(6+12+8) = 52
        assert prism.get_surface_area() == 52.0

    def test_contains_point(self):
        """Test point containment check."""
        prism = PrismFit(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 2, 2]),
            rotation=np.eye(3),
            residual=0.0,
            inlier_ratio=1.0
        )

        # Point at center should be inside
        assert prism.contains_point(np.array([0, 0, 0]))

        # Point at corner should be inside (with small tolerance)
        assert prism.contains_point(np.array([0.9, 0.9, 0.9]))

        # Point outside should not be inside
        assert not prism.contains_point(np.array([2, 0, 0]))

    def test_distance_to_surface(self):
        """Test distance to surface computation."""
        prism = PrismFit(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 2, 2]),
            rotation=np.eye(3),
            residual=0.0,
            inlier_ratio=1.0
        )

        points = np.array([
            [0, 0, 0],   # Center (negative distance)
            [1, 0, 0],   # On surface
            [2, 0, 0]    # Outside (positive distance)
        ])

        distances = prism.distance_to_surface(points)

        assert distances[0] < 0  # Inside
        assert np.abs(distances[1]) < 0.01  # On surface
        assert distances[2] > 0  # Outside


class TestRectangularPrismFitter:
    """Tests for RectangularPrismFitter."""

    def test_initialization(self):
        """Test fitter initialization."""
        fitter = RectangularPrismFitter(
            max_iterations=50,
            inlier_threshold=0.05
        )

        assert fitter.max_iterations == 50
        assert fitter.inlier_threshold == 0.05

    def test_fit_aligned_box(self):
        """Test fitting axis-aligned box."""
        fitter = RectangularPrismFitter(max_iterations=50)

        # Create points on surface of 2x3x4 box centered at origin
        points = []

        # Generate points on each face
        n_per_face = 50
        for _ in range(n_per_face):
            # +X face
            points.append([1, np.random.uniform(-1.5, 1.5), np.random.uniform(-2, 2)])
            # -X face
            points.append([-1, np.random.uniform(-1.5, 1.5), np.random.uniform(-2, 2)])
            # +Y face
            points.append([np.random.uniform(-1, 1), 1.5, np.random.uniform(-2, 2)])
            # -Y face
            points.append([np.random.uniform(-1, 1), -1.5, np.random.uniform(-2, 2)])
            # +Z face
            points.append([np.random.uniform(-1, 1), np.random.uniform(-1.5, 1.5), 2])
            # -Z face
            points.append([np.random.uniform(-1, 1), np.random.uniform(-1.5, 1.5), -2])

        points = np.array(points)

        prism = fitter.fit(points)

        # Check dimensions are approximately correct (sorted)
        dims_sorted = np.sort(prism.dimensions)
        expected_sorted = np.sort([2, 3, 4])

        assert np.allclose(dims_sorted, expected_sorted, atol=0.5)

    def test_fit_rotated_box(self):
        """Test fitting rotated box."""
        fitter = RectangularPrismFitter(max_iterations=100)

        # Create rotated box
        rotation = Rotation.from_euler('z', 45, degrees=True).as_matrix()

        # 2x2x2 box points
        points = []
        n_per_face = 30
        for _ in range(n_per_face):
            # Generate on each face, then rotate
            local_points = [
                [1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                [-1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                [np.random.uniform(-1, 1), 1, np.random.uniform(-1, 1)],
                [np.random.uniform(-1, 1), -1, np.random.uniform(-1, 1)],
                [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 1],
                [np.random.uniform(-1, 1), np.random.uniform(-1, 1), -1]
            ]
            points.extend(local_points)

        points = np.array(points) @ rotation.T  # Apply rotation

        prism = fitter.fit(points)

        # Dimensions should be approximately 2x2x2
        assert np.allclose(prism.dimensions, [2, 2, 2], atol=0.5)

    def test_fit_insufficient_points(self):
        """Test with insufficient points."""
        fitter = RectangularPrismFitter()

        points = np.random.randn(5, 3)

        with pytest.raises(ValueError):
            fitter.fit(points)


class TestBoxConstraints:
    """Tests for BoxConstraints."""

    def test_initialization(self):
        """Test initialization."""
        constraints = BoxConstraints(
            orthogonality_tolerance=10.0,
            parallelism_tolerance=5.0
        )

        assert np.isclose(constraints.orthogonality_tolerance, np.radians(10.0))
        assert np.isclose(constraints.parallelism_tolerance, np.radians(5.0))

    def test_enforce_orthogonality(self):
        """Test orthogonality enforcement."""
        constraints = BoxConstraints(orthogonality_tolerance=10.0)

        # Create planes that are almost orthogonal
        planes = [
            PlaneEstimate(
                normal=np.array([1, 0.1, 0]) / np.linalg.norm([1, 0.1, 0]),
                distance=0,
                inliers=np.array([0]),
                confidence=0.9,
                centroid=np.array([0, 0, 0])
            ),
            PlaneEstimate(
                normal=np.array([0.1, 1, 0]) / np.linalg.norm([0.1, 1, 0]),
                distance=0,
                inliers=np.array([0]),
                confidence=0.9,
                centroid=np.array([0, 0, 0])
            ),
            PlaneEstimate(
                normal=np.array([0, 0.1, 1]) / np.linalg.norm([0, 0.1, 1]),
                distance=0,
                inliers=np.array([0]),
                confidence=0.9,
                centroid=np.array([0, 0, 0])
            )
        ]

        adjusted = constraints.enforce_orthogonality(planes)

        assert len(adjusted) == 3

        # Check that adjusted planes are orthogonal
        n1 = adjusted[0].normal
        n2 = adjusted[1].normal
        n3 = adjusted[2].normal

        assert np.abs(np.dot(n1, n2)) < 0.1
        assert np.abs(np.dot(n1, n3)) < 0.1
        assert np.abs(np.dot(n2, n3)) < 0.1

    def test_enforce_parallelism(self):
        """Test parallelism enforcement."""
        constraints = BoxConstraints(parallelism_tolerance=10.0)

        # Create almost parallel planes
        planes = [
            PlaneEstimate(
                normal=np.array([1, 0.05, 0]) / np.linalg.norm([1, 0.05, 0]),
                distance=0,
                inliers=np.array([0]),
                confidence=0.9
            ),
            PlaneEstimate(
                normal=np.array([1, -0.05, 0]) / np.linalg.norm([1, -0.05, 0]),
                distance=1,
                inliers=np.array([0]),
                confidence=0.9
            )
        ]

        adjusted = constraints.enforce_parallelism(planes)

        assert len(adjusted) == 2

        # Check normals are parallel (same or opposite)
        dot = np.abs(np.dot(adjusted[0].normal, adjusted[1].normal))
        assert dot > 0.99
