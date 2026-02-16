"""
Unit tests for plane detection module.
"""

import pytest
import numpy as np

from src.geometry.plane_detection import (
    PlaneEstimate,
    RANSACPlaneDetector,
    MultiPlaneRANSAC,
    compute_plane_orthogonality,
    compute_plane_parallelism,
    validate_box_topology
)


class TestPlaneEstimate:
    """Tests for PlaneEstimate dataclass."""

    def test_creation(self):
        """Test basic plane creation."""
        normal = np.array([0, 0, 1])
        plane = PlaneEstimate(
            normal=normal,
            distance=1.0,
            inliers=np.array([0, 1, 2]),
            confidence=0.9
        )

        assert np.allclose(plane.normal, [0, 0, 1])
        assert plane.distance == 1.0
        assert plane.confidence == 0.9

    def test_get_plane_equation(self):
        """Test plane equation extraction."""
        plane = PlaneEstimate(
            normal=np.array([1, 0, 0]),
            distance=2.0,
            inliers=np.array([0]),
            confidence=0.8
        )

        a, b, c, d = plane.get_plane_equation()
        assert a == 1
        assert b == 0
        assert c == 0
        assert d == 2.0

    def test_point_to_plane_distance(self):
        """Test distance computation."""
        # Plane z = 0
        plane = PlaneEstimate(
            normal=np.array([0, 0, 1]),
            distance=0.0,
            inliers=np.array([0]),
            confidence=0.9
        )

        points = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, -1]
        ])

        distances = plane.point_to_plane_distance(points)

        assert np.allclose(distances, [0, 1, -1])


class TestRANSACPlaneDetector:
    """Tests for RANSAC plane detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = RANSACPlaneDetector(
            n_iterations=500,
            distance_threshold=0.02,
            min_inliers=20
        )

        assert detector.n_iterations == 500
        assert detector.distance_threshold == 0.02
        assert detector.min_inliers == 20

    def test_detect_horizontal_plane(self):
        """Test detection of horizontal plane."""
        detector = RANSACPlaneDetector(
            n_iterations=100,
            distance_threshold=0.05,
            min_inliers=50
        )

        # Create points on z=0 plane with noise
        n_points = 200
        points = np.random.randn(n_points, 2) * 2
        z = np.random.randn(n_points) * 0.01  # Small noise
        points = np.column_stack([points, z])

        plane = detector.detect(points)

        assert plane is not None
        # Normal should be approximately [0, 0, 1] or [0, 0, -1]
        assert np.abs(plane.normal[2]) > 0.95
        assert plane.confidence > 0.5

    def test_detect_insufficient_points(self):
        """Test with insufficient points."""
        detector = RANSACPlaneDetector(min_inliers=100)

        points = np.random.randn(50, 3)
        plane = detector.detect(points)

        assert plane is None

    def test_detect_no_plane_in_random(self):
        """Test that random points may not form a good plane."""
        detector = RANSACPlaneDetector(
            n_iterations=50,
            distance_threshold=0.001,  # Very tight threshold
            min_inliers=150
        )

        # Random 3D points
        points = np.random.randn(200, 3) * 10

        plane = detector.detect(points)

        # May or may not detect a plane depending on random points
        # Just verify it doesn't crash
        assert plane is None or isinstance(plane, PlaneEstimate)


class TestMultiPlaneRANSAC:
    """Tests for multi-plane RANSAC detector."""

    def test_initialization(self):
        """Test initialization."""
        detector = MultiPlaneRANSAC(max_planes=4)
        assert detector.max_planes == 4

    def test_detect_multiple_planes(self):
        """Test detection of multiple planes."""
        detector = MultiPlaneRANSAC(
            n_iterations=100,
            distance_threshold=0.05,
            min_inliers=30,
            max_planes=3
        )

        # Create points on three orthogonal planes
        n_per_plane = 100

        # XY plane (z=0)
        xy_plane = np.column_stack([
            np.random.randn(n_per_plane, 2),
            np.random.randn(n_per_plane) * 0.01
        ])

        # XZ plane (y=2)
        xz_plane = np.column_stack([
            np.random.randn(n_per_plane),
            np.ones(n_per_plane) * 2 + np.random.randn(n_per_plane) * 0.01,
            np.random.randn(n_per_plane)
        ])

        # YZ plane (x=-2)
        yz_plane = np.column_stack([
            np.ones(n_per_plane) * -2 + np.random.randn(n_per_plane) * 0.01,
            np.random.randn(n_per_plane, 2)
        ])

        points = np.vstack([xy_plane, xz_plane, yz_plane])

        planes = detector.detect_all(points)

        assert len(planes) >= 2  # Should detect at least 2 planes
        assert len(planes) <= 3


class TestPlaneOrthogonality:
    """Tests for orthogonality computation."""

    def test_orthogonal_planes(self):
        """Test with perfectly orthogonal planes."""
        planes = [
            PlaneEstimate(normal=np.array([1, 0, 0]), distance=0, inliers=np.array([0]), confidence=0.9),
            PlaneEstimate(normal=np.array([0, 1, 0]), distance=0, inliers=np.array([0]), confidence=0.9),
            PlaneEstimate(normal=np.array([0, 0, 1]), distance=0, inliers=np.array([0]), confidence=0.9)
        ]

        score, pairs = compute_plane_orthogonality(planes, tolerance_degrees=5.0)

        assert len(pairs) == 3  # All three pairs are orthogonal
        assert score > 0.8

    def test_parallel_planes(self):
        """Test with parallel planes (no orthogonal pairs)."""
        planes = [
            PlaneEstimate(normal=np.array([1, 0, 0]), distance=0, inliers=np.array([0]), confidence=0.9),
            PlaneEstimate(normal=np.array([1, 0, 0]), distance=1, inliers=np.array([0]), confidence=0.9)
        ]

        score, pairs = compute_plane_orthogonality(planes)

        assert len(pairs) == 0


class TestPlaneParallelism:
    """Tests for parallelism computation."""

    def test_parallel_planes(self):
        """Test with parallel planes."""
        planes = [
            PlaneEstimate(normal=np.array([1, 0, 0]), distance=0, inliers=np.array([0]), confidence=0.9),
            PlaneEstimate(normal=np.array([1, 0, 0]), distance=1, inliers=np.array([0]), confidence=0.9)
        ]

        score, pairs = compute_plane_parallelism(planes, tolerance_degrees=5.0)

        assert len(pairs) == 1
        assert score > 0.8

    def test_opposite_normals(self):
        """Test parallel planes with opposite normals."""
        planes = [
            PlaneEstimate(normal=np.array([1, 0, 0]), distance=0, inliers=np.array([0]), confidence=0.9),
            PlaneEstimate(normal=np.array([-1, 0, 0]), distance=1, inliers=np.array([0]), confidence=0.9)
        ]

        score, pairs = compute_plane_parallelism(planes, tolerance_degrees=5.0)

        assert len(pairs) == 1  # Should detect as parallel


class TestBoxTopologyValidation:
    """Tests for box topology validation."""

    def test_valid_box_corner(self):
        """Test valid box corner (3 orthogonal planes)."""
        planes = [
            PlaneEstimate(normal=np.array([1, 0, 0]), distance=0, inliers=np.array([0]), confidence=0.9),
            PlaneEstimate(normal=np.array([0, 1, 0]), distance=0, inliers=np.array([0]), confidence=0.9),
            PlaneEstimate(normal=np.array([0, 0, 1]), distance=0, inliers=np.array([0]), confidence=0.9)
        ]

        is_valid, score, diagnostics = validate_box_topology(planes)

        assert is_valid is True
        assert score > 0.5
        assert 'orthogonality_score' in diagnostics

    def test_insufficient_planes(self):
        """Test with insufficient planes."""
        planes = [
            PlaneEstimate(normal=np.array([1, 0, 0]), distance=0, inliers=np.array([0]), confidence=0.9)
        ]

        is_valid, score, diagnostics = validate_box_topology(planes)

        assert is_valid is False
        assert score == 0.0
