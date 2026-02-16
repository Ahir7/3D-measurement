"""
Unit tests for geometric validator module.
"""

import pytest
import torch
import numpy as np

from src.geometry.geometric_validator import (
    GeometricValidator,
    ValidationResult
)
from src.core.config import GeometricPriorsConfig


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = ValidationResult(
            is_valid=True,
            confidence_score=0.85
        )

        assert result.is_valid is True
        assert result.confidence_score == 0.85
        assert result.refined_bbox is None
        assert result.plane_detections is None

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ValidationResult(
            is_valid=True,
            confidence_score=0.9,
            diagnostics={'test': 'value'}
        )

        d = result.to_dict()

        assert d['is_valid'] is True
        assert d['confidence_score'] == 0.9
        assert 'diagnostics' in d

    def test_to_dict_with_prism(self):
        """Test dictionary conversion with prism fit."""
        from src.geometry.prism_fitting import PrismFit

        prism = PrismFit(
            center=np.array([0, 0, 0]),
            dimensions=np.array([1, 2, 3]),
            rotation=np.eye(3),
            residual=0.01,
            inlier_ratio=0.95
        )

        result = ValidationResult(
            is_valid=True,
            confidence_score=0.9,
            prism_fit=prism
        )

        d = result.to_dict()

        assert 'prism_fit' in d
        assert d['prism_fit']['residual'] == 0.01


class TestGeometricValidator:
    """Tests for GeometricValidator."""

    def test_initialization(self):
        """Test validator initialization."""
        config = GeometricPriorsConfig(
            enable_plane_detection=True,
            enable_prism_fitting=True,
            enable_epipolar_check=False
        )

        validator = GeometricValidator(config)

        assert validator.plane_detector is not None
        assert validator.prism_fitter is not None
        assert validator.epipolar_checker is None

    def test_initialization_all_disabled(self):
        """Test with all validation disabled."""
        config = GeometricPriorsConfig(
            enable_plane_detection=False,
            enable_prism_fitting=False,
            enable_epipolar_check=False,
            enable_box_topology=False
        )

        validator = GeometricValidator(config)

        assert validator.plane_detector is None
        assert validator.prism_fitter is None
        assert validator.epipolar_checker is None
        assert validator.box_constraints is None

    def test_validate_simple_box(self):
        """Test validation with simple box point cloud."""
        config = GeometricPriorsConfig(
            enable_plane_detection=True,
            enable_prism_fitting=True,
            enable_epipolar_check=False,
            ransac_iterations=100,
            min_plane_points=20
        )

        validator = GeometricValidator(config)

        # Create simple box point cloud
        points = []
        n_per_face = 50

        # Generate points on box faces
        for _ in range(n_per_face):
            # +X face
            points.append([1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            # -X face
            points.append([-1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            # +Y face
            points.append([np.random.uniform(-1, 1), 1, np.random.uniform(-1, 1)])
            # -Y face
            points.append([np.random.uniform(-1, 1), -1, np.random.uniform(-1, 1)])
            # +Z face
            points.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 1])
            # -Z face
            points.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1), -1])

        points = np.array(points)

        result = validator.validate_and_refine(points)

        assert isinstance(result, ValidationResult)
        assert result.confidence_score >= 0
        assert result.confidence_score <= 1
        assert 'num_points' in result.diagnostics

    def test_validate_insufficient_points(self):
        """Test validation with insufficient points."""
        config = GeometricPriorsConfig(
            enable_plane_detection=True,
            min_plane_points=100  # High threshold
        )

        validator = GeometricValidator(config)

        # Very few points
        points = np.random.randn(50, 3)

        result = validator.validate_and_refine(points)

        # Should still return a result, but may have low confidence
        assert isinstance(result, ValidationResult)

    def test_get_dimension_corrections(self):
        """Test dimension correction computation."""
        config = GeometricPriorsConfig()
        validator = GeometricValidator(config)

        # Create mock validation result with refined bbox
        from src.utils.geometry import BoundingBox

        refined_bbox = BoundingBox(
            width=0.22,  # 22 cm
            height=0.18,  # 18 cm
            depth=0.12,  # 12 cm
            volume=0.22 * 0.18 * 0.12,
            center=np.array([0, 0, 0]),
            orientation=np.eye(3)
        )

        result = ValidationResult(
            is_valid=True,
            confidence_score=0.9,
            refined_bbox=refined_bbox
        )

        initial_measurements = {
            'width': 20.0,  # 20 cm (vs 22 cm refined)
            'height': 18.0,  # Same
            'depth': 10.0   # 10 cm (vs 12 cm refined)
        }

        corrections = validator.get_dimension_corrections(
            initial_measurements,
            result
        )

        assert 'width_factor' in corrections
        assert 'height_factor' in corrections
        assert 'depth_factor' in corrections

        # Corrections should be limited to reasonable range
        assert 0.8 <= corrections['width_factor'] <= 1.2
        assert 0.8 <= corrections['depth_factor'] <= 1.2

    def test_estimate_geometric_uncertainty(self):
        """Test geometric uncertainty estimation."""
        config = GeometricPriorsConfig()
        validator = GeometricValidator(config)

        result = ValidationResult(
            is_valid=True,
            confidence_score=0.9,
            diagnostics={
                'prism_residual': 0.01,
                'is_box_topology': True
            }
        )

        uncertainty = validator.estimate_geometric_uncertainty(result)

        assert 'overall' in uncertainty
        assert 'width' in uncertainty
        assert 'height' in uncertainty
        assert 'depth' in uncertainty

        # All uncertainties should be positive
        for key, value in uncertainty.items():
            assert value >= 0

    def test_validate_with_depth_maps(self):
        """Test validation with depth maps (no epipolar checker)."""
        config = GeometricPriorsConfig(
            enable_plane_detection=True,
            enable_prism_fitting=True,
            enable_epipolar_check=False  # Disabled for this test
        )

        validator = GeometricValidator(config)

        # Create simple point cloud
        points = np.random.randn(200, 3)

        # Create mock depth maps
        depth_maps = torch.rand(3, 100, 100)

        result = validator.validate_and_refine(
            points,
            depth_maps=depth_maps,
            camera_poses=None,
            camera_intrinsics=None
        )

        assert isinstance(result, ValidationResult)


class TestGeometricValidatorEdgeCases:
    """Edge case tests for GeometricValidator."""

    def test_empty_point_cloud(self):
        """Test with empty point cloud."""
        config = GeometricPriorsConfig(
            enable_prism_fitting=True,
            enable_plane_detection=True
        )
        validator = GeometricValidator(config)

        # Empty array
        points = np.array([]).reshape(0, 3)

        # Should handle gracefully
        result = validator.validate_and_refine(points)
        assert result.confidence_score <= 0.5

    def test_collinear_points(self):
        """Test with collinear points (degenerate case)."""
        config = GeometricPriorsConfig(
            enable_prism_fitting=True,
            min_plane_points=10
        )
        validator = GeometricValidator(config)

        # Points on a line
        t = np.linspace(0, 1, 100)
        points = np.column_stack([t, t, t])

        result = validator.validate_and_refine(points)

        # Should have low confidence for degenerate case
        assert isinstance(result, ValidationResult)

    def test_planar_points(self):
        """Test with planar points (single plane)."""
        config = GeometricPriorsConfig(
            enable_plane_detection=True,
            enable_prism_fitting=True
        )
        validator = GeometricValidator(config)

        # Points on XY plane
        points = np.column_stack([
            np.random.randn(200),
            np.random.randn(200),
            np.zeros(200)
        ])

        result = validator.validate_and_refine(points)

        assert isinstance(result, ValidationResult)
        # Should detect at least one plane
        if result.plane_detections:
            assert len(result.plane_detections) >= 1
