#!/usr/bin/env python3
"""
Validation script for depth-only accuracy enhancement implementation.

This script validates that all components are properly implemented
and integrated without requiring GPU or external dependencies.
"""

import sys
import numpy as np
from pathlib import Path


def check_import(module_path: str, items: list) -> bool:
    """Check if items can be imported from module."""
    try:
        module = __import__(module_path, fromlist=items)
        for item in items:
            if not hasattr(module, item):
                print(f"  MISSING: {item} in {module_path}")
                return False
        return True
    except ImportError as e:
        print(f"  IMPORT ERROR: {module_path} - {e}")
        return False


def validate_config_classes():
    """Validate configuration dataclasses."""
    print("\n1. Validating Configuration Classes...")

    try:
        from src.core.config import (
            SystemConfig, ModelSelectionConfig, UncertaintyConfig,
            GeometricPriorsConfig, Metric3DConfig
        )

        # Test ModelSelectionConfig
        model_config = ModelSelectionConfig(
            primary_model="dpt_large",
            fallback_model="midas_v3"
        )
        assert model_config.primary_model == "dpt_large"
        print("  ModelSelectionConfig: OK")

        # Test UncertaintyConfig
        unc_config = UncertaintyConfig(
            enable_mc_dropout=True,
            mc_dropout_passes=10
        )
        assert unc_config.mc_dropout_passes == 10
        print("  UncertaintyConfig: OK")

        # Test GeometricPriorsConfig
        geo_config = GeometricPriorsConfig(
            enable_plane_detection=True,
            ransac_iterations=1000
        )
        assert geo_config.ransac_iterations == 1000
        print("  GeometricPriorsConfig: OK")

        # Test nested config in Metric3DConfig
        metric_config = Metric3DConfig()
        assert hasattr(metric_config, 'model_selection')
        assert hasattr(metric_config, 'uncertainty')
        print("  Metric3DConfig nested configs: OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_model_registry():
    """Validate model registry system."""
    print("\n2. Validating Model Registry...")

    try:
        from src.depth.model_registry import (
            DepthModelAdapter, ModelRegistry, get_registry, register_model
        )

        # Test registry creation
        registry = ModelRegistry()
        assert registry is not None
        print("  ModelRegistry creation: OK")

        # Test global registry singleton
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2
        print("  Global registry singleton: OK")

        # Test adapter base class
        assert hasattr(DepthModelAdapter, 'load_model')
        assert hasattr(DepthModelAdapter, 'estimate_depth')
        print("  DepthModelAdapter interface: OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_uncertainty_module():
    """Validate uncertainty estimation module."""
    print("\n3. Validating Uncertainty Module...")

    try:
        from src.depth.uncertainty import (
            UncertaintyEstimate, MCDropoutEstimator,
            FlipConsistencyEstimator, UncertaintyFusion
        )
        from src.core.config import UncertaintyConfig

        # Test UncertaintyEstimate
        import torch
        combined = torch.rand(100, 100)
        estimate = UncertaintyEstimate(combined_uncertainty=combined)
        confidence = estimate.get_confidence()
        assert confidence.shape == (100, 100)
        print("  UncertaintyEstimate: OK")

        # Test MCDropoutEstimator
        mc_estimator = MCDropoutEstimator(n_passes=5)
        assert mc_estimator.n_passes == 5
        print("  MCDropoutEstimator: OK")

        # Test UncertaintyFusion
        fusion = UncertaintyFusion(method="weighted_average")
        uncertainties = {
            'a': torch.ones(50, 50) * 0.3,
            'b': torch.ones(50, 50) * 0.5
        }
        result = fusion.fuse(uncertainties)
        assert result.shape == (50, 50)
        print("  UncertaintyFusion: OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_plane_detection():
    """Validate plane detection module."""
    print("\n4. Validating Plane Detection...")

    try:
        from src.geometry.plane_detection import (
            PlaneEstimate, RANSACPlaneDetector, MultiPlaneRANSAC,
            compute_plane_orthogonality, validate_box_topology
        )

        # Test PlaneEstimate
        plane = PlaneEstimate(
            normal=np.array([0, 0, 1]),
            distance=1.0,
            inliers=np.array([0, 1, 2]),
            confidence=0.9
        )
        assert np.allclose(plane.normal, [0, 0, 1])
        print("  PlaneEstimate: OK")

        # Test RANSACPlaneDetector
        detector = RANSACPlaneDetector(n_iterations=100)
        assert detector.n_iterations == 100
        print("  RANSACPlaneDetector: OK")

        # Test plane detection on synthetic data
        n_points = 200
        xy = np.random.randn(n_points, 2)
        z = np.random.randn(n_points) * 0.01
        points = np.column_stack([xy, z])

        detected = detector.detect(points)
        assert detected is not None
        assert np.abs(detected.normal[2]) > 0.9
        print("  Plane detection on synthetic data: OK")

        # Test orthogonality computation
        planes = [
            PlaneEstimate(normal=np.array([1, 0, 0]), distance=0, inliers=np.array([0]), confidence=0.9),
            PlaneEstimate(normal=np.array([0, 1, 0]), distance=0, inliers=np.array([0]), confidence=0.9),
            PlaneEstimate(normal=np.array([0, 0, 1]), distance=0, inliers=np.array([0]), confidence=0.9)
        ]
        score, pairs = compute_plane_orthogonality(planes)
        assert len(pairs) == 3
        print("  Plane orthogonality: OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_prism_fitting():
    """Validate prism fitting module."""
    print("\n5. Validating Prism Fitting...")

    try:
        from src.geometry.prism_fitting import (
            PrismFit, RectangularPrismFitter, BoxConstraints
        )

        # Test PrismFit
        prism = PrismFit(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 3, 4]),
            rotation=np.eye(3),
            residual=0.01,
            inlier_ratio=0.95
        )
        assert prism.get_volume() == 24.0
        assert prism.corners is not None
        assert prism.corners.shape == (8, 3)
        print("  PrismFit: OK")

        # Test RectangularPrismFitter
        fitter = RectangularPrismFitter(max_iterations=50)

        # Create box-like point cloud
        points = []
        for _ in range(50):
            points.extend([
                [1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                [-1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                [np.random.uniform(-1, 1), 1, np.random.uniform(-1, 1)],
                [np.random.uniform(-1, 1), -1, np.random.uniform(-1, 1)],
                [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 1],
                [np.random.uniform(-1, 1), np.random.uniform(-1, 1), -1]
            ])
        points = np.array(points)

        fitted = fitter.fit(points)
        assert fitted.inlier_ratio > 0.5
        print("  RectangularPrismFitter: OK")

        # Test BoxConstraints
        constraints = BoxConstraints(orthogonality_tolerance=5.0)
        assert constraints is not None
        print("  BoxConstraints: OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_geometric_validator():
    """Validate geometric validator module."""
    print("\n6. Validating Geometric Validator...")

    try:
        from src.geometry.geometric_validator import (
            GeometricValidator, ValidationResult
        )
        from src.core.config import GeometricPriorsConfig

        # Test ValidationResult
        result = ValidationResult(
            is_valid=True,
            confidence_score=0.85
        )
        d = result.to_dict()
        assert d['is_valid'] is True
        assert d['confidence_score'] == 0.85
        print("  ValidationResult: OK")

        # Test GeometricValidator initialization
        config = GeometricPriorsConfig(
            enable_plane_detection=True,
            enable_prism_fitting=True,
            enable_epipolar_check=False
        )
        validator = GeometricValidator(config)
        assert validator.plane_detector is not None
        assert validator.prism_fitter is not None
        print("  GeometricValidator initialization: OK")

        # Test validation on synthetic data
        points = np.random.randn(200, 3)
        result = validator.validate_and_refine(points)
        assert isinstance(result, ValidationResult)
        print("  GeometricValidator validation: OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_geometry_utilities():
    """Validate extended geometry utilities."""
    print("\n7. Validating Geometry Utilities...")

    try:
        from src.utils.geometry import (
            fit_rectangular_prism, detect_planes_ransac,
            validate_box_topology, compute_geometric_confidence,
            BoundingBox
        )

        # Create test point cloud
        points = []
        for _ in range(50):
            points.extend([
                [1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                [-1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                [np.random.uniform(-1, 1), 1, np.random.uniform(-1, 1)],
                [np.random.uniform(-1, 1), -1, np.random.uniform(-1, 1)],
            ])
        points = np.array(points)

        # Test fit_rectangular_prism
        bbox, residual = fit_rectangular_prism(points, max_iterations=50)
        assert isinstance(bbox, BoundingBox)
        print("  fit_rectangular_prism: OK")

        # Test detect_planes_ransac
        planes = detect_planes_ransac(points, max_planes=3, min_inliers=20)
        assert isinstance(planes, list)
        print("  detect_planes_ransac: OK")

        # Test compute_geometric_confidence
        conf = compute_geometric_confidence(points, bbox)
        assert 0 <= conf <= 1
        print("  compute_geometric_confidence: OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_depth_estimation_extensions():
    """Validate depth estimation extensions."""
    print("\n8. Validating DepthEstimation Extensions...")

    try:
        import torch
        from src.depth.metric3d_gpu import DepthEstimation

        # Test extended DepthEstimation
        depth_map = torch.randn(480, 640)
        uncertainty_map = torch.rand(480, 640)
        mc_variance = torch.rand(480, 640)

        estimation = DepthEstimation(
            depth_map=depth_map,
            confidence_map=torch.rand(480, 640),
            uncertainty_map=uncertainty_map,
            mc_variance=mc_variance,
            flip_consistency=torch.rand(480, 640),
            model_name="dpt_large"
        )

        assert estimation.model_name == "dpt_large"
        assert estimation.uncertainty_map is not None
        print("  Extended DepthEstimation fields: OK")

        # Test to_dict with new fields
        d = estimation.to_dict()
        assert 'model_name' in d
        assert 'uncertainty_stats' in d
        print("  DepthEstimation.to_dict(): OK")

        # Test get_weighted_confidence
        weighted = estimation.get_weighted_confidence()
        assert weighted.shape == depth_map.shape
        print("  DepthEstimation.get_weighted_confidence(): OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_scale_optimizer_extensions():
    """Validate scale optimizer extensions."""
    print("\n9. Validating Scale Optimizer Extensions...")

    try:
        from src.scale.scale_optimizer import ScaleOptimizer, ScaleEstimate
        from src.core.config import ScaleRecoveryConfig

        # Check new methods exist
        assert hasattr(ScaleOptimizer, '_apply_geometric_priors')
        assert hasattr(ScaleOptimizer, '_validate_with_planes')
        assert hasattr(ScaleOptimizer, 'recover_scale_with_geometric_priors')
        print("  Scale optimizer new methods: OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_measurement_system_extensions():
    """Validate measurement system extensions."""
    print("\n10. Validating Measurement System Extensions...")

    try:
        from src.core.measurement_system_gpu import MeasurementResult

        # Check new fields exist in MeasurementResult
        import inspect
        sig = inspect.signature(MeasurementResult.__init__)
        params = list(sig.parameters.keys())

        assert 'uncertainty_bounds' in params or hasattr(MeasurementResult, '__dataclass_fields__')

        # Check dataclass fields
        fields = MeasurementResult.__dataclass_fields__
        assert 'uncertainty_bounds' in fields
        assert 'geometric_fit_score' in fields
        assert 'plane_detections' in fields
        assert 'model_used' in fields
        print("  MeasurementResult new fields: OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_domain_data_infrastructure():
    """Validate domain data infrastructure."""
    print("\n11. Validating Domain Data Infrastructure...")

    try:
        from src.data.synthetic_pipeline import (
            SyntheticDataGenerator, DomainRandomization, SyntheticScene
        )
        from src.training.fine_tuning import (
            FineTuningTrainer, FineTuningConfig
        )

        # Test SyntheticDataGenerator
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SyntheticDataGenerator(Path(tmpdir))
            scene = generator.generate_box_scene(
                dimensions=(0.2, 0.3, 0.4),
                num_views=5
            )
            assert len(scene.camera_poses) == 5
            print("  SyntheticDataGenerator: OK")

        # Test DomainRandomization
        randomizer = DomainRandomization()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        augmented = randomizer.randomize_lighting(image)
        assert augmented.shape == image.shape
        print("  DomainRandomization: OK")

        # Test FineTuningConfig
        config = FineTuningConfig(
            learning_rate=1e-4,
            head_only=True
        )
        assert config.head_only is True
        print("  FineTuningConfig: OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_enhanced_config():
    """Validate enhanced accuracy configuration."""
    print("\n12. Validating Enhanced Configuration...")

    try:
        from configs.enhanced_accuracy_config import (
            get_enhanced_config, get_fast_config, get_memory_constrained_config
        )

        # Test enhanced config
        config = get_enhanced_config()
        assert config.metric3d.uncertainty.enable_mc_dropout is True
        assert config.geometric_priors.enable_plane_detection is True
        print("  get_enhanced_config(): OK")

        # Test fast config
        fast_config = get_fast_config()
        assert fast_config.metric3d.uncertainty.enable_mc_dropout is False
        print("  get_fast_config(): OK")

        # Test memory constrained config
        mem_config = get_memory_constrained_config()
        assert mem_config.batch_size < config.batch_size
        print("  get_memory_constrained_config(): OK")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Depth-Only Accuracy Enhancement Implementation Validation")
    print("=" * 60)

    results = []

    results.append(("Configuration Classes", validate_config_classes()))
    results.append(("Model Registry", validate_model_registry()))
    results.append(("Uncertainty Module", validate_uncertainty_module()))
    results.append(("Plane Detection", validate_plane_detection()))
    results.append(("Prism Fitting", validate_prism_fitting()))
    results.append(("Geometric Validator", validate_geometric_validator()))
    results.append(("Geometry Utilities", validate_geometry_utilities()))
    results.append(("DepthEstimation Extensions", validate_depth_estimation_extensions()))
    results.append(("Scale Optimizer Extensions", validate_scale_optimizer_extensions()))
    results.append(("Measurement System Extensions", validate_measurement_system_extensions()))
    results.append(("Domain Data Infrastructure", validate_domain_data_infrastructure()))
    results.append(("Enhanced Configuration", validate_enhanced_config()))

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\nAll validations PASSED!")
        return 0
    else:
        print(f"\nWARNING: {total - passed} validation(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
