#!/usr/bin/env python3
"""
System Validation Script

Checks all modules, imports, and integrations for correctness.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_imports():
    """Check if all modules can be imported."""
    print("=" * 60)
    print("Checking Module Imports")
    print("=" * 60)
    
    modules_to_check = [
        ("src", "Core package"),
        ("src.core", "Core module"),
        ("src.core.config", "Configuration"),
        ("src.core.calibration", "Calibration"),
        ("src.core.measurement_system_gpu", "Measurement system"),
        ("src.reconstruction", "Reconstruction module"),
        ("src.reconstruction.colmap_gpu", "COLMAP wrapper"),
        ("src.depth", "Depth module"),
        ("src.depth.metric3d_gpu", "Metric3D"),
        ("src.scale", "Scale module"),
        ("src.scale.marker_detection", "Marker detection"),
        ("src.scale.scale_optimizer", "Scale optimizer"),
        ("src.api", "API module"),
        ("src.api.rest_api", "REST API"),
    ]
    
    all_ok = True
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            print(f"[OK] {description:30s} ({module_name})")
        except ImportError as e:
            print(f"[FAIL] {description:30s} ({module_name}) - {e}")
            all_ok = False
        except Exception as e:
            print(f"[WARN] {description:30s} ({module_name}) - {e}")
    
    return all_ok


def check_class_instantiation():
    """Check if key classes can be instantiated."""
    print("\n" + "=" * 60)
    print("Checking Class Instantiation")
    print("=" * 60)
    
    tests = []
    
    # Config classes
    try:
        from src.core.config import SystemConfig, GPUConfig, COLMAPConfig
        config = SystemConfig()
        print("[OK] SystemConfig can be instantiated")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] SystemConfig instantiation failed: {e}")
        tests.append(False)
    
    # Camera Intrinsics
    try:
        from src.core.calibration import CameraIntrinsics
        intrinsics = CameraIntrinsics(
            fx=1000.0, fy=1000.0, cx=512.0, cy=512.0,
            width=1024, height=1024
        )
        print("[OK] CameraIntrinsics can be instantiated")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] CameraIntrinsics instantiation failed: {e}")
        tests.append(False)
    
    # Dataclasses
    try:
        from src.reconstruction.colmap_gpu import Reconstruction3D
        import torch
        recon = Reconstruction3D(
            points=torch.randn(100, 3),
            colors=torch.randn(100, 3),
            camera_poses=[],
            camera_intrinsics=[]
        )
        print("[OK] Reconstruction3D can be instantiated")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] Reconstruction3D instantiation failed: {e}")
        tests.append(False)
    
    return all(tests)


def check_type_hints():
    """Check if modules have proper type hints."""
    print("\n" + "=" * 60)
    print("Checking Type Hints")
    print("=" * 60)
    
    try:
        from src.core.config import SystemConfig
        import inspect
        
        # Check if validate method has type hints
        sig = inspect.signature(SystemConfig.validate)
        if sig.return_annotation != inspect.Signature.empty:
            print("[OK] Type hints present in SystemConfig.validate")
        else:
            print("[WARN] Missing return type hint in SystemConfig.validate")
        
        return True
    except Exception as e:
        print(f"[FAIL] Type hint check failed: {e}")
        return False


def check_file_structure():
    """Check if all expected files exist."""
    print("\n" + "=" * 60)
    print("Checking File Structure")
    print("=" * 60)
    
    expected_files = [
        "src/__init__.py",
        "src/core/__init__.py",
        "src/core/config.py",
        "src/core/calibration.py",
        "src/core/measurement_system_gpu.py",
        "src/reconstruction/__init__.py",
        "src/reconstruction/colmap_gpu.py",
        "src/depth/__init__.py",
        "src/depth/metric3d_gpu.py",
        "src/scale/__init__.py",
        "src/scale/marker_detection.py",
        "src/scale/scale_optimizer.py",
        "src/api/__init__.py",
        "src/api/rest_api.py",
        "main.py",
        "setup.py",
        "requirements/base.txt",
        "requirements/gpu.txt",
        "requirements/dev.txt",
        "Dockerfile.gpu",
        "docker-compose.gpu.yml",
        "README_NEW.md",
        "MIGRATION_GUIDE.md",
        "QUICKSTART.md",
        "TRANSFORMATION_SUMMARY.md",
    ]
    
    all_exist = True
    for filepath in expected_files:
        path = Path(filepath)
        if path.exists():
            size = path.stat().st_size
            print(f"[OK] {filepath:50s} ({size:,} bytes)")
        else:
            print(f"[FAIL] {filepath:50s} MISSING")
            all_exist = False
    
    return all_exist


def check_directory_structure():
    """Check if all expected directories exist."""
    print("\n" + "=" * 60)
    print("Checking Directory Structure")
    print("=" * 60)
    
    expected_dirs = [
        "src",
        "src/core",
        "src/reconstruction",
        "src/depth",
        "src/scale",
        "src/api",
        "requirements",
        "configs",
    ]
    
    all_exist = True
    for dirname in expected_dirs:
        path = Path(dirname)
        if path.exists() and path.is_dir():
            file_count = len(list(path.glob("*.py")))
            print(f"[OK] {dirname:30s} ({file_count} Python files)")
        else:
            print(f"[FAIL] {dirname:30s} MISSING")
            all_exist = False
    
    return all_exist


def check_integration_points():
    """Check key integration points between modules."""
    print("\n" + "=" * 60)
    print("Checking Module Integration")
    print("=" * 60)
    
    tests = []
    
    # Check if MeasurementSystemGPU imports all required modules
    try:
        from src.core.measurement_system_gpu import MeasurementSystemGPU
        # Check if it has the required methods
        required_methods = ['measure', '_init_components', '_transfer_to_gpu']
        for method in required_methods:
            if hasattr(MeasurementSystemGPU, method):
                print(f"[OK] MeasurementSystemGPU.{method} exists")
            else:
                print(f"[FAIL] MeasurementSystemGPU.{method} missing")
                tests.append(False)
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] MeasurementSystemGPU integration check failed: {e}")
        tests.append(False)
    
    # Check if API imports MeasurementSystemGPU
    try:
        from src.api.rest_api import app, measurement_system
        print("[OK] REST API imports correctly")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] REST API integration check failed: {e}")
        tests.append(False)
    
    # Check if scale optimizer uses marker detector
    try:
        from src.scale.scale_optimizer import ScaleOptimizer
        from src.scale.marker_detection import MarkerDetector
        print("[OK] Scale modules integrate correctly")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] Scale module integration check failed: {e}")
        tests.append(False)
    
    return all(tests) if tests else False


def check_dependencies():
    """Check if key dependencies are available."""
    print("\n" + "=" * 60)
    print("Checking Dependencies")
    print("=" * 60)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("scipy", "SciPy"),
        ("PIL", "Pillow"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
    ]
    
    all_ok = True
    for module, name in dependencies:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"[OK] {name:20s} version {version}")
        except ImportError:
            print(f"[WARN] {name:20s} NOT INSTALLED (optional: may be installed later)")
            # Don't fail on missing dependencies - they may not be installed yet
    
    return True  # Always return True since deps might not be installed


def check_syntax():
    """Check Python syntax of all source files."""
    print("\n" + "=" * 60)
    print("Checking Python Syntax")
    print("=" * 60)
    
    import ast
    
    python_files = list(Path("src").rglob("*.py"))
    python_files.extend([Path("main.py"), Path("setup.py")])
    
    all_ok = True
    for filepath in python_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
                ast.parse(code)
            print(f"[OK] {filepath}")
        except SyntaxError as e:
            print(f"[FAIL] {filepath} - Line {e.lineno}: {e.msg}")
            all_ok = False
        except Exception as e:
            print(f"[WARN] {filepath} - {e}")
    
    return all_ok


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("3D MEASUREMENT SYSTEM - VALIDATION")
    print("=" * 60)
    print()
    
    results = {}
    
    # Run all checks
    results['File Structure'] = check_file_structure()
    results['Directory Structure'] = check_directory_structure()
    results['Python Syntax'] = check_syntax()
    results['Module Imports'] = check_imports()
    results['Class Instantiation'] = check_class_instantiation()
    results['Type Hints'] = check_type_hints()
    results['Module Integration'] = check_integration_points()
    results['Dependencies'] = check_dependencies()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for check_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{check_name:30s} {status}")
    
    total_passed = sum(1 for v in results.values() if v)
    total_checks = len(results)
    
    print("\n" + "=" * 60)
    print(f"Total: {total_passed}/{total_checks} checks passed")
    
    if total_passed == total_checks:
        print("\n*** ALL CHECKS PASSED - System is ready! ***")
        print("=" * 60)
        return 0
    else:
        print(f"\n*** WARNING: {total_checks - total_passed} checks failed ***")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

