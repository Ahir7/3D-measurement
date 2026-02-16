#!/usr/bin/env python3
"""
Test runner for accuracy enhancement modules.

This script runs all unit tests for the new accuracy enhancement features:
- Model registry and adapters
- Uncertainty estimation
- Plane detection
- Prism fitting
- Geometric validator

Usage:
    python run_accuracy_tests.py [--verbose] [--quick]

Options:
    --verbose   Show detailed test output
    --quick     Run only fast tests (skip integration tests)
"""

import subprocess
import sys
import argparse
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import pytest
    except ImportError:
        missing.append("pytest")

    try:
        import scipy
    except ImportError:
        missing.append("scipy")

    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")

    return missing


def run_tests(verbose: bool = False, quick: bool = False) -> int:
    """
    Run accuracy enhancement tests.

    Args:
        verbose: Show detailed output
        quick: Skip slow tests

    Returns:
        Exit code (0 for success)
    """
    # Check dependencies first
    missing = check_dependencies()
    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install -r requirements/base.txt")
        print("  pip install -r requirements/gpu.txt")
        return 1

    # Define test modules
    accuracy_tests = [
        "tests/test_model_registry.py",
        "tests/test_uncertainty.py",
        "tests/test_plane_detection.py",
        "tests/test_prism_fitting.py",
        "tests/test_geometric_validator.py",
    ]

    integration_tests = [
        "tests/test_scale_fusion.py",
        "tests/test_config_validation.py",
    ]

    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Add test files
    cmd.extend(accuracy_tests)

    if not quick:
        cmd.extend(integration_tests)

    # Add markers for quick mode
    if quick:
        cmd.extend(["-m", "not slow"])

    # Show command
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    return result.returncode


def run_validation():
    """Run the validation script."""
    print("\n" + "=" * 60)
    print("Running implementation validation...")
    print("=" * 60)

    result = subprocess.run(
        ["python", "validate_accuracy_implementation.py"],
        cwd=Path(__file__).parent
    )

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run accuracy enhancement tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed test output"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run only fast tests"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation, not tests"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Accuracy Enhancement Test Suite")
    print("=" * 60)
    print()

    if args.validate_only:
        return run_validation()

    # Run tests
    test_result = run_tests(verbose=args.verbose, quick=args.quick)

    # Run validation
    val_result = run_validation()

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    if test_result == 0 and val_result == 0:
        print("All tests and validations passed!")
        return 0
    else:
        if test_result != 0:
            print("Some tests failed.")
        if val_result != 0:
            print("Validation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
