#!/usr/bin/env python3
"""
Setup script for 3D Measurement System.

This script:
1. Checks system requirements
2. Verifies GPU availability
3. Installs dependencies
4. Downloads models (optional)
5. Runs tests (optional)
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or not (8 <= version.minor <= 10):
        print("❌ Python 3.8-3.10 required")
        return False
    
    print("✅ Python version compatible")
    return True


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"✅ GPU available: {gpu_name}")
            print(f"✅ CUDA version: {cuda_version}")
            return True
        else:
            print("❌ No GPU detected")
            print("⚠️  This system requires a CUDA-capable GPU")
            return False
    
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return None  # Unknown


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements/base.txt", "Installing base dependencies"),
        ("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", 
         "Installing PyTorch with CUDA 12.1"),
        ("pip install -r requirements/gpu.txt", "Installing GPU dependencies")
    ]
    
    for command, description in commands:
        print(f"\n{description}...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed: {description}")
            print(result.stderr)
            return False
        else:
            print(f"✅ {description} completed")
    
    return True


def verify_installation():
    """Verify installation by checking imports."""
    print("\n🔍 Verifying installation...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic")
    ]
    
    all_ok = True
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Missing")
            all_ok = False
    
    return all_ok


def run_tests():
    """Run basic tests."""
    print("\n🧪 Running basic tests...")
    
    result = subprocess.run(
        [sys.executable, "main.py", "info"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Basic test passed")
        print(result.stdout)
        return True
    else:
        print("❌ Basic test failed")
        print(result.stderr)
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "output",
        "logs",
        "data/models",
        "configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True


def main():
    """Main setup routine."""
    print("=" * 60)
    print("3D Measurement System - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("\n❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check GPU
    gpu_available = check_gpu()
    if gpu_available is False:
        print("\n⚠️  Warning: No GPU detected")
        print("This system requires a CUDA-capable GPU for operation.")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed")
        sys.exit(1)
    
    # Run tests
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    
    if run_tests():
        print("\n✅ System is ready to use!")
        print("\n🚀 Quick start:")
        print("   python main.py serve          # Start API server")
        print("   python main.py measure img*.jpg  # Measure from CLI")
        print("   python main.py info           # Show system info")
    else:
        print("\n⚠️  Setup complete but tests failed")
        print("Please check the error messages above")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

