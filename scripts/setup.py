#!/usr/bin/env python3
"""
Setup script for 3D dimension measurement system
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8 or version.minor > 10:
        print("❌ Python 3.8-3.10 required")
        return False
    
    print("✅ Python version compatible")
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU available: {gpu_name}")
            return True
        else:
            print("⚠️  No GPU detected, using CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")
        return False

def install_dependencies():
    """Install Python dependencies"""
    commands = [
        #("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing main dependencies"),
        ("pip install -r server/requirements.txt", "Installing server dependencies")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def setup_dust3r():
    """Setup DUSt3R repository"""
    if not Path("dust3r").exists():
        success = run_command(
            "git clone --recursive https://github.com/naver/dust3r.git",
            "Cloning DUSt3R repository"
        )
        if not success:
            return False
    
    os.chdir("dust3r")
    success = run_command("pip install -e .", "Installing DUSt3R")
    os.chdir("..")
    
    return success is not None

def setup_mast3r():
    """Setup MASt3R repository"""
    if not Path("mast3r").exists():
        success = run_command(
            "git clone --recursive https://github.com/naver/mast3r.git",
            "Cloning MASt3R repository"
        )
        if not success:
            return False
    
    os.chdir("mast3r")
    success = run_command("pip install -e .", "Installing MASt3R")
    os.chdir("..")
    
    return success is not None

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/models",
        "data/test_images",
        "output",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Setup 3D dimension measurement system')
    parser.add_argument('--skip-models', action='store_true', help='Skip model download')
    parser.add_argument('--cpu-only', action='store_true', help='Setup for CPU-only inference')
    
    args = parser.parse_args()
    
    print("🚀 Setting up 3D Dimension Measurement System...")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Check GPU (optional)
    has_gpu = check_gpu()
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup DUSt3R
    if not setup_dust3r():
        print("❌ Failed to setup DUSt3R")
        sys.exit(1)
    
    # Setup MASt3R
    if not setup_mast3r():
        print("❌ Failed to setup MASt3R")
        sys.exit(1)
    
    # Download models
    if not args.skip_models:
        success = run_command("python scripts/download_models.py", "Downloading models")
        if not success:
            print("⚠️  Model download failed, but you can try again later")
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("   1. Test installation: python scripts/test_installation.py")
    print("   2. Run server: python server/main.py")
    print("   3. API docs: http://localhost:8000/docs")
    
    if has_gpu:
        print("\n🚀 GPU acceleration enabled for optimal performance!")
    else:
        print("\n⚠️  Running on CPU - consider GPU setup for better performance")

if __name__ == "__main__":
    main()
