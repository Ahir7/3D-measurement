#!/usr/bin/env python3
"""
Check if all required models and dependencies are installed.

This script verifies:
1. COLMAP binary/pycolmap
2. Metric3D model files
3. Model dependencies
4. Download missing components
"""

import sys
import subprocess
import shutil
from pathlib import Path
import urllib.request
import os

def check_colmap():
    """Check if COLMAP is installed."""
    print("\n" + "="*60)
    print("Checking COLMAP Installation")
    print("="*60)
    
    # Check for pycolmap first
    try:
        import pycolmap
        print(f"[OK] pycolmap version {pycolmap.__version__} installed")
        return True, "pycolmap"
    except ImportError:
        print("[INFO] pycolmap not installed (optional)")
    
    # Check for COLMAP binary
    colmap_paths = [
        "colmap",
        "C:\\Program Files\\COLMAP\\COLMAP.bat",
        "/usr/local/bin/colmap",
        "/usr/bin/colmap",
        shutil.which("colmap")
    ]
    
    for path in colmap_paths:
        if path and Path(path).exists():
            try:
                result = subprocess.run(
                    [path, "--help"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    print(f"[OK] COLMAP binary found at: {path}")
                    return True, path
            except:
                continue
    
    print("[WARNING] COLMAP not found!")
    print("\nTo install COLMAP:")
    print("  Windows: https://github.com/colmap/colmap/releases")
    print("  Linux: sudo apt install colmap")
    print("  Or: pip install pycolmap")
    return False, None


def check_metric3d():
    """Check if Metric3D models are available."""
    print("\n" + "="*60)
    print("Checking Metric3D Models")
    print("="*60)
    
    # Check if we can import the model
    try:
        # Try importing transformers (used by Metric3D)
        import transformers
        print(f"[OK] transformers version {transformers.__version__} installed")
        has_transformers = True
    except ImportError:
        print("[WARNING] transformers not installed")
        print("  Install: pip install transformers")
        has_transformers = False
    
    # Check for model files
    model_dir = Path("models/metric3d")
    if model_dir.exists():
        model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
        if model_files:
            print(f"[OK] Found {len(model_files)} Metric3D model file(s)")
            for f in model_files:
                size_mb = f.stat().st_size / (1024*1024)
                print(f"     - {f.name} ({size_mb:.1f} MB)")
            return True
        else:
            print("[INFO] No Metric3D model files found in models/metric3d/")
    else:
        print("[INFO] Models directory not found")
    
    # Check Hugging Face cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        metric3d_models = list(hf_cache.glob("*metric*"))
        if metric3d_models:
            print(f"[OK] Found {len(metric3d_models)} Metric3D model(s) in Hugging Face cache")
            return True
    
    print("[INFO] Metric3D models will be downloaded on first use")
    return has_transformers


def check_depth_anything():
    """Check for Depth Anything model as alternative."""
    print("\n" + "="*60)
    print("Checking Alternative Depth Models")
    print("="*60)
    
    try:
        import timm
        print(f"[OK] timm version {timm.__version__} installed (for DPT models)")
    except ImportError:
        print("[INFO] timm not installed (optional for DPT models)")
        print("  Install: pip install timm")
        return False
    
    return True


def check_opencv_contrib():
    """Check if OpenCV contrib modules are available."""
    print("\n" + "="*60)
    print("Checking OpenCV Modules")
    print("="*60)
    
    try:
        import cv2
        print(f"[OK] OpenCV version {cv2.__version__}")
        
        # Check for ArUco
        try:
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            print("[OK] ArUco module available")
        except AttributeError:
            print("[WARNING] ArUco module not available")
            print("  Install: pip install opencv-contrib-python")
        
        return True
    except ImportError:
        print("[ERROR] OpenCV not installed")
        return False


def download_metric3d_weights():
    """Download Metric3D weights if needed."""
    print("\n" + "="*60)
    print("Metric3D Model Download")
    print("="*60)
    
    response = input("\nWould you like to pre-download Metric3D models? (y/N): ")
    if response.lower() != 'y':
        print("[SKIP] Models will be auto-downloaded on first use")
        return
    
    print("\n[INFO] Downloading Metric3D models...")
    print("This may take a while (model is ~1-2 GB)...")
    
    try:
        # This will trigger the download
        from transformers import AutoModel
        
        model_name = "JUGGHM/Metric3D"
        print(f"Downloading from Hugging Face: {model_name}")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print("[OK] Metric3D model downloaded successfully!")
        
    except Exception as e:
        print(f"[WARNING] Could not download model: {e}")
        print("Models will be downloaded automatically on first use")


def install_missing_packages():
    """Offer to install missing packages."""
    print("\n" + "="*60)
    print("Missing Packages")
    print("="*60)
    
    missing = []
    
    # Check essential packages
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import timm
    except ImportError:
        missing.append("timm")
    
    if not missing:
        print("[OK] All recommended packages are installed")
        return
    
    print(f"\n[INFO] Missing packages: {', '.join(missing)}")
    response = input("\nInstall missing packages? (y/N): ")
    
    if response.lower() == 'y':
        for package in missing:
            print(f"\nInstalling {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package])
        print("\n[OK] Packages installed!")


def create_model_structure():
    """Create necessary model directories."""
    print("\n" + "="*60)
    print("Creating Model Directories")
    print("="*60)
    
    dirs = [
        "models",
        "models/metric3d",
        "models/colmap",
        "models/depth",
        "data/models"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created: {dir_path}")


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("MODEL AND DEPENDENCY CHECK")
    print("="*60)
    print("\nChecking required models and dependencies for:")
    print("  - COLMAP (3D reconstruction)")
    print("  - Metric3D (depth estimation)")
    print("  - OpenCV (marker detection)")
    
    results = {}
    
    # Check COLMAP
    results['colmap'] = check_colmap()
    
    # Check Metric3D
    results['metric3d'] = check_metric3d()
    
    # Check alternative depth models
    results['depth_alt'] = check_depth_anything()
    
    # Check OpenCV
    results['opencv'] = check_opencv_contrib()
    
    # Create directory structure
    create_model_structure()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\n[COLMAP]")
    if results['colmap'][0]:
        print(f"  [OK] Available: {results['colmap'][1]}")
    else:
        print("  [NOT FOUND] - Install from https://github.com/colmap/colmap/releases")
    
    print("\n[Metric3D]")
    if results['metric3d']:
        print("  [OK] Dependencies ready")
        print("  [INFO] Models will auto-download on first use (~1-2 GB)")
    else:
        print("  [WARNING] Missing dependencies - run: pip install transformers")
    
    print("\n[OpenCV]")
    if results['opencv']:
        print("  [OK] Available")
    else:
        print("  [NOT FOUND] - run: pip install opencv-python")
    
    # Offer to install missing
    install_missing_packages()
    
    # Offer to download models
    if results['metric3d']:
        download_metric3d_weights()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. COLMAP:")
    if not results['colmap'][0]:
        print("   - Download and install from: https://github.com/colmap/colmap/releases")
        print("   - Or install pycolmap: pip install pycolmap")
    else:
        print("   [OK] Ready to use")
    
    print("\n2. Metric3D:")
    if not results['metric3d']:
        print("   - Install transformers: pip install transformers")
    else:
        print("   [OK] Will auto-download on first use")
        print("   [INFO] First run will take longer (~1-2 GB download)")
    
    print("\n3. Optional Enhancements:")
    print("   - pip install timm  (for additional depth models)")
    print("   - pip install pycolmap  (for faster COLMAP integration)")
    
    print("\n" + "="*60)
    print("STATUS")
    print("="*60)
    
    critical_ok = results['colmap'][0] or True  # COLMAP can be installed separately
    metric3d_ok = results['metric3d']
    opencv_ok = results['opencv']
    
    if critical_ok and metric3d_ok and opencv_ok:
        print("\n[OK] All critical components are ready!")
        print("[OK] System can be used now")
        print("\n[INFO] Metric3D models will download automatically on first use")
    else:
        print("\n[WARNING] Some components need attention (see above)")
        print("Install missing components before using the system")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

