#!/usr/bin/env python3
"""
Download pre-trained models for 3D dimension measurement
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path
from tqdm import tqdm

# Model configurations
MODELS = {
    "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth": {
        "url": "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "size": 1300000000,  # ~1.3GB
        "sha256": None  # Add if available
    },
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth": {
        "url": "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "size": 1600000000,  # ~1.6GB
        "sha256": None  # Add if available
    }
}

class DownloadProgress:
    def __init__(self, filename):
        self.filename = filename
        self.pbar = None
    
    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=f"Downloading {self.filename}"
            )
        
        self.pbar.update(block_size)
        
        if block_num * block_size >= total_size:
            self.pbar.close()

def verify_checksum(filepath, expected_sha256):
    """Verify file checksum"""
    if not expected_sha256:
        return True
    
    print(f"🔍 Verifying checksum for {filepath.name}...")
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    actual_sha256 = sha256_hash.hexdigest()
    
    if actual_sha256 == expected_sha256:
        print("✅ Checksum verified")
        return True
    else:
        print(f"❌ Checksum mismatch: expected {expected_sha256}, got {actual_sha256}")
        return False

def download_model(model_name, model_info, models_dir):
    """Download a single model"""
    filepath = models_dir / model_name
    
    # Check if already exists
    if filepath.exists():
        if filepath.stat().st_size == model_info["size"]:
            print(f"✅ {model_name} already exists and appears complete")
            return True
        else:
            print(f"⚠️  {model_name} exists but size mismatch, re-downloading...")
            filepath.unlink()
    
    print(f"📥 Downloading {model_name}...")
    print(f"    URL: {model_info['url']}")
    print(f"    Size: {model_info['size'] / 1e9:.1f} GB")
    
    try:
        # Create progress callback
        progress = DownloadProgress(model_name)
        
        # Download with progress bar
        urllib.request.urlretrieve(
            model_info["url"],
            filepath,
            reporthook=progress
        )
        
        # Verify size
        actual_size = filepath.stat().st_size
        expected_size = model_info["size"]
        
        if abs(actual_size - expected_size) > expected_size * 0.01:  # 1% tolerance
            print(f"❌ Size mismatch: expected {expected_size}, got {actual_size}")
            filepath.unlink()
            return False
        
        # Verify checksum if provided
        if not verify_checksum(filepath, model_info.get("sha256")):
            filepath.unlink()
            return False
        
        print(f"✅ Successfully downloaded {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False

def main():
    print("📦 Model Download Script for 3D Dimension Measurement")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check available space
    statvfs = os.statvfs(models_dir)
    free_space = statvfs.f_frsize * statvfs.f_available
    required_space = sum(model["size"] for model in MODELS.values())
    
    print(f"💾 Available space: {free_space / 1e9:.1f} GB")
    print(f"💾 Required space: {required_space / 1e9:.1f} GB")
    
    if free_space < required_space * 1.1:  # 10% buffer
        print("❌ Insufficient disk space")
        sys.exit(1)
    
    # Download models
    success_count = 0
    total_count = len(MODELS)
    
    for model_name, model_info in MODELS.items():
        if download_model(model_name, model_info, models_dir):
            success_count += 1
        print()  # Empty line for readability
    
    # Summary
    print("=" * 60)
    print(f"📊 Download Summary: {success_count}/{total_count} models downloaded")
    
    if success_count == total_count:
        print("✅ All models downloaded successfully!")
        print("\n🎯 Next steps:")
        print("   1. Test installation: python scripts/test_installation.py")
        print("   2. Run server: python server/main.py")
    else:
        print("❌ Some models failed to download")
        print("   Please check your internet connection and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
