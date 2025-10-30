#!/usr/bin/env python3
"""
Test if pycolmap actually uses GPU when we set device=Device.cuda
"""

import pycolmap
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
import time

# Create temp directory
temp_dir = Path(tempfile.mkdtemp())
images_dir = temp_dir / "images"
images_dir.mkdir()

# Create a simple test image
img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
cv2.imwrite(str(images_dir / "test1.jpg"), img)
cv2.imwrite(str(images_dir / "test2.jpg"), img)

database_path = temp_dir / "database.db"

print("Testing pycolmap with Device.cuda...")
print(f"Images: {images_dir}")
print(f"Database: {database_path}")

# Test with explicit CUDA device
sift_opts = pycolmap.SiftExtractionOptions()
sift_opts.use_gpu = True
sift_opts.gpu_index = "0"
sift_opts.max_num_features = 1000

print(f"\nSIFT Options:")
print(f"  use_gpu: {sift_opts.use_gpu}")
print(f"  gpu_index: {sift_opts.gpu_index}")

try:
    print("\n🎮 Attempting GPU feature extraction...")
    start = time.time()
    
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(images_dir),
        sift_options=sift_opts,
        device=pycolmap.Device.cuda  # Force CUDA
    )
    
    elapsed = time.time() - start
    print(f"✅ Feature extraction completed in {elapsed:.2f}s")
    print("\n⚠️  Check the logs above to see if it says 'Creating SIFT GPU' or 'Creating SIFT CPU'")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Cleanup
shutil.rmtree(temp_dir)
print(f"\n✅ Cleaned up temp directory")

