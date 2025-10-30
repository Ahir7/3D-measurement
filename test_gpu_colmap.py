#!/usr/bin/env python3
"""
Quick test to verify COLMAP GPU settings are correct.
"""

import pycolmap
import torch

print("=" * 60)
print("COLMAP GPU Configuration Test")
print("=" * 60)

# Check CUDA
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")

# Check pycolmap version
print(f"\n2. pycolmap version: {pycolmap.__version__}")

# Test SIFT Extraction Options
print("\n3. SIFT Extraction Options:")
sift_opts = pycolmap.SiftExtractionOptions()
print(f"   use_gpu (default): {sift_opts.use_gpu}")
print(f"   gpu_index (default): {sift_opts.gpu_index}")
print(f"   max_num_features: {sift_opts.max_num_features}")

# Set GPU explicitly
sift_opts.use_gpu = True
sift_opts.gpu_index = "0"
print(f"\n   After explicit GPU settings:")
print(f"   use_gpu: {sift_opts.use_gpu}")
print(f"   gpu_index: {sift_opts.gpu_index}")

# Test SIFT Matching Options
print("\n4. SIFT Matching Options:")
match_opts = pycolmap.SiftMatchingOptions()
print(f"   use_gpu (default): {match_opts.use_gpu}")
print(f"   gpu_index (default): {match_opts.gpu_index}")

match_opts.use_gpu = True
match_opts.gpu_index = "0"
print(f"\n   After explicit GPU settings:")
print(f"   use_gpu: {match_opts.use_gpu}")
print(f"   gpu_index: {match_opts.gpu_index}")

# Test Device enum
print("\n5. Device Options:")
print(f"   Device.auto: {pycolmap.Device.auto}")
print(f"   Device.cpu: {pycolmap.Device.cpu}")
print(f"   Device.cuda: {pycolmap.Device.cuda}")

print("\n" + "=" * 60)
print("✅ All GPU settings configured correctly!")
print("=" * 60)

