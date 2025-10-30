"""
Test script to check COLMAP GPU capabilities.
"""
import sys

print("=" * 60)
print("COLMAP GPU CAPABILITY TEST")
print("=" * 60)
print()

# Test 1: Check if pycolmap is installed
try:
    import pycolmap
    print("[OK] pycolmap is installed")
    print(f"     Version: {pycolmap.__version__}")
except ImportError as e:
    print("[ERROR] pycolmap is not installed")
    print(f"        Error: {e}")
    sys.exit(1)

print()

# Test 2: Check CUDA availability
try:
    has_cuda = pycolmap.has_cuda
    if has_cuda:
        print("[OK] pycolmap has CUDA support")
    else:
        print("[WARNING] pycolmap does NOT have CUDA support")
        print("          This means COLMAP will use CPU for SIFT feature extraction")
        print("          which is slower and less accurate.")
except AttributeError:
    print("[WARNING] Cannot determine CUDA support (old pycolmap version)")
    print("          pycolmap.has_cuda attribute not available")

print()

# Test 3: Check available SIFT options
try:
    print("Testing SIFT feature extractor options...")
    
    # Try to create a SIFT extractor config
    from pycolmap import SiftExtractionOptions
    options = SiftExtractionOptions()
    
    if hasattr(options, 'use_gpu'):
        print(f"[INFO] use_gpu option: {options.use_gpu}")
    
    if hasattr(options, 'gpu_index'):
        print(f"[INFO] gpu_index: {options.gpu_index}")
        
except Exception as e:
    print(f"[INFO] Could not test SIFT options: {e}")

print()
print("=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
print()

try:
    if not pycolmap.has_cuda:
        print("Your pycolmap installation does not have CUDA support.")
        print()
        print("TO FIX:")
        print("1. Uninstall current pycolmap:")
        print("   pip uninstall pycolmap")
        print()
        print("2. Install CUDA-enabled pycolmap:")
        print("   Option A: Try pre-built wheel (fastest)")
        print("   pip install pycolmap")
        print()
        print("   Option B: Build from source with CUDA (more reliable)")
        print("   https://github.com/colmap/pycolmap")
        print()
        print("NOTE: Building from source requires:")
        print("   - CMake")
        print("   - CUDA Toolkit 12.1")
        print("   - Visual Studio Build Tools")
    else:
        print("Your pycolmap has CUDA support - this is good!")
        print()
        print("If COLMAP is still using CPU SIFT, possible reasons:")
        print("1. Images are too large (reduce to max 1024px)")
        print("2. GPU memory is full (close other GPU applications)")
        print("3. COLMAP is choosing CPU due to small image set")
except:
    pass

print()
print("=" * 60)

