import torch
import sys

def test_installation():
    print("🔍 Testing installation...")
    
    # Test PyTorch CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test tensor operations
        x = torch.randn(1000, 1000).cuda()
        y = torch.mm(x, x.t())
        print("✅ GPU tensor operations working")
    else:
        print("❌ CUDA not available")
        return False
    
    # Test imports
    try:
        sys.path.append("mast3r")
        sys.path.append("dust3r") 
        from mast3r.model import AsymmetricMASt3R
        print("✅ MASt3R import successful")
    except ImportError as e:
        print(f"❌ MASt3R import failed: {e}")
        return False
    
    print("🎉 All tests passed!")
    return True

if __name__ == "__main__":
    test_installation()
