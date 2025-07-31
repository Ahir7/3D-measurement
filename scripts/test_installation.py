#!/usr/bin/env python3
"""
Test installation of 3D dimension measurement system
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image

def test_python_version():
    """Test Python version"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    
    if version.major == 3 and 8 <= version.minor <= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Unsupported")
        return False

def test_core_dependencies():
    """Test core Python dependencies"""
    print("\n📦 Testing core dependencies...")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("scipy", "SciPy"),
        ("fastapi", "FastAPI")
    ]
    
    success = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - Missing")
            success = False
    
    return success

def test_gpu():
    """Test GPU availability"""
    print("\n🚀 Testing GPU...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✅ GPU available: {gpu_name}")
        print(f"   Count: {gpu_count}")
        print(f"   Memory: {memory:.1f} GB")
        
        # Test GPU operations
        try:
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.mm(test_tensor, test_tensor)
            print("✅ GPU operations - OK")
            return True
        except Exception as e:
            print(f"❌ GPU operations failed: {e}")
            return False
    else:
        print("⚠️  GPU not available, using CPU")
        return False

def test_dust3r_import():
    """Test DUSt3R imports"""
    print("\n🌪️  Testing DUSt3R imports...")
    
    sys.path.append(str(Path("dust3r")))
    
    try:
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        print("✅ DUSt3R imports - OK")
        return True
    except ImportError as e:
        print(f"❌ DUSt3R imports failed: {e}")
        return False

def test_mast3r_import():
    """Test MASt3R imports"""
    print("\n🎭 Testing MASt3R imports...")
    
    sys.path.append(str(Path("mast3r")))
    
    try:
        from mast3r.model import AsymmetricMASt3R
        print("✅ MASt3R imports - OK")
        return True
    except ImportError as e:
        print(f"❌ MASt3R imports failed: {e}")
        return False

def test_models():
    """Test model files"""
    print("\n🧠 Testing model files...")
    
    models_dir = Path("data/models")
    
    expected_models = [
        "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    ]
    
    success = True
    for model_name in expected_models:
        model_path = models_dir / model_name
        if model_path.exists():
            size = model_path.stat().st_size / 1e9
            print(f"✅ {model_name} - {size:.1f} GB")
        else:
            print(f"❌ {model_name} - Missing")
            success = False
    
    return success

def test_server_import():
    """Test server module imports"""
    print("\n🖥️  Testing server imports...")
    
    try:
        from server.main import app, DUSt3RDimensionCalculator
        print("✅ Server main - OK")
        
        from server.models.scale_recovery import MultiMethodScaleRecovery  
        print("✅ Scale recovery - OK")
        
        return True
    except ImportError as e:
        print(f"❌ Server imports failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with dummy data"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Create test images
        test_images = []
        for i in range(3):
            # Create random test image
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                img.save(tmp.name)
                test_images.append(tmp.name)
        
        # Test image loading
        sys.path.append(str(Path("dust3r")))
        from dust3r.utils.image import load_images
        
        loaded_images = load_images(test_images, size=256)
        print(f"✅ Loaded {len(loaded_images)} test images")
        
        # Cleanup
        for img_path in test_images:
            os.unlink(img_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_api_server():
    """Test API server startup"""
    print("\n🌐 Testing API server...")
    
    try:
        from fastapi.testclient import TestClient
        from server.main import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API server - OK")
            print(f"   Status: {data.get('status')}")
            print(f"   GPU: {data.get('gpu_available')}")
            return True
        else:
            print(f"❌ API server returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API server test failed: {e}")
        return False

def main():
    print("🧪 3D Dimension Measurement System - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Core Dependencies", test_core_dependencies),
        ("GPU Support", test_gpu),
        ("DUSt3R Import", test_dust3r_import),
        ("MASt3R Import", test_mast3r_import),
        ("Model Files", test_models),
        ("Server Import", test_server_import),
        ("Basic Functionality", test_basic_functionality),
        ("API Server", test_api_server)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\n🚀 Quick start:")
        print("   python server/main.py image1.jpg image2.jpg image3.jpg")
        print("   # or start API server:")
        print("   python server/main.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the installation.")
        print("\n🔧 Troubleshooting:")
        print("   1. Run setup script: python scripts/setup.py")
        print("   2. Check dependencies: pip install -r requirements.txt")
        print("   3. Download models: python scripts/download_models.py")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
