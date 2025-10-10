# 🔧 Fix CUDA Support

## Problem

You have PyTorch **CPU-only** version installed:
```
PyTorch version: 2.8.0+cpu
CUDA available: False
```

But you have a **CUDA-capable GPU**:
```
GPU: NVIDIA GeForce GTX 1650
CUDA: 12.9
Driver: 577.03
```

---

## Solution

You need to **reinstall PyTorch with CUDA support**.

---

## 🚀 Quick Fix (Automated)

Run the fix script:

```bash
python fix_cuda.py
```

This will:
1. Uninstall CPU-only PyTorch
2. Install PyTorch with CUDA 12.1 support
3. Verify installation
4. Run GPU test

---

## 🔧 Manual Fix (Step by Step)

If the automated script doesn't work, follow these steps:

### Step 1: Uninstall Current PyTorch

```bash
pip uninstall torch torchvision torchaudio
```

Answer `y` when prompted.

### Step 2: Install PyTorch with CUDA 12.1

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

This will download **~2GB** of packages (be patient).

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
CUDA available: True
```

### Step 4: Full Verification

```bash
python main.py info
```

Expected output:
```
GPU: NVIDIA GeForce GTX 1650
CUDA: 12.1
Total GPU memory: 4.00 GB
```

---

## ✅ Verification Checklist

After installation, verify these:

```bash
# 1. Check PyTorch version (should NOT have +cpu)
python -c "import torch; print(torch.__version__)"
# Expected: 2.8.0+cu121 (or similar, without +cpu)

# 2. Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True

# 3. Check GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce GTX 1650

# 4. Quick GPU test
python -c "import torch; x = torch.randn(100, 100).cuda(); print('GPU test OK')"
# Expected: GPU test OK
```

---

## 🐛 Troubleshooting

### Issue 1: "CUDA available: False" after installation

**Solution**: Check CUDA version compatibility

Your CUDA: **12.9**  
PyTorch CUDA: **12.1** (compatible)

CUDA 12.x is backward compatible, so 12.1 should work with 12.9.

If not, try CUDA 12.6:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Issue 2: "No module named 'torch'"

**Solution**: Install was interrupted

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: "RuntimeError: CUDA out of memory" during test

**Solution**: This is actually **GOOD** - it means CUDA is working!

Your GPU only has 4GB, so large tensors may fail. Use the GTX 1650 config.

### Issue 4: Download is very slow

**Solution**: The CUDA version is ~2GB. Be patient or use a mirror.

Alternative (if US/Canada):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Alternative (if Europe):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 5: "Could not find a version that satisfies the requirement"

**Solution**: Update pip first

```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 📊 Before vs After

### Before (Current - Not Working)
```python
import torch
print(torch.__version__)      # 2.8.0+cpu
print(torch.cuda.is_available())  # False
```

### After (Target - Working)
```python
import torch
print(torch.__version__)      # 2.8.0+cu121
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce GTX 1650
```

---

## 🎯 Complete Fix Commands

Copy and paste these commands in order:

```bash
# 1. Uninstall CPU version
pip uninstall -y torch torchvision torchaudio

# 2. Install CUDA version (this will download ~2GB)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Verify
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('SUCCESS: CUDA is working!')"

# 4. Full system check
python main.py info

# 5. Run validation
python validate_system.py
```

---

## 🚀 After CUDA is Working

Once CUDA is detected:

### Test the system
```bash
# 1. Check system info
python main.py info

# 2. Run validation
python validate_system.py

# 3. Test measurement (if you have images)
python main.py measure image1.jpg image2.jpg image3.jpg
```

### Use GTX 1650 optimizations
```python
from configs.gtx1650_config import get_gtx1650_config
from src.core.measurement_system_gpu import MeasurementSystemGPU

config = get_gtx1650_config()
system = MeasurementSystemGPU(config)

# Now it will use GPU!
```

---

## 💡 Why This Happened

You likely installed requirements without specifying CUDA:

```bash
pip install torch  # This installs CPU-only version by default!
```

To get CUDA support, you MUST use the PyTorch index URL:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 📋 Quick Reference

### Install PyTorch with CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify CUDA
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Check GPU
```bash
nvidia-smi
```

### System info
```bash
python main.py info
```

---

## ⏱️ Expected Time

- **Uninstall**: ~10 seconds
- **Download**: ~5-15 minutes (depending on internet speed, ~2GB)
- **Install**: ~1-2 minutes
- **Verify**: ~5 seconds

**Total**: ~10-20 minutes

---

## ✅ Success Indicators

You'll know it's working when you see:

```bash
$ python -c "import torch; print(torch.cuda.is_available())"
True

$ python main.py info
GPU: NVIDIA GeForce GTX 1650
CUDA: 12.1
Total GPU memory: 4.00 GB

$ python validate_system.py
...
*** ALL CHECKS PASSED - System is ready! ***
```

---

## 🆘 Still Not Working?

If CUDA still doesn't work after following these steps:

### Check 1: NVIDIA Drivers
```bash
nvidia-smi
```
Should show your GPU. If not, install NVIDIA drivers from:
https://www.nvidia.com/Download/index.aspx

### Check 2: Python Environment
Make sure you're in the right environment:
```bash
# If using venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Then reinstall
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Check 3: Try Different CUDA Version
```bash
# Try CUDA 11.8 instead
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Check 4: Restart Computer
Sometimes Windows needs a restart after installing CUDA packages.

---

## 📞 Need Help?

If you're still stuck:

1. Run: `python fix_cuda.py` and share the output
2. Run: `nvidia-smi` and share the output
3. Run: `python -c "import torch; print(torch.__version__)"` and share the output

---

**Ready to fix? Run:**
```bash
python fix_cuda.py
```

Or follow the manual steps above! 🚀

