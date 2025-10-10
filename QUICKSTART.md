# Quick Start Guide - 3D Measurement System v2.0

Get up and running in 5 minutes!

## Prerequisites

- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.1 installed
- Python 3.8-3.10

## Installation (2 minutes)

```bash
# 1. Run automated setup
python setup.py

# That's it! The script will:
# - Check Python version
# - Verify GPU availability  
# - Install all dependencies
# - Run basic tests
```

## Quick Test (1 minute)

```bash
# Check system info
python main.py info

# You should see:
# ✅ GPU: NVIDIA GeForce RTX 4090
# ✅ CUDA: 12.1
# ✅ System Ready
```

## First Measurement (2 minutes)

### Option 1: Command Line

```bash
# Measure from your images
python main.py measure image1.jpg image2.jpg image3.jpg

# Output:
# Width:  25.4 cm
# Height: 15.2 cm  
# Depth:  10.8 cm
# Volume: 4156.2 cm³
```

### Option 2: API Server

```bash
# Terminal 1: Start server
python main.py serve

# Terminal 2: Test API
curl -X POST "http://localhost:8000/measure" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### Option 3: Python Script

```python
# measure.py
from src.core.measurement_system_gpu import MeasurementSystemGPU
import cv2

# Initialize
system = MeasurementSystemGPU()

# Load images
images = [cv2.imread(f"img{i}.jpg") for i in range(3)]
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

# Measure
result = system.measure(images)
print(f"Width: {result.measurements['width']:.1f} cm")
```

## Tips for Best Results

### 1. Image Capture

✅ **DO:**
- Take 5-10 images from different angles
- Ensure 30-50% overlap between images
- Use good lighting
- Keep object in focus
- Move around the object

❌ **DON'T:**
- Use blurry images
- Take images from same position
- Have poor lighting
- Move the object

### 2. Add Markers (Optional)

Print ArUco markers for better scale:

```bash
# Generate marker
python -c "import cv2; cv2.aruco.drawMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250), 0, 200).save('marker.png')"
```

Place 100mm marker in scene, system will auto-detect!

### 3. Using IMU Data (Optional)

```python
imu_data = [
    {
        'timestamp': 0,
        'accelerometer': {'x': 0.1, 'y': 0.2, 'z': 9.8}
    },
    # ... more IMU samples
]

result = system.measure(images, imu_data=imu_data)
```

## Common Issues

### "GPU not available"

```bash
# Check GPU
nvidia-smi

# If not working:
# 1. Install NVIDIA drivers
# 2. Install CUDA 12.1
# 3. Restart computer
```

### "ModuleNotFoundError"

```bash
# Reinstall dependencies
pip install -r requirements/gpu.txt
```

### "COLMAP not found"

```bash
# Install pycolmap
pip install pycolmap

# Or COLMAP binary
sudo apt-get install colmap  # Ubuntu
```

## Next Steps

Now that you're up and running:

1. 📖 Read full docs: `README_NEW.md`
2. 🔄 Migration from v1: `MIGRATION_GUIDE.md`
3. 🎯 Best practices: `new-plan.md`
4. 🐳 Docker deploy: `docker-compose.gpu.yml`

## Performance Expectations

| GPU | Processing Time (5 images) |
|-----|--------------------------|
| RTX 3090 | ~3.5 seconds |
| RTX 4090 | ~2.1 seconds |
| A100 | ~2.7 seconds |
| H100 | ~1.8 seconds |

## Getting Help

- Run diagnostics: `python main.py info`
- Check logs: `logs/system.log`
- GitHub Issues: [link]
- Documentation: `README_NEW.md`

## Benchmark Your System

```bash
python main.py benchmark --num-images 5 --num-runs 3
```

This will tell you exact performance on your hardware!

---

**That's it! You're ready to go!** 🚀

For more advanced usage, see `README_NEW.md`.

