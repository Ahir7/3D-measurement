# Migration Guide: DUSt3R to COLMAP+Metric3D

## Overview

This document guides you through migrating from the old DUSt3R-based system to the new GPU-accelerated COLMAP+Metric3D architecture.

## Key Changes

### Architecture

**Old System (v1.x):**
- DUSt3R for 3D reconstruction
- CPU/GPU hybrid processing
- Simple scale recovery (3 methods)
- Basic FastAPI

**New System (v2.0):**
- COLMAP for 3D reconstruction (GPU-accelerated)
- Metric3D for depth estimation
- GPU-only processing (no CPU fallback)
- Advanced scale recovery (4 methods with fusion)
- Production-ready FastAPI

### Directory Structure

**Old:**
```
├── server/
│   ├── main.py
│   ├── models/
│   └── preprocessing/
├── dust3r/
└── mast3r/
```

**New:**
```
├── src/
│   ├── core/
│   ├── reconstruction/
│   ├── depth/
│   ├── scale/
│   └── api/
├── main.py
└── requirements/
```

## Migration Steps

### 1. Backup Old System

```bash
# Backup your old system
cp -r server server_old
cp -r config config_old
```

### 2. Install New Dependencies

```bash
# Run setup script
python setup.py

# Or manual installation
pip install -r requirements/gpu.txt
```

### 3. Update Code

#### Old API Call:
```python
from server.main import DUSt3RDimensionCalculator

calculator = DUSt3RDimensionCalculator(device="cuda")
result = await calculator.calculate_dimensions(
    image_paths,
    imu_data=imu_data
)
```

#### New API Call:
```python
from src.core.measurement_system_gpu import MeasurementSystemGPU
from src.core.config import SystemConfig

config = SystemConfig()
system = MeasurementSystemGPU(config)
result = system.measure(
    images=images,
    imu_data=imu_data
)
```

### 4. Update Configuration

#### Old Config:
```python
# server/config/model_config.py
DUST3R_MODEL_NAME = "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
IMAGE_SIZE = 512
```

#### New Config:
```python
# src/core/config.py
from src.core.config import SystemConfig, COLMAPConfig, Metric3DConfig

config = SystemConfig(
    colmap=COLMAPConfig(num_features=16384),
    metric3d=Metric3DConfig(input_size=(518, 518)),
    max_image_size=2048
)
```

### 5. Update API Endpoints

#### Old Endpoint:
```python
@app.post("/calculate-dimensions-advanced")
async def calculate_dimensions_advanced(...)
```

#### New Endpoint:
```python
@app.post("/measure")
async def measure_endpoint(...)
```

### 6. Update Mobile App

#### Old Request:
```javascript
const response = await fetch(`${serverUrl}/calculate-dimensions-advanced`, {
    method: 'POST',
    body: formData
});
```

#### New Request:
```javascript
const response = await fetch(`${serverUrl}/measure`, {
    method: 'POST',
    body: formData
});
```

## API Changes

### Endpoint Mapping

| Old Endpoint | New Endpoint | Notes |
|-------------|--------------|-------|
| `/health` | `/health` | No change |
| `/calculate-dimensions-advanced` | `/measure` | Simplified name |
| N/A | `/benchmark` | New |
| N/A | `/gpu-stats` | New |

### Response Format

#### Old Response:
```json
{
  "success": true,
  "dimensions_metric": {
    "width_cm": 25.4,
    "height_cm": 15.2
  },
  "confidence_metrics": {
    "overall_confidence": 0.87
  }
}
```

#### New Response:
```json
{
  "success": true,
  "measurements": {
    "width": 25.4,
    "height": 15.2,
    "depth": 10.8,
    "volume_cm3": 4156.2
  },
  "confidence": 0.87,
  "processing_times": {
    "gpu_time": 2.34
  }
}
```

## Feature Comparison

| Feature | Old (v1.x) | New (v2.0) | Notes |
|---------|-----------|-----------|-------|
| 3D Reconstruction | DUSt3R | COLMAP | Better accuracy |
| Depth Estimation | Integrated | Metric3D | Metric scale |
| GPU Support | Optional | Required | Faster processing |
| CPU Fallback | Yes | No | GPU-only architecture |
| Scale Methods | 3 | 4 | Added object detection |
| Marker Support | Limited | Full | ArUco, QR, AprilTag |
| Processing Time | 5-10s | 2-5s | 2x faster |
| Accuracy | ±5% | ±2-3% | Improved |

## Breaking Changes

### 1. GPU Required

The new system **requires** a CUDA-capable GPU. No CPU fallback.

**Migration Action:** Ensure deployment environment has GPU.

### 2. Different Models

DUSt3R models are no longer used. COLMAP and Metric3D are now required.

**Migration Action:** No manual model download needed (handled automatically).

### 3. Configuration Structure

Complete rewrite of configuration system using dataclasses.

**Migration Action:** Update all config references to new format.

### 4. Import Paths

All imports have changed due to new directory structure.

**Migration Action:** Update all import statements.

## Testing Migration

### 1. Verify Installation

```bash
python main.py info
```

### 2. Test Basic Functionality

```bash
python main.py measure test_images/*.jpg
```

### 3. Test API

```bash
# Start server
python main.py serve

# Test endpoint
curl -X POST "http://localhost:8000/measure" \
  -F "files=@test1.jpg" \
  -F "files=@test2.jpg" \
  -F "files=@test3.jpg"
```

### 4. Run Benchmark

```bash
python main.py benchmark
```

## Performance Optimization

### Old System Optimization:
```python
# Limited options
device = "cuda" if available else "cpu"
```

### New System Optimization:
```python
from src.core.config import GPUConfig

gpu_config = GPUConfig(
    mixed_precision=True,  # 2x speedup
    num_streams=4,         # Parallel processing
    allow_tf32=True        # Faster computation
)
```

## Rollback Plan

If you need to rollback:

```bash
# Restore old system
mv server_old server
mv config_old config

# Reinstall old dependencies
pip install -r requirements.txt  # Old requirements
```

## Gradual Migration

You can run both systems in parallel:

```bash
# Old system on port 8000
cd old_system
python server/main.py

# New system on port 8001
cd new_system
python main.py serve --port 8001
```

## Common Issues

### Issue: "GPU not available"

**Solution:** Install NVIDIA drivers and CUDA toolkit.

```bash
nvidia-smi  # Check GPU
nvcc --version  # Check CUDA
```

### Issue: "COLMAP not found"

**Solution:** Install COLMAP or pycolmap.

```bash
pip install pycolmap
# Or: apt-get install colmap
```

### Issue: "Import errors"

**Solution:** Ensure you're running from project root.

```bash
cd /path/to/project
python main.py serve
```

## Support

If you encounter issues during migration:

1. Check the logs: `logs/system.log`
2. Run diagnostics: `python main.py info`
3. Open an issue on GitHub with migration context

## Timeline

Recommended migration timeline:

- **Week 1**: Setup and testing
- **Week 2**: Parallel deployment
- **Week 3**: Switch to new system
- **Week 4**: Deprecate old system

## Checklist

- [ ] Backup old system
- [ ] Install new dependencies
- [ ] Update configuration
- [ ] Update API calls
- [ ] Update mobile app
- [ ] Test basic functionality
- [ ] Test API endpoints
- [ ] Run benchmarks
- [ ] Deploy to production
- [ ] Monitor for issues
- [ ] Deprecate old system

---

**Questions?** Open an issue or discussion on GitHub.

