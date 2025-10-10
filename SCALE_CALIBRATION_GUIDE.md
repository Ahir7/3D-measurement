# 🎯 Scale Calibration Guide

## ❌ **Problem: Inaccurate Measurements**

Your system shows measurements like:
- Width: 1459.24 cm
- Height: 977.74 cm
- Depth: 1495.06 cm
- **Confidence: 0.0%**

These are **arbitrary units**, not real measurements, because:
1. No physical reference detected (no markers)
2. Insufficient scale sources (only 1 of 2 required)
3. System falls back to default scale of 1.0

---

## ✅ **Solution 1: Use Reference Markers (MOST ACCURATE)**

### **Step 1: Print ArUco Markers**

1. Go to: https://chev.me/arucogen/
2. Settings:
   - Dictionary: `DICT_6X6_250`
   - Marker ID: `0` (or any ID 0-249)
   - Marker size: `100mm` (10cm) or whatever size you print
3. Print it at **exact size** (disable "Fit to page")
4. Measure the printed marker **precisely** with a ruler

### **Step 2: Place Markers in Your Scene**

- Place at least **2-3 markers** in the scene you're measuring
- Markers should be **flat** and **clearly visible**
- Ensure good lighting and sharp focus

### **Step 3: Update Configuration**

Edit `configs/gtx1650_config.py` or create `configs/custom_config.py`:

```python
from src.core.config import SystemConfig, ScaleRecoveryConfig

config = SystemConfig()

# Update scale recovery settings
config.scale_recovery = ScaleRecoveryConfig(
    marker_types=['aruco'],  # Enable ArUco markers
    marker_size_mm=100,      # Your actual marker size in mm
    marker_weight=1.0,       # High confidence in markers
    depth_weight=0.5,        # Use depth as secondary
    min_methods_required=2   # Require 2 methods for confidence
)
```

### **Step 4: Run Measurement**

```bash
python main.py measure --config configs/custom_config.py resized\*.jpg
```

---

## ✅ **Solution 2: Manual Scale Calibration**

If you know **one real measurement** in your scene (e.g., a door is 200cm tall):

### **Step 1: Run Measurement Without Scale**

```bash
python main.py measure resized\*.jpg
```

Note the height measurement (e.g., 977.74 cm).

### **Step 2: Calculate Scale Factor**

```
scale_factor = actual_measurement / measured_value
scale_factor = 200 cm / 977.74 cm = 0.2046
```

### **Step 3: Create Calibrated Config**

Create `configs/calibrated_config.py`:

```python
from src.core.config import SystemConfig, ScaleRecoveryConfig

config = SystemConfig()
config.scale_recovery = ScaleRecoveryConfig(
    marker_weight=0.0,
    depth_weight=1.0,
    min_methods_required=1,  # Allow single method
    default_scale=0.2046      # Your calculated scale
)
```

### **Step 4: Re-run with Calibrated Scale**

I'll create a script to apply this calibration automatically.

---

## ✅ **Solution 3: Lower Requirements (QUICK FIX)**

This allows the system to use depth-based scaling alone (less accurate):

### **Create Quick Config**

Create `configs/quick_config.py`:

```python
from src.core.config import SystemConfig, ScaleRecoveryConfig

config = SystemConfig()
config.scale_recovery = ScaleRecoveryConfig(
    marker_weight=0.0,       # No markers required
    depth_weight=1.0,        # Use depth only
    object_weight=0.0,
    imu_weight=0.0,
    min_methods_required=1   # ✅ Allow single method
)
```

### **Run Measurement**

```bash
python main.py measure resized\*.jpg
```

**Note**: Results will be more accurate than default (1.0) but still not perfect without a physical reference.

---

## 📊 **Understanding Confidence Scores**

- **0%**: Default scale used (no reliable estimates)
- **30-50%**: Single method (depth or object detection)
- **60-80%**: Multiple methods agree
- **80-100%**: Markers + depth + IMU all agree

---

## 🎯 **Recommended Workflow**

1. **Use markers** for critical measurements (±1-2% accuracy)
2. **Manual calibration** if you have one known dimension (±5-10% accuracy)
3. **Depth-only** for quick relative measurements (±20-30% accuracy)

---

## 🛠️ **Creating a Calibration Tool**

I'll create a tool to help you calibrate easily...

