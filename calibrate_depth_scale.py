#!/usr/bin/env python3
"""
Calibrate depth scale factor based on ground truth measurements.

Usage:
    python calibrate_depth_scale.py
"""

import json
from pathlib import Path

# Ground truth measurements (in cm)
GROUND_TRUTH = {
    'width': 21.0,   # cm
    'height': 15.0,  # cm
    'depth': 7.8     # cm
}

# Read latest measurement results
results_path = Path('output/results.json')
if not results_path.exists():
    print("❌ No results.json found. Run measurement first:")
    print("   python main.py measure examples/original/resized/*.jpg")
    exit(1)

with open(results_path) as f:
    results = json.load(f)

measured = results['measurements']
scale_info = results['scale_recovery']

print("\n" + "="*70)
print("DEPTH SCALE CALIBRATION")
print("="*70)

print("\n📏 Ground Truth:")
print(f"   Width:  {GROUND_TRUTH['width']:.1f} cm")
print(f"   Height: {GROUND_TRUTH['height']:.1f} cm")
print(f"   Depth:  {GROUND_TRUTH['depth']:.1f} cm")

print("\n📐 Measured (Before Calibration):")
print(f"   Width:  {measured['width']:.2f} cm")
print(f"   Height: {measured['height']:.2f} cm")
print(f"   Depth:  {measured['depth']:.2f} cm")
print(f"   Confidence: {results['confidence']:.1%}")
print(f"   Error Bounds: ±{results['error_bounds']['relative_error_percent']:.1f}%")

# Calculate correction factors
factors = {
    'width': GROUND_TRUTH['width'] / measured['width'],
    'height': GROUND_TRUTH['height'] / measured['height'],
    'depth': GROUND_TRUTH['depth'] / measured['depth']
}

print("\n🔧 Correction Factors Needed:")
for dim, factor in factors.items():
    print(f"   {dim.capitalize():7s}: {factor:.6f}  ({1/factor:.2f}x too large)")

# Recommended calibration factor (average of all dimensions)
import numpy as np
avg_factor = np.mean(list(factors.values()))
median_factor = np.median(list(factors.values()))

print(f"\n✨ Recommended Calibration:")
print(f"   Average:  {avg_factor:.6f}")
print(f"   Median:   {median_factor:.6f}")

# Check consistency
factor_std = np.std(list(factors.values()))
factor_cv = factor_std / avg_factor * 100

if factor_cv < 5:
    print(f"   Consistency: ✅ Excellent (CV={factor_cv:.1f}%)")
    recommended = avg_factor
elif factor_cv < 10:
    print(f"   Consistency: ⚠️  Good (CV={factor_cv:.1f}%)")
    recommended = median_factor  # Use median if some variation
else:
    print(f"   Consistency: ❌ Poor (CV={factor_cv:.1f}%) - measurements may be unreliable")
    recommended = median_factor

print(f"\n🎯 Use This Value: {recommended:.6f}")

# Calculate expected measurements after calibration
expected = {
    'width': measured['width'] * recommended,
    'height': measured['height'] * recommended,
    'depth': measured['depth'] * recommended
}

print("\n📊 Expected After Calibration:")
for dim in ['width', 'height', 'depth']:
    exp = expected[dim]
    truth = GROUND_TRUTH[dim]
    error = abs(exp - truth) / truth * 100
    status = "✅" if error < 5 else "⚠️" if error < 10 else "❌"
    print(f"   {dim.capitalize():7s}: {exp:.2f} cm  (error: {error:.1f}%) {status}")

print("\n" + "="*70)
print("HOW TO APPLY CALIBRATION (POST-SCALE)")
print("="*70)

print(f"""
Edit configs/rtx2060_config.py:

1) Keep pre-scale at 1.0
   metric3d = Metric3DConfig(
       ...
       depth_scale_factor = 1.0,
   )

2) Apply post-scale calibration
   scale_recovery = ScaleRecoveryConfig(
       ...
       depth_only_calibration = {recommended:.6f},
   )

Then re-run measurement:
   python main.py measure examples/original/resized/*.jpg
""")

print("="*70)

# Save calibration to file
calib_file = Path('output/depth_calibration.txt')
with open(calib_file, 'w') as f:
    f.write(f"# Depth Scale Calibration\n")
    f.write(f"# Generated from ground truth: {GROUND_TRUTH['width']}x{GROUND_TRUTH['height']}x{GROUND_TRUTH['depth']} cm\n")
    f.write(f"# Measured: {measured['width']:.2f}x{measured['height']:.2f}x{measured['depth']:.2f} cm\n")
    f.write(f"# Confidence: {results['confidence']:.1%}\n\n")
    f.write(f"depth_only_calibration={recommended:.6f}\n")

print(f"💾 Calibration saved to: {calib_file}")
print()

