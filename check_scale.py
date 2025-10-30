"""Check scale calculation details"""
import numpy as np

# From logs:
marker_scale_mm_per_px = 6.1724  # mm/pixel
depth_scale = 0.8273  # unitless ratio

# Current results (in mm since measurements dict uses mm):
current_width_mm = 7269.22 * 10  # Convert cm to mm = 72692mm
current_height_mm = 6561.30 * 10  # = 65613mm 
current_depth_mm = 1529.95 * 10  # = 15299mm

# Target (in mm):
target_width_mm = 210  # 21cm
target_height_mm = 150  # 15cm
target_depth_mm = 78   # 7.8cm

print("="*60)
print("SCALE ANALYSIS")
print("="*60)
print()
print(f"Marker scale: {marker_scale_mm_per_px:.4f} mm/pixel")
print(f"Depth scale: {depth_scale:.4f}")
print()
print("Current measurements (mm):")
print(f"  Width:  {current_width_mm:.0f} mm ({current_width_mm/10:.1f} cm)")
print(f"  Height: {current_height_mm:.0f} mm ({current_height_mm/10:.1f} cm)")
print(f"  Depth:  {current_depth_mm:.0f} mm ({current_depth_mm/10:.1f} cm)")
print()
print("Target measurements (mm):")
print(f"  Width:  {target_width_mm} mm ({target_width_mm/10} cm)")
print(f"  Height: {target_height_mm} mm ({target_height_mm/10} cm)")
print(f"  Depth:  {target_depth_mm} mm ({target_depth_mm/10} cm)")
print()
print("Required correction factor:")
correction_width = target_width_mm / current_width_mm
correction_height = target_height_mm / current_height_mm
correction_depth = target_depth_mm / current_depth_mm
correction_avg = np.mean([correction_width, correction_height, correction_depth])

print(f"  Width:  {correction_width:.6f}x")
print(f"  Height: {correction_height:.6f}x")
print(f"  Depth:  {correction_depth:.6f}x")
print(f"  Average: {correction_avg:.6f}x")
print()
print(f"This means the scale should be: {marker_scale_mm_per_px * correction_avg:.6f} instead of {marker_scale_mm_per_px:.4f}")
print()
print("="*60)
print("HYPOTHESIS:")
print("="*60)
print()
print("The marker_size_mm configuration (100mm = 10cm) might be WRONG!")
print(f"If the actual marker size is {100 * correction_avg:.2f}mm ({100 * correction_avg/10:.2f}cm),")
print("then the scale calculation would be correct.")
print()
print("OR")
print()
print("The scale might need to be INVERTED or applied differently.")
