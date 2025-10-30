#!/usr/bin/env python3
"""
Calibrate marker size based on known object dimensions.

Usage:
    python calibrate_marker_size.py --actual-width 21 --actual-height 15 --actual-depth 7.8
"""

import argparse
import json
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Calibrate marker size')
    parser.add_argument('--actual-width', type=float, required=True, help='Actual width in cm')
    parser.add_argument('--actual-height', type=float, required=True, help='Actual height in cm')
    parser.add_argument('--actual-depth', type=float, required=True, help='Actual depth in cm')
    parser.add_argument('--results', type=str, default='output/results.json', help='Path to results.json')
    
    args = parser.parse_args()
    
    # Read current results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run measurement first!")
        return 1
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Get current measurements and scale
    current_width = results['measurements']['width']
    current_height = results['measurements']['height']
    current_depth = results['measurements']['depth']
    current_scale = results['scale_recovery']['scale_factor']
    
    print("\n" + "="*60)
    print("MARKER SIZE CALIBRATION")
    print("="*60)
    print()
    print("Current Measurements:")
    print(f"  Width:  {current_width:.2f} cm")
    print(f"  Height: {current_height:.2f} cm")
    print(f"  Depth:  {current_depth:.2f} cm")
    print()
    print("Actual Measurements:")
    print(f"  Width:  {args.actual_width:.2f} cm")
    print(f"  Height: {args.actual_height:.2f} cm")
    print(f"  Depth:  {args.actual_depth:.2f} cm")
    print()
    
    # Calculate scale factors needed
    scale_width = args.actual_width / current_width
    scale_height = args.actual_height / current_height
    scale_depth = args.actual_depth / current_depth
    
    print("Scale correction needed:")
    print(f"  Width:  {scale_width:.4f}x")
    print(f"  Height: {scale_height:.4f}x")
    print(f"  Depth:  {scale_depth:.4f}x")
    print()
    
    # Average scale factor
    avg_scale_factor = np.mean([scale_width, scale_height, scale_depth])
    std_scale_factor = np.std([scale_width, scale_height, scale_depth])
    
    print(f"Average scale factor: {avg_scale_factor:.4f}x")
    print(f"Standard deviation: {std_scale_factor:.4f}")
    print()
    
    if std_scale_factor / avg_scale_factor > 0.2:
        print("⚠️  WARNING: Large variation in scale factors!")
        print("    This might indicate measurement errors or distortion.")
        print()
    
    # Current marker size (from config)
    # We need to find what marker size was used
    # Reading from the marker scale directly
    marker_methods = [m for m in results['scale_recovery']['methods_used'] if m == 'marker']
    
    if not marker_methods:
        print("❌ No marker-based scale found in results!")
        print("   Run measurement again with ArUco markers visible.")
        return 1
    
    # The marker size should be scaled by the inverse of the correction factor
    # Because: larger marker_size_mm → smaller scale → larger measurements
    # So: marker_size_mm_correct = marker_size_mm_current * avg_scale_factor
    
    # But we don't know current marker_size_mm from results, so we'll calculate
    # from the current scale factor
    # scale = marker_size_mm / marker_size_pixels → marker_size_pixels = marker_size_mm / scale_meters
    
    # For now, assume default was 500mm (current setting)
    current_marker_size_mm = 500.0  # Update this if you know what it was
    
    # The correct marker size is:
    correct_marker_size_mm = current_marker_size_mm * avg_scale_factor
    
    print("="*60)
    print("SOLUTION:")
    print("="*60)
    print()
    print(f"Update marker_size_mm in config to: {correct_marker_size_mm:.1f} mm ({correct_marker_size_mm/10:.1f} cm)")
    print()
    print("1. Edit src/core/config.py:")
    print(f"   marker_size_mm: float = {correct_marker_size_mm:.1f}")
    print()
    print("2. Or edit configs/rtx2060_config.py:")
    print(f"   marker_size_mm={correct_marker_size_mm:.1f},")
    print()
    print("3. Then run measurement again")
    print()
    print("="*60)
    
    # Double-check calculation
    expected_width = current_width * avg_scale_factor
    expected_height = current_height * avg_scale_factor
    expected_depth = current_depth * avg_scale_factor
    
    print()
    print("Expected results after calibration:")
    print(f"  Width:  {expected_width:.2f} cm (target: {args.actual_width:.2f} cm)")
    print(f"  Height: {expected_height:.2f} cm (target: {args.actual_height:.2f} cm)")
    print(f"  Depth:  {expected_depth:.2f} cm (target: {args.actual_depth:.2f} cm)")
    print()
    print("="*60)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

