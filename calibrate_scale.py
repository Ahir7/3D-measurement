#!/usr/bin/env python3
"""
Scale Calibration Tool

This tool helps you calibrate the measurement system using a known reference.
"""

import json
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("SCALE CALIBRATION TOOL")
    print("=" * 60)
    print()
    
    # Check if results exist
    results_file = Path("output/results.json")
    if not results_file.exists():
        print("[ERROR] No results found at output/results.json")
        print("       Run a measurement first:")
        print("       python main.py measure resized\\*.jpg")
        return 1
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("Current Measurement Results:")
    print("-" * 60)
    print(f"  Width:  {results['measurements']['width']:.2f} cm")
    print(f"  Height: {results['measurements']['height']:.2f} cm")
    print(f"  Depth:  {results['measurements']['depth']:.2f} cm")
    print(f"  Confidence: {results['confidence']:.1f}%")
    print()
    
    if results['confidence'] > 50:
        print("[OK] Confidence is good! Measurements should be accurate.")
        return 0
    
    print("[WARNING] Low confidence! These measurements are in arbitrary units.")
    print()
    print("To calibrate, you need ONE known measurement from your scene.")
    print("Examples:")
    print("  - A door is 200 cm tall")
    print("  - A table is 80 cm high")
    print("  - A room is 400 cm wide")
    print()
    
    # Get calibration input
    try:
        print("Which dimension do you know? (width/height/depth)")
        dimension = input("> ").strip().lower()
        
        if dimension not in ['width', 'height', 'depth']:
            print("[ERROR] Invalid dimension. Use: width, height, or depth")
            return 1
        
        print(f"\nWhat is the ACTUAL {dimension} in centimeters?")
        actual_value = float(input("> ").strip())
        
        if actual_value <= 0:
            print("[ERROR] Value must be positive")
            return 1
        
    except (EOFError, KeyboardInterrupt):
        print("\n[CANCELLED] Calibration cancelled")
        return 1
    except ValueError:
        print("[ERROR] Invalid number")
        return 1
    
    # Calculate scale factor
    measured_value = results['measurements'][dimension]
    scale_factor = actual_value / measured_value
    
    print()
    print("=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"  Measured {dimension}: {measured_value:.2f} cm")
    print(f"  Actual {dimension}:   {actual_value:.2f} cm")
    print(f"  Scale factor:         {scale_factor:.4f}")
    print()
    
    # Calculate calibrated dimensions
    calibrated_width = results['measurements']['width'] * scale_factor
    calibrated_height = results['measurements']['height'] * scale_factor
    calibrated_depth = results['measurements']['depth'] * scale_factor
    calibrated_volume = calibrated_width * calibrated_height * calibrated_depth
    
    print("CALIBRATED MEASUREMENTS:")
    print("-" * 60)
    print(f"  Width:  {calibrated_width:.2f} cm")
    print(f"  Height: {calibrated_height:.2f} cm")
    print(f"  Depth:  {calibrated_depth:.2f} cm")
    print(f"  Volume: {calibrated_volume:.2f} cm³ ({calibrated_volume/1000000:.4f} m³)")
    print()
    
    # Save calibrated results
    calibrated_results = results.copy()
    calibrated_results['measurements']['width'] = calibrated_width
    calibrated_results['measurements']['height'] = calibrated_height
    calibrated_results['measurements']['depth'] = calibrated_depth
    calibrated_results['measurements']['volume_cm3'] = calibrated_volume
    calibrated_results['scale_recovery']['scale_factor'] = scale_factor
    calibrated_results['scale_recovery']['calibrated'] = True
    calibrated_results['scale_recovery']['calibration_method'] = 'manual'
    calibrated_results['confidence'] = 75.0  # Medium confidence for manual calibration
    
    output_file = Path("output/results_calibrated.json")
    with open(output_file, 'w') as f:
        json.dump(calibrated_results, f, indent=2)
    
    print(f"[SAVED] Calibrated results saved to: {output_file}")
    print()
    
    # Save scale factor for reference (but don't create invalid config)
    scale_file = Path("output/scale_factor.txt")
    with open(scale_file, 'w') as f:
        f.write(f"# Scale factor from calibration\n")
        f.write(f"# Date: {calibrated_results.get('timestamp', 'unknown')}\n")
        f.write(f"# Calibrated dimension: {dimension} = {actual_value:.2f} cm\n")
        f.write(f"# Scale factor: {scale_factor:.6f}\n")
        f.write(f"\n{scale_factor:.6f}\n")
    
    print(f"[SAVED] Scale factor saved to: {scale_file}")
    print()
    print("NOTE: The scale factor has been applied to your measurements.")
    print("      For future measurements, you'll need to calibrate again OR")
    print("      use the same camera setup at similar distances.")
    print()
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

