#!/usr/bin/env python3
"""
Resize images for optimal GPU processing.

Your images (3072x4096) are too large for GPU SIFT extraction.
This script resizes them to 1024px max dimension.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def resize_image(image_path, max_size=1024, output_dir="resized"):
    """Resize image to max dimension while maintaining aspect ratio."""
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Could not read {image_path}")
        return False
    
    h, w = img.shape[:2]
    
    # Calculate new size
    if max(h, w) <= max_size:
        print(f"[SKIP] {image_path.name} already small enough ({w}x{h})")
        return True
    
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize with high-quality interpolation
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Save
    output_path = Path(output_dir) / image_path.name
    cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"[OK] {image_path.name}: {w}x{h} -> {new_w}x{new_h}")
    return True

def main():
    """Resize all images."""
    
    print("\n" + "="*60)
    print("IMAGE RESIZER FOR GPU PROCESSING")
    print("="*60)
    
    # Find all JPG images
    image_files = sorted(Path(".").glob("[0-9]*.jpg"))
    
    if not image_files:
        print("\n[ERROR] No image files found (1.jpg, 2.jpg, etc.)")
        return 1
    
    print(f"\nFound {len(image_files)} images")
    
    # Check first image size
    first_img = cv2.imread(str(image_files[0]))
    if first_img is not None:
        h, w = first_img.shape[:2]
        print(f"Current size: {w}x{h} pixels")
        
        if max(h, w) <= 1024:
            print("\n[INFO] Images are already small enough for GPU processing!")
            print("No resizing needed.")
            return 0
    
    # Create output directory
    output_dir = Path("resized")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nResizing to max 1024px...")
    print(f"Output directory: {output_dir}")
    
    # Resize all images
    success_count = 0
    for img_path in image_files:
        if resize_image(img_path, max_size=1024, output_dir=output_dir):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"Resized {success_count}/{len(image_files)} images successfully")
    print("="*60)
    
    if success_count == len(image_files):
        print("\n✓ All images resized!")
        print("\nNow run measurement with resized images:")
        print(f"  python main.py measure resized\\*.jpg")
        return 0
    else:
        print(f"\n⚠ {len(image_files) - success_count} images failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

