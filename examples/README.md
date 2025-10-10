# Example Images

This directory contains example images for testing the 3D measurement system.

## Directory Structure

- `original/` - Original high-resolution images (3072x4096)
- `resized/` - Resized images (768x1024) optimized for 4GB GPU

## Usage

### Using Original Images
```bash
# Resize them first
python resize_images.py

# Then measure
python main.py measure examples/resized/*.jpg
```

### Using Pre-resized Images
```bash
python main.py measure examples/resized/*.jpg
```

## Image Requirements

For best results:
- **Minimum**: 15 images
- **Optimal**: 20-25 images
- **Maximum**: 30 images (for 4GB GPU)
- 60-80% overlap between consecutive images
- Good lighting and focus
- Cover the object from multiple angles

See `IMAGE_CAPTURE_GUIDE.md` for detailed guidelines.
