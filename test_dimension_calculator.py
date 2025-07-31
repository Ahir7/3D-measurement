import torch
import numpy as np
from PIL import Image, ExifTags, ImageFile
import sys
import os
import time
from pathlib import Path
import argparse
import json

# Allow loading truncated JPEGs for robustness
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Add DUSt3R to Python path
sys.path.append(str(Path(__file__).parent / "dust3r"))

try:
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    print("✓ Successfully imported DUSt3R")
except ImportError as e:
    print(f"✗ Error importing DUSt3R: {e}")
    sys.exit(1)

def sanitize_for_json(data):
    """Recursively convert non-serializable objects to JSON-compatible types."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, tuple):
        return list(sanitize_for_json(item) for item in data)
    elif hasattr(data, 'numerator') and hasattr(data, 'denominator'):
        # Handle IFDRational and other fraction-like objects
        try:
            return float(data)
        except:
            return str(data)
    else:
        try:
            # Test if object is JSON serializable
            json.dumps(data)
            return data
        except:
            return str(data)

class DUSt3RDimensionCalculator:
    def __init__(self, model_path=None, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Use DUSt3R model instead of MASt3R
        if model_path is None:
            model_path = Path(__file__).parent / "dust3r" / "checkpoints" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        
        print(f"Loading DUSt3R model from: {model_path}")
        try:
            self.model = AsymmetricCroCo3DStereo.from_pretrained(str(model_path)).to(self.device)
            self.model.eval()
            print("✓ DUSt3R model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load DUSt3R model: {e}")

    def extract_metadata(self, image_path):
        """Extract EXIF metadata from image."""
        try:
            with Image.open(image_path) as img:
                exif = img._getexif() or {}
                metadata = {}
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    metadata[tag] = value
                metadata['ImageWidth'] = img.width
                metadata['ImageHeight'] = img.height
                # ✅ SANITIZE metadata before returning
                return sanitize_for_json(metadata)
        except Exception as e:
            print(f"Warning: Could not extract metadata from {image_path}: {e}")
            return {}

    def calculate_dimensions(self, image_paths, output_dir="output"):
        if len(image_paths) < 2:
            raise ValueError("At least 2 images required for 3D reconstruction")
        
        print(f"\n🔍 Processing {len(image_paths)} images...")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        start_time = time.time()
        
        try:
            # Use DUSt3R's simpler approach - load images directly
            print("📐 Loading images with DUSt3R...")
            images = load_images(image_paths, size=512, square_ok=False)
            print(f"✓ Loaded {len(images)} images successfully")
            
            # Create pairs using DUSt3R's method
            print("🔗 Creating image pairs...")
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            print(f"✓ Created {len(pairs)} image pairs")
            
            # Run DUSt3R inference (more stable than MASt3R)
            print("🧠 Running DUSt3R neural network inference...")
            
            try:
                from torch.amp import autocast
                autocast_context = autocast(device_type=self.device, enabled=True)
            except ImportError:
                from torch.cuda.amp import autocast
                autocast_context = autocast(enabled=True)
            
            with autocast_context:
                inference_result = inference(pairs, self.model, self.device, batch_size=1, verbose=True)
            
            print("✓ Inference completed successfully")
            
            print("🌐 Performing global alignment...")
            mode = GlobalAlignerMode.PointCloudOptimizer
            scene = global_aligner(inference_result, device=self.device, mode=mode)
            
            print("📊 Extracting 3D points...")
            points3d = scene.get_pts3d()
            all_points = []
            
            for points in points3d:
                if points is not None:
                    valid_points = points[~torch.isnan(points).any(dim=-1)]
                    if len(valid_points) > 0:
                        # ✅ FIXED: Added .detach() to handle gradient tensors
                        all_points.append(valid_points.detach().cpu().numpy())
            
            if not all_points:
                raise ValueError("No valid 3D points found in reconstruction")
            
            combined_points = np.concatenate(all_points, axis=0)
            print(f"✓ Extracted {len(combined_points):,} valid 3D points")
            
            # Calculate dimensions
            min_coords = np.min(combined_points, axis=0)
            max_coords = np.max(combined_points, axis=0)
            dimensions = max_coords - min_coords
            center = (min_coords + max_coords) / 2
            volume = np.prod(dimensions)
            
            # Prepare results
            results = {
                "success": True,
                "processing_time": time.time() - start_time,
                "num_images": len(image_paths),
                "num_3d_points": len(combined_points),
                "dimensions": {
                    "width": float(dimensions[0]),
                    "height": float(dimensions[1]),
                    "depth": float(dimensions[2]),
                    "volume": float(volume)
                },
                "bounding_box": {
                    "min": min_coords.tolist(),
                    "max": max_coords.tolist(),
                    "center": center.tolist()
                },
                "metadata": {
                    "device": self.device,
                    "model": "DUSt3R",
                    "image_metadata": [self.extract_metadata(img_path) for img_path in image_paths]
                }
            }
            
            # ✅ SANITIZE results before JSON serialization
            results = sanitize_for_json(results)
            
            # Save results
            result_file = output_path / "dimension_results.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            np.save(output_path / "point_cloud.npy", combined_points)
            
            print(f"\n✅ Processing completed in {results['processing_time']:.1f} seconds")
            print(f"📊 Results:")
            print(f"   Width:  {dimensions[0]:.3f} units")
            print(f"   Height: {dimensions[1]:.3f} units")
            print(f"   Depth:  {dimensions[2]:.3f} units")
            print(f"   Volume: {volume:.3f} cubic units")
            print(f"   Points: {len(combined_points):,}")
            print(f"📁 Results saved to: {result_file}")
            
            return results
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            print(f"❌ Error: {e}")
            return error_result

def main():
    parser = argparse.ArgumentParser(description='3D dimension calculation with DUSt3R (stable version)')
    parser.add_argument('images', nargs='+', help='Path to input images')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--model', '-m', help='Path to model file')
    
    args = parser.parse_args()
    
    print("🖥️  System Information:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        calculator = DUSt3RDimensionCalculator(model_path=args.model)
        results = calculator.calculate_dimensions(args.images, args.output)
        return 0 if results["success"] else 1
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
