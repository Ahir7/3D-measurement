import torch
import numpy as np
from PIL import Image, ExifTags, ImageFile
import sys
import os
import time
from pathlib import Path
import argparse
import json
import tempfile
import asyncio
from typing import List, Optional
import gc
import importlib

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow loading truncated JPEGs for robustness
ImageFile.LOAD_TRUNCATED_IMAGES = True

def sanitize_for_json(data):
    """Recursively convert non-serializable objects to JSON-compatible types."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_for_json(item) for item in data]
    elif hasattr(data, 'numerator') and hasattr(data, 'denominator'):
        try:
            return float(data)
        except:
            return str(data)
    elif hasattr(data, 'isoformat'):
        return data.isoformat()
    else:
        try:
            json.dumps(data)
            return data
        except:
            return str(data)

def force_import_dust3r():
    """
    Force import DUSt3R components, breaking circular imports.
    """
    # Clear all dust3r and croco related modules from cache
    modules_to_clear = [k for k in sys.modules if k.startswith(('dust3r', 'croco'))]
    for module in modules_to_clear:
        del sys.modules[module]
    gc.collect()
    
    # Add dust3r path to Python path if needed
    root_path = Path(__file__).parent.parent
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    try:
        # Import in specific order to avoid circular dependencies
        utils = importlib.import_module('dust3r.utils.image')
        pairs = importlib.import_module('dust3r.image_pairs')
        inference_mod = importlib.import_module('dust3r.inference')
        cloud_opt = importlib.import_module('dust3r.cloud_opt')

        # Do NOT import model here
        return {
            'inference': inference_mod.inference,
            'load_images': utils.load_images,
            'make_pairs': pairs.make_pairs,
            'global_aligner': cloud_opt.global_aligner,
            'GlobalAlignerMode': cloud_opt.GlobalAlignerMode
        }
    except Exception as e:
        print(f"✗ Error importing DUSt3R: {e}")
        sys.exit(1)

class MultiMethodScaleRecovery:
    def __init__(self):
        self.confidence_weights = {
            'imu': 0.35,
            'metadata': 0.25,
            'geometric': 0.20,
            'ml_depth': 0.15,
            'motion_parallax': 0.05
        }
    
    async def estimate_scale(self, images, imu_data, metadata, timestamps, reconstruction):
        scale_estimates = {}
        confidences = {}
        
        if imu_data:
            print("🔄 IMU-based scale estimation...")
            imu_result = await self.estimate_scale_from_imu(
                imu_data, reconstruction['camera_poses'], timestamps
            )
            if imu_result['success']:
                scale_estimates['imu'] = imu_result['scale']
                confidences['imu'] = imu_result['confidence']
        
        if metadata:
            print("📷 Metadata-based scale estimation...")
            metadata_result = await self.estimate_scale_from_metadata(
                metadata, reconstruction.get('depth_maps', [])
            )
            if metadata_result['success']:
                scale_estimates['metadata'] = metadata_result['scale']
                confidences['metadata'] = metadata_result['confidence']
        
        print("📐 Geometric scale estimation...")
        geometric_result = await self.estimate_scale_from_geometry(
            reconstruction['camera_poses'], reconstruction['point_cloud']
        )
        if geometric_result['success']:
            scale_estimates['geometric'] = geometric_result['scale']
            confidences['geometric'] = geometric_result['confidence']
        
        final_scale, final_confidence = self.fuse_scale_estimates(
            scale_estimates, confidences
        )
        
        return {
            'scale_factor': final_scale,
            'confidence': final_confidence,
            'methods_used': list(scale_estimates.keys()),
            'individual_estimates': scale_estimates,
            'individual_confidences': confidences
        }
    
    async def estimate_scale_from_imu(self, imu_data, camera_poses, timestamps):
        try:
            if not imu_data or len(imu_data) < 2:
                return {'success': False, 'error': 'Insufficient IMU data'}
            
            real_world_motion = self.integrate_imu_motion(imu_data, timestamps)
            camera_motion = self.calculate_camera_motion(camera_poses)
            
            if camera_motion > 0.001:
                scale = real_world_motion / camera_motion
                if 0.1 <= scale <= 10.0:
                    confidence = self.calculate_imu_confidence(imu_data, real_world_motion)
                    return {
                        'success': True,
                        'scale': scale,
                        'confidence': confidence
                    }
            
            return {'success': False, 'error': 'Invalid motion calculation'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def integrate_imu_motion(self, imu_data, timestamps):
        total_motion = 0.0
        gravity = np.array([0, 0, -9.81])
        
        for i, imu_frame in enumerate(imu_data):
            if i == 0:
                continue
                
            dt = (timestamps[i] - timestamps[i-1]) / 1000.0
            accel_data = imu_frame.get('accelerometer', [])
            
            if accel_data:
                accel_readings = np.array([[d['x'], d['y'], d['z']] for d in accel_data])
                avg_accel = np.mean(accel_readings, axis=0)
                linear_accel = avg_accel - gravity
                motion_magnitude = np.linalg.norm(linear_accel) * dt * dt * 0.5
                total_motion += motion_magnitude
        
        return total_motion
    
    def calculate_camera_motion(self, camera_poses):
        if len(camera_poses) < 2:
            return 0.0
        
        total_motion = 0.0
        for i in range(1, len(camera_poses)):
            motion = np.linalg.norm(
                camera_poses[i][:3, 3] - camera_poses[i-1][:3, 3]
            )
            total_motion += motion
        return total_motion
    
    def calculate_imu_confidence(self, imu_data, motion):
        base_confidence = 0.8
        
        if motion < 0.05:
            base_confidence *= 0.5
        elif motion > 5.0:
            base_confidence *= 0.7
        
        total_readings = sum(len(frame.get('accelerometer', [])) for frame in imu_data)
        if total_readings < 50:
            base_confidence *= 0.6
        
        return min(base_confidence, 1.0)
    
    async def estimate_scale_from_metadata(self, metadata, depth_maps):
        try:
            scale_estimates = []
            
            for i, meta in enumerate(metadata):
                if 'FocalLength' not in meta or 'ImageWidth' not in meta:
                    continue
                
                focal_length_mm = float(meta['FocalLength'])
                image_width_px = meta['ImageWidth']
                sensor_width_mm = self.estimate_sensor_width(meta)
                
                fov_radians = 2 * np.arctan(sensor_width_mm / (2 * focal_length_mm))
                
                if i < len(depth_maps):
                    avg_depth = np.median(depth_maps[i])
                    scale = (image_width_px * avg_depth * np.tan(fov_radians / 2)) / (sensor_width_mm / 1000)
                    
                    if 0.1 <= scale <= 10.0:
                        scale_estimates.append(scale)
            
            if scale_estimates:
                final_scale = np.median(scale_estimates)
                confidence = 0.7 * (1.0 - np.std(scale_estimates) / np.mean(scale_estimates))
                return {
                    'success': True,
                    'scale': final_scale,
                    'confidence': max(0.1, confidence)
                }
            
            return {'success': False, 'error': 'No valid metadata estimates'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def estimate_sensor_width(self, metadata):
        sensor_widths = {
            'iPhone': 4.8,
            'Samsung': 5.6,
            'Google': 5.0,
            'OnePlus': 5.4,
            'Xiaomi': 5.2
        }
        
        device_model = metadata.get('Model', '').lower()
        for brand, width in sensor_widths.items():
            if brand.lower() in device_model:
                return width
        return 5.0
    
    async def estimate_scale_from_geometry(self, camera_poses, point_cloud):
        try:
            if len(camera_poses) < 2:
                return {'success': False, 'error': 'Insufficient camera poses'}
            
            baselines = []
            for i in range(len(camera_poses)):
                for j in range(i + 1, len(camera_poses)):
                    baseline = np.linalg.norm(
                        camera_poses[i][:3, 3] - camera_poses[j][:3, 3]
                    )
                    baselines.append(baseline)
            
            median_baseline = np.median(baselines)
            typical_motion_range = np.array([0.3, 0.8])  # meters
            scale_estimates = typical_motion_range / median_baseline
            scale = np.mean(scale_estimates)
            
            baseline_std = np.std(baselines)
            baseline_consistency = 1.0 - (baseline_std / median_baseline)
            confidence = 0.6 * max(0.1, baseline_consistency)
            
            return {
                'success': True,
                'scale': scale,
                'confidence': confidence
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def fuse_scale_estimates(self, scale_estimates, confidences):
        if not scale_estimates:
            return 1.0, 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for method, scale in scale_estimates.items():
            confidence = confidences.get(method, 0.0)
            base_weight = self.confidence_weights.get(method, 0.1)
            
            weight = confidence * base_weight
            weighted_sum += scale * weight
            total_weight += weight
        
        if total_weight > 0:
            final_scale = weighted_sum / total_weight
            final_confidence = total_weight / sum(self.confidence_weights.values())
        else:
            final_scale = 1.0
            final_confidence = 0.0
        
        return final_scale, min(final_confidence, 1.0)

class DUSt3RDimensionCalculator:
    def __init__(self, model_path=None, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Dynamically import DUSt3R components except model
        dust3r_components = force_import_dust3r()

        # Set model path
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "models" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

        # Import model class only here to avoid circular import
        try:
            import importlib
            model_module = importlib.import_module('dust3r.model')
            ModelClass = getattr(model_module, 'AsymmetricCroCo3DStereo')
        except Exception as e:
            raise RuntimeError(f"Failed to import AsymmetricCroCo3DStereo: {e}")

        # Load model
        print(f"Loading DUSt3R model from: {model_path}")
        try:
            if model_path.exists():
                print(f"Loading model from local checkpoint: {model_path}")
                # Load model using the module's load_model function
                self.model = model_module.load_model(str(model_path), self.device)
                self.model.eval()
                print("✓ DUSt3R model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load DUSt3R model: {e}")

        # Store only functions in dust3r_components
        self.dust3r_components = {
            'inference': dust3r_components['inference'],
            'load_images': dust3r_components['load_images'],
            'make_pairs': dust3r_components['make_pairs'],
            'global_aligner': dust3r_components['global_aligner'],
            'GlobalAlignerMode': dust3r_components['GlobalAlignerMode']
        }

        self.scale_recovery = MultiMethodScaleRecovery()

    def extract_metadata(self, image_path):
        try:
            with Image.open(image_path) as img:
                exif = img._getexif() or {}
                metadata = {}
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    metadata[tag] = value
                metadata['ImageWidth'] = img.width
                metadata['ImageHeight'] = img.height
                return sanitize_for_json(metadata)
        except Exception as e:
            print(f"Warning: Could not extract metadata from {image_path}: {e}")
            return {}

    async def calculate_dimensions(self, image_paths, imu_data=None, metadata=None, timestamps=None, output_dir="output"):
        if len(image_paths) < 2:
            raise ValueError("At least 2 images required for 3D reconstruction")
        
        print(f"\n🔍 Processing {len(image_paths)} images...")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        start_time = time.time()
        
        try:
            # Load images using DUSt3R
            print("📐 Loading images with DUSt3R...")
            images = self.dust3r_components['load_images'](image_paths, size=512, square_ok=False)
            print(f"✓ Loaded {len(images)} images successfully")
            
            # Create pairs
            print("🔗 Creating image pairs...")
            pairs = self.dust3r_components['make_pairs'](images, scene_graph='complete', prefilter=None, symmetrize=True)
            print(f"✓ Created {len(pairs)} image pairs")
            
            # Run DUSt3R inference
            print("🧠 Running DUSt3R neural network inference...")
            try:
                from torch.amp import autocast
                autocast_context = autocast(device_type=self.device, enabled=True)
            except ImportError:
                from torch.cuda.amp import autocast
                autocast_context = autocast(enabled=True)
            
            with autocast_context:
                inference_result = self.dust3r_components['inference'](pairs, self.model, self.device, batch_size=1, verbose=True)
            
            print("✓ Inference completed successfully")
            
            # Global alignment
            print("🌐 Performing global alignment...")
            mode = self.dust3r_components['GlobalAlignerMode'].PointCloudOptimizer
            scene = self.dust3r_components['global_aligner'](inference_result, device=self.device, mode=mode)
            
            # Extract 3D points with proper gradient handling
            print("📊 Extracting 3D points...")
            points3d = scene.get_pts3d()
            all_points = []
            
            for points in points3d:
                if points is not None:
                    valid_points = points[~torch.isnan(points).any(dim=-1)]
                    if len(valid_points) > 0:
                        all_points.append(valid_points.detach().cpu().numpy())
            
            if not all_points:
                raise ValueError("No valid 3D points found in reconstruction")
            
            combined_points = np.concatenate(all_points, axis=0)
            print(f"✓ Extracted {len(combined_points):,} valid 3D points")
            
            # Extract camera poses
            try:
                camera_poses_raw = scene.get_im_poses()
                camera_poses = []
                for pose in camera_poses_raw:
                    if pose is not None:
                        camera_poses.append(pose.detach().cpu().numpy())
            except:
                camera_poses = []
            
            # Create reconstruction result for scale recovery
            reconstruction_result = {
                'point_cloud': combined_points,
                'camera_poses': camera_poses,
                'depth_maps': self.extract_depth_maps(points3d),
                'quality_score': self.calculate_quality_score(combined_points, camera_poses)
            }
            
            # Multi-method scale recovery
            print("📏 Performing multi-method scale recovery...")
            scale_result = await self.scale_recovery.estimate_scale(
                images=images,
                imu_data=imu_data,
                metadata=metadata,
                timestamps=timestamps,
                reconstruction=reconstruction_result
            )
            
            # Calculate final dimensions with scale
            scale_factor = scale_result['scale_factor']
            scaled_points = combined_points * scale_factor * 100  # Convert to cm
            
            min_coords = np.min(scaled_points, axis=0)
            max_coords = np.max(scaled_points, axis=0)
            dimensions = max_coords - min_coords
            center = (min_coords + max_coords) / 2
            volume = np.prod(dimensions)
            
            # Calculate surface area (approximate)
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(scaled_points)
                surface_area = hull.area
            except:
                surface_area = 0.0
            
            # Prepare comprehensive results
            results = {
                "success": True,
                "processing_time": time.time() - start_time,
                "scale_recovery": scale_result,
                "reconstruction_stats": {
                    "num_images": len(image_paths),
                    "num_3d_points": len(combined_points),
                    "reconstruction_quality": reconstruction_result['quality_score']
                },
                "dimensions_metric": {
                    "width_cm": float(dimensions[0]),
                    "height_cm": float(dimensions[1]),
                    "depth_cm": float(dimensions[2]),
                    "volume_cm3": float(volume),
                    "surface_area_cm2": float(surface_area)
                },
                "bounding_box": {
                    "min": min_coords.tolist(),
                    "max": max_coords.tolist(),
                    "center": center.tolist()
                },
                "confidence_metrics": {
                    "overall_confidence": scale_result['confidence'],
                    "scale_methods_used": scale_result['methods_used'],
                    "geometric_quality": reconstruction_result['quality_score']
                },
                "metadata": {
                    "device": self.device,
                    "model": "DUSt3R",
                    "image_metadata": [self.extract_metadata(img_path) for img_path in image_paths] if isinstance(image_paths[0], (str, Path)) else []
                }
            }
            
            # Sanitize results for JSON serialization
            results = sanitize_for_json(results)
            
            # Save results
            result_file = output_path / "dimension_results.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            np.save(output_path / "point_cloud.npy", scaled_points)
            
            print(f"\n✅ Processing completed in {results['processing_time']:.1f} seconds")
            print(f"📊 Results:")
            print(f"   Width:  {dimensions[0]:.1f} cm")
            print(f"   Height: {dimensions[1]:.1f} cm")
            print(f"   Depth:  {dimensions[2]:.1f} cm")
            print(f"   Volume: {volume:.1f} cm³")
            print(f"   Points: {len(combined_points):,}")
            print(f"   Scale Confidence: {scale_result['confidence']:.1%}")
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
    
    def extract_depth_maps(self, points3d):
        depth_maps = []
        for points in points3d:
            if points is not None:
                depths = points[:, :, 2].detach().cpu().numpy()
                depth_maps.append(depths)
        return depth_maps
    
    def calculate_quality_score(self, point_cloud, camera_poses):
        try:
            if len(point_cloud) < 1000:
                density_score = 0.3
            elif len(point_cloud) < 10000:
                density_score = 0.6
            else:
                density_score = 1.0
            
            if len(camera_poses) < 3:
                pose_score = 0.4
            else:
                pose_score = min(1.0, len(camera_poses) / 10.0)
            
            quality = 0.6 * density_score + 0.4 * pose_score
            return quality
        except Exception:
            return 0.5

# FastAPI Application
app = FastAPI(
    title="High-Accuracy 3D Dimension Calculator",
    description="Multi-method scale recovery for precise object dimension measurement",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize calculator
calculator = DUSt3RDimensionCalculator()

@app.post("/calculate-dimensions-advanced")
async def calculate_dimensions_advanced(
    images: List[UploadFile] = File(...),
    imu_data: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    timestamps: Optional[str] = Form(None)
):
    try:
        imu_data_parsed = json.loads(imu_data) if imu_data else None
        metadata_parsed = json.loads(metadata) if metadata else None
        timestamps_parsed = json.loads(timestamps) if timestamps else None
        
        if len(images) < 2:
            raise HTTPException(status_code=400, detail="At least 2 images required")
        if len(images) > 25:
            raise HTTPException(status_code=400, detail="Maximum 25 images allowed")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_paths = []
            
            for i, image_file in enumerate(images):
                if not image_file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="Only image files allowed")
                
                file_path = temp_path / f"image_{i:03d}.jpg"
                with open(file_path, 'wb') as f:
                    f.write(await image_file.read())
                image_paths.append(str(file_path))
            
            results = await calculator.calculate_dimensions(
                image_paths,
                imu_data=imu_data_parsed,
                metadata=metadata_parsed,
                timestamps=timestamps_parsed,
                output_dir=str(temp_path / "output")
            )
            
            return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "advanced-dimension-calculator",
        "gpu_available": torch.cuda.is_available(),
        "device": calculator.device
    }

def main():
    parser = argparse.ArgumentParser(description='High-Accuracy 3D Dimension Calculator')
    parser.add_argument('images', nargs='+', help='Path to input images')
    parser.add_argument('--imu', type=str, help='IMU data JSON file')
    parser.add_argument('--metadata', type=str, help='Metadata JSON file')
    parser.add_argument('--timestamps', type=str, help='Timestamps JSON file')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    imu_data = None
    if args.imu:
        with open(args.imu, 'r') as f:
            imu_data = json.load(f)
    
    metadata = None
    if args.metadata:
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
    
    timestamps = None
    if args.timestamps:
        with open(args.timestamps, 'r') as f:
            timestamps = json.load(f)
    
    print("🖥️  System Information:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    calc = DUSt3RDimensionCalculator(device=args.device)
    
    results = asyncio.run(calc.calculate_dimensions(
        args.images,
        imu_data=imu_data,
        metadata=metadata,
        timestamps=timestamps,
        output_dir=args.output
    ))
    
    return 0 if results["success"] else 1

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ['--help', '-h']:
        exit(main())
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
