import torch
import numpy as np
from pathlib import Path
import sys
import logging

# Add DUSt3R to path
sys.path.append(str(Path(__file__).parent.parent.parent / "dust3r"))

from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

logger = logging.getLogger(__name__)

class DUSt3RProcessor:
    def __init__(self, model_path=None, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "data" / "models" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        
        print(f"Loading DUSt3R model from: {model_path}")
        try:
            # Import AsymmetricCroCo3DStereo only here to break circular import
            from dust3r.model import AsymmetricCroCo3DStereo
            if model_path.exists():
                print(f"Loading model from local checkpoint: {model_path}")
                # Load checkpoint using torch.load
                checkpoint = torch.load(str(model_path), map_location='cpu')
                
                # Create model instance
                self.model = AsymmetricCroCo3DStereo()
                
                # Load the state dict from checkpoint
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model = self.model.to(self.device)
                self.model.eval()
                print("✓ DUSt3R model loaded successfully from checkpoint")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            print(f"Failed to load from checkpoint, trying Hugging Face...")
            try:
                # Fallback to Hugging Face
                self.model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(self.device)
                self.model.eval()
                print("✓ DUSt3R model loaded from Hugging Face")
            except Exception as e2:
                raise RuntimeError(f"Failed to load DUSt3R model: {e2}")
    
    async def reconstruct_3d(self, processed_images):
        """
        Perform 3D reconstruction using DUSt3R
        """
        try:
            # Create image pairs
            pairs = make_pairs(processed_images, scene_graph='complete', prefilter=None, symmetrize=True)
            
            # Run inference
            try:
                from torch.amp import autocast
                autocast_context = autocast(device_type=self.device, enabled=True)
            except ImportError:
                from torch.cuda.amp import autocast
                autocast_context = autocast(enabled=True)
            
            with autocast_context:
                inference_result = inference(pairs, self.model, self.device, batch_size=1, verbose=True)
            
            # Global alignment
            mode = GlobalAlignerMode.PointCloudOptimizer
            scene = global_aligner(inference_result, device=self.device, mode=mode)
            
            # Extract results
            points3d = scene.get_pts3d()
            camera_poses = scene.get_im_poses()
            
            # Process point cloud
            all_points = []
            for points in points3d:
                if points is not None:
                    valid_points = points[~torch.isnan(points).any(dim=-1)]
                    if len(valid_points) > 0:
                        all_points.append(valid_points.detach().cpu().numpy())
            
            if not all_points:
                raise ValueError("No valid 3D points found")
            
            combined_points = np.concatenate(all_points, axis=0)
            
            # Extract camera poses
            poses = []
            for pose in camera_poses:
                if pose is not None:
                    poses.append(pose.detach().cpu().numpy())
            
            # Calculate quality metrics
            quality_score = self.calculate_quality_score(combined_points, poses)
            
            return {
                'point_cloud': combined_points,
                'camera_poses': poses,
                'depth_maps': self.extract_depth_maps(points3d),
                'quality_score': quality_score,
                'num_points': len(combined_points)
            }
            
        except Exception as e:
            raise RuntimeError(f"3D reconstruction failed: {e}")
    
    def extract_depth_maps(self, points3d):
        """
        Extract depth maps from 3D points
        """
        depth_maps = []
        for points in points3d:
            if points is not None:
                # Extract depth (z-coordinate)
                depths = points[:, :, 2].detach().cpu().numpy()
                depth_maps.append(depths)
        return depth_maps
    
    def calculate_quality_score(self, point_cloud, camera_poses):
        """
        Calculate reconstruction quality score
        """
        try:
            # Point cloud density
            if len(point_cloud) < 1000:
                density_score = 0.3
            elif len(point_cloud) < 10000:
                density_score = 0.6
            else:
                density_score = 1.0
            
            # Camera pose distribution
            if len(camera_poses) < 3:
                pose_score = 0.4
            else:
                pose_score = min(1.0, len(camera_poses) / 10.0)
            
            # Overall quality
            quality = 0.6 * density_score + 0.4 * pose_score
            
            return quality
            
        except Exception:
            return 0.5  # Default quality score
