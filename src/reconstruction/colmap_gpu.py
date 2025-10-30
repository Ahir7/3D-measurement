"""
GPU-accelerated COLMAP wrapper for 3D reconstruction.

This module provides a Python interface to COLMAP with GPU acceleration.
"""

import torch
import numpy as np
import subprocess
import logging
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import cv2

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False
    logging.warning("pycolmap not available, using subprocess fallback")

from ..core.config import COLMAPConfig
from ..core.calibration import CameraIntrinsics

logger = logging.getLogger(__name__)


@dataclass
class Reconstruction3D:
    """3D reconstruction result from COLMAP."""
    
    points: torch.Tensor  # 3D points [N, 3]
    colors: torch.Tensor  # Point colors [N, 3]
    camera_poses: List[torch.Tensor]  # Camera poses [num_images, 4, 4]
    camera_intrinsics: List[CameraIntrinsics]
    image_names: List[str]
    point_errors: Optional[torch.Tensor] = None  # Reprojection errors
    num_observations: Optional[torch.Tensor] = None  # Observations per point
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'num_points': len(self.points),
            'num_cameras': len(self.camera_poses),
            'points_shape': list(self.points.shape),
            'mean_point_error': float(self.point_errors.mean()) if self.point_errors is not None else None
        }


class COLMAPReconstructor:
    """GPU-accelerated COLMAP 3D reconstruction."""
    
    def __init__(self, config: COLMAPConfig, device: str = 'cuda:0'):
        """
        Initialize COLMAP reconstructor.
        
        Args:
            config: COLMAP configuration
            device: GPU device identifier
            
        Raises:
            RuntimeError: If GPU is not available or COLMAP not found
        """
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for COLMAP reconstruction")
        
        self.config = config
        self.device = torch.device(device)
        self.use_pycolmap = PYCOLMAP_AVAILABLE
        
        # Verify COLMAP installation
        if not self.use_pycolmap:
            self._verify_colmap_binary()
        
        logger.info(f"COLMAP reconstructor initialized on {device}")
        logger.info(f"Using {'pycolmap' if self.use_pycolmap else 'COLMAP binary'}")
    
    def _verify_colmap_binary(self) -> None:
        """Verify COLMAP binary is available."""
        try:
            result = subprocess.run(
                ['colmap', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"COLMAP version: {result.stdout.strip()}")
            else:
                raise RuntimeError("COLMAP binary not found or not working")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"COLMAP verification failed: {e}")
    
    @torch.amp.autocast(device_type='cuda', enabled=True)
    def reconstruct(
        self,
        images: List[torch.Tensor],
        image_paths: Optional[List[Path]] = None,
        known_intrinsics: Optional[CameraIntrinsics] = None
    ) -> Reconstruction3D:
        """
        Perform 3D reconstruction from images.
        
        Args:
            images: List of image tensors on GPU [H, W, 3]
            image_paths: Optional paths to image files
            known_intrinsics: Optional known camera intrinsics
            
        Returns:
            Reconstruction3D object with 3D points and camera poses
            
        Raises:
            RuntimeError: If reconstruction fails
        """
        logger.info(f"Starting 3D reconstruction with {len(images)} images")
        
        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Setup workspace structure
            images_dir = workspace / "images"
            database_path = workspace / "database.db"
            sparse_dir = workspace / "sparse"
            images_dir.mkdir(exist_ok=True)
            sparse_dir.mkdir(exist_ok=True)
            
            # Save images to disk if needed
            if image_paths is None:
                image_paths = self._save_images(images, images_dir)
            else:
                # Copy images to workspace
                for i, img_path in enumerate(image_paths):
                    shutil.copy(img_path, images_dir / f"image_{i:04d}.jpg")
            
            try:
                if self.use_pycolmap:
                    reconstruction = self._reconstruct_pycolmap(
                        images_dir,
                        database_path,
                        sparse_dir,
                        known_intrinsics
                    )
                else:
                    reconstruction = self._reconstruct_binary(
                        images_dir,
                        database_path,
                        sparse_dir,
                        known_intrinsics
                    )
                
                # Transfer to GPU
                reconstruction_gpu = self._transfer_to_gpu(reconstruction)
                
                logger.info(f"Reconstruction complete: {len(reconstruction_gpu.points)} points")
                return reconstruction_gpu
                
            except Exception as e:
                logger.error(f"Reconstruction failed: {e}")
                raise RuntimeError(f"COLMAP reconstruction failed: {e}")
    
    def _save_images(self, images: List[torch.Tensor], output_dir: Path) -> List[Path]:
        """Save GPU tensors as images."""
        image_paths = []
        for i, img_tensor in enumerate(images):
            img_np = img_tensor.cpu().numpy()
            if img_np.dtype == np.float32 or img_np.dtype == np.float16:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            
            img_path = output_dir / f"image_{i:04d}.jpg"
            cv2.imwrite(str(img_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            image_paths.append(img_path)
        
        return image_paths
    
    def _reconstruct_pycolmap(
        self,
        images_dir: Path,
        database_path: Path,
        sparse_dir: Path,
        known_intrinsics: Optional[CameraIntrinsics]
    ) -> Reconstruction3D:
        """Reconstruct using pycolmap library."""
        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! GPU reconstruction requires CUDA.")
        
        # Feature extraction with GPU - GPU ONLY MODE
        logger.info(f"🎮 Extracting features with GPU (CUDA:{self.config.gpu_index})...")
        logger.info("⚠️  GPU-ONLY MODE: pycolmap-cuda required")
        
        # Create Feature Extraction options (new API uses FeatureExtractionOptions)
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.sift.max_num_features = self.config.num_features
        
        # Force CUDA device
        device = pycolmap.Device.cuda
        
        logger.info(f"✅ Device: {device}, Max Features: {extraction_options.sift.max_num_features}")
        
        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(images_dir),
            extraction_options=extraction_options,
            device=device
        )
        logger.info("✅ GPU feature extraction completed!")
        
        # Feature matching with GPU - GPU ONLY MODE
        logger.info("🎮 Matching features with GPU...")
        
        # Use FeatureMatchingOptions which contains sift options
        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.sift.max_ratio = 0.8  # Standard SIFT ratio test
        matching_options.sift.cross_check = True
        
        logger.info(f"✅ Match options: ratio={matching_options.sift.max_ratio}, cross_check={matching_options.sift.cross_check}")
        
        pycolmap.match_exhaustive(
            database_path=str(database_path),
            matching_options=matching_options,
            device=device
        )
        logger.info("✅ GPU feature matching completed!")
        
        # Sparse reconstruction
        logger.info("Running sparse reconstruction...")
        reconstruction = pycolmap.incremental_mapping(
            database_path,
            images_dir,
            sparse_dir,
            options={
                'ba_refine_focal_length': self.config.ba_refine_focal_length,
                'ba_refine_principal_point': self.config.ba_refine_principal_point,
                'ba_refine_extra_params': self.config.ba_refine_extra_params,
                'min_num_matches': self.config.min_num_matches
            }
        )
        
        # Parse reconstruction
        return self._parse_pycolmap_reconstruction(reconstruction[0])
    
    def _reconstruct_binary(
        self,
        images_dir: Path,
        database_path: Path,
        sparse_dir: Path,
        known_intrinsics: Optional[CameraIntrinsics]
    ) -> Reconstruction3D:
        """Reconstruct using COLMAP binary."""
        # Feature extraction
        logger.info("Extracting features with GPU...")
        self._run_colmap_command([
            'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(images_dir),
            '--SiftExtraction.use_gpu', '1',
            '--SiftExtraction.gpu_index', self.config.gpu_index,
            '--SiftExtraction.max_num_features', str(self.config.num_features)
        ])
        
        # Feature matching
        logger.info("Matching features with GPU...")
        self._run_colmap_command([
            'exhaustive_matcher',
            '--database_path', str(database_path),
            '--SiftMatching.use_gpu', '1',
            '--SiftMatching.gpu_index', self.config.gpu_index,
            '--SiftMatching.max_num_matches', str(self.config.max_num_matches)
        ])
        
        # Sparse reconstruction
        logger.info("Running sparse reconstruction...")
        self._run_colmap_command([
            'mapper',
            '--database_path', str(database_path),
            '--image_path', str(images_dir),
            '--output_path', str(sparse_dir),
            '--Mapper.ba_refine_focal_length', '1' if self.config.ba_refine_focal_length else '0',
            '--Mapper.ba_refine_principal_point', '1' if self.config.ba_refine_principal_point else '0',
            '--Mapper.ba_refine_extra_params', '1' if self.config.ba_refine_extra_params else '0',
            '--Mapper.min_num_matches', str(self.config.min_num_matches)
        ])
        
        # Load reconstruction
        return self._load_binary_reconstruction(sparse_dir / '0')
    
    def _run_colmap_command(self, args: List[str]) -> None:
        """Run COLMAP command."""
        cmd = ['colmap'] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP command failed: {result.stderr}")
    
    def _parse_pycolmap_reconstruction(self, reconstruction) -> Reconstruction3D:
        """Parse pycolmap reconstruction object."""
        # Extract 3D points
        points3D = []
        colors = []
        point_errors = []
        num_observations = []
        
        for point3D_id, point3D in reconstruction.points3D.items():
            points3D.append(point3D.xyz)
            colors.append(point3D.color)
            point_errors.append(point3D.error)
            num_observations.append(point3D.track.length())
        
        points = np.array(points3D)
        colors = np.array(colors) / 255.0  # Normalize to [0, 1]
        point_errors = np.array(point_errors)
        num_observations = np.array(num_observations)
        
        # Extract camera poses
        camera_poses = []
        camera_intrinsics = []
        image_names = []
        
        for image_id, image in reconstruction.images.items():
            # Get pose matrix (pycolmap v3.x uses cam_from_world)
            if hasattr(image, 'cam_from_world'):
                # New API (v3.x) - cam_from_world is a method
                cam_from_world = image.cam_from_world()
                R = cam_from_world.rotation.matrix()
                t = cam_from_world.translation
            else:
                # Old API (v2.x)
                R = image.rotmat() if hasattr(image, 'rotmat') else image.rotation_matrix()
                t = image.tvec
            
            # Create 4x4 pose matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            camera_poses.append(pose)
            image_names.append(image.name if hasattr(image, 'name') else str(image_id))
            
            # Get intrinsics (handle camera model mappings correctly)
            camera = reconstruction.cameras[image.camera_id]
            params = camera.params
            model = getattr(camera, 'model', None) or getattr(camera, 'model_name', None)

            if isinstance(model, bytes):
                model = model.decode('utf-8')

            fx = fy = None
            cx = cy = None

            if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
                # COLMAP: [f, cx, cy, (k)] → fx=fy=f
                f = params[0] if len(params) > 0 else max(camera.width, camera.height)
                fx = f
                fy = f
                cx = params[1] if len(params) > 1 else camera.width / 2
                cy = params[2] if len(params) > 2 else camera.height / 2
            elif model in ('PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV', 'THIN_PRISM_FISHEYE'):
                # COLMAP: [fx, fy, cx, cy, ...]
                fx = params[0] if len(params) > 0 else max(camera.width, camera.height)
                fy = params[1] if len(params) > 1 else fx
                cx = params[2] if len(params) > 2 else camera.width / 2
                cy = params[3] if len(params) > 3 else camera.height / 2
            else:
                # Fallback: assume simple pinhole
                f = params[0] if len(params) > 0 else max(camera.width, camera.height)
                fx = f
                fy = f
                cx = params[1] if len(params) > 1 else camera.width / 2
                cy = params[2] if len(params) > 2 else camera.height / 2

            intrinsics = CameraIntrinsics(
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                width=camera.width,
                height=camera.height
            )
            camera_intrinsics.append(intrinsics)
        
        return Reconstruction3D(
            points=torch.from_numpy(points),
            colors=torch.from_numpy(colors),
            camera_poses=[torch.from_numpy(p) for p in camera_poses],
            camera_intrinsics=camera_intrinsics,
            point_errors=torch.from_numpy(point_errors),
            num_observations=torch.from_numpy(num_observations),
            image_names=image_names
        )
    
    def _load_binary_reconstruction(self, sparse_dir: Path) -> Reconstruction3D:
        """Load reconstruction from COLMAP binary format."""
        # Read points3D.bin
        points_file = sparse_dir / "points3D.bin"
        cameras_file = sparse_dir / "cameras.bin"
        images_file = sparse_dir / "images.bin"
        
        if not all([points_file.exists(), cameras_file.exists(), images_file.exists()]):
            raise RuntimeError("COLMAP reconstruction files not found")
        
        # Use pycolmap to read if available, otherwise parse manually
        if PYCOLMAP_AVAILABLE:
            reconstruction = pycolmap.Reconstruction(str(sparse_dir))
            return self._parse_pycolmap_reconstruction(reconstruction)
        else:
            raise RuntimeError("Binary parsing without pycolmap not implemented")
    
    def _transfer_to_gpu(self, reconstruction: Reconstruction3D) -> Reconstruction3D:
        """Transfer reconstruction data to GPU."""
        return Reconstruction3D(
            points=reconstruction.points.to(self.device),
            colors=reconstruction.colors.to(self.device),
            camera_poses=[p.to(self.device) for p in reconstruction.camera_poses],
            camera_intrinsics=reconstruction.camera_intrinsics,
            image_names=reconstruction.image_names,
            point_errors=reconstruction.point_errors.to(self.device) if reconstruction.point_errors is not None else None,
            num_observations=reconstruction.num_observations.to(self.device) if reconstruction.num_observations is not None else None
        )
    
    def save_reconstruction(
        self,
        reconstruction: Reconstruction3D,
        output_path: Path,
        format: str = 'ply'
    ) -> None:
        """
        Save reconstruction to file.
        
        Args:
            reconstruction: Reconstruction to save
            output_path: Output file path
            format: Output format ('ply', 'xyz', 'npy')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        points_cpu = reconstruction.points.cpu().numpy()
        colors_cpu = reconstruction.colors.cpu().numpy()
        
        if format == 'ply':
            self._save_ply(points_cpu, colors_cpu, output_path)
        elif format == 'xyz':
            self._save_xyz(points_cpu, colors_cpu, output_path)
        elif format == 'npy':
            np.save(output_path, np.hstack([points_cpu, colors_cpu]))
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Reconstruction saved to {output_path}")
    
    def _save_ply(self, points: np.ndarray, colors: np.ndarray, path: Path) -> None:
        """Save point cloud as PLY file."""
        header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        colors_uint8 = (colors * 255).astype(np.uint8)
        
        with open(path, 'w') as f:
            f.write(header)
            for point, color in zip(points, colors_uint8):
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
    
    def _save_xyz(self, points: np.ndarray, colors: np.ndarray, path: Path) -> None:
        """Save point cloud as XYZ file."""
        data = np.hstack([points, colors])
        np.savetxt(path, data, fmt='%.6f')

