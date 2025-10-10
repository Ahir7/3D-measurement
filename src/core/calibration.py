"""
Camera calibration for 3D measurement system.

Handles camera intrinsics and extrinsics calibration on GPU.
"""

import torch
import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    k1: float = 0.0  # Radial distortion
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0  # Tangential distortion
    p2: float = 0.0
    width: int = 0
    height: int = 0
    
    def to_matrix(self) -> torch.Tensor:
        """
        Convert to 3x3 camera matrix on GPU.
        
        Returns:
            Camera intrinsics matrix as GPU tensor
        """
        K = torch.tensor([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], device='cuda', dtype=torch.float32)
        return K
    
    def get_distortion_coeffs(self) -> torch.Tensor:
        """
        Get distortion coefficients as GPU tensor.
        
        Returns:
            Distortion coefficients [k1, k2, p1, p2, k3]
        """
        dist = torch.tensor(
            [self.k1, self.k2, self.p1, self.p2, self.k3],
            device='cuda',
            dtype=torch.float32
        )
        return dist
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'k1': self.k1,
            'k2': self.k2,
            'k3': self.k3,
            'p1': self.p1,
            'p2': self.p2,
            'width': self.width,
            'height': self.height
        }


class CameraCalibrator:
    """GPU-accelerated camera calibration."""
    
    def __init__(self, device: str = 'cuda:0'):
        """
        Initialize camera calibrator.
        
        Args:
            device: GPU device identifier
            
        Raises:
            RuntimeError: If GPU is not available
        """
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for calibration")
        
        self.device = torch.device(device)
        logger.info(f"Camera calibrator initialized on {device}")
    
    def calibrate_from_images(
        self,
        images: List[np.ndarray],
        pattern_size: Tuple[int, int] = (9, 6),
        square_size_mm: float = 25.0
    ) -> Optional[CameraIntrinsics]:
        """
        Calibrate camera from checkerboard images.
        
        Args:
            images: List of calibration images
            pattern_size: Checkerboard pattern size (columns, rows)
            square_size_mm: Size of checkerboard squares in mm
            
        Returns:
            CameraIntrinsics if calibration successful, None otherwise
        """
        logger.info(f"Starting calibration with {len(images)} images")
        
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:pattern_size[0],
            0:pattern_size[1]
        ].T.reshape(-1, 2)
        objp *= square_size_mm
        
        obj_points = []  # 3D points in real world
        img_points = []  # 2D points in image plane
        
        img_shape = None
        
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_shape = gray.shape[::-1]
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                obj_points.append(objp)
                img_points.append(corners_refined)
                logger.debug(f"Pattern found in image {i+1}")
            else:
                logger.warning(f"Pattern not found in image {i+1}")
        
        if len(obj_points) < 3:
            logger.error("Insufficient calibration images with detected patterns")
            return None
        
        # Perform calibration
        logger.info(f"Calibrating with {len(obj_points)} valid images")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            img_shape,
            None,
            None
        )
        
        if not ret:
            logger.error("Camera calibration failed")
            return None
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(obj_points)):
            img_points_reproj, _ = cv2.projectPoints(
                obj_points[i],
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs
            )
            error = cv2.norm(img_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
            total_error += error
        
        mean_error = total_error / len(obj_points)
        logger.info(f"Calibration successful - Mean reprojection error: {mean_error:.4f} pixels")
        
        # Create CameraIntrinsics object
        intrinsics = CameraIntrinsics(
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]),
            cy=float(camera_matrix[1, 2]),
            k1=float(dist_coeffs[0, 0]),
            k2=float(dist_coeffs[0, 1]),
            p1=float(dist_coeffs[0, 2]),
            p2=float(dist_coeffs[0, 3]),
            k3=float(dist_coeffs[0, 4]) if dist_coeffs.shape[1] > 4 else 0.0,
            width=img_shape[0],
            height=img_shape[1]
        )
        
        return intrinsics
    
    def estimate_from_exif(
        self,
        focal_length_mm: float,
        sensor_width_mm: float,
        image_width_px: int,
        image_height_px: int
    ) -> CameraIntrinsics:
        """
        Estimate camera intrinsics from EXIF data.
        
        Args:
            focal_length_mm: Focal length in millimeters
            sensor_width_mm: Sensor width in millimeters
            image_width_px: Image width in pixels
            image_height_px: Image height in pixels
            
        Returns:
            Estimated camera intrinsics
        """
        # Calculate focal length in pixels
        fx = (focal_length_mm / sensor_width_mm) * image_width_px
        fy = fx  # Assume square pixels
        
        # Principal point at image center
        cx = image_width_px / 2.0
        cy = image_height_px / 2.0
        
        intrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=image_width_px,
            height=image_height_px
        )
        
        logger.info(f"Estimated intrinsics from EXIF: fx={fx:.2f}, fy={fy:.2f}")
        return intrinsics
    
    def undistort_image_gpu(
        self,
        image: torch.Tensor,
        intrinsics: CameraIntrinsics
    ) -> torch.Tensor:
        """
        Undistort image on GPU.
        
        Args:
            image: Input image tensor on GPU [H, W, 3]
            intrinsics: Camera intrinsics with distortion
            
        Returns:
            Undistorted image tensor on GPU
        """
        # Convert to numpy for OpenCV (currently no pure PyTorch undistortion)
        image_np = image.cpu().numpy()
        
        K = intrinsics.to_matrix().cpu().numpy()
        dist = intrinsics.get_distortion_coeffs().cpu().numpy()
        
        # Undistort
        undistorted_np = cv2.undistort(image_np, K, dist)
        
        # Convert back to GPU tensor
        undistorted = torch.from_numpy(undistorted_np).to(self.device)
        
        return undistorted
    
    def save_calibration(
        self,
        intrinsics: CameraIntrinsics,
        filepath: Path
    ) -> None:
        """
        Save calibration to file.
        
        Args:
            intrinsics: Camera intrinsics to save
            filepath: Output file path (JSON)
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(intrinsics.to_dict(), f, indent=2)
        
        logger.info(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: Path) -> CameraIntrinsics:
        """
        Load calibration from file.
        
        Args:
            filepath: Input file path (JSON)
            
        Returns:
            Loaded camera intrinsics
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        intrinsics = CameraIntrinsics(**data)
        logger.info(f"Calibration loaded from {filepath}")
        
        return intrinsics


def validate_intrinsics(intrinsics: CameraIntrinsics) -> bool:
    """
    Validate camera intrinsics parameters.
    
    Args:
        intrinsics: Camera intrinsics to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check positive focal lengths
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        logger.error("Invalid focal lengths")
        return False
    
    # Check principal point within image bounds
    if not (0 <= intrinsics.cx <= intrinsics.width):
        logger.error("Principal point x outside image bounds")
        return False
    
    if not (0 <= intrinsics.cy <= intrinsics.height):
        logger.error("Principal point y outside image bounds")
        return False
    
    # Check reasonable aspect ratio
    aspect_ratio = intrinsics.fx / intrinsics.fy
    if not (0.5 <= aspect_ratio <= 2.0):
        logger.warning(f"Unusual aspect ratio: {aspect_ratio:.2f}")
    
    return True

