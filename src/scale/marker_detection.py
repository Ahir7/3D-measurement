"""
Marker detection for scale recovery.

Detects and measures ArUco markers, QR codes, and AprilTags.
"""

import torch
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarkerType(Enum):
    """Supported marker types."""
    ARUCO = "aruco"
    QR = "qr"
    APRILTAG = "apriltag"


@dataclass
class DetectedMarker:
    """Detected marker information."""
    
    marker_type: MarkerType
    marker_id: int
    corners: np.ndarray  # 4x2 array of corner positions
    center: Tuple[float, float]
    size_pixels: float
    size_mm: float  # Known real-world size
    confidence: float
    pose: Optional[np.ndarray] = None  # 4x4 transformation matrix


class MarkerDetector:
    """GPU-accelerated marker detection."""
    
    def __init__(self, device: str = 'cuda:0'):
        """
        Initialize marker detector.
        
        Args:
            device: GPU device identifier
        """
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Initialize QR detector
        self.qr_detector = cv2.QRCodeDetector()
        
        logger.info(f"Marker detector initialized on {self.device}")
    
    def detect_markers(
        self,
        image: torch.Tensor,
        marker_types: List[MarkerType],
        known_sizes: Optional[Dict[int, float]] = None
    ) -> List[DetectedMarker]:
        """
        Detect markers in image.
        
        Args:
            image: Input image tensor [H, W, 3] or [H, W]
            marker_types: List of marker types to detect
            known_sizes: Dictionary mapping marker IDs to real sizes in mm
            
        Returns:
            List of detected markers
        """
        # Convert to numpy for OpenCV
        image_np = image.cpu().numpy()
        if image_np.dtype == np.float32 or image_np.dtype == np.float16:
            image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        all_markers = []
        
        # Detect ArUco markers
        if MarkerType.ARUCO in marker_types:
            aruco_markers = self._detect_aruco(gray, known_sizes)
            all_markers.extend(aruco_markers)
        
        # Detect QR codes
        if MarkerType.QR in marker_types:
            qr_markers = self._detect_qr(gray, known_sizes)
            all_markers.extend(qr_markers)
        
        # Detect AprilTags
        if MarkerType.APRILTAG in marker_types:
            apriltag_markers = self._detect_apriltag(gray, known_sizes)
            all_markers.extend(apriltag_markers)
        
        logger.info(f"Detected {len(all_markers)} markers")
        return all_markers
    
    def _detect_aruco(
        self,
        gray: np.ndarray,
        known_sizes: Optional[Dict[int, float]]
    ) -> List[DetectedMarker]:
        """Detect ArUco markers."""
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        markers = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                corner = corners[i][0]  # Shape: (4, 2)
                center = corner.mean(axis=0)
                
                # Estimate size in pixels (perimeter / 4)
                perimeter = np.linalg.norm(corner[0] - corner[1]) + \
                           np.linalg.norm(corner[1] - corner[2]) + \
                           np.linalg.norm(corner[2] - corner[3]) + \
                           np.linalg.norm(corner[3] - corner[0])
                size_pixels = perimeter / 4
                
                # Get known size if available
                size_mm = known_sizes.get(int(marker_id), 100.0) if known_sizes else 100.0
                
                marker = DetectedMarker(
                    marker_type=MarkerType.ARUCO,
                    marker_id=int(marker_id),
                    corners=corner,
                    center=(float(center[0]), float(center[1])),
                    size_pixels=float(size_pixels),
                    size_mm=size_mm,
                    confidence=0.9  # ArUco markers have high confidence
                )
                markers.append(marker)
                logger.debug(f"Detected ArUco marker ID={marker_id} at {center}")
        
        return markers
    
    def _detect_qr(
        self,
        gray: np.ndarray,
        known_sizes: Optional[Dict[int, float]]
    ) -> List[DetectedMarker]:
        """Detect QR codes."""
        retval, decoded_info, points, straight_qrcode = self.qr_detector.detectAndDecodeMulti(gray)
        
        markers = []
        if retval and points is not None:
            for i, qr_points in enumerate(points):
                if qr_points.shape[0] == 4:
                    corner = qr_points.reshape(4, 2)
                    center = corner.mean(axis=0)
                    
                    # Estimate size
                    perimeter = np.linalg.norm(corner[0] - corner[1]) + \
                               np.linalg.norm(corner[1] - corner[2]) + \
                               np.linalg.norm(corner[2] - corner[3]) + \
                               np.linalg.norm(corner[3] - corner[0])
                    size_pixels = perimeter / 4
                    
                    # Try to extract ID from decoded info
                    try:
                        marker_id = int(decoded_info[i]) if i < len(decoded_info) else i
                    except (ValueError, IndexError):
                        marker_id = i
                    
                    size_mm = known_sizes.get(marker_id, 100.0) if known_sizes else 100.0
                    
                    marker = DetectedMarker(
                        marker_type=MarkerType.QR,
                        marker_id=marker_id,
                        corners=corner,
                        center=(float(center[0]), float(center[1])),
                        size_pixels=float(size_pixels),
                        size_mm=size_mm,
                        confidence=0.85
                    )
                    markers.append(marker)
                    logger.debug(f"Detected QR code at {center}")
        
        return markers
    
    def _detect_apriltag(
        self,
        gray: np.ndarray,
        known_sizes: Optional[Dict[int, float]]
    ) -> List[DetectedMarker]:
        """Detect AprilTags."""
        # AprilTag detection requires apriltag library
        try:
            import apriltag
            
            detector = apriltag.Detector()
            detections = detector.detect(gray)
            
            markers = []
            for detection in detections:
                corners = detection.corners
                center = detection.center
                
                # Estimate size
                perimeter = np.linalg.norm(corners[0] - corners[1]) + \
                           np.linalg.norm(corners[1] - corners[2]) + \
                           np.linalg.norm(corners[2] - corners[3]) + \
                           np.linalg.norm(corners[3] - corners[0])
                size_pixels = perimeter / 4
                
                size_mm = known_sizes.get(detection.tag_id, 100.0) if known_sizes else 100.0
                
                marker = DetectedMarker(
                    marker_type=MarkerType.APRILTAG,
                    marker_id=detection.tag_id,
                    corners=corners,
                    center=(float(center[0]), float(center[1])),
                    size_pixels=float(size_pixels),
                    size_mm=size_mm,
                    confidence=detection.decision_margin / 100.0
                )
                markers.append(marker)
                logger.debug(f"Detected AprilTag ID={detection.tag_id} at {center}")
            
            return markers
            
        except ImportError:
            logger.warning("apriltag library not available, skipping AprilTag detection")
            return []
    
    def estimate_scale_from_marker(
        self,
        marker: DetectedMarker
    ) -> Tuple[float, float]:
        """
        Estimate scale factor from marker.
        
        Args:
            marker: Detected marker
            
        Returns:
            Tuple of (scale_factor, confidence)
            Scale factor converts pixels to millimeters
        """
        # Scale = real_size / pixel_size
        scale = marker.size_mm / marker.size_pixels
        
        # Confidence based on marker type and detection quality
        confidence = marker.confidence
        
        logger.debug(f"Marker scale: {scale:.4f} mm/pixel, confidence: {confidence:.2f}")
        
        return scale, confidence
    
    def estimate_pose(
        self,
        marker: DetectedMarker,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Estimate 3D pose of marker.
        
        Args:
            marker: Detected marker
            camera_matrix: Camera intrinsics matrix [3, 3]
            dist_coeffs: Distortion coefficients [5]
            
        Returns:
            4x4 transformation matrix or None if pose estimation fails
        """
        # Define 3D coordinates of marker corners
        half_size = marker.size_mm / 2.0
        obj_points = np.array([
            [-half_size, -half_size, 0],
            [half_size, -half_size, 0],
            [half_size, half_size, 0],
            [-half_size, half_size, 0]
        ], dtype=np.float32)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            marker.corners.astype(np.float32),
            camera_matrix,
            dist_coeffs
        )
        
        if not success:
            logger.warning(f"Failed to estimate pose for marker {marker.marker_id}")
            return None
        
        # Convert to transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.flatten()
        
        return pose
    
    def batch_detect(
        self,
        images: List[torch.Tensor],
        marker_types: List[MarkerType],
        known_sizes: Optional[Dict[int, float]] = None
    ) -> List[List[DetectedMarker]]:
        """
        Detect markers in multiple images.
        
        Args:
            images: List of image tensors
            marker_types: List of marker types to detect
            known_sizes: Dictionary of known marker sizes
            
        Returns:
            List of marker lists for each image
        """
        all_markers = []
        for i, image in enumerate(images):
            markers = self.detect_markers(image, marker_types, known_sizes)
            all_markers.append(markers)
            logger.debug(f"Image {i}: {len(markers)} markers detected")
        
        return all_markers

