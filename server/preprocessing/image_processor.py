import cv2
import numpy as np
from PIL import Image, ExifTags
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
    async def preprocess_sequence(self, image_data: List[bytes], metadata: List[Dict]) -> List[Dict]:
        """
        Preprocess a sequence of images with lens distortion correction and enhancement
        """
        processed_images = []
        
        for i, (img_bytes, meta) in enumerate(zip(image_data, metadata)):
            try:
                # Convert bytes to PIL Image
                img = Image.open(io.BytesIO(img_bytes))
                
                # Apply preprocessing pipeline
                processed_img = await self._preprocess_single_image(img, meta)
                
                processed_images.append(processed_img)
                logger.info(f"Processed image {i+1}/{len(image_data)}")
                
            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                continue
                
        return processed_images
    
    async def _preprocess_single_image(self, img: Image.Image, metadata: Dict) -> Dict:
        """
        Preprocess a single image with various enhancements
        """
        # Convert to OpenCV format
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Apply lens distortion correction if camera parameters available
        if self._has_camera_calibration(metadata):
            cv_img = self._correct_lens_distortion(cv_img, metadata)
        
        # Enhance image quality
        cv_img = self._enhance_image_quality(cv_img)
        
        # Convert back to RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        return {
            'img': rgb_img,
            'metadata': metadata,
            'processed': True
        }
    
    def _has_camera_calibration(self, metadata: Dict) -> bool:
        """Check if camera calibration data is available"""
        required_keys = ['FocalLength', 'FocalLengthIn35mmFilm']
        return all(key in metadata for key in required_keys)
    
    def _correct_lens_distortion(self, img: np.ndarray, metadata: Dict) -> np.ndarray:
        """Apply lens distortion correction"""
        try:
            # Estimate camera matrix from metadata
            h, w = img.shape[:2]
            focal_length = float(metadata.get('FocalLength', 50))
            
            # Simplified camera matrix estimation
            fx = fy = focal_length * max(w, h) / 35  # 35mm equivalent
            cx, cy = w // 2, h // 2
            
            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Simplified distortion coefficients (would be better if calibrated)
            dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)
            
            # Apply undistortion
            undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
            return undistorted
            
        except Exception as e:
            logger.warning(f"Lens distortion correction failed: {e}")
            return img
    
    def _enhance_image_quality(self, img: np.ndarray) -> np.ndarray:
        """Apply image quality enhancements"""
        try:
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply slight sharpening
            kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel * 0.1)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return img
    
    def extract_quality_metrics(self, img: np.ndarray) -> Dict:
        """Extract image quality metrics"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = np.std(gray)
            
            return {
                'sharpness': float(sharpness),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'resolution': img.shape[:2]
            }
            
        except Exception as e:
            logger.error(f"Quality metrics extraction failed: {e}")
            return {}
