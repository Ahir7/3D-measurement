import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MLModelUtils:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}
    
    def load_depth_estimation_model(self, model_name: str = "MiDaS") -> Optional[object]:
        """Load a depth estimation model for scale recovery"""
        try:
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]
            
            if model_name == "MiDaS":
                return self._load_midas_model()
            elif model_name == "DPT":
                return self._load_dpt_model()
            else:
                logger.warning(f"Unknown depth model: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load depth model {model_name}: {e}")
            return None
    
    def _load_midas_model(self):
        """Load MiDaS depth estimation model"""
        try:
            import torch.hub
            
            # Load MiDaS model
            model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
            model.to(self.device)
            model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            transform = midas_transforms.default_transform
            
            model_dict = {
                'model': model,
                'transform': transform,
                'name': 'MiDaS'
            }
            
            self.loaded_models['MiDaS'] = model_dict
            logger.info("MiDaS depth model loaded successfully")
            
            return model_dict
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            return None
    
    def _load_dpt_model(self):
        """Load DPT depth estimation model"""
        try:
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            
            processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
            
            model.to(self.device)
            model.eval()
            
            model_dict = {
                'model': model,
                'processor': processor,
                'name': 'DPT'
            }
            
            self.loaded_models['DPT'] = model_dict
            logger.info("DPT depth model loaded successfully")
            
            return model_dict
            
        except Exception as e:
            logger.error(f"Failed to load DPT model: {e}")
            return None
    
    def estimate_depth_from_image(self, image: np.ndarray, model_name: str = "MiDaS") -> Optional[np.ndarray]:
        """Estimate depth map from a single image"""
        try:
            model_dict = self.load_depth_estimation_model(model_name)
            if not model_dict:
                return None
            
            if model_name == "MiDaS":
                return self._midas_inference(image, model_dict)
            elif model_name == "DPT":
                return self._dpt_inference(image, model_dict)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None
    
    def _midas_inference(self, image: np.ndarray, model_dict: Dict) -> np.ndarray:
        """Run MiDaS inference on image"""
        model = model_dict['model']
        transform = model_dict['transform']
        
        # Preprocess image
        input_tensor = transform(image).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = model(input_tensor)
            
        # Convert to numpy
        depth_map = prediction.squeeze().cpu().numpy()
        
        return depth_map
    
    def _dpt_inference(self, image: np.ndarray, model_dict: Dict) -> np.ndarray:
        """Run DPT inference on image"""
        model = model_dict['model']
        processor = model_dict['processor']
        
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Convert to numpy
        depth_map = predicted_depth.squeeze().cpu().numpy()
        
        return depth_map
    
    def calculate_scale_from_depth(self, depth_map: np.ndarray, 
                                 focal_length: float, 
                                 sensor_width: float,
                                 image_width: int) -> float:
        """Calculate scale factor from depth map and camera parameters"""
        try:
            # Calculate average depth
            valid_depths = depth_map[depth_map > 0]
            if len(valid_depths) == 0:
                return 1.0
            
            avg_depth = np.median(valid_depths)
            
            # Calculate field of view
            fov_radians = 2 * np.arctan(sensor_width / (2 * focal_length))
            
            # Calculate scale
            scale = (image_width * avg_depth * np.tan(fov_radians / 2)) / (sensor_width / 1000)
            
            return scale
            
        except Exception as e:
            logger.error(f"Scale calculation from depth failed: {e}")
            return 1.0
    
    def evaluate_model_confidence(self, depth_map: np.ndarray) -> float:
        """Evaluate confidence in depth estimation"""
        try:
            if depth_map is None or depth_map.size == 0:
                return 0.0
            
            # Calculate depth consistency metrics
            valid_depths = depth_map[depth_map > 0]
            if len(valid_depths) < 100:  # Too few valid depths
                return 0.2
            
            # Check depth distribution
            depth_std = np.std(valid_depths)
            depth_mean = np.mean(valid_depths)
            
            # Coefficient of variation as confidence metric
            cv = depth_std / depth_mean if depth_mean > 0 else 1.0
            
            # Convert to confidence (lower CV = higher confidence)
            confidence = max(0.1, min(1.0, 1.0 - cv))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Model confidence evaluation failed: {e}")
            return 0.3
    
    def cleanup_models(self):
        """Clean up loaded models to free memory"""
        try:
            for model_name in list(self.loaded_models.keys()):
                del self.loaded_models[model_name]
            
            self.loaded_models.clear()
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("ML models cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")
