"""
GPU-accelerated Metric3D depth estimation.

Provides metric-scale depth prediction using Vision Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers library not available")

from ..core.config import Metric3DConfig

logger = logging.getLogger(__name__)


@dataclass
class DepthEstimation:
    """Depth estimation result."""
    
    depth_map: torch.Tensor  # Depth in meters [H, W]
    confidence_map: Optional[torch.Tensor] = None  # Confidence scores [H, W]
    scale_factor: float = 1.0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'shape': list(self.depth_map.shape),
            'min_depth': float(self.depth_map.min()),
            'max_depth': float(self.depth_map.max()),
            'mean_depth': float(self.depth_map.mean()),
            'scale_factor': self.scale_factor,
            'processing_time': self.processing_time
        }


class Metric3DEstimator:
    """GPU-accelerated Metric3D depth estimator."""
    
    def __init__(self, config: Metric3DConfig, device: str = 'cuda:0'):
        """
        Initialize Metric3D depth estimator.
        
        Args:
            config: Metric3D configuration
            device: GPU device identifier
            
        Raises:
            RuntimeError: If GPU is not available
        """
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for Metric3D")
        
        self.config = config
        self.device = torch.device(device)
        self.model = None
        self.processor = None
        
        # Initialize model
        self._load_model()
        
        logger.info(f"Metric3D estimator initialized on {device}")
    
    def _load_model(self) -> None:
        """Load and prepare Metric3D model."""
        logger.info(f"Loading Metric3D model: {self.config.model_name}")
        
        try:
            if self.config.model_name == "metric3d_vit_large":
                self._load_dpt_model()
            else:
                raise ValueError(f"Unknown model: {self.config.model_name}")
            
            # Compile model for faster inference (skip on Windows - Triton not supported)
            import platform
            if self.config.compile_model and hasattr(torch, 'compile') and platform.system() != 'Windows':
                logger.info("Compiling model with torch.compile()...")
                self.model = torch.compile(
                    self.model,
                    mode='reduce-overhead'
                )
                logger.info("Model compiled successfully")
            elif platform.system() == 'Windows':
                logger.info("Skipping torch.compile() on Windows (Triton not supported)")
            
            logger.info("Model loaded and ready")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _load_dpt_model(self) -> None:
        """Load DPT (Dense Prediction Transformer) model."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library required for DPT model")
        
        # Load pre-trained DPT model
        model_id = "Intel/dpt-large"
        self.processor = DPTImageProcessor.from_pretrained(model_id)
        self.model = DPTForDepthEstimation.from_pretrained(model_id)
        
        # Move to GPU and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Compile model with torch.compile() for speedup (skip on Windows - Triton not supported)
        import platform
        if platform.system() != 'Windows':
            logger.info("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model compilation enabled")
        else:
            logger.info("Skipping torch.compile() on Windows (Triton not supported)")
    
    @torch.amp.autocast(device_type='cuda', enabled=True)
    def estimate_depth(
        self,
        images: torch.Tensor,
        return_confidence: bool = False,
        batch_size: int = 2  # Process 2 images at a time for 4GB GPU
    ) -> List[DepthEstimation]:
        """
        Estimate metric depth from images with batched processing for memory efficiency.
        
        Args:
            images: Input images tensor [B, 3, H, W] or [B, H, W, 3]
            return_confidence: Whether to compute confidence maps
            batch_size: Number of images to process at once (default 2 for 4GB GPU)
            
        Returns:
            List of DepthEstimation objects for each image
            
        Raises:
            RuntimeError: If estimation fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Ensure correct format [B, 3, H, W]
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] == 3:  # [B, H, W, 3] -> [B, 3, H, W]
            images = images.permute(0, 3, 1, 2)
        
        total_images = images.shape[0]
        logger.info(f"Estimating depth for {total_images} images in batches of {batch_size}")
        
        # Start timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        try:
            all_results = []
            
            # Process images in batches
            for batch_idx in range(0, total_images, batch_size):
                batch_end = min(batch_idx + batch_size, total_images)
                batch_images = images[batch_idx:batch_end]
                current_batch_size = batch_images.shape[0]
                
                logger.debug(f"Processing batch {batch_idx//batch_size + 1}/{(total_images + batch_size - 1)//batch_size} "
                           f"({current_batch_size} images)")
                
                # Preprocess
                images_processed = self._preprocess(batch_images)
                
                # Run inference
                with torch.no_grad():
                    depth_maps = self._run_inference(images_processed)
                
                # Post-process and collect results
                for i in range(current_batch_size):
                    depth_map = self._postprocess(
                        depth_maps[i],
                        target_size=batch_images.shape[2:]
                    )
                    
                    confidence_map = None
                    if return_confidence:
                        confidence_map = self._compute_confidence(depth_map)
                    
                    all_results.append(DepthEstimation(
                        depth_map=depth_map,
                        confidence_map=confidence_map,
                        scale_factor=1.0
                    ))
                
                # Clear GPU memory after each batch
                del images_processed, depth_maps
                torch.cuda.empty_cache()
                logger.debug(f"Batch {batch_idx//batch_size + 1} complete, GPU memory freed")
            
            # Record timing
            end_time.record()
            torch.cuda.synchronize()
            total_time = start_time.elapsed_time(end_time) / 1000.0
            
            for result in all_results:
                result.processing_time = total_time / total_images
            
            logger.info(f"Depth estimation completed for {total_images} images in {total_time:.2f}s "
                       f"({total_time/total_images:.2f}s per image)")
            return all_results
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            raise RuntimeError(f"Depth estimation failed: {e}")
        
        finally:
            # Clear cache
            if batch_size > 1:
                torch.cuda.empty_cache()
    
    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for depth estimation.
        
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            Preprocessed images
        """
        # Resize to model input size
        target_size = self.config.input_size
        images_resized = F.interpolate(
            images,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize to [0, 1] if needed
        if images_resized.max() > 1.0:
            images_resized = images_resized / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        images_normalized = (images_resized - mean) / std
        
        return images_normalized
    
    def _run_inference(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run model inference.
        
        Args:
            images: Preprocessed images
            
        Returns:
            Raw depth predictions
        """
        if hasattr(self.model, 'forward'):
            outputs = self.model(images)
            if hasattr(outputs, 'predicted_depth'):
                depth = outputs.predicted_depth
            else:
                depth = outputs
        else:
            depth = self.model(images)
        
        return depth
    
    def _postprocess(
        self,
        depth: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Post-process depth map.
        
        Args:
            depth: Raw depth prediction
            target_size: Target output size (H, W)
            
        Returns:
            Processed depth map in meters
        """
        # Ensure 2D
        if depth.dim() == 3:
            depth = depth.squeeze(0)
        elif depth.dim() == 4:
            depth = depth.squeeze(0).squeeze(0)
        
        # Resize to target size
        depth_resized = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Normalize to metric scale
        depth_normalized = self._normalize_depth(depth_resized)
        
        # Clip to valid range
        depth_clipped = torch.clamp(
            depth_normalized,
            self.config.min_depth,
            self.config.max_depth
        )
        
        return depth_clipped
    
    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Normalize depth to metric scale.
        
        Args:
            depth: Raw depth values
            
        Returns:
            Depth in meters
        """
        # Simple normalization - should be calibrated per model
        depth_min = depth.min()
        depth_max = depth.max()
        
        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
            # Scale to reasonable depth range (0.5m to 10m)
            depth_metric = depth_norm * (10.0 - 0.5) + 0.5
        else:
            depth_metric = torch.ones_like(depth)
        
        return depth_metric * self.config.depth_scale_factor
    
    def _compute_confidence(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence map for depth estimation.
        
        Args:
            depth_map: Depth map [H, W]
            
        Returns:
            Confidence map [H, W] in range [0, 1]
        """
        # Compute local depth variance as inverse confidence
        kernel_size = 5
        padding = kernel_size // 2
        
        # Unfold for local patches
        depth_unfold = F.unfold(
            depth_map.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Compute variance
        variance = depth_unfold.var(dim=1)
        variance = variance.view(depth_map.shape)
        
        # Convert variance to confidence (inverse relationship)
        confidence = torch.exp(-variance)
        
        return confidence
    
    def estimate_depth_batch(
        self,
        images: List[torch.Tensor],
        batch_size: int = 4
    ) -> List[DepthEstimation]:
        """
        Estimate depth for multiple images in batches.
        
        Args:
            images: List of image tensors
            batch_size: Batch size for processing
            
        Returns:
            List of depth estimations
        """
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack(batch)
            
            results = self.estimate_depth(batch_tensor)
            all_results.extend(results)
        
        return all_results
    
    def save_depth_map(
        self,
        depth_estimation: DepthEstimation,
        output_path: Path,
        format: str = 'npy'
    ) -> None:
        """
        Save depth map to file.
        
        Args:
            depth_estimation: Depth estimation to save
            output_path: Output file path
            format: Output format ('npy', 'png', 'exr')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        depth_cpu = depth_estimation.depth_map.cpu().numpy()
        
        if format == 'npy':
            np.save(output_path, depth_cpu)
        elif format == 'png':
            # Normalize to 16-bit for PNG
            depth_normalized = (depth_cpu - depth_cpu.min()) / (depth_cpu.max() - depth_cpu.min())
            depth_uint16 = (depth_normalized * 65535).astype(np.uint16)
            import cv2
            cv2.imwrite(str(output_path), depth_uint16)
        elif format == 'exr':
            # OpenEXR format for floating point depth
            import OpenEXR
            import Imath
            header = OpenEXR.Header(depth_cpu.shape[1], depth_cpu.shape[0])
            header['channels'] = {'Y': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))}
            out = OpenEXR.OutputFile(str(output_path), header)
            out.writePixels({'Y': depth_cpu.tobytes()})
            out.close()
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Depth map saved to {output_path}")
    
    def visualize_depth(
        self,
        depth_estimation: DepthEstimation,
        colormap: str = 'turbo'
    ) -> torch.Tensor:
        """
        Create visualization of depth map.
        
        Args:
            depth_estimation: Depth estimation to visualize
            colormap: Matplotlib colormap name
            
        Returns:
            RGB visualization [H, W, 3]
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        depth_cpu = depth_estimation.depth_map.cpu().numpy()
        
        # Normalize
        depth_norm = (depth_cpu - depth_cpu.min()) / (depth_cpu.max() - depth_cpu.min())
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        colored = cmap(depth_norm)[:, :, :3]  # Remove alpha
        
        # Convert back to tensor
        colored_tensor = torch.from_numpy(colored).to(self.device)
        
        return colored_tensor

