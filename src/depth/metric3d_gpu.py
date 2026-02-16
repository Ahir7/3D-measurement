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

from ..core.config import Metric3DConfig, UncertaintyConfig

logger = logging.getLogger(__name__)


def ensure_bchw(images: torch.Tensor) -> torch.Tensor:
    """
    Ensure input tensor is in [B, C, H, W] with C in {1, 3}.
    Accepts shapes: [H, W], [C, H, W], [B, H, W], [H, W, 3], [B, H, W, 3], [B, C, H, W].
    Grayscale inputs (C=1) are repeated to RGB (C=3).
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(images)}")

    # Normalize dimensionality
    if images.dim() == 2:  # [H, W]
        images = images.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif images.dim() == 3:
        if images.shape[0] in (1, 3):  # [C, H, W]
            images = images.unsqueeze(0)  # [1, C, H, W]
        elif images.shape[-1] == 3:  # [H, W, 3]
            images = images.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
        else:  # [B, H, W]
            images = images.unsqueeze(1)  # [B, 1, H, W]
    elif images.dim() == 4:
        if images.shape[-1] == 3:  # [B, H, W, 3]
            images = images.permute(0, 3, 1, 2)  # [B, 3, H, W]
        elif images.shape[1] in (1, 3):  # [B, C, H, W]
            pass
        else:
            raise ValueError(f"Cannot infer channel dimension from shape {tuple(images.shape)}")
    else:
        raise ValueError(f"Unsupported input dim={images.dim()} shape={tuple(images.shape)}")

    # Ensure 3 channels
    channels = images.shape[1]
    if channels == 1:
        images = images.repeat(1, 3, 1, 1)
    elif channels == 3:
        pass
    elif channels == 4:  # RGBA -> RGB
        images = images[:, :3, ...]
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

    return images

@dataclass
class DepthEstimation:
    """Depth estimation result with uncertainty quantification."""

    depth_map: torch.Tensor  # Depth in meters [H, W]
    confidence_map: Optional[torch.Tensor] = None  # Confidence scores [H, W]

    # Uncertainty fields (new for accuracy enhancement)
    uncertainty_map: Optional[torch.Tensor] = None  # Combined uncertainty [H, W]
    mc_variance: Optional[torch.Tensor] = None  # MC Dropout variance [H, W]
    flip_consistency: Optional[torch.Tensor] = None  # Flip consistency score [H, W]

    # Model identification
    model_name: str = "dpt_large"  # Model identifier

    scale_factor: float = 1.0
    processing_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            'shape': list(self.depth_map.shape),
            'min_depth': float(self.depth_map.min()),
            'max_depth': float(self.depth_map.max()),
            'mean_depth': float(self.depth_map.mean()),
            'scale_factor': self.scale_factor,
            'processing_time': self.processing_time,
            'model_name': self.model_name
        }

        # Add uncertainty statistics if available
        if self.uncertainty_map is not None:
            result['uncertainty_stats'] = {
                'mean': float(self.uncertainty_map.mean()),
                'max': float(self.uncertainty_map.max()),
                'min': float(self.uncertainty_map.min())
            }

        if self.mc_variance is not None:
            result['mc_variance_mean'] = float(self.mc_variance.mean())

        if self.flip_consistency is not None:
            result['flip_consistency_mean'] = float(self.flip_consistency.mean())

        return result

    def get_weighted_confidence(self) -> torch.Tensor:
        """
        Get confidence weighted by uncertainty.

        Returns:
            Confidence map combining base confidence and uncertainty
        """
        if self.confidence_map is None:
            base_conf = torch.ones_like(self.depth_map)
        else:
            base_conf = self.confidence_map

        if self.uncertainty_map is not None:
            # Convert uncertainty to confidence weight
            uncertainty_weight = torch.exp(-self.uncertainty_map)
            return base_conf * uncertainty_weight

        return base_conf


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
        
        # torch.compile() DISABLED for GPUs with <8GB VRAM
        # First-time compilation takes 10-20 minutes and uses extra GPU memory
        # The speedup (2-3x) is not worth the overhead for smaller GPUs
        # Re-enable by changing 'if False' to 'if platform.system() != "Windows"'
        import platform
        if False and platform.system() != 'Windows':
            logger.info("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model compilation enabled")
        else:
            logger.info("torch.compile() disabled (recommended for GPUs with <8GB VRAM)")
    
    @torch.amp.autocast(device_type='cuda', enabled=True)
    def estimate_depth(
        self,
        images: torch.Tensor,
        return_confidence: bool = False,
        batch_size: int = 3  # Process 3 images at a time for 6GB GPU
    ) -> List[DepthEstimation]:
        """
        Estimate metric depth from images with batched processing for memory efficiency.
        
        Args:
            images: Input images tensor [B, 3, H, W] or [B, H, W, 3]
            return_confidence: Whether to compute confidence maps
            batch_size: Number of images to process at once (default 3 for 6GB GPU)
            
        Returns:
            List of DepthEstimation objects for each image
            
        Raises:
            RuntimeError: If estimation fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Log initial shape
        try:
            logger.debug(f"estimate_depth: input shape={tuple(images.shape)} dtype={images.dtype} device={images.device}")
        except Exception:
            pass

        # Ensure correct format [B, 3, H, W]
        images = ensure_bchw(images)

        logger.debug(f"estimate_depth: normalized input shape={tuple(images.shape)}")
        
        total_images = images.shape[0]
        logger.info(f"Estimating depth for {total_images} images in batches of {batch_size}")
        
        # Store original image sizes (H, W) for each image before preprocessing
        original_sizes = [(images.shape[2], images.shape[3]) for _ in range(total_images)]
        
        # Start timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        try:
            raw_depth_maps = []
            
            # Process images in batches
            for batch_idx in range(0, total_images, batch_size):
                batch_number = (batch_idx // batch_size) + 1
                batch_end = min(batch_idx + batch_size, total_images)
                batch_images = images[batch_idx:batch_end]
                current_batch_size = batch_images.shape[0]
                
                logger.debug(f"Processing batch {batch_number}/{(total_images + batch_size - 1)//batch_size} "
                           f"({current_batch_size} images)")
                
                # Preprocess
                images_processed = self._preprocess(batch_images)
                logger.debug(f"estimate_depth: preprocessed batch shape={tuple(images_processed.shape)}")
                
                # Run inference
                with torch.no_grad():
                    depth_maps = self._run_inference(images_processed)
                    try:
                        logger.debug(f"estimate_depth: raw depth output shape={tuple(depth_maps.shape)}")
                    except Exception:
                        pass

                # Normalize output shape to [B, H, W]
                if depth_maps.dim() == 4 and depth_maps.shape[1] == 1:
                    depth_maps = depth_maps[:, 0, :, :]
                elif depth_maps.dim() > 3:
                    depth_maps = depth_maps.reshape(depth_maps.shape[0], depth_maps.shape[-2], depth_maps.shape[-1])

                # Keep raw depth maps to compute global normalization stats
                for i in range(current_batch_size):
                    raw_depth_maps.append(depth_maps[i].detach().to('cpu', dtype=torch.float32))
                
                # Clear GPU memory after each batch
                del images_processed, depth_maps
                if batch_number % 4 == 0:
                    torch.cuda.empty_cache()
                    logger.debug(f"Batch {batch_number} complete, periodic cache cleanup")

            # Build normalization stats once for the full image set
            normalization_stats = self._compute_global_normalization_stats(raw_depth_maps)

            # Post-process in original order
            all_results = []
            for img_idx, raw_depth in enumerate(raw_depth_maps):
                target_size = original_sizes[img_idx]

                depth_map = self._postprocess(
                    raw_depth.to(self.device, non_blocking=True),
                    target_size=target_size,
                    normalization_stats=normalization_stats
                )
                logger.debug(f"estimate_depth: postprocessed depth shape={tuple(depth_map.shape)} target={target_size}")

                confidence_map = None
                if return_confidence:
                    confidence_map = self._compute_confidence(depth_map)

                all_results.append(DepthEstimation(
                    depth_map=depth_map,
                    confidence_map=confidence_map,
                    model_name=self.config.model_name,
                    scale_factor=1.0
                ))
            
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
            # Periodic final cache cleanup only for larger jobs
            if total_images >= 12:
                torch.cuda.empty_cache()
    
    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for depth estimation.
        
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            Preprocessed images
        """
        # Ensure device and dtype
        if images.device != self.device:
            images = images.to(self.device, non_blocking=True)
        if not torch.is_floating_point(images):
            images = images.float()

        logger.debug(f"_preprocess: input batch shape={tuple(images.shape)} dtype={images.dtype} device={images.device}")

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
        
        logger.debug(f"_preprocess: output batch shape={tuple(images_normalized.shape)}")

        return images_normalized
    
    def _run_inference(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run model inference.
        
        Args:
            images: Preprocessed images
            
        Returns:
            Raw depth predictions
        """
        logger.debug(f"_run_inference: input shape={tuple(images.shape)}")
        if hasattr(self.model, 'forward'):
            outputs = self.model(images)
            if hasattr(outputs, 'predicted_depth'):
                depth = outputs.predicted_depth
            else:
                depth = outputs
        else:
            depth = self.model(images)
        try:
            logger.debug(f"_run_inference: output shape={tuple(depth.shape)}")
        except Exception:
            pass

        return depth
    
    def _postprocess(
        self,
        depth: torch.Tensor,
        target_size: Tuple[int, int],
        normalization_stats: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Post-process depth map.
        
        Args:
            depth: Raw depth prediction
            target_size: Target output size (H, W)
            
        Returns:
            Processed depth map in meters
        """
        # Ensure 2D [H, W]: collapse any leading dims (batch/channel) by averaging
        if depth.dim() < 2:
            raise ValueError(f"Expected at least 2D depth map, got shape {depth.shape}")
        
        h, w = depth.shape[-2], depth.shape[-1]
        depth = depth.reshape(-1, h, w).mean(dim=0)
        logger.debug(f"_postprocess: collapsed to 2D shape={tuple(depth.shape)}")
        
        # Resize to target size - add batch and channel dims [1, 1, H, W]
        depth_resized = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze()  # Remove batch and channel dims back to [H, W]
        
        # Normalize to metric scale
        depth_normalized = self._normalize_depth(depth_resized, normalization_stats=normalization_stats)
        
        # Clip to valid range
        depth_clipped = torch.clamp(
            depth_normalized,
            self.config.min_depth,
            self.config.max_depth
        )
        
        return depth_clipped
    
    def _normalize_depth(
        self,
        depth: torch.Tensor,
        normalization_stats: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Normalize DPT-Large depth to metric scale.
        
        DPT-Large is trained on multiple datasets (MIX-6) with metric depth.
        The model outputs inverse depth (disparity), which we need to convert
        to absolute metric depth.
        
        Args:
            depth: Raw depth values from DPT-Large
            
        Returns:
            Depth in meters
        """
        mode = getattr(self.config, 'depth_normalization_mode', 'global_percentile')

        if mode == 'none':
            depth_metric = depth
        else:
            if mode == 'global_percentile' and normalization_stats is not None:
                p_low = torch.tensor(normalization_stats['p_low'], device=depth.device, dtype=depth.dtype)
                p_high = torch.tensor(normalization_stats['p_high'], device=depth.device, dtype=depth.dtype)
            else:
                if mode == 'global_percentile':
                    logger.warning("Global depth normalization selected but stats unavailable; falling back to per-image percentiles")
                q = torch.tensor(
                    [self.config.percentile_low, self.config.percentile_high],
                    device=depth.device,
                    dtype=depth.dtype
                )
                p_low, p_high = torch.quantile(depth, q)

            denom = torch.clamp(p_high - p_low, min=1e-6)
            depth_norm = torch.clamp((depth - p_low) / denom, 0, 1)
            depth_metric = depth_norm * (self.config.far_depth - self.config.near_depth) + self.config.near_depth
        
        # Final clipping to physically reasonable range
        depth_metric = torch.clamp(
            depth_metric, 
            self.config.min_depth,
            self.config.max_depth
        )
        
        return depth_metric * self.config.depth_scale_factor

    def _compute_global_normalization_stats(
        self,
        raw_depth_maps: List[torch.Tensor]
    ) -> Optional[Dict[str, float]]:
        """Compute robust global percentile stats across the full image set."""
        if self.config.depth_normalization_mode != 'global_percentile':
            return None

        sampled_values = []
        max_samples_per_map = 20000

        for depth in raw_depth_maps:
            valid = depth[torch.isfinite(depth)]
            if valid.numel() == 0:
                continue

            if valid.numel() > max_samples_per_map:
                indices = torch.randperm(valid.numel())[:max_samples_per_map]
                valid = valid[indices]
            sampled_values.append(valid)

        if not sampled_values:
            return None

        all_values = torch.cat(sampled_values)
        q = torch.tensor([self.config.percentile_low, self.config.percentile_high], dtype=all_values.dtype)
        p_low, p_high = torch.quantile(all_values, q)

        stats = {
            'p_low': float(p_low.item()),
            'p_high': float(p_high.item())
        }
        logger.info(
            f"Global depth normalization stats: p_low={stats['p_low']:.4f}, "
            f"p_high={stats['p_high']:.4f}"
        )
        return stats
    
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

    @torch.amp.autocast(device_type='cuda', enabled=True)
    def estimate_with_uncertainty(
        self,
        images: torch.Tensor,
        batch_size: int = 3
    ) -> List[DepthEstimation]:
        """
        Estimate depth with uncertainty quantification.

        Runs MC Dropout and flip consistency to compute uncertainty maps
        in addition to depth estimates.

        Args:
            images: Input images tensor [B, 3, H, W] or [B, H, W, 3]
            batch_size: Number of images to process at once

        Returns:
            List of DepthEstimation objects with uncertainty fields populated
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Get uncertainty config
        uncertainty_config = getattr(self.config, 'uncertainty', None)
        if uncertainty_config is None:
            # Fallback to standard estimation
            return self.estimate_depth(images, return_confidence=True, batch_size=batch_size)

        # Import uncertainty module
        from .uncertainty import DepthUncertaintyEstimator

        # Ensure correct format
        images = ensure_bchw(images)

        total_images = images.shape[0]
        logger.info(f"Estimating depth with uncertainty for {total_images} images")

        # First get base depth estimates
        base_results = self.estimate_depth(images, return_confidence=True, batch_size=batch_size)

        # Initialize uncertainty estimator
        uncertainty_estimator = DepthUncertaintyEstimator(uncertainty_config)

        # Compute uncertainty for each image
        for i, result in enumerate(base_results):
            img_batch = images[i:i+1]

            try:
                uncertainty_estimate = uncertainty_estimator.estimate(
                    img_batch,
                    self.model,
                    depth_maps=result.depth_map.unsqueeze(0),
                    inference_fn=lambda x: self._run_inference(self._preprocess(x))
                )

                # Populate uncertainty fields
                result.uncertainty_map = uncertainty_estimate.combined_uncertainty
                result.mc_variance = uncertainty_estimate.mc_variance
                result.flip_consistency = uncertainty_estimate.flip_consistency

            except Exception as e:
                logger.warning(f"Uncertainty estimation failed for image {i}: {e}")
                # Keep result without uncertainty

        return base_results

    def _compute_mc_dropout_uncertainty(
        self,
        images: torch.Tensor,
        n_passes: int = 10
    ) -> torch.Tensor:
        """
        Run N forward passes with dropout enabled for MC Dropout uncertainty.

        Args:
            images: Preprocessed images [B, C, H, W]
            n_passes: Number of forward passes

        Returns:
            Variance tensor [B, H, W]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Enable dropout
        self.model.train()
        for module in self.model.modules():
            if not isinstance(module, torch.nn.Dropout):
                module.eval()

        predictions = []
        with torch.no_grad():
            for _ in range(n_passes):
                depth = self._run_inference(images)
                if depth.dim() == 4:
                    depth = depth.squeeze(1)
                predictions.append(depth)

        # Restore eval mode
        self.model.eval()

        # Stack and compute variance
        predictions = torch.stack(predictions, dim=0)
        variance = predictions.var(dim=0)

        return variance

    def _compute_flip_consistency(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute depth consistency under horizontal flip.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Inconsistency map [B, H, W] (higher = more uncertain)
        """
        with torch.no_grad():
            # Original prediction
            images_processed = self._preprocess(images)
            depth_original = self._run_inference(images_processed)
            if depth_original.dim() == 4:
                depth_original = depth_original.squeeze(1)

            # Flipped prediction
            images_flipped = torch.flip(images, dims=[3])
            images_flipped_processed = self._preprocess(images_flipped)
            depth_flipped = self._run_inference(images_flipped_processed)
            if depth_flipped.dim() == 4:
                depth_flipped = depth_flipped.squeeze(1)

            # Flip back
            depth_flipped_back = torch.flip(depth_flipped, dims=[2])

            # Compute normalized difference
            depth_range = depth_original.max() - depth_original.min() + 1e-6
            inconsistency = torch.abs(depth_original - depth_flipped_back) / depth_range

        return inconsistency

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

