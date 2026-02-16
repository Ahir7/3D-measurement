"""
DPT (Dense Prediction Transformer) adapter.

Wraps the Intel DPT-Large model for depth estimation.
This adapter preserves compatibility with the existing Metric3D implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple

from ..model_registry import DepthModelAdapter, DepthOutput, register_model
from ...core.config import ModelSelectionConfig

logger = logging.getLogger(__name__)

# Lazy import to avoid startup overhead
TRANSFORMERS_AVAILABLE = None


def check_transformers():
    """Check if transformers library is available."""
    global TRANSFORMERS_AVAILABLE
    if TRANSFORMERS_AVAILABLE is None:
        try:
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    return TRANSFORMERS_AVAILABLE


@register_model("dpt_large")
class DPTAdapter(DepthModelAdapter):
    """
    Adapter for Intel DPT-Large depth estimation model.

    DPT-Large is trained on MIX-6 dataset with metric depth supervision.
    It outputs inverse depth (disparity) which needs normalization.
    """

    @property
    def name(self) -> str:
        return "dpt_large"

    @property
    def native_resolution(self) -> Tuple[int, int]:
        return (384, 384)

    def load_model(self, device: torch.device) -> None:
        """Load DPT-Large model."""
        if not check_transformers():
            raise RuntimeError("transformers library required for DPT model")

        from transformers import DPTImageProcessor, DPTForDepthEstimation

        logger.info("Loading DPT-Large model...")

        model_id = "Intel/dpt-large"
        self.processor = DPTImageProcessor.from_pretrained(model_id)
        self.model = DPTForDepthEstimation.from_pretrained(model_id)

        # Move to device and set eval mode
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        self._is_loaded = True
        logger.info(f"DPT-Large loaded on {device}")

    @torch.amp.autocast(device_type='cuda', enabled=True)
    def estimate_depth(
        self,
        images: torch.Tensor,
        return_confidence: bool = False
    ) -> List[DepthOutput]:
        """
        Estimate depth from images.

        Args:
            images: Input images [B, C, H, W] in [0, 1] range
            return_confidence: Whether to compute confidence maps

        Returns:
            List of DepthOutput objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure images are on correct device
        if images.device != self.device:
            images = images.to(self.device, non_blocking=True)

        batch_size = images.shape[0]
        original_size = (images.shape[2], images.shape[3])

        # Preprocess: resize to model input size
        target_size = self.native_resolution
        images_resized = F.interpolate(
            images,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        images_normalized = (images_resized - mean) / std

        # Run inference
        with torch.no_grad():
            outputs = self.model(images_normalized)
            depth_maps = outputs.predicted_depth

        # Normalize shape to [B, H, W]
        if depth_maps.dim() == 4 and depth_maps.shape[1] == 1:
            depth_maps = depth_maps.squeeze(1)

        # Process each depth map
        results = []
        for i in range(batch_size):
            depth = depth_maps[i]

            # Resize to original size
            depth_resized = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode='bilinear',
                align_corners=False
            ).squeeze()

            # Compute confidence if requested
            confidence = None
            if return_confidence:
                confidence = self.get_confidence_map(depth_resized)

            results.append(DepthOutput(
                depth_map=depth_resized,
                confidence_map=confidence,
                native_scale=False  # DPT outputs relative depth
            ))

        return results
