"""
MiDaS v3.1 adapter.

Wraps the MiDaS depth estimation model for robust relative depth estimation.
Supports multiple model variants including DPT-BEiT and DPT-Swin2.
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple, Dict

from ..model_registry import DepthModelAdapter, DepthOutput, register_model
from ...core.config import ModelSelectionConfig

logger = logging.getLogger(__name__)

# MiDaS model configurations
MIDAS_CONFIGS: Dict[str, Dict] = {
    "dpt_beit_large_512": {
        "model_type": "dpt_beit_large_512",
        "resolution": (512, 512),
        "hub_id": "intel-isl/MiDaS",
    },
    "dpt_swin2_large_384": {
        "model_type": "dpt_swin2_large_384",
        "resolution": (384, 384),
        "hub_id": "intel-isl/MiDaS",
    },
    "dpt_large_384": {
        "model_type": "dpt_large_384",
        "resolution": (384, 384),
        "hub_id": "intel-isl/MiDaS",
    },
    "dpt_hybrid_384": {
        "model_type": "dpt_hybrid_384",
        "resolution": (384, 384),
        "hub_id": "intel-isl/MiDaS",
    },
    "midas_v21_384": {
        "model_type": "midas_v21_384",
        "resolution": (384, 384),
        "hub_id": "intel-isl/MiDaS",
    }
}


@register_model("midas_v3")
class MiDaSAdapter(DepthModelAdapter):
    """
    Adapter for MiDaS v3.1 depth estimation model.

    MiDaS provides robust relative depth estimation trained on
    multiple mixed datasets. Supports various backbone architectures.

    Variants:
    - dpt_beit_large_512: Best quality, larger (default)
    - dpt_swin2_large_384: High quality, efficient
    - dpt_large_384: Good quality, balanced
    - dpt_hybrid_384: Faster, reasonable quality
    - midas_v21_384: Legacy model, fastest
    """

    def __init__(self, config: ModelSelectionConfig):
        super().__init__(config)
        self.variant = getattr(config, 'midas_variant', 'dpt_beit_large_512')
        if self.variant not in MIDAS_CONFIGS:
            logger.warning(f"Unknown variant {self.variant}, defaulting to 'dpt_beit_large_512'")
            self.variant = 'dpt_beit_large_512'
        self.model_config = MIDAS_CONFIGS[self.variant]
        self.transform = None

    @property
    def name(self) -> str:
        return f"midas_{self.variant}"

    @property
    def native_resolution(self) -> Tuple[int, int]:
        return self.model_config["resolution"]

    def load_model(self, device: torch.device) -> None:
        """Load MiDaS model."""
        logger.info(f"Loading MiDaS ({self.variant})...")

        try:
            # Load MiDaS from torch.hub
            self.model = torch.hub.load(
                self.model_config["hub_id"],
                self.model_config["model_type"],
                trust_repo=True
            )

            # Load transforms
            midas_transforms = torch.hub.load(
                self.model_config["hub_id"],
                "transforms",
                trust_repo=True
            )

            # Select appropriate transform
            if "dpt" in self.variant:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            self.model = self.model.to(device)
            self.model.eval()
            self.device = device
            self._is_loaded = True

            logger.info(f"MiDaS ({self.variant}) loaded on {device}")

        except Exception as e:
            logger.error(f"Failed to load MiDaS from torch.hub: {e}")

            # Fallback: try loading from HuggingFace
            try:
                from transformers import DPTImageProcessor, DPTForDepthEstimation

                model_id = "Intel/dpt-large"  # Fallback to DPT-Large
                self.processor = DPTImageProcessor.from_pretrained(model_id)
                self.model = DPTForDepthEstimation.from_pretrained(model_id)
                self.model = self.model.to(device)
                self.model.eval()
                self.device = device
                self._is_loaded = True
                self._use_hf_fallback = True

                logger.info("Loaded DPT-Large from HuggingFace as MiDaS fallback")

            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load MiDaS. Ensure torch.hub is accessible. Error: {e}"
                )

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

        # Check if using HuggingFace fallback
        if hasattr(self, '_use_hf_fallback') and self._use_hf_fallback:
            return self._estimate_depth_hf_fallback(images, return_confidence)

        # Ensure images are on correct device
        if images.device != self.device:
            images = images.to(self.device, non_blocking=True)

        batch_size = images.shape[0]
        original_size = (images.shape[2], images.shape[3])

        results = []

        # MiDaS expects different input format
        for i in range(batch_size):
            image = images[i]

            # Convert to numpy for MiDaS transform
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

            # Apply MiDaS transform
            if self.transform is not None:
                input_batch = self.transform(image_np).to(self.device)
            else:
                # Manual preprocessing
                target_size = self.native_resolution
                input_tensor = F.interpolate(
                    image.unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                input_batch = (input_tensor - mean) / std

            # Run inference
            with torch.no_grad():
                depth = self.model(input_batch)

            # Handle output shape
            if depth.dim() == 3:
                depth = depth.squeeze(0)
            elif depth.dim() == 4:
                depth = depth.squeeze(0).squeeze(0)

            # Resize to original size
            depth_resized = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode='bilinear',
                align_corners=False
            ).squeeze()

            confidence = None
            if return_confidence:
                confidence = self.get_confidence_map(depth_resized)

            results.append(DepthOutput(
                depth_map=depth_resized,
                confidence_map=confidence,
                native_scale=False
            ))

        return results

    def _estimate_depth_hf_fallback(
        self,
        images: torch.Tensor,
        return_confidence: bool
    ) -> List[DepthOutput]:
        """Fallback estimation using HuggingFace DPT."""
        batch_size = images.shape[0]
        original_size = (images.shape[2], images.shape[3])

        # Preprocess
        target_size = (384, 384)
        images_resized = F.interpolate(
            images.to(self.device),
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        images_normalized = (images_resized - mean) / std

        with torch.no_grad():
            outputs = self.model(images_normalized)
            depth_maps = outputs.predicted_depth

        if depth_maps.dim() == 4:
            depth_maps = depth_maps.squeeze(1)

        results = []
        for i in range(batch_size):
            depth = depth_maps[i]

            depth_resized = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode='bilinear',
                align_corners=False
            ).squeeze()

            confidence = None
            if return_confidence:
                confidence = self.get_confidence_map(depth_resized)

            results.append(DepthOutput(
                depth_map=depth_resized,
                confidence_map=confidence,
                native_scale=False
            ))

        return results
