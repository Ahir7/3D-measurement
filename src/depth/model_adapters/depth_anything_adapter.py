"""
Depth Anything V2 adapter.

Wraps the Depth Anything V2 model for robust relative depth estimation.
Supports multiple backbone sizes: ViT-S, ViT-B, ViT-L, ViT-G.
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple, Dict

from ..model_registry import DepthModelAdapter, DepthOutput, register_model
from ...core.config import ModelSelectionConfig

logger = logging.getLogger(__name__)

# Model configuration for different variants
DEPTH_ANYTHING_CONFIGS: Dict[str, Dict] = {
    "vits": {
        "encoder": "vits",
        "checkpoint": "depth_anything_v2_vits.pth",
        "resolution": (518, 518),
        "features": 64
    },
    "vitb": {
        "encoder": "vitb",
        "checkpoint": "depth_anything_v2_vitb.pth",
        "resolution": (518, 518),
        "features": 128
    },
    "vitl": {
        "encoder": "vitl",
        "checkpoint": "depth_anything_v2_vitl.pth",
        "resolution": (518, 518),
        "features": 256
    },
    "vitg": {
        "encoder": "vitg",
        "checkpoint": "depth_anything_v2_vitg.pth",
        "resolution": (518, 518),
        "features": 384
    }
}


@register_model("depth_anything_v2")
class DepthAnythingAdapter(DepthModelAdapter):
    """
    Adapter for Depth Anything V2 model.

    Depth Anything V2 provides robust relative depth estimation
    with excellent generalization across domains.

    Supports variants:
    - vits: Smallest, fastest
    - vitb: Balanced
    - vitl: Large, high quality (default)
    - vitg: Giant, highest quality (requires >8GB VRAM)
    """

    def __init__(self, config: ModelSelectionConfig):
        super().__init__(config)
        self.variant = getattr(config, 'depth_anything_variant', 'vitl')
        if self.variant not in DEPTH_ANYTHING_CONFIGS:
            logger.warning(f"Unknown variant {self.variant}, defaulting to 'vitl'")
            self.variant = 'vitl'
        self.model_config = DEPTH_ANYTHING_CONFIGS[self.variant]

    @property
    def name(self) -> str:
        return f"depth_anything_v2_{self.variant}"

    @property
    def native_resolution(self) -> Tuple[int, int]:
        return self.model_config["resolution"]

    def load_model(self, device: torch.device) -> None:
        """Load Depth Anything V2 model."""
        logger.info(f"Loading Depth Anything V2 ({self.variant})...")

        try:
            # Try to load from depth_anything_v2 package
            from depth_anything_v2.dpt import DepthAnythingV2

            model = DepthAnythingV2(
                encoder=self.model_config["encoder"],
                features=self.model_config["features"],
                out_channels=[48, 96, 192, 384] if self.variant != 'vitg' else [64, 128, 256, 512]
            )

            # Load checkpoint
            checkpoint_path = self.config.model_weights_dir / self.model_config["checkpoint"]
            if checkpoint_path.exists():
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}, using random weights")

            self.model = model.to(device)
            self.model.eval()
            self.device = device
            self._is_loaded = True

        except ImportError:
            # Fallback: try loading from HuggingFace transformers
            logger.info("depth_anything_v2 package not found, trying HuggingFace...")

            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                model_id = f"depth-anything/Depth-Anything-V2-{self.variant.upper()}"
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
                self.model = self.model.to(device)
                self.model.eval()
                self.device = device
                self._is_loaded = True
                self._use_hf = True
                logger.info(f"Loaded Depth Anything V2 from HuggingFace: {model_id}")

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Depth Anything V2. Install with: "
                    f"pip install depth-anything-v2 or install transformers. Error: {e}"
                )

        logger.info(f"Depth Anything V2 ({self.variant}) loaded on {device}")

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

        # Check if using HuggingFace version
        if hasattr(self, '_use_hf') and self._use_hf:
            return self._estimate_depth_hf(images, return_confidence)

        # Native depth_anything_v2 inference
        # Preprocess: resize to model input size
        target_size = self.native_resolution
        images_resized = F.interpolate(
            images,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        images_normalized = (images_resized - mean) / std

        # Run inference
        with torch.no_grad():
            depth_maps = self.model(images_normalized)

        # Normalize shape to [B, H, W]
        if depth_maps.dim() == 4 and depth_maps.shape[1] == 1:
            depth_maps = depth_maps.squeeze(1)

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

            confidence = None
            if return_confidence:
                confidence = self.get_confidence_map(depth_resized)

            results.append(DepthOutput(
                depth_map=depth_resized,
                confidence_map=confidence,
                native_scale=False
            ))

        return results

    def _estimate_depth_hf(
        self,
        images: torch.Tensor,
        return_confidence: bool
    ) -> List[DepthOutput]:
        """Estimate depth using HuggingFace model."""
        batch_size = images.shape[0]
        original_size = (images.shape[2], images.shape[3])

        results = []

        for i in range(batch_size):
            image = images[i]

            # Convert to PIL for processor
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            from PIL import Image
            pil_image = Image.fromarray(image_np)

            # Process with HF
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth

            # Resize to original size
            depth_resized = F.interpolate(
                depth.unsqueeze(1),
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
