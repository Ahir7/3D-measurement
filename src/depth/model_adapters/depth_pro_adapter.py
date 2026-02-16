"""
Apple Depth Pro adapter.

Wraps Apple's Depth Pro model for high-quality metric depth estimation.
Depth Pro provides metric depth with sharp boundaries.
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple

from ..model_registry import DepthModelAdapter, DepthOutput, register_model
from ...core.config import ModelSelectionConfig

logger = logging.getLogger(__name__)

# Check for depth_pro availability
DEPTH_PRO_AVAILABLE = None


def check_depth_pro():
    """Check if depth_pro library is available."""
    global DEPTH_PRO_AVAILABLE
    if DEPTH_PRO_AVAILABLE is None:
        try:
            import depth_pro
            DEPTH_PRO_AVAILABLE = True
        except ImportError:
            DEPTH_PRO_AVAILABLE = False
    return DEPTH_PRO_AVAILABLE


@register_model("depth_pro")
class DepthProAdapter(DepthModelAdapter):
    """
    Adapter for Apple Depth Pro model.

    Depth Pro provides high-quality metric depth estimation with
    sharp object boundaries. Requires the depth_pro package.

    Installation: pip install depth-pro
    """

    @property
    def name(self) -> str:
        return "depth_pro"

    @property
    def native_resolution(self) -> Tuple[int, int]:
        return (1536, 1536)  # Depth Pro native resolution

    def load_model(self, device: torch.device) -> None:
        """Load Depth Pro model."""
        if not check_depth_pro():
            raise RuntimeError(
                "depth_pro library required. Install with: pip install depth-pro"
            )

        import depth_pro

        logger.info("Loading Depth Pro model...")

        # Load model and transforms
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=str(device),
            precision=torch.float16 if 'cuda' in str(device) else torch.float32
        )
        self.model.eval()
        self.device = device

        self._is_loaded = True
        logger.info(f"Depth Pro loaded on {device}")

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

        import depth_pro

        # Ensure images are on correct device
        if images.device != self.device:
            images = images.to(self.device, non_blocking=True)

        batch_size = images.shape[0]
        original_size = (images.shape[2], images.shape[3])

        results = []

        # Process each image (Depth Pro works best per-image)
        for i in range(batch_size):
            image = images[i]

            # Convert to PIL for transform
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            from PIL import Image
            pil_image = Image.fromarray(image_np)

            # Apply transform
            transformed = self.transform(pil_image)

            # Run inference
            with torch.no_grad():
                prediction = self.model.infer(transformed)
                depth = prediction["depth"]  # Metric depth in meters
                focallength_px = prediction.get("focallength_px", None)

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
                depth_map=depth_resized.to(self.device),
                confidence_map=confidence,
                native_scale=True  # Depth Pro outputs metric depth
            ))

        return results

    def get_confidence_map(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence map with Depth Pro-specific heuristics.

        Depth Pro produces sharper boundaries, so we use edge-aware confidence.
        """
        import torch.nn.functional as F

        # Compute gradient magnitude
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               device=depth_map.device, dtype=depth_map.dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               device=depth_map.device, dtype=depth_map.dtype).view(1, 1, 3, 3)

        depth_4d = depth_map.unsqueeze(0).unsqueeze(0)
        grad_x = F.conv2d(depth_4d, sobel_x, padding=1)
        grad_y = F.conv2d(depth_4d, sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()

        # Lower gradient = higher confidence (smooth regions are more reliable)
        # Normalize gradient
        grad_norm = gradient_magnitude / (gradient_magnitude.max() + 1e-6)

        # Also consider local variance
        kernel_size = 5
        padding = kernel_size // 2
        depth_unfold = F.unfold(
            depth_4d,
            kernel_size=kernel_size,
            padding=padding
        )
        variance = depth_unfold.var(dim=1).view(depth_map.shape)

        # Combined confidence
        variance_conf = torch.exp(-variance * 10)
        gradient_conf = 1.0 - grad_norm * 0.5

        confidence = 0.7 * variance_conf + 0.3 * gradient_conf
        return torch.clamp(confidence, 0, 1)
