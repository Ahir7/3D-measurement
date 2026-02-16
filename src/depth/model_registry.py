"""
Model registry for multi-model depth estimation.

Provides a unified interface for managing multiple depth estimation models
with automatic fallback and model selection capabilities.
"""

import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Tuple
from pathlib import Path

from ..core.config import ModelSelectionConfig

logger = logging.getLogger(__name__)


@dataclass
class DepthOutput:
    """Standardized depth estimation output."""
    depth_map: torch.Tensor  # [H, W] depth in relative units
    confidence_map: Optional[torch.Tensor] = None  # [H, W] confidence scores
    native_scale: bool = False  # True if depth is metric scale


class DepthModelAdapter(ABC):
    """
    Abstract base class for depth model adapters.

    All depth models should be wrapped with an adapter that implements
    this interface to ensure consistent behavior across the system.
    """

    def __init__(self, config: ModelSelectionConfig):
        """
        Initialize the adapter.

        Args:
            config: Model selection configuration
        """
        self.config = config
        self.model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        self._is_loaded = False

    @property
    def name(self) -> str:
        """Return the model name identifier."""
        raise NotImplementedError

    @property
    def requires_rgb(self) -> bool:
        """Return True if model requires RGB input (vs grayscale)."""
        return True

    @property
    def supports_batch(self) -> bool:
        """Return True if model supports batch processing."""
        return True

    @property
    def native_resolution(self) -> Tuple[int, int]:
        """Return the model's native input resolution (H, W)."""
        return (384, 384)

    @abstractmethod
    def load_model(self, device: torch.device) -> None:
        """
        Load the model to the specified device.

        Args:
            device: Target device (cuda or cpu)
        """
        pass

    @abstractmethod
    def estimate_depth(
        self,
        images: torch.Tensor,
        return_confidence: bool = False
    ) -> List[DepthOutput]:
        """
        Estimate depth from input images.

        Args:
            images: Input images tensor [B, C, H, W] normalized to [0, 1]
            return_confidence: Whether to compute confidence maps

        Returns:
            List of DepthOutput objects, one per image
        """
        pass

    def get_confidence_map(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence map from depth map.

        Default implementation uses local variance as inverse confidence.
        Subclasses may override with model-specific confidence estimation.

        Args:
            depth_map: Depth map tensor [H, W]

        Returns:
            Confidence map [H, W] in range [0, 1]
        """
        import torch.nn.functional as F

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

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_loaded = False
            torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {self.name}")

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._is_loaded

    def enable_dropout(self) -> None:
        """Enable dropout for MC Dropout uncertainty estimation."""
        if self.model is not None:
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()

    def disable_dropout(self) -> None:
        """Disable dropout (restore eval mode)."""
        if self.model is not None:
            self.model.eval()


class ModelRegistry:
    """
    Registry for managing multiple depth estimation models.

    Supports model registration, lazy loading, automatic fallback,
    and model switching.
    """

    def __init__(self):
        """Initialize the model registry."""
        self._adapters: Dict[str, Type[DepthModelAdapter]] = {}
        self._instances: Dict[str, DepthModelAdapter] = {}
        self._active_model: Optional[str] = None
        self._device: Optional[torch.device] = None

    def register(self, name: str, adapter_class: Type[DepthModelAdapter]) -> None:
        """
        Register a model adapter class.

        Args:
            name: Model identifier
            adapter_class: Adapter class (not instance)
        """
        if name in self._adapters:
            logger.warning(f"Overwriting existing adapter for: {name}")
        self._adapters[name] = adapter_class
        logger.info(f"Registered model adapter: {name}")

    def unregister(self, name: str) -> None:
        """
        Unregister a model adapter.

        Args:
            name: Model identifier to remove
        """
        if name in self._adapters:
            del self._adapters[name]
            if name in self._instances:
                self._instances[name].unload_model()
                del self._instances[name]
            logger.info(f"Unregistered model adapter: {name}")

    def get_model(
        self,
        name: str,
        config: Optional[ModelSelectionConfig] = None,
        device: Optional[torch.device] = None
    ) -> DepthModelAdapter:
        """
        Get a model adapter instance.

        Creates and loads the model if not already loaded.

        Args:
            name: Model identifier
            config: Optional model configuration
            device: Target device

        Returns:
            Loaded model adapter instance

        Raises:
            KeyError: If model is not registered
            RuntimeError: If model fails to load
        """
        if name not in self._adapters:
            raise KeyError(f"Model not registered: {name}. Available: {self.list_models()}")

        # Create instance if needed
        if name not in self._instances:
            config = config or ModelSelectionConfig()
            adapter = self._adapters[name](config)
            self._instances[name] = adapter

        # Load model if needed
        adapter = self._instances[name]
        if not adapter.is_loaded:
            device = device or self._device or torch.device('cuda:0')
            try:
                adapter.load_model(device)
                logger.info(f"Loaded model: {name} on {device}")
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")
                raise RuntimeError(f"Failed to load model {name}: {e}")

        return adapter

    def list_models(self) -> List[str]:
        """
        List all registered model names.

        Returns:
            List of model identifiers
        """
        return list(self._adapters.keys())

    def list_loaded_models(self) -> List[str]:
        """
        List currently loaded models.

        Returns:
            List of loaded model identifiers
        """
        return [name for name, adapter in self._instances.items() if adapter.is_loaded]

    def set_device(self, device: torch.device) -> None:
        """
        Set default device for loading models.

        Args:
            device: Default device
        """
        self._device = device

    def unload_all(self) -> None:
        """Unload all loaded models."""
        for name, adapter in self._instances.items():
            if adapter.is_loaded:
                adapter.unload_model()
        logger.info("Unloaded all models")

    def get_with_fallback(
        self,
        primary: str,
        fallback: Optional[str] = None,
        config: Optional[ModelSelectionConfig] = None,
        device: Optional[torch.device] = None
    ) -> DepthModelAdapter:
        """
        Get model with automatic fallback on failure.

        Args:
            primary: Primary model name
            fallback: Fallback model name (optional)
            config: Model configuration
            device: Target device

        Returns:
            Loaded model adapter

        Raises:
            RuntimeError: If both primary and fallback fail
        """
        try:
            return self.get_model(primary, config, device)
        except Exception as e:
            if fallback is not None and fallback != primary:
                logger.warning(f"Primary model {primary} failed, trying fallback {fallback}: {e}")
                return self.get_model(fallback, config, device)
            raise


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """
    Get the global model registry instance.

    Returns:
        Global ModelRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


def register_model(name: str) -> callable:
    """
    Decorator for registering model adapters.

    Usage:
        @register_model("my_model")
        class MyModelAdapter(DepthModelAdapter):
            ...

    Args:
        name: Model identifier

    Returns:
        Decorator function
    """
    def decorator(adapter_class: Type[DepthModelAdapter]) -> Type[DepthModelAdapter]:
        get_registry().register(name, adapter_class)
        return adapter_class
    return decorator
