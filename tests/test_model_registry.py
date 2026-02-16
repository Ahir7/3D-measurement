"""
Unit tests for the model registry and adapter system.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.depth.model_registry import (
    DepthModelAdapter,
    DepthOutput,
    ModelRegistry,
    get_registry,
    register_model
)
from src.core.config import ModelSelectionConfig


class TestDepthOutput:
    """Tests for DepthOutput dataclass."""

    def test_creation(self):
        """Test basic DepthOutput creation."""
        depth_map = torch.randn(480, 640)
        output = DepthOutput(depth_map=depth_map)

        assert output.depth_map.shape == (480, 640)
        assert output.confidence_map is None
        assert output.native_scale is False

    def test_with_confidence(self):
        """Test DepthOutput with confidence map."""
        depth_map = torch.randn(480, 640)
        confidence_map = torch.rand(480, 640)

        output = DepthOutput(
            depth_map=depth_map,
            confidence_map=confidence_map,
            native_scale=True
        )

        assert output.confidence_map is not None
        assert output.native_scale is True


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_registry_creation(self):
        """Test registry creation."""
        registry = ModelRegistry()
        assert registry is not None
        assert len(registry.list_models()) == 0

    def test_register_model(self):
        """Test model registration."""
        registry = ModelRegistry()

        class MockAdapter(DepthModelAdapter):
            @property
            def name(self):
                return "mock"

            def load_model(self, device):
                pass

            def estimate_depth(self, images, return_confidence=False):
                return []

        registry.register("mock_model", MockAdapter)
        assert "mock_model" in registry.list_models()

    def test_register_duplicate_warns(self):
        """Test that registering duplicate model warns."""
        registry = ModelRegistry()

        class MockAdapter(DepthModelAdapter):
            @property
            def name(self):
                return "mock"

            def load_model(self, device):
                pass

            def estimate_depth(self, images, return_confidence=False):
                return []

        registry.register("test", MockAdapter)

        # Should warn but not fail
        registry.register("test", MockAdapter)
        assert "test" in registry.list_models()

    def test_unregister_model(self):
        """Test model unregistration."""
        registry = ModelRegistry()

        class MockAdapter(DepthModelAdapter):
            @property
            def name(self):
                return "mock"

            def load_model(self, device):
                pass

            def estimate_depth(self, images, return_confidence=False):
                return []

        registry.register("to_remove", MockAdapter)
        assert "to_remove" in registry.list_models()

        registry.unregister("to_remove")
        assert "to_remove" not in registry.list_models()

    def test_get_nonexistent_model_raises(self):
        """Test that getting nonexistent model raises KeyError."""
        registry = ModelRegistry()

        with pytest.raises(KeyError):
            registry.get_model("nonexistent")

    def test_list_loaded_models(self):
        """Test listing loaded models."""
        registry = ModelRegistry()
        # Initially no models loaded
        assert len(registry.list_loaded_models()) == 0


class TestDepthModelAdapter:
    """Tests for DepthModelAdapter base class."""

    def test_get_confidence_map(self):
        """Test default confidence map computation."""
        config = ModelSelectionConfig()

        class ConcreteAdapter(DepthModelAdapter):
            @property
            def name(self):
                return "test"

            def load_model(self, device):
                self._is_loaded = True

            def estimate_depth(self, images, return_confidence=False):
                return []

        adapter = ConcreteAdapter(config)

        # Create a simple depth map
        depth_map = torch.ones(100, 100) * 5.0

        confidence = adapter.get_confidence_map(depth_map)

        assert confidence.shape == depth_map.shape
        assert confidence.min() >= 0
        assert confidence.max() <= 1

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        config = ModelSelectionConfig()

        class ConcreteAdapter(DepthModelAdapter):
            @property
            def name(self):
                return "test"

            def load_model(self, device):
                self._is_loaded = True

            def estimate_depth(self, images, return_confidence=False):
                return []

        adapter = ConcreteAdapter(config)
        assert adapter.is_loaded is False

        adapter.load_model(torch.device('cpu'))
        assert adapter.is_loaded is True


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_singleton(self):
        """Test that get_registry returns same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_register_decorator(self):
        """Test register_model decorator."""
        @register_model("decorated_model")
        class DecoratedAdapter(DepthModelAdapter):
            @property
            def name(self):
                return "decorated"

            def load_model(self, device):
                pass

            def estimate_depth(self, images, return_confidence=False):
                return []

        registry = get_registry()
        assert "decorated_model" in registry.list_models()
