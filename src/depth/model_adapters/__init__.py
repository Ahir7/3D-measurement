"""
Model adapters for various depth estimation architectures.

Each adapter wraps a specific depth estimation model to provide
a consistent interface through the DepthModelAdapter ABC.
"""

from .dpt_adapter import DPTAdapter
from .depth_pro_adapter import DepthProAdapter
from .depth_anything_adapter import DepthAnythingAdapter
from .midas_adapter import MiDaSAdapter

__all__ = [
    'DPTAdapter',
    'DepthProAdapter',
    'DepthAnythingAdapter',
    'MiDaSAdapter',
]
