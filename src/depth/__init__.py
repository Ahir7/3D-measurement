"""
Depth estimation modules.

Includes multi-model support through model registry, uncertainty estimation,
and the primary Metric3D estimator.
"""

from .metric3d_gpu import Metric3DEstimator, DepthEstimation
from .model_registry import (
    DepthModelAdapter,
    DepthOutput,
    ModelRegistry,
    get_registry,
    register_model
)
from .uncertainty import (
    UncertaintyEstimate,
    MCDropoutEstimator,
    FlipConsistencyEstimator,
    UncertaintyFusion,
    DepthUncertaintyEstimator
)

# Import adapters to trigger registration
from . import model_adapters

__all__ = [
    'Metric3DEstimator',
    'DepthEstimation',
    'DepthModelAdapter',
    'DepthOutput',
    'ModelRegistry',
    'get_registry',
    'register_model',
    'UncertaintyEstimate',
    'MCDropoutEstimator',
    'FlipConsistencyEstimator',
    'UncertaintyFusion',
    'DepthUncertaintyEstimator',
]
