"""
Data infrastructure for synthetic data generation and domain adaptation.

Provides hooks for:
- Synthetic depth data generation (Blender/Omniverse)
- Domain randomization for synthetic-to-real transfer
- Ground truth depth generation
"""

from .synthetic_pipeline import (
    SyntheticDataGenerator,
    DomainRandomization,
    SyntheticScene
)

__all__ = [
    'SyntheticDataGenerator',
    'DomainRandomization',
    'SyntheticScene',
]
