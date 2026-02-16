"""
Training infrastructure for depth model fine-tuning.

Provides support for:
- Head-only fine-tuning (faster, memory efficient)
- Full model fine-tuning
- Domain adaptation training
"""

from .fine_tuning import (
    FineTuningTrainer,
    FineTuningConfig,
    TrainingMetrics
)

__all__ = [
    'FineTuningTrainer',
    'FineTuningConfig',
    'TrainingMetrics',
]
