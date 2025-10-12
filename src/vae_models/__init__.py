"""
VAE Models Package

Contains all VAE-related classes and components.
"""

from .vae_base import (
    VAE,
    vae_loss,
    FraudDataHandler,
    ThresholdOptimizer,
    compute_reconstruction_errors
)

from .vae_trainer import VAEFraudTrainer
from .vae_tuner import VAEGridSearchTuner

__all__ = [
    'VAE',
    'vae_loss',
    'FraudDataHandler',
    'ThresholdOptimizer',
    'compute_reconstruction_errors',
    'VAEFraudTrainer',
    'VAEGridSearchTuner'
]
