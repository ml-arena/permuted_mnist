"""
PermutedMNIST environment module
"""
from .permuted_mnist import PermutedMNISTEnv

# Export the environment as the default
Env = PermutedMNISTEnv

__all__ = ['PermutedMNISTEnv', 'Env']