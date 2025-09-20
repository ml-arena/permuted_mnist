"""
PermutedMNIST environment for meta-learning on MNIST dataset.
"""
import os

# Get the directory containing this file
PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# Import the environment
from .env import Env

__all__ = ['Env', 'PKG_DIR']