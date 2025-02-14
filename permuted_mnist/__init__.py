"""
PermutedMNIST environment for meta-learning on MNIST dataset.
"""
from gymnasium.envs.registration import register
import os

# Get the directory containing this file
PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# Register the environment
register(
    id='PermutedMNIST-v0',
    entry_point='permuted_mnist.env.permuted_mnist:MetaMNISTEnv',
    max_episode_steps=100,
)