"""
MetaMNIST environment for meta-learning on MNIST dataset.
"""
from gymnasium.envs.registration import register
import os

# Get the directory containing this file
PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# Register the environment
register(
    id='MetaMNIST-v0',
    entry_point='metamnist.env.metamnist:MetaMNISTEnv',
    max_episode_steps=100,
)