# PermutedMNIST Environment

A lightweight Gymnasium environment for meta-learning on MNIST dataset. This environment applies random transformations to MNIST images and challenges agents to adapt to these transformations.

## Installation

```bash
pip install -e .
```

## Dependencies

- gymnasium
- numpy
- mnist (lightweight MNIST data loader)
- scipy (for image transformations)

## Environment Description

In each episode:

1. The environment samples 11,000 random images from MNIST
2. 10,000 images are provided as labeled training data
3. 1,000 images are provided as unlabeled test data
4. A random transformation (rotation and shift) is applied to ALL images
5. The agent must predict labels for the test images
6. The reward is the classification accuracy on the test set
7. After each step, a new random transformation is applied

### Observation Space

The observation is a dictionary containing:
- `train_images`: Training images (10000, 28, 28) with values in [0, 1]
- `train_labels`: Labels for training images (10000,) with values in [0, 9]
- `test_images`: Test images (1000, 28, 28) with values in [0, 1]

### Action Space

The action should be predicted labels for test images:
- Shape: (1000,)
- Values: integers in [0, 9]

### Rewards

The reward is the classification accuracy on the test set (between 0 and 1).

### Additional Information

The info dictionary contains:
- `true_labels`: Ground truth labels for the test set
- `transform_params`: Current transformation parameters (angle, shift_x, shift_y)

## Example Usage

```python
import gymnasium as gym
import numpy as np

# Create environment
env = gym.make('PermutedMNIST-v0')

# Reset environment
obs, info = env.reset()

# Random agent example
for _ in range(100):
    # Random predictions
    action = np.random.randint(0, 10, size=1000)
    
    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Reward (accuracy): {reward:.3f}")
    print(f"Transform params: {info['transform_params']}")
```