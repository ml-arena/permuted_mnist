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

1. The environment samples 70,000 random images from MNIST
2. 60,000 images are provided as labeled training data
3. 10,000 images are provided as unlabeled test data
4. A random permutation is applied to ALL images
5. The agent must predict labels for the test images
6. The reward is the classification accuracy on the test set
7. After each step, a new random permutation is applied

### Observation Space

The observation is a dictionary containing:
- `train_images`: Training images (60000, 28, 28) with values in [0, 1]
- `train_labels`: Labels for training images (60000,) with values in [0, 9]
- `test_images`: Test images (10000, 28, 28) with values in [0, 1]

### Action Space

The action should be predicted labels for test images:
- Shape: (10000,)
- Values: integers in [0, 9]

### Rewards

The reward is the classification accuracy on the test set (between 0 and 1).


## Example Usage

```python
import gymnasium as gym
import numpy as np

# Create environment
env = gym.make('PermutedMNIST-v0')

# Reset environment
observation, info = env.reset()

# Random agent example
for _ in range(10):
    # 1 minute to train
    agent.train(observation['X_train'], observation['y_train'])
    # pred
    Y_pred = np.random.randint(0, 10, len(observation['X_test']))
    # Take step
    observation, reward, terminated, truncated, info = env.step(Y_pred)
    
    print(f"Reward (accuracy): {reward:.3f}")
```