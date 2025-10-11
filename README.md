# PermutedMNIST Environment

A competitive meta-learning challenge environment for the [ML Arena Permuted MNIST Competition](https://ml-arena.com/viewcompetition/8).

## Competition Overview

This is a fast adaptation challenge where agents must quickly learn to classify MNIST digits that have been randomly permuted. The key challenge: **both pixels AND labels are randomly shuffled** for each task, requiring agents to adapt from scratch within strict time and resource constraints.

### Competition Link
🏆 **Submit your agent at:** https://ml-arena.com/viewcompetition/8

## Installation

```bash
pip install -e .
```

## Competition Rules & Constraints

### Resource Limits
Your agent will be evaluated under strict constraints:
- **Time Limit:** 1 minute maximum for training AND prediction per task
- **Memory Limit:** 4 GB RAM maximum
- **CPU Limit:** 2 CPU cores only (no GPU)

### Evaluation Process
1. **Initial Evaluation:** 10 runs with different random permutations
2. **Extended Evaluation:** Top leaderboard players face additional runs for final ranking
3. **Scoring:** Average accuracy across all runs determines your position

## The Challenge

### What Makes This Hard?

In each task, the environment applies **two types of permutations**:

1. **Pixel Permutation:** All 784 pixels (28×28) are randomly shuffled in a consistent way across all images
2. **Label Permutation:** The digit labels are randomly remapped (e.g., all 3s might become 7s, all 7s become 1s, etc.)

This means:
- Traditional pre-trained will be harder to use
- You must balance learning speed vs. accuracy within the 1-minute constraint

### Observation Space

The observation is a dictionary containing:
- `train_images`: Training images (60000, 28, 28) with values in [0, 1]
- `train_labels`: Labels for training images (60000,) with values in [0, 9]
- `test_images`: Test images (10000, 28, 28) with values in [0, 1]

### Action Space

The action should be predicted labels for test images:
- Shape: (10000,)
- Values: integers in [0, 9]

### Accuracy

The metric is the classification accuracy on the test set (between 0 and 1).


## Submission Requirements

### What to Submit

To compete, you need to submit:

1. **Agent Class File** (`agent.py`):
   - Must implement the standard Agent interface with `reset()`, `train()`, and `predict()` methods

2. **Util Files** (`.py`):
   - the methods and function you need in your `agent.py`

3. **Model Weights** (torch, jax or tensorflow if applicable):
   - Include any pre-trained weights or saved models your agent needs


### Agent Interface

Your agent MUST follow this interface:

```python
class Agent:
    def __init__(self, output_dim: int = 10, seed: int = None):
        """Initialize your agent"""
        pass

    def reset(self):
        """Reset for a new task (new permutation)"""
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train on the permuted training data"""
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return predictions for test data"""
        pass
```

## Example Usage

```python
import numpy as np
from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv
from permuted_mnist.agent.linear.agent import Agent as LinearAgent

# Create environment
env = PermutedMNISTEnv(number_episodes=10)
env.set_seed(42)

# Initialize your agent
agent = LinearAgent(input_dim=784, output_dim=10, learning_rate=0.01)

# Evaluation loop (this simulates the competition evaluation)
total_time = 0
accuracies = []

for episode in range(10):
    # Get next task with new permutations
    task = env.get_next_task()
    if task is None:
        break

    # Reset agent for new task
    agent.reset()

    import time
    start = time.time()

    # Train (must complete within time limit)
    agent.train(task['X_train'], task['y_train'])

    # Predict
    predictions = agent.predict(task['X_test'])

    elapsed = time.time() - start
    total_time += elapsed

    # Evaluate
    accuracy = env.evaluate(predictions, task['y_test'])
    accuracies.append(accuracy)

    print(f"Episode {episode + 1}: Accuracy: {accuracy:.3f}, Time: {elapsed:.2f}s")

print(f"\nFinal Score: {np.mean(accuracies):.3f}")
print(f"Total Time: {total_time:.2f}s")
print(f"Status: {'PASS ✅' if total_time < 600 else 'FAIL ❌ (timeout)'}")
```

## Getting Started

Check out the `getting_started.ipynb` notebook for:
- Step-by-step tutorial
- Comparison of baseline agents
- Performance analysis
- Tips for improving your agent