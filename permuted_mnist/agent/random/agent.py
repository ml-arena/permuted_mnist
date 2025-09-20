"""
Random Agent for Permuted MNIST
Makes random predictions without learning - useful as a baseline
"""
import numpy as np


class Agent:
    """Random agent that makes random class predictions"""

    def __init__(self, output_dim: int = 10, seed: int = None):
        """
        Initialize the random agent

        Args:
            output_dim: Number of output classes (10 for MNIST digits)
            seed: Random seed for reproducibility
        """
        self.output_dim = output_dim
        self.rng = np.random.RandomState(seed)

    def reset(self):
        """Reset the agent for a new task/simulation"""
        # Random agent has no state to reset
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the agent on the provided data
        Random agent doesn't actually train, just ignores the data

        Args:
            X_train: Training images (N, 28, 28) or (N, 784)
            y_train: Training labels (N, 1) or (N,)
        """
        # Random agent doesn't learn from data
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make random predictions on test data

        Args:
            X_test: Test images (N, 28, 28) or (N, 784)

        Returns:
            Random class labels (N,)
        """
        # Get number of samples
        n_samples = X_test.shape[0]

        # Generate random predictions
        predictions = self.rng.randint(0, self.output_dim, size=n_samples)

        return predictions