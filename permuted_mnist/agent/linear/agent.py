"""
Simple Linear Agent for Permuted MNIST
Implements basic logistic regression with SGD for supervised meta-learning
"""
import numpy as np


class Agent:
    """Simple linear classifier for permuted MNIST tasks"""

    def __init__(self, input_dim: int = 784, output_dim: int = 10, learning_rate: float = 0.01):
        """
        Initialize the linear agent

        Args:
            input_dim: Dimension of flattened input (28*28 for MNIST)
            output_dim: Number of output classes (10 for MNIST digits)
            learning_rate: Learning rate for SGD
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Initialize weights and bias
        self.reset()

    def reset(self):
        """Reset the agent for a new task/simulation"""
        # Xavier initialization
        self.W = np.random.randn(self.input_dim, self.output_dim) * np.sqrt(2.0 / self.input_dim)
        self.b = np.zeros((1, self.output_dim))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Compute stable softmax"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the linear model"""
        # Flatten images if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        # Normalize to [0, 1] if needed
        if X.max() > 1:
            X = X.astype(np.float32) / 255.0

        # Linear transformation
        z = X @ self.W + self.b
        return self._softmax(z)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 5, batch_size: int = 32):
        """
        Train the agent on the provided data

        Args:
            X_train: Training images (N, 28, 28) or (N, 784)
            y_train: Training labels (N, 1) or (N,)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch SGD
        """
        # Flatten images if needed
        if X_train.ndim > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)

        # Normalize to [0, 1]
        if X_train.max() > 1:
            X_train = X_train.astype(np.float32) / 255.0

        # Flatten labels if needed
        y_train = y_train.ravel()

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                # Get batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward pass
                probs = self._forward(X_batch)

                # Convert labels to one-hot
                y_one_hot = np.zeros((y_batch.size, self.output_dim))
                y_one_hot[np.arange(y_batch.size), y_batch] = 1

                # Compute gradients (cross-entropy loss)
                batch_size_actual = X_batch.shape[0]
                dW = (1/batch_size_actual) * X_batch.T @ (probs - y_one_hot)
                db = (1/batch_size_actual) * np.sum(probs - y_one_hot, axis=0, keepdims=True)

                # Update parameters
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data

        Args:
            X_test: Test images (N, 28, 28) or (N, 784)

        Returns:
            Predicted class labels (N,)
        """
        # Get probabilities
        probs = self._forward(X_test)

        # Return class with highest probability
        return np.argmax(probs, axis=1)