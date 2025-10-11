"""
Simplified Permuted MNIST Environment for Supervised Learning
"""
import numpy as np
import os
from typing import Optional, Tuple, Dict, Any
from permuted_mnist import PKG_DIR


class PermutedMNISTEnv:
    """Simplified environment for supervised meta-learning on permuted MNIST"""

    def __init__(self, number_episodes: int = 10):
        self.number_episodes = number_episodes
        self.current_episode = 0
        self.rng = np.random.RandomState()

        # Load MNIST data
        data_path = os.path.join(PKG_DIR, 'data')
        try:
            self.train_images = np.load(os.path.join(data_path, 'mnist_train_images.npy')).astype(np.uint8)
            self.train_labels = np.load(os.path.join(data_path, 'mnist_train_labels.npy')).astype(np.uint8)
            self.test_images = np.load(os.path.join(data_path, 'mnist_test_images.npy')).astype(np.uint8)
            self.test_labels = np.load(os.path.join(data_path, 'mnist_test_labels.npy')).astype(np.uint8)
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(
                "MNIST data files not found. Please run the data preparation script first:\n"
                "python tools/prepare_data.py"
            ) from e

        # Store dataset sizes
        self.train_size = len(self.train_images)
        self.test_size = len(self.test_images)

        # Current permutations
        self.label_permutation = None
        self.pixel_permutation = None

    def set_seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility"""
        self.rng = np.random.RandomState(seed)

    def get_next_task(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get the next permuted MNIST task
        Returns None when all episodes are complete
        """
        if self.current_episode >= self.number_episodes:
            return None

        # Create new permutations for this task
        self.label_permutation = self.rng.permutation(10)
        self.pixel_permutation = self.rng.permutation(28 * 28)

        # Shuffle and permute data
        train_indices = self.rng.permutation(self.train_size)
        test_indices = self.rng.permutation(self.test_size)

        # Get shuffled data
        train_images = self.train_images[train_indices]
        train_labels = self.train_labels[train_indices]
        test_images = self.test_images[test_indices]
        test_labels = self.test_labels[test_indices]

        # Apply label permutation
        train_labels = self.label_permutation[train_labels]
        test_labels = self.label_permutation[test_labels]

        # Apply pixel permutation and task-specific noise
        train_images = self._permute_pixels(train_images, self.current_episode)
        test_images = self._permute_pixels(test_images, self.current_episode)

        self.current_episode += 1

        return {
            'X_train': train_images,
            'y_train': train_labels.reshape(-1, 1),
            'X_test': test_images,
            'y_test': test_labels  # Include true labels for evaluation
        }

    def _permute_pixels(self, images: np.ndarray, task_id: int) -> np.ndarray:
        """Permute pixels consistently across all images and add per-image random noise"""
        flat_images = images.reshape(len(images), -1)
        permuted_images = flat_images[:, self.pixel_permutation]
        permuted_images = permuted_images.reshape(images.shape)

        # Add per-image noise + random brightness/contrast to prevent cheating
        # Use task_id as base seed for reproducibility within the same task
        task_rng = np.random.RandomState(task_id)

        # Convert to float for processing
        permuted_images = permuted_images.astype(np.float32) / 255.0

        n_images = len(permuted_images)

        # Generate per-image random parameters
        scales = task_rng.uniform(0.96, 1.04, size=n_images)  # ±4% brightness
        shifts = task_rng.uniform(-0.02, 0.02, size=n_images)  # ±2% offset

        # Add per-pixel Gaussian noise (std=0.015)
        noise = task_rng.normal(0, 0.015, permuted_images.shape)
        permuted_images = permuted_images + noise

        # Apply per-image random brightness/contrast
        for i in range(n_images):
            permuted_images[i] = permuted_images[i] * scales[i] + shifts[i]

        # Clip to valid range and convert back to uint8
        permuted_images = np.clip(permuted_images, 0, 1)
        permuted_images = (permuted_images * 255).astype(np.uint8)

        return permuted_images

    def evaluate(self, predictions: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate accuracy of predictions"""
        return np.mean(predictions == true_labels)

    def reset(self):
        """Reset environment for new set of episodes"""
        self.current_episode = 0
        self.label_permutation = None
        self.pixel_permutation = None

    def is_complete(self) -> bool:
        """Check if all episodes are complete"""
        return self.current_episode >= self.number_episodes