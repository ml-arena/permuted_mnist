import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import os
from metamnist import PKG_DIR
from .renderer import MetaMNISTRenderer

class MetaMNISTEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
    }
    def __init__(
        self,
        render_mode: str = None,
        number_steps: int = 10,
    ):
        print('init metamnist env')
        super().__init__()
        self.render_mode = render_mode
        self.renderer = MetaMNISTRenderer() if render_mode == "rgb_array" else None
        self.current_test_predictions = None
        self.current_train_predictions = None
        self.number_steps = number_steps
        self.current_step = 0
        self.label_permutation = None
        self.pixel_permutation = None
        # Load pre-split MNIST dataset from package data
        data_path = os.path.join(PKG_DIR, 'data')
        try:
            print('loading mnist data')
            self.train_images = np.load(os.path.join(data_path, 'mnist_train_images.npy')).astype(np.uint8)
            print('train_images loaded')
            self.train_labels = np.load(os.path.join(data_path, 'mnist_train_labels.npy')).astype(np.uint8)
            print('train_labels loaded')
            self.test_images = np.load(os.path.join(data_path, 'mnist_test_images.npy')).astype(np.uint8)
            print('test_images loaded')
            self.test_labels = np.load(os.path.join(data_path, 'mnist_test_labels.npy')).astype(np.uint8)
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(
                "MNIST data files not found. Please run the data preparation script first:\n"
                "python tools/prepare_data.py"
            ) from e
            
        # Store dataset sizes
        self.train_size = len(self.train_images)
        self.test_size = len(self.test_images)
        # Single Box observation space
        # Alternative version using float32 if you really need infinite bounds
        print('creating observation space')
        self.observation_space = spaces.Dict({
            'X_train': spaces.Box(
                low=0,
                high=255,
                shape=(self.train_size, 28, 28),
                dtype=np.uint8
            ),
            'y_train': spaces.Box(
                low=0,
                high=9,
                shape=(self.train_size, 1),
                dtype=np.uint8
            ),
            'X_test': spaces.Box(
                low=0,
                high=255,
                shape=(self.test_size, 28, 28),
                dtype=np.uint8
            )
        })
        print('observation space created')
        self.action_space = spaces.Box(
            low=0,
            high=9,
            shape=(self.test_size, 1),
            dtype=np.uint8
        )
    def _create_permutations(self):
        self.label_permutation = self.np_random.permutation(10)
        self.pixel_permutation = self.np_random.permutation(28 * 28)
    def _permute_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create a random permutation of labels."""
        
        # Apply permutation to both train and test labels
        permuted_train_labels = self.label_permutation[self.current_train_labels]
        permuted_test_labels = self.label_permutation[self.current_test_labels]
        
        return permuted_train_labels, permuted_test_labels

    def _permute_pixels(self, images: np.ndarray) -> np.ndarray:
        """Permute pixels consistently across all images."""
        
        # Reshape images to (N, 784), permute, and reshape back
        flat_images = images.reshape(len(images), -1)
        permuted_images = flat_images[:, self.pixel_permutation]
        return permuted_images.reshape(images.shape)

    def _shuffle_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Shuffle train and test sets independently."""
        # Shuffle training set
        train_indices = self.np_random.permutation(self.train_size)
        shuffled_train_images = self.train_images[train_indices]
        shuffled_train_labels = self.train_labels[train_indices]
        
        # Shuffle test set
        test_indices = self.np_random.permutation(self.test_size)
        shuffled_test_images = self.test_images[test_indices]
        shuffled_test_labels = self.test_labels[test_indices]
        
        return (shuffled_train_images, shuffled_train_labels, 
                shuffled_test_images, shuffled_test_labels)

    # def _get_observation_array(self) -> np.ndarray:
    #     """Convert current state to flattened observation array."""
    #     train_images_flat = self.current_train_images.ravel()
    #     train_labels_norm = self.current_train_labels
    #     test_images_flat = self.current_test_images.ravel()
        
    #     return np.concatenate([
    #         train_images_flat,
    #         train_labels_norm,
    #         test_images_flat
    #     ])

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to start a new episode."""
        print('resetting env')
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset predictions
        self.current_test_predictions = None
        self.current_train_predictions = None
        self._create_permutations()
        print('permutations created')
        # Start with shuffled datasets
        (self.current_train_images, self.current_train_labels,
         self.current_test_images, self.current_test_labels) = self._shuffle_dataset()
        print('datasets shuffled')
        self.current_train_labels, self.current_test_labels = self._permute_labels()
        print('labels permuted')
        self.current_train_images = self._permute_pixels(self.current_train_images)
        print('pixels permuted')
        self.current_test_images = self._permute_pixels(self.current_test_images)
        print('pixels permuted')
        
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        print('getting obs')
        observation = {
            'X_train': self.current_train_images,
            'y_train': self.current_train_labels,
            'X_test': self.current_test_images
        }
        return observation
        # return self._get_observation_array()

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        return {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        print('stepping env')  
        self.current_test_predictions = action
        self._create_permutations()
        if not isinstance(action, np.ndarray) or action.shape != (self.test_size,):
            raise ValueError(f"Expected action shape ({self.test_size},), got {action.shape}")
            
        # Calculate reward (accuracy)
        reward = np.mean(action == self.current_test_labels)
        
        self.current_step += 1
        truncated = self.current_step >= self.number_steps
        terminated = False
        
        if not truncated:
            # Start with shuffled datasets
            (self.current_train_images, self.current_train_labels,
             self.current_test_images, self.current_test_labels) = self._shuffle_dataset()
            
            # Apply transformations
            self.current_train_labels, self.current_test_labels = self._permute_labels()
                
            self.current_train_images = self._permute_pixels(self.current_train_images)
            self.current_test_images = self._permute_pixels(self.current_test_images)
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
        
    def render(self) -> Optional[np.ndarray]:
        """Render the current state of the environment."""
        if self.render_mode != "rgb_array" or self.renderer is None:
            return None
            
        render_state = {
            'train_labels': self.current_train_labels,
            'test_labels': self.current_test_labels,
            'test_predictions': self.current_test_predictions,
            'accuracy': np.mean(self.current_test_predictions == self.current_test_labels) 
                       if self.current_test_predictions is not None else 0.0
        }
        
        if self.current_train_predictions is not None:
            render_state['train_predictions'] = self.current_train_predictions
            
        return self.renderer.render(render_state)

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()