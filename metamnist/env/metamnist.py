import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import rotate, shift
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
        max_rotation: float = 30.0,
        max_shift: float = 0.2,
        render_mode: str = None,
        number_steps: int = 10,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.renderer = MetaMNISTRenderer() if render_mode == "rgb_array" else None
        self.current_test_predictions = None
        self.current_train_predictions = None
        self.number_steps = number_steps
        self.current_step = 0
        
        self.max_rotation = max_rotation
        self.max_shift = max_shift
        
        # Load full MNIST dataset from package data
        data_path = os.path.join(PKG_DIR, 'data')
        try:
            self.images = np.load(os.path.join(data_path, 'mnist_images.npy'))
            self.labels = np.load(os.path.join(data_path, 'mnist_labels.npy'))
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(
                "MNIST data files not found. Please run the data preparation script first:\n"
                "python tools/prepare_data.py"
            ) from e
            
        # Split dataset into train and test
        self.total_samples = len(self.images)
        self.train_size = int(self.total_samples * 0.8)  # 80% for training
        self.test_size = self.total_samples - self.train_size
            
        # Cache for pre-computing transformations (use all available data)
        self.cache_size = self.total_samples
        self._initialize_transform_cache()
        
        # Keep original shapes for internal use
        self.train_images_shape = (self.train_size, 28, 28)
        self.train_labels_shape = (self.train_size,)
        self.test_images_shape = (self.test_size, 28, 28)
        
        # Single Box observation space
        self.observation_space = spaces.Box(
            low=0,
            high=1,  # Both images and normalized labels will be in [0,1]
            shape=(self.train_size * 28 * 28 + self.train_size + self.test_size * 28 * 28,),
            dtype=np.float32
        )
        
        # Action space for test set predictions
        self.action_space = spaces.Box(
            low=0, high=9, shape=(self.test_size,), dtype=np.int64
        )

    def _initialize_transform_cache(self):
        """Initialize empty cache for transformed images."""
        self.transform_cache = {
            'params': None,
            'indices': None,
            'images': np.zeros((self.cache_size, 28, 28), dtype=np.float32)
        }
        
    def _update_transform_cache(self, indices: np.ndarray, params: Tuple[float, float, float]):
        """Update transform cache with new parameters and indices."""
        if (self.transform_cache['params'] == params and 
            np.array_equal(self.transform_cache['indices'], indices)):
            return
            
        self.transform_cache['params'] = params
        self.transform_cache['indices'] = indices
        
        # Apply transformations to original images
        angle, shift_x, shift_y = params
        base_images = self.images[indices].astype(np.float32) / 255.0  # Normalize to [0,1]
        
        # Apply rotation first
        if angle != 0:
            rotated = np.array([
                rotate(img, angle, reshape=False, mode='nearest')
                for img in base_images
            ])
        else:
            rotated = base_images
            
        # Then apply shifts
        if shift_x != 0 or shift_y != 0:
            shifted = np.array([
                shift(img, [shift_y * 28, shift_x * 28], mode='nearest')
                for img in rotated
            ])
        else:
            shifted = rotated
            
        # Update cache
        self.transform_cache['images'][:len(indices)] = np.clip(shifted, 0, 1)

    def _get_random_transform_params(self) -> Tuple[float, float, float]:
        """Generate random transformation parameters."""
        angle = self.np_random.uniform(-self.max_rotation, self.max_rotation)
        shift_x = self.np_random.uniform(-self.max_shift, self.max_shift)
        shift_y = self.np_random.uniform(-self.max_shift, self.max_shift)
        return angle, shift_x, shift_y

    def _get_observation_array(self) -> np.ndarray:
        """Convert current state to flattened observation array."""
        train_images_flat = self.current_train_images.ravel()
        train_labels_norm = self.current_train_labels.astype(np.float32) / 9.0
        test_images_flat = self.current_test_images.ravel()
        
        return np.concatenate([
            train_images_flat,
            train_labels_norm,
            test_images_flat
        ])

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        self.current_step = 0 
        
        # Reset predictions
        self.current_test_predictions = None
        self.current_train_predictions = None
        
        # Split into train and test sets
        all_indices = self.np_random.permutation(self.total_samples)
        self.current_train_indices = all_indices[:self.train_size]
        self.current_test_indices = all_indices[self.train_size:]
        
        # Get labels
        self.current_train_labels = self.labels[self.current_train_indices]
        self.current_test_labels = self.labels[self.current_test_indices]
        
        # Generate and apply random transform
        self.current_params = self._get_random_transform_params()
        
        # Update cache with new transforms for all indices
        all_required_indices = np.concatenate([
            self.current_train_indices,
            self.current_test_indices
        ])
        self._update_transform_cache(all_required_indices, self.current_params)

        # Get transformed images from cache
        self.current_train_images = self.transform_cache['images'][:self.train_size]
        self.current_test_images = self.transform_cache['images'][self.train_size:]
        
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        return self._get_observation_array()

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        return {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        self.current_test_predictions = action

        if not isinstance(action, np.ndarray) or action.shape != (self.test_size,):
            raise ValueError(f"Expected action shape ({self.test_size},), got {action.shape}")
            
        # Calculate reward (accuracy)
        reward = np.mean(action == self.current_test_labels)
        
        self.current_step += 1
        truncated = self.current_step >= self.number_steps
        terminated = False
        
        if not truncated:
            # Generate new random transform
            self.current_params = self._get_random_transform_params()
            
            # Update cache and get transformed images
            all_required_indices = np.concatenate([
                self.current_train_indices,
                self.current_test_indices
            ])
            self._update_transform_cache(all_required_indices, self.current_params)

            # Get transformed images from cache
            self.current_train_images = self.transform_cache['images'][:self.train_size]
            self.current_test_images = self.transform_cache['images'][self.train_size:]
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
        
    def render(self) -> Optional[np.ndarray]:
        """Render the current state of the environment."""
        if self.render_mode != "rgb_array" or self.renderer is None:
            return None
            
        render_state = {
            'train_labels': self.current_train_labels,
            'test_labels': self.current_test_labels,
            'test_predictions': self.current_test_predictions,
            'transform_params': self.current_params,
            'accuracy': np.mean(self.current_test_predictions == self.current_test_labels) 
                       if self.current_test_predictions is not None else 0.0
        }
        
        if self.current_train_predictions is not None:
            render_state['train_predictions'] = self.current_train_predictions
            
        return self.renderer.render(render_state)

    def close(self):
        """Clean up resources."""
        self._initialize_transform_cache()
        if self.renderer is not None:
            self.renderer.close()