import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

def compute_confusion_matrix(y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           num_classes: int = 10,
                           normalize: Optional[str] = 'true') -> np.ndarray:
    """
    Compute confusion matrix without using sklearn.
    """
    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    
    # Count occurrences
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    # Normalize if requested
    if normalize == 'true':
        # Normalize by row (true labels)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm, 0)  # Replace NaN with 0
    elif normalize == 'pred':
        # Normalize by column (predicted labels)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = cm / cm.sum(axis=0, keepdims=True)
        cm = np.nan_to_num(cm, 0)
    elif normalize == 'all':
        # Normalize by total number of samples
        cm = cm / cm.sum()
        
    return cm

class MetaMNISTRenderer:
    """Renderer for the MetaMNIST environment."""
    
    def __init__(self, figure_size: tuple = (8, 6)):
        """
        Initialize the renderer with a single figure for test set visualization.
        """
        self.figure_size = figure_size
        self.fig = None
        self.ax = None

    def _plot_confusion_matrix(self, cm: np.ndarray, ax: plt.Axes):
        """Plot confusion matrix."""
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax, 
                   xticklabels=range(10), yticklabels=range(10))
        ax.set_title('Test Set Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    def render(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Render the current state showing only test set confusion matrix.
        """
        # Clear previous figure if it exists
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create figure with single subplot
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)
        
        if state['test_predictions'] is not None:
            test_cm = compute_confusion_matrix(
                state['test_labels'], 
                state['test_predictions'],
                normalize='true'
            )
            self._plot_confusion_matrix(test_cm, self.ax)
        else:
            self.ax.text(0.5, 0.5, 'No test predictions yet', 
                        ha='center', va='center')
            self.ax.set_title('Test Set Confusion Matrix')
        
        # Add title with current metrics and parameters
        accuracy = state.get('accuracy', 0)
        angle, shift_x, shift_y = state.get('transform_params', (0, 0, 0))
        plt.suptitle(
            f'Accuracy: {accuracy:.3f} | ' 
            f'Transform: rotation={angle:.1f}°, '
            f'shift_x={shift_x:.2f}, shift_y={shift_y:.2f}'
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to RGB array
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        x = np.asarray(buf)
        data = x[:, :, :3]
        
        return data

    def close(self):
        """Clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None