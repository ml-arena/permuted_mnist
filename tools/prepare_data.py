import torch
import torchvision
import numpy as np
import os
from collections import defaultdict

def download_and_process_mnist(n_images=6000):
    """
    Download MNIST dataset using PyTorch and save as numpy arrays.
    Keeps n_images (default 6000) with balanced distribution across classes.
    """
    print(f"Downloading and processing MNIST dataset (keeping {n_images} images)...")
    
    # Create data directory if it doesn't exist
    os.makedirs("metamnist/data", exist_ok=True)
    
    # Download and load MNIST
    train_dataset = torchvision.datasets.MNIST(
        root='./temp_data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    # Group images by label
    label_to_images = defaultdict(list)
    for img, label in train_dataset:
        label_to_images[label].append(img.numpy().squeeze())
    
    # Calculate images per class
    n_classes = len(label_to_images)
    images_per_class = n_images // n_classes
    
    # Select balanced subset of images
    selected_images = []
    selected_labels = []
    
    for label in range(n_classes):
        # Get images for this class
        class_images = label_to_images[label]
        
        # Randomly select subset
        indices = np.random.choice(
            len(class_images), 
            size=images_per_class, 
            replace=False
        )
        
        # Add selected images and labels
        selected_images.extend([class_images[i] for i in indices])
        selected_labels.extend([label] * images_per_class)
    
    # Convert to numpy arrays
    images = np.stack(selected_images).astype(np.float32)
    labels = np.array(selected_labels, dtype=np.int64)
    
    # Shuffle the dataset
    shuffle_idx = np.random.permutation(len(images))
    images = images[shuffle_idx]
    labels = labels[shuffle_idx]
    
    # Save arrays
    np.save("metamnist/data/mnist_images.npy", images)
    np.save("metamnist/data/mnist_labels.npy", labels)
    
    # Print information
    print("\nDataset processed and saved:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Images value range: {images.min():.3f} - {images.max():.3f}")
    print(f"Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} images")
    print("\nFiles saved:")
    print("- metamnist/data/mnist_images.npy")
    print("- metamnist/data/mnist_labels.npy")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree('./temp_data')

if __name__ == "__main__":
    download_and_process_mnist()