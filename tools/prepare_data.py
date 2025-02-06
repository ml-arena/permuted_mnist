import torch
import torchvision
import numpy as np
import os

def download_and_process_mnist():
    """
    Download MNIST dataset using PyTorch and save as 8-bit numpy arrays.
    """
    print("Downloading and processing MNIST dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs("metamnist/data", exist_ok=True)
    
    # Download and load MNIST
    train_dataset = torchvision.datasets.MNIST(
        root='./temp_data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    # Process all images
    all_images = []
    all_labels = []
    
    for img, label in train_dataset:
        # Convert from [0,1] float to [0,255] uint8 range
        img_uint8 = (img.numpy().squeeze() * 255).astype(np.uint8)
        all_images.append(img_uint8)
        all_labels.append(label)
    
    # Convert to numpy arrays
    images = np.stack(all_images)
    labels = np.array(all_labels, dtype=np.uint8)
    
    # Save arrays
    np.save("metamnist/data/mnist_images.npy", images)
    np.save("metamnist/data/mnist_labels.npy", labels)
    
    # Print information
    print("\nDataset processed and saved:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Images value range: {images.min()} - {images.max()}")
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