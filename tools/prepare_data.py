import torch
import torchvision
import numpy as np
import os

def download_and_process_mnist():
    """
    Download MNIST dataset using PyTorch and save train/test sets as 8-bit numpy arrays.
    """
    print("Downloading and processing MNIST dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs("permuted_mnist/data", exist_ok=True)
    
    # Download and load MNIST train set
    train_dataset = torchvision.datasets.MNIST(
        root='./temp_data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    # Download and load MNIST test set
    test_dataset = torchvision.datasets.MNIST(
        root='./temp_data',
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    def process_dataset(dataset, prefix):
        images = []
        labels = []
        
        for img, label in dataset:
            # Convert from [0,1] float to [0,255] uint8 range
            img_uint8 = (img.numpy().squeeze() * 255).astype(np.uint8)
            images.append(img_uint8)
            labels.append(label)
        
        # Convert to numpy arrays
        images = np.stack(images)
        labels = np.array(labels, dtype=np.uint8)
        
        # Save arrays
        np.save(f"permuted_mnist/data/mnist_{prefix}_images.npy", images)
        np.save(f"permuted_mnist/data/mnist_{prefix}_labels.npy", labels)
        
        return images, labels
    
    # Process train and test sets
    train_images, train_labels = process_dataset(train_dataset, "train")
    test_images, test_labels = process_dataset(test_dataset, "test")
    
    # Print information
    def print_dataset_info(prefix, images, labels):
        print(f"\n{prefix.capitalize()} dataset:")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Images dtype: {images.dtype}")
        print(f"Images value range: {images.min()} - {images.max()}")
        print(f"Label distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} images")
    
    print_dataset_info("train", train_images, train_labels)
    print_dataset_info("test", test_images, test_labels)
    
    print("\nFiles saved:")
    print("- permuted_mnist/data/mnist_train_images.npy")
    print("- permuted_mnist/data/mnist_train_labels.npy")
    print("- permuted_mnist/data/mnist_test_images.npy")
    print("- permuted_mnist/data/mnist_test_labels.npy")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree('./temp_data')

if __name__ == "__main__":
    download_and_process_mnist()