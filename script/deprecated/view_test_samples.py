#!/usr/bin/env python3
"""
Script to view test samples from the dataset and their ground truth labels.
This helps understand what the model should be predicting.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random

# Add parent directory to path to import wavelet_lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import wavelet library modules
from wavelet_lib.base import set_seed
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders

def view_test_samples(dataset_path, num_samples=32, save_path=None):
    """
    View random test samples from the dataset.
    
    Args:
        dataset_path: Path to dataset directory
        num_samples: Number of samples to display
        save_path: Path to save the visualization
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create dataset
    transform = get_default_transform(target_size=(32, 32))
    dataset = BalancedDataset(dataset_path, transform=transform, balance=True)
    
    # Get class information
    class_names = dataset.classes
    print(f"Dataset contains {len(class_names)} classes: {class_names}")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        dataset, test_size=0.2, batch_size=num_samples, num_workers=1
    )
    
    # Display dataset statistics
    print("\nDataset Statistics:")
    class_distribution = dataset.get_class_distribution()
    for class_name, count in class_distribution.items():
        print(f"  {class_name}: {count} samples")
    
    print(f"\nTraining set: {len(train_loader.dataset)} samples")
    print(f"Test set: {len(test_loader.dataset)} samples")
    
    # Get a batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create figure
    plt.figure(figsize=(12, 12))
    
    # Display images
    for i in range(min(num_samples, len(images))):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Convert tensor to numpy and denormalize
        img = images[i].numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5  # Denormalize
        img = np.clip(img, 0, 1)
        
        # Get label
        label_idx = labels[i].item()
        class_name = class_names[label_idx]
        
        # Display image
        plt.imshow(img)
        plt.title(f"{class_name}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Test Samples with Ground Truth Labels", fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    # Save figure if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    # Show examples of each class
    print("\nShowing examples of each class...")
    for class_idx, class_name in enumerate(class_names):
        # Find images of this class in the test set
        class_samples = []
        for images, labels in test_loader:
            for i, label in enumerate(labels):
                if label.item() == class_idx:
                    class_samples.append(images[i])
                    if len(class_samples) >= 8:
                        break
            if len(class_samples) >= 8:
                break
        
        # If no samples found, continue
        if not class_samples:
            print(f"No test samples found for class {class_name}")
            continue
        
        # Create figure
        plt.figure(figsize=(12, 3))
        
        # Display images
        for i, img_tensor in enumerate(class_samples[:8]):
            plt.subplot(1, 8, i + 1)
            
            # Convert tensor to numpy and denormalize
            img = img_tensor.numpy().transpose((1, 2, 0))
            img = img * 0.5 + 0.5  # Denormalize
            img = np.clip(img, 0, 1)
            
            # Display image
            plt.imshow(img)
            plt.title(f"{class_name}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Examples of class: {class_name}", fontsize=14)
        plt.subplots_adjust(top=0.85)
        
        # Save figure if requested
        if save_path:
            class_viz_path = os.path.join(os.path.dirname(save_path), 
                                      f"class_{class_name.replace(' ', '_')}_examples.png")
            plt.savefig(class_viz_path, dpi=300, bbox_inches='tight')
            print(f"Class examples saved to: {class_viz_path}")
        
        plt.show()

def main():
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "datasets/HPL_images/custom_dataset")
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    # Create results directory
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set save path
    save_path = os.path.join(results_dir, "test_samples.png")
    
    # View test samples
    view_test_samples(dataset_path, num_samples=36, save_path=save_path)

if __name__ == "__main__":
    main()