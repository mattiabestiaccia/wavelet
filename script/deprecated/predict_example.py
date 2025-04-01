#!/usr/bin/env python3
"""
Example script to show predictions from a wavelet model using test data.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

# Add parent directory to path to import wavelet_lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import wavelet library modules
from wavelet_lib.base import Config, set_seed
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Define paths
    dataset_root = '/home/brus/Projects/wavelet/datasets/HPL_images/custom_dataset'
    
    # Create configuration
    config = Config(
        num_channels=3,
        num_classes=4,  # Using 4 classes dataset
        scattering_order=2,
        J=2,
        shape=(32, 32),
        batch_size=8,
        device=torch.device("cpu")  # Use CPU for inference
    )
    
    # Create dataset with default transform
    transform = get_default_transform(target_size=(32, 32))
    balanced_dataset = BalancedDataset(dataset_root, transform=transform, balance=True)
    
    # Create data loaders with small batch size
    _, test_loader = create_data_loaders(
        balanced_dataset,
        test_size=0.2,
        batch_size=8,
        num_workers=1
    )
    
    # Show class information
    print("\nClasses in dataset:")
    for idx, class_name in enumerate(balanced_dataset.classes):
        print(f"  {idx}: {class_name}")
    
    # Display a random batch of test images with their labels
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    axs = axs.flatten()
    
    # Convert labels to class names
    class_names = balanced_dataset.classes
    
    # Display each image with its label
    for i in range(min(8, len(images))):
        # Convert tensor to numpy array and denormalize
        img = images[i].numpy().transpose((1, 2, 0))
        img = (img * 0.5) + 0.5  # Denormalize
        img = np.clip(img, 0, 1)
        
        # Display image
        axs[i].imshow(img)
        label_idx = labels[i].item()
        label_name = class_names[label_idx]
        axs[i].set_title(f"{label_name}")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Sample Test Images with True Labels", fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(dataset_root), 'test_sample_images.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSample test images saved to: {output_path}")
    
    plt.show()
    
    # Optionally show one full-size original image for reference
    print("\nShowing one original image from each class:")
    for idx, class_name in enumerate(balanced_dataset.classes):
        # Find an image from this class
        class_images = [item for item in balanced_dataset.samples if item[1] == idx]
        if class_images:
            sample_image_path = class_images[0][0]
            print(f"Class {class_name}: {os.path.basename(sample_image_path)}")
            
            # Load and display the original image
            img = Image.open(sample_image_path)
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"Original {class_name} image")
            plt.axis('off')
            
            # Save the image
            output_path = os.path.join(os.path.dirname(dataset_root), f'original_{class_name.replace(" ", "_")}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            plt.show()

if __name__ == "__main__":
    main()