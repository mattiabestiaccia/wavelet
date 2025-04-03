#!/usr/bin/env python3
"""
Utility functions for data manipulation and dataset analysis.

This module provides helper functions for working with datasets,
including data loading, analysis, and preprocessing.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Import wavelet_lib modules
from wavelet_lib.datasets import BalancedDataset


def analyze_dataset(dataset_path, balance=False):
    """
    Analyze a dataset and return statistics.
    
    Args:
        dataset_path: Path to dataset directory
        balance: Whether to analyze balanced dataset
        
    Returns:
        dict: Dataset statistics
    """
    # Check if path exists
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_path}")
        return None
    
    # Create dataset
    dataset = BalancedDataset(dataset_path, transform=None, balance=balance)
    
    # Get class information
    class_names = dataset.classes
    class_counts = defaultdict(int)
    
    for _, label in dataset.samples:
        class_name = class_names[label]
        class_counts[class_name] += 1
    
    # Calculate statistics
    total_samples = len(dataset)
    num_classes = len(class_names)
    min_class_size = min(class_counts.values()) if class_counts else 0
    max_class_size = max(class_counts.values()) if class_counts else 0
    
    # Check for imbalance
    imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
    
    # Collect image information
    image_sizes = defaultdict(int)
    image_channels = defaultdict(int)
    
    for sample_path, _ in dataset.samples:
        try:
            with Image.open(sample_path) as img:
                size = f"{img.width}x{img.height}"
                image_sizes[size] += 1
                image_channels[len(img.getbands())] += 1
        except Exception as e:
            print(f"Warning: Could not open image {sample_path}: {e}")
    
    # Compile statistics
    stats = {
        'total_samples': total_samples,
        'num_classes': num_classes,
        'class_counts': dict(class_counts),
        'min_class_size': min_class_size,
        'max_class_size': max_class_size,
        'imbalance_ratio': imbalance_ratio,
        'image_sizes': dict(image_sizes),
        'image_channels': dict(image_channels),
        'balanced': balance
    }
    
    return stats


def visualize_dataset_samples(dataset_path, num_samples=5, random_seed=None):
    """
    Visualize random samples from each class in a dataset.
    
    Args:
        dataset_path: Path to dataset directory
        num_samples: Number of samples to visualize per class
        random_seed: Random seed for reproducibility
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create dataset
    dataset = BalancedDataset(dataset_path, transform=None, balance=False)
    
    # Get class information
    class_names = dataset.classes
    
    # Group samples by class
    class_samples = defaultdict(list)
    
    for path, label in dataset.samples:
        class_name = class_names[label]
        class_samples[class_name].append(path)
    
    # Determine grid size
    num_classes = len(class_samples)
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(num_samples * 3, num_classes * 3))
    
    # Handle case with only one class
    if num_classes == 1:
        axes = np.array([axes])
    
    # Visualize samples
    for i, (class_name, samples) in enumerate(class_samples.items()):
        # Select random samples
        selected_samples = random.sample(samples, min(num_samples, len(samples)))
        
        for j, sample_path in enumerate(selected_samples):
            try:
                img = Image.open(sample_path)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"{class_name}")
                axes[i, j].axis('off')
            except Exception as e:
                axes[i, j].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                axes[i, j].axis('off')
        
        # Fill empty slots
        for j in range(len(selected_samples), num_samples):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(dataset_path, balance=False, figsize=(10, 6)):
    """
    Plot the class distribution of a dataset.
    
    Args:
        dataset_path: Path to dataset directory
        balance: Whether to analyze balanced dataset
        figsize: Figure size (width, height)
    """
    # Analyze dataset
    stats = analyze_dataset(dataset_path, balance=balance)
    
    if stats is None:
        return
    
    # Extract class counts
    class_counts = stats['class_counts']
    
    # Sort classes by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_names = [c[0] for c in sorted_classes]
    counts = [c[1] for c in sorted_classes]
    
    # Create plot
    plt.figure(figsize=figsize)
    bars = plt.bar(class_names, counts)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(count),
            ha='center'
        )
    
    plt.title(f"Class Distribution {'(Balanced)' if balance else ''}")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Total samples: {stats['total_samples']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Min class size: {stats['min_class_size']}")
    print(f"Max class size: {stats['max_class_size']}")
    print(f"Imbalance ratio: {stats['imbalance_ratio']:.2f}")


def check_dataset_balance(dataset_path):
    """
    Check if a dataset is balanced and suggest balancing strategies.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        dict: Balance analysis results
    """
    # Analyze dataset
    stats = analyze_dataset(dataset_path, balance=False)
    
    if stats is None:
        return None
    
    # Extract class counts
    class_counts = stats['class_counts']
    imbalance_ratio = stats['imbalance_ratio']
    
    # Determine if dataset is balanced
    is_balanced = imbalance_ratio < 1.2  # Less than 20% difference
    
    # Determine balancing strategy
    strategy = None
    if not is_balanced:
        if imbalance_ratio < 2:
            strategy = "Slight imbalance: Consider using class weights"
        elif imbalance_ratio < 5:
            strategy = "Moderate imbalance: Consider oversampling or using BalancedDataset"
        else:
            strategy = "Severe imbalance: Consider data augmentation and BalancedDataset"
    
    # Calculate balanced dataset size
    min_class_size = stats['min_class_size']
    balanced_size = min_class_size * stats['num_classes']
    
    # Compile results
    results = {
        'is_balanced': is_balanced,
        'imbalance_ratio': imbalance_ratio,
        'class_counts': class_counts,
        'balancing_strategy': strategy,
        'balanced_dataset_size': balanced_size
    }
    
    return results


def find_corrupted_images(dataset_path):
    """
    Find corrupted images in a dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        list: Paths to corrupted images
    """
    dataset_path = Path(dataset_path)
    corrupted_images = []
    
    # Walk through dataset directory
    for class_dir in [d for d in dataset_path.iterdir() if d.is_dir()]:
        for image_path in class_dir.glob('*'):
            if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                try:
                    with Image.open(image_path) as img:
                        # Try to load the image data
                        img.load()
                except Exception as e:
                    corrupted_images.append((str(image_path), str(e)))
    
    return corrupted_images


def compute_dataset_stats(dataset_path, channels=3):
    """
    Compute mean and standard deviation for dataset normalization.
    
    Args:
        dataset_path: Path to dataset directory
        channels: Number of image channels
        
    Returns:
        tuple: (mean, std) for each channel
    """
    dataset_path = Path(dataset_path)
    
    # Initialize variables
    pixel_sum = torch.zeros(channels)
    pixel_sum_squared = torch.zeros(channels)
    num_pixels = 0
    
    # Walk through dataset directory
    for class_dir in [d for d in dataset_path.iterdir() if d.is_dir()]:
        for image_path in class_dir.glob('*'):
            if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                try:
                    # Open image
                    with Image.open(image_path) as img:
                        # Convert to tensor
                        img_tensor = torch.tensor(np.array(img), dtype=torch.float32)
                        
                        # Handle grayscale images
                        if len(img_tensor.shape) == 2:
                            img_tensor = img_tensor.unsqueeze(2)
                        
                        # Handle images with alpha channel
                        if img_tensor.shape[2] > channels:
                            img_tensor = img_tensor[:, :, :channels]
                        
                        # Reshape to [C, H*W]
                        img_tensor = img_tensor.permute(2, 0, 1).reshape(img_tensor.shape[2], -1)
                        
                        # Update sums
                        pixel_sum += img_tensor.sum(dim=1)
                        pixel_sum_squared += (img_tensor ** 2).sum(dim=1)
                        num_pixels += img_tensor.shape[1]
                except Exception as e:
                    print(f"Warning: Could not process {image_path}: {e}")
    
    # Compute mean and std
    mean = pixel_sum / num_pixels
    var = (pixel_sum_squared / num_pixels) - (mean ** 2)
    std = torch.sqrt(var)
    
    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset utility functions")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory [NECESSARY]")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset [OPTIONAL]")
    parser.add_argument("--visualize", action="store_true", help="Visualize dataset samples [OPTIONAL]")
    parser.add_argument("--plot-distribution", action="store_true", help="Plot class distribution [OPTIONAL]")
    parser.add_argument("--check-balance", action="store_true", help="Check dataset balance [OPTIONAL]")
    parser.add_argument("--find-corrupted", action="store_true", help="Find corrupted images [OPTIONAL]")
    parser.add_argument("--compute-stats", action="store_true", help="Compute dataset statistics [OPTIONAL]")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels [OPTIONAL, default=3]")
    parser.add_argument("--balance", action="store_true", help="Use balanced dataset [OPTIONAL]")
    
    args = parser.parse_args()
    
    if args.analyze:
        stats = analyze_dataset(args.dataset, balance=args.balance)
        print("Dataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    
    if args.visualize:
        visualize_dataset_samples(args.dataset)
    
    if args.plot_distribution:
        plot_class_distribution(args.dataset, balance=args.balance)
    
    if args.check_balance:
        results = check_dataset_balance(args.dataset)
        print("Balance Analysis:")
        for key, value in results.items():
            print(f"{key}: {value}")
    
    if args.find_corrupted:
        corrupted = find_corrupted_images(args.dataset)
        if corrupted:
            print(f"Found {len(corrupted)} corrupted images:")
            for path, error in corrupted:
                print(f"  {path}: {error}")
        else:
            print("No corrupted images found.")
    
    if args.compute_stats:
        mean, std = compute_dataset_stats(args.dataset, channels=args.channels)
        print(f"Dataset Statistics (for {args.channels} channels):")
        print(f"Mean: {mean}")
        print(f"Std: {std}")
