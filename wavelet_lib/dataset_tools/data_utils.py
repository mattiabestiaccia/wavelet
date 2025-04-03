#!/usr/bin/env python3
"""
Utility functions for data manipulation and dataset analysis.

This module provides helper functions for working with datasets,
including data loading, analysis, and preprocessing.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from collections import defaultdict

# Add the main directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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
    if not os.path.exists(dataset_path):
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
    class_distribution = {name: count/total_samples for name, count in class_counts.items()}
    
    # Print summary
    print(f"\nDataset Analysis: {dataset_path}")
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {len(class_names)}")
    print("\nClass distribution:")
    for name, count in class_counts.items():
        percentage = class_distribution[name] * 100
        print(f"  - {name}: {count} samples ({percentage:.1f}%)")
    
    return {
        'total_samples': total_samples,
        'num_classes': len(class_names),
        'class_names': class_names,
        'class_counts': class_counts,
        'class_distribution': class_distribution
    }

def extract_balanced_dataset(dataset_path, output_path, samples_per_class=None):
    """
    Extract a balanced subset of a dataset.
    
    Args:
        dataset_path: Path to source dataset directory
        output_path: Path to output dataset directory
        samples_per_class: Number of samples per class (if None, use minimum count)
        
    Returns:
        dict: Extraction statistics
    """
    # Check if paths exist
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found: {dataset_path}")
        return None
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Collect class samples
    class_images = defaultdict(list)
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    # Get all images by class
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                filepath = os.path.join(class_dir, filename)
                class_images[class_name].append(filepath)
    
    # Determine samples per class
    if samples_per_class is None:
        min_samples = min(len(images) for images in class_images.values())
        samples_per_class = min_samples
    
    # Create balanced dataset
    extraction_stats = {}
    
    for class_name, images in class_images.items():
        # Create class directory in output
        class_output_dir = os.path.join(output_path, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Select random samples
        selected_images = random.sample(images, min(samples_per_class, len(images)))
        
        # Copy images
        for i, image_path in enumerate(selected_images):
            img = Image.open(image_path)
            output_file = os.path.join(class_output_dir, f"{i:04d}.jpg")
            img.save(output_file)
        
        extraction_stats[class_name] = len(selected_images)
    
    # Print summary
    print(f"\nBalanced Dataset Extraction:")
    print(f"Source: {dataset_path}")
    print(f"Destination: {output_path}")
    print(f"Samples per class: {samples_per_class}")
    print("\nExtractions stats:")
    for class_name, count in extraction_stats.items():
        print(f"  - {class_name}: {count} samples")
    
    return {
        'source': dataset_path,
        'destination': output_path,
        'samples_per_class': samples_per_class,
        'extraction_stats': extraction_stats
    }

def analyze_image_sizes(dataset_path):
    """
    Analyze image sizes in a dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        dict: Image size statistics
    """
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found: {dataset_path}")
        return None
    
    # Get all images
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, filename))
    
    # Analyze sizes
    widths = []
    heights = []
    aspect_ratios = []
    
    for path in image_paths:
        try:
            img = Image.open(path)
            width, height = img.size
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(width / height)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # Calculate statistics
    width_stats = {
        'min': min(widths),
        'max': max(widths),
        'mean': np.mean(widths),
        'median': np.median(widths),
        'std': np.std(widths)
    }
    
    height_stats = {
        'min': min(heights),
        'max': max(heights),
        'mean': np.mean(heights),
        'median': np.median(heights),
        'std': np.std(heights)
    }
    
    aspect_ratio_stats = {
        'min': min(aspect_ratios),
        'max': max(aspect_ratios),
        'mean': np.mean(aspect_ratios),
        'median': np.median(aspect_ratios),
        'std': np.std(aspect_ratios)
    }
    
    # Print summary
    print(f"\nImage Size Analysis: {dataset_path}")
    print(f"Total images: {len(image_paths)}")
    print("\nWidth statistics:")
    print(f"  - Min: {width_stats['min']}")
    print(f"  - Max: {width_stats['max']}")
    print(f"  - Mean: {width_stats['mean']:.1f}")
    print(f"  - Median: {width_stats['median']:.1f}")
    print(f"  - Std: {width_stats['std']:.1f}")
    
    print("\nHeight statistics:")
    print(f"  - Min: {height_stats['min']}")
    print(f"  - Max: {height_stats['max']}")
    print(f"  - Mean: {height_stats['mean']:.1f}")
    print(f"  - Median: {height_stats['median']:.1f}")
    print(f"  - Std: {height_stats['std']:.1f}")
    
    print("\nAspect Ratio statistics:")
    print(f"  - Min: {aspect_ratio_stats['min']:.2f}")
    print(f"  - Max: {aspect_ratio_stats['max']:.2f}")
    print(f"  - Mean: {aspect_ratio_stats['mean']:.2f}")
    print(f"  - Median: {aspect_ratio_stats['median']:.2f}")
    print(f"  - Std: {aspect_ratio_stats['std']:.2f}")
    
    return {
        'total_images': len(image_paths),
        'width_stats': width_stats,
        'height_stats': height_stats,
        'aspect_ratio_stats': aspect_ratio_stats
    }

def plot_size_distribution(dataset_path, save_path=None):
    """
    Plot the distribution of image sizes in a dataset.
    
    Args:
        dataset_path: Path to dataset directory
        save_path: Path to save the plot
        
    Returns:
        dict: Size distribution statistics
    """
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found: {dataset_path}")
        return None
    
    # Get all images
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, filename))
    
    # Analyze sizes
    widths = []
    heights = []
    
    for path in image_paths:
        try:
            img = Image.open(path)
            width, height = img.size
            widths.append(width)
            heights.append(height)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20, alpha=0.7, color='blue')
    plt.axvline(np.median(widths), color='red', linestyle='dashed', linewidth=1)
    plt.title(f"Width Distribution (Median: {np.median(widths):.0f} px)")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20, alpha=0.7, color='green')
    plt.axvline(np.median(heights), color='red', linestyle='dashed', linewidth=1)
    plt.title(f"Height Distribution (Median: {np.median(heights):.0f} px)")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Count")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Size distribution plot saved to: {save_path}")
    
    plt.show()
    
    return {
        'widths': widths,
        'heights': heights,
        'width_median': np.median(widths),
        'height_median': np.median(heights)
    }
