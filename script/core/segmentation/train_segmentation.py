#!/usr/bin/env python3
"""
Training script for WST-UNet segmentation model.

This script trains a Wavelet Scattering Transform U-Net model
for image segmentation tasks. It supports various data augmentations
and automatically splits the dataset into training and validation sets.

Usage:
    python script/core/train_segmentation.py --imgs-dir /path/to/images --masks-dir /path/to/masks --model /path/to/save/model.pth
    python script/core/train_segmentation.py --imgs-dir /path/to/images --masks-dir /path/to/masks --val-split 0.2 --model /path/to/save/model.pth

Author: Claude
Date: 2025-04-01
"""

import os
import sys
import argparse
import time
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the main directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import wavelet_lib modules
from wavelet_lib.single_tile_segmentation import train_segmentation_model
from wavelet_lib.base import Config, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a WST-UNet segmentation model')

    # Input directories
    parser.add_argument('--imgs-dir', type=str, required=True, help='Directory containing all images [NECESSARY]')
    parser.add_argument('--masks-dir', type=str, required=True, help='Directory containing all masks [NECESSARY]')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of data to use for validation [default=0.2]')
    parser.add_argument('--img-pattern', type=str, default='*.jpg,*.png,*.tif', help='Pattern(s) to match image files, comma-separated [default=*.jpg,*.png,*.tif]')
    parser.add_argument('--mask-pattern', type=str, default='*.png', help='Pattern to match mask files [default=*.png]')

    # Model parameters
    parser.add_argument('--model', type=str, required=True, help='Path to save the trained model [NECESSARY]')
    parser.add_argument('--input-size', type=str, default='256,256', help='Input size for the model (W,H) [default=256,256]')
    parser.add_argument('--j', type=int, default=2, help='Number of wavelet scales (J parameter) [default=2]')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training [default=8]')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs [default=50]')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate [default=1e-4]')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers [default=4]')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility [default=42]')

    # Output options
    parser.add_argument('--log-dir', type=str, help='Directory to save training logs')

    return parser.parse_args()


def split_dataset(imgs_dir, masks_dir, val_split=0.2, img_patterns=None, mask_pattern='*.png', seed=42):
    """
    Split dataset into training and validation sets.

    Args:
        imgs_dir: Directory containing images
        masks_dir: Directory containing masks
        val_split: Fraction of data to use for validation
        img_patterns: List of patterns to match image files
        mask_pattern: Pattern to match mask files
        seed: Random seed for reproducibility

    Returns:
        train_imgs, train_masks, val_imgs, val_masks: Lists of file paths
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Default patterns if none provided
    if img_patterns is None:
        img_patterns = ['*.jpg', '*.png', '*.tif']
    elif isinstance(img_patterns, str):
        img_patterns = img_patterns.split(',')

    # Find all image files
    all_imgs = []
    for pattern in img_patterns:
        all_imgs.extend(glob.glob(os.path.join(imgs_dir, pattern)))
    all_imgs = sorted(all_imgs)

    # Find all mask files
    all_masks = sorted(glob.glob(os.path.join(masks_dir, mask_pattern)))

    # Verify that we have the same number of images and masks
    if len(all_imgs) != len(all_masks):
        raise ValueError(f"Number of images ({len(all_imgs)}) does not match number of masks ({len(all_masks)})")

    if len(all_imgs) == 0:
        raise ValueError(f"No images found in {imgs_dir} with patterns {img_patterns}")

    # Create paired list of images and masks
    paired_data = list(zip(all_imgs, all_masks))

    # Shuffle the data
    np.random.shuffle(paired_data)

    # Split into training and validation
    val_size = int(len(paired_data) * val_split)
    train_size = len(paired_data) - val_size

    # Ensure we have at least one sample in each set
    if val_size == 0 and len(paired_data) > 1:
        val_size = 1
        train_size = len(paired_data) - 1

    train_pairs = paired_data[:train_size]
    val_pairs = paired_data[train_size:]

    # Unzip the pairs
    train_imgs, train_masks = zip(*train_pairs) if train_pairs else ([], [])
    val_imgs, val_masks = zip(*val_pairs) if val_pairs else ([], [])

    return list(train_imgs), list(train_masks), list(val_imgs), list(val_masks)


def plot_training_history(history, output_path=None):
    """
    Plot training history.

    Args:
        history: Dictionary containing training history
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss')

    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {output_path}")

    plt.close()


def main(args):
    """Main function to train segmentation model."""
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Parse input size
    input_size = tuple(map(int, args.input_size.split(',')))

    # Create log directory if specified
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    # Split dataset into training and validation sets
    try:
        print(f"Splitting dataset with validation split: {args.val_split}")
        img_patterns = args.img_pattern.split(',') if args.img_pattern else None
        train_imgs, train_masks, val_imgs, val_masks = split_dataset(
            args.imgs_dir,
            args.masks_dir,
            val_split=args.val_split,
            img_patterns=img_patterns,
            mask_pattern=args.mask_pattern,
            seed=args.seed
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Validate that we have enough data
    if len(train_imgs) == 0:
        print(f"Error: No training images found in {args.imgs_dir}")
        return

    if len(val_imgs) == 0:
        print(f"Error: No validation images available. Please check your dataset or reduce val_split.")
        return

    print(f"Dataset split complete: {len(train_imgs)} training images, {len(val_imgs)} validation images")

    # Print training configuration
    print("\n" + "="*80)
    print(" "*30 + "TRAINING CONFIGURATION" + " "*30)
    print("="*80)

    print(f"Training images: {len(train_imgs)}")
    if val_imgs:
        print(f"Validation images: {len(val_imgs)}")

    print(f"\nModel parameters:")
    print(f"  • Input size: {input_size}")
    print(f"  • J parameter: {args.j}")

    print(f"\nTraining parameters:")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Epochs: {args.epochs}")
    print(f"  • Learning rate: {args.lr}")
    print(f"  • Workers: {args.workers}")
    print(f"  • Seed: {args.seed}")

    print(f"\nOutput:")
    print(f"  • Model path: {args.model}")
    if args.log_dir:
        print(f"  • Log directory: {args.log_dir}")

    print("\n" + "="*80)

    # Create directory for model if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.model)), exist_ok=True)

    # Train the model
    start_time = time.time()

    history = train_segmentation_model(
        train_images=train_imgs,
        train_masks=train_masks,
        val_images=val_imgs,
        val_masks=val_masks,
        model_path=args.model,
        J=args.j,
        input_shape=input_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.workers
    )

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # Plot and save training history
    if args.log_dir:
        history_path = os.path.join(args.log_dir, "training_history.png")
        plot_training_history(history, history_path)

        # Save training configuration
        config_path = os.path.join(args.log_dir, "training_config.txt")
        with open(config_path, 'w') as f:
            f.write("WST-UNet Segmentation Training Configuration\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Training Data:\n")
            f.write(f"  • Training images: {len(train_imgs)}\n")
            f.write(f"  • Validation images: {len(val_imgs)}\n")
            f.write(f"  • Validation split: {args.val_split}\n")
            f.write("\n")

            f.write("Model Parameters:\n")
            f.write(f"  • Input size: {input_size}\n")
            f.write(f"  • J parameter: {args.j}\n")
            f.write("\n")

            f.write("Training Parameters:\n")
            f.write(f"  • Batch size: {args.batch_size}\n")
            f.write(f"  • Epochs: {args.epochs}\n")
            f.write(f"  • Learning rate: {args.lr}\n")
            f.write(f"  • Workers: {args.workers}\n")
            f.write(f"  • Seed: {args.seed}\n")
            f.write("\n")

            f.write("Results:\n")
            f.write(f"  • Final training loss: {history['train_loss'][-1]:.6f}\n")
            if 'val_loss' in history and history['val_loss']:
                f.write(f"  • Final validation loss: {history['val_loss'][-1]:.6f}\n")
                f.write(f"  • Best validation loss: {min(history['val_loss']):.6f}\n")
            f.write(f"  • Training duration: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
            f.write(f"  • Model saved to: {args.model}\n")

        print(f"Training configuration saved to: {config_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("\nNota: La validazione è ora obbligatoria e il dataset viene automaticamente diviso in training e validation.")