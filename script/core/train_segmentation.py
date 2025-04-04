#!/usr/bin/env python3
"""
Training script for WST-UNet segmentation model.

This script trains a Wavelet Scattering Transform U-Net model
for image segmentation tasks. It supports various data augmentations
and can be used with or without validation data.

Usage:
    python script/core/train_segmentation.py --train-imgs /path/to/train/images --train-masks /path/to/train/masks --model /path/to/save/model.pth
    python script/core/train_segmentation.py --train-imgs /path/to/train/images --train-masks /path/to/train/masks --val-imgs /path/to/val/images --val-masks /path/to/val/masks --model /path/to/save/model.pth

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
from wavelet_lib.segmentation import train_segmentation_model
from wavelet_lib.base import Config, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a WST-UNet segmentation model')
    
    # Input directories
    parser.add_argument('--train-imgs', type=str, required=True, help='Directory containing training images [NECESSARY]')
    parser.add_argument('--train-masks', type=str, required=True, help='Directory containing training masks [NECESSARY]')
    parser.add_argument('--val-imgs', type=str, help='Directory containing validation images')
    parser.add_argument('--val-masks', type=str, help='Directory containing validation masks')
    
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
    
    # Gather training and validation images
    train_imgs = sorted(glob.glob(os.path.join(args.train_imgs, '*.jpg')) + 
                      glob.glob(os.path.join(args.train_imgs, '*.png')))
    
    train_masks = sorted(glob.glob(os.path.join(args.train_masks, '*.jpg')) + 
                        glob.glob(os.path.join(args.train_masks, '*.png')))
    
    val_imgs = None
    val_masks = None
    
    if args.val_imgs and args.val_masks:
        val_imgs = sorted(glob.glob(os.path.join(args.val_imgs, '*.jpg')) + 
                        glob.glob(os.path.join(args.val_imgs, '*.png')))
        
        val_masks = sorted(glob.glob(os.path.join(args.val_masks, '*.jpg')) + 
                          glob.glob(os.path.join(args.val_masks, '*.png')))
    
    # Validate inputs
    if len(train_imgs) == 0:
        print(f"Error: No training images found in {args.train_imgs}")
        return
    
    if len(train_masks) == 0:
        print(f"Error: No training masks found in {args.train_masks}")
        return
    
    if len(train_imgs) != len(train_masks):
        print(f"Warning: Number of training images ({len(train_imgs)}) does not match number of masks ({len(train_masks)})")
        return
    
    if val_imgs and val_masks and len(val_imgs) != len(val_masks):
        print(f"Warning: Number of validation images ({len(val_imgs)}) does not match number of masks ({len(val_masks)})")
        return
    
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
            if val_imgs:
                f.write(f"  • Validation images: {len(val_imgs)}\n")
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