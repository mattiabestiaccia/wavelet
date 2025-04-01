#!/usr/bin/env python3
"""
Training script for Wavelet Scattering Transform classification models.

This script coordinates the entire training workflow, from data preparation
to model training and evaluation.

Usage:
    python script/core/train.py --dataset /path/to/dataset --num-classes N --epochs E [options]
"""

import os
import sys
import torch
import argparse
import time
from datetime import datetime

# Add the main directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import wavelet_lib modules
from wavelet_lib.base import Config, set_seed, save_model
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders
from wavelet_lib.models import create_model, print_model_summary
from wavelet_lib.training import Trainer, create_optimizer
from wavelet_lib.visualization import plot_training_metrics, plot_class_distribution

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train a Wavelet Scattering Transform model')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--balance', action='store_true', help='Balance classes in the dataset')
    
    # Model parameters
    parser.add_argument('--num-classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--num-channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--scattering-order', type=int, default=2, help='Maximum order of scattering transform')
    parser.add_argument('--j-param', type=int, default=2, help='J parameter for scattering transform')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=90, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--reduce-lr-after', type=int, default=20, help='Reduce learning rate after this many epochs')
    
    # General training parameters
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, help='Device for training (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for dataloaders')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save results')
    
    return parser.parse_args()

def main():
    """
    Main function that coordinates model training.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Configure output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                    f"models/model_output_{args.num_classes}_classes_{timestamp}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration information
    print(f"\n{'='*80}")
    print(f"Wavelet Scattering Transform Model Training")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Classes: {args.num_classes}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Create configuration
    config = Config(
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        scattering_order=args.scattering_order,
        J=args.j_param,
        shape=(32, 32),  # Fixed size for compatibility
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        device=args.device
    )
    
    # Print configuration summary
    config.print_summary()
    
    # Prepare dataset
    print("\nPreparing dataset...")
    transform = get_default_transform(target_size=(32, 32))
    dataset = BalancedDataset(args.dataset, transform=transform, balance=args.balance)
    
    # Visualize class distribution
    plot_class_distribution(dataset, 
                           title="Class distribution in dataset",
                           save_path=os.path.join(args.output_dir, "class_distribution.png"))
    
    # Create dataloaders
    train_loader, test_loader = create_data_loaders(
        dataset,
        test_size=0.2,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model and scattering transform
    model, scattering = create_model(config)
    
    # Print model summary
    print_model_summary(model, scattering, config.device)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create trainer
    trainer = Trainer(model, scattering, config.device, optimizer)
    
    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    
    training_results = trainer.train(
        train_loader,
        test_loader,
        args.epochs,
        save_path=os.path.join(args.output_dir, "model.pth"),
        reduce_lr_after=args.reduce_lr_after,
        class_to_idx=dataset.class_to_idx
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\nTraining completed in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"Best accuracy: {training_results['best_accuracy']:.2f}%")
    
    # Visualize training metrics
    plot_training_metrics(
        args.epochs,
        training_results['train_accuracies'],
        training_results['test_accuracies'],
        training_results['train_losses'],
        training_results['test_losses'],
        os.path.join(args.output_dir, "training_metrics.png")
    )
    
    print(f"\nResults saved in {args.output_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()