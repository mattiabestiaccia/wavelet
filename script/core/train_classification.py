#!/usr/bin/env python3
"""
Training script for Wavelet Scattering Transform classification models.

This script coordinates the entire training workflow for classification models, from data preparation
to model training and evaluation.

Usage:
    python script/core/train_classification.py --dataset /path/to/dataset --num-classes N --epochs E [options]
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
from wavelet_lib.classification import create_classification_model as create_model
from wavelet_lib.classification import print_classifier_summary as print_model_summary
from wavelet_lib.training import Trainer, create_optimizer
from wavelet_lib.visualization import plot_training_metrics, plot_class_distribution

def parse_args():
    """
    Analizza gli argomenti dalla riga di comando.

    Returns:
        args: Namespace contenente gli argomenti analizzati
    """
    parser = argparse.ArgumentParser(description='Addestra un modello Wavelet Scattering Transform')

    # Parametri del dataset
    parser.add_argument('--dataset', type=str, required=True, help='Percorso al dataset (obbligatorio)')
    parser.add_argument('--balance', action='store_true', help='Bilancia le classi nel dataset')

    # Parametri del modello
    parser.add_argument('--num-classes', type=int, default=4, help='Numero di classi (obbligatorio)')
    parser.add_argument('--num-channels', type=int, default=3, help='Numero di canali di input')
    parser.add_argument('--scattering-order', type=int, default=2, help='Ordine massimo della trasformata scattering')
    parser.add_argument('--j', type=int, default=2, help='Parametro J per la trasformata scattering')

    # Parametri di addestramento
    parser.add_argument('--batch-size', type=int, default=128, help='Dimensione del batch')
    parser.add_argument('--epochs', type=int, default=90, help='Numero di epoche di addestramento (obbligatorio)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate iniziale')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum per l\'ottimizzatore')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay per l\'ottimizzatore')
    parser.add_argument('--reduce-lr-after', type=int, default=20, help='Riduci il learning rate dopo questo numero di epoche')

    # Parametri generali di addestramento
    parser.add_argument('--seed', type=int, default=42, help='Seed per la riproducibilità')
    parser.add_argument('--device', type=str, default=None, help='Device per l\'addestramento (cuda o cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Numero di worker per i dataloader')

    # Parametri di output
    parser.add_argument('--output-dir', type=str, default=None, help='Directory per salvare i risultati')
    parser.add_argument('--experiment-name', type=str, default=None, help='Nome per questo esperimento (usato nel percorso di output)')
    parser.add_argument('--output-base', type=str, default=None, help='Directory base per salvare i risultati (default: results)')

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
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Set output base directory
    if args.output_base is None:
        output_base = os.path.join(base_dir, "results")
    else:
        # Handle both absolute and relative paths
        if os.path.isabs(args.output_base):
            output_base = args.output_base
        else:
            output_base = os.path.join(base_dir, args.output_base)

    # Configure output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use experiment name if provided
        if args.experiment_name:
            experiment_folder = f"{args.experiment_name}_{args.num_classes}_classes_{timestamp}"
        else:
            experiment_folder = f"model_output_{args.num_classes}_classes_{timestamp}"

        args.output_dir = os.path.join(output_base, "model_output", experiment_folder)

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
    transform = get_default_transform(target_size=(32, 32), dataset_root=args.dataset)
    dataset = BalancedDataset(args.dataset,
                              transform=transform,
                              balance=args.balance)

    # Visualize class distribution
    plot_class_distribution(dataset,
                            title="Class distribution in dataset",
                            save_path=os.path.join(args.output_dir, "class_distribution.png"))

    # Create dataloaders
    train_loader, test_loader = create_data_loaders(dataset,
                                                    test_size=0.2,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers)

    # Create model and scattering transform
    model, scattering = create_model(config)

    # Print model summary
    print_model_summary(model,
                        scattering,
                        config.device)

    # Create optimizer
    optimizer = create_optimizer(model,
                                 config)

    # Create trainer
    trainer = Trainer(model,
                      scattering,
                      config.device,
                      optimizer)

    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()

    training_results = trainer.train(train_loader,
                                     test_loader,
                                     args.epochs,
                                     save_path=os.path.join(args.output_dir, "model.pth"),
                                     reduce_lr_after=args.reduce_lr_after,
                                     class_to_idx=dataset.class_to_idx)

    # Calculate training time
    training_time = time.time() - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\nTraining completed in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"Best accuracy: {training_results['best_accuracy']:.2f}%")

    # Visualize training metrics
    plot_training_metrics(args.epochs,
                          training_results['train_accuracies'],
                          training_results['test_accuracies'],
                          training_results['train_losses'],
                          training_results['test_losses'],
                          os.path.join(args.output_dir, "training_metrics.png"))

    print(f"\nResults saved in {args.output_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()