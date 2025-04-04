#!/usr/bin/env python3
"""
Visualization script for model results and dataset analysis.

This script provides various visualization functions for model results,
training metrics, and dataset analysis for both classification and segmentation tasks.

Usage:
    python script/core/visualize_results.py metrics --model-dir /path/to/model_output
    python script/core/visualize_results.py samples --model-path /path/to/model.pth --dataset /path/to/dataset
    python script/core/visualize_results.py distribution --dataset /path/to/dataset
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
import random

# Add the main directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import wavelet_lib modules
from wavelet_lib.base import load_model
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders
from wavelet_lib.models import create_scattering_transform, ScatteringClassifier
from wavelet_lib.visualization import plot_training_metrics, plot_class_distribution

def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(description='Visualize Wavelet Scattering Transform model outputs')
    subparsers = parser.add_subparsers(dest='command', help='Visualization command')

    # Metrics visualization
    metrics_parser = subparsers.add_parser('metrics', help='Visualize training metrics')
    metrics_parser.add_argument('--model-dir', type=str, required=True,
                               help='Path to model output directory')
    metrics_parser.add_argument('--output-dir', type=str, default=None,
                               help='Directory to save visualizations')
    metrics_parser.add_argument('--experiment-name', type=str, default=None,
                               help='Name for this experiment (used in output path)')
    metrics_parser.add_argument('--output-base', type=str, default=None,
                               help='Base directory for storing results (default: results)')

    # Sample visualization
    samples_parser = subparsers.add_parser('samples', help='Visualize dataset samples')
    samples_parser.add_argument('--model-path', type=str, required=True,
                               help='Path to model file')
    samples_parser.add_argument('--dataset', type=str, required=True,
                               help='Path to dataset directory')
    samples_parser.add_argument('--num-samples', type=int, default=5,
                               help='Number of samples per class to visualize')
    samples_parser.add_argument('--output-dir', type=str, default=None,
                               help='Directory to save visualizations')
    samples_parser.add_argument('--experiment-name', type=str, default=None,
                               help='Name for this experiment (used in output path)')
    samples_parser.add_argument('--output-base', type=str, default=None,
                               help='Base directory for storing results (default: results)')

    # Distribution visualization
    dist_parser = subparsers.add_parser('distribution', help='Visualize class distribution')
    dist_parser.add_argument('--dataset', type=str, required=True,
                            help='Path to dataset directory')
    dist_parser.add_argument('--output-dir', type=str, default=None,
                            help='Directory to save visualizations')
    dist_parser.add_argument('--experiment-name', type=str, default=None,
                            help='Name for this experiment (used in output path)')
    dist_parser.add_argument('--output-base', type=str, default=None,
                            help='Base directory for storing results (default: results)')

    return parser.parse_args()

def visualize_metrics(model_dir, output_dir=None, experiment_name=None, output_base=None):
    """
    Visualize training metrics from a saved model.

    Args:
        model_dir: Path to model output directory
        output_dir: Directory to save visualizations
    """
    # Handle relative paths
    if not os.path.isabs(model_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # Check if path already includes results/ or models/ prefix
        if model_dir.startswith("results/") or model_dir.startswith("results\\"):
            model_dir = os.path.join(base_dir, model_dir)
        elif model_dir.startswith("models/") or model_dir.startswith("models\\"):
            model_dir = os.path.join(base_dir, model_dir)
        else:
            # Default to results directory
            model_dir = os.path.join(base_dir, "results", model_dir)

    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return

    # Set output directory
    if output_dir is None:
        output_dir = model_dir

    print(f"Loading metrics from: {model_dir}")

    # Try to find metrics files
    metrics_file = os.path.join(model_dir, "metrics.npz")

    if os.path.exists(metrics_file):
        # Load metrics from NPZ file
        metrics = np.load(metrics_file)
        epochs = metrics['epochs']
        train_accuracies = metrics['train_accuracies']
        test_accuracies = metrics['test_accuracies']
        train_losses = metrics['train_losses']
        test_losses = metrics['test_losses']
    else:
        # Look for checkpoint file
        checkpoint_path = os.path.join(model_dir, "checkpoint.pth")
        best_model_path = os.path.join(model_dir, "best_model.pth")
        final_model_path = os.path.join(model_dir, "final_model.pth")

        if os.path.exists(checkpoint_path):
            model_path = checkpoint_path
        elif os.path.exists(best_model_path):
            model_path = best_model_path
        elif os.path.exists(final_model_path):
            model_path = final_model_path
        else:
            print(f"Error: No metrics or model files found in {model_dir}")
            return

        print(f"Loading metrics from checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'training_metrics' in checkpoint:
            metrics = checkpoint['training_metrics']
            epochs = metrics.get('epochs', 90)
            train_accuracies = metrics.get('train_accuracies', [])
            test_accuracies = metrics.get('test_accuracies', [])
            train_losses = metrics.get('train_losses', [])
            test_losses = metrics.get('test_losses', [])
        elif all(key in checkpoint for key in ['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']):
            # Handle single epoch metrics
            print("Found single epoch metrics in checkpoint")
            epochs = 1
            train_accuracies = [checkpoint['train_accuracy']]
            test_accuracies = [checkpoint['test_accuracy']]
            train_losses = [checkpoint['train_loss']]
            test_losses = [checkpoint['test_loss']]
        else:
            print(f"Error: No training metrics found in checkpoint")
            return

    # Plot metrics
    print(f"Plotting metrics for {len(train_accuracies)} epochs")
    plot_training_metrics(
        epochs=len(train_accuracies),
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
        train_losses=train_losses,
        test_losses=test_losses,
        save_path=os.path.join(output_dir, "metrics_plot.png")
    )

    print(f"Metrics visualization saved to: {os.path.join(output_dir, 'metrics_plot.png')}")

def visualize_samples(model_path, dataset_path, num_samples=5, output_dir=None, experiment_name=None, output_base=None):
    """
    Visualize samples from each class with their predictions.

    Args:
        model_path: Path to model file
        dataset_path: Path to dataset directory
        num_samples: Number of samples per class to visualize
        output_dir: Directory to save visualizations
    """
    # Check if paths exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found: {dataset_path}")
        return

    # Configure output directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Set output base directory
    if output_base is None:
        output_base = os.path.join(base_dir, "results")
    else:
        # Handle both absolute and relative paths
        if os.path.isabs(output_base):
            output_base = output_base
        else:
            output_base = os.path.join(base_dir, output_base)

    # Set output directory
    if output_dir is None:
        # Use experiment name if provided
        viz_dir = "visualization"
        if experiment_name:
            viz_dir = os.path.join(viz_dir, experiment_name)

        output_dir = os.path.join(output_base, viz_dir)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Get class information
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        class_names = list(class_to_idx.keys())
        num_classes = len(class_names)
    else:
        # Try to get class names from dataset
        try:
            class_names = sorted([d for d in os.listdir(dataset_path)
                                if os.path.isdir(os.path.join(dataset_path, d))])
            if not class_names:
                raise ValueError("No class directories found")
        except Exception as e:
            print(f"Warning: Could not get class names from dataset: {str(e)}")
            class_names = ["Class_0", "Class_1", "Class_2", "Class_3"]
        num_classes = len(class_names)
        class_to_idx = {name: i for i, name in enumerate(class_names)}

    print(f"Found {num_classes} classes: {', '.join(class_names)}")

    # Create scattering transform
    scattering = create_scattering_transform(
        J=2,
        shape=(32, 32),
        max_order=2,
        device=device
    )

    # Try to get the number of channels from the model checkpoint
    if 'model_state_dict' in checkpoint:
        # Look for the first batch normalization layer to get the channel count
        for key, value in checkpoint['model_state_dict'].items():
            if 'bn.weight' in key:
                in_channels = value.size(0)
                print(f"Detected input channels from checkpoint: {in_channels}")
                break
    else:
        # Fallback to default value if not found
        in_channels = 12
        print(f"Using default input channels: {in_channels}")

    # Create model with correct number of channels
    model = ScatteringClassifier(in_channels=in_channels, num_classes=num_classes).to(device)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("Error: Could not find model weights in checkpoint")
        return

    model.eval()

    # Create dataset
    transform = get_default_transform(target_size=(32, 32))
    dataset = BalancedDataset(dataset_path, transform=transform, balance=False)

    # Visualize samples for each class
    for class_name in class_names:
        class_idx = class_to_idx[class_name]

        # Find samples for this class
        class_samples = [(i, path) for i, (path, idx) in enumerate(dataset.samples) if idx == class_idx]

        if not class_samples:
            print(f"No samples found for class {class_name}")
            continue

        # Randomly select samples
        selected_samples = random.sample(class_samples, min(num_samples, len(class_samples)))

        # Create figure
        fig, axes = plt.subplots(1, len(selected_samples), figsize=(4*len(selected_samples), 4))
        if len(selected_samples) == 1:
            axes = [axes]

        # Process each sample
        for i, (sample_idx, sample_path) in enumerate(selected_samples):
            # Load image
            img = Image.open(sample_path).convert('RGB')

            # Get tensor
            tensor = dataset[sample_idx][0].unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                scattering_coeffs = scattering(tensor)
                output = model(scattering_coeffs)
                probabilities = torch.softmax(output, dim=1)
                max_prob, prediction = torch.max(probabilities, dim=1)

            # Display
            axes[i].imshow(img)
            pred_class = class_names[prediction.item()]
            conf = max_prob.item()

            if pred_class == class_name:
                color = 'green'
            else:
                color = 'red'

            axes[i].set_title(f"Pred: {pred_class}\nConf: {conf:.2f}", color=color)
            axes[i].axis('off')

        plt.suptitle(f"Class: {class_name}")
        plt.tight_layout()

        # Save figure
        save_path = os.path.join(output_dir, f"class_{class_name}_examples.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {class_name} examples to: {save_path}")
        plt.close()

    print(f"All sample visualizations saved to: {output_dir}")

def visualize_distribution(dataset_path, output_dir=None, experiment_name=None, output_base=None):
    """
    Visualize class distribution in a dataset.

    Args:
        dataset_path: Path to dataset directory
        output_dir: Directory to save visualization
    """
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found: {dataset_path}")
        return

    # Configure output directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Set output base directory
    if output_base is None:
        output_base = os.path.join(base_dir, "results")
    else:
        # Handle both absolute and relative paths
        if os.path.isabs(output_base):
            output_base = output_base
        else:
            output_base = os.path.join(base_dir, output_base)

    # Set output directory
    if output_dir is None:
        # Use experiment name if provided
        viz_dir = "visualization"
        if experiment_name:
            viz_dir = os.path.join(viz_dir, experiment_name)

        output_dir = os.path.join(output_base, viz_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Create dataset
    dataset = BalancedDataset(dataset_path, transform=None, balance=False)

    # Visualize class distribution
    print(f"Visualizing class distribution for dataset: {dataset_path}")
    save_path = os.path.join(output_dir, "class_distribution.png")
    plot_class_distribution(dataset, title="Class Distribution in Dataset", save_path=save_path)

    print(f"Class distribution visualization saved to: {save_path}")

def main():
    """
    Main function for visualization.
    """
    # Parse command line arguments
    args = parse_args()

    if args.command is None:
        print("Error: No command specified. Use 'metrics', 'samples', or 'distribution'.")
        return

    if args.command == 'metrics':
        visualize_metrics(
            args.model_dir,
            args.output_dir,
            args.experiment_name,
            args.output_base
        )
    elif args.command == 'samples':
        visualize_samples(
            args.model_path,
            args.dataset,
            args.num_samples,
            args.output_dir,
            args.experiment_name,
            args.output_base
        )
    elif args.command == 'distribution':
        visualize_distribution(
            args.dataset,
            args.output_dir,
            args.experiment_name,
            args.output_base
        )

if __name__ == "__main__":
    main()
