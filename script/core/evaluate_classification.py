#!/usr/bin/env python3
"""
Evaluation script for Wavelet Scattering Transform classification models.

This script performs a comprehensive evaluation of a classification model on a dataset,
providing detailed metrics and visualizations specific to classification tasks.

Usage:
    python script/core/evaluate_classification.py --model-path /path/to/model.pth --dataset /path/to/dataset [options]
"""

import os
import sys
import torch
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add the main directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import wavelet_lib modules
from wavelet_lib.base import Config, load_model
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders
from wavelet_lib.classification import create_scattering_transform, ScatteringClassifier

def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate a Wavelet Scattering Transform model')

    # Model parameters
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')

    # Evaluation parameters
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--balance', action='store_true', help='Balance classes in the dataset')
    parser.add_argument('--sample-count', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--fast', action='store_true',
                        help='Use faster evaluation with reduced sample count')

    # General parameters
    parser.add_argument('--device', type=str, default=None, help='Device for inference (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--experiment-name', type=str, default=None, help='Name for this experiment (used in output path)')
    parser.add_argument('--output-base', type=str, default=None, help='Base directory for storing results (default: results)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for dataloaders')

    args = parser.parse_args()

    # If fast mode is enabled, set sample count to 1000 by default
    if args.fast and args.sample_count is None:
        args.sample_count = 1000
        print(f"Fast mode enabled: Evaluating on {args.sample_count} samples")

    return args

def create_model_for_evaluation(num_classes, device):
    """
    Crea un modello con i parametri corretti per la valutazione.
    """
    config = Config(
        num_channels=3,
        num_classes=num_classes,
        scattering_order=2,
        J=2,
        shape=(32, 32),
        device=device
    )

    # Crea il modello con 12 canali di input come nel checkpoint
    model = ScatteringClassifier(
        in_channels=12,  # Usa 12 canali come nel modello originale
        classifier_type='cnn',
        num_classes=num_classes
    ).to(device)

    # Crea la trasformata scattering
    scattering = create_scattering_transform(
        J=config.J,
        shape=config.shape,
        max_order=config.scattering_order,
        device=device
    )

    return model, scattering

def load_model_checkpoint(model_path, device):
    """
    Carica il checkpoint del modello.
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Ottieni il numero di classi dal checkpoint
    num_classes = len(checkpoint.get('class_to_idx', {}))
    if num_classes == 0:
        # Se class_to_idx non è presente, prova a determinare dal layer finale
        out_features = [v.shape[0] for k, v in checkpoint['model_state_dict'].items() if 'classifier.bias' in k]
        num_classes = out_features[0] if out_features else 4

    # Crea il modello con i parametri corretti
    model, scattering = create_model_for_evaluation(num_classes, device)

    # Carica i pesi
    model.load_state_dict(checkpoint['model_state_dict'])

    # Ottieni i nomi delle classi
    class_names = list(checkpoint.get('class_to_idx', {}).keys())
    if not class_names:
        class_names = [f'Class {i}' for i in range(num_classes)]

    # Ottimizza per l'inferenza
    if device == 'cuda':
        # Utilizziamo torch.compile se disponibile (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                model = torch.compile(model)
                print("Model compiled for faster inference.")
            except Exception as e:
                print(f"Warning: Could not compile model: {str(e)}")

    return model, scattering, class_names, 12  # Ritorna 12 come numero di canali

def evaluate_model(model, scattering, test_loader, device, class_names, output_dir=None, sample_limit=None):
    """
    Evaluate the model on a test dataset.

    Args:
        model: Trained model
        scattering: Scattering transform
        test_loader: DataLoader for testing
        device: Device for inference
        class_names: Class names
        output_dir: Directory to save partial results
        sample_limit: Limit the number of samples to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    # Prepare for metrics
    all_predictions = []
    all_targets = []

    # Calculate total samples to process
    total_samples = len(test_loader.dataset)
    if sample_limit and sample_limit < total_samples:
        total_samples = sample_limit
        print(f"Limiting evaluation to {sample_limit} samples")

    # Check if partial results exist
    partial_results_path = None
    if output_dir:
        partial_results_path = os.path.join(output_dir, "partial_results.npz")

    # Check for existing results
    if partial_results_path and os.path.exists(partial_results_path):
        print(f"Loading partial results from {partial_results_path}")
        try:
            partial_results = np.load(partial_results_path)
            all_predictions = partial_results['predictions'].tolist()
            all_targets = partial_results['targets'].tolist()

            # Verify that predictions and targets have the same length
            if len(all_predictions) != len(all_targets):
                print(f"WARNING: Inconsistent number of samples in partial results: {len(all_predictions)} predictions vs {len(all_targets)} targets")
                print("Discarding partial results and starting fresh")
                all_predictions = []
                all_targets = []
                processed_samples = 0
                # Remove corrupted partial results
                os.remove(partial_results_path)
            else:
                processed_samples = len(all_predictions)
                print(f"Loaded {processed_samples} processed samples")
        except Exception as e:
            print(f"Error loading partial results: {str(e)}")
            print("Starting evaluation from scratch")
            all_predictions = []
            all_targets = []
            processed_samples = 0
            # Remove corrupted partial results
            if os.path.exists(partial_results_path):
                os.remove(partial_results_path)
    else:
        processed_samples = 0

    # Evaluation loop
    try:
        with torch.no_grad():
            total_samples = len(test_loader.dataset)
            progress_bar = tqdm(test_loader, desc="Evaluation")

            for batch_idx, (data, target) in enumerate(progress_bar):
                # Skip already processed batches
                current_batch_start = batch_idx * test_loader.batch_size
                if current_batch_start < processed_samples:
                    # Display progress for skipped batches
                    progress_bar.set_description(f"Skipping (already processed)")
                    continue

                data, target = data.to(device), target.to(device)

                # Apply scattering transform
                scattering_coeffs = scattering(data)

                # The model will handle reshaping internally - pass the scattering coefficients directly

                # Forward pass
                outputs = model(scattering_coeffs)

                # Calculate predictions
                _, predictions = torch.max(outputs, 1)

                # Gather results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                # Check if we've reached the sample limit
                if sample_limit and len(all_predictions) >= sample_limit:
                    # Truncate to sample limit
                    all_predictions = all_predictions[:sample_limit]
                    all_targets = all_targets[:sample_limit]
                    break

                # Update progress information
                progress_bar.set_postfix({
                    'processed': len(all_predictions),
                    'correct': sum(1 for pred, tgt in zip(predictions.cpu().numpy(), target.cpu().numpy()) if pred == tgt),
                    'samples': min(len(all_predictions), total_samples if sample_limit else len(all_predictions)),
                })

    except (KeyboardInterrupt, Exception) as e:
        # Save partial results on interruption
        if partial_results_path and all_predictions:
            print(f"\nSaving partial results to {partial_results_path}")
            np.savez(partial_results_path,
                    predictions=np.array(all_predictions),
                    targets=np.array(all_targets))
            print(f"Processed {len(all_predictions)} samples before interruption")

        if isinstance(e, KeyboardInterrupt):
            print("Evaluation interrupted by user.")
        else:
            print(f"Evaluation interrupted due to error: {str(e)}")
            raise e

    if len(all_predictions) == 0:
        print("No predictions collected. Cannot compute metrics.")
        return None

    # Double check length consistency
    if len(all_predictions) != len(all_targets):
        print(f"WARNING: Inconsistent lengths - {len(all_predictions)} predictions vs {len(all_targets)} targets")
        # Make sure lengths match by truncating to the shorter length
        min_len = min(len(all_predictions), len(all_targets))
        all_predictions = all_predictions[:min_len]
        all_targets = all_targets[:min_len]
        print(f"Truncated to {min_len} samples for evaluation")

    # Calculate evaluation metrics
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    class_report = classification_report(all_targets, all_predictions,
                                        target_names=class_names,
                                        output_dict=True)

    # Accuracy
    accuracy = sum(1 for pred, target in zip(all_predictions, all_targets) if pred == target) / len(all_targets)

    # Save final results
    if partial_results_path:
        # Remove partial results file
        if os.path.exists(partial_results_path):
            os.remove(partial_results_path)

    return {
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'accuracy': accuracy,
        'predictions': all_predictions,
        'targets': all_targets
    }

def plot_confusion_matrix(conf_matrix, class_names, save_path=None):
    """
    Visualize the confusion matrix.

    Args:
        conf_matrix: Confusion matrix
        class_names: Class names
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Prediction')
    plt.ylabel('True Value')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()

def plot_metrics(metrics, class_names, save_path=None):
    """
    Visualize evaluation metrics.

    Args:
        metrics: Dictionary with metrics
        class_names: Class names
        save_path: Path to save the visualization
    """
    report = metrics['classification_report']

    # Extract precision, recall, f1-score for each class
    classes_data = {class_name: report[class_name] for class_name in class_names}

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Set the number of groups
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.25

    # Extract metrics
    precision = [classes_data[name]['precision'] for name in class_names]
    recall = [classes_data[name]['recall'] for name in class_names]
    f1 = [classes_data[name]['f1-score'] for name in class_names]

    # Bar chart
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')

    plt.ylabel('Score')
    plt.title('Evaluation Metrics by Class')
    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add scores above each bar
    for i, v in enumerate(precision):
        plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(recall):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(f1):
        plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation metrics saved to: {save_path}")

    plt.show()

def load_evaluation_data(dataset_path, batch_size=64, num_workers=4):
    """
    Carica il dataset per la valutazione.
    """
    # Crea il dataset
    transform = get_default_transform()
    dataset = BalancedDataset(dataset_path, transform=transform)

    # Crea il dataloader per l'intero dataset
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Non mescoliamo per la valutazione
        num_workers=num_workers
    )

    return loader, dataset.classes

def main():
    """
    Main function for model evaluation.
    """
    # Parse command line arguments
    args = parse_args()

    # Configure device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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
        model_basename = os.path.splitext(os.path.basename(args.model_path))[0]

        # Use experiment name if provided
        if args.experiment_name:
            eval_dir = f"{args.experiment_name}_{model_basename}"
        else:
            eval_dir = model_basename

        args.output_dir = os.path.join(output_base, "evaluation", eval_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Wavelet Scattering Transform Model Evaluation")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    # Print sample count if specified
    if args.fast:
        print(f"Fast mode: enabled (sample count = {args.sample_count})")
    elif args.sample_count:
        print(f"Sample count: {args.sample_count} (limited evaluation)")

    print(f"{'='*80}\n")

    # Load model
    model, scattering, class_names, in_channels = load_model_checkpoint(args.model_path, device)

    # Prepare dataset
    transform = get_default_transform(target_size=(32, 32))
    dataset = BalancedDataset(args.dataset, transform=transform, balance=args.balance)

    # Limit dataset size if sample_count is specified
    if args.sample_count:
        # If it's a subset created by fast mode, update the batch size
        print(f"Original dataset size: {len(dataset)} samples")
        if args.batch_size > args.sample_count // 10:
            # Adjust batch size to be smaller (1/10 of the samples)
            args.batch_size = max(1, args.sample_count // 10)
            print(f"Adjusted batch size to {args.batch_size} for fast evaluation")

        from torch.utils.data import Subset, Dataset
        import random
        import numpy as np

        class CustomSubset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
                self.classes = dataset.classes if hasattr(dataset, 'classes') else None

                # Copy necessary attributes
                if hasattr(dataset, 'samples'):
                    # Get only the selected samples
                    self.samples = [dataset.samples[i] for i in indices]

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]

        # Create a stratified subset
        indices_by_class = {}
        for idx, (_, class_idx) in enumerate(dataset.samples):
            if class_idx not in indices_by_class:
                indices_by_class[class_idx] = []
            indices_by_class[class_idx].append(idx)

        # Calculate samples per class
        num_classes = len(indices_by_class)
        samples_per_class = args.sample_count // num_classes

        # Create balanced subset
        selected_indices = []
        for class_idx, indices in indices_by_class.items():
            selected_indices.extend(random.sample(indices, min(samples_per_class, len(indices))))

        # Shuffle the indices
        np.random.shuffle(selected_indices)

        # Update dataset - use our custom subset class
        dataset = CustomSubset(dataset, selected_indices)
        print(f"Using {len(selected_indices)} samples from {num_classes} classes")

        # Create dataloader directly for evaluation - skip create_data_loaders
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        # Create dataloaders normally for full dataset
        _, test_loader = create_data_loaders(
            dataset,
            test_size=1.0,  # Use the entire dataset for evaluation
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    # Evaluate model
    print("\nStarting model evaluation...")
    metrics = evaluate_model(model, scattering, test_loader, device, class_names, args.output_dir,
                            sample_limit=args.sample_count)

    # Print results
    print("\nEvaluation results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    print("\nClassification report:")
    print(classification_report(metrics['targets'], metrics['predictions'], target_names=class_names))

    # Visualize confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path=os.path.join(args.output_dir, "confusion_matrix.png")
    )

    # Visualize evaluation metrics
    plot_metrics(
        metrics,
        class_names,
        save_path=os.path.join(args.output_dir, "evaluation_metrics.png")
    )

    print(f"\nEvaluation completed! Results saved to {args.output_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
