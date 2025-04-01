#!/usr/bin/env python3
"""
Script for evaluating a trained WST classifier on a test dataset.
Usage: python evaluate_model.py [--model_path <model_path>] [--dataset_path <dataset_path>]
"""

import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt

# Add parent directory to path to import wavelet_lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wavelet_lib.base import Config, load_model
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders
from wavelet_lib.models import ScatteringClassifier, create_scattering_transform
from wavelet_lib.visualization import plot_confusion_matrix, visualize_batch

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a trained WST classifier on a test dataset')
    parser.add_argument('--model_path', type=str, 
                       default='/home/brus/Projects/wavelet/models/model_output_7/best_model.pth',
                       help='Path to the trained model')
    parser.add_argument('--dataset_path', type=str,
                       default='/home/brus/Projects/wavelet/datasets/HPL_images/custom_dataset',
                       help='Path to the test dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation (default: 128)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of the dataset to use for testing (default: 0.2)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save evaluation results')
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset directory not found: {args.dataset_path}")
        sys.exit(1)
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    transform = get_default_transform(target_size=(32, 32))
    dataset = BalancedDataset(args.dataset_path, transform=transform, balance=True)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        dataset, 
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print(f"Dataset loaded. {len(dataset)} samples, {len(dataset.classes)} classes.")
    print(f"Test set: {len(test_loader.dataset)} samples")
    
    # Create configuration
    config = Config(
        num_channels=3,
        num_classes=len(dataset.classes),
        scattering_order=2,
        J=2,
        shape=(32, 32),
        device=device
    )
    
    # Create scattering transform
    scattering = create_scattering_transform(
        J=config.J,
        shape=config.shape,
        max_order=config.scattering_order,
        device=config.device
    )
    
    # Create model
    model = ScatteringClassifier(
        in_channels=config.scattering_coeffs,
        classifier_type='cnn',
        num_classes=config.num_classes
    ).to(config.device)
    
    # Load model weights
    class_to_idx = load_model(model, args.model_path, device)
    print(f"Model loaded from: {args.model_path}")
    
    # Visualize a batch of test data
    if args.output_dir:
        print("Visualizing test batch...")
        visualize_batch(
            test_loader,
            num_images=16,
            save_path=os.path.join(args.output_dir, 'test_batch.png')
        )
    
    # Evaluate model
    print("Evaluating model...")
    confusion_mat, class_metrics = plot_confusion_matrix(
        model,
        scattering,
        test_loader,
        dataset.classes,
        device,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png') if args.output_dir else None
    )
    
    # Calculate per-class and overall performance metrics
    class_metrics_list = []
    for class_name, metrics in class_metrics.items():
        class_metrics_list.append({
            'class': class_name,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
    
    # Calculate overall metrics
    avg_precision = sum(m['precision'] for m in class_metrics_list) / len(class_metrics_list)
    avg_recall = sum(m['recall'] for m in class_metrics_list) / len(class_metrics_list)
    avg_f1 = sum(m['f1'] for m in class_metrics_list) / len(class_metrics_list)
    
    # Print performance summary
    print("\nPERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 60)
    
    for metrics in class_metrics_list:
        print(f"{metrics['class']:<20} | {metrics['precision']:.4f}     | {metrics['recall']:.4f}     | {metrics['f1']:.4f}")
    
    print("-" * 60)
    print(f"{'Average':<20} | {avg_precision:.4f}     | {avg_recall:.4f}     | {avg_f1:.4f}")
    print("=" * 60)
    
    # Detailed examination of misclassifications
    print("\nDetailed evaluation...", end="")
    
    # Function to evaluate model on dataset and collect misclassifications
    def evaluate_detailed(model, scattering, data_loader, device):
        model.eval()
        misclassifications = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                
                # Apply scattering and model
                scattering_coeffs = scattering(data)
                scattering_coeffs = scattering_coeffs.view(data.size(0), -1, 8, 8)
                output = model(scattering_coeffs)
                
                # Get predictions
                probabilities = torch.softmax(output, dim=1)
                confidence, predictions = torch.max(probabilities, dim=1)
                
                # Find misclassifications
                for i in range(len(data)):
                    if predictions[i] != target[i]:
                        misclassifications.append({
                            'true_class': target[i].item(),
                            'predicted_class': predictions[i].item(),
                            'confidence': confidence[i].item(),
                            'batch_idx': batch_idx,
                            'sample_idx': i
                        })
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(".", end="", flush=True)
        
        print(" Done!")
        return misclassifications
    
    # Get misclassifications
    misclassifications = evaluate_detailed(model, scattering, test_loader, device)
    
    # Print misclassification summary
    print(f"\nFound {len(misclassifications)} misclassified samples out of {len(test_loader.dataset)}.")
    
    if len(misclassifications) > 0:
        # Count misclassifications by class pair
        misclass_pairs = {}
        for m in misclassifications:
            true_class = dataset.classes[m['true_class']]
            pred_class = dataset.classes[m['predicted_class']]
            pair = (true_class, pred_class)
            
            if pair not in misclass_pairs:
                misclass_pairs[pair] = 0
            misclass_pairs[pair] += 1
        
        # Print most common misclassifications
        print("\nMOST COMMON MISCLASSIFICATIONS")
        print("=" * 60)
        print(f"{'True Class':<20} | {'Predicted As':<20} | {'Count':<10}")
        print("-" * 60)
        
        # Sort by count
        sorted_pairs = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)
        for (true_class, pred_class), count in sorted_pairs[:10]:  # Show top 10
            print(f"{true_class:<20} | {pred_class:<20} | {count:<10}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()