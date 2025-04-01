#!/usr/bin/env python3
"""
Script to predict and compare results with ground truth labels.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# Add parent directory to path to import wavelet_lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import wavelet library modules
from wavelet_lib.base import Config, set_seed
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders
from wavelet_lib.models import create_model
from wavelet_lib.visualization import plot_confusion_matrix

def run_test_predictions(dataset_path, model_dir, num_samples=16, save_dir=None):
    """
    Run predictions on test samples and compare with ground truth.
    
    Args:
        dataset_path: Path to the dataset directory
        model_dir: Path to the model directory
        num_samples: Number of samples to visualize
        save_dir: Directory to save results
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create dataset
    transform = get_default_transform(target_size=(32, 32))
    dataset = BalancedDataset(dataset_path, transform=transform, balance=True)
    
    # Get class names
    class_names = dataset.classes
    print(f"Classes: {class_names}")
    
    # Create data loaders with small batch size (for visualization)
    _, test_loader = create_data_loaders(
        dataset, 
        test_size=0.2, 
        batch_size=num_samples,  # Using num_samples as batch size
        num_workers=1
    )
    
    # Create configuration
    config = Config(
        num_channels=3,
        num_classes=len(class_names),
        scattering_order=2,
        J=2,
        shape=(32, 32),
        device=device
    )
    
    # Create model and scattering transform
    model, scattering = create_model(config)
    
    # Load model weights
    model_path = os.path.join(model_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load model weights (handling different key formats)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            # Try to load with strict=False to ignore mismatched keys
            model.load_state_dict(checkpoint['model_state'], strict=False)
            print("Loaded model weights with some mismatches (strict=False)")
        else:
            print("Error: No model weights found in checkpoint")
            return
            
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Move to device
    images = images.to(device)
    labels = labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        # Forward pass with scattering
        scattering_coeffs = scattering(images)
        
        # Reshaping for the custom model structure
        batch_size = images.size(0)
        scattering_coeffs = scattering_coeffs.view(batch_size, -1, 8, 8)
        
        # We'll select the first 12 channels if needed for the model
        num_channels = model.in_channels
        num_channels_available = scattering_coeffs.size(1)
        
        if num_channels_available >= num_channels:
            scattering_coeffs = scattering_coeffs[:, :num_channels, :, :]
        else:
            # Repeat channels if we don't have enough
            repeats = (num_channels + num_channels_available - 1) // num_channels_available
            repeated = scattering_coeffs.repeat(1, repeats, 1, 1)
            scattering_coeffs = repeated[:, :num_channels, :, :]
            
        # Forward pass through model
        outputs = model(scattering_coeffs)
        
        # Get predicted classes and confidence scores
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
    
    # Create a figure to display results
    rows = int(np.ceil(num_samples / 4))
    fig, axs = plt.subplots(rows, 4, figsize=(12, 3*rows))
    axs = axs.flatten() if rows > 1 else [axs]
    
    # Display results
    for i in range(min(num_samples, len(images))):
        # Convert tensor to numpy array and denormalize
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5  # Denormalize
        img = np.clip(img, 0, 1)
        
        # Get predicted and true labels
        pred_idx = predictions[i].item()
        true_idx = labels[i].item()
        confidence = confidences[i].item()
        
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Unknown ({pred_idx})"
        true_label = class_names[true_idx] if true_idx < len(class_names) else f"Unknown ({true_idx})"
        
        # Set color based on prediction correctness
        title_color = 'green' if pred_idx == true_idx else 'red'
        
        # Display image
        axs[i].imshow(img)
        axs[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}\nConf: {confidence:.2f}", 
                       color=title_color, fontsize=9)
        axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(images), len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Predictions vs Ground Truth", fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # Save figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, 'prediction_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison image saved to: {output_path}")
    
    # Show figure
    plt.show()
    
    # Now run predictions on the entire test set to calculate accuracy
    print("\nRunning predictions on entire test set...")
    
    # Create a new test loader with a larger batch size
    _, full_test_loader = create_data_loaders(
        dataset, 
        test_size=0.2, 
        batch_size=32,
        num_workers=2
    )
    
    # Collect all predictions and true labels
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for images, labels in full_test_loader:
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass with scattering
            scattering_coeffs = scattering(images)
            batch_size = images.size(0)
            
            # Reshape for model
            scattering_coeffs = scattering_coeffs.view(batch_size, -1, 8, 8)
            
            # Take first 12 channels if needed
            num_channels = model.in_channels
            num_channels_available = scattering_coeffs.size(1)
            
            if num_channels_available >= num_channels:
                scattering_coeffs = scattering_coeffs[:, :num_channels, :, :]
            else:
                # Repeat channels if needed
                repeats = (num_channels + num_channels_available - 1) // num_channels_available
                repeated = scattering_coeffs.repeat(1, repeats, 1, 1)
                scattering_coeffs = repeated[:, :num_channels, :, :]
            
            # Forward pass through model
            outputs = model(scattering_coeffs)
            
            # Get predictions and confidences
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(all_true_labels, all_predictions, 
                                 target_names=class_names, digits=3)
    print(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cm, class_names)
    
    # Save confusion matrix if save_dir is provided
    if save_dir:
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {cm_path}")
    
    plt.show()
    
    # Print high confidence incorrect predictions
    incorrect_indices = np.where(np.array(all_predictions) != np.array(all_true_labels))[0]
    incorrect_confidences = [all_confidences[i] for i in incorrect_indices]
    incorrect_preds = [all_predictions[i] for i in incorrect_indices]
    incorrect_true = [all_true_labels[i] for i in incorrect_indices]
    
    if incorrect_indices.size > 0:
        print("\nTop 5 most confident incorrect predictions:")
        sorted_indices = np.argsort(incorrect_confidences)[::-1][:5]
        
        for i, idx in enumerate(sorted_indices):
            pred_class = class_names[incorrect_preds[idx]]
            true_class = class_names[incorrect_true[idx]]
            confidence = incorrect_confidences[idx]
            print(f"{i+1}. Predicted {pred_class} instead of {true_class} with confidence {confidence:.4f}")
    else:
        print("\nNo incorrect predictions found!")

def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        # Default to model_output_4
        model_dir = "model_output_4"
    
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "datasets/HPL_images/custom_dataset")
    model_dir_path = os.path.join(base_dir, "models", model_dir)
    save_dir = os.path.join(base_dir, "results", model_dir)
    
    # Check if paths exist
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return
        
    if not os.path.exists(model_dir_path):
        print(f"Error: Model directory not found: {model_dir_path}")
        print("Available model directories:")
        models_dir = os.path.join(base_dir, "models")
        for d in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("model_output"):
                print(f"- {d}")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    
    # Run predictions
    run_test_predictions(dataset_path, model_dir_path, num_samples=16, save_dir=save_dir)

if __name__ == "__main__":
    main()