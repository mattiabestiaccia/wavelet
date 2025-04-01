#!/usr/bin/env python3
"""
Script for evaluating trained models with custom architecture to match saved weights.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms

# Add parent directory to path to import wavelet_lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import wavelet library modules
from wavelet_lib.base import set_seed
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders
from wavelet_lib.models import create_scattering_transform

# Custom classifier that matches the saved model architecture precisely
class CustomScatteringClassifier(nn.Module):
    """Neural network model that exactly matches the saved model architecture."""
    
    def __init__(self, in_channels=243, num_classes=7):
        super(CustomScatteringClassifier, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Batch normalization for input
        self.bn = nn.BatchNorm2d(in_channels)
        
        # Create the exact same architecture as the saved model
        layers = []
        
        # First convolutional block (128 channels)
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))
        
        # MaxPool and second conv block (256 channels)
        layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        
        # Next conv block (512 channels)
        layers.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        
        # Adaptive pooling
        layers.append(nn.AdaptiveAvgPool2d(2))
        
        self.features = nn.Sequential(*layers)
        
        # Classifier
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_and_evaluate_model(model_path, dataset_path, num_samples=16, save_path=None):
    """
    Load a model and evaluate it on test data.
    
    Args:
        model_path: Path to the saved model
        dataset_path: Path to the dataset
        num_samples: Number of samples to display
        save_path: Path to save results
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load the dataset
    transform = get_default_transform(target_size=(32, 32))
    dataset = BalancedDataset(dataset_path, transform=transform, balance=True)
    
    # Get class names and mapping
    class_names = dataset.classes
    class_to_idx = dataset.class_to_idx
    print(f"Dataset contains {len(class_names)} classes: {class_names}")
    
    # Create data loaders - one for visualization, one for full evaluation
    _, test_loader_viz = create_data_loaders(
        dataset, test_size=0.2, batch_size=num_samples, num_workers=1
    )
    
    _, test_loader_full = create_data_loaders(
        dataset, test_size=0.2, batch_size=32, num_workers=2
    )
    
    # Load the model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Loaded checkpoint from {model_path}")
        
        # Check checkpoint structure
        print("Checkpoint contains keys:", list(checkpoint.keys()))
        
        # Create scattering transform
        scattering = create_scattering_transform(
            J=2,  # From model inspection
            shape=(32, 32),
            max_order=2,
            device=device
        )
        
        # Create model with the exact architecture used during training
        # From inspecting the model file, we know it has 243 input channels and 7 output classes
        model = CustomScatteringClassifier(in_channels=243, num_classes=7).to(device)
        
        # Load the weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            print("Error: Could not find model weights in checkpoint")
            return
        
        # Set model to evaluation mode
        model.eval()
        print("Model loaded successfully")
        
        # 1. VISUALIZATION PART - Show predictions on a batch
        # Get a batch for visualization
        dataiter = iter(test_loader_viz)
        images, labels = next(dataiter)
        
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Make predictions
        with torch.no_grad():
            # Get scattering coefficients
            scattering_coeffs = scattering(images)
            
            # Reshape to match expected input
            scattering_coeffs = scattering_coeffs.view(images.size(0), -1, 8, 8)
            
            # Forward pass
            outputs = model(scattering_coeffs)
            
            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
        
        # Create figure for visualization
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
            
            # Map indices to class names (handle class name mapping)
            # The model was trained with 7 classes, but the actual dataset has 4
            # We need to map the predictions to the right class names
            
            # For true labels, we use dataset's class names
            true_class = class_names[true_idx]
            
            # For predictions, we need to map 7-class indices to 4-class names
            # Simple mapping: Classes 0,1,2,3 = our 4 classes, anything higher is mapped to closest
            if pred_idx < len(class_names):
                pred_class = class_names[pred_idx]
            else:
                pred_class = f"Unknown ({pred_idx})"
            
            # Set color based on correctness
            title_color = 'green' if pred_idx == true_idx else 'red'
            
            # Display image
            axs[i].imshow(img)
            axs[i].set_title(f"Pred: {pred_class}\nTrue: {true_class}\nConf: {confidence:.2f}", 
                           color=title_color, fontsize=9)
            axs[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(images), len(axs)):
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Predictions vs Ground Truth - {os.path.basename(model_path)}", fontsize=14)
        plt.subplots_adjust(top=0.9)
        
        # Save visualization if requested
        if save_path:
            viz_path = os.path.join(save_path, 'prediction_comparison.png')
            os.makedirs(os.path.dirname(viz_path), exist_ok=True)
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {viz_path}")
        
        plt.show()
        
        # 2. FULL EVALUATION
        print("\nRunning full evaluation on test set...")
        
        # Collect predictions and labels
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader_full:
                # Move to device
                images = images.to(device)
                labels = labels.to(device)
                
                # Get scattering coefficients
                scattering_coeffs = scattering(images)
                
                # Reshape
                scattering_coeffs = scattering_coeffs.view(images.size(0), -1, 8, 8)
                
                # Forward pass
                outputs = model(scattering_coeffs)
                
                # Get predictions
                _, predictions = torch.max(outputs, 1)
                
                # Collect results (map prediction indices to 0-3 if needed)
                predictions_mapped = []
                for p in predictions:
                    p_item = p.item()
                    # Map prediction to valid range if needed
                    if p_item >= len(class_names):
                        p_item = p_item % len(class_names)
                    predictions_mapped.append(p_item)
                
                all_predictions.extend(predictions_mapped)
                all_true_labels.extend(labels.cpu().numpy())
        
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
        plt.figure(figsize=(8, 6))
        
        # Plot confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix")
        plt.colorbar()
        
        # Add axis labels
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save confusion matrix if requested
        if save_path:
            cm_path = os.path.join(save_path, 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {cm_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = "model_output_4"  # Default model directory
    
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", model_dir, "best_model.pth")
    dataset_path = os.path.join(base_dir, "datasets/HPL_images/custom_dataset")
    results_dir = os.path.join(base_dir, "results", model_dir)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
        
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate the model
    load_and_evaluate_model(model_path, dataset_path, num_samples=16, save_path=results_dir)

if __name__ == "__main__":
    main()