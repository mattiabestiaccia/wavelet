#!/usr/bin/env python3
"""
Utility functions for model management and analysis.

This module provides helper functions for working with models,
including loading, saving, analyzing, and converting models.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Import wavelet_lib modules
from wavelet_lib.base import load_model
from wavelet_lib.models import create_scattering_transform, ScatteringClassifier


def list_model_checkpoints(directory=None):
    """
    List all model checkpoints in a directory.
    
    Args:
        directory: Directory to search (defaults to models/)
        
    Returns:
        list: List of model checkpoints
    """
    if directory is None:
        directory = Path.cwd() / "models"
    else:
        directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return []
    
    # Find all model checkpoint files
    checkpoints = []
    for checkpoint_path in directory.glob("**/*.pth"):
        checkpoints.append(str(checkpoint_path))
    
    return checkpoints


def load_latest_checkpoint(directory=None, model=None):
    """
    Load the latest model checkpoint.
    
    Args:
        directory: Directory to search (defaults to models/)
        model: Model to load checkpoint into
        
    Returns:
        tuple: (model, checkpoint_info)
    """
    if directory is None:
        directory = Path.cwd() / "models"
    else:
        directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return None, None
    
    # Find all model checkpoint files
    checkpoints = []
    for checkpoint_path in directory.glob("**/*.pth"):
        checkpoints.append(checkpoint_path)
    
    if not checkpoints:
        print(f"Error: No checkpoints found in {directory}")
        return None, None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_checkpoint = checkpoints[0]
    
    print(f"Loading latest checkpoint: {latest_checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(latest_checkpoint)
    
    # Create model if not provided
    if model is None:
        if 'config' in checkpoint:
            config = checkpoint['config']
            model = create_scattering_transform(config)
        else:
            print("Error: No model provided and no config in checkpoint")
            return None, None
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def visualize_model_performance(history):
    """
    Visualize model training history.
    
    Args:
        history: Training history dictionary
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training and validation accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), normalize=True):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size (width, height)
        normalize: Whether to normalize the confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_training_history(history_path):
    """
    Plot training history from a saved history file.
    
    Args:
        history_path: Path to history file
    """
    # Load history
    history = torch.load(history_path)
    
    # Visualize performance
    visualize_model_performance(history)
    
    # Print final metrics
    print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
    
    # Print best epoch
    best_epoch = np.argmax(history['val_acc'])
    print(f"Best Epoch: {best_epoch + 1}")
    print(f"Best Validation Accuracy: {history['val_acc'][best_epoch]:.4f}")
    print(f"Training Accuracy at Best Epoch: {history['train_acc'][best_epoch]:.4f}")


def analyze_model_predictions(model, dataloader, class_names=None, num_samples=5):
    """
    Analyze model predictions on a dataset.
    
    Args:
        model: Model to analyze
        dataloader: DataLoader with samples
        class_names: List of class names
        num_samples: Number of samples to visualize per class
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables
    all_preds = []
    all_labels = []
    samples_by_class = {}
    
    # Collect predictions and samples
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move to device
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store samples for visualization
            for i in range(inputs.size(0)):
                label = labels[i].item()
                pred = preds[i].item()
                
                # Initialize if needed
                if label not in samples_by_class:
                    samples_by_class[label] = {'correct': [], 'incorrect': []}
                
                # Store sample
                sample = inputs[i].cpu()
                if label == pred:
                    if len(samples_by_class[label]['correct']) < num_samples:
                        samples_by_class[label]['correct'].append((sample, pred))
                else:
                    if len(samples_by_class[label]['incorrect']) < num_samples:
                        samples_by_class[label]['incorrect'].append((sample, pred))
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names)
    
    # Visualize samples
    for label in samples_by_class:
        class_name = class_names[label] if class_names else f"Class {label}"
        
        # Visualize correct samples
        if samples_by_class[label]['correct']:
            plt.figure(figsize=(15, 3))
            plt.suptitle(f"Correct Predictions for {class_name}")
            
            for i, (sample, _) in enumerate(samples_by_class[label]['correct']):
                plt.subplot(1, num_samples, i + 1)
                
                # Convert tensor to image
                if sample.shape[0] == 1:
                    # Grayscale
                    plt.imshow(sample.squeeze().numpy(), cmap='gray')
                else:
                    # RGB
                    img = sample.permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    plt.imshow(img)
                
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Visualize incorrect samples
        if samples_by_class[label]['incorrect']:
            plt.figure(figsize=(15, 3))
            plt.suptitle(f"Incorrect Predictions for {class_name}")
            
            for i, (sample, pred) in enumerate(samples_by_class[label]['incorrect']):
                plt.subplot(1, num_samples, i + 1)
                
                # Convert tensor to image
                if sample.shape[0] == 1:
                    # Grayscale
                    plt.imshow(sample.squeeze().numpy(), cmap='gray')
                else:
                    # RGB
                    img = sample.permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    plt.imshow(img)
                
                pred_name = class_names[pred] if class_names else f"Class {pred}"
                plt.title(f"Pred: {pred_name}")
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()


def export_model_to_onnx(model, input_shape, output_path, dynamic_axes=None):
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        input_shape: Input shape (batch_size, channels, height, width)
        output_path: Path to save ONNX model
        dynamic_axes: Dynamic axes for ONNX export
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Default dynamic axes
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify export
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid")
    except ImportError:
        print("ONNX not installed, skipping verification")
    except Exception as e:
        print(f"ONNX model verification failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model utility functions")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory containing model checkpoints [OPTIONAL]")
    parser.add_argument("--list-checkpoints", action="store_true", help="List model checkpoints [OPTIONAL]")
    parser.add_argument("--load-latest", action="store_true", help="Load latest checkpoint [OPTIONAL]")
    parser.add_argument("--plot-history", type=str, help="Path to history file to plot [OPTIONAL]")
    parser.add_argument("--export-onnx", type=str, help="Path to save ONNX model [OPTIONAL]")
    parser.add_argument("--input-shape", type=str, default="1,3,224,224", 
                      help="Input shape for ONNX export (batch_size,channels,height,width) [OPTIONAL, default=1,3,224,224]")
    
    args = parser.parse_args()
    
    if args.list_checkpoints:
        checkpoints = list_model_checkpoints(args.checkpoint_dir)
        print(f"Found {len(checkpoints)} checkpoints:")
        for checkpoint in checkpoints:
            print(f"  {checkpoint}")
    
    if args.load_latest:
        model, checkpoint = load_latest_checkpoint(args.checkpoint_dir)
        if model is not None:
            print("Model loaded successfully")
            if 'epoch' in checkpoint:
                print(f"Epoch: {checkpoint['epoch']}")
            if 'val_acc' in checkpoint:
                print(f"Validation Accuracy: {checkpoint['val_acc']:.4f}")
    
    if args.plot_history:
        plot_training_history(args.plot_history)
    
    if args.export_onnx:
        if not args.load_latest:
            print("Error: Must load a model with --load-latest to export to ONNX")
        else:
            # Parse input shape
            input_shape = tuple(map(int, args.input_shape.split(',')))
            export_model_to_onnx(model, input_shape, args.export_onnx)
