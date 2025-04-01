#!/usr/bin/env python3
"""
Utility functions for model management and analysis.

This module provides helper functions for working with models,
including loading, saving, analyzing, and converting models.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix

# Add the main directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return []
    
    # Find all model checkpoint files
    checkpoints = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth'):
                checkpoint_path = os.path.join(root, file)
                checkpoints.append(checkpoint_path)
    
    # Print summary
    print(f"\nFound {len(checkpoints)} model checkpoints:")
    for i, checkpoint in enumerate(checkpoints):
        rel_path = os.path.relpath(checkpoint, directory)
        print(f"{i+1}. {rel_path}")
    
    return checkpoints

def analyze_model(model_path, device=None):
    """
    Analyze a model and print a summary of its architecture and parameters.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load the model on
        
    Returns:
        dict: Model analysis results
    """
    # Check if path exists
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found: {model_path}")
        return None
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model information
    model_info = {}
    
    # Get class information
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        model_info['class_names'] = list(class_to_idx.keys())
        model_info['num_classes'] = len(model_info['class_names'])
    else:
        print("Warning: class_to_idx not found in checkpoint.")
        model_info['num_classes'] = 4  # Default fallback
        model_info['class_names'] = [f"Class {i}" for i in range(model_info['num_classes'])]
    
    # Create scattering transform
    scattering = create_scattering_transform(
        J=2,
        shape=(32, 32),
        max_order=2,
        device=device
    )
    
    # Determine input channels
    in_channels = 243  # Default for standard architecture
    model_info['in_channels'] = in_channels
    
    # Create model
    model = ScatteringClassifier(
        in_channels=in_channels,
        num_classes=model_info['num_classes']
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    
    # Analyze parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info['total_params'] = total_params
    model_info['trainable_params'] = trainable_params
    
    # Check if training metrics are available
    if 'training_metrics' in checkpoint:
        metrics = checkpoint['training_metrics']
        model_info['epochs'] = metrics.get('epochs', 0)
        model_info['best_accuracy'] = metrics.get('best_accuracy', 0)
        model_info['best_epoch'] = metrics.get('best_epoch', 0)
        model_info['final_train_loss'] = metrics.get('train_losses', [0])[-1]
        model_info['final_test_loss'] = metrics.get('test_losses', [0])[-1]
    
    # Print summary
    print(f"\nModel Analysis: {os.path.basename(model_path)}")
    print(f"Classes: {model_info['num_classes']} - {', '.join(model_info['class_names'])}")
    print(f"Input channels: {model_info['in_channels']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if 'epochs' in model_info:
        print(f"\nTraining Information:")
        print(f"Epochs: {model_info['epochs']}")
        print(f"Best accuracy: {model_info['best_accuracy']:.2f}%")
        print(f"Best epoch: {model_info['best_epoch']}")
        print(f"Final train loss: {model_info['final_train_loss']:.4f}")
        print(f"Final test loss: {model_info['final_test_loss']:.4f}")
    
    return model_info

def compare_models(model_paths, device=None):
    """
    Compare multiple models and print a summary of their performance.
    
    Args:
        model_paths: List of model checkpoint paths
        device: Device to load the models on
        
    Returns:
        dict: Comparison results
    """
    # Check inputs
    if not model_paths:
        print("Error: No model paths provided")
        return None
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Analyze each model
    model_analyses = []
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: Model checkpoint not found: {path}")
            continue
        
        print(f"\nAnalyzing model: {os.path.basename(path)}")
        analysis = analyze_model(path, device)
        if analysis:
            analysis['path'] = path
            model_analyses.append(analysis)
    
    # Create comparison
    if not model_analyses:
        print("No models analyzed.")
        return None
    
    # Print comparison table
    print("\nModel Comparison:")
    print("-" * 100)
    print(f"{'Model':<30} {'Classes':<10} {'Parameters':<15} {'Best Accuracy':<15} {'Final Test Loss':<15}")
    print("-" * 100)
    
    for analysis in model_analyses:
        model_name = os.path.basename(analysis['path'])
        num_classes = analysis['num_classes']
        params = f"{analysis['total_params']:,}"
        best_acc = f"{analysis.get('best_accuracy', 'N/A')}"
        test_loss = f"{analysis.get('final_test_loss', 'N/A')}"
        
        print(f"{model_name:<30} {num_classes:<10} {params:<15} {best_acc:<15} {test_loss:<15}")
    
    print("-" * 100)
    
    return {
        'models': model_analyses
    }

def convert_model_format(source_path, target_path, format_type='standalone'):
    """
    Convert a model checkpoint to a different format.
    
    Args:
        source_path: Path to source model checkpoint
        target_path: Path to save the converted model
        format_type: Target format type ('standalone' or 'full')
        
    Returns:
        dict: Conversion results
    """
    # Check if source exists
    if not os.path.exists(source_path):
        print(f"Error: Source model not found: {source_path}")
        return None
    
    # Load source checkpoint
    checkpoint = torch.load(source_path, map_location='cpu')
    
    # Extract model information
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        num_classes = len(class_to_idx)
    else:
        print("Warning: class_to_idx not found in checkpoint.")
        num_classes = 4  # Default fallback
        class_to_idx = {f"Class {i}": i for i in range(num_classes)}
    
    # Create model
    model = ScatteringClassifier(
        in_channels=243,  # Default
        num_classes=num_classes
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("Error: Could not find model weights in checkpoint")
        return None
    
    # Create new checkpoint
    if format_type == 'standalone':
        # Minimal checkpoint with just the model and class mapping
        new_checkpoint = {
            'model_state_dict': model.state_dict(),
            'class_to_idx': class_to_idx
        }
    else:  # 'full'
        # Full checkpoint with all information
        new_checkpoint = checkpoint
    
    # Save converted model
    torch.save(new_checkpoint, target_path)
    print(f"Model converted and saved to: {target_path}")
    
    return {
        'source': source_path,
        'target': target_path,
        'format': format_type,
        'num_classes': num_classes,
        'class_to_idx': class_to_idx
    }

def predict_batch(model_path, image_paths, device=None):
    """
    Make predictions on a batch of images.
    
    Args:
        model_path: Path to model checkpoint
        image_paths: List of image paths
        device: Device to load the model on
        
    Returns:
        dict: Prediction results
    """
    # Check inputs
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found: {model_path}")
        return None
    
    if not image_paths:
        print("Error: No image paths provided")
        return None
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get class information
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        class_names = list(class_to_idx.keys())
        num_classes = len(class_names)
    else:
        print("Warning: class_to_idx not found in checkpoint.")
        num_classes = 4  # Default fallback
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Create scattering transform
    scattering = create_scattering_transform(
        J=2,
        shape=(32, 32),
        max_order=2,
        device=device
    )
    
    # Create model
    model = ScatteringClassifier(
        in_channels=243,  # Default
        num_classes=num_classes
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("Error: Could not find model weights in checkpoint")
        return None
    
    model.eval()
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Make predictions
    results = []
    
    with torch.no_grad():
        for path in image_paths:
            if not os.path.exists(path):
                print(f"Warning: Image not found: {path}")
                results.append({
                    'path': path,
                    'error': 'File not found'
                })
                continue
            
            try:
                # Load and process image
                img = Image.open(path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                
                # Apply scattering transform
                scattering_coeffs = scattering(tensor)
                scattering_coeffs = scattering_coeffs.view(tensor.size(0), -1, 8, 8)
                
                # Get prediction
                output = model(scattering_coeffs)
                probabilities = torch.softmax(output, dim=1)
                max_prob, prediction = torch.max(probabilities, dim=1)
                
                # Get top-3 predictions
                top3_probs, top3_indices = torch.topk(probabilities, k=min(3, num_classes))
                top3_classes = [class_names[idx] for idx in top3_indices[0].cpu().numpy()]
                top3_probs = top3_probs[0].cpu().numpy()
                
                # Add result
                results.append({
                    'path': path,
                    'class_name': class_names[prediction.item()],
                    'class_index': prediction.item(),
                    'confidence': max_prob.item(),
                    'top3_classes': top3_classes,
                    'top3_confidences': top3_probs.tolist()
                })
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append({
                    'path': path,
                    'error': str(e)
                })
    
    # Print summary
    print(f"\nBatch Prediction Summary:")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Images processed: {len(results)}")
    
    # Count predictions by class
    class_counts = {}
    for result in results:
        if 'class_name' in result:
            class_name = result['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\nPrediction distribution:")
    for class_name, count in class_counts.items():
        percentage = 100 * count / len(results)
        print(f"  - {class_name}: {count} images ({percentage:.1f}%)")
    
    return {
        'model_path': model_path,
        'class_names': class_names,
        'predictions': results,
        'class_counts': class_counts
    }