#!/usr/bin/env python3
"""
Script for making predictions using a trained wavelet scattering model.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path to import wavelet_lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import wavelet library modules
from wavelet_lib.models import ScatteringClassifier, create_scattering_transform
from wavelet_lib.processors import ImageProcessor
from wavelet_lib.base import Config

def load_model(model_path, device=None):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        
    Returns:
        model: Loaded model
        scattering: Scattering transform
        class_names: List of class names
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model file
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get class mappings
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
    else:
        print("Warning: No class mapping found in model file.")
        class_to_idx = {}
    
    # Determine number of classes
    num_classes = len(class_to_idx) if class_to_idx else 4
    print(f"Number of classes detected: {num_classes}")
    
    # Create scattering transform
    config = Config(num_classes=num_classes)
    scattering = create_scattering_transform(
        J=config.J, 
        shape=config.shape, 
        max_order=config.scattering_order,
        device=device
    )
    
    # Create model
    model = ScatteringClassifier(
        in_channels=config.scattering_coeffs,
        classifier_type='cnn',
        num_classes=num_classes
    ).to(device)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("Error: No model state found in checkpoint file.")
        return None, None, []
    
    # Set model to evaluation mode
    model.eval()
    
    # Get class names
    class_names = ["Unknown"] * num_classes
    for name, idx in class_to_idx.items():
        class_names[idx] = name
    
    return model, scattering, class_names

def make_prediction(image_path, model, scattering, class_names, device=None):
    """
    Make a prediction on a single image.
    
    Args:
        image_path: Path to the image file
        model: Trained model
        scattering: Scattering transform
        class_names: List of class names
        device: Device to use for prediction
        
    Returns:
        pred_class: Predicted class name
        confidence: Confidence score
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create image processor
    processor = ImageProcessor(model, scattering, device, class_names)
    
    # Process the image
    result = processor.process_image(image_path)
    
    return result

def main():
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python make_prediction.py <model_dir> <image_path>")
        
        # List available models
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_dirs = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d)) and 
                     d.startswith('model_output')]
        
        print("\nAvailable model directories:")
        for model_dir in model_dirs:
            print(f"- {model_dir}")
        
        # List sample images
        datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
        if os.path.exists(datasets_dir):
            print("\nSample image paths to try (adjust according to your dataset structure):")
            for root, dirs, files in os.walk(datasets_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')) and 'sample' in root.lower():
                        print(f"- {os.path.join(root, file)}")
                        # Only show a few examples
                        if len(files) > 5:
                            print(f"- ... and {len(files)-5} more in {root}")
                            break
        return
    
    # Get arguments
    model_dir = sys.argv[1]
    image_path = sys.argv[2]
    
    # Build full model path
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(models_dir, model_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Load the model
    model, scattering, class_names = load_model(model_path)
    
    if model is None:
        return
    
    print(f"Loaded model with {len(class_names)} classes: {', '.join(class_names)}")
    
    # Make prediction
    result = make_prediction(image_path, model, scattering, class_names)
    
    # Display results
    print(f"\nPrediction results for {os.path.basename(image_path)}:")
    print(f"Class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Display the image with prediction
    img = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Predicted: {result['class_name']} (Confidence: {result['confidence']:.2f})")
    plt.axis('off')
    
    # Save the visualization
    output_path = os.path.join(os.path.dirname(image_path), 
                              f"{os.path.splitext(os.path.basename(image_path))[0]}_prediction.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()