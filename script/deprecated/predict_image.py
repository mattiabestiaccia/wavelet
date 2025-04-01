#!/usr/bin/env python3
"""
Script for making predictions on images using a trained model.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary modules
from wavelet_lib.processors import ImageProcessor
from wavelet_lib.models import ScatteringClassifier, create_scattering_transform
from wavelet_lib.base import Config

def process_image(image_path, model_path, output_path=None, device=None):
    """
    Process an image with a trained model and show/save the results.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the model directory
        output_path: Path to save the output visualization
        device: Device to use for processing
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure paths exist
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found: {model_path}")
        return
    
    # Load the image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image: {os.path.basename(image_path)} - Size: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Set up image display
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Try to use the model with direct file loading
    best_model_path = os.path.join(model_path, 'best_model.pth')
    if not os.path.exists(best_model_path):
        print(f"Error: Best model file not found at {best_model_path}")
        return
    
    # Create a temporary inference setup
    try:
        # Load model checkpoint
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # Get class mapping
        if 'class_to_idx' in checkpoint:
            class_to_idx = checkpoint['class_to_idx']
            class_names = list(class_to_idx.keys())
        else:
            print("Warning: No class mapping found. Using generic class names.")
            # Determine which model we're using based on the directory name
            model_name = os.path.basename(model_path)
            if "4" in model_name:
                class_names = ["vegetation 1", "vegetation 2", "vegetation 3", "water"]
            elif "7" in model_name:
                class_names = ["class1", "class2", "class3", "class4", "class5", "class6", "class7"]
            else:
                class_names = [f"class{i}" for i in range(4)]  # Default to 4 classes
        
        num_classes = len(class_names)
        print(f"Using model with {num_classes} classes: {', '.join(class_names)}")
        
        # Create a clean configuration and model for inference
        config = Config(num_classes=num_classes)
        
        # Create the scattering transform
        scattering = create_scattering_transform(
            J=config.J,
            shape=config.shape,
            max_order=config.scattering_order,
            device=device
        )
        
        # Create a model with the same architecture
        model = ScatteringClassifier(
            in_channels=config.scattering_coeffs,
            classifier_type='cnn',
            num_classes=num_classes
        ).to(device)
        
        # Try loading model weights, handling different key formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            print("Error: Could not find model weights in checkpoint")
            return
        
        model.eval()
        
        # Create image processor
        processor = ImageProcessor(model, scattering, device, class_names)
        
        # Process the image for classification
        result = processor.process_image(image_path, confidence_threshold=0.0)
        
        # Display the result
        print(f"\nImage classification result:")
        print(f"Predicted class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        # Create a visualization
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title(f"Prediction: {result['class_name']}\nConfidence: {result['confidence']:.2f}")
        plt.axis('off')
        
        # Save the visualization if requested
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process an image with a trained model')
    parser.add_argument('model_dir', help='Name of the model directory')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--output', '-o', help='Path to save the output visualization')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get the full model path
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(models_dir, args.model_dir)
    
    # Generate output path if not provided
    output_path = args.output
    if not output_path:
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_path = os.path.join(os.path.dirname(args.image_path), f"{base_name}_prediction.png")
    
    # Process the image
    process_image(args.image_path, model_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments, print usage
        print("Usage: python predict_image.py <model_dir> <image_path> [--output OUTPUT_PATH]")
        
        # List available models
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_dirs = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d)) and 
                     d.startswith('model_output')]
        
        print("\nAvailable model directories:")
        for model_dir in model_dirs:
            print(f"- {model_dir}")
        
        # List available images
        sample_dirs = ['/home/brus/Projects/wavelet/datasets/HPL_images/custom_dataset/water',
                      '/home/brus/Projects/wavelet/datasets/HPL_images/custom_dataset/vegetation 1']
        print("\nSample images to try:")
        for sample_dir in sample_dirs:
            if os.path.exists(sample_dir):
                files = os.listdir(sample_dir)[:3]  # List first 3 files
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        print(f"- {os.path.join(sample_dir, file)}")
    else:
        main()