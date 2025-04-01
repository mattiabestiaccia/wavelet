#!/usr/bin/env python3
"""
Script for making predictions using the existing model structure.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import wavelet library modules
from wavelet_lib.models import create_scattering_transform

# Custom ScatteringClassifier class that matches the saved model architecture
class CustomScatteringClassifier(nn.Module):
    """Custom classifier that matches the saved model architecture."""
    
    def __init__(self, in_channels=243, num_classes=7):
        super(CustomScatteringClassifier, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Batch normalization for input
        self.bn = nn.BatchNorm2d(in_channels)
        
        # Feature extractor (matching the saved model architecture)
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Adaptive pool
            nn.AdaptiveAvgPool2d(2)
        )
        
        # Classifier
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_model(model_path, device=None):
    """
    Load the model with the exact architecture used during training.
    
    Args:
        model_path: Path to the model file
        device: Device to load the model on
        
    Returns:
        model: Loaded model
        scattering: Scattering transform
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model file
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create scattering transform
    scattering = create_scattering_transform(
        J=2,  # Based on model inspection
        shape=(32, 32),
        max_order=2,
        device=device
    )
    
    # Create the model with matching architecture
    model = CustomScatteringClassifier(
        in_channels=243,  # From model inspection
        num_classes=7  # From model inspection
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    
    # Set to evaluation mode
    model.eval()
    
    return model, scattering

def predict_image(model, scattering, image_path, device=None):
    """
    Make a prediction for a single image.
    
    Args:
        model: Loaded model
        scattering: Scattering transform
        image_path: Path to the image
        device: Device to use
        
    Returns:
        prediction: Class index
        confidence: Confidence score
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Compute scattering coefficients and make prediction
    with torch.no_grad():
        # Forward pass with scattering
        scattering_coeffs = scattering(image_tensor)
        
        # Match the format from the training code (reshape to expected format)
        scattering_coeffs = scattering_coeffs.view(image_tensor.size(0), -1, 8, 8)
        
        # Forward pass through the model
        outputs = model(scattering_coeffs)
        
        # Get prediction and confidence
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    return prediction.item(), confidence.item()

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict_custom.py <model_dir> <image_path>")
        return
    
    # Get arguments
    model_dir = sys.argv[1]
    image_path = sys.argv[2]
    
    # Get full paths
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(models_dir, model_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Define class names based on model directory
    if "4" in model_dir:
        class_names = ["vegetation 1", "vegetation 2", "vegetation 3", "water"]
    else:
        class_names = [f"class{i+1}" for i in range(7)]
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, scattering = load_model(model_path)
    
    # Make prediction
    print(f"Making prediction for {image_path}...")
    prediction, confidence = predict_image(model, scattering, image_path)
    
    # Map prediction index to class name
    # Note: The model was trained with 7 classes even though dataset had 4 classes
    # We'll need to map the prediction to the correct class
    if 0 <= prediction < len(class_names):
        class_name = class_names[prediction]
    else:
        class_name = f"Unknown class {prediction}"
    
    # Display results
    print(f"\nPrediction results:")
    print(f"Class index: {prediction}")
    print(f"Class name: {class_name}")
    print(f"Confidence: {confidence:.4f}")
    
    # Display the image with prediction
    image = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"Predicted: {class_name} (Confidence: {confidence:.4f})")
    plt.axis('off')
    
    # Save the visualization
    output_path = os.path.join(os.path.dirname(image_path), 
                              f"{os.path.splitext(os.path.basename(image_path))[0]}_custom_prediction.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()