"""
Base module for the Wavelet Scattering Transform Library.
Contains configurations and utility functions.
"""

import os
import torch
import numpy as np

class Config:
    """Configuration class for the Wavelet Scattering Transform Library."""
    
    def __init__(self, 
                 num_channels=3, 
                 num_classes=7, 
                 scattering_order=2,
                 J=2,
                 shape=(32, 32),
                 device=None,
                 batch_size=128,
                 epochs=90,
                 learning_rate=0.1,
                 momentum=0.9,
                 weight_decay=5e-4,
                 max_channels=10):
        """
        Initialize configuration parameters.
        
        Args:
            num_channels: Number of channels in input images
            num_classes: Number of classes for classification
            scattering_order: Order of scattering transform
            J: Number of scales
            shape: Shape of input images for scattering
            device: Device to use (cuda or cpu)
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            momentum: Momentum for optimization
            weight_decay: Weight decay for optimization
            max_channels: Maximum number of channels supported (default: 10)
        """
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.scattering_order = scattering_order
        self.J = J
        self.shape = shape
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_channels = max_channels
        
        # Validate num_channels
        if num_channels > max_channels:
            raise ValueError(f"Number of channels ({num_channels}) exceeds maximum supported channels ({max_channels})")
        
        # Computed fields
        self.scattering_coeffs = (1 + self.J + self.J * (self.J - 1) // 2) * self.num_channels
        
    def print_summary(self):
        """Print a summary of the configuration."""
        print("\n" + "="*80)
        print(" "*30 + "CONFIGURATION SUMMARY" + " "*30)
        print("="*80)
        
        print("\nGENERAL SETTINGS:")
        print(f"  • Device: {self.device}")
        print(f"  • Number of channels: {self.num_channels}")
        print(f"  • Maximum channels supported: {self.max_channels}")
        print(f"  • Number of classes: {self.num_classes}")
        
        print("\nSCATTERING SETTINGS:")
        print(f"  • Scattering order: {self.scattering_order}")
        print(f"  • J parameter: {self.J}")
        print(f"  • Input shape: {self.shape}")
        print(f"  • Scattering coefficients: {self.scattering_coeffs}")
        
        print("\nTRAINING SETTINGS:")
        print(f"  • Batch size: {self.batch_size}")
        print(f"  • Epochs: {self.epochs}")
        print(f"  • Learning rate: {self.learning_rate}")
        print(f"  • Momentum: {self.momentum}")
        print(f"  • Weight decay: {self.weight_decay}")
        
        print("\n" + "="*80)

def set_seed(seed=42):
    """
    Set seed for reproducibility.
    
    Args:
        seed: Seed value for random number generators
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, path, class_mapping):
    """
    Save model weights and class mapping.
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
        class_mapping: Dictionary mapping class names to indices
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_mapping
    }, path)
    print(f"Model saved to {path}")

def load_model(model, path, device=None):
    """
    Load model weights and class mapping.
    
    Args:
        model: PyTorch model to load weights into
        path: Path to saved model
        device: Device to load the model to
        
    Returns:
        class_mapping: Dictionary mapping class names to indices
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_to_idx = checkpoint.get('class_to_idx', None)
    
    model.to(device)
    model.eval()
    
    return class_to_idx