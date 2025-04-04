"""
Classification models module for the Wavelet Scattering Transform Library.
Contains neural network models for classification with scattering transforms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D

class ScatteringClassifier(nn.Module):
    """Neural network model for classification with scattering transform."""
    
    def __init__(self, in_channels, classifier_type='cnn', num_classes=4):
        """
        Initialize the scattering classifier.
        
        Args:
            in_channels: Number of input channels (scattering coefficients)
            classifier_type: Type of classifier ('cnn', 'mlp', or 'linear')
            num_classes: Number of output classes
        """
        super(ScatteringClassifier, self).__init__()
        self.in_channels = in_channels
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.build()

    def build(self):
        """Build the classifier architecture based on the classifier type."""
        self.K = self.in_channels
        self.bn = nn.BatchNorm2d(self.K)
        
        if self.classifier_type == 'cnn':
            # CNN classifier with deep convolutional layers
            cfg = [256, 256, 256, 'M', 512, 512, 512, 1024, 1024]
            layers = []
            current_in_channels = self.K
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [
                        nn.Conv2d(current_in_channels, v, kernel_size=3, padding=1),
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ]
                    current_in_channels = v
            layers += [nn.AdaptiveAvgPool2d(2)]
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(1024 * 4, self.num_classes)
        elif self.classifier_type == 'mlp':
            # MLP classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.K * 8 * 8, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.num_classes))
            self.features = None
        elif self.classifier_type == 'linear':
            # Linear classifier
            self.classifier = nn.Linear(self.K * 8 * 8, self.num_classes)
            self.features = None

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, 8, 8)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Get batch size and reshape directly using the expected input channels (self.K)
        batch_size = x.size(0)
        total_elements = x.numel()
        
        # If input is already the right shape, use it directly
        if x.shape[1] == self.K and x.shape[2] == 8 and x.shape[3] == 8:
            pass
        # If the input is from a scattering transform and needs reshaping
        elif total_elements // batch_size // 64 == self.K:
            # Already the correct number of elements
            x = x.view(batch_size, self.K, 8, 8)
        else:
            # Handle case where scattering output has a different number of channels
            # This is specifically for the case when the scattering output is shape [1, 3, 81, 8, 8]
            # and needs to be flattened to match the channels
            flattened = x.reshape(batch_size, -1)
            if flattened.shape[1] % 64 == 0:  # Can be reshaped to [batch, channels, 8, 8]
                # Get first self.K * 64 elements and reshape
                x = flattened[:, :self.K * 64].view(batch_size, self.K, 8, 8)
            else:
                raise ValueError(f"Input shape {x.shape} with {flattened.shape[1]} elements can't be reshaped to [batch, {self.K}, 8, 8]")
                
        x = self.bn(x)
        if self.features:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def create_scattering_transform(J=2, shape=(32, 32), max_order=2, device=None):
    """
    Create a scattering transform.
    
    Args:
        J: Number of scales
        shape: Shape of input images
        max_order: Maximum order of scattering
        device: Device to create the scattering transform on
        
    Returns:
        Scattering2D object
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    scattering = Scattering2D(
        J=J,
        shape=shape,
        max_order=max_order
    ).to(device)
    
    return scattering

def create_classification_model(config):
    """
    Create a scattering classifier model.
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        ScatteringClassifier model and Scattering2D transform
    """
    # Create scattering transform
    scattering = create_scattering_transform(
        J=config.J,
        shape=config.shape,
        max_order=config.scattering_order,
        device=config.device
    )
    
    # Create classifier model
    model = ScatteringClassifier(
        in_channels=config.scattering_coeffs,
        classifier_type='cnn',
        num_classes=config.num_classes
    ).to(config.device)
    
    return model, scattering

def print_classifier_summary(model, scattering, device, input_shape=(1, 3, 32, 32)):
    """
    Print a summary of the classification model.
    
    Args:
        model: ScatteringClassifier model
        scattering: Scattering2D transform
        device: Device to use
        input_shape: Shape of input images
    """
    print("\n" + "="*80)
    print(" "*30 + "CLASSIFICATION MODEL SUMMARY" + " "*30)
    print("="*80)
    
    # Create dummy input
    dummy_input = torch.zeros(input_shape).to(device)
    
    # Get scattering output shape
    with torch.no_grad():
        scat_output = scattering(dummy_input)
        scat_shape = scat_output.shape
    
    print("\nSCATTERING TRANSFORM:")
    print(f"  • Input shape: {input_shape}")
    print(f"  • J parameter: {scattering.J}")
    print(f"  • Max order: {scattering.max_order}")
    print(f"  • Output shape: {scat_shape}")
    
    print("\nMODEL ARCHITECTURE:")
    print(f"  • Type: {model.classifier_type}")
    print(f"  • Input channels: {model.in_channels}")
    print(f"  • Number of classes: {model.num_classes}")
    print(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nLAYERS:")
    if model.features:
        print("  • Batch Normalization")
        print("  • Feature extractor (CNN layers)")
        for idx, layer in enumerate(model.features):
            print(f"    - {idx}: {layer}")
    else:
        print("  • Batch Normalization")
    
    print("  • Classifier:")
    if isinstance(model.classifier, nn.Linear):
        print(f"    - Linear: {model.classifier}")
    else:
        for idx, layer in enumerate(model.classifier):
            print(f"    - {idx}: {layer}")
    
    print("\n" + "="*80)