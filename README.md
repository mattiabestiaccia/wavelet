# Wavelet Scattering Transform Classification Library

A Python library for image classification using Wavelet Scattering Transforms (WST) features.

## Overview

This library provides tools for leveraging Wavelet Scattering Transforms for image classification tasks, including:

- Advanced image classification with wavelet scattering features
- Multi-class classification support with balanced dataset handling
- Comprehensive training and evaluation workflow
- Wavelet analysis tools for feature visualization and inspection
- Dataset management and processing utilities

## Features

### Image Classification

- Wavelet Scattering Transform feature extraction for robust classification
- Configurable neural network classifiers optimized for wavelet features
- Support for multi-class classification with class balancing
- Complete workflow for model training, evaluation, and prediction
- Performance metrics visualization and model analysis

### Wavelet Analysis Tools

- Discrete Wavelet Transform (DWT) feature extraction
- Wavelet Scattering Transform (WST) coefficient visualization
- Multi-channel and multi-band image support
- Statistical analysis of wavelet coefficients for feature importance

### Dataset Utilities

- Dataset inspection and validation for quality assurance
- Class balancing for handling imbalanced datasets
- Size and distribution analysis for dataset understanding
- Image preprocessing optimized for wavelet transforms

## Installation

```bash
# Clone the repository
git clone https://github.com/mattiabestiaccia/wavelet.git
cd wavelet

# Create a virtual environment and install requirements
python3 -m venv wavelet_venv
source wavelet_venv/bin/activate
pip install -r requirements.txt

# Install the library in development mode
pip install -e .
```

## Quick Start

### Classification

```python
import torch
from wavelet_lib.base import Config
from wavelet_lib.classification import create_classification_model, ClassificationProcessor

# Create model configuration
config = Config(
    num_classes=4,             # Number of classification classes
    num_channels=3,            # RGB input images
    scattering_order=2,        # Maximum scattering order
    J=2,                       # Wavelet scale parameter
    shape=(32, 32),            # Input image size
    batch_size=128,            # Training batch size
    learning_rate=0.1,         # Initial learning rate
    weight_decay=5e-4          # Regularization strength
)

# Create model and scattering transform
model, scattering = create_classification_model(config)

# Load pretrained weights
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create processor for inference
processor = ClassificationProcessor(model, scattering, device=config.device)

# Process an image and get classification results
result = processor.process_image('image.jpg')
print(f"Prediction: {result['class_name']} with {result['confidence']:.2f} confidence")

# Get classification with detailed feature information
detailed_result = processor.process_image('image.jpg', return_features=True)
print(f"WST Features shape: {detailed_result['features'].shape}")
```

## Command Line Usage

### Classification

```bash
# Train a classification model
python script/core/classification/train_classification.py --dataset /path/to/dataset --num-classes 4 --epochs 90

# Evaluate a model
python script/core/classification/evaluate_classification.py --model /path/to/model.pth --dataset /path/to/test_dataset

# Make predictions
python script/core/classification/predict_classification.py --model /path/to/model.pth --image /path/to/image.jpg
```

## Experiment Structure

Each classification experiment is organized as follows:

```
experiments/dataset_name/
├── classification_result/    # Classification results and confusion matrices
├── dataset_info/             # Dataset statistics and class distributions
│   ├── dataset_report.txt    # Detailed dataset analysis
│   └── dataset_stats.png     # Visualizations of dataset properties
├── evaluation/               # Evaluation metrics and performance analysis
├── models/                   # Model checkpoints and saved weights
│   ├── class_distribution.png  # Class distribution visualization
│   └── training_metrics.png    # Accuracy and loss curves
├── model_output/             # Training logs and intermediate outputs
└── visualization/            # Feature visualizations and model interpretability
```

## Advanced Usage

For more advanced usage and detailed instructions, see the [Usage Guide](USAGE.md).

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Kymatio 0.3.0+
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib
- Albumentations (for data augmentation)

## Credits

- Wavelet Scattering Transform: [Kymatio](https://github.com/kymatio/kymatio)
- The segmentation components are inspired by the U-Net architecture