# Wavelet Scattering Transform Library

A Python library for image analysis, classification, and segmentation using Wavelet Scattering Transforms (WST).

## Overview

This library provides tools for working with Wavelet Scattering Transforms for image processing tasks, including:

- Image classification with wavelet scattering features
- Image segmentation using WST-UNet architecture 
- Wavelet analysis tools for image inspection
- Dataset management and processing utilities
- Visualization tools for wavelet coefficients

## Features

### Image Classification

- Wavelet Scattering Transform feature extraction
- Configurable neural network classifiers
- Support for multi-class classification
- Tools for model training, evaluation, and prediction

### Image Segmentation

- WST-UNet architecture for semantic segmentation
- Binary and multi-class segmentation support
- Training pipeline with data augmentation
- Evaluation and visualization utilities

### Wavelet Analysis Tools

- Discrete Wavelet Transform (DWT) analysis
- Wavelet Scattering Transform (WST) visualization
- Multi-channel and multi-band image support
- Statistical analysis of wavelet coefficients

### Dataset Utilities

- Dataset inspection and validation
- Class balancing and augmentation
- Size and distribution analysis
- Image preprocessing for wavelet transforms

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

# Create model
config = Config(num_classes=4, J=2)
model, scattering = create_classification_model(config)

# Load weights
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Process an image
processor = ClassificationProcessor(model, scattering, device=config.device)
result = processor.process_image('image.jpg')

print(f"Prediction: {result['class_name']} with {result['confidence']:.2f} confidence")
```

### Segmentation

```python
from wavelet_lib.segmentation import ScatteringSegmenter

# Initialize segmenter with a trained model
segmenter = ScatteringSegmenter(
    model_path='segmentation_model.pth',
    J=2, 
    input_shape=(256, 256)
)

# Segment an image
mask = segmenter.predict('image.jpg', threshold=0.5)
```

## Command Line Usage

### Classification

```bash
# Train a classification model
python script/core/train.py --dataset /path/to/dataset --model-save model.pth

# Evaluate a model
python script/core/evaluate.py --model model.pth --dataset /path/to/test_dataset

# Make predictions
python script/core/predict.py --model model.pth --image image.jpg
```

### Segmentation

```bash
# Train a segmentation model
python script/core/train_segmentation.py \
  --train-imgs /path/to/train/images \
  --train-masks /path/to/train/masks \
  --model model.pth

# Run segmentation
python script/core/segment.py --model model.pth --image image.jpg --output results
```

## Experiment Structure

Each experiment is organized as follows:

```
experiments/dataset0/
├── classification_result/    # Classification results
├── dataset_info/             # Dataset statistics
├── evaluation/               # Evaluation metrics
├── models/                   # Model checkpoints
├── model_output/             # Training output
├── visualization/            # Visualizations
└── README.md                 # Experiment documentation
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