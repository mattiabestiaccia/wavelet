# Wavelet Scattering Transform Library Usage Guide

This document provides instructions for using the Wavelet Scattering Transform (WST) Library for image analysis and classification tasks.

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Library Structure

The library is organized into the following modules:

- **base.py**: Configuration and utility functions
- **datasets.py**: Dataset loading and processing
- **training.py**: Training functions
- **visualization.py**: Visualization utilities
- **classification/**: Classification module
  - **models.py**: Neural network models for classification
  - **processors.py**: Image classification utilities
- **segmentation/**: Segmentation module
  - **models.py**: Image segmentation with WST-UNet

Utility modules:
- **dataset_tools/**: Tools for dataset analysis and manipulation
- **image_tools/**: Image analysis and wavelet transformations
- **utils/**: General utility functions

## Basic Usage

### Classification

#### 1. Training a Classification Model

```bash
python script/core/train.py \
  --dataset /path/to/dataset \
  --model-save /path/to/save/model.pth \
  --epochs 50 \
  --batch-size 64
```

#### 2. Evaluating a Classification Model

```bash
python script/core/evaluate.py \
  --model /path/to/model.pth \
  --dataset /path/to/test_dataset
```

#### 3. Making Predictions

```bash
python script/core/predict.py \
  --model /path/to/model.pth \
  --image /path/to/image.jpg
```

or for batch processing:

```bash
python script/core/predict.py \
  --model /path/to/model.pth \
  --folder /path/to/images \
  --output /path/to/output
```

### Segmentation

#### 1. Training a Segmentation Model

```bash
python script/core/train_segmentation.py \
  --train-imgs /path/to/train/images \
  --train-masks /path/to/train/masks \
  --model /path/to/save/model.pth \
  --epochs 50
```

With validation:

```bash
python script/core/train_segmentation.py \
  --train-imgs /path/to/train/images \
  --train-masks /path/to/train/masks \
  --val-imgs /path/to/val/images \
  --val-masks /path/to/val/masks \
  --model /path/to/save/model.pth
```

#### 2. Running Segmentation

```bash
python script/core/segment.py \
  --image /path/to/image.jpg \
  --model /path/to/model.pth \
  --output /path/to/output
```

or for batch processing:

```bash
python script/core/segment.py \
  --folder /path/to/images \
  --model /path/to/model.pth \
  --output /path/to/output \
  --overlay
```

## Using the Library in Python

### Classification

```python
import torch
from wavelet_lib.base import Config
from wavelet_lib.classification import create_classification_model, ClassificationProcessor

# Initialize configuration
config = Config(num_classes=4, J=2)

# Create model
model, scattering = create_classification_model(config)

# Load weights
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create processor
processor = ClassificationProcessor(model, scattering, device=config.device)

# Process an image
result = processor.process_image('image.jpg')
print(f"Prediction: {result['class_name']} with {result['confidence']:.2f} confidence")
```

### Segmentation

```python
import torch
import matplotlib.pyplot as plt
from wavelet_lib.segmentation import ScatteringSegmenter

# Initialize segmenter
segmenter = ScatteringSegmenter(
    model_path='segmentation_model.pth',
    J=2,
    input_shape=(256, 256)
)

# Segment an image
mask, probabilities = segmenter.predict('image.jpg', threshold=0.5, return_raw=True)

# Visualize results
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(plt.imread('image.jpg'))
plt.title('Original Image')

plt.subplot(132)
plt.imshow(mask, cmap='gray')
plt.title('Segmentation Mask')

plt.subplot(133)
plt.imshow(probabilities, cmap='jet')
plt.title('Probability Heatmap')
plt.colorbar()

plt.tight_layout()
plt.show()
```

### Wavelet Analysis

```python
from wavelet_lib.image_tools import WaveletAnalyzer

# Initialize analyzer
analyzer = WaveletAnalyzer(max_level=3, wavelet='db4', J=2, L=8)

# Load and analyze image
img = analyzer.load_image('image.jpg')

# Discrete wavelet transform analysis
dwt_results = analyzer.analyze_dwt(img, plot=True)

# Wavelet scattering transform analysis
wst_results = analyzer.analyze_wst(img, plot=True)

# Visualize WST coefficients in disk format
analyzer.visualize_wst_disk(img)
```

## Advanced Features

### Dataset Analysis

```python
from wavelet_lib.dataset_tools import analyze_dataset, plot_size_distribution

# Analyze a dataset
stats = analyze_dataset('/path/to/dataset')

# Plot size distribution
plot_size_distribution('/path/to/dataset', save_path='size_dist.png')
```

### WST Dataset Analysis

```python
from wavelet_lib.image_tools import WSTDatasetAnalyzer

# Initialize analyzer
analyzer = WSTDatasetAnalyzer('processed_dataset.pkl', 'analysis_output')

# Run analysis
analyzer.analyze()

# Generate visualizations
analyzer.plot_analysis()
```

### Synthetic Test Images

```python
from wavelet_lib.image_tools import PhantomGenerator

# Create generator
generator = PhantomGenerator('phantoms_output')

# Generate test images
phantoms = generator.generate_all_phantoms(base_size=512)
```

## Command Line Tools

The library provides command-line tools for common tasks:

### Dataset Inspection

```bash
python -m wavelet_lib.dataset_tools.dataset_inspector --dataset /path/to/dataset
```

### Channel Visualization

```bash
python -m wavelet_lib.image_tools.visualize_channels /path/to/multiband_image.tif --delay 0.5
```

### Create Multiband TIFF

```bash
python -m wavelet_lib.utils.merge_bands /path/to/bands /path/to/output
```

## Examples

See the `experiments` directory for example usage in different scenarios.