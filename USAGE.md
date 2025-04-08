# Wavelet Scattering Transform Classification Library Usage Guide

This document provides detailed instructions for using the Wavelet Scattering Transform (WST) Library for image classification tasks using wavelet features.

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Classification Library Structure

The library is organized into the following modules:

- **base.py**: Configuration and utility functions for classification workflows
- **datasets.py**: Dataset loading and processing with class balancing support
- **training.py**: Training functions for classification models
- **visualization.py**: Visualization utilities for classification results and metrics
- **classification/**: Classification module
  - **models.py**: Neural network models optimized for wavelet feature classification
  - **processors.py**: Image classification utilities and inference pipelines

Supporting modules:
- **dataset_tools/**: Tools for dataset analysis and manipulation
  - **data_utils.py**: Utilities for data preprocessing and normalization
  - **dataset_inspector.py**: Tools for analyzing dataset properties
- **image_tools/**: Wavelet analysis tools for feature extraction
  - **wavelet_analyzer.py**: Tools for wavelet transform analysis
  - **wst_dataset_analyzer.py**: Dataset-level wavelet feature analysis
- **utils/**: General utility functions
  - **model_utils.py**: Model management and configuration utilities

## Classification Model Usage

### 1. Training a Classification Model

```bash
python script/core/classification/train_classification.py \
  --dataset /path/to/dataset \
  --num-classes 4 \
  --epochs 90 \
  --batch-size 128 \
  --balance \
  --j 2 \
  --scattering-order 2 \
  --output-dir /path/to/save/results
```

Key parameters:
- `--dataset`: Path to the image dataset organized in class folders
- `--num-classes`: Number of classification classes
- `--balance`: Enable class balancing for imbalanced datasets
- `--j`: Wavelet scale parameter (controls feature granularity)
- `--scattering-order`: Maximum order of wavelet scattering coefficients

### 2. Evaluating a Classification Model

```bash
python script/core/classification/evaluate_classification.py \
  --model /path/to/model.pth \
  --dataset /path/to/test_dataset \
  --output /path/to/evaluation/results
```

This will generate:
- Confusion matrix
- Classification report with precision, recall, and F1-score
- Performance visualization across classes

### 3. Making Predictions

For single image classification:

```bash
python script/core/classification/predict_classification.py \
  --model /path/to/model.pth \
  --image /path/to/image.jpg \
  --output /path/to/result
```

For batch processing multiple images:

```bash
python script/core/classification/predict_classification.py \
  --model /path/to/model.pth \
  --folder /path/to/images \
  --output /path/to/output \
  --save-features  # Optional: save extracted wavelet features
```


## Python API for Classification

### Basic Image Classification

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from wavelet_lib.base import Config
from wavelet_lib.classification import create_classification_model, ClassificationProcessor

# Initialize configuration with wavelet parameters
config = Config(
    num_classes=4,             # Number of classes
    num_channels=3,            # RGB images
    scattering_order=2,        # Maximum scattering order
    J=2,                       # Wavelet scale parameter
    shape=(32, 32)             # Input image size
)

# Create model and scattering transform
model, scattering = create_classification_model(config)

# Load pretrained weights
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create processor for inference
processor = ClassificationProcessor(
    model=model,
    scattering=scattering,
    device=config.device,
    class_names=['class1', 'class2', 'class3', 'class4']  # Optional class names
)

# Process an image and get classification
result = processor.process_image('image.jpg')
print(f"Prediction: {result['class_name']} (Class {result['class_id']})")
print(f"Confidence: {result['confidence']:.2f}")

# Get detailed classification results with feature extraction
detailed_result = processor.process_image('image.jpg', return_features=True)

# Access wavelet scattering features
wst_features = detailed_result['features']
print(f"Feature shape: {wst_features.shape}")

# Visualize class probabilities
plt.figure(figsize=(8, 4))
plt.bar(range(config.num_classes), detailed_result['probabilities'])
plt.xticks(range(config.num_classes), processor.class_names)
plt.ylabel('Probability')
plt.title('Classification Probability Distribution')
plt.tight_layout()
plt.show()
```

### Working with Wavelet Features

```python
import numpy as np
import matplotlib.pyplot as plt
from wavelet_lib.image_tools import WaveletAnalyzer
from wavelet_lib.classification import create_classification_model, ClassificationProcessor

# Create wavelet analyzer
analyzer = WaveletAnalyzer(max_level=3, wavelet='db4', J=2, L=8)

# Load and analyze image
img = analyzer.load_image('image.jpg')

# Get wavelet scattering features
wst_coeffs = analyzer.analyze_wst(img, plot=False)

# Visualize wavelet coefficient distribution
plt.figure(figsize=(10, 6))
plt.hist(wst_coeffs.flatten(), bins=50)
plt.title('Wavelet Scattering Coefficient Distribution')
plt.xlabel('Coefficient Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Visualize coefficients as feature map
plt.figure(figsize=(12, 10))
analyzer.visualize_wst_disk(img, 
                           title='Wavelet Scattering Transform Coefficients',
                           save_path='wst_visualization.png')
```

### Training and Evaluating Classification Models

```python
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from wavelet_lib.base import Config, save_model
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders
from wavelet_lib.classification import create_classification_model
from wavelet_lib.training import Trainer, create_optimizer
from wavelet_lib.visualization import plot_training_metrics, plot_class_distribution

# Create dataset with class balancing
dataset_path = '/path/to/dataset'
transform = get_default_transform(target_size=(32, 32))
dataset = BalancedDataset(dataset_path, transform=transform, balance=True)

# Visualize class distribution
plot_class_distribution(dataset, 
                        title="Class distribution in dataset",
                        save_path="class_distribution.png")

# Create dataloaders with train/test split
train_loader, test_loader = create_data_loaders(dataset,
                                               test_size=0.2,
                                               batch_size=128,
                                               num_workers=4)

# Create model configuration
config = Config(
    num_classes=len(dataset.classes),
    num_channels=3,
    scattering_order=2,
    J=2,
    shape=(32, 32),
    batch_size=128,
    epochs=90,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

# Create model and optimizer
model, scattering = create_classification_model(config)
optimizer = create_optimizer(model, config)

# Create trainer
trainer = Trainer(model, scattering, config.device, optimizer)

# Train the model
results = trainer.train(train_loader, 
                       test_loader,
                       config.epochs,
                       save_path='model.pth',
                       reduce_lr_after=20,
                       class_to_idx=dataset.class_to_idx)

# Plot training metrics
plot_training_metrics(config.epochs,
                     results['train_accuracies'],
                     results['test_accuracies'],
                     results['train_losses'],
                     results['test_losses'],
                     'training_metrics.png')

print(f"Best accuracy: {results['best_accuracy']:.2f}%")
```

## Advanced Classification Features

### Dataset Analysis and Preparation

```python
from wavelet_lib.dataset_tools import dataset_inspector
from wavelet_lib.dataset_tools.data_utils import analyze_dataset, plot_size_distribution

# Analyze a classification dataset
stats = analyze_dataset('/path/to/dataset')
print(f"Total images: {stats['total_images']}")
print(f"Classes: {stats['classes']}")
print(f"Class distribution: {stats['class_distribution']}")

# Plot size distribution of images in dataset
plot_size_distribution('/path/to/dataset', save_path='size_distribution.png')

# Run comprehensive dataset inspection
inspector = dataset_inspector.DatasetInspector('/path/to/dataset')
report = inspector.generate_report(save_path='dataset_report.txt')
inspector.plot_statistics(save_dir='dataset_stats')
```

### Wavelet Feature Analysis for Classification

```python
from wavelet_lib.image_tools import WSTDatasetAnalyzer

# Analyze wavelet features across dataset classes
analyzer = WSTDatasetAnalyzer(
    dataset_path='/path/to/dataset',
    output_dir='wst_analysis',
    J=2,
    L=8
)

# Compute and analyze features
analyzer.analyze(sample_size=50)  # Analyze 50 samples per class

# Generate visualizations of feature distributions by class
analyzer.plot_class_feature_distributions()

# Visualize feature importance for classification
analyzer.plot_feature_importance() 

# Generate t-SNE visualization of feature space
analyzer.plot_feature_space_tsne(perplexity=30)
```

## Classification Command Line Tools

The library provides command-line tools for classification-related tasks:

### Dataset Inspection for Classification

```bash
# Generate comprehensive dataset report for classification
python -m wavelet_lib.dataset_tools.dataset_inspector \
  --dataset /path/to/dataset \
  --output dataset_report.txt \
  --visualize

# Analyze class distribution
python -m wavelet_lib.dataset_tools.dataset_inspector \
  --dataset /path/to/dataset \
  --class-distribution \
  --output class_dist.png
```

### Wavelet Feature Visualization

```bash
# Visualize wavelet scattering coefficients for an image
python -m wavelet_lib.image_tools.visualize_wst \
  --image /path/to/image.jpg \
  --output wst_viz.png \
  --j 2 \
  --order 2
  
# Compare wavelet features between classes
python -m wavelet_lib.image_tools.compare_class_features \
  --dataset /path/to/dataset \
  --samples 10 \
  --output feature_comparison.png
```

### Feature Extraction Utilities

```bash
# Extract wavelet features from dataset for external analysis
python -m wavelet_lib.classification.extract_features \
  --dataset /path/to/dataset \
  --model /path/to/model.pth \
  --output features.pkl \
  --batch-size 64
```

## Classification Experiments

The `experiments` directory contains examples of classification tasks with different datasets:

- `experiments/custom_dataset_4_classe/`: Multi-class classification example with 4 classes
- `experiments/dataset1/`: Binary classification example
- `experiments/dataset2/`: Another classification example with different parameters

Each experiment includes:
- Training configuration
- Evaluation metrics
- Visualizations of results
- Analysis of model performance

To replicate an experiment, see the README.md file in each experiment directory.