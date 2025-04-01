# Wavelet Scattering Transform (WST) Image Classification

A modular framework for image classification using Wavelet Scattering Transform representations. The framework provides tools for training, evaluating, and deploying models on various types of image data.

## Features

- Wavelet Scattering Transform based feature extraction
- Support for various neural network architectures
- Balanced dataset handling
- Training and evaluation tools
- Tile-based image classification for large images
- Visualization tools for model analysis

## Installation

1. Clone the repository:

```bash
git clone https://github.com/brus/wavelet.git
cd wavelet
```

2. Create a virtual environment and install requirements:

```bash
python3 -m venv wavelet_venv
source wavelet_venv/bin/activate
pip install -r wavelet_venv/requirements.txt
```

3. Install the library in development mode:

```bash
pip install -e .
```

## Project Structure

```
wavelet/
├── datasets/                  # Dataset storage
├── models/                    # Saved model checkpoints
│   ├── model_output_4/        # 4-class model outputs
│   └── model_output_7/        # 7-class model outputs
├── script/                    # Scripts for training and inference
│   ├── wst_train.py           # Training script
│   ├── wst_predict.py         # Prediction script
│   └── view_metrics.py        # Metrics visualization
├── wavelet_lib/               # Core library
│   ├── base.py                # Base configurations and utilities
│   ├── datasets.py            # Dataset handling
│   ├── models.py              # Neural network models
│   ├── processors.py          # Image processing utilities
│   ├── training.py            # Training tools
│   └── visualization.py       # Visualization tools
└── README.md
```

## Basic Usage

```python
from wavelet_lib.base import Config
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders
from wavelet_lib.models import create_model
from wavelet_lib.training import Trainer, create_optimizer

# Create configuration
config = Config(
    num_channels=3,
    num_classes=4,
    scattering_order=2
)

# Create dataset
transform = get_default_transform()
dataset = BalancedDataset('/path/to/dataset', transform=transform)
train_loader, test_loader = create_data_loaders(dataset)

# Create model
model, scattering = create_model(config)

# Train model
optimizer = create_optimizer(model, config)
trainer = Trainer(model, scattering, config.device, optimizer)
results = trainer.train(train_loader, test_loader, config.epochs, save_path='model.pth')
```

## Training a Model

To train a new model, use the `train.py` script:

```bash
python script/core/train.py --dataset /path/to/dataset --num-classes 4 --epochs 90
```

Additional training parameters:

- `--balance`: Balance class distribution
- `--num-channels`: Number of input channels (default: 3)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.1)
- `--device`: Device to use (cuda or cpu)
- `--output-dir`: Directory to save results

## Making Predictions

To make predictions using a trained model:

```bash
python script/core/predict.py --model-path /path/to/model.pth --image-path /path/to/image.jpg
```

For tile-based classification of large images:

```bash
python script/core/predict.py --model-path /path/to/model.pth --image-path /path/to/image.jpg --tile-mode --tile-size 32
```

Additional prediction parameters:

- `--tile-mode`: Enable tile-based classification
- `--tile-size`: Size of tiles (default: 32)
- `--confidence-threshold`: Confidence threshold for visualization (default: 0.7)
- `--device`: Device to use (cuda or cpu)
- `--output-dir`: Directory to save results

## Evaluation

To evaluate a model on a test dataset:

```bash
python script/core/evaluate.py --model-path /path/to/model.pth --dataset /path/to/dataset
```

## Visualization

To visualize training metrics from a saved model:

```bash
python script/core/visualize.py metrics --model-dir models/model_output_4_classes_YYYYMMDD_HHMMSS
```

## Dataset Inspection

To check if a dataset is properly formatted for the model:

```bash
python script/utility/dataset_inspector.py --dataset /path/to/dataset --expected-dims 32x32
```

## Example Workflow

1. Prepare your dataset with class subdirectories:

```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

2. Check if your dataset is suitable for the model:

```bash
python script/utility/dataset_inspector.py --dataset /path/to/dataset --expected-dims 32x32
```

3. Train a model:

```bash
python script/core/train.py --dataset /path/to/dataset --num-classes 4 --epochs 90 --balance
```

4. View training metrics:

```bash
python script/core/visualize.py metrics --model-dir models/model_output_4_classes_YYYYMMDD_HHMMSS
```

5. Evaluate the model on test data:

```bash
python script/core/evaluate.py --model-path models/model_output_4_classes_YYYYMMDD_HHMMSS/best_model.pth --dataset /path/to/dataset
```

6. Make predictions on new images:

```bash
python script/core/predict.py --model-path models/model_output_4_classes_YYYYMMDD_HHMMSS/best_model.pth --image-path /path/to/test_image.jpg
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- Torchvision 0.8+
- Kymatio 0.2+
- NumPy
- Matplotlib
- scikit-learn
- Pillow
- tqdm

## Credits

- PyTorch Scattering: https://github.com/kymatio/kymatio
- PyTorch: https://pytorch.org/
