#!/usr/bin/env python3
"""
Example script demonstrating how to use the Wavelet Scattering Transform Library
for image classification.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt

# Add parent directory to path to import wavelet_lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import wavelet library modules
from wavelet_lib.base import Config, set_seed, save_model
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders, print_dataset_summary
from wavelet_lib.models import create_model, print_model_summary
from wavelet_lib.training import Trainer, create_optimizer
from wavelet_lib.visualization import plot_class_distribution, visualize_batch, plot_confusion_matrix
from wavelet_lib.processors import ImageProcessor

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Define paths
    dataset_root = '/home/brus/Projects/wavelet/datasets/HPL_images/custom_dataset'
    models_dir = '/home/brus/Projects/wavelet/models'
    model_output_dir = os.path.join(models_dir, 'model_output_example')
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Create configuration
    config = Config(
        num_channels=3,
        num_classes=7,  # Adjust based on your dataset
        scattering_order=2,
        J=2,
        shape=(32, 32),
        batch_size=128,
        epochs=5,  # Reduced for example
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # Print configuration summary
    config.print_summary()
    
    # Create dataset with default transform
    transform = get_default_transform(target_size=(32, 32))
    balanced_dataset = BalancedDataset(dataset_root, transform=transform, balance=True)
    
    # Plot class distribution
    plot_class_distribution(balanced_dataset, 
                           title="Class Distribution in Dataset",
                           save_path=os.path.join(model_output_dir, 'class_distribution.png'))
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        balanced_dataset,
        test_size=0.2,
        batch_size=config.batch_size,
        num_workers=4
    )
    
    # Print dataset summary
    print_dataset_summary(balanced_dataset, train_loader, test_loader)
    
    # Visualize a batch of training data
    visualize_batch(train_loader, 
                   num_images=16, 
                   save_path=os.path.join(model_output_dir, 'batch_visualization.png'))
    
    # Create model and scattering transform
    model, scattering = create_model(config)
    
    # Print model summary
    print_model_summary(model, scattering, config.device)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create trainer
    trainer = Trainer(model, scattering, config.device, optimizer)
    
    # Train model
    training_results = trainer.train(
        train_loader,
        test_loader,
        config.epochs,
        save_path=os.path.join(model_output_dir, 'model.pth'),
        class_to_idx=balanced_dataset.class_to_idx
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        model,
        scattering,
        test_loader,
        balanced_dataset.classes,
        config.device,
        save_path=os.path.join(model_output_dir, 'confusion_matrix.png')
    )
    
    # Example of using the image processor to classify an image
    # (Comment out if you don't have a suitable test image)
    """
    test_image_path = '/path/to/test/image.jpg'
    image_processor = ImageProcessor(
        model, 
        scattering, 
        config.device,
        class_names=balanced_dataset.classes
    )
    
    # Process single image
    result = image_processor.process_image(test_image_path)
    print(f"Image classified as: {result['class_name']} with confidence {result['confidence']:.2f}")
    
    # Classify image by tiles
    tile_results = image_processor.classify_image_tiles(
        test_image_path,
        tile_size=32,
        confidence_threshold=0.7
    )
    
    # Visualize tile classification
    visualize_classification_results(
        tile_results,
        class_names=balanced_dataset.classes,
        save_path=os.path.join(model_output_dir, 'tile_classification.png')
    )
    """
    
    print(f"Model training and evaluation complete. Results saved in {model_output_dir}")

if __name__ == "__main__":
    main()