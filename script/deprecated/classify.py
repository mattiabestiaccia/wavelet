#!/usr/bin/env python3
"""
Script for classifying images using the trained WST classifier.
Usage: python classify.py <image_path> [--model_path <model_path>] [--tile_size <tile_size>] [--confidence <threshold>]
"""

import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt

# Add parent directory to path to import wavelet_lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wavelet_lib.base import Config, load_model
from wavelet_lib.models import ScatteringClassifier, create_scattering_transform
from wavelet_lib.processors import ImageProcessor
from wavelet_lib.visualization import visualize_classification_results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Classify images using a trained WST classifier')
    parser.add_argument('image_path', type=str, help='Path to the image to classify')
    parser.add_argument('--model_path', type=str, 
                       default='/home/brus/Projects/wavelet/models/model_output_7/best_model.pth',
                       help='Path to the trained model')
    parser.add_argument('--dataset_root', type=str,
                        default='/home/brus/Projects/wavelet/datasets/HPL_images/custom_dataset',
                        help='Path to dataset root directory (for class names)')
    parser.add_argument('--tile_size', type=int, default=32,
                       help='Size of tiles for classification (default: 32)')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold for classification (default: 0.7)')
    parser.add_argument('--process_30x30', action='store_true',
                       help='Process 30x30 tiles (special case)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the visualization output')
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create configuration
    config = Config(
        num_channels=3,
        scattering_order=2,
        J=2,
        shape=(32, 32),
        device=device
    )
    
    # Get class names from dataset directory
    if os.path.exists(args.dataset_root):
        class_names = sorted([d for d in os.listdir(args.dataset_root) 
                            if os.path.isdir(os.path.join(args.dataset_root, d))])
        config.num_classes = len(class_names)
        print(f"Found {config.num_classes} classes: {', '.join(class_names)}")
    else:
        print(f"Warning: Dataset directory not found: {args.dataset_root}")
        print("Class names will not be available.")
        class_names = None
    
    # Create scattering transform
    scattering = create_scattering_transform(
        J=config.J,
        shape=config.shape,
        max_order=config.scattering_order,
        device=config.device
    )
    
    # Create model
    model = ScatteringClassifier(
        in_channels=config.scattering_coeffs,
        classifier_type='cnn',
        num_classes=config.num_classes
    ).to(config.device)
    
    # Load model weights
    class_to_idx = load_model(model, args.model_path, device)
    
    # If class_to_idx is available from the saved model, use it to get class names
    if class_to_idx is not None:
        class_names = [None] * len(class_to_idx)
        for cls, idx in class_to_idx.items():
            class_names[idx] = cls
        print(f"Class names from model: {', '.join(class_names)}")
    
    # Create image processor
    processor = ImageProcessor(model, scattering, device, class_names)
    
    # Process image
    print(f"Processing image: {args.image_path}")
    print(f"Tile size: {args.tile_size}, Confidence threshold: {args.confidence}")
    
    # Classify image tiles
    results = processor.classify_image_tiles(
        args.image_path,
        tile_size=args.tile_size,
        confidence_threshold=args.confidence,
        process_30x30_tiles=args.process_30x30
    )
    
    # Visualize results
    print("Generating visualization...")
    visualize_classification_results(
        results,
        class_names=class_names,
        save_path=args.output
    )
    
    # Print class distribution summary
    print("\nCLASS DISTRIBUTION SUMMARY:")
    print("-" * 50)
    total_tiles = results['total_tiles']
    classified_tiles = sum(results['class_counts'].values())
    print(f"Total tiles: {total_tiles}")
    print(f"Classified tiles (confidence ≥ {args.confidence}): {classified_tiles} ({classified_tiles/total_tiles*100:.1f}%)")
    print(f"Unclassified tiles (confidence < {args.confidence}): {total_tiles - classified_tiles} ({(total_tiles - classified_tiles)/total_tiles*100:.1f}%)")
    print("-" * 50)
    for class_idx, count in results['class_counts'].items():
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        percentage = 100 * count / total_tiles
        print(f"{class_name}: {count} tiles ({percentage:.1f}%)")
    
    if args.output:
        print(f"\nVisualization saved to: {args.output}")

if __name__ == "__main__":
    main()