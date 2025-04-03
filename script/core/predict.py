#!/usr/bin/env python3
"""
Prediction script for Wavelet Scattering Transform models.

This script allows for image classification using trained models,
supporting both whole-image classification and tile-based analysis.

Usage:
    python script/core/predict.py --model-path /path/to/model.pth --image-path /path/to/image.jpg [options]
    python script/core/predict.py --model-path /path/to/model.pth --image-path /path/to/image.jpg --tile-mode [options]
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from torchvision import transforms

# Add the main directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import wavelet_lib modules
from wavelet_lib.base import load_model
from wavelet_lib.models import create_scattering_transform, ScatteringClassifier
from wavelet_lib.processors import ImageProcessor
from wavelet_lib.visualization import visualize_classification_results

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(description='Make predictions with Wavelet Scattering Transform model')
    
    # Model parameters
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the image to classify')
    
    # Prediction parameters
    parser.add_argument('--tile-mode', action='store_true', help='Enable tile mode')
    parser.add_argument('--tile-size', type=int, default=32, help='Tile size')
    parser.add_argument('--process-30x30', action='store_true', help='Process 30x30 tiles (cropped in some datasets)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Confidence threshold for visualization')
    
    # General parameters
    parser.add_argument('--device', type=str, default=None, help='Device for inference (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--experiment-name', type=str, default=None, help='Name for this experiment (used in output path)')
    parser.add_argument('--output-base', type=str, default=None, help='Base directory for storing results (default: results)')
    parser.add_argument('--dataset-root', type=str, default=None, 
                       help='Path to dataset root directory (for class names, optional)')
    
    return parser.parse_args()

def classify_image_tiles(image_path, model, scattering, device, class_names, 
                   tile_size=32, process_30x30_tiles=False, confidence_threshold=0.7):
    """
    Classify an image using a trained model and Wavelet Scattering Transform, using tile-based approach.
    
    Args:
        image_path: Path to the image to classify
        model: Trained model
        scattering: Scattering transform
        device: Device for inference
        class_names: List of class names
        tile_size: Tile size (default: 32)
        process_30x30_tiles: Whether to process 30x30 tiles (default: False)
        confidence_threshold: Confidence threshold (default: 0.7)
        
    Returns:
        dict: Classification results
    """
    # Load image with multiband support
    try:
        # Try to load with rasterio for multiband support
        import rasterio
        with rasterio.open(image_path) as src:
            # Read all bands
            img_array = src.read()
            num_bands = src.count
            
            # Convert to format (height, width, bands)
            image = np.transpose(img_array, (1, 2, 0))
    except:
        # Fallback to PIL for standard image formats
        image = Image.open(image_path)
        image = np.array(image)

    # Handle cropping for images with 30x30 tiles
    if process_30x30_tiles:
        tile_size = 30
        target_size = 32
        h, w, _ = image.shape
        center_y, center_x = h // 2, w // 2
        crop_size = 30 * 32
        y_start = max(0, center_y - crop_size // 2)
        x_start = max(0, center_x - crop_size // 2)
        cropped_image = image[y_start:y_start + crop_size, x_start:x_start + crop_size, :]
        img_height, img_width, _ = cropped_image.shape
    else:
        img_height, img_width, _ = image.shape
        cropped_image = image
        target_size = tile_size

    # Calculate number of tiles
    num_tiles_x = img_width // tile_size
    num_tiles_y = img_height // tile_size
    
    # Matrix for labels and confidences
    label_matrix = np.full((num_tiles_y, num_tiles_x), -1, dtype=int)
    confidence_matrix = np.zeros((num_tiles_y, num_tiles_x), dtype=float)

    # Prepare transforms
    transform_steps = []
    if tile_size != target_size:
        transform_steps.append(transforms.Resize((target_size, target_size)))
    # Determine number of channels from model or default to model's input channels
    if hasattr(model, 'in_channels'):
        num_channels = model.in_channels
    else:
        # Default to match image array if available
        if len(cropped_image.shape) == 3:
            num_channels = cropped_image.shape[2]
        else:
            num_channels = 3  # Default fallback
    
    # Create dynamic normalization arrays
    mean_values = [0.5] * num_channels
    std_values = [0.5] * num_channels
    
    transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values, std=std_values)
    ]
    transform = transforms.Compose(transform_steps)

    # Classify tiles
    total_tiles = num_tiles_x * num_tiles_y
    print(f"Processing {total_tiles} tiles...")

    with torch.no_grad():
        processed_tiles = 0
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                tile = cropped_image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size, :]
                tile_img = Image.fromarray(tile)
                tile_tensor = transform(tile_img).unsqueeze(0).to(device)

                scattering_coeffs = scattering(tile_tensor)
                
                # The model will handle reshaping internally - pass the scattering coefficients directly
                output = model(scattering_coeffs)
                
                # Calculate softmax for probabilities
                probabilities = torch.softmax(output, dim=1)
                max_prob, label = torch.max(probabilities, dim=1)
                
                # Store label and confidence
                if max_prob.item() >= confidence_threshold:
                    label_matrix[i, j] = label.item()
                    confidence_matrix[i, j] = max_prob.item()

                # Update progress
                processed_tiles += 1
                if processed_tiles % 100 == 0 or processed_tiles == total_tiles:
                    progress_percent = (processed_tiles / total_tiles) * 100
                    print(f"Progress: {processed_tiles}/{total_tiles} tiles ({progress_percent:.1f}%)")

    print("Classification complete.")
    
    # Count classes
    class_counts = {}
    for class_idx, name in enumerate(class_names):
        class_counts[class_idx] = np.sum(label_matrix == class_idx)
        
    # Create results dictionary
    results = {
        'image': cropped_image,
        'label_matrix': label_matrix,
        'confidence_matrix': confidence_matrix,
        'tile_size': tile_size,
        'class_names': class_names,
        'class_counts': class_counts,
        'total_tiles': total_tiles
    }
    
    return results

def visualize_tiles(results, save_path=None):
    """
    Visualize classification results for tile-based prediction.
    
    Args:
        results: Classification results
        save_path: Path to save the visualization
    """
    image = results['image']
    label_matrix = results['label_matrix']
    confidence_matrix = results['confidence_matrix']
    tile_size = results['tile_size']
    class_names = results['class_names']
    
    num_classes = len(class_names)
    colors = list(mcolors.TABLEAU_COLORS.values())[:num_classes]
    
    # Count classes
    class_counts = results['class_counts']
    total_tiles = results['total_tiles']
    classified_tiles = sum(class_counts.values())
    
    plt.figure(figsize=(15, 12))
    
    # Display image
    plt.imshow(image)

    # Draw colored tiles
    ax = plt.gca()
    for i in range(label_matrix.shape[0]):
        for j in range(label_matrix.shape[1]):
            label = label_matrix[i, j]
            if label >= 0 and label < num_classes:
                color = colors[label]
                rect = plt.Rectangle(
                    (j * tile_size, i * tile_size),
                    tile_size, tile_size,
                    linewidth=1,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.3  # Semi-transparent
                )
                ax.add_patch(rect)
    
    # Create legend
    legend_patches = []
    for class_idx, class_name in enumerate(class_names):
        count = class_counts.get(class_idx, 0)
        percentage = 100 * count / total_tiles
        patch = plt.Rectangle((0, 0), 1, 1,
                               linewidth=1,
                               edgecolor=colors[class_idx],
                               facecolor=colors[class_idx],
                               label=f"{class_name}: {count} tiles ({percentage:.1f}%)")
        legend_patches.append(patch)
    
    # Add legend
    plt.legend(handles=legend_patches,
               loc='center left',
               bbox_to_anchor=(1, 0.5),
               fontsize=10,
               framealpha=0.8)
    
    unclassified = total_tiles - classified_tiles
    unclassified_percentage = 100 * unclassified / total_tiles
    
    plt.title(f'Tile Classification - {total_tiles} tiles ({label_matrix.shape[0]}×{label_matrix.shape[1]})\n' +
              f'Classified: {classified_tiles} ({100*classified_tiles/total_tiles:.1f}%), ' +
              f'Unclassified: {unclassified} ({unclassified_percentage:.1f}%)')
    plt.tight_layout()
    plt.axis('off')
    
    # Save the image if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def classify_single_image(image_path, model, scattering, device, class_names):
    """
    Classify a whole image without dividing it into tiles.
    
    Args:
        image_path: Path to the image to classify
        model: Trained model
        scattering: Scattering transform
        device: Device for inference
        class_names: List of class names
        
    Returns:
        tuple: (class_name, confidence)
    """
    # Create image processor
    processor = ImageProcessor(model, scattering, device, class_names)
    
    # Classify image
    result = processor.process_image(image_path)
    
    return result['class_name'], result['confidence']

def main():
    """
    Main function for image prediction.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Configure device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*80}")
    print(f"Wavelet Scattering Transform Model Prediction")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Image: {args.image_path}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get class names
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        class_names = list(class_to_idx.keys())
    else:
        print("Warning: Class mapping not found in model file.")
        
        # Try to get class names from dataset root
        if args.dataset_root and os.path.exists(args.dataset_root):
            class_names = sorted([d for d in os.listdir(args.dataset_root) 
                            if os.path.isdir(os.path.join(args.dataset_root, d))])
            print(f"Class names from dataset: {class_names}")
        else:
            class_names = [f"Class {i}" for i in range(10)]  # Generic fallback
    
    print(f"Detected classes: {class_names}")
    
    # Create scattering transform
    scattering = create_scattering_transform(
        J=2,
        shape=(32, 32),
        max_order=2,
        device=device
    )
    
    # Create model from scratch with the same architecture
    num_classes = len(class_names)
    
    # Try to get the number of channels from the model checkpoint
    if 'model_state_dict' in checkpoint:
        # Look for the first batch normalization layer to get the channel count
        for key, value in checkpoint['model_state_dict'].items():
            if 'bn.weight' in key:
                num_channels = value.size(0)
                print(f"Detected input channels from checkpoint: {num_channels}")
                break
    else:
        # Fallback to default value if not found
        num_channels = 12
        print(f"Using default input channels: {num_channels}")
    
    # Create model with the correct number of channels
    model = ScatteringClassifier(in_channels=num_channels, num_classes=num_classes).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("Error: Could not find model weights in checkpoint")
        return
    
    model.eval()
    print("Model loaded successfully.")
    
    # Configure output directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Set output base directory
    if args.output_base is None:
        output_base = os.path.join(base_dir, "results")
    else:
        # Handle both absolute and relative paths
        if os.path.isabs(args.output_base):
            output_base = args.output_base
        else:
            output_base = os.path.join(base_dir, args.output_base)
    
    # Configure output directory
    if args.output_dir is None:
        filename = os.path.basename(args.image_path)
        base_filename, _ = os.path.splitext(filename)
        
        # Use experiment name if provided
        if args.experiment_name:
            result_dir = f"{args.experiment_name}_{base_filename}"
        else:
            result_dir = base_filename
            
        args.output_dir = os.path.join(output_base, "classification_result", result_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run prediction
    if args.tile_mode:
        # Tile mode
        print(f"Running prediction in tile mode (size: {args.tile_size})...")
        
        results = classify_image_tiles(
            args.image_path,
            model,
            scattering,
            device,
            class_names,
            tile_size=args.tile_size,
            process_30x30_tiles=args.process_30x30,
            confidence_threshold=args.confidence_threshold
        )
        
        # Visualize and save results
        save_path = os.path.join(args.output_dir, "tile_classification.png")
        visualize_tiles(results, save_path=save_path)
        
        # Print class distribution summary
        print("\nCLASS DISTRIBUTION SUMMARY:")
        print("-" * 50)
        total_tiles = results['total_tiles']
        classified_tiles = sum(results['class_counts'].values())
        print(f"Total tiles: {total_tiles}")
        print(f"Classified tiles (confidence ≥ {args.confidence_threshold}): {classified_tiles} ({classified_tiles/total_tiles*100:.1f}%)")
        print(f"Unclassified tiles (confidence < {args.confidence_threshold}): {total_tiles - classified_tiles} ({(total_tiles - classified_tiles)/total_tiles*100:.1f}%)")
        print("-" * 50)
        for class_idx, count in results['class_counts'].items():
            class_name = class_names[class_idx]
            percentage = 100 * count / total_tiles
            print(f"{class_name}: {count} tiles ({percentage:.1f}%)")
    else:
        # Single image mode
        print("Running prediction on whole image...")
        
        class_name, confidence = classify_single_image(
            args.image_path,
            model,
            scattering,
            device,
            class_names
        )
        
        print(f"\nClassification result:")
        print(f"Class: {class_name}")
        print(f"Confidence: {confidence:.4f}")
        
        # Visualize and save image with label
        img = Image.open(args.image_path).convert('RGB')
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(img))
        plt.title(f"Class: {class_name}\nConfidence: {confidence:.4f}")
        plt.axis('off')
        
        save_path = os.path.join(args.output_dir, "classification_result.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.show()
    
    print(f"\nPrediction completed!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()