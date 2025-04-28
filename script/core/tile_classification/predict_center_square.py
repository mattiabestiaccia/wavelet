#!/usr/bin/env python3
"""
Center Square Prediction script for Wavelet Scattering Transform classification models.

This script processes a square grid of tiles in the center of an image and visualizes
the classification results overlaid on the original image.

Usage:
    python script/core/classification/predict_center_square.py --model-path /path/to/model.pth --image-path /path/to/image.jpg [options]
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms
import time

# Add the main directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import wavelet_lib modules
from wavelet_lib.base import load_model
from wavelet_lib.single_tile_classification import create_scattering_transform, ScatteringClassifier
from wavelet_lib.single_tile_classification import ClassificationProcessor as ImageProcessor

def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(description='Make predictions on a center square grid with Wavelet Scattering Transform model')

    # Model parameters
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the image to classify')

    # Prediction parameters
    parser.add_argument('--tile-size', type=int, default=32, help='Tile size')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Confidence threshold for visualization')
    parser.add_argument('--grid-size', type=int, default=5, help='Number of tiles per side of the center square grid')

    # General parameters
    parser.add_argument('--device', type=str, default=None, help='Device for inference (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--experiment-name', type=str, default=None, help='Name for this experiment (used in output path)')
    parser.add_argument('--output-base', type=str, default=None, help='Base directory for storing results (default: results)')
    parser.add_argument('--dataset-root', type=str, default=None,
                       help='Path to dataset root directory (for class names, optional)')

    return parser.parse_args()

def classify_center_square(image_path, model, scattering, device, class_names,
                          tile_size=32, grid_size=5, confidence_threshold=0.7):
    """
    Classify a center square grid of tiles in an image using a trained model and Wavelet Scattering Transform.

    Args:
        image_path: Path to the image to classify
        model: Trained model
        scattering: Scattering transform
        device: Device for inference
        class_names: List of class names
        tile_size: Tile size (default: 32)
        grid_size: Number of tiles per side of the square grid (default: 5)
        confidence_threshold: Confidence threshold (default: 0.7)

    Returns:
        dict: Classification results
    """
    # Load and convert image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    img_height, img_width, _ = image.shape

    # Calculate the total size of the grid in pixels
    grid_width_px = grid_size * tile_size
    grid_height_px = grid_size * tile_size

    # Check if the image is large enough for the grid
    if img_width < grid_width_px or img_height < grid_height_px:
        raise ValueError(f"Image too small for {grid_size}x{grid_size} grid with tile size {tile_size}")

    # Calculate the starting position for the center grid
    start_y = (img_height - grid_height_px) // 2
    start_x = (img_width - grid_width_px) // 2

    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((tile_size, tile_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Matrix for labels and confidences
    label_matrix = np.full((grid_size, grid_size), -1, dtype=int)
    confidence_matrix = np.zeros((grid_size, grid_size), dtype=float)

    # Process each tile in the grid
    print(f"Processing {grid_size}x{grid_size} center square grid...")
    total_tiles = grid_size * grid_size
    processed_tiles = 0

    # Initialize timing variables
    start_time = time.time()
    last_update_time = start_time

    with torch.no_grad():
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate absolute position
                y = start_y + i * tile_size
                x = start_x + j * tile_size

                # Extract tile
                tile = image[y:y+tile_size, x:x+tile_size, :]

                # Convert to tensor
                tile_img = Image.fromarray(tile)
                tile_tensor = transform(tile_img).unsqueeze(0).to(device)

                # Apply scattering transform
                scattering_coeffs = scattering(tile_tensor)

                # Get model prediction
                output = model(scattering_coeffs)

                # Calculate softmax for probabilities
                probabilities = torch.softmax(output, dim=1)
                max_prob, label = torch.max(probabilities, dim=1)

                # Store result
                confidence = max_prob.item()
                if confidence >= confidence_threshold:
                    label_matrix[i, j] = label.item()
                    confidence_matrix[i, j] = confidence

                # Update progress
                processed_tiles += 1
                progress_percent = (processed_tiles / total_tiles) * 100

                # Show progress update for every 10% or for the last tile
                if processed_tiles % max(1, total_tiles // 10) == 0 or processed_tiles == total_tiles:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    tiles_per_second = processed_tiles / max(0.001, elapsed_time)

                    # Estimate remaining time
                    remaining_tiles = total_tiles - processed_tiles
                    estimated_remaining_time = remaining_tiles / max(0.001, tiles_per_second)

                    # Format time as minutes:seconds
                    elapsed_min, elapsed_sec = divmod(int(elapsed_time), 60)
                    remaining_min, remaining_sec = divmod(int(estimated_remaining_time), 60)

                    print(f"Progress: {processed_tiles}/{total_tiles} tiles ({progress_percent:.1f}%) | " +
                          f"Elapsed: {elapsed_min:02d}:{elapsed_sec:02d} | " +
                          f"Remaining: {remaining_min:02d}:{remaining_sec:02d} | " +
                          f"Speed: {tiles_per_second:.2f} tiles/sec")

    print("Classification complete.")

    # Count classes
    class_counts = {}
    for class_idx, name in enumerate(class_names):
        class_counts[class_idx] = np.sum(label_matrix == class_idx)

    # Create results dictionary
    results = {
        'image': image,
        'label_matrix': label_matrix,
        'confidence_matrix': confidence_matrix,
        'tile_size': tile_size,
        'grid_size': grid_size,
        'start_position': (start_y, start_x),
        'class_names': class_names,
        'class_counts': class_counts,
        'total_tiles': grid_size * grid_size
    }

    return results

def visualize_center_square(results, save_path=None):
    """
    Visualize classification results for center square grid.

    Args:
        results: Classification results
        save_path: Path to save the visualization
    """
    image = results['image']
    label_matrix = results['label_matrix']
    confidence_matrix = results['confidence_matrix']
    tile_size = results['tile_size']
    grid_size = results['grid_size']
    start_y, start_x = results['start_position']
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

    # Draw grid boundary
    grid_width = grid_size * tile_size
    grid_height = grid_size * tile_size

    # Draw a white border around the entire grid
    rect = plt.Rectangle(
        (start_x, start_y),
        grid_width, grid_height,
        linewidth=2,
        edgecolor='white',
        facecolor='none'
    )
    plt.gca().add_patch(rect)

    # Draw colored tiles
    ax = plt.gca()
    for i in range(grid_size):
        for j in range(grid_size):
            label = label_matrix[i, j]
            if label >= 0 and label < num_classes:
                color = colors[label]
                rect = plt.Rectangle(
                    (start_x + j * tile_size, start_y + i * tile_size),
                    tile_size, tile_size,
                    linewidth=1,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.5  # Semi-transparent
                )
                ax.add_patch(rect)

    # Create legend
    legend_patches = []
    for class_idx, class_name in enumerate(class_names):
        count = class_counts.get(class_idx, 0)
        percentage = 100 * count / total_tiles
        patch = mpatches.Patch(
            color=colors[class_idx],
            label=f"{class_name}: {count} tiles ({percentage:.1f}%)"
        )
        legend_patches.append(patch)

    # Add legend
    plt.legend(handles=legend_patches,
               loc='center left',
               bbox_to_anchor=(1, 0.5),
               fontsize=10,
               framealpha=0.8)

    unclassified = total_tiles - classified_tiles
    unclassified_percentage = 100 * unclassified / total_tiles

    plt.title(f'Center Square Classification - {grid_size}×{grid_size} grid ({total_tiles} tiles)\n' +
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

def main():
    """
    Main function for center square grid prediction.
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
    print(f"Wavelet Scattering Transform Center Square Grid Prediction")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Image: {args.image_path}")
    print(f"Device: {device}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Tile size: {args.tile_size}")
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
            result_dir = f"center_square_{base_filename}"

        args.output_dir = os.path.join(output_base, "classification_result", result_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Run prediction
    print(f"Running prediction on center square grid...")

    results = classify_center_square(
        args.image_path,
        model,
        scattering,
        device,
        class_names,
        tile_size=args.tile_size,
        grid_size=args.grid_size,
        confidence_threshold=args.confidence_threshold
    )

    # Visualize and save results
    save_path = os.path.join(args.output_dir, "center_square_classification.png")
    visualize_center_square(results, save_path=save_path)

    # Print class distribution summary
    print("\nCLASS DISTRIBUTION SUMMARY:")
    print("-" * 50)
    total_tiles = results['total_tiles']
    classified_tiles = sum(results['class_counts'].values())
    print(f"Grid size: {args.grid_size}x{args.grid_size} ({total_tiles} tiles)")
    print(f"Classified tiles (confidence ≥ {args.confidence_threshold}): {classified_tiles} ({classified_tiles/total_tiles*100:.1f}%)")
    print(f"Unclassified tiles (confidence < {args.confidence_threshold}): {total_tiles - classified_tiles} ({(total_tiles - classified_tiles)/total_tiles*100:.1f}%)")
    print("-" * 50)
    for class_idx, count in results['class_counts'].items():
        class_name = class_names[class_idx]
        percentage = 100 * count / total_tiles
        print(f"{class_name}: {count} tiles ({percentage:.1f}%)")

    print(f"\nPrediction completed!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
