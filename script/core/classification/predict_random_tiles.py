#!/usr/bin/env python3
"""
Random Tile Prediction script for Wavelet Scattering Transform classification models.

This script processes 5 random groups of 3x3 tiles in an image and visualizes the classification
results overlaid on the original image.

Usage:
    python script/core/classification/predict_random_tiles.py --model-path /path/to/model.pth --image-path /path/to/image.jpg [options]
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
import random

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
    parser = argparse.ArgumentParser(description='Make predictions on random tile groups with Wavelet Scattering Transform model')

    # Model parameters
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the image to classify')

    # Prediction parameters
    parser.add_argument('--tile-size', type=int, default=32, help='Tile size')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Confidence threshold for visualization')
    parser.add_argument('--num-groups', type=int, default=5, help='Number of 3x3 tile groups to process')
    parser.add_argument('--group-size', type=int, default=3, help='Size of each tile group (default: 3x3)')

    # General parameters
    parser.add_argument('--device', type=str, default=None, help='Device for inference (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--experiment-name', type=str, default=None, help='Name for this experiment (used in output path)')
    parser.add_argument('--output-base', type=str, default=None, help='Base directory for storing results (default: results)')
    parser.add_argument('--dataset-root', type=str, default=None,
                       help='Path to dataset root directory (for class names, optional)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    return parser.parse_args()

def classify_random_tile_groups(image_path, model, scattering, device, class_names,
                          tile_size=32, num_groups=5, group_size=3, confidence_threshold=0.7, seed=None):
    """
    Classify random groups of tiles in an image using a trained model and Wavelet Scattering Transform.
    Avoids reclassifying tiles that have already been processed.

    Args:
        image_path: Path to the image to classify
        model: Trained model
        scattering: Scattering transform
        device: Device for inference
        class_names: List of class names
        tile_size: Tile size (default: 32)
        num_groups: Number of tile groups to process (default: 5)
        group_size: Size of each tile group (default: 3)
        confidence_threshold: Confidence threshold (default: 0.7)
        seed: Random seed for reproducibility

    Returns:
        dict: Classification results
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Load and convert image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    img_height, img_width, _ = image.shape

    # Calculate maximum valid starting positions for tile groups
    max_start_y = img_height - (group_size * tile_size)
    max_start_x = img_width - (group_size * tile_size)

    if max_start_y <= 0 or max_start_x <= 0:
        raise ValueError(f"Image too small for {group_size}x{group_size} tile groups with tile size {tile_size}")

    # Keep track of processed tiles to avoid reclassification
    processed_tiles = set()

    # Generate random starting positions for tile groups
    group_positions = []
    attempts = 0
    max_attempts = num_groups * 10  # Limit attempts to avoid infinite loops

    while len(group_positions) < num_groups and attempts < max_attempts:
        start_y = random.randint(0, max_start_y)
        start_x = random.randint(0, max_start_x)

        # Check if any tile in this group overlaps with already processed tiles
        overlap = False
        for i in range(group_size):
            for j in range(group_size):
                y = start_y + i * tile_size
                x = start_x + j * tile_size
                if (y, x) in processed_tiles:
                    overlap = True
                    break
            if overlap:
                break

        if not overlap:
            group_positions.append((start_y, start_x))
            # Mark all tiles in this group as processed
            for i in range(group_size):
                for j in range(group_size):
                    y = start_y + i * tile_size
                    x = start_x + j * tile_size
                    processed_tiles.add((y, x))

        attempts += 1

    if len(group_positions) < num_groups:
        print(f"Warning: Could only generate {len(group_positions)} non-overlapping groups (requested {num_groups})")

    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((tile_size, tile_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Store classification results
    group_results = []

    # Process each tile group
    print(f"Processing {len(group_positions)} groups of {group_size}x{group_size} tiles...")

    with torch.no_grad():
        for group_idx, (start_y, start_x) in enumerate(group_positions):
            print(f"Processing group {group_idx+1}/{len(group_positions)} at position ({start_y}, {start_x})")

            # Store results for this group
            group_result = {
                'position': (start_y, start_x),
                'tiles': []
            }

            # Process each tile in the group
            for i in range(group_size):
                for j in range(group_size):
                    # Extract tile
                    y = start_y + i * tile_size
                    x = start_x + j * tile_size
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
                    class_idx = label.item() if confidence >= confidence_threshold else -1

                    tile_result = {
                        'position': (y, x),
                        'class_idx': class_idx,
                        'confidence': confidence,
                        'relative_position': (i, j)
                    }

                    group_result['tiles'].append(tile_result)

            group_results.append(group_result)

    print("Classification complete.")

    # Count classes
    class_counts = {}
    total_tiles = num_groups * group_size * group_size
    classified_tiles = 0

    for group in group_results:
        for tile in group['tiles']:
            class_idx = tile['class_idx']
            if class_idx >= 0:
                class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
                classified_tiles += 1

    # Create results dictionary
    results = {
        'image': image,
        'group_results': group_results,
        'tile_size': tile_size,
        'group_size': group_size,
        'class_names': class_names,
        'class_counts': class_counts,
        'total_tiles': total_tiles,
        'classified_tiles': classified_tiles
    }

    return results

def visualize_random_tile_groups(results, save_path=None):
    """
    Visualize classification results for random tile groups.

    Args:
        results: Classification results
        save_path: Path to save the visualization
    """
    image = results['image']
    group_results = results['group_results']
    tile_size = results['tile_size']
    group_size = results['group_size']
    class_names = results['class_names']
    class_counts = results['class_counts']
    total_tiles = results['total_tiles']
    classified_tiles = results['classified_tiles']

    num_classes = len(class_names)
    colors = list(mcolors.TABLEAU_COLORS.values())[:num_classes]

    plt.figure(figsize=(15, 12))

    # Display image
    plt.imshow(image)

    # Draw colored tiles
    ax = plt.gca()

    for group in group_results:
        # Draw group boundary
        start_y, start_x = group['position']
        group_width = group_size * tile_size
        group_height = group_size * tile_size

        # Draw a white border around the group
        rect = plt.Rectangle(
            (start_x, start_y),
            group_width, group_height,
            linewidth=2,
            edgecolor='white',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Draw individual tiles
        for tile in group['tiles']:
            y, x = tile['position']
            class_idx = tile['class_idx']

            if class_idx >= 0 and class_idx < num_classes:
                color = colors[class_idx]
                rect = plt.Rectangle(
                    (x, y),
                    tile_size, tile_size,
                    linewidth=1,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.5  # Semi-transparent
                )
                ax.add_patch(rect)

    # Create legend
    legend_patches = []
    for class_idx in range(num_classes):
        class_name = class_names[class_idx]
        count = class_counts.get(class_idx, 0)
        percentage = 100 * count / total_tiles if total_tiles > 0 else 0
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
    unclassified_percentage = 100 * unclassified / total_tiles if total_tiles > 0 else 0

    plt.title(f'Random Tile Group Classification - {total_tiles} tiles in {len(group_results)} groups\n' +
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
    Main function for random tile group prediction.
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
    print(f"Wavelet Scattering Transform Random Tile Group Prediction")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Image: {args.image_path}")
    print(f"Device: {device}")
    print(f"Number of groups: {args.num_groups}")
    print(f"Group size: {args.group_size}x{args.group_size}")
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
            result_dir = f"random_tiles_{base_filename}"

        args.output_dir = os.path.join(output_base, "classification_result", result_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Run prediction
    print(f"Running prediction on random tile groups...")

    results = classify_random_tile_groups(
        args.image_path,
        model,
        scattering,
        device,
        class_names,
        tile_size=args.tile_size,
        num_groups=args.num_groups,
        group_size=args.group_size,
        confidence_threshold=args.confidence_threshold,
        seed=args.seed
    )

    # Visualize and save results
    save_path = os.path.join(args.output_dir, "random_tile_classification.png")
    visualize_random_tile_groups(results, save_path=save_path)

    # Print class distribution summary
    print("\nCLASS DISTRIBUTION SUMMARY:")
    print("-" * 50)
    total_tiles = results['total_tiles']
    classified_tiles = results['classified_tiles']
    actual_groups = len(results['group_results'])
    print(f"Groups processed: {actual_groups} (requested: {args.num_groups})")
    print(f"Total tiles: {total_tiles}")
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
