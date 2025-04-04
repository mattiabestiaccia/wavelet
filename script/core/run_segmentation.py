#!/usr/bin/env python3
"""
Segmentation execution script for the Wavelet Scattering Transform Library.

This script runs image segmentation using a pre-trained WST-UNet model.
It can process individual images or batches of images for area segmentation.

Usage:
    python script/core/run_segmentation.py --image /path/to/image.jpg --model /path/to/model.pth
    python script/core/run_segmentation.py --folder /path/to/images --model /path/to/model.pth --output /path/to/output

Author: Claude
Date: 2025-04-01
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add the main directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import wavelet_lib modules
from wavelet_lib.segmentation import ScatteringSegmenter
from wavelet_lib.base import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run image segmentation with WST-UNet model')

    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to a single image to segment')
    input_group.add_argument('--folder', type=str, help='Path to a folder of images to process')

    # Model arguments
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--input-size', type=str, default='256,256', help='Input size for the model (W,H)')
    parser.add_argument('--j', type=int, default=2, help='Number of wavelet scales (J parameter)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    parser.add_argument('--no-morphology', action='store_true', help='Disable morphological operations')

    # Output arguments
    parser.add_argument('--output', type=str, help='Output directory for segmentation results')
    parser.add_argument('--overlay', action='store_true', help='Create overlay of segmentation on original image')
    parser.add_argument('--no-display', action='store_true', help='Do not display results (for headless mode)')

    return parser.parse_args()


def segment_image(segmenter, image_path, threshold, output_dir=None, overlay=False, display=True):
    """
    Segment a single image and save/display the results.

    Args:
        segmenter: Initialized ScatteringSegmenter
        image_path: Path to the image to segment
        threshold: Threshold for binary segmentation
        output_dir: Directory to save the results
        overlay: Whether to create overlay of segmentation on original image
        display: Whether to display the results

    Returns:
        binary_mask: Binary segmentation mask
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return None

    image_name = os.path.basename(image_path)
    print(f"\nProcessing: {image_name}")

    # Time the segmentation
    start_time = time.time()

    # Make prediction
    binary_mask, raw_pred = segmenter.predict(image_path, threshold=threshold, return_raw=True)

    # Measure elapsed time
    elapsed_time = time.time() - start_time
    print(f"Segmentation completed in {elapsed_time:.2f} seconds")

    # Load original image for visualization
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Calculate segmented area
    total_pixels = binary_mask.size
    segmented_pixels = np.sum(binary_mask)
    segmented_percentage = (segmented_pixels / total_pixels) * 100
    print(f"Segmented area: {segmented_percentage:.2f}% of the image")

    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Generate file paths
        mask_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_mask.png")
        heatmap_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_heatmap.png")

        # Save binary mask
        cv2.imwrite(mask_path, binary_mask * 255)

        # Create and save heatmap visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_pred, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(label='Prediction Confidence')
        plt.title(f'Segmentation Heatmap')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Create and save overlay if requested
        if overlay:
            overlay_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_overlay.png")
            overlay_img = original_rgb.copy()
            overlay_mask = np.zeros_like(original_rgb)
            overlay_mask[binary_mask > 0] = [0, 255, 0]  # Green overlay
            overlay_img = cv2.addWeighted(overlay_img, 0.7, overlay_mask, 0.3, 0)
            plt.figure(figsize=(10, 10))
            plt.imshow(overlay_img)
            plt.title('Segmentation Overlay')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Results saved to: {output_dir}")

    # Display results if requested
    if display and not args.no_display:
        # Create visualization grid
        plt.figure(figsize=(18, 6))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_rgb)
        plt.title('Original Image')
        plt.axis('off')

        # Binary mask
        plt.subplot(1, 3, 2)
        plt.imshow(binary_mask, cmap='gray')
        plt.title(f'Binary Mask (Area: {segmented_percentage:.2f}%)')
        plt.axis('off')

        # Heatmap
        plt.subplot(1, 3, 3)
        plt.imshow(raw_pred, cmap='jet', vmin=0, vmax=1)
        plt.title('Prediction Heatmap')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return binary_mask


def process_folder(segmenter, folder_path, threshold, output_dir, overlay, display):
    """
    Process a folder of images.

    Args:
        segmenter: Initialized ScatteringSegmenter
        folder_path: Path to the folder of images
        threshold: Threshold for binary segmentation
        output_dir: Directory to save the results
        overlay: Whether to create overlay of segmentation on original image
        display: Whether to display the results

    Returns:
        list: List of results for each image
    """
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return None

    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get list of images
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"No images found in: {folder_path}")
        return None

    print(f"Found {len(image_paths)} images to process")

    # Process each image
    results = []
    for image_path in image_paths:
        mask = segment_image(
            segmenter, image_path, threshold,
            output_dir=output_dir,
            overlay=overlay,
            display=display
        )

        if mask is not None:
            # Calculate segmented area
            total_pixels = mask.size
            segmented_pixels = np.sum(mask)
            segmented_percentage = (segmented_pixels / total_pixels) * 100

            results.append({
                'image_path': image_path,
                'segmented_percentage': segmented_percentage
            })

    # Create summary report
    if output_dir and results:
        report_path = os.path.join(output_dir, "segmentation_report.txt")
        with open(report_path, 'w') as f:
            f.write("Wavelet Scattering Transform Segmentation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Images processed: {len(results)}\n")
            f.write(f"Threshold: {threshold}\n\n")

            f.write("Image Results:\n")
            f.write("-" * 50 + "\n")

            # Sort by segmented percentage
            results.sort(key=lambda x: x['segmented_percentage'], reverse=True)

            for result in results:
                image_name = os.path.basename(result['image_path'])
                segmented_percentage = result['segmented_percentage']
                f.write(f"{image_name}: {segmented_percentage:.2f}%\n")

            # Calculate average segmentation percentage
            avg_percentage = sum(r['segmented_percentage'] for r in results) / len(results)
            f.write("\nAverage segmented area: {:.2f}%\n".format(avg_percentage))

        print(f"\nSegmentation report saved to: {report_path}")

    return results


def main(args):
    """Main function to run segmentation."""
    # Parse input size
    input_size = tuple(map(int, args.input_size.split(',')))

    # Create segmenter
    segmenter = ScatteringSegmenter(
        model_path=args.model,
        J=args.j,
        input_shape=input_size,
        apply_morphology=not args.no_morphology
    )

    # Process input based on arguments
    if args.image:
        segment_image(
            segmenter,
            args.image,
            args.threshold,
            output_dir=args.output,
            overlay=args.overlay,
            display=not args.no_display
        )
    elif args.folder:
        process_folder(
            segmenter,
            args.folder,
            args.threshold,
            args.output,
            args.overlay,
            display=not args.no_display
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)