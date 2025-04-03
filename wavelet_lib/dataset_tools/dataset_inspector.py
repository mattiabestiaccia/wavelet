#!/usr/bin/env python3
"""
Modulo per l'ispezione e la validazione di dataset.
Fornisce funzioni per analizzare un dataset e assicurarsi che sia strutturato
correttamente per l'addestramento e la valutazione di modelli.
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import traceback
from pathlib import Path

# Import wavelet_lib modules
from wavelet_lib.datasets import BalancedDataset, get_default_transform
from wavelet_lib.base import Config


def inspect_dataset_structure(dataset_path):
    """
    Ispeziona la struttura della directory del dataset.
    
    Args:
        dataset_path (str): Percorso della directory del dataset
        
    Returns:
        tuple: (classes, class_counts, total_files, structure_valid)
            - classes: lista delle classi trovate
            - class_counts: dizionario con il conteggio dei file per classe
            - total_files: numero totale di file
            - structure_valid: True se la struttura è valida, False altrimenti
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        return [], {}, 0, False
    
    if not dataset_path.is_dir():
        print(f"[ERROR] Dataset path is not a directory: {dataset_path}")
        return [], {}, 0, False
    
    # Check for class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"[ERROR] No class directories found in {dataset_path}")
        print("Expected structure: dataset_root/class_name/image_files")
        return [], {}, 0, False
    
    classes = [d.name for d in class_dirs]
    class_counts = {}
    total_files = 0
    
    print(f"[INFO] Found {len(classes)} classes: {', '.join(classes)}")
    
    # Count files in each class
    for class_dir in class_dirs:
        image_files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']]
        class_counts[class_dir.name] = len(image_files)
        total_files += len(image_files)
    
    return classes, class_counts, total_files, True


def analyze_image_properties(dataset_path, max_channels=10, sample_size=None):
    """
    Analizza le proprietà delle immagini nel dataset.
    
    Args:
        dataset_path (str): Percorso della directory del dataset
        max_channels (int): Numero massimo di canali supportati
        sample_size (int): Numero di immagini da analizzare per classe (None = tutte)
        
    Returns:
        dict: Statistiche sulle immagini (dimensioni, canali, formati)
    """
    dataset_path = Path(dataset_path)
    
    # Initialize statistics
    stats = {
        'dimensions': Counter(),
        'channels': Counter(),
        'formats': Counter(),
        'errors': [],
        'min_width': float('inf'),
        'max_width': 0,
        'min_height': float('inf'),
        'max_height': 0,
        'min_channels': float('inf'),
        'max_channels': 0,
    }
    
    # Get class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    # Collect all image files
    all_image_files = []
    for class_dir in class_dirs:
        image_files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']]
        
        if sample_size and len(image_files) > sample_size:
            # Random sampling
            import random
            image_files = random.sample(image_files, sample_size)
            
        all_image_files.extend(image_files)
    
    # Analyze images
    print(f"[INFO] Analyzing {len(all_image_files)} images...")
    
    for image_file in tqdm(all_image_files, desc="Analyzing images"):
        try:
            # Try to open with PIL first
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
                    channels = len(img.getbands())
                    format_name = img.format
            except:
                # If PIL fails, try with rasterio (for multi-band GeoTIFFs)
                import rasterio
                with rasterio.open(image_file) as src:
                    width, height = src.width, src.height
                    channels = src.count
                    format_name = "TIFF"
            
            # Update statistics
            stats['dimensions'][(width, height)] += 1
            stats['channels'][channels] += 1
            stats['formats'][format_name] += 1
            
            stats['min_width'] = min(stats['min_width'], width)
            stats['max_width'] = max(stats['max_width'], width)
            stats['min_height'] = min(stats['min_height'], height)
            stats['max_height'] = max(stats['max_height'], height)
            stats['min_channels'] = min(stats['min_channels'], channels)
            stats['max_channels'] = max(stats['max_channels'], channels)
            
            # Check if channels exceed maximum
            if channels > max_channels:
                stats['errors'].append(f"Image {image_file} has {channels} channels, which exceeds the maximum of {max_channels}")
                
        except Exception as e:
            error_msg = f"Error processing {image_file}: {str(e)}"
            stats['errors'].append(error_msg)
    
    return stats


def validate_dataset(dataset_path, min_samples=10, expected_dims=None, max_channels=10):
    """
    Valida un dataset per l'addestramento e la valutazione di modelli.
    
    Args:
        dataset_path (str): Percorso della directory del dataset
        min_samples (int): Numero minimo di campioni per classe
        expected_dims (tuple): Dimensioni attese (larghezza, altezza)
        max_channels (int): Numero massimo di canali supportati
        
    Returns:
        tuple: (is_valid, validation_results)
            - is_valid: True se il dataset è valido, False altrimenti
            - validation_results: dizionario con i risultati della validazione
    """
    results = {
        'structure_valid': False,
        'class_balance_valid': False,
        'dimensions_valid': True,
        'channels_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check dataset structure
    classes, class_counts, total_files, structure_valid = inspect_dataset_structure(dataset_path)
    results['structure_valid'] = structure_valid
    results['classes'] = classes
    results['class_counts'] = class_counts
    results['total_files'] = total_files
    
    if not structure_valid:
        results['errors'].append("Dataset structure is invalid")
        return False, results
    
    # Check class balance
    min_class_count = min(class_counts.values()) if class_counts else 0
    max_class_count = max(class_counts.values()) if class_counts else 0
    
    results['min_class_count'] = min_class_count
    results['max_class_count'] = max_class_count
    
    if min_class_count < min_samples:
        results['errors'].append(f"Some classes have fewer than {min_samples} samples")
        results['class_balance_valid'] = False
    else:
        results['class_balance_valid'] = True
    
    # Check imbalance ratio
    if min_class_count > 0:
        imbalance_ratio = max_class_count / min_class_count
        results['imbalance_ratio'] = imbalance_ratio
        
        if imbalance_ratio > 5:
            results['warnings'].append(f"High class imbalance detected (ratio: {imbalance_ratio:.2f})")
    
    # Analyze image properties
    stats = analyze_image_properties(dataset_path, max_channels)
    results.update(stats)
    
    # Check dimensions
    if expected_dims:
        expected_width, expected_height = expected_dims
        
        if stats['min_width'] != stats['max_width'] or stats['min_height'] != stats['max_height']:
            results['errors'].append(f"Images have inconsistent dimensions (width: {stats['min_width']}-{stats['max_width']}, height: {stats['min_height']}-{stats['max_height']})")
            results['dimensions_valid'] = False
        
        if stats['min_width'] != expected_width or stats['min_height'] != expected_height:
            results['errors'].append(f"Images do not match expected dimensions {expected_width}x{expected_height}")
            results['dimensions_valid'] = False
    
    # Check channels
    if stats['max_channels'] > max_channels:
        results['errors'].append(f"Some images have more than {max_channels} channels")
        results['channels_valid'] = False
    
    # Final validation
    is_valid = (
        results['structure_valid'] and
        results['class_balance_valid'] and
        results['dimensions_valid'] and
        results['channels_valid'] and
        not results['errors']
    )
    
    return is_valid, results


def generate_report(validation_results, output_path=None):
    """
    Genera un report di validazione del dataset.
    
    Args:
        validation_results (dict): Risultati della validazione
        output_path (str): Percorso dove salvare il report
        
    Returns:
        str: Report di validazione
    """
    report = []
    report.append("=" * 80)
    report.append("DATASET VALIDATION REPORT")
    report.append("=" * 80)
    
    # Overall status
    is_valid = (
        validation_results['structure_valid'] and
        validation_results['class_balance_valid'] and
        validation_results['dimensions_valid'] and
        validation_results['channels_valid'] and
        not validation_results['errors']
    )
    
    status = "PASSED" if is_valid else "FAILED"
    report.append(f"Overall Status: {status}")
    report.append("")
    
    # Dataset structure
    report.append("-" * 80)
    report.append("Dataset Structure:")
    report.append(f"- Structure Valid: {'Yes' if validation_results['structure_valid'] else 'No'}")
    report.append(f"- Total Files: {validation_results.get('total_files', 0)}")
    report.append(f"- Classes: {', '.join(validation_results.get('classes', []))}")
    report.append("")
    
    # Class balance
    report.append("-" * 80)
    report.append("Class Balance:")
    report.append(f"- Class Balance Valid: {'Yes' if validation_results.get('class_balance_valid', False) else 'No'}")
    
    if 'class_counts' in validation_results:
        for cls, count in validation_results['class_counts'].items():
            report.append(f"  - {cls}: {count} samples")
    
    if 'imbalance_ratio' in validation_results:
        report.append(f"- Imbalance Ratio: {validation_results['imbalance_ratio']:.2f}")
    
    report.append("")
    
    # Image properties
    report.append("-" * 80)
    report.append("Image Properties:")
    
    if 'dimensions' in validation_results:
        report.append("- Dimensions:")
        for (width, height), count in validation_results['dimensions'].most_common(5):
            report.append(f"  - {width}x{height}: {count} images")
        
        report.append(f"- Width Range: {validation_results.get('min_width', 'N/A')} - {validation_results.get('max_width', 'N/A')}")
        report.append(f"- Height Range: {validation_results.get('min_height', 'N/A')} - {validation_results.get('max_height', 'N/A')}")
    
    if 'channels' in validation_results:
        report.append("- Channels:")
        for channels, count in validation_results['channels'].most_common():
            report.append(f"  - {channels} channels: {count} images")
    
    if 'formats' in validation_results:
        report.append("- Formats:")
        for format_name, count in validation_results['formats'].most_common():
            report.append(f"  - {format_name}: {count} images")
    
    report.append("")
    
    # Warnings and errors
    if validation_results.get('warnings', []):
        report.append("-" * 80)
        report.append("Warnings:")
        for warning in validation_results['warnings']:
            report.append(f"- {warning}")
        report.append("")
    
    if validation_results.get('errors', []):
        report.append("-" * 80)
        report.append("Errors:")
        for error in validation_results['errors']:
            report.append(f"- {error}")
        report.append("")
    
    # Save report if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"Report saved to {output_path}")
    
    return '\n'.join(report)


def visualize_dataset_stats(validation_results):
    """
    Visualizza le statistiche del dataset.
    
    Args:
        validation_results (dict): Risultati della validazione
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Class distribution
    if 'class_counts' in validation_results:
        ax1 = fig.add_subplot(2, 2, 1)
        classes = list(validation_results['class_counts'].keys())
        counts = list(validation_results['class_counts'].values())
        
        ax1.bar(classes, counts)
        ax1.set_title('Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Samples')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels
        for i, count in enumerate(counts):
            ax1.text(i, count + 0.1, str(count), ha='center')
    
    # 2. Image dimensions
    if 'dimensions' in validation_results:
        ax2 = fig.add_subplot(2, 2, 2)
        dimensions = list(validation_results['dimensions'].keys())
        dim_counts = list(validation_results['dimensions'].values())
        
        # Sort by count
        sorted_dims = sorted(zip(dimensions, dim_counts), key=lambda x: x[1], reverse=True)
        dimensions = [f"{w}x{h}" for (w, h), _ in sorted_dims[:10]]  # Top 10
        dim_counts = [count for _, count in sorted_dims[:10]]
        
        ax2.bar(dimensions, dim_counts)
        ax2.set_title('Image Dimensions (Top 10)')
        ax2.set_xlabel('Dimensions (WxH)')
        ax2.set_ylabel('Number of Images')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Channel distribution
    if 'channels' in validation_results:
        ax3 = fig.add_subplot(2, 2, 3)
        channels = list(validation_results['channels'].keys())
        channel_counts = list(validation_results['channels'].values())
        
        ax3.bar(channels, channel_counts)
        ax3.set_title('Channel Distribution')
        ax3.set_xlabel('Number of Channels')
        ax3.set_ylabel('Number of Images')
        
        # Add count labels
        for i, count in enumerate(channel_counts):
            ax3.text(i, count + 0.1, str(count), ha='center')
    
    # 4. Format distribution
    if 'formats' in validation_results:
        ax4 = fig.add_subplot(2, 2, 4)
        formats = list(validation_results['formats'].keys())
        format_counts = list(validation_results['formats'].values())
        
        ax4.bar(formats, format_counts)
        ax4.set_title('Format Distribution')
        ax4.set_xlabel('Format')
        ax4.set_ylabel('Number of Images')
        
        # Add count labels
        for i, count in enumerate(format_counts):
            ax4.text(i, count + 0.1, str(count), ha='center')
    
    plt.tight_layout()
    plt.show()


def main():
    """Funzione principale per l'esecuzione come script."""
    parser = argparse.ArgumentParser(description='Inspect and validate a dataset for model evaluation')
    
    parser.add_argument('--dataset', type=str, required=True, 
                      help='Path to the dataset directory [NECESSARY]')
    parser.add_argument('--min-samples', type=int, default=10, 
                      help='Minimum number of samples per class [OPTIONAL, default=10]')
    parser.add_argument('--expected-dims', type=str, default=None,
                      help='Expected dimensions (WxH) for the model, e.g., "32x32" [OPTIONAL]')
    parser.add_argument('--max-channels', type=int, default=10,
                      help='Maximum number of channels supported [OPTIONAL, default=10]')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to save the inspection report [OPTIONAL]')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize dataset statistics [OPTIONAL, default=False]')
    
    args = parser.parse_args()
    
    # Parse expected dimensions if provided
    expected_dims = None
    if args.expected_dims:
        try:
            width, height = map(int, args.expected_dims.split('x'))
            expected_dims = (width, height)
        except:
            print(f"[ERROR] Invalid format for expected dimensions: {args.expected_dims}")
            print("Expected format: WIDTHxHEIGHT (e.g., 32x32)")
            return 1
    
    # Validate dataset
    is_valid, validation_results = validate_dataset(
        args.dataset,
        args.min_samples,
        expected_dims,
        args.max_channels
    )
    
    # Generate and print report
    report = generate_report(validation_results, args.output)
    print(report)
    
    # Visualize statistics if requested
    if args.visualize:
        visualize_dataset_stats(validation_results)
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
