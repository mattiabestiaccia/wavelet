#!/usr/bin/env python3
"""
Dataset Inspector for Wavelet Scattering Transform.

This script analyzes a dataset to ensure it's correctly structured for the wavelet model.
It checks image dimensions, formats, class structure, and other parameters to validate
the dataset for proper evaluation.

Usage:
    python script/utility/dataset_inspector.py --dataset /path/to/dataset

Author: Claude
Date: 2025-04-01
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

# Add the main directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import wavelet_lib modules
from wavelet_lib.datasets import BalancedDataset, get_default_transform
from wavelet_lib.base import Config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inspect and validate a dataset for WST evaluation')
    
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--min-samples', type=int, default=10, 
                      help='Minimum number of samples per class (default: 10)')
    parser.add_argument('--expected-dims', type=str, default=None,
                      help='Expected dimensions (WxH) for the model, e.g., "32x32"')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to save the inspection report (default: dataset_report.txt)')
    
    return parser.parse_args()

def inspect_dataset_structure(dataset_path):
    """Inspect the dataset directory structure."""
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'class_counts': {},
        'total_samples': 0,
        'classes': []
    }
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_path):
        results['is_valid'] = False
        results['errors'].append(f"Dataset path '{dataset_path}' does not exist")
        return results
    
    # Check if it's a directory
    if not os.path.isdir(dataset_path):
        results['is_valid'] = False
        results['errors'].append(f"'{dataset_path}' is not a directory")
        return results
    
    # List directories (classes)
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Check if there are any class directories
    if not class_dirs:
        results['is_valid'] = False
        results['errors'].append(f"No class directories found in '{dataset_path}'")
        return results
    
    # Store class names
    results['classes'] = sorted(class_dirs)
    
    # Count samples in each class
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    for cls in class_dirs:
        cls_path = os.path.join(dataset_path, cls)
        files = [f for f in os.listdir(cls_path) 
                if os.path.isfile(os.path.join(cls_path, f)) and 
                os.path.splitext(f)[1].lower() in allowed_extensions]
        
        results['class_counts'][cls] = len(files)
        results['total_samples'] += len(files)
    
    return results

def check_image_properties(dataset_path, min_samples=10):
    """Check properties of the images in the dataset."""
    results = {
        'dimensions': defaultdict(int),
        'channels': defaultdict(int),
        'formats': defaultdict(int),
        'problematic_files': [],
        'sizes': [],
        'warnings': [],
        'errors': []
    }
    
    # Get class directories
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # Total files to process for progress bar
    total_files = sum(len([f for f in os.listdir(os.path.join(dataset_path, cls)) 
                         if os.path.isfile(os.path.join(dataset_path, cls, f)) and 
                         os.path.splitext(f)[1].lower() in allowed_extensions])
                    for cls in class_dirs)
    
    with tqdm(total=total_files, desc="Inspecting images") as pbar:
        for cls in class_dirs:
            cls_path = os.path.join(dataset_path, cls)
            files = [f for f in os.listdir(cls_path) 
                    if os.path.isfile(os.path.join(cls_path, f)) and 
                    os.path.splitext(f)[1].lower() in allowed_extensions]
            
            # Check if class has enough samples
            if len(files) < min_samples:
                results['warnings'].append(f"Class '{cls}' has only {len(files)} samples, which is below the recommended minimum of {min_samples}")
            
            for file in files:
                file_path = os.path.join(cls_path, file)
                try:
                    with Image.open(file_path) as img:
                        # Record image dimensions
                        dimensions = f"{img.width}x{img.height}"
                        results['dimensions'][dimensions] += 1
                        results['sizes'].append((img.width, img.height))
                        
                        # Record image format
                        results['formats'][img.format] += 1
                        
                        # Record number of channels
                        if img.mode == 'RGB':
                            channels = 3
                        elif img.mode == 'RGBA':
                            channels = 4
                        elif img.mode == 'L':
                            channels = 1
                        else:
                            channels = img.mode
                        results['channels'][channels] += 1
                except (UnidentifiedImageError, OSError, IOError) as e:
                    results['problematic_files'].append((file_path, str(e)))
                    results['errors'].append(f"Problem loading '{file_path}': {str(e)}")
                except Exception as e:
                    results['problematic_files'].append((file_path, str(e)))
                    results['errors'].append(f"Unexpected error with '{file_path}': {str(e)}")
                
                pbar.update(1)
    
    return results

def validate_for_model(dataset_path, expected_dims=None):
    """Validate if the dataset is suitable for the wavelet model."""
    results = {
        'is_suitable': True,
        'recommendations': [],
        'transformations_needed': [],
        'warnings': []
    }
    
    # Get dataset metrics
    structure = inspect_dataset_structure(dataset_path)
    image_props = check_image_properties(dataset_path)
    
    # Check if dataset structure is valid
    if not structure['is_valid']:
        results['is_suitable'] = False
        results['warnings'].extend(structure['errors'])
        return results
    
    # Check for problematic files
    if image_props['problematic_files']:
        results['warnings'].append(f"Found {len(image_props['problematic_files'])} problematic files")
    
    # Check if dimensions are consistent
    if len(image_props['dimensions']) > 1:
        results['is_suitable'] = False
        results['warnings'].append(f"Inconsistent image dimensions: {dict(image_props['dimensions'])}")
        results['transformations_needed'].append("Resize all images to the same dimensions")
    
    # Check if expected dimensions are provided and match
    if expected_dims:
        expected_w, expected_h = map(int, expected_dims.split('x'))
        current_dims = list(image_props['dimensions'].keys())[0] if image_props['dimensions'] else None
        
        if current_dims:
            current_w, current_h = map(int, current_dims.split('x'))
            if current_w != expected_w or current_h != expected_h:
                results['is_suitable'] = False
                results['warnings'].append(f"Images are {current_dims}, but model expects {expected_dims}")
                results['transformations_needed'].append(f"Resize all images to {expected_dims}")
    
    # Check if number of channels is consistent and supported
    if len(image_props['channels']) > 1:
        results['warnings'].append(f"Inconsistent channel counts: {dict(image_props['channels'])}")
        results['transformations_needed'].append("Convert all images to the same color mode (preferably RGB)")
    
    # Typical expectation for wavelet model is RGB images
    if 3 not in image_props['channels']:
        results['warnings'].append("No RGB images found, wavelet model typically expects RGB (3-channel) images")
        results['transformations_needed'].append("Convert images to RGB format")
    
    # Check class balance
    class_counts = structure['class_counts']
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    
    if max_count > min_count * 2:
        results['warnings'].append(f"Dataset is imbalanced. Min samples: {min_count}, Max samples: {max_count}")
        results['recommendations'].append("Consider using --balance option when loading dataset")
    
    # Minimum number of classes for meaningful classification
    if len(class_counts) < 2:
        results['is_suitable'] = False
        results['warnings'].append(f"Found only {len(class_counts)} classes, at least 2 are needed for classification")
    
    return results

def validate_with_dataset_loader(dataset_path):
    """Try loading the dataset with the BalancedDataset class to check compatibility."""
    results = {
        'success': False,
        'errors': [],
        'dataset_size': 0,
        'classes': []
    }
    
    try:
        # Try to load the dataset
        transform = get_default_transform()
        dataset = BalancedDataset(dataset_path, transform=transform)
        
        # Record dataset information
        results['success'] = True
        results['dataset_size'] = len(dataset)
        results['classes'] = dataset.classes
        
        # Check a few samples
        try:
            sample_image, sample_label = dataset[0]
            results['sample_shape'] = tuple(sample_image.shape)
            results['sample_type'] = type(sample_image).__name__
        except Exception as e:
            results['errors'].append(f"Error accessing sample: {str(e)}")
    
    except Exception as e:
        results['errors'].append(f"Error loading dataset: {str(e)}")
        results['errors'].append(traceback.format_exc())
    
    return results

def plot_dataset_stats(structure, image_props, output_path):
    """Plot statistics about the dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot class distribution
    class_names = list(structure['class_counts'].keys())
    class_counts = list(structure['class_counts'].values())
    
    axes[0, 0].bar(class_names, class_counts)
    axes[0, 0].set_title("Class Distribution")
    axes[0, 0].set_ylabel("Number of Samples")
    axes[0, 0].set_xlabel("Class")
    for tick in axes[0, 0].get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')
    
    # Plot image formats
    format_names = list(image_props['formats'].keys())
    format_counts = list(image_props['formats'].values())
    
    axes[0, 1].pie(format_counts, labels=format_names, autopct='%1.1f%%')
    axes[0, 1].set_title("Image Formats")
    
    # Plot image dimensions
    dim_names = list(image_props['dimensions'].keys())
    dim_counts = list(image_props['dimensions'].values())
    
    if len(dim_names) > 10:
        # Trim to top 10 if there are too many
        dim_data = sorted(zip(dim_names, dim_counts), key=lambda x: x[1], reverse=True)[:10]
        dim_names, dim_counts = zip(*dim_data)
        axes[1, 0].set_title("Top 10 Image Dimensions")
    else:
        axes[1, 0].set_title("Image Dimensions")
    
    axes[1, 0].bar(dim_names, dim_counts)
    axes[1, 0].set_ylabel("Number of Images")
    axes[1, 0].set_xlabel("Dimensions (WxH)")
    for tick in axes[1, 0].get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')
    
    # Plot channel distribution
    channel_names = [f"{ch} channels" if isinstance(ch, int) else ch for ch in image_props['channels'].keys()]
    channel_counts = list(image_props['channels'].values())
    
    axes[1, 1].pie(channel_counts, labels=channel_names, autopct='%1.1f%%')
    axes[1, 1].set_title("Image Channels")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(structure, image_props, model_validation, loader_validation, args):
    """Generate a comprehensive report of the dataset inspection."""
    report = []
    
    report.append("=" * 80)
    report.append("DATASET INSPECTION REPORT")
    report.append("=" * 80)
    report.append(f"Dataset path: {args.dataset}")
    report.append(f"Date: {os.popen('date').read().strip()}")
    report.append("=" * 80)
    
    # Dataset structure summary
    report.append("\nDATASET STRUCTURE:")
    report.append(f"  • Total classes: {len(structure['classes'])}")
    report.append(f"  • Total samples: {structure['total_samples']}")
    report.append("\nCLASS DISTRIBUTION:")
    for cls, count in structure['class_counts'].items():
        report.append(f"  • {cls}: {count} samples")
    
    # Image properties
    report.append("\nIMAGE PROPERTIES:")
    report.append(f"  • Unique dimensions: {len(image_props['dimensions'])}")
    for dim, count in image_props['dimensions'].items():
        report.append(f"    - {dim}: {count} images")
    
    report.append(f"  • Image formats: {dict(image_props['formats'])}")
    report.append(f"  • Channel counts: {dict(image_props['channels'])}")
    
    # Model validation
    report.append("\nMODEL COMPATIBILITY:")
    report.append(f"  • Suitable for model: {'YES' if model_validation['is_suitable'] else 'NO'}")
    
    if model_validation['warnings']:
        report.append("  • Warnings:")
        for warning in model_validation['warnings']:
            report.append(f"    - {warning}")
    
    if model_validation['transformations_needed']:
        report.append("  • Recommended transformations:")
        for transform in model_validation['transformations_needed']:
            report.append(f"    - {transform}")
    
    if model_validation['recommendations']:
        report.append("  • General recommendations:")
        for rec in model_validation['recommendations']:
            report.append(f"    - {rec}")
    
    # Dataset loader validation
    report.append("\nDATASET LOADER TEST:")
    report.append(f"  • Successfully loaded: {'YES' if loader_validation['success'] else 'NO'}")
    
    if loader_validation['success']:
        report.append(f"  • Dataset size: {loader_validation['dataset_size']} samples")
        report.append(f"  • Classes: {loader_validation['classes']}")
        if 'sample_shape' in loader_validation:
            report.append(f"  • Sample shape: {loader_validation['sample_shape']}")
    else:
        report.append("  • Errors:")
        for error in loader_validation['errors']:
            report.append(f"    - {error}")
    
    # Problematic files
    if image_props['problematic_files']:
        report.append("\nPROBLEMATIC FILES:")
        for file_path, error in image_props['problematic_files'][:10]:  # Limit to first 10
            report.append(f"  • {file_path}: {error}")
        
        if len(image_props['problematic_files']) > 10:
            report.append(f"  • ... and {len(image_props['problematic_files']) - 10} more")
    
    # Final verdict
    report.append("\nFINAL VERDICT:")
    if model_validation['is_suitable'] and loader_validation['success']:
        report.append("✅ Dataset is ready for wavelet model evaluation.")
    elif loader_validation['success']:
        report.append("⚠️ Dataset can be loaded but has issues that may affect model performance.")
    else:
        report.append("❌ Dataset has critical issues and cannot be used for model evaluation.")
    
    # Add commands to fix issues
    if model_validation['transformations_needed']:
        report.append("\nSUGGESTED FIXES:")
        if "Resize all images to the same dimensions" in model_validation['transformations_needed']:
            report.append("  • To resize all images to 32x32:")
            report.append("    python -c \"import os, sys; from PIL import Image; [Image.open(os.path.join(root, f)).resize((32, 32)).save(os.path.join(root, f)) for root, _, files in os.walk(sys.argv[1]) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]\" \"" + args.dataset + "\"")
        
        if "Convert all images to the same color mode" in model_validation['transformations_needed']:
            report.append("  • To convert all images to RGB:")
            report.append("    python -c \"import os, sys; from PIL import Image; [Image.open(os.path.join(root, f)).convert('RGB').save(os.path.join(root, f)) for root, _, files in os.walk(sys.argv[1]) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]\" \"" + args.dataset + "\"")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)

def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"Wavelet Dataset Inspector")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Expected dimensions: {args.expected_dims or 'Not specified'}")
    print(f"Minimum samples per class: {args.min_samples}")
    print(f"{'='*80}\n")
    
    # Set output filename if not provided
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.dataset), "dataset_report.txt")
    
    # Inspect dataset structure
    print("Step 1: Inspecting dataset structure...")
    structure = inspect_dataset_structure(args.dataset)
    
    if not structure['is_valid']:
        print("\n❌ Dataset structure is invalid:")
        for error in structure['errors']:
            print(f"  • {error}")
        sys.exit(1)
    
    print(f"✅ Found {len(structure['classes'])} classes with {structure['total_samples']} total samples")
    
    # Check image properties
    print("\nStep 2: Checking image properties...")
    image_props = check_image_properties(args.dataset, args.min_samples)
    
    # Validate for model
    print("\nStep 3: Validating for wavelet model...")
    model_validation = validate_for_model(args.dataset, args.expected_dims)
    
    # Try loading with BalancedDataset
    print("\nStep 4: Testing dataset loader...")
    loader_validation = validate_with_dataset_loader(args.dataset)
    
    # Plot statistics
    stats_path = os.path.join(os.path.dirname(args.output), "dataset_stats.png")
    print("\nStep 5: Generating statistics chart...")
    try:
        plot_dataset_stats(structure, image_props, stats_path)
        print(f"✅ Statistics chart saved to: {stats_path}")
    except Exception as e:
        print(f"❌ Error generating statistics chart: {str(e)}")
    
    # Generate final report
    print("\nStep 6: Generating final report...")
    report = generate_report(structure, image_props, model_validation, loader_validation, args)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {args.output}")
    
    # Print final verdict
    print("\nFINAL VERDICT:")
    if model_validation['is_suitable'] and loader_validation['success']:
        print("✅ Dataset is ready for wavelet model evaluation.")
    elif loader_validation['success']:
        print("⚠️ Dataset can be loaded but has issues that may affect model performance.")
    else:
        print("❌ Dataset has critical issues and cannot be used for model evaluation.")
    
    if model_validation['warnings']:
        print("\nWarnings:")
        for warning in model_validation['warnings'][:5]:
            print(f"  • {warning}")
        if len(model_validation['warnings']) > 5:
            print(f"  • ... and {len(model_validation['warnings']) - 5} more (see report)")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()