#!/usr/bin/env python3
"""
Wavelet Scattering Transform Dataset Analyzer.

This module provides tools for analyzing datasets processed with Wavelet 
Scattering Transform, including statistical analysis and visualization.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from pathlib import Path
import torch


class WSTDatasetAnalyzer:
    """
    Analyzer for datasets processed with Wavelet Scattering Transform.
    
    This class provides methods for analyzing and visualizing statistics about
    wavelet scattering transform representations within datasets.
    """
    
    def __init__(self, dataset_path, output_dir=None):
        """
        Initialize the WST dataset analyzer.
        
        Args:
            dataset_path (str): Path to the WST processed dataset pickle file
            output_dir (str, optional): Directory to save analysis results
        """
        self.dataset_path = dataset_path
        
        # Create output directory if specified
        if output_dir:
            self.output_dir = Path(output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = Path("wst_analysis")
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Will store analysis results
        self.dataset = None
        self.analysis_results = None
    
    def load_dataset(self):
        """
        Load the WST processed dataset from the pickle file.
        
        Returns:
            dict: The loaded dataset
        """
        print(f"Loading dataset from: {self.dataset_path}")
        with open(self.dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        return self.dataset
    
    def analyze(self):
        """
        Analyze the WST dataset and generate statistics.
        
        Returns:
            dict: Analysis results
        """
        if self.dataset is None:
            self.load_dataset()
        
        print("\n" + "="*50)
        print("WST DATASET ANALYSIS")
        print("="*50)
        
        # 1. General Information
        print("\n1. GENERAL INFORMATION:")
        total_samples = len(self.dataset['samples'])
        num_classes = len(self.dataset['classes'])
        classes = self.dataset['classes']
        
        print(f"• Total samples: {total_samples}")
        print(f"• Number of classes: {num_classes}")
        print(f"• Classes: {classes}")
        
        # 2. Class Distribution
        print("\n2. CLASS DISTRIBUTION:")
        class_distribution = Counter([label for _, label in self.dataset['samples']])
        
        class_counts = {}
        for class_name in classes:
            class_idx = self.dataset['class_to_idx'][class_name]
            count = class_distribution[class_idx]
            percentage = (count / total_samples) * 100
            print(f"• {class_name}: {count} samples ({percentage:.1f}%)")
            class_counts[class_name] = count
        
        # 3. WST Representation Analysis
        print("\n3. WST REPRESENTATION ANALYSIS:")
        
        # Get sample representation for analysis
        sample_path = list(self.dataset['wavelet_representations'].keys())[0]
        sample_repr = self.dataset['wavelet_representations'][sample_path]
        
        # Determine representation dimensions and stats
        if isinstance(sample_repr, torch.Tensor):
            repr_shape = tuple(sample_repr.shape)
            repr_dtype = str(sample_repr.dtype)
            repr_min = float(sample_repr.min())
            repr_max = float(sample_repr.max())
            repr_mean = float(sample_repr.mean())
            repr_std = float(sample_repr.std())
        else:
            repr_shape = sample_repr.shape
            repr_dtype = str(sample_repr.dtype)
            repr_min = float(sample_repr.min())
            repr_max = float(sample_repr.max())
            repr_mean = float(sample_repr.mean())
            repr_std = float(sample_repr.std())
        
        print(f"• Representation shape: {repr_shape}")
        print(f"• Data type: {repr_dtype}")
        print(f"• Value range: [{repr_min:.3f}, {repr_max:.3f}]")
        print(f"• Mean: {repr_mean:.3f}")
        print(f"• Standard deviation: {repr_std:.3f}")
        
        # 4. Per-Class Analysis
        print("\n4. PER-CLASS ANALYSIS:")
        
        # Gather per-class statistics
        class_stats = {}
        for class_name in classes:
            class_idx = self.dataset['class_to_idx'][class_name]
            class_samples = [path for path, label in self.dataset['samples'] if label == class_idx]
            
            if not class_samples:
                continue
            
            # Collect representation statistics for this class
            class_reprs = []
            for path in class_samples:
                if path in self.dataset['wavelet_representations']:
                    repr_data = self.dataset['wavelet_representations'][path]
                    if isinstance(repr_data, torch.Tensor):
                        class_reprs.append(repr_data.cpu().numpy())
                    else:
                        class_reprs.append(repr_data)
            
            if not class_reprs:
                continue
                
            # Stack for easier analysis
            class_reprs = np.stack(class_reprs)
            
            # Calculate statistics
            class_stats[class_name] = {
                'count': len(class_reprs),
                'mean': np.mean(class_reprs),
                'std': np.std(class_reprs),
                'min': np.min(class_reprs),
                'max': np.max(class_reprs)
            }
            
            print(f"• {class_name}:")
            print(f"  - Count: {class_stats[class_name]['count']}")
            print(f"  - Mean: {class_stats[class_name]['mean']:.3f}")
            print(f"  - Std: {class_stats[class_name]['std']:.3f}")
            print(f"  - Range: [{class_stats[class_name]['min']:.3f}, {class_stats[class_name]['max']:.3f}]")
        
        # Store analysis results
        self.analysis_results = {
            'total_samples': total_samples,
            'num_classes': num_classes,
            'classes': classes,
            'class_distribution': class_distribution,
            'class_counts': class_counts,
            'repr_shape': repr_shape,
            'repr_dtype': repr_dtype,
            'repr_range': (repr_min, repr_max),
            'repr_mean': repr_mean,
            'repr_std': repr_std,
            'class_stats': class_stats,
            'sample_repr': sample_repr
        }
        
        return self.analysis_results
    
    def plot_analysis(self):
        """
        Create and save plots visualizing the analysis results.
        """
        if self.analysis_results is None:
            self.analyze()
        
        # 1. Class Distribution Plot
        plt.figure(figsize=(12, 6))
        classes = self.analysis_results['classes']
        counts = [self.analysis_results['class_counts'][cls] for cls in classes]
        
        sns.barplot(x=classes, y=counts)
        plt.title('Class Distribution in Dataset')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Number of Samples')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. WST Representation Visualizations
        sample_repr = self.analysis_results['sample_repr']
        
        if isinstance(sample_repr, torch.Tensor):
            sample_repr = sample_repr.cpu().numpy()
        
        # Determine dimensionality of the representation
        if len(sample_repr.shape) >= 3:
            # Multi-channel representation
            num_channels = sample_repr.shape[0]
            
            # Visualize first few channels/slices of the representation
            for c in range(min(3, num_channels)):
                if len(sample_repr.shape) == 3:
                    # Shape is like (C, H, W)
                    channel_data = sample_repr[c]
                elif len(sample_repr.shape) == 4:
                    # Shape is like (C, D, H, W), take first slice of D
                    channel_data = sample_repr[c, 0]
                else:
                    # Just use first indices for higher dimensions
                    channel_data = sample_repr[c, 0, 0]
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(channel_data, cmap='viridis')
                plt.title(f'WST Representation - Channel {c}')
                plt.tight_layout()
                
                plt.savefig(self.output_dir / f'wst_heatmap_channel_{c}.png', dpi=300, bbox_inches='tight')
                plt.close()
        else:
            # 1D representation, create histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(sample_repr.flatten(), kde=True)
            plt.title('Distribution of WST Coefficients')
            plt.xlabel('Coefficient Value')
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'wst_coefficient_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Per-Class Statistics Comparison
        if self.analysis_results['class_stats']:
            # Mean values comparison
            plt.figure(figsize=(12, 6))
            class_names = list(self.analysis_results['class_stats'].keys())
            means = [stats['mean'] for stats in self.analysis_results['class_stats'].values()]
            
            sns.barplot(x=class_names, y=means)
            plt.title('Mean WST Coefficient Value by Class')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'class_mean_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Standard deviation comparison
            plt.figure(figsize=(12, 6))
            stds = [stats['std'] for stats in self.analysis_results['class_stats'].values()]
            
            sns.barplot(x=class_names, y=stds)
            plt.title('Standard Deviation of WST Coefficients by Class')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'class_std_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\nAnalysis plots saved to: {self.output_dir}")
    
    def get_dataset_stats(self):
        """
        Return key statistics for use in other modules.
        
        Returns:
            dict: Dictionary of key dataset statistics
        """
        if self.analysis_results is None:
            self.analyze()
        
        return {
            'total_samples': self.analysis_results['total_samples'],
            'num_classes': self.analysis_results['num_classes'],
            'class_distribution': {
                cls: self.analysis_results['class_counts'][cls] 
                for cls in self.analysis_results['classes']
            },
            'representation_shape': self.analysis_results['repr_shape'],
            'mean': self.analysis_results['repr_mean'],
            'std': self.analysis_results['repr_std']
        }


def main():
    """Command line entry point for WST dataset analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='WST Dataset Analyzer')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to the WST processed dataset pickle file')
    parser.add_argument('--output', type=str, default='wst_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create and use the analyzer
    analyzer = WSTDatasetAnalyzer(args.dataset, args.output)
    analyzer.analyze()
    analyzer.plot_analysis()
    
    stats = analyzer.get_dataset_stats()
    print("\nDataset statistics summary:")
    for key, value in stats.items():
        if key != 'class_distribution':
            print(f"• {key}: {value}")
        else:
            print("• Class distribution:")
            for cls, count in value.items():
                print(f"  - {cls}: {count}")


if __name__ == "__main__":
    main()