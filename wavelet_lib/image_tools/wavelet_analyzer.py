#!/usr/bin/env python3
"""
Wavelet Analysis Module - Tools for wavelet-based image analysis.

This module provides utilities for analyzing images using various wavelet transforms,
including the Discrete Wavelet Transform (DWT) and Wavelet Scattering Transform (WST).
It supports multi-channel images and various visualization methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2
from pathlib import Path
import os
import torch
from PIL import Image
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib import gridspec

# Try to import kymatio if available
try:
    from kymatio.torch import Scattering2D
    KYMATIO_AVAILABLE = True
except ImportError:
    KYMATIO_AVAILABLE = False


class WaveletAnalyzer:
    """
    Analyzer for wavelet transformations of images.
    
    This class provides methods for analyzing images using both discrete
    wavelet transform and wavelet scattering transform, with visualization options.
    """
    
    def __init__(self, max_level=3, wavelet='haar', J=2, L=8, output_dir=None):
        """
        Initialize the wavelet analyzer.
        
        Args:
            max_level (int): Maximum decomposition level for DWT
            wavelet (str): Wavelet type for DWT (e.g., 'haar', 'db4', 'sym4')
            J (int): Number of scales for Wavelet Scattering Transform
            L (int): Number of angles for Wavelet Scattering Transform
            output_dir (str, optional): Directory to save output images
        """
        self.max_level = max_level
        self.wavelet = wavelet
        self.J = J
        self.L = L
        
        # Create output directory if specified
        if output_dir:
            self.output_dir = Path(output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = None
    
    def load_image(self, image_path, convert_mode='L'):
        """
        Load an image file.
        
        Args:
            image_path (str): Path to the image file
            convert_mode (str): PIL image conversion mode ('L' for grayscale, 'RGB' for color)
            
        Returns:
            numpy.ndarray: Loaded image array
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Check file extension
            if image_path.suffix.lower() in ['.tif', '.tiff']:
                # Handle TIFF files differently to support multiband images
                import tifffile as tiff
                img = tiff.imread(str(image_path))
                # If we have a multiband image but requested grayscale, use first band
                if len(img.shape) > 2 and convert_mode == 'L':
                    img = img[:, :, 0]
            else:
                # Use PIL for other formats
                img = Image.open(image_path).convert(convert_mode)
                img = np.array(img)
            
            print(f"Loaded image: {image_path.name}, Shape: {img.shape}")
            return img
            
        except Exception as e:
            raise IOError(f"Error loading image {image_path}: {str(e)}")
    
    def analyze_dwt(self, image, max_level=None, wavelet=None, plot=True, save_path=None):
        """
        Perform discrete wavelet transform analysis on the image.
        
        Args:
            image (numpy.ndarray): Input image array
            max_level (int, optional): Maximum decomposition level, defaults to self.max_level
            wavelet (str, optional): Wavelet type, defaults to self.wavelet
            plot (bool): Whether to display plots
            save_path (str, optional): Path to save the output
            
        Returns:
            dict: Dictionary containing wavelet coefficients
        """
        if max_level is None:
            max_level = self.max_level
            
        if wavelet is None:
            wavelet = self.wavelet
        
        # Ensure image is 2D (convert if RGB)
        if len(image.shape) > 2:
            print("Converting multi-channel image to grayscale for DWT")
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                # Use first channel for other types
                image = image[:, :, 0]
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(image, wavelet=wavelet, level=max_level)
        LL = coeffs[0]           # Approximation coefficients at the last level
        details = coeffs[1:]     # List of tuples: (LH, HL, HH) for each level
        
        # Store results
        results = {
            'LL': LL,
            'details': details,
            'max_level': max_level,
            'wavelet': wavelet
        }
        
        # Plot results if requested
        if plot:
            self._plot_dwt_results(image, LL, details, max_level, save_path)
        
        return results
    
    def _plot_dwt_results(self, original, LL, details, max_level, save_path=None):
        """
        Plot DWT decomposition results.
        
        Args:
            original (numpy.ndarray): Original image
            LL (numpy.ndarray): Approximation coefficients
            details (list): Detail coefficients
            max_level (int): Decomposition level
            save_path (str, optional): Path to save the output
        """
        # Plot original image
        plt.figure(figsize=(6, 6))
        plt.imshow(original, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        
        if save_path:
            plt.savefig(f"{save_path}_original.png", dpi=300, bbox_inches='tight')
        if self.output_dir:
            plt.savefig(self.output_dir / "dwt_original.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot approximation at the last level
        plt.figure(figsize=(6, 6))
        plt.imshow(LL, cmap='gray')
        plt.title(f"Approximation (LL) - Level {max_level}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(f"{save_path}_LL.png", dpi=300, bbox_inches='tight')
        if self.output_dir:
            plt.savefig(self.output_dir / "dwt_LL.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot details for each level
        for i, (LH, HL, HH) in enumerate(details, start=1):
            level = max_level - i + 1
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(LH, cmap='gray')
            axes[0].set_title(f"Horizontal Details (LH) - Level {level}")
            axes[0].axis('off')
            
            axes[1].imshow(HL, cmap='gray')
            axes[1].set_title(f"Vertical Details (HL) - Level {level}")
            axes[1].axis('off')
            
            axes[2].imshow(HH, cmap='gray')
            axes[2].set_title(f"Diagonal Details (HH) - Level {level}")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_level_{level}.png", dpi=300, bbox_inches='tight')
            if self.output_dir:
                plt.savefig(self.output_dir / f"dwt_level_{level}.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def analyze_wst(self, image, J=None, L=None, plot=True, save_path=None):
        """
        Perform wavelet scattering transform analysis.
        
        Args:
            image (numpy.ndarray): Input image array
            J (int, optional): Number of scales, defaults to self.J
            L (int, optional): Number of angles, defaults to self.L
            plot (bool): Whether to display plots
            save_path (str, optional): Path to save the output
            
        Returns:
            numpy.ndarray: Scattering coefficients
        """
        if not KYMATIO_AVAILABLE:
            raise ImportError("Kymatio package is required for Wavelet Scattering Transform. "
                             "Please install it with 'pip install kymatio'.")
        
        if J is None:
            J = self.J
            
        if L is None:
            L = self.L
        
        # Process image to prepare for WST
        # Normalize to [0, 1] if needed
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Handle multichannel images
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image: process each channel separately
            red = image[:, :, 0]
            green = image[:, :, 1]
            blue = image[:, :, 2]
            
            scattering = Scattering2D(J=J, L=L, shape=red.shape)
            
            # Calculate coefficients for each channel
            red_coeffs = self._compute_channel_coefficients(red, scattering)
            green_coeffs = self._compute_channel_coefficients(green, scattering)
            blue_coeffs = self._compute_channel_coefficients(blue, scattering)
            
            # Store results
            results = {
                'red_coeffs': red_coeffs,
                'green_coeffs': green_coeffs,
                'blue_coeffs': blue_coeffs,
                'J': J,
                'L': L
            }
            
            # Plot results if requested
            if plot:
                self._plot_wst_results_multichannel(image, red_coeffs, green_coeffs, blue_coeffs, save_path)
        else:
            # Grayscale image
            if len(image.shape) == 3:
                image = image[:, :, 0]  # Take first channel
                
            scattering = Scattering2D(J=J, L=L, shape=image.shape)
            coeffs = self._compute_channel_coefficients(image, scattering)
            
            # Store results
            results = {
                'coeffs': coeffs,
                'J': J,
                'L': L
            }
            
            # Plot results if requested
            if plot:
                self._plot_wst_results(image, coeffs, save_path)
        
        return results
    
    def _compute_channel_coefficients(self, channel, scattering):
        """
        Compute scattering coefficients for a single channel.
        
        Args:
            channel (numpy.ndarray): Single image channel
            scattering (Scattering2D): Configured scattering transform
            
        Returns:
            numpy.ndarray: Scattering coefficients
        """
        # Convert to tensor format
        img_tensor = torch.from_numpy(channel).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        img_tensor = img_tensor.contiguous()
        
        # Compute scattering transform
        scattering_coeffs = scattering(img_tensor)
        return scattering_coeffs.squeeze().numpy()
    
    def _plot_wst_results(self, original, coeffs, save_path=None):
        """
        Plot WST decomposition results for grayscale image.
        
        Args:
            original (numpy.ndarray): Original image
            coeffs (numpy.ndarray): Scattering coefficients
            save_path (str, optional): Path to save the output
        """
        # Plot original image
        plt.figure(figsize=(6, 6))
        plt.imshow(original, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        
        if save_path:
            plt.savefig(f"{save_path}_original.png", dpi=300, bbox_inches='tight')
        if self.output_dir:
            plt.savefig(self.output_dir / "wst_original.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot a subset of coefficients
        num_coeffs = min(10, len(coeffs))
        fig, axes = plt.subplots(2, 5, figsize=(18, 9))
        
        for i in range(num_coeffs):
            ax = axes[i // 5, i % 5]
            im = ax.imshow(coeffs[i], cmap="viridis", aspect='auto')
            ax.axis("off")
            ax.set_title(f"Coeff {i+1}")
        
        # Add colorbar
        fig.suptitle("Wavelet Scattering Coefficients", fontsize=16)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Coefficient Intensity')
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        if save_path:
            plt.savefig(f"{save_path}_coeffs.png", dpi=300, bbox_inches='tight')
        if self.output_dir:
            plt.savefig(self.output_dir / "wst_coeffs.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_wst_results_multichannel(self, original, red_coeffs, green_coeffs, blue_coeffs, save_path=None):
        """
        Plot WST decomposition results for RGB image.
        
        Args:
            original (numpy.ndarray): Original RGB image
            red_coeffs (numpy.ndarray): Red channel scattering coefficients
            green_coeffs (numpy.ndarray): Green channel scattering coefficients
            blue_coeffs (numpy.ndarray): Blue channel scattering coefficients
            save_path (str, optional): Path to save the output
        """
        # Plot original image
        plt.figure(figsize=(8, 8))
        plt.imshow(original)
        plt.title("Original RGB Image")
        plt.axis('off')
        
        if save_path:
            plt.savefig(f"{save_path}_original.png", dpi=300, bbox_inches='tight')
        if self.output_dir:
            plt.savefig(self.output_dir / "wst_rgb_original.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Define function for plotting channel coefficients
        def plot_channel_coeffs(coeffs, channel_name, color_map):
            num_coeffs = min(5, len(coeffs))
            fig, axes = plt.subplots(1, num_coeffs, figsize=(18, 4))
            fig.suptitle(f"{channel_name} Channel Scattering Coefficients", fontsize=16)
            
            for i in range(num_coeffs):
                ax = axes[i]
                im = ax.imshow(coeffs[i], cmap=color_map, aspect='auto')
                ax.axis("off")
                ax.set_title(f"Coeff {i+1}")
            
            # Add colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax, label='Coefficient Intensity')
            
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
            
            if save_path:
                plt.savefig(f"{save_path}_{channel_name.lower()}_coeffs.png", dpi=300, bbox_inches='tight')
            if self.output_dir:
                plt.savefig(self.output_dir / f"wst_{channel_name.lower()}_coeffs.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot each channel
        plot_channel_coeffs(red_coeffs, "Red", "Reds")
        plot_channel_coeffs(green_coeffs, "Green", "Greens")
        plot_channel_coeffs(blue_coeffs, "Blue", "Blues")
    
    def visualize_wst_disk(self, image, J=None, L=None, save_path=None):
        """
        Visualize WST coefficients within a disk, following the representation from
        'Invariant Scattering Convolution Networks' by Bruna and Mallat.
        
        Args:
            image (numpy.ndarray): Input image
            J (int, optional): Number of scales, defaults to self.J
            L (int, optional): Number of angles, defaults to self.L
            save_path (str, optional): Path to save the output
            
        Returns:
            dict: Dictionary with visualization details
        """
        if not KYMATIO_AVAILABLE:
            raise ImportError("Kymatio package is required for Wavelet Scattering Transform. "
                             "Please install it with 'pip install kymatio'.")
            
        if J is None:
            J = self.J
            
        if L is None:
            L = self.L
        
        # Ensure image is of appropriate size and channels
        if len(image.shape) == 3:
            # Convert RGB to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize to ensure appropriate dimensions for WST
        image = cv2.resize(image, (32, 32))
        
        # Normalize image
        image_tensor = image.astype(np.float32) / 255.
        
        # Configure and compute scattering transform
        scattering = Scattering2D(J=J, shape=image.shape, L=L, max_order=2, frontend='numpy')
        scat_coeffs = scattering(image_tensor)
        
        # Invert colors for better visualization
        scat_coeffs = -scat_coeffs
        
        # Separate coefficients by order
        len_order_1 = J*L
        scat_coeffs_order_1 = scat_coeffs[1:1+len_order_1, :, :]
        norm_order_1 = mpl.colors.Normalize(scat_coeffs_order_1.min(), scat_coeffs_order_1.max(), clip=True)
        mapper_order_1 = cm.ScalarMappable(norm=norm_order_1, cmap="gray")
        
        len_order_2 = (J*(J-1)//2)*(L**2)
        scat_coeffs_order_2 = scat_coeffs[1+len_order_1:, :, :]
        norm_order_2 = mpl.colors.Normalize(scat_coeffs_order_2.min(), scat_coeffs_order_2.max(), clip=True)
        mapper_order_2 = cm.ScalarMappable(norm=norm_order_2, cmap="gray")
        
        # Retrieve spatial size
        window_rows, window_columns = scat_coeffs.shape[1:]
        
        # Display original image
        plt.figure(figsize=(10, 8))
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        if save_path:
            plt.savefig(f"{save_path}_original.png", dpi=300, bbox_inches='tight')
        if self.output_dir:
            plt.savefig(self.output_dir / "wst_disk_original.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create visualization
        fig = plt.figure(figsize=(47, 15))
        gs = gridspec.GridSpec(1, 3, wspace=0.1)
        gs_order_1 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[1])
        gs_order_2 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[2])
        
        # Add title
        fig.suptitle('Wavelet Scattering Coefficients Visualization', fontsize=20)
        
        # Add column titles
        ax1 = plt.subplot(gs[0])
        ax1.set_title('Zero-Order Scattering Coefficients', fontsize=16)
        
        ax2 = plt.subplot(gs[1])
        ax2.set_title('First-Order Scattering Coefficients', fontsize=16)
        
        ax3 = plt.subplot(gs[2])
        ax3.set_title('Second-Order Scattering Coefficients', fontsize=16)
        
        # Plot zero-order coefficients (original image)
        ax = plt.subplot(gs[0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image, cmap='gray', interpolation='nearest', aspect='auto')
        ax.axis('off')
        
        # Plot first-order coefficients
        ax = plt.subplot(gs[1])
        ax.set_xticks([])
        ax.set_yticks([])
        
        l_offset = int(L - L / 2 - 1)  # Follow same ordering as Kymatio
        
        for row in range(window_rows):
            for column in range(window_columns):
                ax = fig.add_subplot(gs_order_1[row, column], projection='polar')
                ax.axis('off')
                coefficients = scat_coeffs_order_1[:, row, column]
                for j in range(J):
                    for l in range(L):
                        coeff = coefficients[l + j * L]
                        color = mapper_order_1.to_rgba(coeff)
                        angle = (l_offset - l) * np.pi / L
                        radius = 2 ** (-j - 1)
                        ax.bar(x=angle,
                              height=radius,
                              width=np.pi / L,
                              bottom=radius,
                              color=color)
                        ax.bar(x=angle + np.pi,
                              height=radius,
                              width=np.pi / L,
                              bottom=radius,
                              color=color)
        
        # Plot second-order coefficients
        ax = plt.subplot(gs[2])
        ax.set_xticks([])
        ax.set_yticks([])
        
        for row in range(window_rows):
            for column in range(window_columns):
                ax = fig.add_subplot(gs_order_2[row, column], projection='polar')
                ax.axis('off')
                coefficients = scat_coeffs_order_2[:, row, column]
                for j1 in range(J - 1):
                    for j2 in range(j1 + 1, J):
                        for l1 in range(L):
                            for l2 in range(L):
                                coeff_index = l1 * L * (J - j1 - 1) + l2 + L * (j2 - j1 - 1) + (L ** 2) * \
                                            (j1 * (J - 1) - j1 * (j1 - 1) // 2)
                                coeff = coefficients[coeff_index]
                                color = mapper_order_2.to_rgba(coeff)
                                angle = (l_offset - l1) * np.pi / L + (L // 2 - l2 - 0.5) * np.pi / (L ** 2)
                                radius = 2 ** (-j1 - 1)
                                ax.bar(x=angle,
                                      height=radius / 2 ** (J - 2 - j1),
                                      width=np.pi / L ** 2,
                                      bottom=radius + (radius / 2 ** (J - 2 - j1)) * (J - j2 - 1),
                                      color=color)
                                ax.bar(x=angle + np.pi,
                                      height=radius / 2 ** (J - 2 - j1),
                                      width=np.pi / L ** 2,
                                      bottom=radius + (radius / 2 ** (J - 2 - j1)) * (J - j2 - 1),
                                      color=color)
        
        if save_path:
            plt.savefig(f"{save_path}_disk.png", dpi=300, bbox_inches='tight')
        if self.output_dir:
            plt.savefig(self.output_dir / "wst_disk.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'scat_coeffs': scat_coeffs,
            'order1_coeffs': scat_coeffs_order_1,
            'order2_coeffs': scat_coeffs_order_2
        }
    
    def extract_features(self, image, method='wst', resize=None, J=None, L=None):
        """
        Extract features from an image using wavelet transforms.
        
        Args:
            image (numpy.ndarray): Input image
            method (str): Method to use ('dwt' or 'wst')
            resize (tuple, optional): Size to resize image to (width, height)
            J (int, optional): Number of scales for WST
            L (int, optional): Number of angles for WST
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Resize image if needed
        if resize is not None:
            image = cv2.resize(image, resize)
        
        if method.lower() == 'dwt':
            # Use DWT for feature extraction
            if len(image.shape) > 2:
                # Convert to grayscale for DWT
                if image.shape[2] == 3:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    image_gray = image[:, :, 0]
            else:
                image_gray = image
            
            # Perform DWT
            coeffs = pywt.wavedec2(image_gray, wavelet=self.wavelet, level=self.max_level)
            
            # Extract features from the coefficients
            # We'll use the approximation coefficients and statistical measures of the detail coefficients
            features = []
            
            # Add approximation coefficients (downsampled)
            LL = coeffs[0]
            LL_downsampled = cv2.resize(LL, (8, 8)).flatten()
            features.extend(LL_downsampled)
            
            # Add statistical measures of detail coefficients
            for detail_level in coeffs[1:]:
                for detail_type in detail_level:
                    features.append(np.mean(detail_type))
                    features.append(np.std(detail_type))
                    features.append(np.max(detail_type))
            
            return np.array(features)
            
        elif method.lower() == 'wst':
            # Use WST for feature extraction
            if not KYMATIO_AVAILABLE:
                raise ImportError("Kymatio package is required for Wavelet Scattering Transform.")
            
            # Set parameters
            if J is None:
                J = self.J
            if L is None:
                L = self.L
            
            # Normalize image
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # Handle different channels
            features = []
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Process each channel
                for c in range(3):
                    channel = image[:, :, c]
                    scattering = Scattering2D(J=J, L=L, shape=channel.shape)
                    img_tensor = torch.from_numpy(channel).unsqueeze(0).unsqueeze(0)
                    coeffs = scattering(img_tensor).squeeze().cpu().numpy()
                    
                    # Average pooling across spatial dimensions
                    channel_features = np.mean(coeffs, axis=(1, 2))
                    features.extend(channel_features)
            else:
                # Grayscale image
                if len(image.shape) == 3:
                    image = image[:, :, 0]
                
                scattering = Scattering2D(J=J, L=L, shape=image.shape)
                img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
                coeffs = scattering(img_tensor).squeeze().cpu().numpy()
                
                # Average pooling across spatial dimensions
                features = np.mean(coeffs, axis=(1, 2))
            
            return np.array(features)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'dwt' or 'wst'.")


def main():
    """Command line entry point for wavelet analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wavelet Analysis Tool')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--method', type=str, choices=['dwt', 'wst', 'disk'], default='dwt',
                       help='Analysis method (dwt, wst, or disk visualization)')
    parser.add_argument('--max_level', type=int, default=3, help='Maximum decomposition level for DWT')
    parser.add_argument('--wavelet', type=str, default='haar', help='Wavelet type for DWT')
    parser.add_argument('--J', type=int, default=2, help='Number of scales for WST')
    parser.add_argument('--L', type=int, default=8, help='Number of angles for WST')
    parser.add_argument('--output_dir', type=str, default='wavelet_analysis', 
                       help='Directory to save output images')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = WaveletAnalyzer(
        max_level=args.max_level,
        wavelet=args.wavelet,
        J=args.J,
        L=args.L,
        output_dir=args.output_dir
    )
    
    # Load image
    image = analyzer.load_image(args.image_path)
    
    # Perform analysis
    if args.method == 'dwt':
        analyzer.analyze_dwt(image)
    elif args.method == 'wst':
        analyzer.analyze_wst(image)
    elif args.method == 'disk':
        analyzer.visualize_wst_disk(image)
    
    print(f"Analysis completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()