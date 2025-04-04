#!/usr/bin/env python3
"""
Phantom Generator for Wavelet Analysis.

This module provides tools for generating synthetic test images (phantoms)
for evaluating wavelet transform algorithms. These synthetic images have
known properties making them useful for algorithm validation.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path


class PhantomGenerator:
    """
    Generator for synthetic test images (phantoms) with controlled properties.
    
    This class creates various types of synthetic test images that are useful
    for testing and benchmarking wavelet transform algorithms.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the phantom generator.
        
        Args:
            output_dir (str, optional): Directory to save generated images
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = None
    
    def generate_octagon_phantom(self, size=4000, radius1=1000, radius2=2000, 
                                fade_width=200, noise_sigma=1.0, save_path=None):
        """
        Generate a synthetic octagonal phantom with fade and noise.
        
        Creates a test image with concentric octagons of different intensities,
        with controlled fade between regions and optional Gaussian noise.
        
        Args:
            size (int): Size of the square image
            radius1 (int): Radius of inner octagon
            radius2 (int): Radius of outer octagon
            fade_width (int): Width of the fade region
            noise_sigma (float): Standard deviation of the Gaussian noise
            save_path (str, optional): Path to save the generated image
            
        Returns:
            numpy.ndarray: Generated phantom image
        """
        # Create a black image
        image = np.zeros((size, size), dtype=np.uint8)
        
        # Calculate octagon vertices
        center = (size // 2, size // 2)
        points1 = self._calculate_octagon_points(center, radius1)
        points2 = self._calculate_octagon_points(center, radius2)
        points3 = self._calculate_octagon_points(center, radius2 - 100)  # Inner edge
        
        # Create mask for fade effect
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask, [points1], 255)  # Small octagon is white
        
        # Create distance transform for smooth fade
        distance_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        normalized = np.clip(1 - (distance_transform / fade_width), 0, 1) * 255
        faded_octagon = normalized.astype(np.uint8)
        
        # Draw medium octagon (white)
        cv2.fillPoly(image, [points2], 255)
        
        # Draw inner octagon (black)
        cv2.fillPoly(image, [points3], 0)
        
        # Overlay the faded region
        image = np.maximum(image, faded_octagon)
        
        # Add Gaussian noise
        if noise_sigma > 0:
            gaussian_noise = np.random.normal(0, noise_sigma, image.shape).astype(np.uint8)
            image = cv2.add(image, gaussian_noise)
        
        # Save the image if path provided
        if save_path:
            cv2.imwrite(save_path, image)
        
        if self.output_dir:
            cv2.imwrite(str(self.output_dir / "octagon_phantom.jpg"), image)
        
        return image
    
    def generate_circular_phantom(self, size=512, num_circles=5, min_radius=50, 
                                  max_radius=None, noise_sigma=0.0, save_path=None):
        """
        Generate a phantom with concentric circles of varying intensities.
        
        Args:
            size (int): Size of the square image
            num_circles (int): Number of concentric circles
            min_radius (int): Radius of innermost circle
            max_radius (int, optional): Radius of outermost circle. If None, uses size/2 - 10
            noise_sigma (float): Standard deviation of the Gaussian noise
            save_path (str, optional): Path to save the generated image
            
        Returns:
            numpy.ndarray: Generated phantom image
        """
        # Create a black image
        image = np.zeros((size, size), dtype=np.float32)
        
        # Set default max radius if not provided
        if max_radius is None:
            max_radius = size // 2 - 10
        
        # Calculate radii for each circle
        radii = np.linspace(min_radius, max_radius, num_circles)
        
        # Generate circle intensities (alternating between bright and dark)
        intensities = []
        for i in range(num_circles):
            if i % 2 == 0:
                intensities.append(1.0)  # Bright
            else:
                intensities.append(0.2)  # Dark
        
        # Draw the circles from outside in
        center = (size // 2, size // 2)
        for i in range(num_circles):
            r = int(radii[i])
            cv2.circle(image, center, r, intensities[i], -1)  # -1 for filled circle
        
        # Add Gaussian noise
        if noise_sigma > 0:
            gaussian_noise = np.random.normal(0, noise_sigma, image.shape)
            image = np.clip(image + gaussian_noise, 0, 1)
        
        # Convert to 8-bit for saving
        image_8bit = (image * 255).astype(np.uint8)
        
        # Save the image if path provided
        if save_path:
            cv2.imwrite(save_path, image_8bit)
        
        if self.output_dir:
            cv2.imwrite(str(self.output_dir / "circular_phantom.jpg"), image_8bit)
        
        return image
    
    def generate_checkerboard_phantom(self, size=512, grid_size=8, noise_sigma=0.0, save_path=None):
        """
        Generate a checkerboard pattern phantom.
        
        Args:
            size (int): Size of the square image
            grid_size (int): Number of squares in each dimension
            noise_sigma (float): Standard deviation of the Gaussian noise
            save_path (str, optional): Path to save the generated image
            
        Returns:
            numpy.ndarray: Generated phantom image
        """
        # Create checkerboard pattern
        cell_size = size // grid_size
        image = np.zeros((size, size), dtype=np.float32)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 0:
                    # White square
                    image[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = 1.0
        
        # Add Gaussian noise
        if noise_sigma > 0:
            gaussian_noise = np.random.normal(0, noise_sigma, image.shape)
            image = np.clip(image + gaussian_noise, 0, 1)
        
        # Convert to 8-bit for saving
        image_8bit = (image * 255).astype(np.uint8)
        
        # Save the image if path provided
        if save_path:
            cv2.imwrite(save_path, image_8bit)
        
        if self.output_dir:
            cv2.imwrite(str(self.output_dir / "checkerboard_phantom.jpg"), image_8bit)
        
        return image
    
    def generate_frequency_phantom(self, size=512, min_freq=2, max_freq=16, 
                                  orientation=0, noise_sigma=0.0, save_path=None):
        """
        Generate a phantom with varying frequency sinusoidal patterns.
        
        Args:
            size (int): Size of the square image
            min_freq (int): Minimum frequency (cycles per image)
            max_freq (int): Maximum frequency (cycles per image)
            orientation (float): Orientation in degrees (0 for horizontal, 90 for vertical)
            noise_sigma (float): Standard deviation of the Gaussian noise
            save_path (str, optional): Path to save the generated image
            
        Returns:
            numpy.ndarray: Generated phantom image
        """
        # Create coordinate grid
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Calculate rotated coordinates
        orientation_rad = np.radians(orientation)
        X_rot = X * np.cos(orientation_rad) + Y * np.sin(orientation_rad)
        
        # Create a gradient of frequencies
        image = np.zeros((size, size), dtype=np.float32)
        
        # Apply frequency gradient along Y axis
        for i in range(size):
            # Calculate frequency for this row
            freq = min_freq + (max_freq - min_freq) * (i / size)
            # Generate sinusoidal pattern for this row
            image[i, :] = 0.5 + 0.5 * np.sin(2 * np.pi * freq * X_rot[i, :])
        
        # Add Gaussian noise
        if noise_sigma > 0:
            gaussian_noise = np.random.normal(0, noise_sigma, image.shape)
            image = np.clip(image + gaussian_noise, 0, 1)
        
        # Convert to 8-bit for saving
        image_8bit = (image * 255).astype(np.uint8)
        
        # Save the image if path provided
        if save_path:
            cv2.imwrite(save_path, image_8bit)
        
        if self.output_dir:
            cv2.imwrite(str(self.output_dir / "frequency_phantom.jpg"), image_8bit)
        
        return image
    
    def generate_all_phantoms(self, base_size=512, save_dir=None):
        """
        Generate all available phantom types and save to directory.
        
        Args:
            base_size (int): Base size for the phantoms
            save_dir (str, optional): Directory to save generated images
            
        Returns:
            dict: Dictionary mapping phantom types to generated images
        """
        if save_dir:
            save_path = Path(save_dir)
            os.makedirs(save_path, exist_ok=True)
        elif self.output_dir:
            save_path = self.output_dir
        else:
            save_path = None
        
        # Generate all phantom types
        phantoms = {}
        
        # Octagon phantom (smaller size)
        octagon_size = min(base_size * 2, 1024)
        phantoms['octagon'] = self.generate_octagon_phantom(
            size=octagon_size, 
            radius1=octagon_size//4, 
            radius2=octagon_size//2,
            fade_width=octagon_size//20,
            save_path=None if save_path is None else str(save_path / "octagon_phantom.jpg")
        )
        
        # Circular phantom
        phantoms['circular'] = self.generate_circular_phantom(
            size=base_size, 
            num_circles=5,
            save_path=None if save_path is None else str(save_path / "circular_phantom.jpg")
        )
        
        # Checkerboard phantom
        phantoms['checkerboard'] = self.generate_checkerboard_phantom(
            size=base_size, 
            grid_size=8,
            save_path=None if save_path is None else str(save_path / "checkerboard_phantom.jpg")
        )
        
        # Frequency phantoms at different orientations
        for angle in [0, 45, 90]:
            phantoms[f'frequency_{angle}deg'] = self.generate_frequency_phantom(
                size=base_size, 
                orientation=angle,
                save_path=None if save_path is None else str(save_path / f"frequency_{angle}deg_phantom.jpg")
            )
        
        # Display summary
        print(f"Generated {len(phantoms)} phantom images")
        if save_path:
            print(f"Saved to: {save_path}")
        
        return phantoms
    
    def _calculate_octagon_points(self, center, radius, num_sides=8, angle_offset=np.pi/8):
        """
        Calculate the vertices of an octagon.
        
        Args:
            center (tuple): Center point (x, y)
            radius (int): Radius of octagon
            num_sides (int): Number of sides
            angle_offset (float): Angle offset in radians
            
        Returns:
            numpy.ndarray: Array of octagon vertices
        """
        return np.array([
            [int(center[0] + radius * np.cos(2 * np.pi * i / num_sides + angle_offset)),
             int(center[1] + radius * np.sin(2 * np.pi * i / num_sides + angle_offset))]
            for i in range(num_sides)
        ], np.int32).reshape((-1, 1, 2))


def main():
    """Command line entry point for phantom generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic phantom images for wavelet analysis')
    parser.add_argument('--type', type=str, choices=['octagon', 'circular', 'checkerboard', 'frequency', 'all'],
                       default='all', help='Type of phantom to generate')
    parser.add_argument('--size', type=int, default=512, help='Size of the image')
    parser.add_argument('--output', type=str, default='phantoms', help='Output directory')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level (sigma)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = PhantomGenerator(args.output)
    
    if args.type == 'all':
        generator.generate_all_phantoms(base_size=args.size)
    elif args.type == 'octagon':
        generator.generate_octagon_phantom(
            size=args.size * 2,  # Octagon looks better with more space
            radius1=args.size // 2,
            radius2=args.size,
            noise_sigma=args.noise
        )
    elif args.type == 'circular':
        generator.generate_circular_phantom(
            size=args.size,
            noise_sigma=args.noise
        )
    elif args.type == 'checkerboard':
        generator.generate_checkerboard_phantom(
            size=args.size,
            noise_sigma=args.noise
        )
    elif args.type == 'frequency':
        # Generate horizontal, diagonal, and vertical patterns
        for angle in [0, 45, 90]:
            generator.generate_frequency_phantom(
                size=args.size,
                orientation=angle,
                noise_sigma=args.noise
            )


if __name__ == "__main__":
    main()