#!/usr/bin/env python3
"""
Channel Visualizer for Multiband Images.

This module provides functionality to visualize channels in multiband
images (such as satellite imagery) sequentially, allowing visual
inspection of each channel individually.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import argparse
import time

def visualize_channels_sequence(input_file, delay=1.0, cmap='viridis'):
    """
    Visualize each channel of a multiband image in sequence.
    
    Args:
        input_file (str): Path to multiband TIFF file
        delay (float): Time to wait between channels in seconds
        cmap (str): Matplotlib colormap to use for visualization
        
    Returns:
        int: Number of channels in the image
    """
    input_path = Path(input_file)
    
    # Open image with rasterio
    with rasterio.open(input_file) as src:
        num_bands = src.count
        print(f"Image: {input_path.name}")
        print(f"Number of channels: {num_bands}")
        
        # Create figure and axes once
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.ion()  # Enable interactive mode
        
        while True:
            for idx in range(num_bands):
                # Read channel
                band = src.read(idx + 1)
                
                # Normalize for visualization
                band_norm = (band - band.min()) / (band.max() - band.min())
                
                # Clear axes
                ax.clear()
                
                # Plot
                im = ax.imshow(band_norm, cmap=cmap)
                ax.set_title(f'Channel {idx + 1}')
                ax.axis('off')
                
                # Force display update
                plt.draw()
                plt.pause(delay)
            
            # Temporarily disable interactive mode for input
            plt.ioff()
            user_input = input("\nPress Enter to repeat the sequence or 'q' to exit: ")
            plt.ion()
            
            if user_input.lower() == 'q':
                break
        
        plt.ioff()
        plt.close()
        
        return num_bands

def main():
    """Command line entry point for the channel visualizer."""
    parser = argparse.ArgumentParser(description='Visualize channels of a multiband image in sequence')
    parser.add_argument('input_file', type=str, help='TIFF file to visualize')
    parser.add_argument('--delay', type=float, default=1.0, help='Time to wait between channels in seconds')
    parser.add_argument('--cmap', type=str, default='viridis', help='Matplotlib colormap to use')
    args = parser.parse_args()
    
    try:
        visualize_channels_sequence(args.input_file, args.delay, args.cmap)
    except Exception as e:
        print(f"Error processing {args.input_file}: {e}")
        raise e  # Show full error for debugging

if __name__ == "__main__":
    main()
