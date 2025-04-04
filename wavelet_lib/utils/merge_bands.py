#!/usr/bin/env python3
"""
Multiband TIFF Generator.

This module provides functionality to combine individual band TIFF files 
into multiband TIFF files, useful for satellite and aerial imagery processing.
"""

import os
import sys
import argparse
import glob
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.windows import Window


def create_multiband_tiffs(input_dir, output_dir, pattern=None, max_files=None, compression="lzw"):
    """
    Create multiband TIFF files from individual band TIFFs.
    
    Args:
        input_dir (str): Directory containing individual band TIFF files
        output_dir (str): Directory where multiband TIFFs will be saved
        pattern (str, optional): Glob pattern to match individual band files. 
                                If None, uses "IMG_[0-9][0-9][0-9][0-9]_[0-9]*.tif"
        max_files (int, optional): Maximum number of files to process (for testing)
        compression (str, optional): Compression to use for output files (lzw, deflate, etc.)
    
    Returns:
        list: Paths to the created multiband TIFF files
    """
    # Convert paths to Path objects and handle spaces
    input_dir = Path(os.path.expanduser(str(input_dir).strip()))
    output_dir = Path(os.path.expanduser(str(output_dir).strip()))
    
    # Set default pattern if None
    if pattern is None:
        pattern = "IMG_[0-9][0-9][0-9][0-9]_[0-9]*.tif"
    
    # Create output directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory configured: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return []
    
    print(f"Scanning for files in {input_dir}...")
    try:
        # Use glob directly with the pattern
        all_files = sorted(str(f) for f in input_dir.glob(pattern))
        
        if not all_files:
            print(f"No files found in {input_dir} matching pattern {pattern}")
            print("Files present in directory:")
            for f in input_dir.iterdir():
                print(f"  {f.name}")
            return []
        
        print(f"Found {len(all_files)} files")
        
    except Exception as e:
        print(f"Error scanning files: {e}")
        return []
    
    # Group files by base name (without band number)
    file_groups = {}
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        # Modify regex pattern to be more flexible
        match = re.match(r'(.+?)[-_]?(\d+)\.tif', file_name, re.IGNORECASE)
        if match:
            base_name = match.group(1)
            band_num = int(match.group(2))
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append((band_num, file_path))
    
    if not file_groups:
        print("No files match the expected pattern. Files found:")
        for f in all_files[:10]:  # Show only the first 10 files
            print(f"  {os.path.basename(f)}")
        return []
    
    # Determine the number of bands for each group
    band_counts = {base: len(files) for base, files in file_groups.items()}
    
    # Find the maximum number of bands among all groups
    max_bands = max(band_counts.values()) if band_counts else 0
    print(f"Maximum number of bands found: {max_bands}")
    
    # Filter groups that have all bands
    complete_groups = {
        base: sorted(files, key=lambda x: x[0])
        for base, files in file_groups.items()
        if len(files) == max_bands
    }
    
    print(f"Found {len(complete_groups)} complete image sets with {max_bands} bands each")
    if not complete_groups:
        print("No complete image sets found. Check file pattern and directory.")
        print("\nIncomplete groups found:")
        for base, files in file_groups.items():
            print(f"  {base}: {len(files)} bands")
        return []
    
    # Limit the number of files to process if specified
    if max_files:
        keys_to_keep = list(complete_groups.keys())[:max_files]
        complete_groups = {k: complete_groups[k] for k in keys_to_keep}
        print(f"Processing {len(complete_groups)} image sets (limited by max_files)")
    
    created_files = []
    
    # Process each group
    for base_name, file_info in tqdm(complete_groups.items(), desc="Processing image sets"):
        file_paths = [f[1] for f in file_info]
        
        # Read metadata from first file
        with rasterio.open(file_paths[0]) as src:
            meta = src.meta.copy()
            meta.update({
                'count': max_bands,
                'compress': compression
            })
        
        # Output file path
        output_path = output_dir / f"{base_name}_multiband.tif"
        
        # Create new multiband file
        with rasterio.open(output_path, 'w', **meta) as dst:
            for band_idx, file_path in enumerate(file_paths, 1):
                with rasterio.open(file_path) as src:
                    data = src.read(1)
                    dst.write(data, band_idx)
                    band_desc = src.descriptions[0] if src.descriptions else f"Band {band_idx}"
                    dst.set_band_description(band_idx, band_desc)
        
        created_files.append(str(output_path))
        print(f"Created {output_path}")
    
    print(f"Successfully created {len(created_files)} multiband TIFF files in {output_dir}")
    print(f"Each file contains {max_bands} bands")
    
    return created_files


def main():
    """Command line entry point for multiband TIFF creation."""
    parser = argparse.ArgumentParser(description="Create multiband TIFF files from individual bands")
    parser.add_argument("input_dir", 
                      help="Directory containing individual band TIFF files")
    parser.add_argument("output_dir", 
                      help="Output directory for multiband TIFFs (will be created if it doesn't exist)")
    parser.add_argument("--pattern", 
                      default="IMG_[0-9][0-9][0-9][0-9]_[0-9]*.tif", 
                      help="File pattern to match (default: IMG_[0-9][0-9][0-9][0-9]_[0-9]*.tif)")
    parser.add_argument("--max-files", 
                      type=int, 
                      help="Maximum number of files to process (for testing)")
    parser.add_argument("--compression", 
                      default="lzw", 
                      help="Compression for output files (default: lzw)")
    
    args = parser.parse_args()
    
    create_multiband_tiffs(
        args.input_dir,
        args.output_dir,
        args.pattern,
        args.max_files,
        args.compression
    )


if __name__ == "__main__":
    main()