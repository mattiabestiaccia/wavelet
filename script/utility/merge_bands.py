#!/usr/bin/env python3
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


def create_multiband_tiffs(input_dir, output_dir=None, pattern="IMG_*_[1-5].tif", max_files=None, compression="lzw"):
    """
    Create multiband TIFF files from individual band TIFFs.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing individual band TIFF files
    output_dir : str, optional
        Directory where multiband TIFFs will be saved. If None, creates 'multiband' subdirectory
    pattern : str, optional
        Glob pattern to match individual band files
    max_files : int, optional
        Maximum number of files to process (for testing)
    compression : str, optional
        Compression to use for output files (lzw, deflate, etc.)
    """
    # Setup input and output directories
    input_dir = Path(input_dir)
    if not output_dir:
        output_dir = input_dir / "multiband"
    else:
        output_dir = Path(output_dir)
        
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Scanning for files in {input_dir}...")
    # Get all matching files
    all_files = sorted(glob.glob(str(input_dir / pattern)))
    
    # Group files by base name (without band number)
    file_groups = {}
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        # Extract base name (e.g., "IMG_0001" from "IMG_0001_1.tif")
        match = re.match(r'(.+)_[1-5]\.tif', file_name)
        if match:
            base_name = match.group(1)
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(file_path)
    
    # Filter out incomplete groups (less than 5 bands)
    complete_groups = {base: files for base, files in file_groups.items() if len(files) == 5}
    
    print(f"Found {len(complete_groups)} complete image sets")
    if not complete_groups:
        print("No complete image sets found. Check file pattern and directory.")
        return
    
    # Limit the number of files to process if specified
    if max_files:
        keys_to_keep = list(complete_groups.keys())[:max_files]
        complete_groups = {k: complete_groups[k] for k in keys_to_keep}
        print(f"Processing {len(complete_groups)} image sets (limited by max_files)")
        
    # Process each group
    for base_name, file_paths in tqdm(complete_groups.items(), desc="Processing image sets"):
        # Sort by band number to ensure correct order (1-5)
        file_paths = sorted(file_paths, key=lambda x: int(re.match(r'.+_([1-5])\.tif', os.path.basename(x)).group(1)))
        
        # Read metadata from first file
        with rasterio.open(file_paths[0]) as src:
            meta = src.meta.copy()
            
            # Update metadata for multiband output
            meta.update({
                'count': 5,  # 5 bands
                'compress': compression
            })
        
        # Output file path
        output_path = output_dir / f"{base_name}_multiband.tif"
        
        # Create new multiband file
        with rasterio.open(output_path, 'w', **meta) as dst:
            # Read and write each band
            for band_idx, file_path in enumerate(file_paths, 1):
                with rasterio.open(file_path) as src:
                    # Read the entire band data
                    data = src.read(1)
                    # Write to output
                    dst.write(data, band_idx)
                    
                    # Copy band description if available
                    band_desc = src.descriptions[0] if src.descriptions else f"Band {band_idx}"
                    dst.set_band_description(band_idx, band_desc)
                    
        print(f"Created {output_path}")
    
    print(f"Successfully created {len(complete_groups)} multiband TIFF files in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create multiband TIFF files from individual bands")
    parser.add_argument("input_dir", help="Directory containing individual band TIFF files")
    parser.add_argument("--output-dir", help="Output directory for multiband TIFFs (default: input_dir/multiband)")
    parser.add_argument("--pattern", default="IMG_*_[1-5].tif", help="File pattern to match (default: IMG_*_[1-5].tif)")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process (for testing)")
    parser.add_argument("--compression", default="lzw", help="Compression for output files (default: lzw)")
    
    args = parser.parse_args()
    
    create_multiband_tiffs(
        args.input_dir, 
        args.output_dir, 
        args.pattern, 
        args.max_files,
        args.compression
    )