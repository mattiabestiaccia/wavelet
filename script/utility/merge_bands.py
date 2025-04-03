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


def create_multiband_tiffs(input_dir, output_dir, pattern="IMG_*_*.tif", max_files=None, compression="lzw", max_bands=10):
    """
    Create multiband TIFF files from individual band TIFFs.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing individual band TIFF files
    output_dir : str
        Directory where multiband TIFFs will be saved
    pattern : str, optional
        Glob pattern to match individual band files
    max_files : int, optional
        Maximum number of files to process (for testing)
    compression : str, optional
        Compression to use for output files (lzw, deflate, etc.)
    max_bands : int, optional
        Maximum number of bands to include in the output TIFF (default: 10)
    """
    # Converti i percorsi in oggetti Path e gestisci gli spazi
    input_dir = Path(os.path.expanduser(str(input_dir).strip()))
    output_dir = Path(os.path.expanduser(str(output_dir).strip()))
    
    # Crea la directory di output se non esiste
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Directory di output configurata: {output_dir}")
    except Exception as e:
        print(f"Errore nella creazione della directory di output: {e}")
        return
    
    print(f"Scanning for files in {input_dir}...")
    try:
        # Usa glob direttamente con il pattern
        all_files = sorted(str(f) for f in input_dir.glob("IMG_[0-9][0-9][0-9][0-9]_[0-9]*.tif"))
        
        if not all_files:
            print(f"Nessun file trovato in {input_dir}")
            print("Files presenti nella directory:")
            for f in input_dir.iterdir():
                print(f"  {f.name}")
            return
        
        print(f"Trovati {len(all_files)} files")
        
    except Exception as e:
        print(f"Errore durante la scansione dei file: {e}")
        return
    
    # Group files by base name (without band number)
    file_groups = {}
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        # Modifica il pattern regex per essere più flessibile
        match = re.match(r'(.+?)[-_]?(\d+)\.tif', file_name, re.IGNORECASE)
        if match:
            base_name = match.group(1)
            band_num = int(match.group(2))
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append((band_num, file_path))
    
    if not file_groups:
        print("Nessun file corrisponde al pattern atteso. Files trovati:")
        for f in all_files[:10]:  # Mostra solo i primi 10 file
            print(f"  {os.path.basename(f)}")
        return
    
    # Determina il numero di bande per ogni gruppo
    band_counts = {base: len(files) for base, files in file_groups.items()}
    
    # Trova il numero massimo di bande tra tutti i gruppi
    found_bands = max(band_counts.values()) if band_counts else 0
    print(f"Numero massimo di bande trovate: {found_bands}")
    
    # Limita il numero di bande al massimo specificato
    if found_bands > max_bands:
        print(f"Attenzione: Limitando il numero di bande a {max_bands} (trovate {found_bands})")
        bands_to_use = max_bands
    else:
        bands_to_use = found_bands
    
    # Filtra i gruppi che hanno almeno il numero di bande richiesto
    complete_groups = {
        base: sorted(files, key=lambda x: x[0])[:bands_to_use]
        for base, files in file_groups.items()
        if len(files) >= bands_to_use
    }
    
    print(f"Found {len(complete_groups)} complete image sets with {bands_to_use} bands each")
    if not complete_groups:
        print("No complete image sets found. Check file pattern and directory.")
        print("\nGruppi incompleti trovati:")
        for base, files in file_groups.items():
            print(f"  {base}: {len(files)} bande")
        return
    
    # Limita il numero di file da processare se specificato
    if max_files:
        keys_to_keep = list(complete_groups.keys())[:max_files]
        complete_groups = {k: complete_groups[k] for k in keys_to_keep}
        print(f"Processing {len(complete_groups)} image sets (limited by max_files)")
    
    # Processa ogni gruppo
    for base_name, file_info in tqdm(complete_groups.items(), desc="Processing image sets"):
        file_paths = [f[1] for f in file_info]
        
        # Leggi i metadati dal primo file
        with rasterio.open(file_paths[0]) as src:
            meta = src.meta.copy()
            meta.update({
                'count': bands_to_use,
                'compress': compression
            })
        
        # Percorso del file di output
        output_path = output_dir / f"{base_name}_multiband.tif"
        
        # Crea il nuovo file multibanda
        with rasterio.open(output_path, 'w', **meta) as dst:
            for band_idx, file_path in enumerate(file_paths, 1):
                with rasterio.open(file_path) as src:
                    data = src.read(1)
                    dst.write(data, band_idx)
                    band_desc = src.descriptions[0] if src.descriptions else f"Band {band_idx}"
                    dst.set_band_description(band_idx, band_desc)
                    
        print(f"Created {output_path}")
    
    print(f"Successfully created {len(complete_groups)} multiband TIFF files in {output_dir}")
    print(f"Each file contains {bands_to_use} bands")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create multiband TIFF files from individual bands")
    parser.add_argument("input_dir", 
                       help="Directory containing individual band TIFF files")
    parser.add_argument("output_dir", 
                       help="Output directory for multiband TIFFs (will be created if it doesn't exist)")
    parser.add_argument("--pattern", 
                       default="*.tif", 
                       help="File pattern to match (default: *.tif)")
    parser.add_argument("--max-files", 
                       type=int, 
                       help="Maximum number of files to process (for testing)")
    parser.add_argument("--compression", 
                       default="lzw", 
                       help="Compression for output files (default: lzw)")
    parser.add_argument("--max-bands", 
                       type=int,
                       default=10,
                       help="Maximum number of bands to include in output files (default: 10)")
    
    args = parser.parse_args()
    
    create_multiband_tiffs(
        args.input_dir,
        args.output_dir,
        args.pattern,
        args.max_files,
        args.compression,
        args.max_bands
    )
