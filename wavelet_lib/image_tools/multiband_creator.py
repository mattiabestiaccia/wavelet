#!/usr/bin/env python3
"""
Modulo per la creazione di file TIFF multibanda da singole bande.
Fornisce funzioni per combinare più file TIFF monobanda in un unico file multibanda.
"""

import os
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.warp import reproject, Resampling
import cv2  # Importa OpenCV per le operazioni di ridimensionamento

# Import scikit-image for resize operations on non-georeferenced images (optional)
try:
    from skimage.transform import resize
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False
    print("NOTE: scikit-image not found, using OpenCV for image resizing")


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
        print(f"\n[INFO] Directory di output: {output_dir}")
    except Exception as e:
        print(f"\n[ERROR] Errore nella creazione della directory di output: {e}")
        return
    
    print(f"[INFO] Scanning for files in {input_dir}...")
    try:
        # Usa glob direttamente con il pattern
        all_files = sorted(str(f) for f in input_dir.glob("IMG_[0-9][0-9][0-9][0-9]_[0-9]*.tif"))
        
        if not all_files:
            print(f"[ERROR] Nessun file trovato in {input_dir}")
            print("Files presenti nella directory:")
            for f in input_dir.iterdir():
                print(f"  {f.name}")
            return
        
        print(f"[INFO] Trovati {len(all_files)} files")
        
    except Exception as e:
        print(f"[ERROR] Errore durante la scansione dei file: {e}")
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
        print("[ERROR] Nessun file corrisponde al pattern atteso. Files trovati:")
        for f in all_files[:10]:  # Mostra solo i primi 10 file
            print(f"  {os.path.basename(f)}")
        return
    
    # Determina il numero di bande per ogni gruppo
    band_counts = {base: len(files) for base, files in file_groups.items()}
    
    # Trova il numero massimo di bande tra tutti i gruppi
    found_bands = max(band_counts.values()) if band_counts else 0
    print(f"[INFO] Numero massimo di bande trovate: {found_bands}")
    
    # Limita il numero di bande al massimo specificato
    if found_bands > max_bands:
        print(f"[WARNING] Limitando il numero di bande a {max_bands} (trovate {found_bands})")
        bands_to_use = max_bands
    else:
        bands_to_use = found_bands
    
    # Filtra i gruppi che hanno almeno il numero di bande richiesto
    complete_groups = {
        base: sorted(files, key=lambda x: x[0])[:bands_to_use]
        for base, files in file_groups.items()
        if len(files) >= bands_to_use
    }
    
    print(f"[INFO] Found {len(complete_groups)} complete image sets with {bands_to_use} bands each")
    if not complete_groups:
        print("[ERROR] No complete image sets found. Check file pattern and directory.")
        print("\nGruppi incompleti trovati:")
        for base, files in file_groups.items():
            print(f"  {base}: {len(files)} bande")
        return
    
    # Limita il numero di file da processare se specificato
    if max_files:
        keys_to_keep = list(complete_groups.keys())[:max_files]
        complete_groups = {k: complete_groups[k] for k in keys_to_keep}
        print(f"[INFO] Processing {len(complete_groups)} image sets (limited by max_files)")
    
    # Configura tqdm per un output più compatto
    progress_bar = tqdm(complete_groups.items(), desc="Processing image sets",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    # Processa ogni gruppo
    for base_name, file_info in progress_bar:
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
        
        # Leggi tutte le bande e prepara l'allineamento
        all_data = []
        all_descs = []
        reference_transform = None
        reference_shape = None
        
        # Prima, leggi tutte le bande e la trasformazione di riferimento
        for band_idx, file_path in enumerate(file_paths, 1):
            with rasterio.open(file_path) as src:
                data = src.read(1)
                if reference_shape is None:
                    reference_shape = data.shape
                    reference_transform = src.transform
                all_data.append(data)
                band_desc = src.descriptions[0] if src.descriptions else f"Band {band_idx}"
                all_descs.append(band_desc)
        
        # Verifica se c'è bisogno di allineamento (se ci sono differenze di dimensione)
        shapes_differ = any(data.shape != reference_shape for data in all_data)
        
        # Crea il nuovo file multibanda
        with rasterio.open(output_path, 'w', **meta) as dst:
            for band_idx, (data, band_desc) in enumerate(zip(all_data, all_descs), 1):
                # Se necessario, esegui l'allineamento usando rasterio warp
                if shapes_differ or band_idx > 1:  # Allinea tutte le bande tranne la prima
                    # Determina le dimensioni di output
                    out_shape = reference_shape
                    # Crea una matrice vuota per i dati allineati
                    aligned_data = np.zeros(out_shape, dtype=data.dtype)
                    
                    # Apri nuovamente il file per ottenere la trasformazione specifica della banda
                    with rasterio.open(file_paths[band_idx-1]) as src:
                        # Usa rasterio.warp per allineare la banda alla riferimento
                        # Se non c'è CRS, usa una trasformazione diretta senza CRS
                        if src.crs:
                            reproject(
                                source=data,
                                destination=aligned_data,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=reference_transform,
                                dst_crs=src.crs,
                                resampling=Resampling.bilinear
                            )
                        else:
                            # Per immagini senza georeferenziazione, usa resize scikit-image o OpenCV
                            if HAVE_SKIMAGE:
                                # Resize usando scikit-image se disponibile
                                aligned_data = resize(
                                    data, 
                                    output_shape=reference_shape,
                                    preserve_range=True,
                                    order=1
                                ).astype(data.dtype)
                            else:
                                # Ridimensiona l'immagine usando OpenCV
                                aligned_data = cv2.resize(
                                    data, 
                                    (reference_shape[1], reference_shape[0]),
                                    interpolation=cv2.INTER_LINEAR
                                ).astype(data.dtype)
                    dst.write(aligned_data, band_idx)
                else:
                    dst.write(data, band_idx)
                
                dst.set_band_description(band_idx, band_desc)
        
        # Aggiorna la descrizione della barra di progresso con il file appena creato
        progress_bar.set_postfix_str(f"Last: {base_name}_multiband.tif")
    
    print(f"\n[SUCCESS] Created {len(complete_groups)} multiband TIFF files in {output_dir}")
    print(f"[INFO] Each file contains {bands_to_use} bands")


def main():
    """Funzione principale per l'esecuzione come script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create multiband TIFF files from individual bands")
    parser.add_argument("input_dir",
                       help="Directory containing individual band TIFF files [NECESSARY]")
    parser.add_argument("output_dir",
                       help="Output directory for multiband TIFFs (will be created if it doesn't exist) [NECESSARY]")
    parser.add_argument("--pattern",
                       default="*.tif",
                       help="File pattern to match [OPTIONAL, default=*.tif]")
    parser.add_argument("--max-files",
                       type=int,
                       help="Maximum number of files to process (for testing) [OPTIONAL]")
    parser.add_argument("--compression",
                       default="lzw",
                       help="Compression for output files [OPTIONAL, default=lzw]")
    parser.add_argument("--max-bands",
                       type=int,
                       default=10,
                       help="Maximum number of bands to include in output files [OPTIONAL, default=10]")
    
    args = parser.parse_args()
    
    create_multiband_tiffs(
        args.input_dir,
        args.output_dir,
        args.pattern,
        args.max_files,
        args.compression,
        args.max_bands
    )


if __name__ == "__main__":
    main()
