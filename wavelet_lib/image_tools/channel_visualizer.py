#!/usr/bin/env python3
"""
Modulo per la visualizzazione di canali in immagini multibanda.
Fornisce funzioni per visualizzare singoli canali o sequenze di canali
in immagini multibanda, utile per l'analisi di immagini satellitari o multispettrali.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import argparse
import time


def visualize_channels_sequence(input_file, delay=1.0, max_channels=10):
    """
    Visualizza in sequenza ogni canale di un'immagine multibanda.
    
    Args:
        input_file (str): Percorso del file TIFF multibanda
        delay (float): Tempo di attesa tra un canale e l'altro in secondi
        max_channels (int): Numero massimo di canali da visualizzare (default: 10)
    """
    input_path = Path(input_file)
    
    # Apertura dell'immagine con rasterio
    with rasterio.open(input_file) as src:
        num_bands = src.count
        print(f"Immagine: {input_path.name}")
        print(f"Numero di canali: {num_bands}")
        
        # Check if the number of bands exceeds the maximum
        if num_bands > max_channels:
            print(f"Attenzione: L'immagine ha {num_bands} canali, ma verranno visualizzati solo i primi {max_channels}")
            num_bands = max_channels
        
        # Crea una figura per la visualizzazione
        plt.figure(figsize=(10, 8))
        
        # Visualizza ogni canale in sequenza
        for band_idx in range(1, num_bands + 1):
            # Leggi il canale
            band = src.read(band_idx)
            
            # Normalizza i valori per la visualizzazione
            band_norm = (band - band.min()) / (band.max() - band.min() + 1e-10)
            
            # Ottieni la descrizione del canale se disponibile
            band_desc = src.descriptions[band_idx-1] if src.descriptions and src.descriptions[band_idx-1] else f"Band {band_idx}"
            
            # Visualizza il canale
            plt.clf()
            plt.imshow(band_norm, cmap='viridis')
            plt.colorbar(label='Normalized Value')
            plt.title(f"Channel {band_idx}: {band_desc}")
            plt.axis('off')
            plt.tight_layout()
            plt.draw()
            plt.pause(delay)
        
        plt.show()


def visualize_channels(input_file, max_channels=10, figsize=(15, 10)):
    """
    Visualizza tutti i canali di un'immagine multibanda in un'unica figura.
    
    Args:
        input_file (str): Percorso del file TIFF multibanda
        max_channels (int): Numero massimo di canali da visualizzare (default: 10)
        figsize (tuple): Dimensioni della figura (default: (15, 10))
    """
    input_path = Path(input_file)
    
    # Apertura dell'immagine con rasterio
    with rasterio.open(input_file) as src:
        num_bands = src.count
        print(f"Immagine: {input_path.name}")
        print(f"Numero di canali: {num_bands}")
        
        # Check if the number of bands exceeds the maximum
        if num_bands > max_channels:
            print(f"Attenzione: L'immagine ha {num_bands} canali, ma verranno visualizzati solo i primi {max_channels}")
            num_bands = max_channels
        
        # Calcola il layout della griglia
        n_cols = min(4, num_bands)
        n_rows = (num_bands + n_cols - 1) // n_cols
        
        # Crea una figura per la visualizzazione
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Visualizza ogni canale
        for band_idx in range(1, num_bands + 1):
            ax = axes[band_idx-1]
            
            # Leggi il canale
            band = src.read(band_idx)
            
            # Normalizza i valori per la visualizzazione
            band_norm = (band - band.min()) / (band.max() - band.min() + 1e-10)
            
            # Ottieni la descrizione del canale se disponibile
            band_desc = src.descriptions[band_idx-1] if src.descriptions and src.descriptions[band_idx-1] else f"Band {band_idx}"
            
            # Visualizza il canale
            im = ax.imshow(band_norm, cmap='viridis')
            ax.set_title(f"Channel {band_idx}: {band_desc}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Nascondi gli assi non utilizzati
        for idx in range(num_bands, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Funzione principale per l'esecuzione come script."""
    parser = argparse.ArgumentParser(description="Visualizza i canali di un'immagine multibanda")
    parser.add_argument("input_file", 
                       help="Percorso del file TIFF multibanda [NECESSARY]")
    parser.add_argument("--sequence", 
                       action="store_true", 
                       help="Visualizza i canali in sequenza [OPTIONAL, default=False]")
    parser.add_argument("--delay", 
                       type=float, 
                       default=1.0, 
                       help="Tempo di attesa tra un canale e l'altro in secondi [OPTIONAL, default=1.0]")
    parser.add_argument("--max-channels", 
                       type=int, 
                       default=10, 
                       help="Numero massimo di canali da visualizzare [OPTIONAL, default=10]")
    
    args = parser.parse_args()
    
    if args.sequence:
        visualize_channels_sequence(args.input_file, args.delay, args.max_channels)
    else:
        visualize_channels(args.input_file, args.max_channels)


if __name__ == "__main__":
    main()
