#!/usr/bin/env python3

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
        
        # Crea la figura e gli assi una sola volta
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.ion()  # Abilita la modalità interattiva
        
        while True:
            for idx in range(num_bands):
                # Lettura del canale
                band = src.read(idx + 1)
                
                # Normalizzazione per la visualizzazione
                band_norm = (band - band.min()) / (band.max() - band.min())
                
                # Pulisci gli assi
                ax.clear()
                
                # Plot
                im = ax.imshow(band_norm, cmap='viridis')
                ax.set_title(f'Canale {idx + 1}')
                ax.axis('off')
                
                # Forza l'aggiornamento del display
                plt.draw()
                plt.pause(delay)
            
            # Disabilita temporaneamente la modalità interattiva per l'input
            plt.ioff()
            user_input = input("\nPremi Enter per ripetere la sequenza o 'q' per uscire: ")
            plt.ion()
            
            if user_input.lower() == 'q':
                break
        
        plt.ioff()
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualizza canali di un\'immagine multibanda in sequenza')
    parser.add_argument('input_file', type=str, help='File TIFF da visualizzare')
    parser.add_argument('--delay', type=float, default=1.0, help='Tempo di attesa tra i canali in secondi')
    parser.add_argument('--max-channels', type=int, default=10, help='Numero massimo di canali da visualizzare')
    args = parser.parse_args()
    
    try:
        visualize_channels_sequence(args.input_file, args.delay, args.max_channels)
    except Exception as e:
        print(f"Errore nel processare {args.input_file}: {e}")
        raise e  # Mostra l'errore completo per debug

if __name__ == "__main__":
    main()
