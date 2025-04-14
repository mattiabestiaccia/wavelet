#!/usr/bin/env python3
"""
Script per eseguire la classificazione pixel-wise utilizzando la trasformata wavelet scattering.

Questo script esegue la classificazione pixel-wise di immagini multibanda
utilizzando un modello pre-addestrato con trasformata wavelet scattering.

Utilizzo:
    python script/core/pixel_classification/run_pixel_classification.py --image /path/to/image.jpg --model /path/to/model.pth --output /path/to/output
    python script/core/pixel_classification/run_pixel_classification.py --folder /path/to/images --model /path/to/model.pth --output /path/to/output
"""

import os
import sys
import argparse
from pathlib import Path

# Aggiungi la directory principale al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from wavelet_lib.single_pixel_classification.processors import (
    PixelClassificationProcessor,
    process_image
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Classificazione pixel-wise con WST')
    
    # Modalità di input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Percorso dell\'immagine da processare')
    input_group.add_argument('--folder', type=str, help='Directory contenente le immagini da processare')
    
    # Modello e output
    parser.add_argument('--model', type=str, required=True, help='Percorso del modello di classificazione')
    parser.add_argument('--output', type=str, help='Directory di output per i risultati')
    
    # Parametri opzionali
    parser.add_argument('--patch_size', type=int, default=32, help='Dimensione delle patch')
    parser.add_argument('--stride', type=int, default=16, help='Passo per l\'inferenza')
    parser.add_argument('--j', type=int, default=2, help='Numero di scale per la trasformata scattering')
    parser.add_argument('--overlay', action='store_true', help='Crea un overlay con l\'immagine originale')
    parser.add_argument('--alpha', type=float, default=0.5, help='Opacità dell\'overlay')
    parser.add_argument('--no_display', action='store_true', help='Non visualizzare i risultati')
    
    return parser.parse_args()


def main(args):
    """Funzione principale per eseguire la classificazione."""
    # Crea processore
    processor = PixelClassificationProcessor(
        model_path=args.model,
        patch_size=args.patch_size,
        stride=args.stride,
        J=args.j
    )
    
    # Crea directory di output se necessario
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Processa input in base agli argomenti
    if args.image:
        # Processa una singola immagine
        print(f"Processamento immagine: {args.image}")
        
        # Crea percorso di output
        output_path = None
        if args.output:
            output_path = os.path.join(args.output, f"{Path(args.image).stem}_classification.png")
        
        # Processa immagine
        classification_map = processor.process_image(
            image_path=args.image,
            output_path=output_path,
            overlay=args.overlay,
            alpha=args.alpha
        )
        
        # Visualizza risultati
        if not args.no_display:
            processor.visualize_results(
                image_path=args.image,
                classification_map=classification_map,
                output_path=os.path.join(args.output, f"{Path(args.image).stem}_results.png") if args.output else None
            )
        
        # Crea legenda
        if args.output:
            processor.create_legend(
                output_path=os.path.join(args.output, "legend.png")
            )
        
    elif args.folder:
        # Processa una cartella di immagini
        print(f"Processamento cartella: {args.folder}")
        
        # Processa immagini
        output_paths = processor.process_folder(
            folder_path=args.folder,
            output_dir=args.output,
            overlay=args.overlay,
            alpha=args.alpha
        )
        
        # Crea legenda
        if args.output:
            processor.create_legend(
                output_path=os.path.join(args.output, "legend.png")
            )
        
        print(f"\nProcessate {len(output_paths)} immagini")
    
    print("\nClassificazione completata.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
