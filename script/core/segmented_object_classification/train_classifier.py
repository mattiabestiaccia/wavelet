#!/usr/bin/env python3
"""
Script per addestrare un classificatore di oggetti segmentati.

Questo script addestra un modello di classificazione per oggetti segmentati
utilizzando la trasformata wavelet scattering.

Utilizzo:
    python script/core/segmented_object_classification/train_classifier.py --train_dir /path/to/train_data --model_path /path/to/model.pth
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Aggiungi la directory principale al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from wavelet_lib.segmented_object_classification.training import train_segmented_object_classifier


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Addestramento classificatore di oggetti segmentati')
    
    # Input e output
    parser.add_argument('--train_dir', type=str, required=True, help='Directory contenente i dati di training')
    parser.add_argument('--val_dir', type=str, help='Directory contenente i dati di validazione (opzionale)')
    parser.add_argument('--model_path', type=str, required=True, help='Percorso dove salvare il modello')
    
    # Parametri di training
    parser.add_argument('--input_size', type=str, default='32,32', help='Dimensione di input per il modello (altezza,larghezza)')
    parser.add_argument('--batch_size', type=int, default=32, help='Dimensione del batch')
    parser.add_argument('--num_epochs', type=int, default=50, help='Numero di epoche di training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate per l\'ottimizzatore')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay per l\'ottimizzatore')
    parser.add_argument('--val_split', type=float, default=0.2, help='Frazione dei dati da usare per la validazione se val_dir non è specificato')
    
    # Parametri della trasformata scattering
    parser.add_argument('--j', type=int, default=2, help='Numero di scale per la trasformata scattering')
    parser.add_argument('--scattering_order', type=int, default=2, help='Ordine della trasformata scattering')
    
    # Altre opzioni
    parser.add_argument('--no_balance', action='store_true', help='Non bilanciare le classi')
    parser.add_argument('--max_samples', type=int, help='Numero massimo di campioni per classe')
    parser.add_argument('--no_augment', action='store_true', help='Non applicare data augmentation')
    parser.add_argument('--num_workers', type=int, default=4, help='Numero di worker per il data loading')
    
    return parser.parse_args()


def main(args):
    """Funzione principale per addestrare il classificatore."""
    # Parsing delle dimensioni di input
    input_size = tuple(map(int, args.input_size.split(',')))
    
    # Determina il device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    # Addestra il modello
    result = train_segmented_object_classifier(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        val_split=args.val_split,
        model_path=args.model_path,
        input_size=input_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        J=args.j,
        scattering_order=args.scattering_order,
        device=device,
        num_workers=args.num_workers,
        balance_classes=not args.no_balance,
        max_samples_per_class=args.max_samples,
        augment=not args.no_augment
    )
    
    print("\nAddestramento completato con successo!")
    print(f"Modello salvato in: {args.model_path}")
    print(f"Classi: {result['classes']}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
