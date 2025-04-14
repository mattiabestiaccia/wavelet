#!/usr/bin/env python3
"""
Script per addestrare un classificatore pixel-wise utilizzando la trasformata wavelet scattering.

Questo script addestra un modello per la classificazione pixel-wise di immagini multibanda
utilizzando la trasformata wavelet scattering.

Utilizzo:
    python script/core/pixel_classification/train_pixel_classifier.py --images_dir /path/to/images --masks_dir /path/to/masks --model /path/to/model.pth
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Aggiungi la directory principale al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from wavelet_lib.base import Config
from wavelet_lib.single_pixel_classification.models import (
    create_pixel_classifier,
    train_pixel_classifier,
    PixelWiseDataset
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Addestramento classificatore pixel-wise con WST')
    
    # Input e output
    parser.add_argument('--images_dir', type=str, required=True, help='Directory contenente le immagini di training')
    parser.add_argument('--masks_dir', type=str, required=True, help='Directory contenente le maschere di classe')
    parser.add_argument('--model', type=str, required=True, help='Percorso dove salvare il modello')
    
    # Parametri di training
    parser.add_argument('--patch_size', type=int, default=32, help='Dimensione delle patch')
    parser.add_argument('--stride', type=int, default=16, help='Passo per l\'estrazione delle patch')
    parser.add_argument('--batch_size', type=int, default=16, help='Dimensione del batch')
    parser.add_argument('--epochs', type=int, default=50, help='Numero di epoche')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Frazione dei dati da usare per la validazione')
    
    # Parametri della trasformata scattering
    parser.add_argument('--j', type=int, default=2, help='Numero di scale per la trasformata scattering')
    parser.add_argument('--scattering_order', type=int, default=2, help='Ordine della trasformata scattering')
    
    # Altre opzioni
    parser.add_argument('--no_augment', action='store_true', help='Disabilita data augmentation')
    parser.add_argument('--seed', type=int, default=42, help='Seed per la riproducibilità')
    parser.add_argument('--num_classes', type=int, default=5, help='Numero di classi (default: 5)')
    parser.add_argument('--class_names', type=str, help='Nomi delle classi separati da virgola')
    
    return parser.parse_args()


def main(args):
    """Funzione principale per addestrare il classificatore."""
    # Imposta seed per la riproducibilità
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determina il device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    # Crea mapping delle classi
    if args.class_names:
        class_names = args.class_names.split(',')
        class_mapping = {i: name for i, name in enumerate(class_names)}
    else:
        class_mapping = {
            0: "background",
            1: "water",
            2: "vegetation",
            3: "streets",
            4: "buildings"
        }
    
    print(f"Classi: {class_mapping}")
    
    # Crea dataset
    print(f"Creazione dataset da {args.images_dir} e {args.masks_dir}")
    dataset = PixelWiseDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        augment=not args.no_augment,
        class_mapping=class_mapping
    )
    
    print(f"Dataset creato con {len(dataset)} patch")
    
    # Dividi in training e validation
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=args.val_split,
        random_state=args.seed
    )
    
    # Crea subset
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Training set: {len(train_dataset)} patch")
    print(f"Validation set: {len(val_dataset)} patch")
    
    # Crea configurazione
    config = Config(
        num_channels=3,  # RGB
        num_classes=args.num_classes,
        scattering_order=args.scattering_order,
        J=args.j,
        shape=(args.patch_size, args.patch_size),
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Crea modello e trasformata scattering
    model, scattering = create_pixel_classifier(config)
    
    # Stampa riepilogo della configurazione
    config.print_summary()
    
    # Addestra il modello
    history = train_pixel_classifier(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_path=args.model,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        scattering=scattering,
        model=model
    )
    
    print("\nAddestramento completato con successo!")
    print(f"Modello salvato in: {args.model}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
