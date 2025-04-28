#!/usr/bin/env python3
"""
Script di addestramento per il modulo di classificazione pixel-wise.

Questo script coordina l'intero flusso di lavoro per l'addestramento dei modelli di classificazione pixel-wise,
dalla preparazione dei dati all'addestramento e alla valutazione del modello.

Utilizzo:
    python train.py --images_dir /path/to/images --masks_dir /path/to/masks --model /path/to/model.pth [opzioni]
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

# Importa i moduli del pacchetto
from pixel_classification.utils import Config, set_seed
from pixel_classification.dataset import PixelWiseDataset
from pixel_classification.models import create_pixel_classifier, train_pixel_classifier
from pixel_classification.visualization import plot_class_distribution

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
    parser.add_argument('--no_scattering', action='store_true', help='Disabilita la trasformata scattering')
    
    # Opzioni di checkpoint e ripresa
    parser.add_argument('--resume', action='store_true', help='Riprendi l\'addestramento da un checkpoint esistente')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Intervallo (in epoche) per salvare i checkpoint')
    
    # Opzioni di ottimizzazione della memoria
    parser.add_argument('--lazy_loading', action='store_true', help='Carica le patch solo quando necessario (riduce l\'uso di memoria)')
    parser.add_argument('--max_patches_in_memory', type=int, default=100000, help='Numero massimo di patch da tenere in memoria')
    parser.add_argument('--max_images', type=int, help='Numero massimo di immagini da elaborare (None = tutte)')
    parser.add_argument('--workers', type=int, default=4, help='Numero di worker per il data loading')
    parser.add_argument('--pin_memory', action='store_true', help='Usa pin_memory per accelerare il trasferimento alla GPU')
    
    # Opzioni per la GPU
    parser.add_argument('--disable_cudnn', action='store_true', help='Disabilita completamente cuDNN (utile se si verificano errori)')
    parser.add_argument('--no_amp', action='store_true', help='Disabilita la precisione mista automatica (AMP)')
    
    # Opzioni per il caching dei metadati
    parser.add_argument('--metadata_cache', type=str, help='File per salvare/caricare i metadati delle patch')
    parser.add_argument('--save_metadata', action='store_true', help='Salva i metadati delle patch dopo l\'estrazione')
    
    # Altre opzioni
    parser.add_argument('--no_augment', action='store_true', help='Disabilita data augmentation')
    parser.add_argument('--seed', type=int, default=42, help='Seed per la riproducibilità')
    parser.add_argument('--num_classes', type=int, default=5, help='Numero di classi (default: 5)')
    parser.add_argument('--class_names', type=str, help='Nomi delle classi separati da virgola')
    parser.add_argument('--quiet', action='store_true', help='Modalità silenziosa (meno output)')
    parser.add_argument('--output_dir', type=str, help='Directory per salvare i risultati (default: directory del modello)')
    
    return parser.parse_args()

def main():
    """Funzione principale per addestrare il classificatore."""
    # Analizza gli argomenti dalla riga di comando
    args = parse_args()
    
    # Imposta seed per la riproducibilità
    set_seed(args.seed)
    
    # Determina il device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    # Configura la directory di output
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model)
    
    # Crea la directory di output se non esiste
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    print(f"Parametri: patch_size={args.patch_size}, stride={args.stride}, augment={not args.no_augment}")
    
    # Disabilita l'output verboso se richiesto
    if args.quiet:
        import sys
        from contextlib import contextmanager
        
        @contextmanager
        def suppress_stdout():
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                yield
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout
        
        with suppress_stdout():
            dataset = PixelWiseDataset(
                images_dir=args.images_dir,
                masks_dir=args.masks_dir,
                patch_size=args.patch_size,
                stride=args.stride,
                augment=not args.no_augment,
                class_mapping=class_mapping,
                lazy_loading=args.lazy_loading,
                max_patches_in_memory=args.max_patches_in_memory,
                max_images=args.max_images,
                verbose=not args.quiet,
                metadata_cache_file=args.metadata_cache,
                save_metadata=args.save_metadata
            )
    else:
        dataset = PixelWiseDataset(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            patch_size=args.patch_size,
            stride=args.stride,
            augment=not args.no_augment,
            class_mapping=class_mapping,
            lazy_loading=args.lazy_loading,
            max_patches_in_memory=args.max_patches_in_memory,
            max_images=args.max_images,
            verbose=not args.quiet,
            metadata_cache_file=args.metadata_cache,
            save_metadata=args.save_metadata
        )
    
    print(f"Dataset creato con {len(dataset)} patch")
    
    # Visualizza la distribuzione delle classi
    plot_class_distribution(
        dataset,
        title="Distribuzione delle classi nel dataset",
        save_path=os.path.join(args.output_dir, "class_distribution.png")
    )
    
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
        learning_rate=args.learning_rate,
        use_scattering=not args.no_scattering
    )
    
    # Stampa informazioni sulla modalità
    print(f"Modalità: {'Senza' if args.no_scattering else 'Con'} trasformata scattering")
    
    # Crea modello e trasformata scattering
    model, scattering = create_pixel_classifier(config)
    
    # Stampa riepilogo della configurazione
    config.print_summary()
    
    # Stampa informazioni sulle ottimizzazioni di memoria
    if args.lazy_loading:
        print(f"Modalità lazy loading attivata: le patch verranno caricate on-demand")
        print(f"Cache massima: {args.max_patches_in_memory} patch in memoria")
    if args.max_images:
        print(f"Limitazione: verranno utilizzate solo {args.max_images} immagini")
    if args.metadata_cache:
        print(f"Cache metadati: {args.metadata_cache}")
        if args.save_metadata:
            print(f"I metadati verranno salvati dopo l'estrazione")
    
    # Registra il tempo di inizio
    start_time = time.time()
    
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
        model=model,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        disable_cudnn=args.disable_cudnn,
        use_amp=not args.no_amp,
        num_workers=args.workers
    )
    
    # Calcola il tempo di addestramento
    training_time = time.time() - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\nAddestramento completato in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"Modello salvato in: {args.model}")
    
    # Visualizza le metriche di addestramento
    from pixel_classification.visualization import plot_training_metrics
    plot_training_metrics(
        history,
        save_path=os.path.join(args.output_dir, "training_metrics.png")
    )
    
    # Mostra informazioni sui checkpoint
    checkpoint_dir = os.path.dirname(args.model)
    checkpoint_base = os.path.splitext(os.path.basename(args.model))[0]
    print(f"\nPer riprendere l'addestramento in futuro, usa:")
    print(f"python train.py --resume --model {args.model} ...")
    
    # Informazioni sul checkpoint temporaneo
    print(f"Checkpoint temporaneo: {os.path.join(checkpoint_dir, checkpoint_base + '_temp.pth')}")
    
    # Suggerimento per il riutilizzo dei metadati
    if args.metadata_cache and args.save_metadata:
        print(f"\nPer saltare l'analisi delle immagini in futuro, usa:")
        print(f"python train.py --metadata_cache {args.metadata_cache} ...")

if __name__ == "__main__":
    main()
