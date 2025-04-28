#!/usr/bin/env python3
"""
Script di addestramento per il modulo di classificazione tile.

Questo script coordina l'intero flusso di lavoro per l'addestramento dei modelli di classificazione,
dalla preparazione dei dati all'addestramento e alla valutazione del modello.

Utilizzo:
    python train.py --dataset /path/to/dataset --num-classes N --epochs E [opzioni]
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Importa i moduli del pacchetto
from tile_classification.utils import Config, set_seed
from tile_classification.dataset import BalancedDataset, get_default_transform, create_data_loaders
from tile_classification.models import create_classification_model, print_classifier_summary
from tile_classification.training import Trainer, create_optimizer
from tile_classification.visualization import plot_training_metrics, plot_class_distribution

def parse_args():
    """
    Analizza gli argomenti dalla riga di comando.

    Returns:
        args: Namespace contenente gli argomenti analizzati
    """
    parser = argparse.ArgumentParser(description='Addestra un modello Wavelet Scattering Transform')

    # Parametri del dataset
    parser.add_argument('--dataset', type=str, required=True, help='Percorso al dataset (obbligatorio)')
    parser.add_argument('--balance', action='store_true', help='Bilancia le classi nel dataset')

    # Parametri del modello
    parser.add_argument('--num-classes', type=int, default=4, help='Numero di classi (obbligatorio)')
    parser.add_argument('--num-channels', type=int, default=3, help='Numero di canali di input')
    parser.add_argument('--scattering-order', type=int, default=2, help='Ordine massimo della trasformata scattering')
    parser.add_argument('--j', type=int, default=2, help='Parametro J per la trasformata scattering')

    # Parametri di addestramento
    parser.add_argument('--batch-size', type=int, default=128, help='Dimensione del batch')
    parser.add_argument('--epochs', type=int, default=90, help='Numero di epoche di addestramento (obbligatorio)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate iniziale')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum per l\'ottimizzatore')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay per l\'ottimizzatore')
    parser.add_argument('--reduce-lr-after', type=int, default=20, help='Riduci il learning rate dopo questo numero di epoche')

    # Parametri generali di addestramento
    parser.add_argument('--seed', type=int, default=42, help='Seed per la riproducibilità')
    parser.add_argument('--device', type=str, default=None, help='Device per l\'addestramento (cuda o cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Numero di worker per i dataloader')

    # Parametri di output
    parser.add_argument('--output-dir', type=str, default=None, help='Directory per salvare i risultati')
    parser.add_argument('--experiment-name', type=str, default=None, help='Nome per questo esperimento (usato nel percorso di output)')

    return parser.parse_args()

def main():
    """
    Funzione principale che coordina l'addestramento del modello.
    """
    # Analizza gli argomenti dalla riga di comando
    args = parse_args()

    # Imposta il seed per la riproducibilità
    set_seed(args.seed)

    # Configura la directory di output
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Usa il nome dell'esperimento se fornito
        if args.experiment_name:
            experiment_folder = f"{args.experiment_name}_{args.num_classes}_classi_{timestamp}"
        else:
            experiment_folder = f"output_modello_{args.num_classes}_classi_{timestamp}"

        args.output_dir = os.path.join("risultati", experiment_folder)

    # Crea la directory di output se non esiste
    os.makedirs(args.output_dir, exist_ok=True)

    # Stampa le informazioni di configurazione
    print(f"\n{'='*80}")
    print(f"Addestramento del modello Wavelet Scattering Transform")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Classi: {args.num_classes}")
    print(f"Epoche: {args.epochs}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Crea la configurazione
    config = Config(
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        scattering_order=args.scattering_order,
        J=args.j,
        shape=(32, 32),  # Dimensione fissa per compatibilità
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        device=args.device
    )

    # Stampa il riepilogo della configurazione
    config.print_summary()

    # Prepara il dataset
    print("\nPreparazione del dataset...")
    transform = get_default_transform(target_size=(32, 32), dataset_root=args.dataset)
    dataset = BalancedDataset(args.dataset,
                          transform=transform,
                          balance=args.balance)

    # Visualizza la distribuzione delle classi
    plot_class_distribution(dataset,
                        title="Distribuzione delle classi nel dataset",
                        save_path=os.path.join(args.output_dir, "distribuzione_classi.png"))

    # Crea i dataloader
    train_loader, test_loader = create_data_loaders(dataset,
                                                test_size=0.2,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers)

    # Crea il modello e la trasformata scattering
    model, scattering = create_classification_model(config)

    # Stampa il riepilogo del modello
    print_classifier_summary(model,
                          scattering,
                          config.device)

    # Crea l'ottimizzatore
    optimizer = create_optimizer(model,
                               config)

    # Crea il trainer
    trainer = Trainer(model,
                    scattering,
                    config.device,
                    optimizer)

    # Addestra il modello
    print(f"\nInizio addestramento per {args.epochs} epoche...")
    start_time = time.time()

    training_results = trainer.train(train_loader,
                                   test_loader,
                                   args.epochs,
                                   save_path=os.path.join(args.output_dir, "modello.pth"),
                                   reduce_lr_after=args.reduce_lr_after,
                                   class_to_idx=dataset.class_to_idx)

    # Calcola il tempo di addestramento
    training_time = time.time() - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\nAddestramento completato in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"Miglior accuratezza: {training_results['best_accuracy']:.2f}%")

    # Visualizza le metriche di addestramento
    plot_training_metrics(args.epochs,
                        training_results['train_accuracies'],
                        training_results['test_accuracies'],
                        training_results['train_losses'],
                        training_results['test_losses'],
                        os.path.join(args.output_dir, "metriche_addestramento.png"))

    print(f"\nRisultati salvati in {args.output_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
