#!/usr/bin/env python3
"""
Script di predizione per il modulo di classificazione tile.

Questo script permette di classificare immagini utilizzando modelli di classificazione addestrati,
supportando sia la classificazione dell'intera immagine che l'analisi basata su tile.

Utilizzo:
    python predict.py --model-path /path/to/model.pth --image-path /path/to/image.jpg [opzioni]
    python predict.py --model-path /path/to/model.pth --image-path /path/to/image.jpg --tile-mode [opzioni]
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importa i moduli del pacchetto
from tile_classification.models import ScatteringClassifier, create_scattering_transform
from tile_classification.processors import ClassificationProcessor
from tile_classification.utils import load_model
from tile_classification.visualization import visualize_classification_results

def parse_args():
    """
    Analizza gli argomenti dalla riga di comando.

    Returns:
        args: Namespace contenente gli argomenti analizzati
    """
    parser = argparse.ArgumentParser(description='Fai predizioni con un modello Wavelet Scattering Transform')

    # Parametri del modello
    parser.add_argument('--model-path', type=str, required=True, help='Percorso del file del modello')
    parser.add_argument('--image-path', type=str, required=True, help='Percorso dell\'immagine da classificare')

    # Parametri di predizione
    parser.add_argument('--tile-mode', action='store_true', help='Abilita la modalità tile')
    parser.add_argument('--tile-size', type=int, default=32, help='Dimensione del tile')
    parser.add_argument('--process-30x30', action='store_true', help='Elabora tile 30x30 (ritagliati in alcuni dataset)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Soglia di confidenza per la visualizzazione')

    # Parametri generali
    parser.add_argument('--device', type=str, default=None, help='Device per l\'inferenza (cuda o cpu)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory per salvare i risultati')
    parser.add_argument('--dataset-root', type=str, default=None,
                       help='Percorso alla directory radice del dataset (per i nomi delle classi, opzionale)')

    return parser.parse_args()

def main():
    """
    Funzione principale per la predizione delle immagini.
    """
    # Analizza gli argomenti dalla riga di comando
    args = parse_args()

    # Verifica se l'immagine esiste
    if not os.path.exists(args.image_path):
        print(f"Errore: File immagine non trovato: {args.image_path}")
        sys.exit(1)

    # Verifica se il modello esiste
    if not os.path.exists(args.model_path):
        print(f"Errore: File modello non trovato: {args.model_path}")
        sys.exit(1)

    # Configura il device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*80}")
    print(f"Predizione con modello Wavelet Scattering Transform")
    print(f"{'='*80}")
    print(f"Modello: {args.model_path}")
    print(f"Immagine: {args.image_path}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Carica il modello
    print("Caricamento del modello...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Ottieni i nomi delle classi
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        class_names = list(class_to_idx.keys())
    else:
        print("Attenzione: Mappatura delle classi non trovata nel file del modello.")

        # Prova a ottenere i nomi delle classi dalla directory radice del dataset
        if args.dataset_root and os.path.exists(args.dataset_root):
            class_names = sorted([d for d in os.listdir(args.dataset_root)
                            if os.path.isdir(os.path.join(args.dataset_root, d))])
            print(f"Nomi delle classi dal dataset: {class_names}")
        else:
            class_names = [f"Classe {i}" for i in range(10)]  # Fallback generico

    print(f"Classi rilevate: {class_names}")

    # Crea la trasformata scattering
    scattering_params = checkpoint.get('scattering_params', {})
    J = scattering_params.get('J', 2)
    shape = scattering_params.get('shape', (32, 32))
    max_order = scattering_params.get('max_order', 2)
    
    scattering = create_scattering_transform(
        J=J,
        shape=shape,
        max_order=max_order,
        device=device
    )

    # Crea il modello da zero con la stessa architettura
    num_classes = len(class_names)

    # Prova a ottenere il numero di canali dal checkpoint del modello
    if 'model_state_dict' in checkpoint:
        # Cerca il primo layer di batch normalization per ottenere il conteggio dei canali
        for key, value in checkpoint['model_state_dict'].items():
            if 'bn.weight' in key:
                num_channels = value.size(0)
                print(f"Canali di input rilevati dal checkpoint: {num_channels}")
                break
    else:
        # Fallback al valore predefinito se non trovato
        num_channels = 12
        print(f"Utilizzo del numero di canali predefinito: {num_channels}")

    # Crea il modello con il numero corretto di canali
    model = ScatteringClassifier(in_channels=num_channels, num_classes=num_classes).to(device)

    # Carica i pesi
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("Errore: Impossibile trovare i pesi del modello nel checkpoint")
        return

    model.eval()
    print("Modello caricato con successo.")

    # Configura la directory di output
    if args.output_dir is None:
        filename = os.path.basename(args.image_path)
        base_filename, _ = os.path.splitext(filename)
        args.output_dir = os.path.join("risultati", "classificazione", base_filename)

    os.makedirs(args.output_dir, exist_ok=True)

    # Esegui la predizione
    if args.tile_mode:
        # Modalità tile
        print(f"Esecuzione della predizione in modalità tile (dimensione: {args.tile_size})...")

        # Crea il processore di immagini
        processor = ClassificationProcessor(model, scattering, device, class_names)
        
        # Classifica l'immagine
        results = processor.classify_image_tiles(
            args.image_path,
            tile_size=args.tile_size,
            process_30x30_tiles=args.process_30x30,
            confidence_threshold=args.confidence_threshold
        )

        # Visualizza e salva i risultati
        save_path = os.path.join(args.output_dir, "classificazione_tile.png")
        visualize_classification_results(results, save_path=save_path)

        # Stampa il riepilogo della distribuzione delle classi
        print("\nRIEPILOGO DELLA DISTRIBUZIONE DELLE CLASSI:")
        print("-" * 50)
        total_tiles = results['total_tiles']
        classified_tiles = sum(results['class_counts'].values())
        print(f"Tile totali: {total_tiles}")
        print(f"Tile classificati (confidenza ≥ {args.confidence_threshold}): {classified_tiles} ({classified_tiles/total_tiles*100:.1f}%)")
        print(f"Tile non classificati (confidenza < {args.confidence_threshold}): {total_tiles - classified_tiles} ({(total_tiles - classified_tiles)/total_tiles*100:.1f}%)")
        print("-" * 50)
        for class_idx, count in results['class_counts'].items():
            class_name = class_names[class_idx]
            percentage = 100 * count / total_tiles
            print(f"{class_name}: {count} tile ({percentage:.1f}%)")
    else:
        # Modalità immagine singola
        print("Esecuzione della predizione sull'intera immagine...")

        # Crea il processore di immagini
        processor = ClassificationProcessor(model, scattering, device, class_names)
        
        # Classifica l'immagine
        result = processor.process_image(args.image_path)

        print(f"\nRisultato della classificazione:")
        print(f"Classe: {result['class_name']}")
        print(f"Confidenza: {result['confidence']:.4f}")

        # Visualizza e salva l'immagine con l'etichetta
        img = Image.open(args.image_path).convert('RGB')
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(img))
        plt.title(f"Classe: {result['class_name']}\nConfidenza: {result['confidence']:.4f}")
        plt.axis('off')

        save_path = os.path.join(args.output_dir, "risultato_classificazione.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualizzazione salvata in: {save_path}")
        plt.show()

    print(f"\nPredizione completata!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
