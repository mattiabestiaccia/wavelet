#!/usr/bin/env python3
"""
Script per creare un dataset di immagini classificate a partire da immagini e annotazioni COCO RLE.

Questo script estrae oggetti segmentati da immagini utilizzando annotazioni in formato COCO RLE
e li organizza in un dataset strutturato per l'addestramento di un classificatore.

Utilizzo:
    python script/core/segmented_object_classification/create_dataset_from_annotations.py --images_dir /path/to/images --annotations_dir /path/to/annotations --output_dir /path/to/output
"""

import os
import sys
import argparse
from pathlib import Path

# Aggiungi la directory principale al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from wavelet_lib.segmented_object_classification.rle_utils import create_dataset_from_annotations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Creazione dataset da annotazioni COCO RLE')
    
    # Input e output
    parser.add_argument('--images_dir', type=str, required=True, help='Directory contenente le immagini originali')
    parser.add_argument('--annotations_dir', type=str, required=True, help='Directory contenente i file di annotazione JSON')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory di output per il dataset')
    
    # Opzioni
    parser.add_argument('--no_mask_apply', action='store_true', help='Non applicare la maschera agli oggetti estratti')
    
    return parser.parse_args()


def main(args):
    """Funzione principale per creare il dataset."""
    print(f"Creazione dataset da annotazioni COCO RLE")
    print(f"Directory immagini: {args.images_dir}")
    print(f"Directory annotazioni: {args.annotations_dir}")
    print(f"Directory output: {args.output_dir}")
    
    # Crea dataset
    stats = create_dataset_from_annotations(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        apply_mask=not args.no_mask_apply
    )
    
    if stats:
        print("\nCreazione dataset completata con successo!")
        print(f"Immagini processate: {stats['total_images']}")
        print(f"Oggetti totali estratti: {stats['total_objects']}")
        print(f"Distribuzione degli oggetti per classe:")
        for cls, count in stats['objects_per_class'].items():
            print(f"  - {cls}: {count}")
    else:
        print("\nErrore nella creazione del dataset.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
