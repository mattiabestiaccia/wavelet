#!/usr/bin/env python3
"""
Script di esecuzione per la classificazione di oggetti segmentati nella Wavelet Scattering Transform Library.

Questo script esegue la classificazione di oggetti estratti da immagini segmentate
utilizzando un modello pre-addestrato con trasformata wavelet scattering.

Utilizzo:
    python script/core/segmented_object_classification/run_segmented_classification.py --image /path/to/image.jpg --mask /path/to/mask.png --model /path/to/model.pth
    python script/core/segmented_object_classification/run_segmented_classification.py --images_dir /path/to/images --masks_dir /path/to/masks --model /path/to/model.pth --output /path/to/output
    python script/core/segmented_object_classification/run_segmented_classification.py --image /path/to/image.jpg --annotation /path/to/annotation.json --model /path/to/model.pth --output /path/to/output
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Aggiungi la directory principale al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from wavelet_lib.segmented_object_classification.models import (
    create_scattering_transform,
    load_segmented_object_classifier
)
from wavelet_lib.segmented_object_classification.processors import (
    SegmentedObjectProcessor,
    extract_objects_from_mask
)
from wavelet_lib.segmented_object_classification.rle_utils import (
    load_coco_rle_annotations,
    extract_objects_from_coco_annotations
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Classificazione di oggetti segmentati con WST')

    # Modalità di input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Percorso dell\'immagine da processare')
    input_group.add_argument('--images_dir', type=str, help='Directory contenente le immagini da processare')

    # Modalità di segmentazione
    segmentation_group = parser.add_mutually_exclusive_group(required=True)
    segmentation_group.add_argument('--mask', type=str, help='Percorso della maschera di segmentazione')
    segmentation_group.add_argument('--masks_dir', type=str, help='Directory contenente le maschere di segmentazione')
    segmentation_group.add_argument('--annotation', type=str, help='Percorso del file di annotazione COCO RLE')
    segmentation_group.add_argument('--annotations_dir', type=str, help='Directory contenente i file di annotazione COCO RLE')

    # Modello e output
    parser.add_argument('--model', type=str, required=True, help='Percorso del modello di classificazione')
    parser.add_argument('--output', type=str, help='Directory di output per i risultati')

    # Parametri opzionali
    parser.add_argument('--j', type=int, default=2, help='Numero di scale per la trasformata scattering')
    parser.add_argument('--input_size', type=str, default='32,32', help='Dimensione di input per il modello (altezza,larghezza)')
    parser.add_argument('--min_size', type=int, default=100, help='Dimensione minima degli oggetti in pixel')
    parser.add_argument('--no_mask_apply', action='store_true', help='Non applicare la maschera agli oggetti estratti')
    parser.add_argument('--no_save_objects', action='store_true', help='Non salvare gli oggetti estratti')
    parser.add_argument('--visualize', action='store_true', help='Visualizza i risultati')
    parser.add_argument('--num_classes', type=int, default=7, help='Numero di classi del modello')
    parser.add_argument('--class_names', type=str, help='Nomi delle classi separati da virgola')

    return parser.parse_args()


def main(args):
    """Funzione principale per eseguire la classificazione."""
    # Parsing delle dimensioni di input
    input_size = tuple(map(int, args.input_size.split(',')))

    # Determina il device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")

    # Crea trasformata scattering
    scattering = create_scattering_transform(
        J=args.j,
        shape=input_size,
        max_order=2,
        device=device
    )

    # Carica modello
    print(f"Caricamento modello da: {args.model}")
    model, class_to_idx = load_segmented_object_classifier(
        model_path=args.model,
        num_classes=args.num_classes,
        device=device
    )

    # Se il mapping delle classi non è disponibile, creane uno predefinito o usa quello fornito
    if class_to_idx is None:
        if args.class_names:
            class_names = args.class_names.split(',')
            class_to_idx = {name: i for i, name in enumerate(class_names)}
        else:
            # Classi predefinite
            class_names = ['vegetazione_bassa', 'vegetazione_alta', 'strade', 'corsi_acqua', 'acqua_esterna', 'edifici', 'mudflat']
            class_to_idx = {name: i for i, name in enumerate(class_names)}

    print(f"Classi del modello: {class_to_idx}")

    # Crea processore
    processor = SegmentedObjectProcessor(
        model=model,
        scattering=scattering,
        class_mapping=class_to_idx,
        device=device,
        target_size=input_size
    )

    # Processa input in base agli argomenti
    if args.image and args.mask:
        # Processa una singola immagine con maschera binaria
        print(f"Processamento immagine: {args.image} con maschera: {args.mask}")
        result = processor.process_from_segmentation(
            image_path=args.image,
            mask_path=args.mask,
            output_dir=args.output,
            min_size=args.min_size,
            apply_mask=not args.no_mask_apply,
            save_objects=not args.no_save_objects,
            visualize=args.visualize
        )

        # Stampa risultati
        print(f"\nRisultati della classificazione:")
        print(f"Numero di oggetti: {result['num_objects']}")
        print(f"Distribuzione delle classi: {result['class_distribution']}")

    elif args.image and args.annotation:
        # Processa una singola immagine con annotazione COCO RLE
        print(f"Processamento immagine: {args.image} con annotazione: {args.annotation}")

        # Carica immagine e annotazioni
        import cv2
        image = cv2.imread(args.image)
        annotations = load_coco_rle_annotations(args.annotation)

        # Estrai oggetti
        objects = extract_objects_from_coco_annotations(image, annotations)
        print(f"Estratti {len(objects)} oggetti dall'immagine")

        # Crea directory di output se necessario
        if args.output:
            os.makedirs(args.output, exist_ok=True)

            # Salva oggetti se richiesto
            if not args.no_save_objects:
                from wavelet_lib.segmented_object_classification.rle_utils import save_extracted_objects_with_class
                base_filename = os.path.splitext(os.path.basename(args.image))[0]
                objects_dir = os.path.join(args.output, "objects")
                saved_paths = save_extracted_objects_with_class(
                    objects, objects_dir, base_filename, apply_mask=not args.no_mask_apply
                )

        # Classifica oggetti
        results = []
        for i, (obj_img, bbox, obj_mask, class_name, score) in enumerate(objects):
            # Applica maschera se richiesto
            if not args.no_mask_apply:
                # Crea una maschera RGB per immagini a colori
                if len(obj_img.shape) == 3:
                    mask_rgb = np.stack([obj_mask] * 3, axis=2)
                    # Applica la maschera (mantieni solo i pixel dell'oggetto)
                    masked_img = obj_img * mask_rgb
                else:
                    # Per immagini in scala di grigi
                    masked_img = obj_img * obj_mask

                # Classifica l'immagine mascherata
                result = processor.process_object(masked_img)
            else:
                # Classifica l'immagine originale
                result = processor.process_object(obj_img)

            # Aggiungi informazioni sull'oggetto
            result['object_id'] = i
            result['bbox'] = bbox
            result['size'] = obj_img.shape[:2]
            result['area'] = np.sum(obj_mask)
            result['original_class'] = class_name
            result['original_score'] = score

            results.append(result)

        # Calcola statistiche
        from collections import Counter
        class_distribution = Counter([r['class'] for r in results])

        # Crea risultato finale
        final_result = {
            'image_path': args.image,
            'annotation_path': args.annotation,
            'num_objects': len(objects),
            'class_distribution': dict(class_distribution),
            'objects': results
        }

        # Salva risultati se richiesto
        if args.output:
            import json
            base_filename = os.path.splitext(os.path.basename(args.image))[0]
            results_path = os.path.join(args.output, f"{base_filename}_classification.json")
            with open(results_path, 'w') as f:
                json.dump(final_result, f, indent=2)

        # Visualizza risultati se richiesto
        if args.visualize:
            processor.visualize_results(image, np.zeros_like(image[:,:,0]), results, args.output)

        # Stampa risultati
        print(f"\nRisultati della classificazione:")
        print(f"Numero di oggetti: {final_result['num_objects']}")
        print(f"Distribuzione delle classi: {final_result['class_distribution']}")

    elif args.images_dir and args.masks_dir:
        # Processa una cartella di immagini con maschere binarie
        print(f"Processamento cartella: {args.images_dir} con maschere: {args.masks_dir}")
        result = processor.process_folder(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            output_dir=args.output,
            min_size=args.min_size,
            apply_mask=not args.no_mask_apply,
            save_objects=not args.no_save_objects,
            visualize=args.visualize
        )

        if result:
            # Stampa risultati
            print(f"\nRisultati della classificazione:")
            print(f"Immagini processate: {result['total_images']}")
            print(f"Oggetti totali: {result['total_objects']}")
            print(f"Distribuzione globale delle classi: {result['global_class_distribution']}")

    elif args.images_dir and args.annotations_dir:
        # Processa una cartella di immagini con annotazioni COCO RLE
        print(f"Processamento cartella: {args.images_dir} con annotazioni: {args.annotations_dir}")

        # Crea dataset e classifica
        from wavelet_lib.segmented_object_classification.rle_utils import create_dataset_from_annotations
        stats = create_dataset_from_annotations(
            images_dir=args.images_dir,
            annotations_dir=args.annotations_dir,
            output_dir=args.output,
            apply_mask=not args.no_mask_apply
        )

        if stats:
            print(f"\nRisultati della classificazione:")
            print(f"Immagini processate: {stats['total_images']}")
            print(f"Oggetti totali estratti: {stats['total_objects']}")
            print(f"Distribuzione degli oggetti per classe: {stats['objects_per_class']}")

    print("\nClassificazione completata.")


if __name__ == "__main__":
    args = parse_args()
    main(args)