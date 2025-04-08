"""
Utilità per la gestione di maschere in formato RLE (Run-Length Encoding) nel formato COCO.
"""

import numpy as np
import cv2
import json
import pycocotools.mask as mask_util
from pathlib import Path


def load_coco_rle_annotations(annotation_file):
    """
    Carica le annotazioni in formato COCO RLE da un file JSON.
    
    Args:
        annotation_file: Percorso del file JSON con le annotazioni
        
    Returns:
        Lista di dizionari con le annotazioni
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    return annotations


def rle_to_mask(rle_data, shape=None):
    """
    Converte una maschera in formato RLE in un array binario.
    
    Args:
        rle_data: Dizionario con 'counts' e 'size' o stringa RLE
        shape: Forma dell'immagine (height, width) se non specificata in rle_data
        
    Returns:
        Array binario della maschera
    """
    if isinstance(rle_data, dict):
        if 'counts' in rle_data and 'size' in rle_data:
            return mask_util.decode(rle_data)
    
    # Se rle_data è una stringa o non ha il formato atteso
    if shape is None:
        raise ValueError("È necessario specificare la forma dell'immagine se rle_data non contiene 'size'")
    
    # Crea un dizionario RLE nel formato atteso da pycocotools
    rle = {'counts': rle_data if isinstance(rle_data, str) else rle_data['counts'], 
           'size': shape}
    
    return mask_util.decode(rle)


def extract_objects_from_coco_annotations(image, annotations):
    """
    Estrae oggetti da un'immagine utilizzando annotazioni COCO RLE.
    
    Args:
        image: Immagine originale (numpy array)
        annotations: Lista di dizionari con annotazioni COCO RLE
        
    Returns:
        Lista di tuple (oggetto, bbox, maschera_oggetto, classe, punteggio)
    """
    objects = []
    
    for ann in annotations:
        # Estrai informazioni dall'annotazione
        segmentation = ann['segmentation']
        bbox = ann['bbox']
        class_name = ann.get('class', 'unknown')
        score = ann.get('predicted_iou', ann.get('stability_score', 1.0))
        
        # Converti RLE in maschera binaria
        mask = rle_to_mask(segmentation)
        
        # Estrai bounding box
        x, y, w, h = [int(coord) for coord in bbox]
        
        # Aggiungi padding al bounding box (opzionale)
        padding = 5
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(image.shape[1], x + w + padding)
        y_max = min(image.shape[0], y + h + padding)
        
        # Estrai l'oggetto dall'immagine originale
        object_image = image[y_min:y_max, x_min:x_max].copy()
        
        # Estrai la maschera dell'oggetto
        object_mask = mask[y_min:y_max, x_min:x_max]
        
        # Salva l'oggetto, il bounding box, la maschera e la classe
        bbox_with_padding = (x_min, y_min, x_max, y_max)
        objects.append((object_image, bbox_with_padding, object_mask, class_name, score))
    
    return objects


def save_extracted_objects_with_class(objects, output_dir, base_filename, apply_mask=True):
    """
    Salva gli oggetti estratti come immagini separate, organizzate per classe.
    
    Args:
        objects: Lista di tuple (oggetto, bbox, maschera_oggetto, classe, punteggio)
        output_dir: Directory di output
        base_filename: Nome base del file
        apply_mask: Se applicare la maschera all'oggetto
        
    Returns:
        Lista di percorsi dei file salvati
    """
    saved_paths = []
    
    for i, (obj_img, bbox, obj_mask, class_name, score) in enumerate(objects):
        # Crea directory per la classe
        class_dir = Path(output_dir) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Crea nome file
        filename = f"{base_filename}_obj_{i:03d}_{class_name}_{score:.2f}.png"
        filepath = str(class_dir / filename)
        
        # Applica maschera se richiesto
        if apply_mask:
            # Crea una copia dell'immagine
            masked_img = obj_img.copy()
            
            # Crea una maschera RGB per immagini a colori
            if len(obj_img.shape) == 3:
                mask_rgb = np.stack([obj_mask] * 3, axis=2)
                # Applica la maschera (mantieni solo i pixel dell'oggetto)
                masked_img = masked_img * mask_rgb
            else:
                # Per immagini in scala di grigi
                masked_img = masked_img * obj_mask
            
            # Salva l'immagine mascherata
            cv2.imwrite(filepath, masked_img)
        else:
            # Salva l'immagine originale con bounding box
            cv2.imwrite(filepath, obj_img)
        
        saved_paths.append(filepath)
    
    return saved_paths


def create_dataset_from_annotations(images_dir, annotations_dir, output_dir, apply_mask=True):
    """
    Crea un dataset di immagini classificate a partire da immagini e annotazioni COCO RLE.
    
    Args:
        images_dir: Directory contenente le immagini originali
        annotations_dir: Directory contenente i file di annotazione JSON
        output_dir: Directory di output per il dataset
        apply_mask: Se applicare la maschera agli oggetti estratti
        
    Returns:
        Dizionario con statistiche sul dataset creato
    """
    # Crea directory di output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Trova tutte le immagini
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(Path(images_dir).glob(f"**/*{ext}")))
    
    if not image_paths:
        print(f"Nessuna immagine trovata in: {images_dir}")
        return None
    
    print(f"Trovate {len(image_paths)} immagini da processare")
    
    # Statistiche
    stats = {
        'total_images': 0,
        'total_objects': 0,
        'objects_per_class': {},
        'processed_images': []
    }
    
    # Processa ogni immagine
    for img_path in image_paths:
        img_name = img_path.stem
        
        # Cerca il file di annotazione corrispondente
        ann_path = None
        for ext in ['.json']:
            potential_ann = Path(annotations_dir) / f"{img_name}{ext}"
            if potential_ann.exists():
                ann_path = potential_ann
                break
        
        if ann_path is None:
            print(f"Annotazione non trovata per l'immagine: {img_path}")
            continue
        
        try:
            # Carica immagine e annotazioni
            image = cv2.imread(str(img_path))
            annotations = load_coco_rle_annotations(ann_path)
            
            # Estrai oggetti
            objects = extract_objects_from_coco_annotations(image, annotations)
            
            if not objects:
                print(f"Nessun oggetto trovato nell'immagine: {img_path}")
                continue
            
            # Salva oggetti estratti
            saved_paths = save_extracted_objects_with_class(
                objects, output_dir, img_name, apply_mask=apply_mask
            )
            
            # Aggiorna statistiche
            stats['total_images'] += 1
            stats['total_objects'] += len(objects)
            stats['processed_images'].append(str(img_path))
            
            # Conta oggetti per classe
            for _, _, _, class_name, _ in objects:
                if class_name not in stats['objects_per_class']:
                    stats['objects_per_class'][class_name] = 0
                stats['objects_per_class'][class_name] += 1
            
            print(f"Processata immagine {img_path}: {len(objects)} oggetti estratti")
            
        except Exception as e:
            print(f"Errore nel processare {img_path}: {e}")
    
    # Salva statistiche
    with open(output_path / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nCreazione dataset completata:")
    print(f"- Immagini processate: {stats['total_images']}")
    print(f"- Oggetti totali estratti: {stats['total_objects']}")
    print(f"- Oggetti per classe: {stats['objects_per_class']}")
    
    return stats
