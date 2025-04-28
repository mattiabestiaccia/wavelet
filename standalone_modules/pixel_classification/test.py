#!/usr/bin/env python3
"""
Script di test per il modulo di classificazione pixel-wise.

Questo script permette di testare un modello di classificazione pixel-wise su un'immagine
o su un dataset di test, calcolando metriche di valutazione.

Utilizzo:
    python test.py --model /path/to/model.pth --images_dir /path/to/images --masks_dir /path/to/masks [opzioni]
    python test.py --model /path/to/model.pth --image /path/to/image.jpg --mask /path/to/mask.png [opzioni]
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Importa i moduli del pacchetto
from pixel_classification.models import PixelWiseClassifier, create_scattering_transform
from pixel_classification.utils import load_model
from pixel_classification.visualization import visualize_results
from pixel_classification.dataset import PixelWiseDataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test di un classificatore pixel-wise')
    
    # Modalità di test
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Percorso dell\'immagine da testare')
    group.add_argument('--images_dir', type=str, help='Directory contenente le immagini di test')
    
    # Input e output
    parser.add_argument('--model', type=str, required=True, help='Percorso del modello addestrato')
    parser.add_argument('--mask', type=str, help='Percorso della maschera di verità (per --image)')
    parser.add_argument('--masks_dir', type=str, help='Directory contenente le maschere di verità (per --images_dir)')
    parser.add_argument('--output_dir', type=str, help='Directory dove salvare i risultati')
    
    # Parametri di test
    parser.add_argument('--patch_size', type=int, default=32, help='Dimensione delle patch')
    parser.add_argument('--stride', type=int, default=16, help='Passo per l\'estrazione delle patch')
    parser.add_argument('--batch_size', type=int, default=16, help='Dimensione del batch')
    parser.add_argument('--max_images', type=int, help='Numero massimo di immagini da testare')
    
    # Opzioni per la GPU
    parser.add_argument('--disable_cudnn', action='store_true', help='Disabilita completamente cuDNN')
    parser.add_argument('--no_amp', action='store_true', help='Disabilita la precisione mista automatica (AMP)')
    
    # Altre opzioni
    parser.add_argument('--class_names', type=str, help='Nomi delle classi separati da virgola')
    parser.add_argument('--max_size', type=int, default=1024, help='Dimensione massima dell\'immagine (ridimensiona se più grande)')
    
    return parser.parse_args()

def predict_image(model, scattering, image_path, mask_path=None, patch_size=32, stride=16, batch_size=16, device=None, use_amp=True, max_size=None):
    """
    Classifica un'immagine utilizzando un modello pixel-wise.
    
    Args:
        model: Modello addestrato
        scattering: Trasformata scattering
        image_path: Percorso dell'immagine
        mask_path: Percorso della maschera di verità (opzionale)
        patch_size: Dimensione delle patch
        stride: Passo per l'estrazione delle patch
        batch_size: Dimensione del batch
        device: Device da utilizzare
        use_amp: Se utilizzare la precisione mista automatica
        max_size: Dimensione massima dell'immagine
        
    Returns:
        Immagine originale, maschera di verità e predizione
    """
    # Imposta il device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica l'immagine
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Impossibile caricare l'immagine {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Carica la maschera se specificata
    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Ridimensiona l'immagine se necessario
    original_size = None
    if max_size is not None:
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            # Calcola il fattore di scala
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Salva le dimensioni originali per riferimento
            original_size = (h, w)
            
            # Ridimensiona l'immagine
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Ridimensiona anche la maschera se presente
            if mask is not None:
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            print(f"Immagine ridimensionata da {h}x{w} a {new_h}x{new_w}")
    
    # Crea una mappa di predizione vuota
    h, w = img.shape[:2]
    prediction = np.zeros((h, w), dtype=np.uint8)
    
    # Estrai patch dall'immagine
    patches = []
    coords = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((x, y))
    
    # Converti le patch in tensori
    tensor_patches = []
    for patch in patches:
        tensor_patch = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
        tensor_patches.append(tensor_patch)
    
    # Classifica le patch in batch
    num_patches = len(patches)
    num_batches = (num_patches + batch_size - 1) // batch_size
    
    print(f"Classificazione di {num_patches} patch in {num_batches} batch...")
    
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_patches)
            
            batch_patches = torch.stack(tensor_patches[start_idx:end_idx]).to(device)
            
            # Forward pass
            if scattering is not None and model.use_scattering:
                with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda' and use_amp):
                    # Applica la trasformata scattering
                    scattering_coeffs = scattering(batch_patches)
                    
                    # Gestisci la dimensionalità dell'output
                    if scattering_coeffs.dim() == 5:
                        if scattering_coeffs.shape[-1] == 1:
                            scattering_coeffs = scattering_coeffs.squeeze(-1)
                        else:
                            scattering_coeffs = scattering_coeffs[..., 0]
                    
                    # Passa i coefficienti al modello
                    outputs = model(scattering_coeffs)
            else:
                with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda' and use_amp):
                    outputs = model(batch_patches)
            
            # Ottieni le predizioni
            _, predicted = outputs.max(1)
            
            # Aggiorna la mappa di predizione
            for j in range(end_idx - start_idx):
                x, y = coords[start_idx + j]
                pred_patch = predicted[j].cpu().numpy()
                
                # Ottieni le dimensioni effettive della patch di predizione
                pred_h, pred_w = pred_patch.shape
                
                # Usa il voto di maggioranza per ogni pixel
                for py in range(min(patch_size, pred_h)):
                    for px in range(min(patch_size, pred_w)):
                        if y + py < h and x + px < w:
                            prediction[y + py, x + px] = pred_patch[py, px]
    
    # Ridimensiona la predizione alle dimensioni originali se necessario
    if original_size is not None:
        prediction = cv2.resize(prediction, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (original_size[1], original_size[0]), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return img, mask, prediction

def calculate_metrics(mask, prediction, num_classes):
    """
    Calcola le metriche di valutazione.
    
    Args:
        mask: Maschera di verità
        prediction: Predizione del modello
        num_classes: Numero di classi
        
    Returns:
        Dizionario con le metriche
    """
    # Appiattisci le maschere
    mask_flat = mask.flatten()
    pred_flat = prediction.flatten()
    
    # Calcola la matrice di confusione
    cm = confusion_matrix(mask_flat, pred_flat, labels=range(num_classes))
    
    # Calcola le metriche
    accuracy = accuracy_score(mask_flat, pred_flat)
    
    # Calcola precision, recall e F1-score per ogni classe
    precision = precision_score(mask_flat, pred_flat, average=None, labels=range(num_classes), zero_division=0)
    recall = recall_score(mask_flat, pred_flat, average=None, labels=range(num_classes), zero_division=0)
    f1 = f1_score(mask_flat, pred_flat, average=None, labels=range(num_classes), zero_division=0)
    
    # Calcola le metriche medie (escludendo lo sfondo)
    precision_mean = np.mean(precision[1:]) if len(precision) > 1 else 0
    recall_mean = np.mean(recall[1:]) if len(recall) > 1 else 0
    f1_mean = np.mean(f1[1:]) if len(f1) > 1 else 0
    
    # Calcola IoU per ogni classe
    iou = np.zeros(num_classes)
    for i in range(num_classes):
        iou[i] = cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i] + 1e-10)
    
    # Calcola mIoU (escludendo lo sfondo)
    miou = np.mean(iou[1:]) if len(iou) > 1 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'precision_mean': precision_mean,
        'recall_mean': recall_mean,
        'f1_mean': f1_mean,
        'miou': miou,
        'confusion_matrix': cm
    }

def print_metrics(metrics, class_mapping):
    """
    Stampa le metriche di valutazione.
    
    Args:
        metrics: Dizionario con le metriche
        class_mapping: Mappatura delle classi
    """
    print("\n" + "="*80)
    print(" "*30 + "METRICHE DI VALUTAZIONE" + " "*30)
    print("="*80)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean Precision: {metrics['precision_mean']:.4f}")
    print(f"Mean Recall: {metrics['recall_mean']:.4f}")
    print(f"Mean F1-score: {metrics['f1_mean']:.4f}")
    print(f"Mean IoU: {metrics['miou']:.4f}")
    
    print("\nMetriche per classe:")
    for cls in range(len(metrics['precision'])):
        if cls in class_mapping:
            print(f"  Classe {cls} ({class_mapping[cls]}):")
            print(f"    Precision: {metrics['precision'][cls]:.4f}")
            print(f"    Recall: {metrics['recall'][cls]:.4f}")
            print(f"    F1-score: {metrics['f1'][cls]:.4f}")
            print(f"    IoU: {metrics['iou'][cls]:.4f}")
    
    print("\nMatrice di confusione:")
    print(metrics['confusion_matrix'])
    print("="*80)

def main():
    """Funzione principale per il test."""
    # Analizza gli argomenti dalla riga di comando
    args = parse_args()
    
    # Verifica se il modello esiste
    if not os.path.exists(args.model):
        print(f"Errore: Il modello {args.model} non esiste")
        sys.exit(1)
    
    # Determina il device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    # Configurazione per stabilità su GPU
    if device.type == 'cuda':
        if args.disable_cudnn:
            torch.backends.cudnn.enabled = False
            print("GPU: cuDNN completamente disabilitato")
        else:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print("GPU: configurazione ottimizzata per stabilità")
    
    # Carica il modello
    print(f"Caricamento del modello da {args.model}...")
    checkpoint = load_model(args.model, device)
    
    # Estrai i parametri
    model_state_dict = checkpoint['model_state_dict']
    class_mapping = checkpoint['class_mapping']
    scattering_params = checkpoint['scattering_params']
    use_scattering = checkpoint['use_scattering']
    
    # Crea la trasformata scattering se necessario
    if use_scattering and scattering_params:
        scattering = create_scattering_transform(
            J=scattering_params.get('J', 2),
            shape=scattering_params.get('shape', (args.patch_size, args.patch_size)),
            max_order=scattering_params.get('max_order', 2),
            device=device
        )
    else:
        scattering = None
    
    # Determina il numero di canali di input
    if use_scattering and scattering:
        # Calcola il numero di canali di input
        dummy_input = torch.randn(1, 3, *scattering_params.get('shape', (args.patch_size, args.patch_size))).to(device)
        scattering_output = scattering(dummy_input)
        
        # Gestisci la dimensionalità dell'output
        if scattering_output.dim() == 5:
            if scattering_output.shape[-1] == 1:
                scattering_output = scattering_output.squeeze(-1)
            else:
                scattering_output = scattering_output[..., 0]
        
        in_channels = scattering_output.shape[1]
    else:
        in_channels = 3  # RGB
    
    # Crea il modello
    model = PixelWiseClassifier(
        in_channels=in_channels,
        hidden_dim=128,
        num_classes=len(class_mapping),
        use_scattering=use_scattering
    ).to(device)
    
    # Carica i pesi
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Sostituisci i nomi delle classi se specificati
    if args.class_names:
        class_names = args.class_names.split(',')
        if len(class_names) != len(class_mapping):
            print(f"Attenzione: Il numero di nomi di classe specificati ({len(class_names)}) "
                  f"non corrisponde al numero di classi nel modello ({len(class_mapping)})")
        
        # Aggiorna il mapping delle classi
        for i, name in enumerate(class_names):
            if i in class_mapping:
                class_mapping[i] = name
    
    print(f"Modello caricato con {len(class_mapping)} classi: {class_mapping}")
    
    # Crea la directory di output se specificata
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Modalità di test
    if args.image:
        # Test su una singola immagine
        if not os.path.exists(args.image):
            print(f"Errore: L'immagine {args.image} non esiste")
            sys.exit(1)
        
        # Classifica l'immagine
        img, mask, prediction = predict_image(
            model,
            scattering,
            args.image,
            args.mask,
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
            use_amp=not args.no_amp,
            max_size=args.max_size
        )
        
        # Calcola le metriche se la maschera è disponibile
        if mask is not None:
            metrics = calculate_metrics(mask, prediction, len(class_mapping))
            print_metrics(metrics, class_mapping)
        
        # Visualizza i risultati
        output_path = None
        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"result_{Path(args.image).stem}.png")
        
        visualize_results(
            img,
            mask if mask is not None else np.zeros_like(prediction),
            prediction,
            class_mapping,
            save_path=output_path
        )
    else:
        # Test su un dataset
        if not os.path.exists(args.images_dir):
            print(f"Errore: La directory {args.images_dir} non esiste")
            sys.exit(1)
        
        if not args.masks_dir or not os.path.exists(args.masks_dir):
            print(f"Attenzione: La directory delle maschere {args.masks_dir} non esiste o non è specificata")
            print("Il test verrà eseguito senza calcolare le metriche")
            args.masks_dir = None
        
        # Trova tutte le immagini
        image_paths = sorted(list(Path(args.images_dir).glob("*.jpg")) +
                           list(Path(args.images_dir).glob("*.JPG")) +
                           list(Path(args.images_dir).glob("*.png")) +
                           list(Path(args.images_dir).glob("*.PNG")) +
                           list(Path(args.images_dir).glob("*.tif")) +
                           list(Path(args.images_dir).glob("*.TIF")))
        
        # Limita il numero di immagini se specificato
        if args.max_images and args.max_images < len(image_paths):
            image_paths = image_paths[:args.max_images]
        
        print(f"Test su {len(image_paths)} immagini...")
        
        # Inizializza le metriche aggregate
        all_metrics = {
            'accuracy': [],
            'precision_mean': [],
            'recall_mean': [],
            'f1_mean': [],
            'miou': []
        }
        
        # Processa ogni immagine
        for img_path in tqdm(image_paths, desc="Elaborazione immagini"):
            # Trova la maschera corrispondente
            mask_path = None
            if args.masks_dir:
                possible_mask_paths = [
                    Path(args.masks_dir) / f"{img_path.stem}_mask.png",
                    Path(args.masks_dir) / f"{img_path.stem}.png",
                    Path(args.masks_dir) / f"{img_path.stem}.jpg",
                    Path(args.masks_dir) / f"{img_path.stem}.tif"
                ]
                
                for p in possible_mask_paths:
                    if p.exists():
                        mask_path = p
                        break
            
            # Classifica l'immagine
            img, mask, prediction = predict_image(
                model,
                scattering,
                img_path,
                mask_path,
                patch_size=args.patch_size,
                stride=args.stride,
                batch_size=args.batch_size,
                device=device,
                use_amp=not args.no_amp,
                max_size=args.max_size
            )
            
            # Calcola le metriche se la maschera è disponibile
            if mask is not None:
                metrics = calculate_metrics(mask, prediction, len(class_mapping))
                
                # Aggiorna le metriche aggregate
                for key in all_metrics.keys():
                    all_metrics[key].append(metrics[key])
            
            # Salva i risultati se specificato
            if args.output_dir:
                output_path = os.path.join(args.output_dir, f"result_{img_path.stem}.png")
                visualize_results(
                    img,
                    mask if mask is not None else np.zeros_like(prediction),
                    prediction,
                    class_mapping,
                    save_path=output_path
                )
        
        # Calcola le metriche medie
        if all_metrics['accuracy']:
            print("\n" + "="*80)
            print(" "*30 + "METRICHE MEDIE" + " "*30)
            print("="*80)
            
            print(f"Accuracy: {np.mean(all_metrics['accuracy']):.4f}")
            print(f"Mean Precision: {np.mean(all_metrics['precision_mean']):.4f}")
            print(f"Mean Recall: {np.mean(all_metrics['recall_mean']):.4f}")
            print(f"Mean F1-score: {np.mean(all_metrics['f1_mean']):.4f}")
            print(f"Mean IoU: {np.mean(all_metrics['miou']):.4f}")
            print("="*80)
    
    print("\nTest completato con successo!")

if __name__ == "__main__":
    main()
