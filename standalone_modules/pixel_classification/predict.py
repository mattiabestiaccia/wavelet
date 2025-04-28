#!/usr/bin/env python3
"""
Script di predizione per il modulo di classificazione pixel-wise.

Questo script permette di classificare immagini utilizzando modelli di classificazione pixel-wise addestrati.

Utilizzo:
    python predict.py --model /path/to/model.pth --image /path/to/image.jpg [opzioni]
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Importa i moduli del pacchetto
from pixel_classification.models import PixelWiseClassifier, create_scattering_transform
from pixel_classification.utils import load_model
from pixel_classification.visualization import visualize_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predizione con classificatore pixel-wise')
    
    # Input e output
    parser.add_argument('--model', type=str, required=True, help='Percorso del modello addestrato')
    parser.add_argument('--image', type=str, required=True, help='Percorso dell\'immagine da classificare')
    parser.add_argument('--mask', type=str, help='Percorso della maschera di verità (opzionale)')
    parser.add_argument('--output', type=str, help='Percorso dove salvare il risultato')
    
    # Parametri di predizione
    parser.add_argument('--patch_size', type=int, default=32, help='Dimensione delle patch')
    parser.add_argument('--stride', type=int, default=16, help='Passo per l\'estrazione delle patch')
    parser.add_argument('--batch_size', type=int, default=16, help='Dimensione del batch')
    
    # Opzioni per la GPU
    parser.add_argument('--disable_cudnn', action='store_true', help='Disabilita completamente cuDNN')
    parser.add_argument('--no_amp', action='store_true', help='Disabilita la precisione mista automatica (AMP)')
    
    # Altre opzioni
    parser.add_argument('--class_names', type=str, help='Nomi delle classi separati da virgola')
    parser.add_argument('--max_size', type=int, default=1024, help='Dimensione massima dell\'immagine (ridimensiona se più grande)')
    
    return parser.parse_args()

def predict_image(model, scattering, image_path, patch_size=32, stride=16, batch_size=16, device=None, use_amp=True, max_size=None):
    """
    Classifica un'immagine utilizzando un modello pixel-wise.
    
    Args:
        model: Modello addestrato
        scattering: Trasformata scattering
        image_path: Percorso dell'immagine
        patch_size: Dimensione delle patch
        stride: Passo per l'estrazione delle patch
        batch_size: Dimensione del batch
        device: Device da utilizzare
        use_amp: Se utilizzare la precisione mista automatica
        max_size: Dimensione massima dell'immagine
        
    Returns:
        Immagine originale e predizione
    """
    # Imposta il device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica l'immagine
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Impossibile caricare l'immagine {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
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
    
    return img, prediction

def main():
    """Funzione principale per la predizione."""
    # Analizza gli argomenti dalla riga di comando
    args = parse_args()
    
    # Verifica se il modello esiste
    if not os.path.exists(args.model):
        print(f"Errore: Il modello {args.model} non esiste")
        sys.exit(1)
    
    # Verifica se l'immagine esiste
    if not os.path.exists(args.image):
        print(f"Errore: L'immagine {args.image} non esiste")
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
    
    # Classifica l'immagine
    img, prediction = predict_image(
        model,
        scattering,
        args.image,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
        use_amp=not args.no_amp,
        max_size=args.max_size
    )
    
    # Carica la maschera di verità se specificata
    mask = None
    if args.mask and os.path.exists(args.mask):
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        
        # Ridimensiona la maschera se necessario
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Visualizza i risultati
    visualize_results(
        img,
        mask if mask is not None else np.zeros_like(prediction),
        prediction,
        class_mapping,
        save_path=args.output
    )
    
    print("Predizione completata con successo!")

if __name__ == "__main__":
    main()
