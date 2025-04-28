#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm

# Aggiungi la directory principale al path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(project_dir)

from wavelet_lib.single_pixel_classification.models import PixelWiseClassifier, create_scattering_transform

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test classificatore pixel-wise su un\'immagine casuale')

    # Input e output
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images_dir', type=str, help='Directory contenente le immagini di test')
    input_group.add_argument('--input_image', type=str, help='Percorso di un\'immagine specifica da testare')

    parser.add_argument('--masks_dir', type=str, help='Directory contenente le maschere di classe (richiesto se si usa --images_dir)')
    parser.add_argument('--model', type=str, required=True, help='Percorso del modello addestrato')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory dove salvare i risultati (opzionale)')
    parser.add_argument('--output_file', type=str, default=None, help='Nome del file di output (opzionale, solo con --input_image)')

    # Parametri di test
    parser.add_argument('--patch_size', type=int, default=32, help='Dimensione delle patch')
    parser.add_argument('--stride', type=int, default=16, help='Passo per l\'estrazione delle patch')
    parser.add_argument('--batch_size', type=int, default=16, help='Dimensione del batch')
    parser.add_argument('--j', type=int, default=2, help='Numero di scale per la trasformata scattering')
    parser.add_argument('--scattering_order', type=int, default=2, help='Ordine della trasformata scattering')
    parser.add_argument('--no_scattering', action='store_true', help='Disabilita la trasformata scattering')
    parser.add_argument('--num_classes', type=int, default=None, help='Numero di classi (se non specificato, viene rilevato dal modello)')
    parser.add_argument('--class_names', type=str, help='Nomi delle classi separati da virgola (es. "sfondo,acqua,vegetazione")')
    parser.add_argument('--seed', type=int, default=None, help='Seed per la selezione casuale dell\'immagine')
    parser.add_argument('--image_index', type=int, default=None, help='Indice specifico dell\'immagine da testare')
    parser.add_argument('--max_size', type=int, default=None, help='Dimensione massima dell\'immagine (ridimensiona se più grande)')

    # Opzioni per la GPU
    parser.add_argument('--disable_cudnn', action='store_true', help='Disabilita completamente cuDNN')
    parser.add_argument('--no_amp', action='store_true', help='Disabilita la precisione mista automatica')

    return parser.parse_args()

def load_model(model_path, device, num_classes, use_scattering, patch_size, j, scattering_order):
    """Carica il modello addestrato."""
    print(f"Caricamento del modello da {model_path}...")

    # Carica il checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Estrai informazioni dal checkpoint
    if 'class_mapping' in checkpoint:
        class_mapping = checkpoint['class_mapping']
        print(f"Classi trovate nel modello: {class_mapping}")
        num_classes = len(class_mapping)
    elif 'model_state_dict' in checkpoint:
        # Se non c'è class_mapping, prova a determinare il numero di classi dai pesi
        if 'final_conv.bias' in checkpoint['model_state_dict']:
            detected_num_classes = checkpoint['model_state_dict']['final_conv.bias'].size(0)
            print(f"Numero di classi rilevato dai pesi: {detected_num_classes}")
            num_classes = detected_num_classes
            # Crea un mapping di default
            class_mapping = {i: f"Classe {i}" for i in range(num_classes)}

    use_scattering = checkpoint.get('use_scattering', use_scattering)

    # Crea la trasformata scattering se necessario
    if use_scattering:
        scattering = create_scattering_transform(
            J=j,
            shape=(patch_size, patch_size),
            max_order=scattering_order,
            device=device
        )

        # Calcola il numero di canali di input
        dummy_input = torch.randn(1, 3, patch_size, patch_size).to(device)
        scattering_output = scattering(dummy_input)

        # Gestisci la dimensionalità dell'output
        if scattering_output.dim() == 5:
            if scattering_output.shape[-1] == 1:
                scattering_output = scattering_output.squeeze(-1)
            else:
                scattering_output = scattering_output[..., 0]

        in_channels = scattering_output.shape[1]
        print(f"Trasformata scattering: {in_channels} canali di input")
    else:
        scattering = None
        in_channels = 3
        print("Nessuna trasformata scattering")

    # Ottieni il numero corretto di classi dal modello salvato
    if 'model_state_dict' in checkpoint and 'final_conv.bias' in checkpoint['model_state_dict']:
        actual_num_classes = checkpoint['model_state_dict']['final_conv.bias'].size(0)
        print(f"Numero di classi nei pesi del modello: {actual_num_classes}")
        num_classes = actual_num_classes

    # Crea il modello
    model = PixelWiseClassifier(
        in_channels=in_channels,
        hidden_dim=128,
        num_classes=num_classes,
        use_scattering=use_scattering
    ).to(device)

    # Carica i pesi
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, scattering, class_mapping

def get_random_image(images_dir, masks_dir, seed=None, image_index=None):
    """Seleziona un'immagine casuale dal dataset."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    # Trova tutte le immagini
    image_paths = sorted(list(images_dir.glob("*.jpg")) +
                        list(images_dir.glob("*.JPG")) +
                        list(images_dir.glob("*.png")) +
                        list(images_dir.glob("*.PNG")) +
                        list(images_dir.glob("*.tif")) +
                        list(images_dir.glob("*.TIF")))

    if not image_paths:
        raise ValueError(f"Nessuna immagine trovata in {images_dir}")

    # Seleziona un'immagine casuale o specifica
    if image_index is not None:
        if image_index < 0 or image_index >= len(image_paths):
            raise ValueError(f"Indice immagine {image_index} fuori range (0-{len(image_paths)-1})")
        img_path = image_paths[image_index]
    else:
        if seed is not None:
            random.seed(seed)
        img_path = random.choice(image_paths)

    # Trova la maschera corrispondente
    possible_mask_paths = [
        masks_dir / f"{img_path.stem}_mask.png",
        masks_dir / f"{img_path.stem}.png",
        masks_dir / f"{img_path.stem}.jpg",
        masks_dir / f"{img_path.stem}.tif"
    ]

    mask_path = None
    for p in possible_mask_paths:
        if p.exists():
            mask_path = p
            break

    if mask_path is None:
        raise ValueError(f"Nessuna maschera trovata per {img_path}")

    print(f"Immagine selezionata: {img_path}")
    print(f"Maschera corrispondente: {mask_path}")

    return img_path, mask_path

def classify_image(img_path, mask_path, model, scattering, device, patch_size, stride, batch_size, use_amp=True):
    """Classifica un'immagine intera utilizzando il modello addestrato."""
    # Carica l'immagine e la maschera
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Verifica le dimensioni
    if img.shape[:2] != mask.shape[:2]:
        print(f"Ridimensionamento maschera: {img.shape[:2]} vs {mask.shape[:2]}")
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Crea una mappa di predizione vuota
    prediction = np.zeros(img.shape[:2], dtype=np.uint8)

    # Estrai patch dall'immagine
    h, w = img.shape[:2]
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

    return img, mask, prediction

def visualize_results(img, mask, prediction, class_mapping, output_path=None):
    """Visualizza i risultati della classificazione."""
    # Crea una mappa di colori per le classi
    colors = [
        [0, 0, 0],        # Classe 0: Nero (sfondo)
        [0, 0, 255],      # Classe 1: Blu
        [0, 255, 0],      # Classe 2: Verde
        [255, 0, 0],      # Classe 3: Rosso
        [255, 255, 0],    # Classe 4: Giallo
        [0, 255, 255],    # Classe 5: Ciano
        [255, 0, 255],    # Classe 6: Magenta
        [128, 0, 0],      # Classe 7: Marrone
        [0, 128, 0],      # Classe 8: Verde scuro
        [0, 0, 128],      # Classe 9: Blu scuro
    ]

    # Crea immagini colorate per maschera e predizione
    mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    pred_colored = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

    for cls in range(min(len(colors), len(class_mapping))):
        mask_colored[mask == cls] = colors[cls]
        pred_colored[prediction == cls] = colors[cls]

    # Crea una legenda
    legend_items = []
    for cls, name in class_mapping.items():
        if cls < len(colors):
            color = np.array(colors[cls]) / 255.0
            legend_items.append((color, name))

    # Visualizza i risultati
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title("Immagine originale")
    axes[0].axis('off')

    axes[1].imshow(mask_colored)
    axes[1].set_title("Maschera di verità")
    axes[1].axis('off')

    axes[2].imshow(pred_colored)
    axes[2].set_title("Predizione")
    axes[2].axis('off')

    # Aggiungi la legenda
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color, _ in legend_items]
    legend_labels = [name for _, name in legend_items]
    fig.legend(legend_patches, legend_labels, loc='lower center', ncol=len(legend_items))

    plt.tight_layout()

    # Salva o mostra i risultati
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Risultati salvati in {output_path}")
    else:
        plt.show()

def calculate_metrics(mask, prediction, num_classes):
    """Calcola le metriche di valutazione."""
    # Calcola la matrice di confusione
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((mask == i) & (prediction == j))

    # Calcola accuracy
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    # Calcola precision, recall e F1-score per ogni classe
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for i in range(num_classes):
        # Precision: TP / (TP + FP)
        precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i]) if np.sum(confusion_matrix[:, i]) > 0 else 0

        # Recall: TP / (TP + FN)
        recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]) if np.sum(confusion_matrix[i, :]) > 0 else 0

        # F1-score: 2 * (precision * recall) / (precision + recall)
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    # Calcola metriche medie
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1_score = np.mean(f1_score)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1_score': mean_f1_score,
        'confusion_matrix': confusion_matrix
    }

def print_metrics(metrics, class_mapping):
    """Stampa le metriche di valutazione."""
    print("\n=== METRICHE DI VALUTAZIONE ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean Precision: {metrics['mean_precision']:.4f}")
    print(f"Mean Recall: {metrics['mean_recall']:.4f}")
    print(f"Mean F1-score: {metrics['mean_f1_score']:.4f}")

    print("\nMetriche per classe:")
    for cls, name in class_mapping.items():
        if cls < len(metrics['precision']):
            print(f"  Classe {cls} ({name}):")
            print(f"    Precision: {metrics['precision'][cls]:.4f}")
            print(f"    Recall: {metrics['recall'][cls]:.4f}")
            print(f"    F1-score: {metrics['f1_score'][cls]:.4f}")

    print("\nMatrice di confusione:")
    print(metrics['confusion_matrix'])

def classify_external_image(img_path, model, scattering, device, patch_size, stride, batch_size, use_amp=True, max_size=None):
    """Classifica un'immagine esterna utilizzando il modello addestrato."""
    # Carica l'immagine
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Impossibile caricare l'immagine: {img_path}")

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
    prediction = np.zeros(img.shape[:2], dtype=np.uint8)

    # Estrai patch dall'immagine
    h, w = img.shape[:2]
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

    return img, prediction

def visualize_external_image(img, prediction, class_mapping, output_path=None):
    """Visualizza i risultati della classificazione di un'immagine esterna."""
    # Crea una mappa di colori per le classi
    colors = [
        [0, 0, 0],        # Classe 0: Nero (sfondo)
        [0, 0, 255],      # Classe 1: Blu
        [0, 255, 0],      # Classe 2: Verde
        [255, 0, 0],      # Classe 3: Rosso
        [255, 255, 0],    # Classe 4: Giallo
        [0, 255, 255],    # Classe 5: Ciano
        [255, 0, 255],    # Classe 6: Magenta
        [128, 0, 0],      # Classe 7: Marrone
        [0, 128, 0],      # Classe 8: Verde scuro
        [0, 0, 128],      # Classe 9: Blu scuro
    ]

    # Crea immagine colorata per la predizione
    pred_colored = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

    for cls in range(min(len(colors), len(class_mapping))):
        pred_colored[prediction == cls] = colors[cls]

    # Crea una legenda
    legend_items = []
    for cls, name in class_mapping.items():
        if cls < len(colors):
            color = np.array(colors[cls]) / 255.0
            legend_items.append((color, name))

    # Visualizza i risultati
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img)
    axes[0].set_title("Immagine originale")
    axes[0].axis('off')

    axes[1].imshow(pred_colored)
    axes[1].set_title("Predizione")
    axes[1].axis('off')

    # Aggiungi la legenda
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color, _ in legend_items]
    legend_labels = [name for _, name in legend_items]
    fig.legend(legend_patches, legend_labels, loc='lower center', ncol=len(legend_items))

    plt.tight_layout()

    # Salva o mostra i risultati
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Risultati salvati in {output_path}")
    else:
        plt.show()

def main(args):
    """Funzione principale."""
    # Imposta il device
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
    model, scattering, class_mapping = load_model(
        args.model,
        device,
        args.num_classes,
        not args.no_scattering,
        args.patch_size,
        args.j,
        args.scattering_order
    )

    # Sostituisci i nomi delle classi se specificati dall'utente
    if args.class_names:
        class_names = args.class_names.split(',')
        if len(class_names) != len(class_mapping):
            print(f"ATTENZIONE: Il numero di nomi di classe specificati ({len(class_names)}) "
                  f"non corrisponde al numero di classi nel modello ({len(class_mapping)})")
            print("Verranno utilizzati solo i primi nomi di classe disponibili")

        # Aggiorna il mapping delle classi
        for i, name in enumerate(class_names):
            if i in class_mapping:
                class_mapping[i] = name.strip()

        print(f"Nomi delle classi aggiornati: {class_mapping}")

    # Modalità di esecuzione: immagine dal dataset o immagine esterna
    if args.input_image:
        # Modalità immagine esterna
        print(f"Classificazione dell'immagine esterna: {args.input_image}")

        # Classifica l'immagine
        img, prediction = classify_external_image(
            args.input_image,
            model,
            scattering,
            device,
            args.patch_size,
            args.stride,
            args.batch_size,
            not args.no_amp,
            args.max_size
        )

        # Crea la directory di output se necessario
        output_path = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.output_file:
                output_path = os.path.join(args.output_dir, args.output_file)
            else:
                img_path = Path(args.input_image)
                output_path = os.path.join(args.output_dir, f"result_{img_path.stem}.png")

        # Visualizza i risultati
        visualize_external_image(img, prediction, class_mapping, output_path)

    else:
        # Modalità immagine dal dataset
        if not args.masks_dir:
            raise ValueError("L'argomento --masks_dir è richiesto quando si usa --images_dir")

        # Seleziona un'immagine casuale
        img_path, mask_path = get_random_image(
            args.images_dir,
            args.masks_dir,
            args.seed,
            args.image_index
        )

        # Classifica l'immagine
        img, mask, prediction = classify_image(
            img_path,
            mask_path,
            model,
            scattering,
            device,
            args.patch_size,
            args.stride,
            args.batch_size,
            not args.no_amp
        )

        # Calcola le metriche
        metrics = calculate_metrics(mask, prediction, len(class_mapping))
        print_metrics(metrics, class_mapping)

        # Crea la directory di output se necessario
        output_path = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, f"result_{img_path.stem}.png")

        # Visualizza i risultati
        visualize_results(img, mask, prediction, class_mapping, output_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
