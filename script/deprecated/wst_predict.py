#!/usr/bin/env python3
"""
Script per la predizione di immagini utilizzando modelli basati su Wavelet Scattering Transform.

Questo script consente di classificare immagini intere o suddividerle in tile per l'analisi.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from PIL import Image
from torchvision import transforms

# Aggiungi la directory principale al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa i moduli wavelet_lib
from wavelet_lib.base import load_model
from wavelet_lib.models import create_scattering_transform
from wavelet_lib.processors import ImageProcessor

def parse_args():
    """
    Analizza gli argomenti dalla riga di comando.
    
    Returns:
        args: Namespace contenente gli argomenti analizzati
    """
    parser = argparse.ArgumentParser(description='Predizioni con modello Wavelet Scattering Transform')
    
    # Parametri del modello
    parser.add_argument('--model-path', type=str, required=True, help='Percorso al file del modello')
    parser.add_argument('--image-path', type=str, required=True, help='Percorso all\'immagine da classificare')
    
    # Parametri di predizione
    parser.add_argument('--tile-mode', action='store_true', help='Abilita la modalità tile')
    parser.add_argument('--tile-size', type=int, default=32, help='Dimensione dei tile')
    parser.add_argument('--process-30x30', action='store_true', help='Elabora i tile 30x30 (ritagliati in alcuni dataset)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Soglia di confidenza per la visualizzazione')
    
    # Parametri generali
    parser.add_argument('--device', type=str, default=None, help='Device per l\'inferenza (cuda o cpu)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory per salvare i risultati')
    
    return parser.parse_args()

def classify_image(image_path, model, scattering, device, class_names, 
                   tile_size=32, process_30x30_tiles=False, confidence_threshold=0.7):
    """
    Classifica un'immagine utilizzando un modello addestrato con Wavelet Scattering Transform.
    
    Args:
        image_path: Percorso all'immagine da classificare
        model: Modello addestrato
        scattering: Trasformata scattering
        device: Device per l'inferenza
        class_names: Lista dei nomi delle classi
        tile_size: Dimensione dei tile (default: 32)
        process_30x30_tiles: Se elaborare tile 30x30 (default: False)
        confidence_threshold: Soglia di confidenza (default: 0.7)
        
    Returns:
        tuple: (label_matrix, confidence_matrix, cropped_image, tile_size)
    """
    # Caricamento e conversione dell'immagine
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)

    # Gestione del ritaglio per immagini con tile 30x30
    if process_30x30_tiles:
        tile_size = 30
        target_size = 32
        h, w, _ = image.shape
        center_y, center_x = h // 2, w // 2
        crop_size = 30 * 32
        y_start = max(0, center_y - crop_size // 2)
        x_start = max(0, center_x - crop_size // 2)
        cropped_image = image[y_start:y_start + crop_size, x_start:x_start + crop_size, :]
        img_height, img_width, _ = cropped_image.shape
    else:
        img_height, img_width, _ = image.shape
        cropped_image = image
        target_size = tile_size

    # Calcolo del numero di tile
    num_tiles_x = img_width // tile_size
    num_tiles_y = img_height // tile_size
    
    # Matrice per le etichette e per le confidenze
    label_matrix = np.full((num_tiles_y, num_tiles_x), -1, dtype=int)
    confidence_matrix = np.zeros((num_tiles_y, num_tiles_x), dtype=float)

    # Preparazione delle trasformazioni
    transform_steps = []
    if tile_size != target_size:
        transform_steps.append(transforms.Resize((target_size, target_size)))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    transform = transforms.Compose(transform_steps)

    # Classificazione dei tile
    total_tiles = num_tiles_x * num_tiles_y
    print(f"Elaborazione di {total_tiles} tile...")

    with torch.no_grad():
        processed_tiles = 0
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                tile = cropped_image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size, :]
                tile_img = Image.fromarray(tile)
                tile_tensor = transform(tile_img).unsqueeze(0).to(device)

                scattering_coeffs = scattering(tile_tensor)
                scattering_coeffs = scattering_coeffs.view(tile_tensor.size(0), -1, 8, 8)
                output = model(scattering_coeffs)
                
                # Calcolo softmax per ottenere le probabilità
                probabilities = torch.softmax(output, dim=1)
                max_prob, label = torch.max(probabilities, dim=1)
                
                # Memorizzazione dell'etichetta e della confidenza
                if max_prob.item() >= confidence_threshold:
                    label_matrix[i, j] = label.item()
                    confidence_matrix[i, j] = max_prob.item()

                # Aggiornamento del progresso
                processed_tiles += 1
                if processed_tiles % 100 == 0 or processed_tiles == total_tiles:
                    progress_percent = (processed_tiles / total_tiles) * 100
                    print(f"Progresso: {processed_tiles}/{total_tiles} tile ({progress_percent:.1f}%)")

    print("Classificazione completata.")
    return label_matrix, confidence_matrix, cropped_image, tile_size

def visualize_classification(image, label_matrix, confidence_matrix, tile_size, class_names, save_path=None):
    """
    Visualizza i risultati della classificazione.
    
    Args:
        image: Immagine classificata
        label_matrix: Matrice delle etichette
        confidence_matrix: Matrice delle confidenze
        tile_size: Dimensione dei tile
        class_names: Lista dei nomi delle classi
        save_path: Percorso dove salvare la visualizzazione
    """
    num_classes = len(class_names)
    colors = list(mcolors.TABLEAU_COLORS.values())[:num_classes]
    
    # Conteggio delle classi
    class_counts = {}
    for class_idx, name in enumerate(class_names):
        class_counts[name] = np.sum(label_matrix == class_idx)
    total_tiles = label_matrix.size
    classified_tiles = sum(class_counts.values())
    
    plt.figure(figsize=(15, 12))
    
    # Visualizzazione dell'immagine
    plt.imshow(image)

    # Disegno dei tile colorati
    ax = plt.gca()
    for i in range(label_matrix.shape[0]):
        for j in range(label_matrix.shape[1]):
            label = label_matrix[i, j]
            if label >= 0 and label < num_classes:
                color = colors[label]
                rect = plt.Rectangle(
                    (j * tile_size, i * tile_size),
                    tile_size, tile_size,
                    linewidth=1,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.3  # Semi-trasparente
                )
                ax.add_patch(rect)
    
    # Creazione della legenda
    legend_patches = []
    for class_idx, class_name in enumerate(class_names):
        count = class_counts.get(class_name, 0)
        percentage = 100 * count / total_tiles
        patch = plt.Rectangle((0, 0), 1, 1,
                               linewidth=1,
                               edgecolor=colors[class_idx],
                               facecolor=colors[class_idx],
                               label=f"{class_name}: {count} tile ({percentage:.1f}%)")
        legend_patches.append(patch)
    
    # Aggiunta della legenda
    plt.legend(handles=legend_patches,
               loc='center left',
               bbox_to_anchor=(1, 0.5),
               fontsize=10,
               framealpha=0.8)
    
    unclassified = total_tiles - classified_tiles
    unclassified_percentage = 100 * unclassified / total_tiles
    
    plt.title(f'Classificazione a Tile - {total_tiles} tile ({label_matrix.shape[0]}×{label_matrix.shape[1]})\n' +
              f'Classificati: {classified_tiles} ({100*classified_tiles/total_tiles:.1f}%), ' +
              f'Non classificati: {unclassified} ({unclassified_percentage:.1f}%)')
    plt.tight_layout()
    plt.axis('off')
    
    # Salva l'immagine se richiesto
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualizzazione salvata in: {save_path}")
    
    plt.show()

def classify_single_image(image_path, model, scattering, device, class_names):
    """
    Classifica un'immagine intera senza suddividerla in tile.
    
    Args:
        image_path: Percorso all'immagine da classificare
        model: Modello addestrato
        scattering: Trasformata scattering
        device: Device per l'inferenza
        class_names: Lista dei nomi delle classi
        
    Returns:
        tuple: (class_name, confidence)
    """
    # Crea il processore di immagini
    processor = ImageProcessor(model, scattering, device, class_names)
    
    # Classifica l'immagine
    result = processor.process_image(image_path)
    
    return result['class_name'], result['confidence']

def main():
    """
    Funzione principale per la predizione di immagini.
    """
    # Analizza gli argomenti dalla riga di comando
    args = parse_args()
    
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
    
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        class_names = list(class_to_idx.keys())
    else:
        print("Avviso: Mapping delle classi non trovato nel file del modello.")
        class_names = [f"Classe {i}" for i in range(10)]  # Fallback generico
    
    print(f"Classi rilevate: {class_names}")
    
    # Crea la trasformata scattering
    scattering = create_scattering_transform(
        J=2,
        shape=(32, 32),
        max_order=2,
        device=device
    )
    
    # Riprova a creare il modello da zero con la stessa architettura
    num_classes = len(class_names)
    num_channels = 243  # Valore fisso basato sull'architettura del modello
    
    # Crea un'istanza di CustomScatteringClassifier che corrisponde esattamente al modello salvato
    # Nota: questa classe è definita qui per garantire la compatibilità con il modello salvato
    class CustomScatteringClassifier(torch.nn.Module):
        def __init__(self, in_channels=243, num_classes=7):
            super(CustomScatteringClassifier, self).__init__()
            self.in_channels = in_channels
            self.num_classes = num_classes
            
            # Batch normalization for input
            self.bn = torch.nn.BatchNorm2d(in_channels)
            
            # Feature extractor (matching the saved model architecture)
            layers = []
            
            # First convolutional block (128 channels)
            layers.append(torch.nn.Conv2d(in_channels, 128, kernel_size=3, padding=1))
            layers.append(torch.nn.BatchNorm2d(128))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Conv2d(128, 128, kernel_size=3, padding=1))
            layers.append(torch.nn.BatchNorm2d(128))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            
            # Second conv block (256 channels)
            layers.append(torch.nn.Conv2d(128, 256, kernel_size=3, padding=1))
            layers.append(torch.nn.BatchNorm2d(256))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Conv2d(256, 256, kernel_size=3, padding=1))
            layers.append(torch.nn.BatchNorm2d(256))
            layers.append(torch.nn.ReLU(inplace=True))
            
            # Third conv block (512 channels)
            layers.append(torch.nn.Conv2d(256, 512, kernel_size=3, padding=1))
            layers.append(torch.nn.BatchNorm2d(512))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Conv2d(512, 512, kernel_size=3, padding=1))
            layers.append(torch.nn.BatchNorm2d(512))
            layers.append(torch.nn.ReLU(inplace=True))
            
            # Adaptive pooling
            layers.append(torch.nn.AdaptiveAvgPool2d(2))
            
            self.features = torch.nn.Sequential(*layers)
            
            # Classifier
            self.classifier = torch.nn.Linear(2048, num_classes)
        
        def forward(self, x):
            x = self.bn(x)
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    # Crea il modello
    model = CustomScatteringClassifier(in_channels=num_channels, num_classes=num_classes).to(device)
    
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
    
    # Configura la directory di output se non specificata
    if args.output_dir is None:
        filename = os.path.basename(args.image_path)
        base_filename, _ = os.path.splitext(filename)
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", base_filename)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Esegui la predizione
    if args.tile_mode:
        # Modalità tile
        print(f"Esecuzione della predizione in modalità tile (dimensione: {args.tile_size})...")
        
        label_matrix, confidence_matrix, cropped_image, tile_size = classify_image(
            args.image_path,
            model,
            scattering,
            device,
            class_names,
            tile_size=args.tile_size,
            process_30x30_tiles=args.process_30x30,
            confidence_threshold=args.confidence_threshold
        )
        
        # Visualizza e salva i risultati
        save_path = os.path.join(args.output_dir, "tile_classification.png")
        visualize_classification(
            cropped_image,
            label_matrix,
            confidence_matrix,
            tile_size,
            class_names,
            save_path=save_path
        )
    else:
        # Modalità immagine singola
        print("Esecuzione della predizione su immagine intera...")
        
        class_name, confidence = classify_single_image(
            args.image_path,
            model,
            scattering,
            device,
            class_names
        )
        
        print(f"\nRisultato della classificazione:")
        print(f"Classe: {class_name}")
        print(f"Confidenza: {confidence:.4f}")
        
        # Visualizza e salva l'immagine con l'etichetta
        img = Image.open(args.image_path).convert('RGB')
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(img))
        plt.title(f"Classe: {class_name}\nConfidenza: {confidence:.4f}")
        plt.axis('off')
        
        save_path = os.path.join(args.output_dir, "classification_result.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualizzazione salvata in: {save_path}")
        plt.show()
    
    print(f"\nPredizione completata!")

if __name__ == "__main__":
    main()