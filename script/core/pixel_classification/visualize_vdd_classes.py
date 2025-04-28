#!/usr/bin/env python3
"""
Script per visualizzare un'immagine del dataset VDD con la sua maschera di segmentazione e la legenda delle classi.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Definizione del mapping delle classi VDD
VDD_CLASS_MAPPING = {
    0: 'background',
    1: 'wall',
    2: 'roads',
    3: 'vegetation',
    4: 'vehicles',
    5: 'roof',
    6: 'others'
}

# Colori per le classi (in formato BGR per OpenCV, RGB per matplotlib)
COLORS_BGR = [
    [0, 0, 0],      # 0: background - nero
    [0, 0, 255],    # 1: wall - rosso
    [255, 0, 0],    # 2: roads - blu
    [0, 255, 0],    # 3: vegetation - verde
    [255, 0, 255],  # 4: vehicles - magenta
    [0, 128, 255],  # 5: roof - arancione
    [128, 128, 128] # 6: others - grigio
]

COLORS_RGB = [
    [0, 0, 0],      # 0: background - nero
    [255, 0, 0],    # 1: wall - rosso
    [0, 0, 255],    # 2: roads - blu
    [0, 255, 0],    # 3: vegetation - verde
    [255, 0, 255],  # 4: vehicles - magenta
    [255, 128, 0],  # 5: roof - arancione
    [128, 128, 128] # 6: others - grigio
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualizza immagine VDD con segmentazione e legenda')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory contenente le immagini')
    parser.add_argument('--masks_dir', type=str, required=True, help='Directory contenente le maschere')
    parser.add_argument('--output', type=str, default=None, help='Percorso dove salvare la visualizzazione')
    parser.add_argument('--image_index', type=int, default=None, help='Indice immagine specifico da visualizzare')
    parser.add_argument('--seed', type=int, default=42, help='Seed per selezione casuale')
    return parser.parse_args()

def get_random_image(images_dir, masks_dir, seed=None, image_index=None):
    """Seleziona un'immagine casuale o specifica dal dataset."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    # Trova tutte le immagini
    image_paths = sorted(list(images_dir.glob("*.jpg")) +
                        list(images_dir.glob("*.JPG")) +
                        list(images_dir.glob("*.png")) +
                        list(images_dir.glob("*.PNG")))

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

def visualize_classes(img_path, mask_path, output_path=None):
    """Visualizza l'immagine, la maschera e la legenda delle classi."""
    # Carica l'immagine e la maschera
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Crea una versione colorata della maschera
    mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Trova classi uniche nella maschera
    unique_classes = np.unique(mask)
    print(f"Classi presenti nell'immagine: {unique_classes}")
    
    # Colora ogni classe
    for cls in unique_classes:
        if cls < len(COLORS_RGB):
            mask_colored[mask == cls] = COLORS_RGB[cls]
    
    # Crea una figura con 3 sottografici (immagine, maschera, statistica classi)
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Immagine originale
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img)
    ax1.set_title('Immagine originale')
    ax1.axis('off')
    
    # 2. Maschera di segmentazione
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(mask_colored)
    ax2.set_title('Maschera di segmentazione')
    ax2.axis('off')
    
    # 3. Overlay dell'immagine con la maschera
    ax3 = fig.add_subplot(2, 2, 3)
    overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
    ax3.imshow(overlay)
    ax3.set_title('Overlay immagine e segmentazione')
    ax3.axis('off')
    
    # 4. Statistica delle classi
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Calcola le statistiche delle classi
    class_counts = {}
    total_pixels = mask.size
    
    for cls in unique_classes:
        count = np.sum(mask == cls)
        percentage = (count / total_pixels) * 100
        class_name = VDD_CLASS_MAPPING.get(cls, f"Classe {cls}")
        class_counts[class_name] = percentage
    
    # Crea un grafico a barre
    classes = list(class_counts.keys())
    percentages = list(class_counts.values())
    colors = [tuple(c/255 for c in COLORS_RGB[list(VDD_CLASS_MAPPING.keys())[classes.index(cls)]]) 
              if cls in VDD_CLASS_MAPPING.values() else 'gray' 
              for cls in classes]
    
    ax4.bar(classes, percentages, color=colors)
    ax4.set_title('Distribuzione delle classi')
    ax4.set_ylabel('Percentuale (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Legenda delle classi
    legend_elements = []
    for cls, name in VDD_CLASS_MAPPING.items():
        if cls < len(COLORS_RGB):
            color = tuple(c/255 for c in COLORS_RGB[cls])
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f"{cls}: {name}"))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Spazio per la legenda
    
    # Salva o mostra l'immagine
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualizzazione salvata in {output_path}")
    else:
        plt.show()

def main():
    """Funzione principale."""
    args = parse_args()
    
    # Ottieni un'immagine casuale o specifica
    img_path, mask_path = get_random_image(
        args.images_dir, 
        args.masks_dir, 
        args.seed, 
        args.image_index
    )
    
    # Visualizza le classi
    visualize_classes(img_path, mask_path, args.output)

if __name__ == "__main__":
    main()