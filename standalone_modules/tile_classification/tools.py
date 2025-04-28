#!/usr/bin/env python3
"""
Strumenti di utilità per la classificazione tile-wise.

Questo modulo fornisce funzioni di utilità per l'analisi dei dataset,
l'estrazione di tile e la gestione dei modelli.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
from collections import defaultdict
from tqdm import tqdm
import math

def analyze_dataset(images_dir, masks_dir, max_images=None, verbose=True):
    """
    Analizza un dataset di immagini e maschere e restituisce statistiche.
    
    Args:
        images_dir: Directory contenente le immagini
        masks_dir: Directory contenente le maschere
        max_images: Numero massimo di immagini da analizzare
        verbose: Se stampare informazioni dettagliate
        
    Returns:
        dict: Statistiche del dataset
    """
    # Verifica se le directory esistono
    if not os.path.exists(images_dir):
        print(f"Errore: Directory delle immagini non trovata: {images_dir}")
        return None
    
    if not os.path.exists(masks_dir):
        print(f"Errore: Directory delle maschere non trovata: {masks_dir}")
        return None
    
    # Trova tutte le immagini
    image_paths = sorted(
        [os.path.join(images_dir, f) for f in os.listdir(images_dir)
         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    )
    
    # Limita il numero di immagini se specificato
    if max_images and max_images < len(image_paths):
        image_paths = image_paths[:max_images]
    
    if verbose:
        print(f"Analisi di {len(image_paths)} immagini...")
    
    # Statistiche
    image_sizes = []
    mask_sizes = []
    class_counts = defaultdict(int)
    total_pixels = 0
    
    # Analizza ogni immagine
    for img_path in tqdm(image_paths, desc="Analisi immagini", disable=not verbose):
        # Trova la maschera corrispondente
        img_name = os.path.basename(img_path)
        img_stem = os.path.splitext(img_name)[0]
        
        mask_candidates = [
            os.path.join(masks_dir, f"{img_stem}_mask.png"),
            os.path.join(masks_dir, f"{img_stem}.png"),
            os.path.join(masks_dir, f"{img_stem}.jpg"),
            os.path.join(masks_dir, f"{img_stem}.tif")
        ]
        
        mask_path = None
        for candidate in mask_candidates:
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        if mask_path is None:
            if verbose:
                print(f"Attenzione: Maschera non trovata per {img_name}")
            continue
        
        # Carica immagine e maschera
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            if verbose:
                print(f"Errore nel caricamento di {img_path} o {mask_path}")
            continue
        
        # Verifica le dimensioni
        if img.shape[:2] != mask.shape[:2]:
            if verbose:
                print(f"Attenzione: Dimensioni diverse per {img_name}: {img.shape[:2]} vs {mask.shape[:2]}")
        
        # Raccogli statistiche
        image_sizes.append(img.shape[:2])
        mask_sizes.append(mask.shape[:2])
        
        # Conta le classi
        unique_classes, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            class_counts[int(cls)] += count
        
        total_pixels += mask.size
    
    # Calcola statistiche
    if not image_sizes:
        print("Nessuna immagine valida trovata")
        return None
    
    # Dimensioni delle immagini
    heights = [size[0] for size in image_sizes]
    widths = [size[1] for size in image_sizes]
    
    size_stats = {
        'min_height': min(heights),
        'max_height': max(heights),
        'mean_height': np.mean(heights),
        'min_width': min(widths),
        'max_width': max(widths),
        'mean_width': np.mean(widths)
    }
    
    # Distribuzione delle classi
    class_distribution = {cls: count / total_pixels for cls, count in class_counts.items()}
    
    # Stampa il riepilogo
    if verbose:
        print("\nStatistiche del dataset:")
        print(f"Immagini analizzate: {len(image_sizes)}")
        print(f"Dimensione media: {size_stats['mean_width']:.1f} x {size_stats['mean_height']:.1f}")
        print(f"Dimensione minima: {size_stats['min_width']} x {size_stats['min_height']}")
        print(f"Dimensione massima: {size_stats['max_width']} x {size_stats['max_height']}")
        
        print("\nDistribuzione delle classi:")
        for cls in sorted(class_distribution.keys()):
            percentage = class_distribution[cls] * 100
            print(f"  Classe {cls}: {percentage:.2f}% ({class_counts[cls]} pixel)")
    
    return {
        'num_images': len(image_sizes),
        'size_stats': size_stats,
        'class_counts': dict(class_counts),
        'class_distribution': class_distribution
    }

def extract_tiles(image_path, output_dir, tile_size=32, stride=16, min_class_pixels=0, class_threshold=None):
    """
    Estrae tile da un'immagine e dalla sua maschera.
    
    Args:
        image_path: Percorso dell'immagine
        output_dir: Directory di output
        tile_size: Dimensione dei tile
        stride: Passo per l'estrazione dei tile
        min_class_pixels: Numero minimo di pixel di classe (non sfondo) per salvare un tile
        class_threshold: Dizionario con soglie per classe {classe: min_pixel}
        
    Returns:
        dict: Statistiche dell'estrazione
    """
    # Verifica se l'immagine esiste
    if not os.path.exists(image_path):
        print(f"Errore: Immagine non trovata: {image_path}")
        return None
    
    # Determina il percorso della maschera
    img_dir = os.path.dirname(image_path)
    mask_dir = os.path.join(os.path.dirname(img_dir), "masks")
    if not os.path.exists(mask_dir):
        print(f"Errore: Directory delle maschere non trovata: {mask_dir}")
        return None
    
    img_name = os.path.basename(image_path)
    img_stem = os.path.splitext(img_name)[0]
    
    mask_candidates = [
        os.path.join(mask_dir, f"{img_stem}_mask.png"),
        os.path.join(mask_dir, f"{img_stem}.png"),
        os.path.join(mask_dir, f"{img_stem}.jpg"),
        os.path.join(mask_dir, f"{img_stem}.tif")
    ]
    
    mask_path = None
    for candidate in mask_candidates:
        if os.path.exists(candidate):
            mask_path = candidate
            break
    
    if mask_path is None:
        print(f"Errore: Maschera non trovata per {img_name}")
        return None
    
    # Carica immagine e maschera
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        print(f"Errore nel caricamento di {image_path} o {mask_path}")
        return None
    
    # Verifica le dimensioni
    if img.shape[:2] != mask.shape[:2]:
        print(f"Ridimensionamento della maschera per {img_name}")
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    # Estrai tile
    h, w = img.shape[:2]
    tile_count = 0
    class_tiles = defaultdict(int)
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            # Estrai tile
            img_tile = img[y:y+tile_size, x:x+tile_size]
            mask_tile = mask[y:y+tile_size, x:x+tile_size]
            
            # Conta i pixel per classe
            unique_classes, counts = np.unique(mask_tile, return_counts=True)
            class_pixels = {cls: count for cls, count in zip(unique_classes, counts)}
            
            # Verifica se il tile contiene abbastanza pixel di classe
            non_background_pixels = sum(class_pixels.get(cls, 0) for cls in class_pixels if cls > 0)
            
            if non_background_pixels >= min_class_pixels:
                # Verifica le soglie per classe se specificate
                save_tile = True
                if class_threshold:
                    for cls, threshold in class_threshold.items():
                        if class_pixels.get(cls, 0) < threshold:
                            save_tile = False
                            break
                
                if save_tile:
                    # Salva il tile
                    tile_filename = f"{img_stem}_x{x}_y{y}.png"
                    tile_path = os.path.join(output_dir, tile_filename)
                    cv2.imwrite(tile_path, img_tile)
                    
                    # Salva la maschera del tile
                    mask_filename = f"{img_stem}_x{x}_y{y}_mask.png"
                    mask_path = os.path.join(output_dir, mask_filename)
                    cv2.imwrite(mask_path, mask_tile)
                    
                    # Aggiorna i contatori
                    tile_count += 1
                    for cls in unique_classes:
                        if cls > 0:  # Ignora lo sfondo
                            class_tiles[int(cls)] += 1
    
    # Stampa il riepilogo
    print(f"Estrazione completata per {img_name}:")
    print(f"  Tile estratti: {tile_count}")
    print("  Tile per classe:")
    for cls in sorted(class_tiles.keys()):
        print(f"    Classe {cls}: {class_tiles[cls]} tile")
    
    return {
        'image': image_path,
        'tile_count': tile_count,
        'class_tiles': dict(class_tiles)
    }

def extract_tiles_batch(images_dir, output_dir, tile_size=32, stride=16, min_class_pixels=0, max_images=None):
    """
    Estrae tile da un batch di immagini.
    
    Args:
        images_dir: Directory contenente le immagini
        output_dir: Directory di output
        tile_size: Dimensione dei tile
        stride: Passo per l'estrazione dei tile
        min_class_pixels: Numero minimo di pixel di classe (non sfondo) per salvare un tile
        max_images: Numero massimo di immagini da elaborare
        
    Returns:
        dict: Statistiche dell'estrazione
    """
    # Verifica se la directory esiste
    if not os.path.exists(images_dir):
        print(f"Errore: Directory delle immagini non trovata: {images_dir}")
        return None
    
    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova tutte le immagini
    image_paths = sorted(
        [os.path.join(images_dir, f) for f in os.listdir(images_dir)
         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    )
    
    # Limita il numero di immagini se specificato
    if max_images and max_images < len(image_paths):
        image_paths = image_paths[:max_images]
    
    print(f"Estrazione di tile da {len(image_paths)} immagini...")
    
    # Estrai tile da ogni immagine
    results = []
    total_tiles = 0
    
    for img_path in tqdm(image_paths, desc="Estrazione tile"):
        result = extract_tiles(
            img_path,
            output_dir,
            tile_size=tile_size,
            stride=stride,
            min_class_pixels=min_class_pixels
        )
        
        if result:
            results.append(result)
            total_tiles += result['tile_count']
    
    # Stampa il riepilogo
    print(f"\nEstrazione completata:")
    print(f"Immagini elaborate: {len(results)}")
    print(f"Tile totali estratti: {total_tiles}")
    
    # Calcola il numero di tile per classe
    class_tiles = defaultdict(int)
    for result in results:
        for cls, count in result['class_tiles'].items():
            class_tiles[cls] += count
    
    print("Tile per classe:")
    for cls in sorted(class_tiles.keys()):
        print(f"  Classe {cls}: {class_tiles[cls]} tile")
    
    return {
        'images_processed': len(results),
        'total_tiles': total_tiles,
        'class_tiles': dict(class_tiles),
        'results': results
    }

def analyze_model(model_path, device=None):
    """
    Analizza un modello e stampa un riepilogo della sua architettura e parametri.
    
    Args:
        model_path: Percorso del modello
        device: Device su cui caricare il modello
        
    Returns:
        dict: Risultati dell'analisi
    """
    # Verifica se il modello esiste
    if not os.path.exists(model_path):
        print(f"Errore: Modello non trovato: {model_path}")
        return None
    
    # Imposta il device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica il checkpoint
    print(f"Caricamento del modello da {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Estrai informazioni
    model_info = {}
    
    # Informazioni sulle classi
    if 'class_mapping' in checkpoint:
        model_info['class_mapping'] = checkpoint['class_mapping']
        model_info['num_classes'] = len(checkpoint['class_mapping'])
    else:
        print("Attenzione: class_mapping non trovato nel checkpoint")
        model_info['num_classes'] = 0
    
    # Informazioni sul modello
    if 'model_state_dict' in checkpoint:
        # Conta i parametri
        num_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        model_info['num_params'] = num_params
    
    # Informazioni sull'addestramento
    if 'epoch' in checkpoint:
        model_info['epoch'] = checkpoint['epoch']
    
    if 'loss' in checkpoint:
        model_info['loss'] = checkpoint['loss']
    
    # Informazioni sulla trasformata scattering
    if 'scattering_params' in checkpoint:
        model_info['scattering_params'] = checkpoint['scattering_params']
    
    if 'use_scattering' in checkpoint:
        model_info['use_scattering'] = checkpoint['use_scattering']
    
    # Stampa il riepilogo
    print("\nRiepilogo del modello:")
    print(f"Percorso: {model_path}")
    
    if 'num_classes' in model_info:
        print(f"Numero di classi: {model_info['num_classes']}")
        if 'class_mapping' in model_info:
            print("Mapping delle classi:")
            for cls, name in model_info['class_mapping'].items():
                print(f"  {cls}: {name}")
    
    if 'num_params' in model_info:
        print(f"Numero di parametri: {model_info['num_params']:,}")
    
    if 'epoch' in model_info:
        print(f"Epoca: {model_info['epoch']}")
    
    if 'loss' in model_info:
        print(f"Loss: {model_info['loss']:.4f}")
    
    if 'use_scattering' in model_info:
        print(f"Uso scattering: {model_info['use_scattering']}")
    
    if 'scattering_params' in model_info:
        print("Parametri scattering:")
        for key, value in model_info['scattering_params'].items():
            print(f"  {key}: {value}")
    
    return model_info

def interactive_tile_selection(image_path, output_dir, tile_size=32, tiles_per_subwin=30):
    """
    Strumento interattivo per la selezione di tile da un'immagine.
    
    Args:
        image_path: Percorso dell'immagine
        output_dir: Directory di output
        tile_size: Dimensione dei tile
        tiles_per_subwin: Numero di tile per dimensione della sottofinestra
        
    Returns:
        int: Numero di tile estratti
    """
    # Verifica se l'immagine esiste
    if not os.path.exists(image_path):
        print(f"Errore: Immagine non trovata: {image_path}")
        return 0
    
    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    # Carica l'immagine
    img = cv2.imread(image_path)
    if img is None:
        print(f"Errore nel caricamento di {image_path}")
        return 0
    
    # Dimensioni della sottofinestra
    subwin_width = tiles_per_subwin * tile_size
    subwin_height = tiles_per_subwin * tile_size
    
    # Dimensioni dell'immagine
    h, w = img.shape[:2]
    
    # Calcola il numero di sottofinestre
    num_subwins_x = math.ceil(w / subwin_width)
    num_subwins_y = math.ceil(h / subwin_height)
    
    # Variabili di stato per la selezione con il mouse
    drawing = False
    mouse_start = None
    mouse_end = None
    preview_tiles = set()
    
    # Funzione di callback per il mouse
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, mouse_start, mouse_end, preview_tiles
        image, image_shape = param
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            mouse_start = (x, y)
            mouse_end = (x, y)
            preview_tiles = set()
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                mouse_end = (x, y)
                # Calcola i tile nell'area selezionata
                x1, y1 = mouse_start
                x2, y2 = mouse_end
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(image_shape[1], x_max)
                y_max = min(image_shape[0], y_max)
                tile_x_start = x_min // tile_size
                tile_y_start = y_min // tile_size
                tile_x_end = (x_max - 1) // tile_size
                tile_y_end = (y_max - 1) // tile_size
                preview_tiles = set()
                for ty in range(tile_y_start, tile_y_end + 1):
                    for tx in range(tile_x_start, tile_x_end + 1):
                        preview_tiles.add((tx, ty))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            mouse_end = (x, y)
            # Calcola i tile nell'area selezionata
            x1, y1 = mouse_start
            x2, y2 = mouse_end
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image_shape[1], x_max)
            y_max = min(image_shape[0], y_max)
            tile_x_start = x_min // tile_size
            tile_y_start = y_min // tile_size
            tile_x_end = (x_max - 1) // tile_size
            tile_y_end = (y_max - 1) // tile_size
            preview_tiles = set()
            for ty in range(tile_y_start, tile_y_end + 1):
                for tx in range(tile_x_start, tile_x_end + 1):
                    preview_tiles.add((tx, ty))
    
    # Funzione per disegnare la griglia
    def draw_grid(image, tile_size):
        h, w = image.shape[:2]
        for i in range(0, w, tile_size):
            cv2.line(image, (i, 0), (i, h), (200, 200, 200), 1)
        for j in range(0, h, tile_size):
            cv2.line(image, (0, j), (w, j), (200, 200, 200), 1)
    
    # Funzione per disegnare i tile selezionati
    def draw_selected_tiles(image, tile_size, selected_tiles):
        overlay = image.copy()
        alpha = 0.4
        for (tx, ty) in selected_tiles:
            pt1 = (tx * tile_size, ty * tile_size)
            pt2 = ((tx + 1) * tile_size, (ty + 1) * tile_size)
            cv2.rectangle(overlay, pt1, pt2, (0, 0, 255), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Funzione per disegnare i tile in anteprima
    def draw_preview_tiles(image, tile_size, preview_tiles):
        overlay = image.copy()
        alpha = 0.4
        for (tx, ty) in preview_tiles:
            pt1 = (tx * tile_size, ty * tile_size)
            pt2 = ((tx + 1) * tile_size, (ty + 1) * tile_size)
            cv2.rectangle(overlay, pt1, pt2, (0, 255, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Funzione per salvare i tile selezionati
    def save_tiles(image, tile_size, selected_tiles, output_dir, subwin_index):
        img_name = os.path.basename(image_path)
        img_stem = os.path.splitext(img_name)[0]
        
        for (tx, ty) in selected_tiles:
            x = tx * tile_size
            y = ty * tile_size
            tile = image[y:y+tile_size, x:x+tile_size]
            
            # Salva il tile
            tile_filename = f"{img_stem}_subwin{subwin_index}_x{tx}_y{ty}.png"
            tile_path = os.path.join(output_dir, tile_filename)
            cv2.imwrite(tile_path, tile)
    
    # Processa ogni sottofinestra
    total_tiles = 0
    
    for j in range(num_subwins_y):
        for i in range(num_subwins_x):
            # Estrai la sottofinestra
            x_start = i * subwin_width
            y_start = j * subwin_height
            x_end = min(x_start + subwin_width, w)
            y_end = min(y_start + subwin_height, h)
            
            subwin = img[y_start:y_end, x_start:x_end].copy()
            subwin_index = j * num_subwins_x + i + 1
            
            # Inizializza le variabili per questa sottofinestra
            selected_tiles = set()
            preview_tiles = set()
            
            # Crea la finestra
            window_name = f"Sottofinestra {subwin_index}/{num_subwins_x*num_subwins_y}"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, mouse_callback, param=(subwin, subwin.shape))
            
            print(f"\nSottofinestra {subwin_index}/{num_subwins_x*num_subwins_y}")
            print("Comandi:")
            print("  - Seleziona un'area con il mouse per vedere i tile in anteprima")
            print("  - Premi 'c' per confermare la selezione")
            print("  - Premi 'd' per rimuovere i tile selezionati")
            print("  - Premi 'r' per annullare l'anteprima")
            print("  - Premi 's' per salvare e passare alla prossima sottofinestra")
            print("  - Premi 'q' per uscire")
            
            while True:
                # Crea la copia su cui disegnare
                disp_img = subwin.copy()
                
                # Disegna la griglia
                draw_grid(disp_img, tile_size)
                
                # Disegna i tile selezionati
                if selected_tiles:
                    draw_selected_tiles(disp_img, tile_size, selected_tiles)
                
                # Disegna i tile in anteprima
                if preview_tiles:
                    draw_preview_tiles(disp_img, tile_size, preview_tiles)
                
                # Mostra l'immagine
                cv2.imshow(window_name, disp_img)
                
                # Gestisci i tasti
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return total_tiles
                elif key == ord('s'):
                    # Salva i tile selezionati
                    save_tiles(subwin, tile_size, selected_tiles, output_dir, subwin_index)
                    total_tiles += len(selected_tiles)
                    cv2.destroyWindow(window_name)
                    break
                elif key == ord('c'):
                    # Conferma la selezione
                    selected_tiles.update(preview_tiles)
                    preview_tiles = set()
                elif key == ord('d'):
                    # Rimuovi i tile selezionati
                    selected_tiles.difference_update(preview_tiles)
                    preview_tiles = set()
                elif key == ord('r'):
                    # Annulla l'anteprima
                    preview_tiles = set()
    
    cv2.destroyAllWindows()
    print(f"\nEstrazione completata: {total_tiles} tile estratti")
    return total_tiles
