#!/usr/bin/env python3
"""
Tile Extractor - Strumento per l'estrazione e il salvataggio di tiles selezionati da immagini.

Questo script permette di selezionare interattivamente i tiles da un'immagine e salvarli
direttamente in una cartella di output. L'utente può navigare attraverso sottofinestre
dell'immagine, selezionare i tiles di interesse, e questi verranno estratti e salvati
come file separati.
"""

import cv2
import numpy as np
import os
import math

# Costanti di dimensione
tile_size = 32
tiles_per_subwin = 30
subwin_width = tiles_per_subwin * tile_size
subwin_height = tiles_per_subwin * tile_size

# Output directories base
output_dir_selected = '/home/brus/Projects/wavelet/datasets/HPL_images/'
# output_dir_unselected = 'unselected_tiles'

def create_output_dirs():
    os.makedirs(output_dir_selected, exist_ok=True)
    # os.makedirs(output_dir_unselected, exist_ok=True)

def draw_grid(image, tile_size):
    h, w = image.shape[:2]
    for i in range(0, w, tile_size):
        cv2.line(image, (i, 0), (i, h), (200, 200, 200), 1)
    for j in range(0, h, tile_size):
        cv2.line(image, (0, j), (w, j), (200, 200, 200), 1)

def mask_selected_tiles(image, tile_size, selected_tiles):
    # Oscura i tile confermati con una tinta grigia
    masked_image = image.copy()
    for (tx, ty) in selected_tiles:
        cv2.rectangle(masked_image,
                      (tx * tile_size, ty * tile_size),
                      ((tx + 1) * tile_size, (ty + 1) * tile_size),
                      (50, 50, 50), -1)
    return masked_image

def draw_cursor(image, cursor_tile, tile_size):
    x, y = cursor_tile
    pt1 = (x * tile_size, y * tile_size)
    pt2 = ((x + 1) * tile_size, (y + 1) * tile_size)
    cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)  # bordo rosso

def draw_preview_tiles(image, preview_tiles, tile_size):
    overlay = image.copy()
    alpha = 0.4  # trasparenza per il verde
    for (tx, ty) in preview_tiles:
        pt1 = (tx * tile_size, ty * tile_size)
        pt2 = ((tx + 1) * tile_size, (ty + 1) * tile_size)
        cv2.rectangle(overlay, pt1, pt2, (0, 255, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def save_subwindow_tiles(sub_img, tile_size, selected_tiles, subwin_index):
    h, w = sub_img.shape[:2]
    # Itera in base alla dimensione della sottofinestra
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = sub_img[y:y + tile_size, x:x + tile_size]
            tile_coord = (x // tile_size, y // tile_size)
            # Includiamo l'indice della sottofinestra nel nome del file
            if tile_coord in selected_tiles:
                filename = f'{output_dir_selected}/subwin_{subwin_index}_tile_{tile_coord[0]}_{tile_coord[1]}.jpg'
            # else:
            #     filename = f'{output_dir_unselected}/subwin_{subwin_index}_tile_{tile_coord[0]}_{tile_coord[1]}.jpg'
                cv2.imwrite(filename, tile)

def compute_tiles_in_rect(pt1, pt2, tile_size, image_shape):
    h, w = image_shape[:2]
    x1, y1 = pt1
    x2, y2 = pt2
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    tile_x_start = x_min // tile_size
    tile_y_start = y_min // tile_size
    tile_x_end = (x_max - 1) // tile_size
    tile_y_end = (y_max - 1) // tile_size
    tiles = set()
    for ty in range(tile_y_start, tile_y_end + 1):
        for tx in range(tile_x_start, tile_x_end + 1):
            tiles.add((tx, ty))
    return tiles

# Variabili di stato per la selezione con il mouse (per ogni sottofinestra)
drawing = False
mouse_start = None
mouse_end = None
preview_tiles = set()

def mouse_callback(event, x, y, flags, param):
    global drawing, mouse_start, mouse_end, preview_tiles
    image, image_shape = param
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mouse_start = (x, y)
        mouse_end = (x, y)
        preview_tiles = set()
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            mouse_end = (x, y)
            preview_tiles = compute_tiles_in_rect(mouse_start, mouse_end, tile_size, image_shape)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        mouse_end = (x, y)
        preview_tiles = compute_tiles_in_rect(mouse_start, mouse_end, tile_size, image_shape)
        # In attesa di una conferma o rimozione:
        # 'c' per confermare (aggiungere), 'd' per rimuovere i tile dalla selezione, 'r' per annullare l'anteprima

def process_subwindow(sub_img, subwin_index):
    """
    Funzione interattiva per processare una sottofinestra.
    Restituisce l'insieme dei tile selezionati (coordinate relative alla sottofinestra).
    """
    global drawing, mouse_start, mouse_end, preview_tiles
    # Inizializza le variabili per questa sottofinestra
    selected_tiles = set()
    preview_tiles = set()
    cursor_tile = [0, 0]
    sub_img_shape = sub_img.shape

    window_name = f'Sottofinestra {subwin_index}'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, param=(sub_img, sub_img_shape))

    while True:
        # Crea la copia su cui disegnare
        disp_img = mask_selected_tiles(sub_img, tile_size, selected_tiles)
        draw_grid(disp_img, tile_size)
        draw_cursor(disp_img, cursor_tile, tile_size)
        if preview_tiles:
            draw_preview_tiles(disp_img, preview_tiles, tile_size)
        cv2.imshow(window_name, disp_img)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            cv2.destroyWindow(window_name)
            exit(0)
        elif key == ord('s'):
            # Salva la sottofinestra e passa alla successiva
            cv2.destroyWindow(window_name)
            return selected_tiles
        elif key == ord('c'):
            # Conferma la selezione in anteprima: aggiungi i tile
            selected_tiles.update(preview_tiles)
            preview_tiles = set()
        elif key == ord('d'):
            # Rimuove i tile in anteprima dalla selezione se presenti
            selected_tiles.difference_update(preview_tiles)
            preview_tiles = set()
        elif key == ord('r'):
            # Annulla l'anteprima senza modifiche
            preview_tiles = set()
        elif key == 32:
            # Barra spaziatrice: seleziona/deseleziona il tile corrente
            tile_coord = (cursor_tile[0], cursor_tile[1])
            if tile_coord in selected_tiles:
                selected_tiles.remove(tile_coord)
            else:
                selected_tiles.add(tile_coord)
        # Gestione del cursore con le frecce (i codici possono variare a seconda della piattaforma)
        elif key == 81:  # freccia sinistra
            cursor_tile[0] = max(0, cursor_tile[0] - 1)
        elif key == 82:  # freccia su
            cursor_tile[1] = max(0, cursor_tile[1] - 1)
        elif key == 83:  # freccia destra
            max_x = sub_img_shape[1] // tile_size - 1
            cursor_tile[0] = min(max_x, cursor_tile[0] + 1)
        elif key == 84:  # freccia giù
            max_y = sub_img_shape[0] // tile_size - 1
            cursor_tile[1] = min(max_y, cursor_tile[1] + 1)
        # Altri tasti possono essere gestiti se necessario

def extract_tiles(input_image_path, output_dir=None, tile_size=32, tiles_per_subwin=30):
    """
    Extract tiles from an image through interactive selection.
    
    Args:
        input_image_path (str): Path to the input image
        output_dir (str, optional): Directory to save extracted tiles. 
                                   Defaults to /home/brus/Projects/wavelet/datasets/HPL_images/
        tile_size (int, optional): Size of each tile in pixels. Defaults to 32.
        tiles_per_subwin (int, optional): Number of tiles per subwindow dimension. Defaults to 30.
    
    Returns:
        int: Number of tiles extracted
    """
    global output_dir_selected
    
    # Update global variables if parameters are provided
    if output_dir:
        output_dir_selected = output_dir
    
    subwin_width = tiles_per_subwin * tile_size
    subwin_height = tiles_per_subwin * tile_size
    
    create_output_dirs()
    full_img = cv2.imread(input_image_path)
    if full_img is None:
        print(f"Errore: il file in input {input_image_path} non esiste o non è accessibile.")
        return 0
    
    full_h, full_w = full_img.shape[:2]

    # Calcola il numero di sottofinestre in orizzontale e verticale
    num_subwins_x = math.ceil(full_w / subwin_width)
    num_subwins_y = math.ceil(full_h / subwin_height)

    tile_counter = 0
    subwin_counter = 0
    # Itera su tutte le sottofinestre
    for j in range(num_subwins_y):
        for i in range(num_subwins_x):
            x_start = i * subwin_width
            y_start = j * subwin_height
            x_end = min(x_start + subwin_width, full_w)
            y_end = min(y_start + subwin_height, full_h)
            sub_img = full_img[y_start:y_end, x_start:x_end]
            subwin_counter += 1

            # Processa la sottofinestra: l'utente interagisce per selezionare tile
            selected_tiles = process_subwindow(sub_img, subwin_index=subwin_counter)
            # Salva i tile della sottofinestra (i nomi dei file includono il numero della sottofinestra)
            save_subwindow_tiles(sub_img, tile_size, selected_tiles, subwin_index=subwin_counter)
            tile_counter += len(selected_tiles)
            print(f"Sottofinestra {subwin_counter} salvata: {len(selected_tiles)} tiles.")

    print(f"Tutte le sottofinestre sono state elaborate. {tile_counter} tiles estratti.")
    return tile_counter

def main():
    """Command line entry point for the tile extraction tool."""
    extract_tiles('/home/brus/Projects/wavelet/elaborations/results/DJI_0981.JPG')

if __name__ == '__main__':
    main()
