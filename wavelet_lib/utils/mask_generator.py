#!/usr/bin/env python3
"""
Mask Generator - Strumento per la creazione di maschere binarie basate sulla selezione di tiles.

Questo script permette di selezionare interattivamente i tiles da un'immagine e generare
una maschera binaria dove i pixels corrispondenti ai tiles selezionati sono impostati a 255 (bianco).
L'utente può navigare tra le sottofinestre dell'immagine, selezionare i tiles di interesse,
e alla fine viene generata una maschera binaria completa dell'immagine.
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

def update_mask(mask, selected_tiles, tile_size, offset_x, offset_y):
    """
    Aggiorna la maschera con i tile selezionati nella posizione corretta
    """
    for (tx, ty) in selected_tiles:
        # Calcola la posizione assoluta nel contesto dell'immagine completa
        abs_x = offset_x + tx * tile_size
        abs_y = offset_y + ty * tile_size
        # Assicura che il rettangolo sia all'interno della maschera
        if (abs_y + tile_size <= mask.shape[0] and abs_x + tile_size <= mask.shape[1]):
            # Imposta la regione corrispondente al tile come bianca (255)
            mask[abs_y:abs_y + tile_size, abs_x:abs_x + tile_size] = 255
    return mask

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

def process_subwindow(sub_img, subwin_index, previous_selection=None):
    """
    Funzione interattiva per processare una sottofinestra.
    Restituisce l'insieme dei tile selezionati (coordinate relative alla sottofinestra).
    e un flag che indica se tornare alla sottofinestra precedente
    """
    global drawing, mouse_start, mouse_end, preview_tiles
    # Inizializza le variabili per questa sottofinestra
    selected_tiles = set() if previous_selection is None else previous_selection.copy()
    preview_tiles = set()
    cursor_tile = [0, 0]
    sub_img_shape = sub_img.shape

    window_name = f'Sottofinestra {subwin_index}'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, param=(sub_img, sub_img_shape))

    # Mostra istruzioni
    print(f"\nSottofinestra {subwin_index}:")
    print("- 'c' per confermare selezione")
    print("- 'd' per rimuovere selezione")
    print("- 'r' per annullare anteprima")
    print("- Spazio per selezionare/deselezionare il tile corrente")
    print("- 's' per salvare e passare alla prossima sottofinestra")
    print("- 'b' per tornare alla sottofinestra precedente")
    print("- 'q' per uscire")

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
            return selected_tiles, False  # False significa "non tornare indietro"
        elif key == ord('b'):
            # Torna alla sottofinestra precedente
            cv2.destroyWindow(window_name)
            return selected_tiles, True  # True significa "torna indietro"
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

def create_mask(input_image_path, output_dir=None, output_filename=None, tile_size=32, tiles_per_subwin=30):
    """
    Create a binary mask through interactive tile selection.
    
    Args:
        input_image_path (str): Path to the input image
        output_dir (str, optional): Directory to save the mask. 
                                   Defaults to /home/brus/Projects/wavelet/datasets/HPL_images/segmentation/mask
        output_filename (str, optional): Filename for the output mask. If None, uses input filename with _mask suffix.
        tile_size (int, optional): Size of each tile in pixels. Defaults to 32.
        tiles_per_subwin (int, optional): Number of tiles per subwindow dimension. Defaults to 30.
    
    Returns:
        str: Path to the generated mask
    """
    # Setup output directories and files
    if output_dir is None:
        output_dir = '/home/brus/Projects/wavelet/datasets/HPL_images/segmentation/mask'
    
    if output_filename is None:
        output_filename = os.path.basename(input_image_path).split('.')[0] + '_mask.jpg'
    
    output_mask_file = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Update global variables
    global subwin_width, subwin_height
    subwin_width = tiles_per_subwin * tile_size
    subwin_height = tiles_per_subwin * tile_size
    
    # Load image
    full_img = cv2.imread(input_image_path)
    if full_img is None:
        print(f"Errore: il file in input {input_image_path} non esiste o non è accessibile.")
        return None
    
    full_h, full_w = full_img.shape[:2]

    # Crea una maschera vuota (nera) delle stesse dimensioni dell'immagine originale
    mask = np.zeros((full_h, full_w), dtype=np.uint8)

    # Calcola il numero di sottofinestre in orizzontale e verticale
    num_subwins_x = math.ceil(full_w / subwin_width)
    num_subwins_y = math.ceil(full_h / subwin_height)
    total_subwins = num_subwins_x * num_subwins_y

    # Memorizza le coordinate e le selezioni di ogni sottofinestra
    subwin_coords = []
    subwin_selections = []

    # Prepara la lista di coordinate delle sottofinestre
    for j in range(num_subwins_y):
        for i in range(num_subwins_x):
            x_start = i * subwin_width
            y_start = j * subwin_height
            x_end = min(x_start + subwin_width, full_w)
            y_end = min(y_start + subwin_height, full_h)
            subwin_coords.append((x_start, y_start, x_end, y_end))
            subwin_selections.append(set())  # Selezioni iniziali vuote

    # Processa le sottofinestre con la possibilità di tornare indietro
    current_subwin_idx = 0

    while 0 <= current_subwin_idx < total_subwins:
        x_start, y_start, x_end, y_end = subwin_coords[current_subwin_idx]
        sub_img = full_img[y_start:y_end, x_start:x_end]

        # Recupera le selezioni precedenti per questa sottofinestra (se esistono)
        previous_selection = subwin_selections[current_subwin_idx]

        # Processa la sottofinestra corrente
        selected_tiles, go_back = process_subwindow(sub_img, subwin_index=current_subwin_idx+1, previous_selection=previous_selection)

        # Salva le selezioni aggiornate
        subwin_selections[current_subwin_idx] = selected_tiles

        # Aggiorna la maschera
        mask = update_mask(mask, selected_tiles, tile_size, x_start, y_start)

        # Torna indietro o vai avanti
        if go_back:
            if current_subwin_idx > 0:
                current_subwin_idx -= 1
                print(f"Tornando alla sottofinestra {current_subwin_idx+1}")
            else:
                print("Sei già alla prima sottofinestra.")
        else:
            current_subwin_idx += 1
            print(f"Sottofinestra {current_subwin_idx}/{total_subwins} elaborata.")

    # Salva la maschera finale
    cv2.imwrite(output_mask_file, mask)
    print(f"Maschera salvata in {output_mask_file}")

    # Facoltativo: visualizza la maschera finale
    cv2.namedWindow("Maschera finale", cv2.WINDOW_NORMAL)
    cv2.imshow("Maschera finale", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return output_mask_file

def main():
    """Command line entry point for the mask generator tool."""
    create_mask('/home/brus/Projects/wavelet/elaborations/results/DJI_0981.JPG')

if __name__ == '__main__':
    main()