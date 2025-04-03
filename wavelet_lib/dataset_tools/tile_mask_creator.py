#!/usr/bin/env python3
"""
Modulo per la creazione interattiva di maschere di selezione dei tiles.
Fornisce strumenti per dividere un'immagine in sottofinestre e selezionare
interattivamente i tiles di interesse, generando una maschera binaria.
"""

import cv2
import numpy as np
import os
import math
from pathlib import Path
from typing import Set, Tuple, List, Dict, Optional, Union

class TileMaskCreator:
    """
    Classe per la creazione interattiva di maschere di selezione dei tiles.
    Permette di dividere un'immagine in sottofinestre e selezionare
    interattivamente i tiles di interesse, generando una maschera binaria.
    """

    def __init__(
        self,
        tile_size: int = 32,
        tiles_per_subwin: int = 30
    ):
        """
        Inizializza il creatore di maschere per tiles.

        Args:
            tile_size: Dimensione di un singolo tile in pixel
            tiles_per_subwin: Numero di tiles per lato in una sottofinestra
        """
        self.tile_size = tile_size
        self.tiles_per_subwin = tiles_per_subwin
        self.subwin_width = tiles_per_subwin * tile_size
        self.subwin_height = tiles_per_subwin * tile_size

        # Variabili di stato per la selezione con il mouse
        self.drawing = False
        self.mouse_start = None
        self.mouse_end = None
        self.preview_tiles = set()
        self.selected_tiles = set()
        self.cursor_tile = [0, 0]

    def draw_grid(self, image: np.ndarray) -> None:
        """
        Disegna una griglia sull'immagine in base alla dimensione dei tiles.

        Args:
            image: Immagine su cui disegnare la griglia
        """
        h, w = image.shape[:2]
        for i in range(0, w, self.tile_size):
            cv2.line(image, (i, 0), (i, h), (200, 200, 200), 1)
        for j in range(0, h, self.tile_size):
            cv2.line(image, (0, j), (w, j), (200, 200, 200), 1)

    def mask_selected_tiles(self, image: np.ndarray, selected_tiles: Set[Tuple[int, int]]) -> np.ndarray:
        """
        Oscura i tile selezionati con una tinta grigia.

        Args:
            image: Immagine originale
            selected_tiles: Set di coordinate (x, y) dei tiles selezionati

        Returns:
            Immagine con i tiles selezionati oscurati
        """
        masked_image = image.copy()
        for (tx, ty) in selected_tiles:
            cv2.rectangle(
                masked_image,
                (tx * self.tile_size, ty * self.tile_size),
                ((tx + 1) * self.tile_size, (ty + 1) * self.tile_size),
                (50, 50, 50), -1
            )
        return masked_image

    def draw_cursor(self, image: np.ndarray, cursor_tile: List[int]) -> None:
        """
        Disegna un cursore intorno al tile corrente.

        Args:
            image: Immagine su cui disegnare il cursore
            cursor_tile: Coordinate [x, y] del tile corrente
        """
        x, y = cursor_tile
        pt1 = (x * self.tile_size, y * self.tile_size)
        pt2 = ((x + 1) * self.tile_size, (y + 1) * self.tile_size)
        cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)  # bordo rosso

    def draw_preview_tiles(self, image: np.ndarray, preview_tiles: Set[Tuple[int, int]]) -> None:
        """
        Disegna un'anteprima dei tiles che verranno selezionati.

        Args:
            image: Immagine su cui disegnare l'anteprima
            preview_tiles: Set di coordinate (x, y) dei tiles in anteprima
        """
        overlay = image.copy()
        alpha = 0.4  # trasparenza per il verde
        for (tx, ty) in preview_tiles:
            pt1 = (tx * self.tile_size, ty * self.tile_size)
            pt2 = ((tx + 1) * self.tile_size, (ty + 1) * self.tile_size)
            cv2.rectangle(overlay, pt1, pt2, (0, 255, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def compute_tiles_in_rect(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        image_shape: Tuple[int, ...]
    ) -> Set[Tuple[int, int]]:
        """
        Calcola i tiles contenuti in un rettangolo.

        Args:
            pt1: Punto iniziale del rettangolo (x, y)
            pt2: Punto finale del rettangolo (x, y)
            image_shape: Dimensioni dell'immagine

        Returns:
            Set di coordinate (x, y) dei tiles contenuti nel rettangolo
        """
        h, w = image_shape[:2]
        x1, y1 = pt1
        x2, y2 = pt2
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        tile_x_start = x_min // self.tile_size
        tile_y_start = y_min // self.tile_size
        tile_x_end = (x_max - 1) // self.tile_size
        tile_y_end = (y_max - 1) // self.tile_size
        tiles = set()
        for ty in range(tile_y_start, tile_y_end + 1):
            for tx in range(tile_x_start, tile_x_end + 1):
                tiles.add((tx, ty))
        return tiles

    def mouse_callback(self, event, x, y, flags, param):
        """
        Callback per gli eventi del mouse.

        Args:
            event: Tipo di evento del mouse
            x, y: Coordinate del mouse
            flags: Flag dell'evento
            param: Parametri aggiuntivi (immagine e dimensioni)
        """
        image, image_shape = param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.mouse_start = (x, y)
            self.mouse_end = (x, y)
            self.preview_tiles = set()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.mouse_end = (x, y)
                self.preview_tiles = self.compute_tiles_in_rect(
                    self.mouse_start, self.mouse_end, image_shape
                )
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.mouse_end = (x, y)
            self.preview_tiles = self.compute_tiles_in_rect(
                self.mouse_start, self.mouse_end, image_shape
            )

    def process_subwindow(
        self,
        sub_img: np.ndarray,
        subwin_index: int,
        previous_selection: Optional[Set[Tuple[int, int]]] = None
    ) -> Tuple[Set[Tuple[int, int]], bool]:
        """
        Funzione interattiva per processare una sottofinestra.

        Args:
            sub_img: Immagine della sottofinestra
            subwin_index: Indice della sottofinestra
            previous_selection: Selezione precedente (se disponibile)

        Returns:
            Tuple contenente:
                - Set di coordinate (x, y) dei tiles selezionati
                - Flag che indica se tornare alla sottofinestra precedente
        """
        # Inizializza le variabili per questa sottofinestra
        self.selected_tiles = set() if previous_selection is None else previous_selection.copy()
        self.preview_tiles = set()
        self.cursor_tile = [0, 0]
        sub_img_shape = sub_img.shape

        window_name = f'Sottofinestra {subwin_index}'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(
            window_name,
            lambda event, x, y, flags, param: self.mouse_callback(event, x, y, flags, param),
            param=(sub_img, sub_img_shape)
        )

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
            disp_img = self.mask_selected_tiles(sub_img, self.selected_tiles)
            self.draw_grid(disp_img)
            self.draw_cursor(disp_img, self.cursor_tile)
            if self.preview_tiles:
                self.draw_preview_tiles(disp_img, self.preview_tiles)
            cv2.imshow(window_name, disp_img)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                cv2.destroyWindow(window_name)
                exit(0)
            elif key == ord('s'):
                # Salva la sottofinestra e passa alla successiva
                cv2.destroyWindow(window_name)
                return self.selected_tiles, False  # False significa "non tornare indietro"
            elif key == ord('b'):
                # Torna alla sottofinestra precedente
                cv2.destroyWindow(window_name)
                return self.selected_tiles, True  # True significa "torna indietro"
            elif key == ord('c'):
                # Conferma la selezione in anteprima: aggiungi i tile
                self.selected_tiles.update(self.preview_tiles)
                self.preview_tiles = set()
            elif key == ord('d'):
                # Rimuove i tile in anteprima dalla selezione se presenti
                self.selected_tiles.difference_update(self.preview_tiles)
                self.preview_tiles = set()
            elif key == ord('r'):
                # Annulla l'anteprima senza modifiche
                self.preview_tiles = set()
            elif key == 32:  # Barra spaziatrice
                # Seleziona/deseleziona il tile corrente
                tile_coord = (self.cursor_tile[0], self.cursor_tile[1])
                if tile_coord in self.selected_tiles:
                    self.selected_tiles.remove(tile_coord)
                else:
                    self.selected_tiles.add(tile_coord)
            # Gestione del cursore con le frecce
            elif key == 81:  # freccia sinistra
                self.cursor_tile[0] = max(0, self.cursor_tile[0] - 1)
            elif key == 82:  # freccia su
                self.cursor_tile[1] = max(0, self.cursor_tile[1] - 1)
            elif key == 83:  # freccia destra
                max_x = sub_img_shape[1] // self.tile_size - 1
                self.cursor_tile[0] = min(max_x, self.cursor_tile[0] + 1)
            elif key == 84:  # freccia giù
                max_y = sub_img_shape[0] // self.tile_size - 1
                self.cursor_tile[1] = min(max_y, self.cursor_tile[1] + 1)

    def update_mask(
        self,
        mask: np.ndarray,
        selected_tiles: Set[Tuple[int, int]],
        offset_x: int,
        offset_y: int
    ) -> np.ndarray:
        """
        Aggiorna la maschera con i tile selezionati nella posizione corretta.

        Args:
            mask: Maschera da aggiornare
            selected_tiles: Set di coordinate (x, y) dei tiles selezionati
            offset_x: Offset orizzontale della sottofinestra
            offset_y: Offset verticale della sottofinestra

        Returns:
            Maschera aggiornata
        """
        for (tx, ty) in selected_tiles:
            # Calcola la posizione assoluta nel contesto dell'immagine completa
            abs_x = offset_x + tx * self.tile_size
            abs_y = offset_y + ty * self.tile_size
            # Assicura che il rettangolo sia all'interno della maschera
            if (abs_y + self.tile_size <= mask.shape[0] and abs_x + self.tile_size <= mask.shape[1]):
                # Imposta la regione corrispondente al tile come bianca (255)
                mask[abs_y:abs_y + self.tile_size, abs_x:abs_x + self.tile_size] = 255
        return mask

    def create_mask(
        self,
        image_path: Union[str, Path],
        output_mask_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Processa un'immagine completa, dividendola in sottofinestre e permettendo
        la selezione interattiva dei tiles per creare una maschera binaria.

        Args:
            image_path: Percorso dell'immagine da processare
            output_mask_path: Percorso dove salvare la maschera risultante

        Returns:
            Maschera binaria con i tiles selezionati
        """
        image_path = Path(image_path)
        full_img = cv2.imread(str(image_path))
        if full_img is None:
            raise FileNotFoundError(f"Errore: il file {image_path} non esiste o non è accessibile.")

        full_h, full_w = full_img.shape[:2]

        # Crea una maschera vuota (nera) delle stesse dimensioni dell'immagine originale
        mask = np.zeros((full_h, full_w), dtype=np.uint8)

        # Calcola il numero di sottofinestre in orizzontale e verticale
        num_subwins_x = math.ceil(full_w / self.subwin_width)
        num_subwins_y = math.ceil(full_h / self.subwin_height)
        total_subwins = num_subwins_x * num_subwins_y

        # Memorizza le coordinate e le selezioni di ogni sottofinestra
        subwin_coords = []
        subwin_selections = []

        # Prepara la lista di coordinate delle sottofinestre
        for j in range(num_subwins_y):
            for i in range(num_subwins_x):
                x_start = i * self.subwin_width
                y_start = j * self.subwin_height
                x_end = min(x_start + self.subwin_width, full_w)
                y_end = min(y_start + self.subwin_height, full_h)
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
            selected_tiles, go_back = self.process_subwindow(
                sub_img,
                subwin_index=current_subwin_idx+1,
                previous_selection=previous_selection
            )

            # Salva le selezioni aggiornate
            subwin_selections[current_subwin_idx] = selected_tiles

            # Aggiorna la maschera
            mask = self.update_mask(mask, selected_tiles, x_start, y_start)

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

        # Salva la maschera finale se richiesto
        if output_mask_path:
            output_mask_path = Path(output_mask_path)
            os.makedirs(output_mask_path.parent, exist_ok=True)
            cv2.imwrite(str(output_mask_path), mask)
            print(f"Maschera salvata in {output_mask_path}")

        return mask


def main():
    """Funzione principale per l'esecuzione come script."""
    import argparse

    parser = argparse.ArgumentParser(description='Crea una maschera di selezione dei tiles da un\'immagine')
    parser.add_argument('input_image', type=str,
                        help='Percorso dell\'immagine da processare [OBBLIGATORIO]')
    parser.add_argument('--output-mask', type=str,
                        help='Percorso dove salvare la maschera risultante')
    parser.add_argument('--tile-size', type=int, default=32,
                        help='Dimensione di un singolo tile in pixel [default=32]')
    parser.add_argument('--tiles-per-subwin', type=int, default=30,
                        help='Numero di tiles per lato in una sottofinestra [default=30]')

    args = parser.parse_args()

    mask_creator = TileMaskCreator(
        tile_size=args.tile_size,
        tiles_per_subwin=args.tiles_per_subwin
    )

    try:
        mask = mask_creator.create_mask(
            args.input_image,
            args.output_mask
        )

        # Visualizza la maschera finale
        cv2.namedWindow("Maschera finale", cv2.WINDOW_NORMAL)
        cv2.imshow("Maschera finale", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Errore: {e}")
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
