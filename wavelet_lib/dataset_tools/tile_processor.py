#!/usr/bin/env python3
"""
Modulo per l'elaborazione e il salvataggio di tiles da immagini in dataset classificati.
Fornisce strumenti per estrarre e salvare tiles da immagini in cartelle specifiche per classe,
con supporto per immagini multibanda e creazione di dataset per machine learning.
"""

import cv2
import numpy as np
import os
import math
import rasterio
from pathlib import Path
from typing import Set, Tuple, List, Dict, Optional, Union, Callable


class TileProcessor:
    """
    Classe per l'elaborazione e il salvataggio di tiles da immagini in dataset classificati.
    Supporta sia immagini RGB standard che immagini multibanda, e organizza i tiles
    in cartelle specifiche per classe per facilitare l'addestramento di modelli di machine learning.

    Per impostazione predefinita, la classe è configurata per:
    - Supportare immagini multibanda (use_rasterio=True)
    - Organizzare i tiles in sottocartelle per classe (dataset_mode=True)

    Esempi:
        # Modalità dataset (predefinita)
        processor = TileProcessor(output_dir='/path/to/dataset')
        processor.process_image_to_tiles(
            'path/to/image.tif',
            class_name='healthy'
        )
        # I tiles verranno salvati in: /path/to/dataset/healthy/

        # Modalità flat (disabilitando dataset_mode)
        processor = TileProcessor(output_dir='/path/to/output', dataset_mode=False)
        processor.process_image_to_tiles('path/to/image.tif')
        # I tiles verranno salvati in: /path/to/output/
    """

    def __init__(
        self,
        tile_size: int = 32,
        output_dir: Optional[Union[str, Path]] = None,
        use_rasterio: bool = True,
        dataset_mode: bool = True
    ):
        """
        Inizializza il processore di tiles.

        Args:
            tile_size: Dimensione di un singolo tile in pixel
            output_dir: Directory di output per i tiles salvati
            use_rasterio: Se True, usa rasterio per il caricamento delle immagini (supporta multibanda)
            dataset_mode: Se True, organizza i tiles in sottocartelle per classe
        """
        self.tile_size = tile_size
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_rasterio = use_rasterio
        self.dataset_mode = dataset_mode

    def set_output_directory(self, output_dir: Union[str, Path]) -> None:
        """
        Imposta la directory di output per i tiles salvati.

        Args:
            output_dir: Directory di output
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Directory di output configurata: {self.output_dir}")

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Carica un'immagine dal disco.

        Args:
            image_path: Percorso dell'immagine

        Returns:
            Immagine caricata come array numpy
        """
        image_path = Path(image_path)

        if self.use_rasterio:
            with rasterio.open(str(image_path)) as src:
                # Leggi tutte le bande
                image = src.read()
                # Riorganizza le dimensioni da (bands, height, width) a (height, width, bands)
                image = np.transpose(image, (1, 2, 0))
                return image
        else:
            # Usa OpenCV per immagini standard (RGB, grayscale)
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Errore: il file {image_path} non esiste o non è accessibile.")
            return image

    def save_tile(
        self,
        tile: np.ndarray,
        output_path: Union[str, Path],
        format: str = 'jpg'
    ) -> None:
        """
        Salva un tile su disco.

        Args:
            tile: Array numpy contenente il tile
            output_path: Percorso dove salvare il tile
            format: Formato di output ('jpg', 'png', 'tif')
        """
        output_path = Path(output_path)

        # Assicurati che la directory esista
        os.makedirs(output_path.parent, exist_ok=True)

        if format.lower() in ['jpg', 'jpeg', 'png']:
            # Per formati standard, usa OpenCV
            cv2.imwrite(str(output_path), tile)
        elif format.lower() in ['tif', 'tiff'] and self.use_rasterio:
            # Per TIFF multibanda, usa rasterio
            # Riorganizza le dimensioni da (height, width, bands) a (bands, height, width)
            if len(tile.shape) == 3:
                tile_transposed = np.transpose(tile, (2, 0, 1))
                num_bands = tile.shape[2]
            else:
                # Immagine a singola banda
                tile_transposed = tile[np.newaxis, :, :]
                num_bands = 1

            # Crea il profilo per il file di output
            profile = {
                'driver': 'GTiff',
                'height': tile.shape[0],
                'width': tile.shape[1],
                'count': num_bands,
                'dtype': tile.dtype,
                'compress': 'lzw'
            }

            with rasterio.open(str(output_path), 'w', **profile) as dst:
                if num_bands == 1:
                    dst.write(tile_transposed)
                else:
                    for band_idx in range(num_bands):
                        dst.write(tile_transposed[band_idx], band_idx + 1)
        else:
            raise ValueError(f"Formato non supportato: {format}")

    def extract_tiles(
        self,
        image: np.ndarray,
        tile_size: Optional[int] = None,
        overlap: int = 0,
        min_valid_fraction: float = 0.0,
        mask: Optional[np.ndarray] = None
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Estrae tiles da un'immagine.

        Args:
            image: Immagine da cui estrarre i tiles
            tile_size: Dimensione dei tiles (se None, usa self.tile_size)
            overlap: Sovrapposizione tra tiles adiacenti in pixel
            min_valid_fraction: Frazione minima di pixel validi (non zero in mask) per considerare un tile
            mask: Maschera opzionale per filtrare i tiles (solo i tiles con pixel non zero nella maschera vengono estratti)

        Returns:
            Dizionario di tiles estratti, con chiavi (x, y) e valori gli array dei tiles
        """
        if tile_size is None:
            tile_size = self.tile_size

        h, w = image.shape[:2]
        tiles = {}

        # Calcola il passo tra i tiles (considerando l'overlap)
        step = tile_size - overlap

        for y in range(0, h - tile_size + 1, step):
            for x in range(0, w - tile_size + 1, step):
                # Estrai il tile
                tile = image[y:y + tile_size, x:x + tile_size]

                # Se c'è una maschera, controlla se il tile contiene abbastanza pixel validi
                if mask is not None:
                    mask_tile = mask[y:y + tile_size, x:x + tile_size]
                    valid_fraction = np.count_nonzero(mask_tile) / (tile_size * tile_size)
                    if valid_fraction < min_valid_fraction:
                        continue

                # Aggiungi il tile al dizionario
                tile_coord = (x // step, y // step)
                tiles[tile_coord] = tile

        return tiles

    def save_tiles(
        self,
        tiles: Dict[Tuple[int, int], np.ndarray],
        base_filename: str,
        output_dir: Optional[Union[str, Path]] = None,
        format: str = 'jpg'
    ) -> List[Path]:
        """
        Salva i tiles su disco.

        Args:
            tiles: Dizionario di tiles da salvare
            base_filename: Nome base per i file di output
            output_dir: Directory di output (se None, usa self.output_dir)
            format: Formato di output ('jpg', 'png', 'tif')

        Returns:
            Lista dei percorsi dei files salvati
        """
        if output_dir is None:
            if self.output_dir is None:
                raise ValueError("Nessuna directory di output specificata")
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        saved_files = []

        for (tx, ty), tile in tiles.items():
            filename = f"{base_filename}_tile_{tx}_{ty}.{format}"
            output_path = output_dir / filename
            self.save_tile(tile, output_path, format)
            saved_files.append(output_path)

        return saved_files

    def process_image_to_tiles(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        mask_path: Optional[Union[str, Path]] = None,
        tile_size: Optional[int] = None,
        overlap: int = 0,
        min_valid_fraction: float = 0.0,
        format: str = 'jpg'
    ) -> List[Path]:
        """
        Processa un'immagine completa, estraendo e salvando i tiles.

        Args:
            image_path: Percorso dell'immagine da processare
            output_dir: Directory di output (se None, usa self.output_dir)
            mask_path: Percorso opzionale di una maschera per filtrare i tiles
            tile_size: Dimensione dei tiles (se None, usa self.tile_size)
            overlap: Sovrapposizione tra tiles adiacenti in pixel
            min_valid_fraction: Frazione minima di pixel validi per considerare un tile
            format: Formato di output ('jpg', 'png', 'tif')

        Returns:
            Lista dei percorsi dei files salvati
        """
        # Imposta la directory di output
        if output_dir is not None:
            self.set_output_directory(output_dir)
        elif self.output_dir is None:
            raise ValueError("Nessuna directory di output specificata")

        # Carica l'immagine
        image_path = Path(image_path)
        image = self.load_image(image_path)

        # Carica la maschera se specificata
        mask = None
        if mask_path:
            mask_path = Path(mask_path)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Errore: il file maschera {mask_path} non esiste o non è accessibile.")

        # Estrai i tiles
        tiles = self.extract_tiles(
            image,
            tile_size=tile_size,
            overlap=overlap,
            min_valid_fraction=min_valid_fraction,
            mask=mask
        )

        # Salva i tiles
        base_filename = image_path.stem
        saved_files = self.save_tiles(
            tiles,
            base_filename,
            self.output_dir,
            format
        )

        print(f"Salvati {len(saved_files)} tiles da {image_path}")
        return saved_files

    def save_subwindow_tiles(
        self,
        sub_img: np.ndarray,
        selected_tiles: Set[Tuple[int, int]],
        subwin_index: int,
        base_filename: str,
        output_dir: Optional[Union[str, Path]] = None,
        format: str = 'jpg'
    ) -> List[Path]:
        """
        Salva i tiles selezionati da una sottofinestra.

        Args:
            sub_img: Immagine della sottofinestra
            selected_tiles: Set di coordinate (x, y) dei tiles selezionati
            subwin_index: Indice della sottofinestra
            base_filename: Nome base per i file di output
            output_dir: Directory di output (se None, usa self.output_dir)
            format: Formato di output ('jpg', 'png', 'tif')

        Returns:
            Lista dei percorsi dei files salvati
        """
        if output_dir is None:
            if self.output_dir is None:
                raise ValueError("Nessuna directory di output specificata")
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        h, w = sub_img.shape[:2]

        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                if y + self.tile_size > h or x + self.tile_size > w:
                    continue

                tile_coord = (x // self.tile_size, y // self.tile_size)

                if tile_coord in selected_tiles:
                    tile = sub_img[y:y + self.tile_size, x:x + self.tile_size]
                    filename = f"{base_filename}_subwin_{subwin_index}_tile_{tile_coord[0]}_{tile_coord[1]}.{format}"
                    output_path = output_dir / filename
                    self.save_tile(tile, output_path, format)
                    saved_files.append(output_path)

        return saved_files


def main():
    """Funzione principale per l'esecuzione come script."""
    import argparse

    parser = argparse.ArgumentParser(description='Estrai e salva tiles da un\'immagine')
    parser.add_argument('input_image',
                        type=str,
                        help='Percorso dell\'immagine da processare [NECESSARY]')
    parser.add_argument('output_dir',
                        type=str,
                        help='Directory di output per i tiles [NECESSARY]')
    parser.add_argument('--mask',
                        type=str,
                        help='Percorso di una maschera per filtrare i tiles')
    parser.add_argument('--tile-size',
                        type=int,
                        default=32,
                        help='Dimensione dei tiles in pixel [default=32]')
    parser.add_argument('--overlap',
                        type=int,
                        default=0,
                        help='Sovrapposizione tra tiles adiacenti in pixel [default=0]')
    parser.add_argument('--min-valid-fraction',
                        type=float,
                        default=0.5,
                        help='Frazione minima di pixel validi per considerare un tile [default=0.5]')
    parser.add_argument('--format',
                        type=str,
                        default='tif',
                        choices=['jpg', 'png', 'tif'],
                        help='Formato di output per i tiles (jpg, png, tif) [default=tif]')
    parser.add_argument('--no-rasterio',
                        action='store_true',
                        help='Disabilita l\'uso di rasterio per il caricamento delle immagini [default=False]')
    parser.add_argument('--class-name',
                        type=str,
                        help='Nome della classe per organizzare i tiles in sottocartelle')
    parser.add_argument('--no-dataset-mode',
                        action='store_true',
                        help='Disabilita la modalità dataset (non organizza i tiles in sottocartelle per classe) [default=False]')

    args = parser.parse_args()

    processor = TileProcessor(
        tile_size=args.tile_size,
        output_dir=args.output_dir,
        use_rasterio=not args.no_rasterio,
        dataset_mode=not args.no_dataset_mode
    )

    try:
        saved_files = processor.process_image_to_tiles(
            args.input_image,
            mask_path=args.mask,
            overlap=args.overlap,
            min_valid_fraction=args.min_valid_fraction,
            format=args.format,
            class_name=args.class_name
        )

        print(f"Salvati {len(saved_files)} tiles in {args.output_dir}")
        if args.class_name and not args.no_dataset_mode:
            print(f"I tiles sono stati organizzati nella classe: {args.class_name}")

    except Exception as e:
        print(f"Errore: {e}")
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
