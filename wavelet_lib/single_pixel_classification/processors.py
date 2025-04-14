"""
Modulo di processori per la classificazione pixel-wise nella Wavelet Scattering Transform Library.
Contiene classi e funzioni per l'inferenza e la visualizzazione dei risultati.
"""

import os
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
from kymatio.torch import Scattering2D

from wavelet_lib.single_pixel_classification.models import create_scattering_transform, load_pixel_classifier


class PixelClassificationProcessor:
    """Classe per processare e classificare immagini pixel per pixel."""

    def __init__(self, model=None, scattering=None, model_path=None, class_mapping=None,
                 device=None, patch_size=32, stride=16, J=2, use_scattering=None):
        """
        Inizializza il processore di classificazione pixel-wise.

        Args:
            model: Modello di classificazione
            scattering: Trasformata scattering
            model_path: Percorso del modello (alternativa a model)
            class_mapping: Mapping delle classi (dict)
            device: Device per l'inferenza
            patch_size: Dimensione delle patch per l'inferenza
            stride: Passo per l'inferenza
            J: Numero di scale per la trasformata scattering
            use_scattering: Se utilizzare la trasformata scattering (se None, usa il valore del modello)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.stride = stride
        self.J = J

        # Carica modello se fornito un percorso
        if model_path and model is None:
            self.model, self.class_mapping = load_pixel_classifier(model_path, self.device)

            # Determina se usare la trasformata scattering
            self.use_scattering = use_scattering if use_scattering is not None else getattr(self.model, 'use_scattering', True)

            # Crea trasformata scattering se necessario
            if self.use_scattering:
                self.scattering = create_scattering_transform(
                    J=J,
                    shape=(patch_size, patch_size),
                    device=self.device
                )
            else:
                self.scattering = None
        else:
            self.model = model
            self.scattering = scattering
            self.class_mapping = class_mapping
            self.use_scattering = use_scattering if use_scattering is not None else getattr(self.model, 'use_scattering', True)

        # Verifica che il modello sia definito
        if self.model is None:
            raise ValueError("È necessario fornire model o model_path")

        # Verifica che la trasformata scattering sia definita se necessaria
        if self.use_scattering and self.scattering is None:
            raise ValueError("È necessario fornire scattering quando use_scattering=True")

        # Crea mapping inverso
        if self.class_mapping:
            self.idx_to_class = {v: k for k, v in self.class_mapping.items()} if isinstance(self.class_mapping, dict) else {i: c for i, c in enumerate(self.class_mapping)}
        else:
            self.idx_to_class = {
                0: "background",
                1: "water",
                2: "vegetation",
                3: "streets",
                4: "buildings"
            }
            self.class_mapping = {v: k for k, v in self.idx_to_class.items()}

    def process_patch(self, patch):
        """
        Processa una singola patch.

        Args:
            patch: Patch di immagine (numpy array)

        Returns:
            Tensore con le predizioni di classe
        """
        # Converti a tensore
        if isinstance(patch, np.ndarray):
            patch = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0

        # Aggiungi dimensione batch
        if len(patch.shape) == 3:
            patch = patch.unsqueeze(0)

        # Sposta su device
        patch = patch.to(self.device)

        # Applica modello (con o senza scattering)
        with torch.no_grad():
            if self.use_scattering and self.scattering is not None:
                # Con trasformata scattering
                scattering_coeffs = self.scattering(patch)
                outputs = self.model(scattering_coeffs)
            else:
                # Senza trasformata scattering
                outputs = self.model(patch)

            # Ottieni predizione
            _, prediction = torch.max(outputs, dim=1)

        return prediction.cpu().numpy()[0]

    def process_image(self, image_path, output_path=None, overlay=False, alpha=0.5):
        """
        Processa un'immagine completa.

        Args:
            image_path: Percorso dell'immagine
            output_path: Percorso di output per la mappa di classificazione
            overlay: Se creare un overlay con l'immagine originale
            alpha: Opacità dell'overlay

        Returns:
            Mappa di classificazione
        """
        # Carica immagine
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Impossibile caricare l'immagine: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path

        # Crea mappa di classificazione
        h, w = image.shape[:2]
        classification_map = np.zeros((h, w), dtype=np.uint8)

        # Processa l'immagine a patch
        for y in tqdm(range(0, h - self.patch_size + 1, self.stride), desc="Processando righe"):
            for x in range(0, w - self.patch_size + 1, self.stride):
                # Estrai patch
                patch = image[y:y+self.patch_size, x:x+self.patch_size]

                # Classifica patch
                prediction = self.process_patch(patch)

                # Aggiorna mappa di classificazione (solo al centro della patch)
                center_size = self.stride
                cy, cx = y + (self.patch_size - center_size) // 2, x + (self.patch_size - center_size) // 2
                classification_map[cy:cy+center_size, cx:cx+center_size] = prediction[
                    (self.patch_size - center_size) // 2:(self.patch_size + center_size) // 2,
                    (self.patch_size - center_size) // 2:(self.patch_size + center_size) // 2
                ]

        # Processa i bordi
        # Bordo inferiore
        if h > self.patch_size:
            y = h - self.patch_size
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                prediction = self.process_patch(patch)
                cy = y + (self.patch_size - center_size) // 2
                cx = x + (self.patch_size - center_size) // 2
                classification_map[cy:h, cx:cx+center_size] = prediction[
                    (self.patch_size - center_size) // 2:self.patch_size,
                    (self.patch_size - center_size) // 2:(self.patch_size + center_size) // 2
                ]

        # Bordo destro
        if w > self.patch_size:
            x = w - self.patch_size
            for y in range(0, h - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                prediction = self.process_patch(patch)
                cy = y + (self.patch_size - center_size) // 2
                cx = x + (self.patch_size - center_size) // 2
                classification_map[cy:cy+center_size, cx:w] = prediction[
                    (self.patch_size - center_size) // 2:(self.patch_size + center_size) // 2,
                    (self.patch_size - center_size) // 2:self.patch_size
                ]

        # Angolo in basso a destra
        if h > self.patch_size and w > self.patch_size:
            y, x = h - self.patch_size, w - self.patch_size
            patch = image[y:y+self.patch_size, x:x+self.patch_size]
            prediction = self.process_patch(patch)
            cy = y + (self.patch_size - center_size) // 2
            cx = x + (self.patch_size - center_size) // 2
            classification_map[cy:h, cx:w] = prediction[
                (self.patch_size - center_size) // 2:self.patch_size,
                (self.patch_size - center_size) // 2:self.patch_size
            ]

        # Salva o visualizza risultati
        if output_path or overlay:
            # Crea mappa colorata
            color_map = self.create_color_map(classification_map)

            if overlay:
                # Crea overlay
                overlay_img = image.copy()
                overlay_img = cv2.addWeighted(overlay_img, 1-alpha, color_map, alpha, 0)

                if output_path:
                    # Salva overlay
                    overlay_path = str(Path(output_path).with_suffix('')) + "_overlay.png"
                    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
                    print(f"Overlay salvato in: {overlay_path}")

                # Visualizza overlay
                plt.figure(figsize=(12, 8))
                plt.imshow(overlay_img)
                plt.title("Classificazione Pixel-Wise (Overlay)")
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            if output_path:
                # Salva mappa di classificazione
                cv2.imwrite(output_path, classification_map)

                # Salva mappa colorata
                color_path = str(Path(output_path).with_suffix('')) + "_color.png"
                cv2.imwrite(color_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
                print(f"Mappa di classificazione salvata in: {output_path}")
                print(f"Mappa colorata salvata in: {color_path}")

        return classification_map

    def create_color_map(self, classification_map):
        """
        Crea una mappa colorata dalla mappa di classificazione.

        Args:
            classification_map: Mappa di classificazione

        Returns:
            Mappa colorata
        """
        # Definisci colori per le classi
        colors = {
            0: [0, 0, 0],       # Background (nero)
            1: [0, 0, 255],     # Acqua (blu)
            2: [0, 255, 0],     # Vegetazione (verde)
            3: [128, 128, 128], # Strade (grigio)
            4: [255, 0, 0]      # Edifici (rosso)
        }

        # Personalizza colori se necessario
        if self.class_mapping:
            for idx, class_name in self.idx_to_class.items():
                if class_name.lower() in ["water", "acqua", "corso_acqua", "acqua_esterna"]:
                    colors[idx] = [0, 0, 255]  # Blu
                elif class_name.lower() in ["vegetation", "vegetazione", "vegetazione_bassa", "vegetazione_alta", "alberi"]:
                    colors[idx] = [0, 255, 0]  # Verde
                elif class_name.lower() in ["street", "strada", "strade"]:
                    colors[idx] = [128, 128, 128]  # Grigio
                elif class_name.lower() in ["building", "edificio", "edifici"]:
                    colors[idx] = [255, 0, 0]  # Rosso
                elif class_name.lower() in ["sand", "sabbia", "mudflat"]:
                    colors[idx] = [255, 255, 0]  # Giallo

        # Crea mappa colorata
        h, w = classification_map.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)

        for idx, color in colors.items():
            color_map[classification_map == idx] = color

        return color_map

    def process_folder(self, folder_path, output_dir, overlay=False, alpha=0.5):
        """
        Processa tutte le immagini in una cartella.

        Args:
            folder_path: Percorso della cartella con le immagini
            output_dir: Directory di output
            overlay: Se creare overlay
            alpha: Opacità dell'overlay

        Returns:
            Lista di percorsi delle mappe di classificazione
        """
        # Crea directory di output
        os.makedirs(output_dir, exist_ok=True)

        # Trova tutte le immagini
        folder_path = Path(folder_path)
        image_paths = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + list(folder_path.glob("*.tif"))

        if not image_paths:
            print(f"Nessuna immagine trovata in: {folder_path}")
            return []

        # Processa ogni immagine
        output_paths = []
        for img_path in tqdm(image_paths, desc="Processando immagini"):
            # Crea percorso di output
            output_path = Path(output_dir) / f"{img_path.stem}_classification.png"

            try:
                # Processa immagine
                self.process_image(
                    image_path=img_path,
                    output_path=str(output_path),
                    overlay=overlay,
                    alpha=alpha
                )

                output_paths.append(output_path)
            except Exception as e:
                print(f"Errore nel processare {img_path}: {e}")

        return output_paths

    def visualize_results(self, image_path, classification_map=None, output_path=None):
        """
        Visualizza i risultati della classificazione.

        Args:
            image_path: Percorso dell'immagine originale
            classification_map: Mappa di classificazione (opzionale)
            output_path: Percorso di output per l'immagine
        """
        # Carica immagine
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Impossibile caricare l'immagine: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path

        # Processa immagine se la mappa non è fornita
        if classification_map is None:
            classification_map = self.process_image(image)

        # Crea mappa colorata
        color_map = self.create_color_map(classification_map)

        # Visualizza risultati
        plt.figure(figsize=(15, 5))

        # Immagine originale
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Immagine Originale')
        plt.axis('off')

        # Mappa di classificazione
        plt.subplot(1, 3, 2)
        plt.imshow(color_map)
        plt.title('Mappa di Classificazione')
        plt.axis('off')

        # Overlay
        plt.subplot(1, 3, 3)
        overlay = cv2.addWeighted(image, 0.7, color_map, 0.3, 0)
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')

        plt.tight_layout()

        # Salva se richiesto
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        plt.show()

    def create_legend(self, output_path=None):
        """
        Crea una legenda per la mappa di classificazione.

        Args:
            output_path: Percorso di output per la legenda
        """
        # Crea figura
        plt.figure(figsize=(8, 4))

        # Definisci colori per le classi
        colors = {
            0: [0, 0, 0],       # Background (nero)
            1: [0, 0, 255],     # Acqua (blu)
            2: [0, 255, 0],     # Vegetazione (verde)
            3: [128, 128, 128], # Strade (grigio)
            4: [255, 0, 0]      # Edifici (rosso)
        }

        # Personalizza colori se necessario
        if self.class_mapping:
            for idx, class_name in self.idx_to_class.items():
                if class_name.lower() in ["water", "acqua", "corso_acqua", "acqua_esterna"]:
                    colors[idx] = [0, 0, 255]  # Blu
                elif class_name.lower() in ["vegetation", "vegetazione", "vegetazione_bassa", "vegetazione_alta", "alberi"]:
                    colors[idx] = [0, 255, 0]  # Verde
                elif class_name.lower() in ["street", "strada", "strade"]:
                    colors[idx] = [128, 128, 128]  # Grigio
                elif class_name.lower() in ["building", "edificio", "edifici"]:
                    colors[idx] = [255, 0, 0]  # Rosso
                elif class_name.lower() in ["sand", "sabbia", "mudflat"]:
                    colors[idx] = [255, 255, 0]  # Giallo

        # Crea legenda
        for i, (idx, color) in enumerate(colors.items()):
            if idx in self.idx_to_class:
                class_name = self.idx_to_class[idx]
                plt.bar(0, 0, color=[c/255 for c in color], label=class_name)

        plt.legend(loc='center', fontsize=12)
        plt.axis('off')
        plt.title('Legenda della Classificazione')

        # Salva se richiesto
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        plt.show()


# Funzioni di utilità per l'uso diretto

def process_image(image_path, model_path, output_path=None, overlay=False, patch_size=32, stride=16, J=2):
    """
    Processa un'immagine con un modello di classificazione pixel-wise.

    Args:
        image_path: Percorso dell'immagine
        model_path: Percorso del modello
        output_path: Percorso di output
        overlay: Se creare un overlay
        patch_size: Dimensione delle patch
        stride: Passo per l'inferenza
        J: Numero di scale per la trasformata scattering

    Returns:
        Mappa di classificazione
    """
    processor = PixelClassificationProcessor(
        model_path=model_path,
        patch_size=patch_size,
        stride=stride,
        J=J
    )

    return processor.process_image(
        image_path=image_path,
        output_path=output_path,
        overlay=overlay
    )


def create_classification_map(image_path, model_path, output_dir, overlay=False, patch_size=32, stride=16, J=2):
    """
    Crea una mappa di classificazione per un'immagine.

    Args:
        image_path: Percorso dell'immagine
        model_path: Percorso del modello
        output_dir: Directory di output
        overlay: Se creare un overlay
        patch_size: Dimensione delle patch
        stride: Passo per l'inferenza
        J: Numero di scale per la trasformata scattering

    Returns:
        Percorso della mappa di classificazione
    """
    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)

    # Crea percorso di output
    output_path = os.path.join(output_dir, f"{Path(image_path).stem}_classification.png")

    # Processa immagine
    process_image(
        image_path=image_path,
        model_path=model_path,
        output_path=output_path,
        overlay=overlay,
        patch_size=patch_size,
        stride=stride,
        J=J
    )

    return output_path
