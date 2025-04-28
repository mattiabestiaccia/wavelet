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
                 device=None, patch_size=32, stride=16, J=2, use_scattering=None, num_classes=6):
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
            num_classes: Numero di classi nel modello
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.stride = stride
        self.J = J

        # Carica modello se fornito un percorso
        if model_path and model is None:
            self.model, self.class_mapping = load_pixel_classifier(model_path, self.device, num_classes=num_classes)

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
                4: "buildings",
                5: "other"
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

    def process_image(self, image_path, output_path=None, overlay=False, alpha=0.5, batch_size=4, resize_factor=None, show_progress=True):
        """
        Processa un'immagine completa con ottimizzazioni per immagini di grandi dimensioni.

        Args:
            image_path: Percorso dell'immagine o array numpy
            output_path: Percorso di output per la mappa di classificazione
            overlay: Se creare un overlay con l'immagine originale
            alpha: Opacità dell'overlay
            batch_size: Numero di patch da processare in batch (per accelerare l'inferenza)
            resize_factor: Fattore di ridimensionamento dell'immagine (None = nessun ridimensionamento)
            show_progress: Se mostrare la barra di progresso

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

        # Ridimensiona l'immagine se richiesto
        original_size = None
        if resize_factor is not None and resize_factor != 1.0:
            original_size = image.shape[:2]
            new_h, new_w = int(original_size[0] * resize_factor), int(original_size[1] * resize_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Immagine ridimensionata da {original_size} a {image.shape[:2]}")

        # Crea mappa di classificazione
        h, w = image.shape[:2]
        classification_map = np.zeros((h, w), dtype=np.uint8)

        # Calcola il numero totale di patch
        y_positions = list(range(0, h - self.patch_size + 1, self.stride))
        x_positions = list(range(0, w - self.patch_size + 1, self.stride))
        total_patches = len(y_positions) * len(x_positions)

        # Inizializza la barra di progresso
        if show_progress:
            progress_bar = tqdm(total=total_patches, desc="Processando patch")

        # Processa l'immagine a batch di patch per migliorare le prestazioni
        center_size = self.stride

        # Prepara i batch di patch
        for y_batch_start in range(0, len(y_positions), batch_size):
            y_batch_end = min(y_batch_start + batch_size, len(y_positions))
            y_batch = y_positions[y_batch_start:y_batch_end]

            for x_batch_start in range(0, len(x_positions), batch_size):
                x_batch_end = min(x_batch_start + batch_size, len(x_positions))
                x_batch = x_positions[x_batch_start:x_batch_end]

                # Prepara il batch di patch
                batch_patches = []
                batch_positions = []

                for y in y_batch:
                    for x in x_batch:
                        # Estrai patch
                        patch = image[y:y+self.patch_size, x:x+self.patch_size]
                        batch_patches.append(patch)
                        batch_positions.append((y, x))

                # Processa il batch di patch
                batch_results = self.process_patch_batch(batch_patches)

                # Aggiorna la mappa di classificazione
                for i, (y, x) in enumerate(batch_positions):
                    prediction = batch_results[i]

                    # Calcola le coordinate centrali
                    cy, cx = y + (self.patch_size - center_size) // 2, x + (self.patch_size - center_size) // 2

                    # Aggiorna la mappa di classificazione (solo al centro della patch)
                    classification_map[cy:cy+center_size, cx:cx+center_size] = prediction[
                        (self.patch_size - center_size) // 2:(self.patch_size + center_size) // 2,
                        (self.patch_size - center_size) // 2:(self.patch_size + center_size) // 2
                    ]

                # Aggiorna la barra di progresso
                if show_progress:
                    progress_bar.update(len(batch_positions))

        # Chiudi la barra di progresso
        if show_progress:
            progress_bar.close()

        # Processa i bordi
        # Bordo inferiore
        if h > self.patch_size:
            y = h - self.patch_size
            border_patches = []
            border_positions = []

            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                border_patches.append(patch)
                border_positions.append((y, x))

            # Processa il batch di patch del bordo
            if border_patches:
                border_results = self.process_patch_batch(border_patches)

                # Aggiorna la mappa di classificazione
                for i, (y, x) in enumerate(border_positions):
                    prediction = border_results[i]
                    cy = y + (self.patch_size - center_size) // 2
                    cx = x + (self.patch_size - center_size) // 2
                    classification_map[cy:h, cx:cx+center_size] = prediction[
                        (self.patch_size - center_size) // 2:self.patch_size,
                        (self.patch_size - center_size) // 2:(self.patch_size + center_size) // 2
                    ]

        # Bordo destro
        if w > self.patch_size:
            x = w - self.patch_size
            border_patches = []
            border_positions = []

            for y in range(0, h - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                border_patches.append(patch)
                border_positions.append((y, x))

            # Processa il batch di patch del bordo
            if border_patches:
                border_results = self.process_patch_batch(border_patches)

                # Aggiorna la mappa di classificazione
                for i, (y, x) in enumerate(border_positions):
                    prediction = border_results[i]
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

        # Ridimensiona la mappa di classificazione alle dimensioni originali se necessario
        if original_size is not None:
            classification_map = cv2.resize(classification_map, (original_size[1], original_size[0]),
                                           interpolation=cv2.INTER_NEAREST)

        # Salva o visualizza risultati
        if output_path or overlay:
            # Crea mappa colorata
            color_map = self.create_color_map(classification_map)

            if overlay:
                # Ridimensiona l'immagine originale se necessario
                if original_size is not None:
                    image = cv2.resize(image, (original_size[1], original_size[0]), interpolation=cv2.INTER_AREA)

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

    def process_patch_batch(self, patches):
        """
        Processa un batch di patch contemporaneamente per migliorare le prestazioni.

        Args:
            patches: Lista di patch di immagine (numpy array)

        Returns:
            Lista di tensori con le predizioni di classe
        """
        if not patches:
            return []

        # Converti a tensori
        batch_tensors = []
        for patch in patches:
            if isinstance(patch, np.ndarray):
                tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
                batch_tensors.append(tensor)

        # Crea batch
        batch = torch.stack(batch_tensors).to(self.device)

        # Applica modello (con o senza scattering)
        with torch.no_grad():
            if self.use_scattering and self.scattering is not None:
                # Con trasformata scattering
                # Processa ogni patch separatamente per evitare problemi di memoria
                outputs = []
                for i in range(batch.size(0)):
                    scattering_coeffs = self.scattering(batch[i:i+1])
                    
                    # Gestisci la dimensionalità dell'output
                    if scattering_coeffs.dim() == 5:
                        # Rimuovi l'ultima dimensione o prendi solo la parte reale
                        if scattering_coeffs.shape[-1] == 1:
                            scattering_coeffs = scattering_coeffs.squeeze(-1)
                        else:
                            # Prendi solo la parte reale (prima componente)
                            scattering_coeffs = scattering_coeffs[..., 0]
                    
                    output = self.model(scattering_coeffs)
                    outputs.append(output)
                outputs = torch.cat(outputs, dim=0)
            else:
                # Senza trasformata scattering
                outputs = self.model(batch)

            # Ottieni predizioni
            _, predictions = torch.max(outputs, dim=1)

        # Converti a numpy
        return predictions.cpu().numpy()

    def create_color_map(self, classification_map):
        """
        Crea una mappa colorata dalla mappa di classificazione.

        Args:
            classification_map: Mappa di classificazione

        Returns:
            Mappa colorata
        """
        # Ottieni i valori unici nella mappa di classificazione
        unique_classes = np.unique(classification_map)

        # Definisci colori predefiniti per le classi comuni
        default_colors = {
            0: [0, 0, 0],       # Background (nero)
            1: [0, 0, 255],     # Acqua (blu)
            2: [0, 255, 0],     # Vegetazione (verde)
            3: [128, 128, 128], # Strade (grigio)
            4: [255, 0, 0],     # Edifici (rosso)
            5: [255, 255, 0],   # Altro/Sabbia (giallo)
            6: [255, 0, 255],   # Magenta
            7: [0, 255, 255],   # Ciano
            8: [128, 0, 0],     # Marrone scuro
            9: [0, 128, 0]      # Verde scuro
        }

        # Colori aggiuntivi per classi oltre le 10 predefinite
        additional_colors = [
            [255, 128, 0],   # Arancione
            [128, 0, 128],   # Viola
            [128, 128, 0],   # Oliva
            [0, 128, 128],   # Teal
            [128, 255, 0],   # Lime chiaro
            [255, 0, 128],   # Rosa
            [0, 128, 255],   # Azzurro
            [255, 128, 128], # Rosa chiaro
            [128, 255, 128], # Verde chiaro
            [128, 128, 255]  # Lavanda
        ]

        # Inizializza dizionario dei colori
        colors = {}

        # Assegna colori alle classi
        for idx in unique_classes:
            if idx in default_colors:
                colors[idx] = default_colors[idx]
            else:
                # Assegna un colore dall'elenco aggiuntivo per classi non predefinite
                color_idx = (idx - len(default_colors)) % len(additional_colors)
                colors[idx] = additional_colors[color_idx]

        # Personalizza colori in base ai nomi delle classi se disponibili
        if self.class_mapping:
            for idx, class_name in self.idx_to_class.items():
                # Skip if class_name is not a string
                if not isinstance(class_name, str):
                    continue
                    
                class_name_lower = class_name.lower()

                # Assegna colori in base al nome della classe
                if any(water_term in class_name_lower for water_term in ["water", "acqua", "corso_acqua", "acqua_esterna", "fiume", "lago"]):
                    colors[idx] = [0, 0, 255]  # Blu
                elif any(veg_term in class_name_lower for veg_term in ["vegetation", "vegetazione", "alberi", "foresta", "piante"]):
                    colors[idx] = [0, 255, 0]  # Verde
                elif any(road_term in class_name_lower for road_term in ["street", "strada", "strade", "asfalto", "autostrada"]):
                    colors[idx] = [128, 128, 128]  # Grigio
                elif any(building_term in class_name_lower for building_term in ["building", "edificio", "edifici", "costruzione", "casa"]):
                    colors[idx] = [255, 0, 0]  # Rosso
                elif any(sand_term in class_name_lower for sand_term in ["sand", "sabbia", "mudflat", "spiaggia", "deserto"]):
                    colors[idx] = [255, 255, 0]  # Giallo
                elif any(other_term in class_name_lower for other_term in ["other", "altro", "vario", "misc"]):
                    colors[idx] = [255, 0, 255]  # Magenta

        # Crea mappa colorata
        h, w = classification_map.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)

        # Applica i colori alla mappa
        for idx in unique_classes:
            if idx in colors:
                color_map[classification_map == idx] = colors[idx]
            else:
                # Colore di fallback per classi senza colore assegnato
                color_map[classification_map == idx] = [128, 128, 128]  # Grigio

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
        Visualizza i risultati della classificazione con statistiche dettagliate.

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

        # Calcola statistiche
        unique_classes, counts = np.unique(classification_map, return_counts=True)
        total_pixels = classification_map.size
        class_percentages = {cls: (count / total_pixels) * 100 for cls, count in zip(unique_classes, counts)}

        # Crea figura con layout flessibile
        fig = plt.figure(figsize=(18, 10))

        # Definisci il layout della griglia
        gs = plt.GridSpec(2, 3, height_ratios=[3, 1], figure=fig)

        # Immagine originale
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title('Immagine Originale', fontsize=12)
        ax1.axis('off')

        # Mappa di classificazione
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(color_map)
        ax2.set_title('Mappa di Classificazione', fontsize=12)
        ax2.axis('off')

        # Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = cv2.addWeighted(image, 0.7, color_map, 0.3, 0)
        ax3.imshow(overlay)
        ax3.set_title('Overlay', fontsize=12)
        ax3.axis('off')

        # Statistiche e legenda
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')

        # Crea tabella di statistiche
        table_data = []
        colors_for_legend = []

        for cls in sorted(unique_classes):
            if cls in self.idx_to_class:
                class_name = self.idx_to_class[cls]
            else:
                class_name = f"Classe {cls}"

            count = counts[np.where(unique_classes == cls)[0][0]]
            percentage = class_percentages[cls]

            # Ottieni il colore per questa classe
            color_idx = np.where(classification_map == cls)
            if len(color_idx[0]) > 0:
                color = color_map[color_idx[0][0], color_idx[1][0]]
                colors_for_legend.append(color / 255.0)  # Normalizza per matplotlib
            else:
                colors_for_legend.append([0.5, 0.5, 0.5])  # Grigio di default

            table_data.append([class_name, f"{count:,}", f"{percentage:.2f}%"])

        # Crea tabella
        table = ax4.table(
            cellText=table_data,
            colLabels=["Classe", "Pixel", "Percentuale"],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.3, 0.3]
        )

        # Formatta tabella
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Colora le celle della tabella in base alle classi
        for i, color in enumerate(colors_for_legend):
            table[(i+1, 0)].set_facecolor(color)
            # Imposta il testo in bianco o nero in base alla luminosità del colore
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = 'white' if luminance < 0.5 else 'black'
            table[(i+1, 0)].get_text().set_color(text_color)

        plt.tight_layout()

        # Aggiungi titolo generale
        plt.suptitle(f"Risultati Classificazione Pixel-Wise - {Path(image_path).stem if isinstance(image_path, (str, Path)) else 'Immagine'}",
                    fontsize=14, y=0.98)

        # Salva se richiesto
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            print(f"Visualizzazione salvata in: {output_path}")

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
            4: [255, 0, 0],     # Edifici (rosso)
            5: [255, 255, 0]    # Altro (giallo)
        }

        # Personalizza colori se necessario
        if self.class_mapping:
            for idx, class_name in self.idx_to_class.items():
                # Handle non-string class names
                if not isinstance(class_name, str):
                    continue
                    
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
                # Handle non-string class names
                if not isinstance(class_name, str):
                    class_name = f"Class {idx}"
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
