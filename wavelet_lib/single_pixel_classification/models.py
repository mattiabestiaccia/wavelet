"""
Modulo di modelli per la classificazione pixel-wise nella Wavelet Scattering Transform Library.
Contiene modelli di reti neurali per la classificazione di singoli pixel con trasformate scattering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D
import numpy as np
import os
# Importa PIL.Image solo se necessario
try:
    from PIL import Image
except ImportError:
    pass
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


def create_scattering_transform(J=2, shape=(32, 32), max_order=2, device=None):
    """
    Crea una trasformata scattering.

    Args:
        J: Numero di scale
        shape: Forma delle immagini di input
        max_order: Ordine massimo di scattering
        device: Device su cui creare la trasformata scattering

    Returns:
        Oggetto Scattering2D
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scattering = Scattering2D(
        J=J,
        shape=shape,
        max_order=max_order
    ).to(device)

    return scattering


class PixelWiseClassifier(nn.Module):
    """Modello di rete neurale per la classificazione pixel-wise con o senza trasformata scattering."""

    def __init__(self, in_channels, hidden_dim=128, num_classes=4, use_scattering=True):
        """
        Inizializza il classificatore pixel-wise.

        Args:
            in_channels: Numero di canali di input (coefficienti scattering o canali immagine)
            hidden_dim: Dimensione dello strato nascosto
            num_classes: Numero di classi di output
            use_scattering: Se utilizzare la trasformata scattering
        """
        super(PixelWiseClassifier, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_scattering = use_scattering

        # Definisci l'architettura della rete
        self.bn = nn.BatchNorm2d(in_channels)

        # Rete fully convolutional per preservare la dimensione spaziale
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # Aggiungi più strati per la versione senza scattering per compensare
        if not use_scattering:
            self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(hidden_dim)
            self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(hidden_dim)

        # Strato finale di classificazione
        self.final_conv = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass del modello.

        Args:
            x: Tensore di input (coefficienti scattering o immagine diretta)

        Returns:
            Tensore di output con le probabilità di classe per ogni pixel
        """
        # Normalizzazione batch
        x = self.bn(x)

        # Rete fully convolutional
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Strati aggiuntivi per la versione senza scattering
        if not self.use_scattering:
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))

        # Strato finale
        x = self.final_conv(x)

        return x


class PixelWiseDataset(Dataset):
    """Dataset per la classificazione pixel-wise."""

    def __init__(self, images_dir, masks_dir, patch_size=32, transform=None,
                 augment=True, stride=16, class_mapping=None, lazy_loading=False,
                 max_patches_in_memory=100000, max_images=None, verbose=True,
                 metadata_cache_file=None, save_metadata=False):
        """
        Inizializza il dataset con opzioni per ottimizzare l'uso della memoria.

        Args:
            images_dir: Directory contenente le immagini
            masks_dir: Directory contenente le maschere di classe
            patch_size: Dimensione delle patch estratte
            transform: Trasformazioni da applicare
            augment: Se applicare data augmentation
            stride: Passo per l'estrazione delle patch
            class_mapping: Mapping delle classi (dict)
            lazy_loading: Se caricare le patch solo quando richieste (riduce l'uso di memoria)
            max_patches_in_memory: Numero massimo di patch da tenere in memoria (per modalità lazy)
            max_images: Numero massimo di immagini da elaborare (None = tutte)
            verbose: Se stampare messaggi informativi durante l'elaborazione
            metadata_cache_file: File da cui caricare/salvare i metadati delle patch
            save_metadata: Se salvare i metadati delle patch dopo l'estrazione
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.augment = augment
        self.lazy_loading = lazy_loading
        self.max_patches_in_memory = max_patches_in_memory
        self.max_images = max_images
        self.verbose = verbose
        self.metadata_cache_file = metadata_cache_file
        self.save_metadata = save_metadata

        # Cache per le patch caricate (usata in modalità lazy)
        self.patch_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Trova tutte le immagini (supporta sia maiuscole che minuscole)
        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")) +
                                 list(self.images_dir.glob("*.JPG")) +
                                 list(self.images_dir.glob("*.png")) +
                                 list(self.images_dir.glob("*.PNG")) +
                                 list(self.images_dir.glob("*.tif")) +
                                 list(self.images_dir.glob("*.TIF")))

        # Trova le maschere corrispondenti
        self.mask_paths = []
        for img_path in self.image_paths:
            # Prova diversi formati di nome per le maschere
            possible_mask_paths = [
                self.masks_dir / f"{img_path.stem}_mask.png",
                self.masks_dir / f"{img_path.stem}.png",
                self.masks_dir / f"{img_path.stem}.jpg",
                self.masks_dir / f"{img_path.stem}.tif"
            ]

            # Cerca la prima maschera che esiste
            mask_path = None
            for p in possible_mask_paths:
                if p.exists():
                    mask_path = p
                    break

            # Se non è stata trovata nessuna maschera, salta questa immagine
            if mask_path is None:
                if self.verbose:
                    print(f"Maschera non trovata per {img_path}")
                continue

            # Aggiungi il percorso della maschera senza stampare messaggi verbosi
            self.mask_paths.append(mask_path)

        # Filtra le immagini senza maschera
        self.image_paths = [img for i, img in enumerate(self.image_paths)
                           if i < len(self.mask_paths)]

        # Crea il mapping delle classi
        if class_mapping is None:
            self.class_mapping = {
                0: "background",
                1: "water",
                2: "vegetation",
                3: "streets",
                4: "buildings",
                5: "other"
            }
        else:
            self.class_mapping = class_mapping

        self.num_classes = len(self.class_mapping)
        self.class_to_idx = {v: k for k, v in self.class_mapping.items()}

        # Crea trasformazioni di augmentation
        if augment:
            self.aug_transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
            ])
        else:
            self.aug_transform = None

        # In modalità lazy, memorizziamo solo i metadati delle patch
        # altrimenti estraiamo tutte le patch in memoria
        if self.lazy_loading:
            self.patch_metadata = []

            # Prova a caricare i metadati da file se specificato
            if self.metadata_cache_file and os.path.exists(self.metadata_cache_file):
                if self.verbose:
                    print(f"Caricamento metadati da {self.metadata_cache_file}...")
                self._load_metadata()
            else:
                # Estrai i metadati se non è stato possibile caricarli
                self.extract_patch_metadata()

                # Salva i metadati se richiesto
                if self.save_metadata and self.metadata_cache_file:
                    self._save_metadata()

            if self.verbose:
                print(f"Modalità lazy loading attivata: {len(self.patch_metadata)} metadati di patch estratti")
        else:
            self.patches = []
            self.extract_patches()

    def extract_patch_metadata(self):
        """Estrae solo i metadati delle patch (non le patch stesse) per risparmiare memoria."""
        total_images = len(self.image_paths)
        if self.max_images is not None and self.max_images < total_images:
            total_images = self.max_images
            self.image_paths = self.image_paths[:self.max_images]
            self.mask_paths = self.mask_paths[:self.max_images]

        if self.verbose:
            print(f"Estraendo metadati delle patch da {total_images} immagini...")

        # Crea una barra di progresso principale per le immagini
        main_pbar = tqdm(total=total_images, desc="Analisi immagini", position=0)

        # Contatori per le statistiche
        total_patches = 0
        processed_images = 0
        unique_classes = set()

        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            # Carica solo la maschera per determinare le posizioni delle patch
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                main_pbar.write(f"⚠️ Impossibile caricare la maschera {mask_path}")
                continue

            # Ottieni le dimensioni dell'immagine
            img_info = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img_info is None:
                main_pbar.write(f"⚠️ Impossibile caricare l'immagine {img_path}")
                continue

            h, w = img_info.shape[:2]

            # Verifica le dimensioni
            if img_info.shape[:2] != mask.shape[:2]:
                main_pbar.write(f"ℹ️ Ridimensionamento maschera per {os.path.basename(str(img_path))}: {img_info.shape[:2]} vs {mask.shape[:2]}")
                # Ridimensiona la maschera per adattarla all'immagine
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Libera memoria
            del img_info

            # Aggiorna le classi uniche trovate
            img_unique_values = np.unique(mask)
            unique_classes.update(img_unique_values)

            # Estrai metadati delle patch
            patches_count = 0

            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    # Estrai solo la patch della maschera per verificare se contiene classi
                    mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]

                    # Verifica che la patch contenga almeno un pixel di classe
                    if np.any(mask_patch > 0):
                        # Salva solo i metadati, non la patch stessa
                        self.patch_metadata.append((str(img_path), str(mask_path), (x, y)))
                        patches_count += 1

            # Aggiorna contatori
            total_patches += patches_count
            processed_images += 1

            # Aggiorna la barra di progresso con informazioni sintetiche
            main_pbar.set_postfix({
                'metadati': total_patches,
                'img': os.path.basename(str(img_path)),
                'classi': len(unique_classes) - 1  # Sottraiamo 1 per escludere lo sfondo (0)
            })
            main_pbar.update(1)

        # Chiudi la barra di progresso
        main_pbar.close()

        # Mostra un riepilogo finale
        if self.verbose:
            print(f"✅ Estrazione metadati completata: {total_patches} patch da {processed_images} immagini")
            print(f"📊 Classi trovate: {sorted(unique_classes)}")
            print(f"💾 Memoria risparmiata: patch caricate solo quando necessario")

    def extract_patches(self):
        """Estrae patch da tutte le immagini con visualizzazione compatta."""
        total_images = len(self.image_paths)
        if self.max_images is not None and self.max_images < total_images:
            total_images = self.max_images
            self.image_paths = self.image_paths[:self.max_images]
            self.mask_paths = self.mask_paths[:self.max_images]

        if self.verbose:
            print(f"Estraendo patch da {total_images} immagini...")

        # Crea una barra di progresso principale per le immagini
        main_pbar = tqdm(total=total_images, desc="Elaborazione immagini", position=0)

        # Contatori per le statistiche
        total_patches = 0
        processed_images = 0
        unique_classes = set()

        # Gestione della memoria
        memory_warning_shown = False

        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            # Carica immagine e maschera
            img = cv2.imread(str(img_path))
            if img is None:
                main_pbar.write(f"⚠️ Impossibile caricare l'immagine {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                main_pbar.write(f"⚠️ Impossibile caricare la maschera {mask_path}")
                continue

            # Verifica le dimensioni
            if img.shape[:2] != mask.shape[:2]:
                main_pbar.write(f"ℹ️ Ridimensionamento maschera per {os.path.basename(str(img_path))}: {img.shape[:2]} vs {mask.shape[:2]}")
                # Ridimensiona la maschera per adattarla all'immagine
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Aggiorna le classi uniche trovate
            img_unique_values = np.unique(mask)
            unique_classes.update(img_unique_values)

            # Estrai patch
            h, w = img.shape[:2]
            patches_count = 0

            # Verifica se stiamo per superare il limite di memoria
            if len(self.patches) > self.max_patches_in_memory and not memory_warning_shown:
                main_pbar.write(f"⚠️ Attenzione: Numero di patch ({len(self.patches)}) sta superando il limite consigliato ({self.max_patches_in_memory})")
                main_pbar.write(f"   Considera di usare lazy_loading=True per ridurre l'uso di memoria")
                memory_warning_shown = True

            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
                    mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]

                    # Verifica che la patch contenga almeno un pixel di classe
                    if np.any(mask_patch > 0):
                        self.patches.append((img_patch, mask_patch, img_path.stem, (x, y)))
                        patches_count += 1

            # Aggiorna contatori
            total_patches += patches_count
            processed_images += 1

            # Aggiorna la barra di progresso con informazioni sintetiche
            main_pbar.set_postfix({
                'patches': total_patches,
                'img': os.path.basename(str(img_path)),
                'classi': len(unique_classes) - 1  # Sottraiamo 1 per escludere lo sfondo (0)
            })
            main_pbar.update(1)

        # Chiudi la barra di progresso
        main_pbar.close()

        # Mostra un riepilogo finale
        if self.verbose:
            print(f"✅ Estrazione completata: {total_patches} patch da {processed_images} immagini")
            print(f"📊 Classi trovate: {sorted(unique_classes)}")

    def __len__(self):
        """Restituisce il numero di patch nel dataset."""
        if self.lazy_loading:
            return len(self.patch_metadata)
        else:
            return len(self.patches)

    def _load_patch(self, img_path, mask_path, coords):
        """Carica una patch dalle coordinate specificate."""
        x, y = coords

        # Carica l'immagine e la maschera
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Impossibile caricare l'immagine {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Impossibile caricare la maschera {mask_path}")

        # Verifica le dimensioni
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Estrai la patch
        img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]

        return img_patch, mask_patch, os.path.basename(img_path).split('.')[0], coords

    def _manage_cache(self):
        """Gestisce la cache delle patch per il lazy loading."""
        # Se la cache è piena, rimuovi gli elementi meno recenti
        if len(self.patch_cache) >= self.max_patches_in_memory:
            # Rimuovi il 10% delle patch meno recenti
            num_to_remove = max(1, int(self.max_patches_in_memory * 0.1))
            keys_to_remove = list(self.patch_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self.patch_cache[key]

    def _save_metadata(self):
        """Salva i metadati delle patch su file."""
        if not self.metadata_cache_file:
            return

        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(self.metadata_cache_file), exist_ok=True)

        # Prepara i dati da salvare
        metadata = {
            'patch_metadata': self.patch_metadata,
            'patch_size': self.patch_size,
            'stride': self.stride,
            'class_mapping': self.class_mapping,
            'images_dir': str(self.images_dir),
            'masks_dir': str(self.masks_dir)
        }

        # Salva i metadati
        try:
            with open(self.metadata_cache_file, 'wb') as f:
                torch.save(metadata, f)
            if self.verbose:
                print(f"✅ Metadati salvati in {self.metadata_cache_file}")
        except Exception as e:
            if self.verbose:
                print(f"❌ Errore nel salvataggio dei metadati: {e}")

    def _load_metadata(self):
        """Carica i metadati delle patch da file."""
        if not self.metadata_cache_file or not os.path.exists(self.metadata_cache_file):
            return False

        try:
            # Carica i metadati
            metadata = torch.load(self.metadata_cache_file)

            # Verifica la compatibilità
            if (metadata['patch_size'] != self.patch_size or
                metadata['stride'] != self.stride or
                str(metadata['images_dir']) != str(self.images_dir) or
                str(metadata['masks_dir']) != str(self.masks_dir)):
                if self.verbose:
                    print("⚠️ I metadati salvati non sono compatibili con i parametri attuali.")
                    print(f"   Salvati: patch_size={metadata['patch_size']}, stride={metadata['stride']}")
                    print(f"   Attuali: patch_size={self.patch_size}, stride={self.stride}")
                return False

            # Carica i metadati
            self.patch_metadata = metadata['patch_metadata']

            # Aggiorna il mapping delle classi se non è stato specificato
            if self.class_mapping is None and 'class_mapping' in metadata:
                self.class_mapping = metadata['class_mapping']
                self.num_classes = len(self.class_mapping)
                self.class_to_idx = {v: k for k, v in self.class_mapping.items()}

            if self.verbose:
                print(f"✅ Caricati {len(self.patch_metadata)} metadati di patch da {self.metadata_cache_file}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"❌ Errore nel caricamento dei metadati: {e}")
            return False

    def __getitem__(self, idx):
        """Ottiene una patch dal dataset, caricandola on-demand se in modalità lazy."""
        if self.lazy_loading:
            # Verifica se la patch è già in cache
            if idx in self.patch_cache:
                self.cache_hits += 1
                img_patch, mask_patch, img_name, coords = self.patch_cache[idx]
            else:
                # Carica la patch dai metadati
                self.cache_misses += 1
                img_path, mask_path, coords = self.patch_metadata[idx]
                img_patch, mask_patch, img_name, coords = self._load_patch(img_path, mask_path, coords)

                # Gestisci la cache
                self._manage_cache()

                # Aggiungi alla cache
                self.patch_cache[idx] = (img_patch, mask_patch, img_name, coords)

                # Stampa statistiche della cache ogni 1000 miss
                if self.verbose and self.cache_misses % 1000 == 0:
                    total = self.cache_hits + self.cache_misses
                    hit_rate = (self.cache_hits / total) * 100 if total > 0 else 0
                    print(f"Cache: {len(self.patch_cache)}/{self.max_patches_in_memory} patch, "
                          f"Hit rate: {hit_rate:.1f}% ({self.cache_hits}/{total})")
        else:
            # Modalità standard: carica dalla lista di patch pre-estratte
            img_patch, mask_patch, img_name, coords = self.patches[idx]

        # Applica augmentation se richiesto
        if self.augment and self.aug_transform:
            augmented = self.aug_transform(image=img_patch, mask=mask_patch)
            img_patch = augmented['image']
            mask_patch = augmented['mask']

        # Applica trasformazioni
        if self.transform:
            img_patch = self.transform(img_patch)
        else:
            # Converti a tensore
            img_patch = torch.from_numpy(img_patch.transpose(2, 0, 1)).float() / 255.0

        # Converti maschera a tensore
        mask_patch = torch.from_numpy(mask_patch).long()

        return {
            'image': img_patch,
            'mask': mask_patch,
            'name': img_name,
            'coords': coords
        }


def create_pixel_classifier(config):
    """
    Crea un modello di classificatore pixel-wise.

    Args:
        config: Oggetto di configurazione con parametri del modello

    Returns:
        Modello PixelWiseClassifier e trasformata Scattering2D (o None se use_scattering=False)
    """
    # Determina se usare la trasformata scattering
    use_scattering = getattr(config, 'use_scattering', True)

    if use_scattering:
        # Crea trasformata scattering
        scattering = create_scattering_transform(
            J=config.J,
            shape=config.shape,
            max_order=config.scattering_order,
            device=config.device
        )

        # Calcola il numero di coefficienti scattering
        dummy_input = torch.randn(1, config.num_channels, *config.shape).to(config.device)
        scattering_output = scattering(dummy_input)

        # Gestisci la dimensionalità dell'output
        if scattering_output.dim() == 5:
            # Rimuovi l'ultima dimensione o prendi solo la parte reale
            if scattering_output.shape[-1] == 1:
                scattering_output = scattering_output.squeeze(-1)
            else:
                # Prendi solo la parte reale (prima componente)
                scattering_output = scattering_output[..., 0]

        in_channels = scattering_output.shape[1]
        print(f"Forma dell'output della trasformata scattering: {scattering_output.shape}")
    else:
        # Senza scattering, usa direttamente i canali dell'immagine
        scattering = None
        in_channels = config.num_channels

    # Crea modello classificatore
    model = PixelWiseClassifier(
        in_channels=in_channels,
        hidden_dim=128,
        num_classes=config.num_classes,
        use_scattering=use_scattering
    ).to(config.device)

    return model, scattering


def train_pixel_classifier(
    train_dataset,
    val_dataset=None,
    model_path='pixel_classifier.pth',
    batch_size=16,
    num_epochs=50,
    learning_rate=1e-4,
    device=None,
    scattering=None,
    model=None,
    config=None,
    use_scattering=True,
    resume=False,
    checkpoint_interval=1,
    disable_cudnn=False,
    use_amp=True
):
    """
    Addestra un classificatore pixel-wise con supporto per interruzione e ripresa.

    Args:
        train_dataset: Dataset di training
        val_dataset: Dataset di validazione
        model_path: Percorso dove salvare il modello
        batch_size: Dimensione del batch
        num_epochs: Numero di epoche
        learning_rate: Learning rate
        device: Device per l'addestramento
        scattering: Trasformata scattering
        model: Modello pre-inizializzato
        config: Configurazione
        use_scattering: Se utilizzare la trasformata scattering
        resume: Se riprendere l'addestramento da un checkpoint esistente
        checkpoint_interval: Intervallo (in epoche) per salvare i checkpoint
        disable_cudnn: Se disabilitare completamente cuDNN (utile in caso di errori)
        use_amp: Se utilizzare la precisione mista automatica (AMP) per accelerare l'addestramento

    Returns:
        Dizionario con la storia dell'addestramento
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Utilizzo device: {device}")

    # Configurazione per stabilità su GPU
    if device.type == 'cuda':
        if disable_cudnn:
            # Disabilita completamente cuDNN
            torch.backends.cudnn.enabled = False
            print("GPU: cuDNN completamente disabilitato")
        else:
            # Disabilita il benchmark di cuDNN per maggiore stabilità
            torch.backends.cudnn.benchmark = False
            # Imposta la modalità deterministica per maggiore stabilità
            torch.backends.cudnn.deterministic = True
            print("GPU: configurazione ottimizzata per stabilità")

        # Informazioni sulla precisione mista
        if use_amp:
            print("GPU: precisione mista (AMP) attivata per accelerare l'addestramento")
        else:
            print("GPU: precisione mista (AMP) disattivata per maggiore stabilità")

    # Determina il numero di worker ottimale
    # Se lazy_loading è attivo, usa più worker per caricare le patch in parallelo
    # altrimenti usa meno worker per ridurre l'overhead
    if hasattr(train_dataset, 'lazy_loading') and train_dataset.lazy_loading:
        num_workers = min(8, os.cpu_count() or 4)  # Usa più worker per lazy loading
    else:
        num_workers = min(4, os.cpu_count() or 2)  # Usa meno worker per dataset in memoria

    # Adatta il batch size in base al device
    effective_batch_size = batch_size
    if device.type == 'cuda':
        # Riduci il batch size se necessario per evitare problemi di memoria
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if gpu_mem < 8:  # Meno di 8GB di VRAM
            effective_batch_size = min(4, batch_size)
            print(f"GPU con memoria limitata ({gpu_mem:.1f}GB): batch size ridotto a {effective_batch_size}")

    # Crea data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0  # Mantieni i worker attivi tra le epoche
    )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=effective_batch_size,  # Usa lo stesso batch size ridotto
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0  # Mantieni i worker attivi tra le epoche
        )
    else:
        val_loader = None

    # Inizializza variabili per il training
    start_epoch = 0
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Crea directory per i checkpoint se non esiste
    checkpoint_dir = os.path.dirname(model_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Percorso per il checkpoint temporaneo
    temp_checkpoint_path = os.path.join(
        os.path.dirname(model_path),
        f"{os.path.splitext(os.path.basename(model_path))[0]}_temp.pth"
    )

    # Ripresa dell'addestramento da un checkpoint esistente
    if resume and os.path.exists(model_path):
        print(f"Ripresa dell'addestramento dal checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Carica lo stato del modello
        if model is None:
            # Determina i parametri del modello dal checkpoint
            use_scattering = checkpoint.get('use_scattering', use_scattering)
            num_classes = len(checkpoint.get('class_mapping', {}))
            in_channels = next(iter(checkpoint['model_state_dict'].items()))[1].shape[0]

            # Crea un nuovo modello con i parametri corretti
            model = PixelWiseClassifier(
                in_channels=in_channels,
                hidden_dim=128,
                num_classes=num_classes,
                use_scattering=use_scattering
            ).to(device)

            # Carica i pesi
            model.load_state_dict(checkpoint['model_state_dict'])

            # Crea la trasformata scattering se necessario
            if use_scattering and scattering is None:
                if config is not None:
                    scattering = create_scattering_transform(
                        J=config.J,
                        shape=(train_dataset.patch_size, train_dataset.patch_size),
                        device=device
                    )
        else:
            # Carica i pesi nel modello esistente
            model.load_state_dict(checkpoint['model_state_dict'])

        # Carica lo stato dell'ottimizzatore
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Carica lo stato dello scheduler
        try:
            # Prova con verbose=True (versioni più recenti di PyTorch)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        except TypeError:
            # Fallback per versioni meno recenti di PyTorch
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            print("Nota: Scheduler creato senza parametro verbose (non supportato in questa versione di PyTorch)")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Carica altre informazioni
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint.get('history', history)

        print(f"Addestramento ripreso dall'epoca {start_epoch}")
    else:
        # Crea modello e scattering se non forniti
        if model is None:
            if config is None:
                raise ValueError("È necessario fornire config se model non è fornito")

            # Aggiungi il parametro use_scattering alla configurazione
            if not hasattr(config, 'use_scattering'):
                config.use_scattering = use_scattering

            model, scattering = create_pixel_classifier(config)

        # Definisci loss function e ottimizzatore
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Crea scheduler con gestione della compatibilità
        try:
            # Prova con verbose=True (versioni più recenti di PyTorch)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        except TypeError:
            # Fallback per versioni meno recenti di PyTorch
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            print("Nota: Scheduler creato senza parametro verbose (non supportato in questa versione di PyTorch)")

    # Definisci la loss function (se non è già stata definita)
    criterion = nn.CrossEntropyLoss()

    # Crea una barra di progresso principale per le epoche
    epochs_pbar = tqdm(total=num_epochs, initial=start_epoch, desc="Addestramento", position=0)

    # Gestione dell'interruzione
    try:
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Processa i batch di training senza barra di progresso separata
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                # Forward pass
                if scattering is not None and model.use_scattering:
                    # Con trasformata scattering
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda' and use_amp):
                        # Applica la trasformata scattering
                        scattering_coeffs = scattering(images)

                        # Gestisci la dimensionalità dell'output
                        # La trasformata scattering può produrre un tensore 5D [B, C, H, W, 1] o [B, C, H, W, 2]
                        # Dobbiamo convertirlo in un tensore 4D [B, C, H, W]
                        if scattering_coeffs.dim() == 5:
                            # Rimuovi l'ultima dimensione o prendi solo la parte reale
                            if scattering_coeffs.shape[-1] == 1:
                                scattering_coeffs = scattering_coeffs.squeeze(-1)  # Rimuovi l'ultima dimensione
                            else:
                                # Prendi solo la parte reale (prima componente)
                                scattering_coeffs = scattering_coeffs[..., 0]

                        # Stampa le dimensioni per debug
                        if batch_idx == 0:
                            print(f"Dimensioni scattering_coeffs: {scattering_coeffs.shape}")
                            print(f"Dimensioni masks: {masks.shape}")

                        # Passa i coefficienti al modello
                        outputs = model(scattering_coeffs)

                        # Ridimensiona l'output se necessario per adattarlo alla maschera
                        if outputs.shape[-2:] != masks.shape[-2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                            if batch_idx == 0:
                                print(f"Output ridimensionato a: {outputs.shape}")

                        # Calcola loss
                        loss = criterion(outputs, masks)
                else:
                    # Senza trasformata scattering
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda' and use_amp):
                        # Stampa le dimensioni per debug
                        if batch_idx == 0:
                            print(f"Dimensioni images: {images.shape}")
                            print(f"Dimensioni masks: {masks.shape}")

                        outputs = model(images)

                        # Ridimensiona l'output se necessario per adattarlo alla maschera
                        if outputs.shape[-2:] != masks.shape[-2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                            if batch_idx == 0:
                                print(f"Output ridimensionato a: {outputs.shape}")

                        # Calcola loss
                        loss = criterion(outputs, masks)

                # Backward pass e ottimizzazione
                optimizer.zero_grad(set_to_none=True)  # Più efficiente di zero_grad()
                loss.backward()
                optimizer.step()

                # Libera memoria
                if device.type == 'cuda':
                    # Libera memoria cache CUDA non utilizzata
                    if batch_idx % 10 == 0:  # Non farlo ad ogni batch per evitare overhead
                        torch.cuda.empty_cache()

                # Statistiche
                batch_loss = loss.item() * images.size(0)
                train_loss += batch_loss
                _, predicted = outputs.max(1)
                batch_total = masks.numel()
                train_total += batch_total
                batch_correct = predicted.eq(masks).sum().item()
                train_correct += batch_correct

                # Aggiorna la barra di progresso principale con lo stato corrente
                if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:  # Aggiorna ogni 10 batch o all'ultimo batch
                    current_acc = 100. * train_correct / train_total if train_total > 0 else 0
                    epochs_pbar.set_postfix({
                        'epoca': f"{epoch+1}/{num_epochs}",
                        'fase': "train",
                        'batch': f"{batch_idx+1}/{len(train_loader)}",
                        'loss': f"{batch_loss/images.size(0):.4f}",
                        'acc': f"{current_acc:.2f}%"
                    })

            # Calcola metriche di training
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = 100. * train_correct / train_total if train_total > 0 else 0

            # Validazione
            val_loss = 0.0
            val_acc = 0.0

            if val_loader:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        images = batch['image'].to(device)
                        masks = batch['mask'].to(device)

                        # Forward pass
                        if scattering is not None and model.use_scattering:
                            # Con trasformata scattering
                            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda' and use_amp):
                                # Applica la trasformata scattering
                                scattering_coeffs = scattering(images)

                                # Gestisci la dimensionalità dell'output
                                # La trasformata scattering può produrre un tensore 5D [B, C, H, W, 1] o [B, C, H, W, 2]
                                # Dobbiamo convertirlo in un tensore 4D [B, C, H, W]
                                if scattering_coeffs.dim() == 5:
                                    # Rimuovi l'ultima dimensione o prendi solo la parte reale
                                    if scattering_coeffs.shape[-1] == 1:
                                        scattering_coeffs = scattering_coeffs.squeeze(-1)  # Rimuovi l'ultima dimensione
                                    else:
                                        # Prendi solo la parte reale (prima componente)
                                        scattering_coeffs = scattering_coeffs[..., 0]

                                # Stampa le dimensioni per debug (solo per il primo batch)
                                if batch_idx == 0 and epoch == start_epoch:
                                    print(f"[Val] Dimensioni scattering_coeffs: {scattering_coeffs.shape}")
                                    print(f"[Val] Dimensioni masks: {masks.shape}")

                                # Passa i coefficienti al modello
                                outputs = model(scattering_coeffs)

                                # Ridimensiona l'output se necessario per adattarlo alla maschera
                                if outputs.shape[-2:] != masks.shape[-2:]:
                                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                                    if batch_idx == 0 and epoch == start_epoch:
                                        print(f"[Val] Output ridimensionato a: {outputs.shape}")

                                # Calcola loss
                                loss = criterion(outputs, masks)
                        else:
                            # Senza trasformata scattering
                            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda' and use_amp):
                                # Stampa le dimensioni per debug (solo per il primo batch)
                                if batch_idx == 0 and epoch == start_epoch:
                                    print(f"[Val] Dimensioni images: {images.shape}")
                                    print(f"[Val] Dimensioni masks: {masks.shape}")

                                outputs = model(images)

                                # Ridimensiona l'output se necessario per adattarlo alla maschera
                                if outputs.shape[-2:] != masks.shape[-2:]:
                                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                                    if batch_idx == 0 and epoch == start_epoch:
                                        print(f"[Val] Output ridimensionato a: {outputs.shape}")

                                # Calcola loss
                                loss = criterion(outputs, masks)

                        # Libera memoria
                        if device.type == 'cuda' and batch_idx % 10 == 0:
                            torch.cuda.empty_cache()

                        # Statistiche
                        batch_loss = loss.item() * images.size(0)
                        val_loss += batch_loss
                        _, predicted = outputs.max(1)
                        batch_total = masks.numel()
                        val_total += batch_total
                        batch_correct = predicted.eq(masks).sum().item()
                        val_correct += batch_correct

                        # Aggiorna la barra di progresso principale con lo stato corrente
                        if batch_idx % 10 == 0 or batch_idx == len(val_loader) - 1:  # Aggiorna ogni 10 batch o all'ultimo batch
                            current_acc = 100. * val_correct / val_total if val_total > 0 else 0
                            epochs_pbar.set_postfix({
                                'epoca': f"{epoch+1}/{num_epochs}",
                                'fase': "val",
                                'batch': f"{batch_idx+1}/{len(val_loader)}",
                                'loss': f"{batch_loss/images.size(0):.4f}",
                                'acc': f"{current_acc:.2f}%"
                            })

                # Calcola metriche di validazione
                val_loss = val_loss / len(val_loader.dataset)
                val_acc = 100. * val_correct / val_total if val_total > 0 else 0

                # Aggiorna learning rate
                scheduler.step(val_loss)

            # Aggiorna storia
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if val_loader:
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

            # Aggiorna la barra di progresso principale con il riepilogo dell'epoca
            epochs_pbar.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'train_acc': f"{train_acc:.2f}%",
                'val_loss': f"{val_loss:.4f}" if val_loader else "N/A",
                'val_acc': f"{val_acc:.2f}%" if val_loader else "N/A"
            })
            epochs_pbar.update(1)

            # Prepara il checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'history': history,
                'class_mapping': getattr(train_dataset, 'class_mapping', 
                                       getattr(train_dataset.dataset, 'class_mapping', None)),
                'use_scattering': model.use_scattering
            }

            # Salva checkpoint temporaneo
            temp_checkpoint_path = os.path.join(
                os.path.dirname(model_path) if os.path.dirname(model_path) else '.',
                f"{os.path.splitext(os.path.basename(model_path))[0]}_temp.pth"
            )
            torch.save(checkpoint, temp_checkpoint_path)

            # Salva checkpoint a intervalli regolari
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    os.path.dirname(model_path) if os.path.dirname(model_path) else '.',
                    f"{os.path.splitext(os.path.basename(model_path))[0]}_epoch_{epoch+1}.pth"
                )
                torch.save(checkpoint, checkpoint_path)
                print(f"Salvato checkpoint all'epoca {epoch+1}: {checkpoint_path}")

            # Salva il modello se è il migliore
            if val_loader and val_loss < best_val_loss:
                best_val_loss = val_loss
                # Salva il modello migliore
                torch.save(checkpoint, model_path)
                print(f"Salvato nuovo miglior modello con val_loss: {val_loss:.4f}")

            # Stampa metriche
            if val_loader:
                print(f"Epoca {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                print(f"Epoca {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    except KeyboardInterrupt:
        print("\nAddestramento interrotto dall'utente!")
        print(f"Ultimo checkpoint salvato in: {temp_checkpoint_path}")
        print("Puoi riprendere l'addestramento usando il parametro resume=True")

    finally:
        # Chiudi la barra di progresso principale
        epochs_pbar.close()

    # Visualizza curve di apprendimento
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if val_loader:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    if val_loader:
        plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.tight_layout()

    # Salva il grafico
    plot_path = os.path.splitext(model_path)[0] + '_learning_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training completato. Modello salvato in: {model_path}")
    print(f"Curve di apprendimento salvate in: {plot_path}")

    return history


def load_pixel_classifier(model_path, device=None, num_classes=None):
    """
    Carica un modello di classificatore pixel-wise da un file.

    Args:
        model_path: Percorso del file del modello
        device: Device su cui caricare il modello
        num_classes: Numero di classi nel modello

    Returns:
        Modello PixelWiseClassifier caricato e mapping delle classi
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        return None, None

    # Ottieni mapping delle classi
    class_mapping = checkpoint.get('class_mapping', None)

    # Determina il numero di classi
    if num_classes is None:
        num_classes = len(class_mapping) if class_mapping else 6
    
    # Determina se il modello usa la trasformata scattering
    use_scattering = checkpoint.get('use_scattering', True)

    # Determina il tipo di modello
    model_type = checkpoint.get('model_type', 'PixelWiseClassifier')

    # Determina i parametri del modello
    in_channels = 81  # Valore predefinito per scattering
    hidden_dim = checkpoint.get('hidden_dim', 128)

    # Prova a determinare il numero di canali di input dai pesi
    if 'model_state_dict' in checkpoint:
        for key, value in checkpoint['model_state_dict'].items():
            if 'bn.weight' in key:
                in_channels = value.size(0)
                print(f"Rilevato numero di canali di input: {in_channels}")
                break

    # Crea modello in base al tipo
    if model_type == 'PixelWiseClassifier' or model_type == 'default':
        model = PixelWiseClassifier(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_scattering=use_scattering
        ).to(device)
    else:
        # Supporto per altri tipi di modelli in futuro
        print(f"Tipo di modello non supportato: {model_type}")
        return None, None

    # Carica pesi
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("Errore: Impossibile trovare i pesi del modello nel checkpoint")
            return None, None
    except Exception as e:
        print(f"Errore nel caricamento dei pesi: {e}")
        # Prova a caricare con strict=False
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Pesi caricati con strict=False")
        else:
            return None, None

    model.eval()
    return model, class_mapping
