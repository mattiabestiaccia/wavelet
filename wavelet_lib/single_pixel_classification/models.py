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
from PIL import Image
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
                 augment=True, stride=16, class_mapping=None):
        """
        Inizializza il dataset.

        Args:
            images_dir: Directory contenente le immagini
            masks_dir: Directory contenente le maschere di classe
            patch_size: Dimensione delle patch estratte
            transform: Trasformazioni da applicare
            augment: Se applicare data augmentation
            stride: Passo per l'estrazione delle patch
            class_mapping: Mapping delle classi (dict)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.augment = augment

        # Trova tutte le immagini
        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")) +
                                 list(self.images_dir.glob("*.png")) +
                                 list(self.images_dir.glob("*.tif")))

        # Trova le maschere corrispondenti
        self.mask_paths = []
        for img_path in self.image_paths:
            mask_path = self.masks_dir / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                mask_path = self.masks_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                print(f"Maschera non trovata per {img_path}")
                continue
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
                4: "buildings"
            }
        else:
            self.class_mapping = class_mapping

        self.num_classes = len(self.class_mapping)
        self.class_to_idx = {v: k for k, v in self.class_mapping.items()}

        # Crea trasformazioni di augmentation
        if augment:
            self.aug_transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
            ])
        else:
            self.aug_transform = None

        # Estrai patch da tutte le immagini
        self.patches = []
        self.extract_patches()

    def extract_patches(self):
        """Estrae patch da tutte le immagini."""
        print("Estraendo patch dalle immagini...")
        for img_path, mask_path in tqdm(zip(self.image_paths, self.mask_paths),
                                       total=len(self.image_paths)):
            # Carica immagine e maschera
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Impossibile caricare {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Estrai patch
            h, w = img.shape[:2]
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
                    mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]

                    # Verifica che la patch contenga almeno un pixel di classe
                    if np.any(mask_patch > 0):
                        self.patches.append((img_patch, mask_patch, img_path.stem, (x, y)))

    def __len__(self):
        """Restituisce il numero di patch nel dataset."""
        return len(self.patches)

    def __getitem__(self, idx):
        """Ottiene una patch dal dataset."""
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
        in_channels = scattering_output.shape[1]
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
    use_scattering=True
):
    """
    Addestra un classificatore pixel-wise.

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

    Returns:
        Dizionario con la storia dell'addestramento
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crea data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        val_loader = None

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Inizializza variabili per il training
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # Forward pass
            if scattering is not None and model.use_scattering:
                # Con trasformata scattering
                scattering_coeffs = scattering(images)
                outputs = model(scattering_coeffs)
            else:
                # Senza trasformata scattering
                outputs = model(images)

            # Calcola loss
            loss = criterion(outputs, masks)

            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistiche
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += masks.numel()
            train_correct += predicted.eq(masks).sum().item()

            # Aggiorna progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * train_correct / train_total:.2f}%"
            })

        # Calcola metriche di training
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total

        # Validazione
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for batch in val_pbar:
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)

                    # Forward pass
                    if scattering is not None and model.use_scattering:
                        # Con trasformata scattering
                        scattering_coeffs = scattering(images)
                        outputs = model(scattering_coeffs)
                    else:
                        # Senza trasformata scattering
                        outputs = model(images)

                    # Calcola loss
                    loss = criterion(outputs, masks)

                    # Statistiche
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += masks.numel()
                    val_correct += predicted.eq(masks).sum().item()

                    # Aggiorna progress bar
                    val_pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{100. * val_correct / val_total:.2f}%"
                    })

            # Calcola metriche di validazione
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = 100. * val_correct / val_total

            # Aggiorna learning rate
            scheduler.step(val_loss)

            # Salva il modello se è il migliore
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Crea directory se non esiste
                os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
                # Salva il modello
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_mapping': train_dataset.class_mapping,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'use_scattering': model.use_scattering
                }, model_path)
                print(f"Salvato nuovo miglior modello con val_loss: {val_loss:.4f}")

            # Aggiorna storia
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

        # Aggiorna storia
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Stampa metriche
        if val_loader:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

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


def load_pixel_classifier(model_path, device=None):
    """
    Carica un modello di classificatore pixel-wise da un file.

    Args:
        model_path: Percorso del file del modello
        device: Device su cui caricare il modello

    Returns:
        Modello PixelWiseClassifier caricato e mapping delle classi
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Ottieni mapping delle classi
    class_mapping = checkpoint.get('class_mapping', None)

    # Determina il numero di classi
    num_classes = len(class_mapping) if class_mapping else 5

    # Determina se il modello usa la trasformata scattering
    use_scattering = checkpoint.get('use_scattering', True)

    # Crea modello
    model = PixelWiseClassifier(
        in_channels=81,  # Valore predefinito, verrà sovrascritto dai pesi
        hidden_dim=128,
        num_classes=num_classes,
        use_scattering=use_scattering
    ).to(device)

    # Carica pesi
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Errore: Impossibile trovare i pesi del modello nel checkpoint")
        return None, None

    model.eval()
    return model, class_mapping
