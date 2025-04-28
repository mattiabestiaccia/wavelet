"""
Modelli di classificazione pixel-wise per Wavelet Scattering Transform.

Questo modulo contiene le definizioni dei modelli neurali per la classificazione
pixel-wise di immagini utilizzando la trasformata scattering wavelet.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from kymatio.torch import Scattering2D
from tqdm import tqdm
import numpy as np

from .utils import save_model

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
    
    def __init__(self, in_channels, hidden_dim=128, num_classes=5, use_scattering=True):
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

def create_pixel_classifier(config):
    """
    Crea un modello di classificazione pixel-wise.
    
    Args:
        config: Oggetto di configurazione con i parametri del modello
        
    Returns:
        Modello PixelWiseClassifier e trasformata Scattering2D
    """
    # Crea la trasformata scattering se richiesta
    if config.use_scattering:
        scattering = create_scattering_transform(
            J=config.J,
            shape=config.shape,
            max_order=config.scattering_order,
            device=config.device
        )
        
        # Calcola il numero di canali di input
        dummy_input = torch.randn(1, config.num_channels, *config.shape).to(config.device)
        scattering_output = scattering(dummy_input)
        
        # Gestisci la dimensionalità dell'output
        if scattering_output.dim() == 5:
            if scattering_output.shape[-1] == 1:
                scattering_output = scattering_output.squeeze(-1)
            else:
                scattering_output = scattering_output[..., 0]
        
        in_channels = scattering_output.shape[1]
        print(f"Trasformata scattering: {in_channels} canali di input")
    else:
        scattering = None
        in_channels = config.num_channels
        print("Nessuna trasformata scattering: utilizzo diretto dell'immagine")
    
    # Crea il modello di classificazione
    model = PixelWiseClassifier(
        in_channels=in_channels,
        hidden_dim=128,
        num_classes=config.num_classes,
        use_scattering=config.use_scattering
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
    use_amp=True,
    num_workers=4
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
        num_workers: Numero di worker per il data loading
        
    Returns:
        Dizionario con la storia dell'addestramento
    """
    # Imposta il device
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
    
    # Crea il modello se non fornito
    if model is None:
        if config is None:
            raise ValueError("È necessario fornire un modello o una configurazione")
        
        model, scattering = create_pixel_classifier(config)
    
    # Crea l'ottimizzatore
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Crea lo scheduler per ridurre il learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Crea il criterio di loss
    criterion = nn.CrossEntropyLoss()
    
    # Variabili per il checkpoint
    start_epoch = 0
    best_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    # Carica il checkpoint se richiesto
    if resume and os.path.exists(model_path):
        print(f"Caricamento checkpoint da {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        
        # Carica la storia se presente
        if 'history' in checkpoint:
            history = checkpoint['history']
        
        print(f"Ripresa dell'addestramento dall'epoca {start_epoch}")
    
    # Percorso per il checkpoint temporaneo
    temp_checkpoint_path = os.path.splitext(model_path)[0] + '_temp.pth'
    
    # Parametri della trasformata scattering
    scattering_params = {}
    if scattering is not None:
        scattering_params = {
            'J': scattering.J,
            'shape': scattering.shape,
            'max_order': scattering.max_order
        }
    
    # Estrai il class mapping dal dataset
    class_mapping = {}
    if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'class_mapping'):
        class_mapping = train_dataset.dataset.class_mapping
    
    # Loop di addestramento
    print(f"\nInizio addestramento per {num_epochs} epoche (da {start_epoch} a {start_epoch + num_epochs - 1})")
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        # Crea una barra di progresso per l'addestramento
        train_pbar = tqdm(train_loader, desc=f"Addestramento: Epoca {epoch+1}/{start_epoch + num_epochs}")
        
        for batch_idx, batch in enumerate(train_pbar):
            # Estrai immagini e maschere
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Azzera i gradienti
            optimizer.zero_grad()
            
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
            loss.backward()
            optimizer.step()
            
            # Aggiorna la loss totale
            train_loss += loss.item()
            
            # Aggiorna la barra di progresso
            train_pbar.set_postfix({'loss': train_loss / (batch_idx + 1)})
        
        # Calcola la loss media
        train_loss /= len(train_loader)
        
        # Validazione
        val_loss = 0.0
        if val_loader:
            model.eval()
            
            # Crea una barra di progresso per la validazione
            val_pbar = tqdm(val_loader, desc=f"Validazione: Epoca {epoch+1}/{start_epoch + num_epochs}")
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_pbar):
                    # Estrai immagini e maschere
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
                    
                    # Aggiorna la loss totale
                    val_loss += loss.item()
                    
                    # Aggiorna la barra di progresso
                    val_pbar.set_postfix({'loss': val_loss / (batch_idx + 1)})
            
            # Calcola la loss media
            val_loss /= len(val_loader)
            
            # Aggiorna lo scheduler
            scheduler.step(val_loss)
        
        # Aggiorna la storia
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss if val_loader else train_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Stampa le metriche
        print(f"Epoca {epoch+1}/{start_epoch + num_epochs} - Train Loss: {train_loss:.4f}" + 
              (f" - Val Loss: {val_loss:.4f}" if val_loader else ""))
        
        # Salva il checkpoint temporaneo ad ogni epoca
        save_model(
            model,
            scattering_params,
            optimizer,
            epoch,
            val_loss if val_loader else train_loss,
            class_mapping,
            temp_checkpoint_path
        )
        
        # Salva il miglior modello
        current_loss = val_loss if val_loader else train_loss
        if current_loss < best_loss:
            best_loss = current_loss
            save_model(
                model,
                scattering_params,
                optimizer,
                epoch,
                best_loss,
                class_mapping,
                model_path
            )
            print(f"Nuovo miglior modello salvato con loss: {best_loss:.4f}")
        
        # Salva checkpoint periodici
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.splitext(model_path)[0] + f'_epoch_{epoch+1}.pth'
            save_model(
                model,
                scattering_params,
                optimizer,
                epoch,
                current_loss,
                class_mapping,
                checkpoint_path
            )
            print(f"Checkpoint salvato all'epoca {epoch+1}")
    
    # Restituisci la storia dell'addestramento
    return history
