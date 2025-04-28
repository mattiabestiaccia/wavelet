"""
Funzioni di utilità per il modulo di classificazione pixel-wise.
"""

import os
import random
import torch
import numpy as np

class Config:
    """Classe di configurazione per i modelli di classificazione pixel-wise."""
    
    def __init__(self, num_channels=3, num_classes=5, scattering_order=2, J=2, shape=(32, 32),
                 batch_size=16, epochs=50, learning_rate=1e-4, use_scattering=True,
                 device=None):
        """
        Inizializza la configurazione.
        
        Args:
            num_channels: Numero di canali di input
            num_classes: Numero di classi
            scattering_order: Ordine massimo della trasformata scattering
            J: Parametro J per la trasformata scattering
            shape: Forma delle immagini di input
            batch_size: Dimensione del batch
            epochs: Numero di epoche
            learning_rate: Learning rate
            use_scattering: Se utilizzare la trasformata scattering
            device: Device per l'addestramento
        """
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.scattering_order = scattering_order
        self.J = J
        self.shape = shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_scattering = use_scattering
        
        # Imposta il device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
    
    def print_summary(self):
        """Stampa un riepilogo della configurazione."""
        print("\n" + "="*80)
        print(" "*30 + "CONFIGURAZIONE" + " "*30)
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Canali di input: {self.num_channels}")
        print(f"Numero di classi: {self.num_classes}")
        print(f"Parametri scattering: J={self.J}, order={self.scattering_order}")
        print(f"Uso scattering: {self.use_scattering}")
        print(f"Dimensione batch: {self.batch_size}")
        print(f"Epoche: {self.epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print("="*80 + "\n")

def set_seed(seed):
    """
    Imposta il seed per la riproducibilità.
    
    Args:
        seed: Seed per la generazione di numeri casuali
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed impostato a {seed} per la riproducibilità")

def save_model(model, scattering_params, optimizer, epoch, loss, class_mapping, save_path):
    """
    Salva il modello e i parametri di addestramento.
    
    Args:
        model: Modello da salvare
        scattering_params: Parametri della trasformata scattering
        optimizer: Ottimizzatore
        epoch: Epoca corrente
        loss: Loss del modello
        class_mapping: Mappatura delle classi
        save_path: Percorso dove salvare il modello
    """
    # Crea la directory se non esiste
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepara il checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'class_mapping': class_mapping,
        'scattering_params': scattering_params,
        'use_scattering': model.use_scattering
    }
    
    # Salva il checkpoint
    torch.save(checkpoint, save_path)
    print(f"Modello salvato in {save_path}")

def load_model(model_path, device=None):
    """
    Carica un modello salvato.
    
    Args:
        model_path: Percorso del modello da caricare
        device: Device su cui caricare il modello
        
    Returns:
        Modello caricato e parametri
    """
    # Imposta il device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica il checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Estrai i parametri
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    class_mapping = checkpoint.get('class_mapping', {})
    scattering_params = checkpoint.get('scattering_params', {})
    use_scattering = checkpoint.get('use_scattering', True)
    
    return {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'epoch': epoch,
        'loss': loss,
        'class_mapping': class_mapping,
        'scattering_params': scattering_params,
        'use_scattering': use_scattering
    }
