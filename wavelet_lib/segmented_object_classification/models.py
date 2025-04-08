"""
Modulo di modelli per la classificazione di oggetti segmentati nella Wavelet Scattering Transform Library.
Contiene modelli di reti neurali per la classificazione con trasformate scattering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D
import numpy as np
import os
from PIL import Image
import cv2

class SegmentedObjectClassifier(nn.Module):
    """Modello di rete neurale per la classificazione di oggetti segmentati con trasformata scattering."""
    
    def __init__(self, in_channels, classifier_type='cnn', num_classes=7):
        """
        Inizializza il classificatore di oggetti segmentati.
        
        Args:
            in_channels: Numero di canali di input (coefficienti scattering)
            classifier_type: Tipo di classificatore ('cnn', 'mlp', o 'linear')
            num_classes: Numero di classi di output
        """
        super(SegmentedObjectClassifier, self).__init__()
        self.in_channels = in_channels
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.build()

    def build(self):
        """Costruisce l'architettura del classificatore in base al tipo di classificatore."""
        self.K = self.in_channels
        self.bn = nn.BatchNorm2d(self.K)
        
        if self.classifier_type == 'cnn':
            # Architettura CNN per classificazione
            self.features = nn.Sequential(
                nn.Conv2d(self.K, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 2 * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, self.num_classes),
            )
        elif self.classifier_type == 'mlp':
            # Architettura MLP per classificazione
            self.features = None
            self.classifier = nn.Sequential(
                nn.Linear(self.K * 8 * 8, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, self.num_classes),
            )
        elif self.classifier_type == 'linear':
            # Classificatore lineare semplice
            self.features = None
            self.classifier = nn.Linear(self.K * 8 * 8, self.num_classes)
        else:
            raise ValueError(f"Tipo di classificatore non supportato: {self.classifier_type}")

    def forward(self, x):
        """
        Forward pass del modello.
        
        Args:
            x: Tensore di input, può essere coefficienti scattering o tensore appiattito
            
        Returns:
            Tensore di output con le probabilità di classe
        """
        # Gestisce diversi formati di input
        if len(x.shape) == 2:
            # Input già appiattito
            flattened = x
            batch_size = x.size(0)
            try:
                x = flattened.view(batch_size, self.K, 8, 8)
            except RuntimeError:
                raise ValueError(f"Forma dell'input {x.shape} con {flattened.shape[1]} elementi non può essere ridimensionata a [batch, {self.K}, 8, 8]")
                
        x = self.bn(x)
        if self.features:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


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


def create_segmented_object_classifier(config):
    """
    Crea un modello di classificatore per oggetti segmentati.
    
    Args:
        config: Oggetto di configurazione con parametri del modello
        
    Returns:
        Modello SegmentedObjectClassifier e trasformata Scattering2D
    """
    # Crea trasformata scattering
    scattering = create_scattering_transform(
        J=config.J,
        shape=config.shape,
        max_order=config.scattering_order,
        device=config.device
    )
    
    # Crea modello classificatore
    model = SegmentedObjectClassifier(
        in_channels=config.scattering_coeffs,
        classifier_type='cnn',
        num_classes=config.num_classes
    ).to(config.device)
    
    return model, scattering


def load_segmented_object_classifier(model_path, num_classes=7, in_channels=12, device=None):
    """
    Carica un modello di classificatore per oggetti segmentati da un file.
    
    Args:
        model_path: Percorso del file del modello
        num_classes: Numero di classi
        in_channels: Numero di canali di input
        device: Device su cui caricare il modello
        
    Returns:
        Modello SegmentedObjectClassifier caricato e mapping delle classi
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Crea modello
    model = SegmentedObjectClassifier(
        in_channels=in_channels,
        classifier_type='cnn',
        num_classes=num_classes
    ).to(device)
    
    # Carica pesi
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("Errore: Impossibile trovare i pesi del modello nel checkpoint")
        return None, None
    
    # Ottieni mapping delle classi
    class_to_idx = checkpoint.get('class_to_idx', None)
    
    model.eval()
    
    return model, class_to_idx
