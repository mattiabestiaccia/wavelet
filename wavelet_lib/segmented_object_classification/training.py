"""
Modulo per l'addestramento di modelli di classificazione di oggetti segmentati.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import random
import cv2
from sklearn.model_selection import train_test_split

from wavelet_lib.segmented_object_classification.models import (
    SegmentedObjectClassifier,
    create_scattering_transform
)


class SegmentedObjectDataset(Dataset):
    """Dataset per oggetti segmentati organizzati in cartelle per classe."""
    
    def __init__(self, root_dir, transform=None, balance=True, max_samples_per_class=None):
        """
        Inizializza il dataset.
        
        Args:
            root_dir: Directory principale del dataset
            transform: Trasformazioni da applicare alle immagini
            balance: Se bilanciare le classi
            max_samples_per_class: Numero massimo di campioni per classe
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Raccogli immagini per classe
        class_images = {}
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            images = []
            for fname in os.listdir(cls_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in allowed_extensions:
                    filepath = cls_dir / fname
                    images.append((str(filepath), self.class_to_idx[cls]))
            class_images[cls] = images
        
        # Bilanciamento opzionale
        if balance:
            min_samples = min(len(images) for images in class_images.values())
            if max_samples_per_class is not None:
                min_samples = min(min_samples, max_samples_per_class)
            
            for cls, images in class_images.items():
                if len(images) > min_samples:
                    # Campionamento casuale per bilanciare
                    selected_images = random.sample(images, min_samples)
                    self.samples.extend(selected_images)
                else:
                    self.samples.extend(images)
        else:
            # Usa tutte le immagini senza bilanciamento
            for cls, images in class_images.items():
                if max_samples_per_class is not None and len(images) > max_samples_per_class:
                    # Limita il numero di campioni per classe
                    selected_images = random.sample(images, max_samples_per_class)
                    self.samples.extend(selected_images)
                else:
                    self.samples.extend(images)
    
    def __len__(self):
        """Restituisce il numero di campioni nel dataset."""
        return len(self.samples)
    
    def __getitem__(self, index):
        """Ottiene un campione dal dataset."""
        filepath, label = self.samples[index]
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_class_names(self):
        """Restituisce la lista dei nomi delle classi."""
        return self.classes
    
    def get_class_distribution(self):
        """Restituisce un dizionario con il numero di campioni per classe."""
        class_counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        return class_counts


def get_transforms(input_size=(32, 32), augment=True):
    """
    Ottiene le trasformazioni per il dataset.
    
    Args:
        input_size: Dimensione di input per il modello
        augment: Se applicare data augmentation
        
    Returns:
        Trasformazioni per training e validazione
    """
    # Trasformazioni di base
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Trasformazioni per il training (con augmentation)
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            normalize
        ])
    
    # Trasformazioni per la validazione (senza augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def train_segmented_object_classifier(
    train_dir,
    val_dir=None,
    val_split=0.2,
    model_path='segmented_object_classifier.pth',
    input_size=(32, 32),
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    weight_decay=1e-4,
    J=2,
    scattering_order=2,
    device=None,
    num_workers=4,
    balance_classes=True,
    max_samples_per_class=None,
    augment=True
):
    """
    Addestra un classificatore per oggetti segmentati.
    
    Args:
        train_dir: Directory contenente i dati di training
        val_dir: Directory contenente i dati di validazione (opzionale)
        val_split: Frazione dei dati da usare per la validazione se val_dir non è specificato
        model_path: Percorso dove salvare il modello
        input_size: Dimensione di input per il modello
        batch_size: Dimensione del batch
        num_epochs: Numero di epoche di training
        learning_rate: Learning rate per l'ottimizzatore
        weight_decay: Weight decay per l'ottimizzatore
        J: Numero di scale per la trasformata scattering
        scattering_order: Ordine della trasformata scattering
        device: Device per il training
        num_workers: Numero di worker per il data loading
        balance_classes: Se bilanciare le classi
        max_samples_per_class: Numero massimo di campioni per classe
        augment: Se applicare data augmentation
        
    Returns:
        Dizionario con la storia del training e il modello addestrato
    """
    # Determina il device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Utilizzo device: {device}")
    
    # Ottieni le trasformazioni
    train_transform, val_transform = get_transforms(input_size, augment)
    
    # Carica il dataset di training
    train_dataset = SegmentedObjectDataset(
        root_dir=train_dir,
        transform=train_transform,
        balance=balance_classes,
        max_samples_per_class=max_samples_per_class
    )
    
    # Ottieni le classi
    classes = train_dataset.get_class_names()
    num_classes = len(classes)
    class_to_idx = train_dataset.class_to_idx
    
    print(f"Classi trovate: {classes}")
    print(f"Distribuzione delle classi nel training set: {train_dataset.get_class_distribution()}")
    
    # Prepara dataset di validazione
    if val_dir:
        # Usa una directory separata per la validazione
        val_dataset = SegmentedObjectDataset(
            root_dir=val_dir,
            transform=val_transform,
            balance=False  # Non bilanciare il validation set
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        # Dividi il dataset di training
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)),
            test_size=val_split,
            stratify=[label for _, label in train_dataset.samples],
            random_state=42
        )
        
        # Crea subset
        from torch.utils.data import Subset
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        # Crea data loader
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    
    # Crea trasformata scattering
    scattering = create_scattering_transform(
        J=J,
        shape=input_size,
        max_order=scattering_order,
        device=device
    )
    
    # Calcola il numero di coefficienti scattering
    dummy_input = torch.randn(1, 3, *input_size).to(device)
    scattering_output = scattering(dummy_input)
    scattering_coeffs = scattering_output.shape[1]
    
    print(f"Numero di coefficienti scattering: {scattering_coeffs}")
    
    # Crea modello
    model = SegmentedObjectClassifier(
        in_channels=scattering_coeffs,
        classifier_type='cnn',
        num_classes=num_classes
    ).to(device)
    
    # Definisci loss function e ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
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
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            scattering_coeffs = scattering(inputs)
            outputs = model(scattering_coeffs)
            loss = criterion(outputs, labels)
            
            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistiche
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Aggiorna progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * train_correct / train_total:.2f}%"
            })
        
        # Calcola metriche di training
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total
        
        # Validazione
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                scattering_coeffs = scattering(inputs)
                outputs = model(scattering_coeffs)
                loss = criterion(outputs, labels)
                
                # Statistiche
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
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
                'class_to_idx': class_to_idx,
                'scattering_coeffs': scattering_coeffs,
                'input_size': input_size,
                'J': J,
                'scattering_order': scattering_order,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, model_path)
            print(f"Salvato nuovo miglior modello con val_loss: {val_loss:.4f}")
        
        # Aggiorna storia
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Stampa metriche
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Visualizza curve di apprendimento
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
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
    
    print(f"Training completato. Miglior val_loss: {best_val_loss:.4f}")
    print(f"Modello salvato in: {model_path}")
    print(f"Curve di apprendimento salvate in: {plot_path}")
    
    return {
        'history': history,
        'model': model,
        'class_to_idx': class_to_idx,
        'classes': classes
    }
