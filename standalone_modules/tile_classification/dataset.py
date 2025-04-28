"""
Modulo per la gestione dei dataset di classificazione.
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from collections import Counter

class BalancedDataset(Dataset):
    """Dataset bilanciato per la classificazione di immagini."""
    
    def __init__(self, root_dir, transform=None, balance=False, max_samples_per_class=None):
        """
        Inizializza il dataset.
        
        Args:
            root_dir: Directory radice del dataset
            transform: Trasformazioni da applicare alle immagini
            balance: Se bilanciare le classi
            max_samples_per_class: Numero massimo di campioni per classe
        """
        self.root_dir = root_dir
        self.transform = transform
        self.balance = balance
        self.max_samples_per_class = max_samples_per_class
        
        # Trova tutte le classi (cartelle)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        
        # Crea la mappatura delle classi
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Raccogli tutti i file di immagine
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Trova tutte le immagini nella directory della classe
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            
            # Aggiungi i percorsi delle immagini e le etichette
            for img_file in image_files:
                self.samples.append((os.path.join(class_dir, img_file), class_idx))
        
        # Bilancia il dataset se richiesto
        if balance:
            self._balance_dataset()
        
        # Limita il numero di campioni per classe se richiesto
        if max_samples_per_class is not None:
            self._limit_samples_per_class()
        
        # Stampa le statistiche del dataset
        self._print_stats()
    
    def _balance_dataset(self):
        """Bilancia il dataset in modo che tutte le classi abbiano lo stesso numero di campioni."""
        # Conta i campioni per classe
        class_counts = Counter([label for _, label in self.samples])
        min_count = min(class_counts.values())
        
        # Raggruppa i campioni per classe
        samples_by_class = {}
        for path, label in self.samples:
            if label not in samples_by_class:
                samples_by_class[label] = []
            samples_by_class[label].append((path, label))
        
        # Bilancia il dataset
        balanced_samples = []
        for label, samples in samples_by_class.items():
            # Seleziona casualmente min_count campioni
            selected_samples = random.sample(samples, min_count)
            balanced_samples.extend(selected_samples)
        
        # Aggiorna i campioni
        self.samples = balanced_samples
        
        print(f"Dataset bilanciato: {min_count} campioni per classe")
    
    def _limit_samples_per_class(self):
        """Limita il numero di campioni per classe."""
        # Raggruppa i campioni per classe
        samples_by_class = {}
        for path, label in self.samples:
            if label not in samples_by_class:
                samples_by_class[label] = []
            samples_by_class[label].append((path, label))
        
        # Limita il numero di campioni per classe
        limited_samples = []
        for label, samples in samples_by_class.items():
            # Seleziona casualmente max_samples_per_class campioni
            num_samples = min(len(samples), self.max_samples_per_class)
            selected_samples = random.sample(samples, num_samples)
            limited_samples.extend(selected_samples)
        
        # Aggiorna i campioni
        self.samples = limited_samples
        
        print(f"Dataset limitato: massimo {self.max_samples_per_class} campioni per classe")
    
    def _print_stats(self):
        """Stampa le statistiche del dataset."""
        # Conta i campioni per classe
        class_counts = Counter([label for _, label in self.samples])
        
        print(f"\nStatistiche del dataset:")
        print(f"  • Numero totale di campioni: {len(self.samples)}")
        print(f"  • Numero di classi: {len(self.classes)}")
        print("  • Campioni per classe:")
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            count = class_counts[class_idx]
            print(f"    - {class_name}: {count} campioni")
    
    def __len__(self):
        """Restituisce il numero di campioni nel dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Restituisce un campione dal dataset.
        
        Args:
            idx: Indice del campione
            
        Returns:
            Dizionario con l'immagine e l'etichetta
        """
        # Ottieni il percorso dell'immagine e l'etichetta
        img_path, label = self.samples[idx]
        
        # Carica l'immagine
        image = Image.open(img_path).convert('RGB')
        
        # Applica le trasformazioni se disponibili
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'label': label, 'path': img_path}

def get_default_transform(target_size=(32, 32), dataset_root=None):
    """
    Restituisce la trasformazione predefinita per le immagini.
    
    Args:
        target_size: Dimensione target per le immagini
        dataset_root: Directory radice del dataset (per calcolare le statistiche)
        
    Returns:
        Trasformazione predefinita
    """
    # Trasformazione predefinita
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform

def create_data_loaders(dataset, test_size=0.2, batch_size=32, num_workers=4):
    """
    Crea i data loader per l'addestramento e il test.
    
    Args:
        dataset: Dataset da dividere
        test_size: Frazione del dataset da usare per il test
        batch_size: Dimensione del batch
        num_workers: Numero di worker per il caricamento dei dati
        
    Returns:
        Data loader per l'addestramento e il test
    """
    # Calcola le dimensioni dei set di addestramento e test
    test_size = int(test_size * len(dataset))
    train_size = len(dataset) - test_size
    
    # Dividi il dataset
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Crea i data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nData loader creati:")
    print(f"  • Set di addestramento: {train_size} campioni")
    print(f"  • Set di test: {test_size} campioni")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Numero di batch di addestramento: {len(train_loader)}")
    print(f"  • Numero di batch di test: {len(test_loader)}")
    
    return train_loader, test_loader
