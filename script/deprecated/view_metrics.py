#!/usr/bin/env python3
"""
Script per visualizzare le metriche di addestramento dai modelli salvati.

Questo script carica e visualizza le metriche di addestramento salvate nei modelli.
"""

import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Aggiungi la directory principale al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa i moduli wavelet_lib
from wavelet_lib.visualization import plot_training_metrics

def parse_args():
    """
    Analizza gli argomenti dalla riga di comando.
    
    Returns:
        args: Namespace contenente gli argomenti analizzati
    """
    parser = argparse.ArgumentParser(description='Visualizzazione delle metriche di addestramento')
    
    # Parametri principali
    parser.add_argument('model_dir', type=str, help='Directory del modello')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory per salvare i risultati')
    
    return parser.parse_args()

def find_metrics_file(model_dir):
    """
    Trova il file che contiene le metriche di addestramento.
    
    Args:
        model_dir: Directory del modello
        
    Returns:
        Path al file con le metriche
    """
    # Cerca in ordine di priorità
    candidate_files = [
        os.path.join(model_dir, 'final_model.pth'),
        os.path.join(model_dir, 'checkpoint.pth'),
        os.path.join(model_dir, 'best_model.pth')
    ]
    
    for file_path in candidate_files:
        if os.path.exists(file_path):
            try:
                checkpoint = torch.load(file_path, map_location='cpu')
                if isinstance(checkpoint, dict) and ('metrics' in checkpoint or 
                                                    ('train_losses' in checkpoint and 'test_losses' in checkpoint)):
                    print(f"Metriche trovate in: {file_path}")
                    return file_path, checkpoint
            except Exception as e:
                print(f"Errore nel caricamento di {file_path}: {e}")
    
    return None, None

def extract_metrics(checkpoint):
    """
    Estrae le metriche di addestramento dal checkpoint.
    
    Args:
        checkpoint: Dizionario del checkpoint
        
    Returns:
        Tuple con (train_accuracies, test_accuracies, train_losses, test_losses)
    """
    # Gestisce diversi formati di checkpoint
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        train_accuracies = metrics.get('train_acc', [])
        test_accuracies = metrics.get('test_acc', [])
        train_losses = metrics.get('train_loss', [])
        test_losses = metrics.get('test_loss', [])
    elif 'train_accuracies' in checkpoint and 'test_accuracies' in checkpoint:
        train_accuracies = checkpoint['train_accuracies']
        test_accuracies = checkpoint['test_accuracies']
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
    else:
        # Formato legacy con diverse chiavi
        train_accuracies = checkpoint.get('train_accuracies', [])
        test_accuracies = checkpoint.get('test_accuracies', [])
        train_losses = checkpoint.get('train_losses', [])
        test_losses = checkpoint.get('test_losses', [])
    
    return train_accuracies, test_accuracies, train_losses, test_losses

def main():
    """
    Funzione principale che coordina la visualizzazione delle metriche.
    """
    # Analizza gli argomenti dalla riga di comando
    args = parse_args()
    
    # Normalizza il percorso della directory del modello
    model_dir = args.model_dir
    if not os.path.isabs(model_dir):
        # Se il percorso è relativo, assumiamo che sia relativo alla directory models/
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        model_dir = os.path.join(models_dir, model_dir)
    
    if not os.path.exists(model_dir):
        print(f"Errore: La directory {model_dir} non esiste.")
        return
    
    print(f"\n{'='*80}")
    print(f"Visualizzazione delle metriche di addestramento")
    print(f"{'='*80}")
    print(f"Directory del modello: {model_dir}")
    
    # Trova e carica il file con le metriche
    metrics_file, checkpoint = find_metrics_file(model_dir)
    if metrics_file is None:
        print("Errore: Nessun file con metriche trovato nella directory specificata.")
        return
    
    # Estrai le metriche
    train_accuracies, test_accuracies, train_losses, test_losses = extract_metrics(checkpoint)
    
    if not train_accuracies or not test_accuracies:
        print("Errore: Metriche di addestramento mancanti nel checkpoint.")
        return
    
    # Determina il numero di epoche
    epochs = len(train_accuracies)
    print(f"Numero di epoche: {epochs}")
    
    # Visualizza le statistiche
    max_train_acc = max(train_accuracies)
    max_test_acc = max(test_accuracies)
    min_train_loss = min(train_losses)
    min_test_loss = min(test_losses)
    
    print(f"\nStatistiche:")
    print(f"Accuratezza massima di addestramento: {max_train_acc:.2f}%")
    print(f"Accuratezza massima di test: {max_test_acc:.2f}%")
    print(f"Loss minima di addestramento: {min_train_loss:.4f}")
    print(f"Loss minima di test: {min_test_loss:.4f}")
    
    # Configura la directory di output
    if args.output_dir is None:
        output_dir = model_dir
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # Visualizza le metriche
    save_path = os.path.join(output_dir, "training_metrics.png")
    plot_training_metrics(
        epochs,
        train_accuracies,
        test_accuracies,
        train_losses,
        test_losses,
        save_path
    )
    
    print(f"\nVisualization completata e salvata in: {save_path}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()