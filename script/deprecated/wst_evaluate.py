#!/usr/bin/env python3
"""
Script per la valutazione completa di un modello su un dataset.

Questo script esegue una valutazione dettagliata del modello fornendo metriche complete.
"""

import os
import sys
import torch
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Aggiungi la directory principale al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa i moduli wavelet_lib
from wavelet_lib.base import Config, load_model
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders
from wavelet_lib.models import create_scattering_transform, ScatteringClassifier

def parse_args():
    """
    Analizza gli argomenti dalla riga di comando.
    
    Returns:
        args: Namespace contenente gli argomenti analizzati
    """
    parser = argparse.ArgumentParser(description='Valutazione modello Wavelet Scattering Transform')
    
    # Parametri del modello
    parser.add_argument('--model-path', type=str, required=True, help='Percorso al file del modello')
    parser.add_argument('--dataset', type=str, required=True, help='Percorso al dataset')
    
    # Parametri di valutazione
    parser.add_argument('--batch-size', type=int, default=64, help='Dimensione del batch')
    parser.add_argument('--balance', action='store_true', help='Bilancia le classi nel dataset')
    
    # Parametri generali
    parser.add_argument('--device', type=str, default=None, help='Device per l\'inferenza (cuda o cpu)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory per salvare i risultati')
    parser.add_argument('--num-workers', type=int, default=4, help='Numero di worker per i dataloader')
    
    return parser.parse_args()

def load_model_checkpoint(model_path, device):
    """
    Carica un modello da un checkpoint.
    
    Args:
        model_path: Percorso al file del modello
        device: Device per il modello
    
    Returns:
        model, scattering, class_names, in_channels
    """
    print(f"Caricamento del modello: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Estrazione delle informazioni sulla classe
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        class_names = list(class_to_idx.keys())
        num_classes = len(class_names)
    else:
        print("Avviso: class_to_idx non trovato nel checkpoint.")
        
        # Prova a determinare il numero di classi dalla directory del dataset
        dataset_dir = os.path.dirname(model_path)
        if "model_output_" in dataset_dir:
            parts = os.path.basename(dataset_dir).split("_")
            for part in parts:
                if part.isdigit():
                    num_classes = int(part)
                    break
            else:
                num_classes = 4  # Default fallback
        else:
            num_classes = 4  # Default fallback
        
        class_names = [f"Classe {i}" for i in range(num_classes)]
    
    print(f"Numero di classi rilevato: {num_classes}")
    print(f"Nomi delle classi: {class_names}")
    
    # Crea il modello e la trasformata scattering
    scattering = create_scattering_transform(
        J=2,
        shape=(32, 32),
        max_order=2,
        device=device
    )
    
    # Determina il numero di canali di input
    in_channels = 243  # Default per l'architettura standard
    
    # Crea e carica il modello
    model = ScatteringClassifier(in_channels=in_channels, num_classes=num_classes).to(device)
    
    # Carica i pesi del modello
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("Avviso: chiave di stato del modello non standard. Tentativo di adattamento...")
        
        # Ottieni le chiavi disponibili nel checkpoint
        state_dict_keys = [k for k in checkpoint.keys() if k.startswith('model') or 'state' in k]
        if state_dict_keys:
            print(f"Chiavi disponibili: {state_dict_keys}")
            for key in state_dict_keys:
                try:
                    model.load_state_dict(checkpoint[key])
                    print(f"Modello caricato con successo dalla chiave: {key}")
                    break
                except:
                    continue
        else:
            print("Errore: Impossibile trovare uno stato del modello valido nel checkpoint")
            sys.exit(1)
    
    model.eval()
    print("Modello caricato con successo.")
    
    return model, scattering, class_names, in_channels

def evaluate_model(model, scattering, test_loader, device, class_names):
    """
    Valuta il modello su un dataset di test.
    
    Args:
        model: Modello addestrato
        scattering: Trasformata scattering
        test_loader: DataLoader per il test
        device: Device per l'inferenza
        class_names: Nomi delle classi
        
    Returns:
        Dictionary con le metriche di valutazione
    """
    model.eval()
    
    # Preparazione per le metriche
    all_predictions = []
    all_targets = []
    
    # Valutazione
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Valutazione"):
            data, target = data.to(device), target.to(device)
            
            # Applica la trasformata scattering
            scattering_coeffs = scattering(data)
            
            # Reshape per adattare al modello
            batch_size = data.size(0)
            scattering_coeffs = scattering_coeffs.view(batch_size, -1, 8, 8)
            
            # Forward pass
            outputs = model(scattering_coeffs)
            
            # Calcola le predizioni
            _, predictions = torch.max(outputs, 1)
            
            # Raccogli i risultati
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calcola le metriche di valutazione
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    class_report = classification_report(all_targets, all_predictions, 
                                         target_names=class_names, 
                                         output_dict=True)
    
    # Accuracy
    accuracy = sum(1 for pred, target in zip(all_predictions, all_targets) if pred == target) / len(all_targets)
    
    return {
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'accuracy': accuracy,
        'predictions': all_predictions,
        'targets': all_targets
    }

def plot_confusion_matrix(conf_matrix, class_names, save_path=None):
    """
    Visualizza la matrice di confusione.
    
    Args:
        conf_matrix: Matrice di confusione
        class_names: Nomi delle classi
        save_path: Percorso dove salvare la visualizzazione
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predizione')
    plt.ylabel('Valore reale')
    plt.title('Matrice di Confusione')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrice di confusione salvata in: {save_path}")
    
    plt.show()

def plot_metrics(metrics, class_names, save_path=None):
    """
    Visualizza le metriche di valutazione.
    
    Args:
        metrics: Dictionary con le metriche
        class_names: Nomi delle classi
        save_path: Percorso dove salvare la visualizzazione
    """
    report = metrics['classification_report']
    
    # Estrai precision, recall, f1-score per ogni classe
    classes_data = {class_name: report[class_name] for class_name in class_names}
    
    # Crea il grafico
    plt.figure(figsize=(12, 8))
    
    # Imposta il numero di gruppi
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.25
    
    # Estrai le metriche
    precision = [classes_data[name]['precision'] for name in class_names]
    recall = [classes_data[name]['recall'] for name in class_names]
    f1 = [classes_data[name]['f1-score'] for name in class_names]
    
    # Grafico a barre
    plt.bar(x - width, precision, width, label='Precisione')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')
    
    plt.ylabel('Punteggio')
    plt.title('Metriche di valutazione per classe')
    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Aggiungi punteggi sopra ogni barra
    for i, v in enumerate(precision):
        plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(recall):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(f1):
        plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metriche di valutazione salvate in: {save_path}")
    
    plt.show()

def main():
    """
    Funzione principale per la valutazione del modello.
    """
    # Analizza gli argomenti dalla riga di comando
    args = parse_args()
    
    # Configura il device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Configura la directory di output
    if args.output_dir is None:
        model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "evaluation", model_basename)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Valutazione modello Wavelet Scattering Transform")
    print(f"{'='*80}")
    print(f"Modello: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Carica il modello
    model, scattering, class_names, in_channels = load_model_checkpoint(args.model_path, device)
    
    # Prepara il dataset
    transform = get_default_transform(target_size=(32, 32))
    dataset = BalancedDataset(args.dataset, transform=transform, balance=args.balance)
    
    # Crea i dataloader
    _, test_loader = create_data_loaders(
        dataset,
        test_size=1.0,  # Usa tutto il dataset per la valutazione
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Valuta il modello
    print("\nAvvio della valutazione del modello...")
    metrics = evaluate_model(model, scattering, test_loader, device, class_names)
    
    # Stampa i risultati
    print("\nRisultati della valutazione:")
    print(f"Accuratezza: {metrics['accuracy']:.4f}")
    
    print("\nReport di classificazione:")
    print(classification_report(metrics['targets'], metrics['predictions'], target_names=class_names))
    
    # Visualizza la matrice di confusione
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        class_names, 
        save_path=os.path.join(args.output_dir, "confusion_matrix.png")
    )
    
    # Visualizza le metriche di valutazione
    plot_metrics(
        metrics, 
        class_names, 
        save_path=os.path.join(args.output_dir, "evaluation_metrics.png")
    )
    
    print(f"\nValutazione completata! Risultati salvati in {args.output_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()