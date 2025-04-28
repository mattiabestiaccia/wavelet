"""
Modulo per la visualizzazione dei risultati di classificazione.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter

def plot_training_metrics(epochs, train_accuracies, test_accuracies, train_losses, test_losses, save_path=None):
    """
    Visualizza le metriche di addestramento.
    
    Args:
        epochs: Numero di epoche
        train_accuracies: Accuratezze di addestramento
        test_accuracies: Accuratezze di test
        train_losses: Loss di addestramento
        test_losses: Loss di test
        save_path: Percorso dove salvare la visualizzazione
    """
    # Crea la figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Asse x
    x = range(1, epochs + 1)
    
    # Grafico dell'accuratezza
    ax1.plot(x, train_accuracies, label='Train', marker='o')
    ax1.plot(x, test_accuracies, label='Test', marker='s')
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Accuratezza (%)')
    ax1.set_title('Accuratezza di addestramento e test')
    ax1.legend()
    ax1.grid(True)
    
    # Grafico della loss
    ax2.plot(x, train_losses, label='Train', marker='o')
    ax2.plot(x, test_losses, label='Test', marker='s')
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss di addestramento e test')
    ax2.legend()
    ax2.grid(True)
    
    # Aggiusta il layout
    plt.tight_layout()
    
    # Salva la figura se richiesto
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metriche di addestramento salvate in {save_path}")
    
    # Mostra la figura
    plt.show()

def plot_class_distribution(dataset, title="Distribuzione delle classi", save_path=None):
    """
    Visualizza la distribuzione delle classi nel dataset.
    
    Args:
        dataset: Dataset da visualizzare
        title: Titolo del grafico
        save_path: Percorso dove salvare la visualizzazione
    """
    # Conta le classi
    class_counts = Counter([label for _, label in dataset.samples])
    
    # Ordina le classi
    classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    
    # Ottieni i nomi delle classi
    class_names = [dataset.classes[cls] for cls in classes]
    
    # Crea la figura
    plt.figure(figsize=(12, 6))
    
    # Crea il grafico a barre
    bars = plt.bar(class_names, counts)
    
    # Aggiungi le etichette
    plt.xlabel('Classe')
    plt.ylabel('Numero di campioni')
    plt.title(title)
    
    # Aggiungi i valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}', ha='center', va='bottom')
    
    # Ruota le etichette se ce ne sono molte
    if len(class_names) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Aggiusta il layout
    plt.tight_layout()
    
    # Salva la figura se richiesto
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribuzione delle classi salvata in {save_path}")
    
    # Mostra la figura
    plt.show()

def visualize_classification_results(results, save_path=None):
    """
    Visualizza i risultati della classificazione per tile.
    
    Args:
        results: Risultati della classificazione
        save_path: Percorso dove salvare la visualizzazione
    """
    # Estrai i dati
    image = results['cropped_image']
    label_matrix = results['label_matrix']
    confidence_matrix = results['confidence_matrix']
    tile_size = results['tile_size']
    class_names = results['class_names']
    
    # Conta le classi
    class_counts = results['class_counts']
    total_tiles = results['total_tiles']
    classified_tiles = sum(class_counts.values())
    
    # Crea i colori per le classi
    num_classes = len(class_names)
    colors = list(mcolors.TABLEAU_COLORS.values())[:num_classes]
    
    # Crea la figura
    plt.figure(figsize=(15, 12))
    
    # Mostra l'immagine
    plt.imshow(image)
    
    # Disegna i tile colorati
    ax = plt.gca()
    for i in range(label_matrix.shape[0]):
        for j in range(label_matrix.shape[1]):
            label = label_matrix[i, j]
            if label >= 0 and label < num_classes:
                color = colors[label]
                rect = plt.Rectangle(
                    (j * tile_size, i * tile_size),
                    tile_size, tile_size,
                    linewidth=1,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.3  # Semi-trasparente
                )
                ax.add_patch(rect)
    
    # Crea la legenda
    legend_patches = []
    for class_idx, class_name in enumerate(class_names):
        count = class_counts.get(class_idx, 0)
        percentage = 100 * count / total_tiles
        patch = plt.Rectangle((0, 0), 1, 1,
                               linewidth=1,
                               edgecolor=colors[class_idx],
                               facecolor=colors[class_idx],
                               label=f"{class_name}: {count} tile ({percentage:.1f}%)")
        legend_patches.append(patch)
    
    # Aggiungi la legenda
    plt.legend(handles=legend_patches,
               loc='center left',
               bbox_to_anchor=(1, 0.5),
               fontsize=10,
               framealpha=0.8)
    
    # Calcola i tile non classificati
    unclassified = total_tiles - classified_tiles
    unclassified_percentage = 100 * unclassified / total_tiles
    
    # Aggiungi il titolo
    plt.title(f'Classificazione per tile - {total_tiles} tile ({label_matrix.shape[0]}×{label_matrix.shape[1]})\n' +
              f'Classificati: {classified_tiles} ({100*classified_tiles/total_tiles:.1f}%), ' +
              f'Non classificati: {unclassified} ({unclassified_percentage:.1f}%)')
    
    # Aggiusta il layout
    plt.tight_layout()
    plt.axis('off')
    
    # Salva la figura se richiesto
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Risultati della classificazione salvati in {save_path}")
    
    # Mostra la figura
    plt.show()
