"""
Modulo per la visualizzazione dei risultati di classificazione pixel-wise.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
import torch
import torch.nn.functional as F

def visualize_results(image, mask, prediction, class_mapping, save_path=None):
    """
    Visualizza i risultati della classificazione pixel-wise.
    
    Args:
        image: Immagine originale
        mask: Maschera di verità
        prediction: Predizione del modello
        class_mapping: Mappatura delle classi
        save_path: Percorso dove salvare la visualizzazione
    """
    # Crea una mappa di colori per le classi
    colors = [
        [0, 0, 0],        # Classe 0: Nero (sfondo)
        [0, 0, 255],      # Classe 1: Blu
        [0, 255, 0],      # Classe 2: Verde
        [255, 0, 0],      # Classe 3: Rosso
        [255, 255, 0],    # Classe 4: Giallo
        [0, 255, 255],    # Classe 5: Ciano
        [255, 0, 255],    # Classe 6: Magenta
        [128, 0, 0],      # Classe 7: Marrone
        [0, 128, 0],      # Classe 8: Verde scuro
        [0, 0, 128],      # Classe 9: Blu scuro
    ]
    
    # Converti tensori in numpy array se necessario
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if isinstance(prediction, torch.Tensor):
        if prediction.dim() > 2:  # Se è un tensore di logits [C, H, W]
            prediction = torch.argmax(prediction, dim=0).cpu().numpy()
        else:
            prediction = prediction.cpu().numpy()
    
    # Crea immagini colorate per maschera e predizione
    mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    pred_colored = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    
    for cls in range(min(len(colors), len(class_mapping))):
        mask_colored[mask == cls] = colors[cls]
        pred_colored[prediction == cls] = colors[cls]
    
    # Crea una legenda
    legend_items = []
    for cls, name in class_mapping.items():
        if cls < len(colors):
            color = np.array(colors[cls]) / 255.0
            legend_items.append((color, name))
    
    # Visualizza i risultati
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Immagine originale")
    axes[0].axis('off')
    
    axes[1].imshow(mask_colored)
    axes[1].set_title("Maschera di verità")
    axes[1].axis('off')
    
    axes[2].imshow(pred_colored)
    axes[2].set_title("Predizione")
    axes[2].axis('off')
    
    # Aggiungi la legenda
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color, _ in legend_items]
    legend_labels = [name for _, name in legend_items]
    fig.legend(legend_patches, legend_labels, loc='lower center', ncol=len(legend_items))
    
    plt.tight_layout()
    
    # Salva o mostra i risultati
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Risultati salvati in {save_path}")
    
    plt.show()

def plot_training_metrics(history, save_path=None):
    """
    Visualizza le metriche di addestramento.
    
    Args:
        history: Dizionario con le metriche di addestramento
        save_path: Percorso dove salvare la visualizzazione
    """
    # Estrai le metriche
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    learning_rates = history['learning_rates']
    
    # Crea la figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Asse x
    epochs = range(1, len(train_loss) + 1)
    
    # Grafico della loss
    ax1.plot(epochs, train_loss, 'b-', label='Training')
    ax1.plot(epochs, val_loss, 'r-', label='Validation')
    ax1.set_title('Loss di addestramento e validazione')
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Grafico del learning rate
    ax2.plot(epochs, learning_rates, 'g-')
    ax2.set_title('Learning rate')
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Learning rate')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Salva o mostra i risultati
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metriche di addestramento salvate in {save_path}")
    
    plt.show()

def plot_class_distribution(dataset, title="Distribuzione delle classi", save_path=None):
    """
    Visualizza la distribuzione delle classi nel dataset.
    
    Args:
        dataset: Dataset da visualizzare
        title: Titolo del grafico
        save_path: Percorso dove salvare la visualizzazione
    """
    # Estrai le maschere dal dataset
    class_counts = Counter()
    
    # Verifica se il dataset è un subset
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'class_mapping'):
        class_mapping = dataset.dataset.class_mapping
    elif hasattr(dataset, 'class_mapping'):
        class_mapping = dataset.class_mapping
    else:
        class_mapping = {i: f"Classe {i}" for i in range(10)}
    
    # Campiona un sottoinsieme di patch per efficienza
    num_samples = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        batch = dataset[idx]
        mask = batch['mask']
        
        # Conta le classi
        for cls in range(len(class_mapping)):
            count = (mask == cls).sum().item()
            if count > 0:
                class_counts[cls] += count
    
    # Normalizza i conteggi
    total = sum(class_counts.values())
    class_percentages = {cls: count / total * 100 for cls, count in class_counts.items()}
    
    # Ordina le classi
    classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    percentages = [class_percentages[cls] for cls in classes]
    
    # Ottieni i nomi delle classi
    class_names = [class_mapping.get(cls, f"Classe {cls}") for cls in classes]
    
    # Crea la figura
    plt.figure(figsize=(12, 6))
    
    # Crea il grafico a barre
    bars = plt.bar(class_names, percentages)
    
    # Aggiungi le etichette
    plt.xlabel('Classe')
    plt.ylabel('Percentuale di pixel (%)')
    plt.title(title)
    
    # Aggiungi i valori sopra le barre
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{percentage:.1f}%', ha='center', va='bottom')
    
    # Ruota le etichette se ce ne sono molte
    if len(class_names) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Salva o mostra i risultati
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribuzione delle classi salvata in {save_path}")
    
    plt.show()
