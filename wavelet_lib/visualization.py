"""
Visualization module for the Wavelet Scattering Transform Library.
Contains functions for visualizing results.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image

def plot_class_distribution(dataset, title="Class Distribution", figsize=(10, 6), save_path=None):
    """
    Plot the class distribution of a dataset.
    
    Args:
        dataset: Dataset object
        title: Title for the plot
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Get class distribution - handle different dataset types
    if hasattr(dataset, 'get_class_distribution'):
        class_distribution = dataset.get_class_distribution()
    else:
        # Fallback for dataset without get_class_distribution method
        from collections import defaultdict
        class_distribution = defaultdict(int)
        for _, label in dataset:
            if hasattr(dataset, 'classes'):
                class_name = dataset.classes[label]
            else:
                class_name = f"Class {label}"
            class_distribution[class_name] += 1
    
    # Create sorted distribution for better visualization
    classes = list(class_distribution.keys())
    values = list(class_distribution.values())
    
    plt.figure(figsize=figsize)
    bars = plt.bar(classes, values, color='skyblue')
    
    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height}', ha='center', va='bottom', fontsize=10)
    
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    
    plt.show()

def visualize_batch(dataloader, num_images=16, figsize=(12, 12), save_path=None):
    """
    Visualize a batch of images from a dataloader.
    
    Args:
        dataloader: DataLoader object
        num_images: Number of images to display
        figsize: Figure size
        save_path: Path to save the visualization
    """
    # Get a batch of data
    images, labels = next(iter(dataloader))
    
    # Get class names if available
    if hasattr(dataloader.dataset, 'classes'):
        class_names = dataloader.dataset.classes
    elif hasattr(dataloader.dataset.dataset, 'classes'):
        class_names = dataloader.dataset.dataset.classes
    else:
        class_names = None
    
    # Limit to num_images
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Define grid size
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot images
    for i, (image, label) in enumerate(zip(images, labels)):
        # Convert tensor to numpy
        img = image.permute(1, 2, 0).numpy()
        
        # Denormalize if needed
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        
        # Add subplot
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img)
        plt.axis('off')
        
        # Add label
        if class_names:
            class_name = class_names[label]
            plt.title(f"{class_name}")
        else:
            plt.title(f"Class {label}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def visualize_classification_results(results, class_names=None, figsize=(15, 12), save_path=None):
    """
    Visualize the results of tile-based classification.
    
    Args:
        results: Dictionary containing classification results
        class_names: List of class names
        figsize: Figure size
        save_path: Path to save the visualization
    """
    # Extract data from results
    image = results['cropped_image']
    label_matrix = results['label_matrix']
    confidence_matrix = results['confidence_matrix']
    tile_size = results['tile_size']
    
    # If class_names is not provided, use class indices
    if class_names is None:
        max_class = np.max(label_matrix)
        class_names = [f"Class {i}" for i in range(max_class + 1)]
    
    num_classes = len(class_names)
    colors = list(mcolors.TABLEAU_COLORS.values())[:num_classes]
    
    # Count class occurrences
    class_counts = {}
    for class_idx, name in enumerate(class_names):
        class_counts[name] = np.sum(label_matrix == class_idx)
    total_tiles = label_matrix.size
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Show the image
    plt.imshow(image)
    
    # Draw colored rectangles for classified tiles
    ax = plt.gca()
    for i in range(label_matrix.shape[0]):
        for j in range(label_matrix.shape[1]):
            label = label_matrix[i, j]
            if label >= 0 and label < num_classes:
                confidence = confidence_matrix[i, j]
                # Adjust alpha based on confidence
                alpha = min(0.3 + confidence * 0.3, 0.7)
                
                color = colors[label]
                rect = plt.Rectangle(
                    (j * tile_size, i * tile_size),
                    tile_size, tile_size,
                    linewidth=1,
                    edgecolor=color,
                    facecolor=color,
                    alpha=alpha
                )
                ax.add_patch(rect)
    
    # Create legend
    legend_patches = []
    for class_idx, class_name in enumerate(class_names):
        count = class_counts[class_name]
        percentage = 100 * count / total_tiles
        patch = plt.Rectangle((0, 0), 1, 1,
                               linewidth=1,
                               edgecolor=colors[class_idx],
                               facecolor=colors[class_idx],
                               label=f"{class_name}: {count} tiles ({percentage:.1f}%)")
        legend_patches.append(patch)
    
    # Add legend
    plt.legend(handles=legend_patches,
               loc='center left',
               bbox_to_anchor=(1, 0.5),
               fontsize=10,
               framealpha=0.8)
    
    plt.title(f'Tile Classification - {total_tiles} tiles ({label_matrix.shape[0]}×{label_matrix.shape[1]})')
    plt.tight_layout()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def visualize_scattering_coeffs(scattering, image_tensor, figsize=(15, 10), save_path=None):
    """
    Visualize scattering coefficients for an image.
    
    Args:
        scattering: Scattering transform
        image_tensor: Image tensor of shape (1, C, H, W)
        figsize: Figure size
        save_path: Path to save the visualization
    """
    # Apply scattering transform
    with torch.no_grad():
        S = scattering(image_tensor)
    
    # Get shape information
    batch_size, num_coeffs, h, w = S.shape
    order0_size = 1
    order1_size = scattering.J
    order2_size = (scattering.J * (scattering.J - 1)) // 2
    
    # Initialize figure
    plt.figure(figsize=figsize)
    
    # Subplot layout
    total_plots = min(36, num_coeffs)  # Limit number of plots
    grid_size = int(np.ceil(np.sqrt(total_plots)))
    
    # Plot original image
    plt.subplot(grid_size, grid_size, 1)
    img = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    img = img * 0.5 + 0.5  # Denormalize if needed
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot scattering coefficients
    for i in range(1, total_plots):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Determine coefficient order
        if i < order0_size:
            title = "Order 0"
        elif i < order0_size + order1_size:
            title = f"Order 1 ({i - order0_size})"
        else:
            title = f"Order 2 ({i - order0_size - order1_size})"
        
        # Plot coefficient
        plt.imshow(S[0, i].cpu().numpy(), cmap='viridis')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def plot_training_metrics(epochs, train_accuracies, test_accuracies, train_losses, test_losses, save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        epochs: Number of epochs
        train_accuracies: List of training accuracies
        test_accuracies: List of validation accuracies
        train_losses: List of training losses
        test_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 6))
    
    # For single epoch case, we'll create a bar chart instead of a line plot
    if len(train_accuracies) == 1:
        # Plot accuracy
        plt.subplot(1, 2, 1)
        labels = ['Training', 'Validation']
        values = [train_accuracies[0], test_accuracies[0]]
        bars = plt.bar(labels, values, color=['blue', 'red'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
        
        plt.ylim(0, max(values) * 1.2)  # Add some space at the top
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        loss_values = [train_losses[0], test_losses[0]]
        bars = plt.bar(labels, loss_values, color=['blue', 'red'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.ylim(0, max(loss_values) * 1.2)  # Add some space at the top
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    else:
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_accuracies, 'b-', label='Training')
        plt.plot(range(1, epochs + 1), test_accuracies, 'r-', label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Add annotation for best accuracy
        max_acc = max(test_accuracies)
        max_epoch = test_accuracies.index(max_acc) + 1
        plt.annotate(f'Best: {max_acc:.2f}%', 
                    xy=(max_epoch, max_acc),
                    xytext=(max_epoch, max_acc-5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    ha='center', fontsize=9)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Training')
        plt.plot(range(1, epochs + 1), test_losses, 'r-', label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Add annotation for best loss
        min_loss = min(test_losses)
        min_epoch = test_losses.index(min_loss) + 1
        plt.annotate(f'Best: {min_loss:.4f}', 
                    xy=(min_epoch, min_loss),
                    xytext=(min_epoch, min_loss+0.2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to: {save_path}")
    
    plt.show()

def plot_confusion_matrix(model, scattering, test_loader, class_names, device,
                         figsize=(10, 8), save_path=None):
    """
    Plot a confusion matrix for model evaluation.
    
    Args:
        model: Trained model
        scattering: Scattering transform
        test_loader: DataLoader for test data
        class_names: List of class names
        device: Device to use for computation
        figsize: Figure size
        save_path: Path to save the visualization
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize confusion matrix
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Evaluate model
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Apply scattering and model
            scattering_coeffs = scattering(data)
            output = model(scattering_coeffs)
            
            # Get predictions
            _, predictions = torch.max(output, 1)
            
            # Update confusion matrix
            for t, p in zip(target.cpu().numpy(), predictions.cpu().numpy()):
                confusion_matrix[t, p] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=figsize)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add counts to each cell
    thresh = confusion_matrix.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    # Calculate metrics
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Per-class metrics
    class_metrics = {}
    for i in range(num_classes):
        # True positives, false positives, false negatives
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"{class_names[i]}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    return confusion_matrix, class_metrics

def visualize_class_samples(dataset, model=None, num_samples=16, figsize=(15, 10), save_path=None):
    """
    Visualizza campioni dal dataset, opzionalmente con predizioni del modello.
    
    Args:
        dataset: Dataset object
        model: Modello addestrato (opzionale)
        num_samples: Numero di campioni da visualizzare
        figsize: Dimensioni della figura
        save_path: Percorso per salvare la visualizzazione
    """
    import torch
    from torch.utils.data import DataLoader
    
    # Crea un dataloader con batch size = num_samples
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    
    # Ottieni un batch di immagini
    images, labels = next(iter(loader))
    
    # Ottieni i nomi delle classi se disponibili
    class_names = dataset.classes if hasattr(dataset, 'classes') else None
    
    # Calcola le predizioni se il modello è fornito
    predictions = None
    if model is not None:
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
    
    # Crea la griglia di visualizzazione
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = rows
    
    plt.figure(figsize=figsize)
    for idx in range(min(num_samples, len(images))):
        plt.subplot(rows, cols, idx + 1)
        
        # Converti e denormalizza l'immagine
        img = images[idx].permute(1, 2, 0).numpy()
        img = img * 0.5 + 0.5  # denormalizza
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.axis('off')
        
        # Aggiungi etichetta e predizione
        if class_names:
            true_label = class_names[labels[idx]]
            title = f"True: {true_label}"
            if predictions is not None:
                pred_label = class_names[predictions[idx]]
                title += f"\nPred: {pred_label}"
        else:
            title = f"Class {labels[idx]}"
            if predictions is not None:
                title += f"\nPred: {predictions[idx]}"
        
        plt.title(title, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualizzazione salvata in: {save_path}")
    
    plt.show()
