"""
Modulo autonomo per la classificazione pixel-wise con Wavelet Scattering Transform.

Questo modulo contiene tutte le funzionalità necessarie per addestrare e utilizzare
modelli di classificazione pixel-wise basati su Wavelet Scattering Transform.
"""

from .models import PixelWiseClassifier, create_scattering_transform, train_pixel_classifier
from .dataset import PixelWiseDataset
from .utils import Config, set_seed, save_model, load_model
from .visualization import visualize_results, plot_training_metrics, plot_class_distribution
from .tools import (
    analyze_dataset,
    extract_tiles,
    extract_tiles_batch,
    analyze_model,
    interactive_tile_selection
)

__all__ = [
    # Modelli e classificazione
    'PixelWiseClassifier',
    'create_scattering_transform',
    'train_pixel_classifier',

    # Dataset e utilità
    'PixelWiseDataset',
    'Config',
    'set_seed',
    'save_model',
    'load_model',

    # Visualizzazione
    'visualize_results',
    'plot_training_metrics',
    'plot_class_distribution',

    # Strumenti aggiuntivi
    'analyze_dataset',
    'extract_tiles',
    'extract_tiles_batch',
    'analyze_model',
    'interactive_tile_selection'
]
