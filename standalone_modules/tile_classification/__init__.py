"""
Modulo autonomo per la classificazione di tile con Wavelet Scattering Transform.

Questo modulo contiene tutte le funzionalità necessarie per addestrare e utilizzare
modelli di classificazione basati su Wavelet Scattering Transform per immagini suddivise in tile.
"""

from .models import ScatteringClassifier, create_scattering_transform
from .processors import ClassificationProcessor
from .utils import Config, set_seed, save_model, load_model
from .dataset import BalancedDataset, get_default_transform, create_data_loaders
from .training import Trainer, create_optimizer
from .visualization import (
    visualize_classification_results, 
    plot_training_metrics, 
    plot_class_distribution
)

__all__ = [
    'ScatteringClassifier',
    'create_scattering_transform',
    'ClassificationProcessor',
    'Config',
    'set_seed',
    'save_model',
    'load_model',
    'BalancedDataset',
    'get_default_transform',
    'create_data_loaders',
    'Trainer',
    'create_optimizer',
    'visualize_classification_results',
    'plot_training_metrics',
    'plot_class_distribution'
]
