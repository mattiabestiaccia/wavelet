"""
Strumenti per la creazione e gestione di dataset di immagini.
Questo package fornisce classi e funzioni per la creazione di maschere,
l'estrazione e l'elaborazione di tiles da immagini, con supporto per immagini multibanda,
e strumenti per l'ispezione e la validazione di dataset.
"""

from .tile_mask_creator import TileMaskCreator
from .tile_processor import TileProcessor
from .dataset_inspector import (
    inspect_dataset_structure,
    analyze_image_properties,
    validate_dataset,
    generate_report,
    visualize_dataset_stats
)

__all__ = [
    'TileMaskCreator',
    'TileProcessor',
    'inspect_dataset_structure',
    'analyze_image_properties',
    'validate_dataset',
    'generate_report',
    'visualize_dataset_stats'
]
