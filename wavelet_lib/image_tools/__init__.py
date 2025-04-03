"""
Strumenti per la manipolazione e visualizzazione di immagini multibanda.
Questo package fornisce classi e funzioni per la creazione, elaborazione
e visualizzazione di immagini con più di 3 bande.
"""

from .multiband_creator import create_multiband_tiffs
from .channel_visualizer import visualize_channels, visualize_channels_sequence

__all__ = ['create_multiband_tiffs', 'visualize_channels', 'visualize_channels_sequence']
