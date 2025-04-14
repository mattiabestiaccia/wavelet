"""
Single Pixel Classification module for the Wavelet Scattering Transform Library.

Questo modulo fornisce strumenti per la classificazione pixel-wise di immagini multibanda
utilizzando la trasformata wavelet scattering, inclusi modelli, processori e utility.
"""

# Importa tutti i componenti per l'accesso a livello di libreria
from wavelet_lib.single_pixel_classification.models import *
from wavelet_lib.single_pixel_classification.processors import *

# Ri-esporta funzioni specifiche per import più puliti
from wavelet_lib.single_pixel_classification.models import (
    PixelWiseClassifier,
    create_pixel_classifier,
    create_scattering_transform,
)

from wavelet_lib.single_pixel_classification.processors import (
    PixelClassificationProcessor,
    process_image,
    create_classification_map,
)
