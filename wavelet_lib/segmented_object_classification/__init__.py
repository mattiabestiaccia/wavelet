"""
Segmented Object Classification module for the Wavelet Scattering Transform Library.

Questo modulo fornisce strumenti per la classificazione di oggetti estratti da immagini segmentate
utilizzando la Wavelet Scattering Transform, inclusi modelli, processori e utility.
"""

# Importa tutti i componenti per l'accesso a livello di libreria
from wavelet_lib.segmented_object_classification.models import *
from wavelet_lib.segmented_object_classification.processors import *
from wavelet_lib.segmented_object_classification.rle_utils import *
from wavelet_lib.segmented_object_classification.training import *

# Ri-esporta funzioni specifiche per import più puliti
from wavelet_lib.segmented_object_classification.models import (
    SegmentedObjectClassifier,
    create_segmented_object_classifier,
)

from wavelet_lib.segmented_object_classification.processors import (
    SegmentedObjectProcessor,
    extract_objects_from_mask,
)

from wavelet_lib.segmented_object_classification.rle_utils import (
    load_coco_rle_annotations,
    rle_to_mask,
    extract_objects_from_coco_annotations,
    create_dataset_from_annotations,
)

from wavelet_lib.segmented_object_classification.training import (
    SegmentedObjectDataset,
    train_segmented_object_classifier,
)
