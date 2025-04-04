"""
Classification module for the Wavelet Scattering Transform Library.

This module provides tools for image classification using Wavelet Scattering Transforms,
including model definitions, processors, and utilities.
"""

# Import all components for library-level access
from wavelet_lib.classification.models import *
from wavelet_lib.classification.processors import *

# Re-export specific functions for cleaner imports
from wavelet_lib.classification.models import (
    ScatteringClassifier,
    create_scattering_transform,
    create_classification_model,
    print_classifier_summary,
)

from wavelet_lib.classification.processors import (
    ClassificationProcessor,
)