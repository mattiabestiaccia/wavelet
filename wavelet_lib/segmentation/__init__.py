"""
Segmentation module for the Wavelet Scattering Transform Library.

This module provides tools for image segmentation using Wavelet Scattering Transforms,
including model definitions, processors, and utilities.
"""

# Import all components for library-level access
from wavelet_lib.segmentation.models import *

# Re-export specific functions for cleaner imports
from wavelet_lib.segmentation.models import (
    ScatteringPreprocessor,
    ScatteringUNet,
    ScatteringSegmenter,
    SegmentationDataset,
    train_segmentation_model,
)