#!/bin/bash

# Script per aggiornare il mapping delle classi nel modello

# Directory per il modello
MODEL_DIR="/home/brus/Projects/HPL/wavelet/experiments/pixel_classification/vdd"
MODEL_PATH="${MODEL_DIR}/pixel_classifier_vdd.pth"
OUTPUT_PATH="${MODEL_DIR}/pixel_classifier_vdd_corrected.pth"

# Esegui lo script per aggiornare il mapping delle classi
python /home/brus/Projects/HPL/wavelet/vdd_class_mapping.py \
  --model ${MODEL_PATH} \
  --output ${OUTPUT_PATH}

echo "Mapping delle classi aggiornato. Nuovo modello salvato in: ${OUTPUT_PATH}"
