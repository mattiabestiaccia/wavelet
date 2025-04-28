#!/bin/bash

# Script per visualizzare un esempio del dataset VDD con legenda delle classi

# Directory del dataset
DATASET_DIR="/home/brus/Projects/HPL/wavelet/datasets/classification_datasets/_VDD"
IMAGES_DIR="${DATASET_DIR}/train/src"
MASKS_DIR="${DATASET_DIR}/train/gt"

# Directory per i risultati
OUTPUT_DIR="/home/brus/Projects/HPL/wavelet/experiments/pixel_classification/vdd/visualizations"
mkdir -p $OUTPUT_DIR

# Esegui lo script di visualizzazione
python /home/brus/Projects/HPL/wavelet/visualize_vdd_classes.py \
  --images_dir ${IMAGES_DIR} \
  --masks_dir ${MASKS_DIR} \
  --output "${OUTPUT_DIR}/vdd_sample_visualization.png"

echo "Visualizzazione completata. Immagine salvata in: ${OUTPUT_DIR}/vdd_sample_visualization.png"