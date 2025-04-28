#!/bin/bash

# Script per eseguire la classificazione pixel-wise sul dataset VDD con mapping corretto

# Directory del dataset
DATASET_DIR="/home/brus/Projects/HPL/wavelet/datasets/classification_datasets/_VDD"
TEST_IMAGES_DIR="${DATASET_DIR}/test/src"

# Directory per il modello
MODEL_DIR="/home/brus/Projects/HPL/wavelet/experiments/pixel_classification/vdd"
MODEL_PATH="${MODEL_DIR}/pixel_classifier_vdd_corrected.pth"

# Directory per i risultati
OUTPUT_DIR="${MODEL_DIR}/results_corrected"
mkdir -p $OUTPUT_DIR

# Parametri di inferenza
PATCH_SIZE=32
STRIDE=16
# Numero di classi corretto per VDD
NUM_CLASSES=7

# Esegui la classificazione con i parametri corretti
python /home/brus/Projects/HPL/wavelet/script/core/pixel_classification/run_pixel_classification.py \
  --folder ${TEST_IMAGES_DIR} \
  --model ${MODEL_PATH} \
  --output ${OUTPUT_DIR} \
  --patch_size ${PATCH_SIZE} \
  --stride ${STRIDE} \
  --j 2 \
  --batch_size 4 \
  --overlay \
  --num_classes ${NUM_CLASSES}

echo "Classificazione completata. Risultati salvati in: ${OUTPUT_DIR}"
