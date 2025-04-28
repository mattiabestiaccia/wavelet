#!/bin/bash

# Script per testare il classificatore pixel-wise sul dataset VDD con mapping corretto

# Directory del dataset
DATASET_DIR="/home/brus/Projects/HPL/wavelet/datasets/classification_datasets/_VDD"
TEST_IMAGES_DIR="${DATASET_DIR}/test/src"
TEST_MASKS_DIR="${DATASET_DIR}/test/gt"

# Directory per il modello
MODEL_DIR="/home/brus/Projects/HPL/wavelet/experiments/pixel_classification/vdd"
MODEL_PATH="${MODEL_DIR}/pixel_classifier_vdd_corrected.pth"

# Directory per i risultati
OUTPUT_DIR="${MODEL_DIR}/test_results_corrected"
mkdir -p $OUTPUT_DIR

# Parametri di test
PATCH_SIZE=32
STRIDE=16
BATCH_SIZE=16
# Numero di classi nel dataset VDD (0-6 = 7 classi)
NUM_CLASSES=7

# Esegui il test su un'immagine casuale
python /home/brus/Projects/HPL/wavelet/script/core/pixel_classification/test_pixel_classifier.py \
  --images_dir ${TEST_IMAGES_DIR} \
  --masks_dir ${TEST_MASKS_DIR} \
  --model ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --patch_size ${PATCH_SIZE} \
  --stride ${STRIDE} \
  --batch_size ${BATCH_SIZE} \
  --j 2 \
  --num_classes ${NUM_CLASSES} \
  --class_names "background,wall,roads,vegetation,vehicles,roof,others"

echo "Test completato. Risultati salvati in: ${OUTPUT_DIR}"
