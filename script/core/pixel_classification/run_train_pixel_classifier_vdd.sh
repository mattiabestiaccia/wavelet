#!/bin/bash

# Script per addestrare il classificatore pixel-wise sul dataset VDD
# Creato per risolvere problemi di compatibilità col dataset

# Directory del dataset
DATASET_DIR="/home/brus/Projects/HPL/wavelet/datasets/classification_datasets/_VDD"
TRAIN_IMAGES_DIR="${DATASET_DIR}/train/src"
TRAIN_MASKS_DIR="${DATASET_DIR}/train/gt"

# Directory per il modello
MODEL_DIR="/home/brus/Projects/HPL/wavelet/experiments/pixel_classification/vdd"
mkdir -p $MODEL_DIR

# Nome del modello
MODEL_PATH="${MODEL_DIR}/pixel_classifier_vdd.pth"

# Parametri di addestramento
PATCH_SIZE=32
STRIDE=16
BATCH_SIZE=16
EPOCHS=50
NUM_CLASSES=6

# Esegui l'addestramento con i parametri corretti
python /home/brus/Projects/HPL/wavelet/script/core/pixel_classification/train_pixel_classifier.py \
  --images_dir ${TRAIN_IMAGES_DIR} \
  --masks_dir ${TRAIN_MASKS_DIR} \
  --model ${MODEL_PATH} \
  --patch_size ${PATCH_SIZE} \
  --stride ${STRIDE} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --num_classes ${NUM_CLASSES} \
  --j 2 \
  --val_split 0.2 \
  --checkpoint_interval 5 \
  --max_images 20 \
  --metadata_cache "${MODEL_DIR}/metadata_cache.pth" \
  --save_metadata

echo "Addestramento completato. Modello salvato in: ${MODEL_PATH}"