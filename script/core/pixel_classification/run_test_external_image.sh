#!/bin/bash

# Script per testare il classificatore pixel-wise su un'immagine esterna

# Verifica se è stata fornita un'immagine
if [ $# -lt 1 ]; then
    echo "Utilizzo: $0 <percorso_immagine> [output_file] [max_size]"
    echo "Esempio: $0 /path/to/image.jpg result.png 1024"
    echo ""
    echo "Parametri:"
    echo "  <percorso_immagine>: Percorso dell'immagine da classificare (obbligatorio)"
    echo "  [output_file]: Nome del file di output (opzionale)"
    echo "  [max_size]: Dimensione massima dell'immagine (opzionale, default: 1024)"
    exit 1
fi

# Percorso dell'immagine da testare
INPUT_IMAGE="$1"

# Nome del file di output (opzionale)
OUTPUT_FILE=""
if [ $# -ge 2 ]; then
    OUTPUT_FILE="$2"
    OUTPUT_ARGS="--output_file $OUTPUT_FILE"
else
    OUTPUT_ARGS=""
fi

# Dimensione massima dell'immagine (opzionale)
MAX_SIZE=1024
if [ $# -ge 3 ]; then
    MAX_SIZE="$3"
fi

# Directory del progetto
PROJECT_DIR="$(dirname "$0")"

# Directory di output per i risultati
OUTPUT_DIR="$PROJECT_DIR/experiments/pixel_classification/vdd/results"

# Percorso del modello addestrato
MODEL_PATH="$PROJECT_DIR/experiments/pixel_classification/vdd/pixel_classifier_vdd.pth"

# Nomi delle classi
CLASS_NAMES="sfondo,acqua,vegetazione,strade,edifici,altro"

# Esegui lo script di test
python "$PROJECT_DIR/script/core/pixel_classification/test_pixel_classifier.py" \
    --input_image "$INPUT_IMAGE" \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $OUTPUT_ARGS \
    --class_names "$CLASS_NAMES" \
    --patch_size 32 \
    --stride 16 \
    --batch_size 16 \
    --max_size $MAX_SIZE

echo ""
echo "Risultati salvati in: $OUTPUT_DIR"
