# Wavelet Scattering Transform (WST) - Guida all'uso

## Gestione degli Esperimenti

Il framework è organizzato per gestire esperimenti multipli. Ogni esperimento ha la propria directory con tutti i risultati e le metriche associate.

### Creazione Nuovo Esperimento

```bash
python script/core/train.py \
    --dataset /path/to/dataset \
    --num-classes 4 \
    --epochs 90 \
    --output-base experiments/dataset0 \
    --experiment-name first_run \
    --balance
```

Parametri principali:
- `--dataset`: Percorso al dataset
- `--output-base`: Directory base dell'esperimento
- `--experiment-name`: Nome dell'esperimento
- `--num-classes`: Numero di classi
- `--epochs`: Numero di epoche
- `--balance`: Bilancia le classi

### Valutazione nell'Esperimento

```bash
python script/core/evaluate.py \
    --model-path experiments/dataset0/models/first_run/best_model.pth \
    --dataset /path/to/dataset \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

### Predizione nell'Esperimento

```bash
# Predizione singola immagine
python script/core/predict.py \
    --model-path experiments/dataset0/models/first_run/best_model.pth \
    --image-path /path/to/image.jpg \
    --output-base experiments/dataset0 \
    --experiment-name first_run

# Predizione con tile
python script/core/predict.py \
    --model-path experiments/dataset0/models/first_run/best_model.pth \
    --image-path /path/to/image.jpg \
    --tile-mode \
    --tile-size 32 \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

### Visualizzazione Risultati

```bash
# Metriche di training
python script/core/visualize.py metrics \
    --model-dir experiments/dataset0/model_output/first_run

# Confronto tra esperimenti
python script/core/visualize.py metrics \
    --model-dir experiments/dataset0/model_output/first_run \
    --compare-with experiments/dataset1/model_output/second_run
```

## Workflow Tipico

1. Crea un nuovo esperimento:
```bash
python script/core/train.py \
    --dataset /path/to/dataset \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

2. Valuta il modello:
```bash
python script/core/evaluate.py \
    --model-path experiments/dataset0/models/first_run/best_model.pth \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

3. Analizza i risultati:
```bash
python script/core/visualize.py metrics \
    --model-dir experiments/dataset0/model_output/first_run
```

## Confronto tra Esperimenti

### Stesso Modello, Dataset Diverso
```bash
# Training su nuovo dataset
python script/core/train.py \
    --dataset /path/to/new_dataset \
    --output-base experiments/dataset1 \
    --experiment-name wst_run

# Confronto risultati
python script/core/visualize.py metrics \
    --model-dir experiments/dataset0/model_output/first_run \
    --compare-with experiments/dataset1/model_output/wst_run
```

### Dataset Stesso, Modello Diverso
```bash
# Training con parametri diversi
python script/core/train.py \
    --dataset /path/to/dataset \
    --output-base experiments/dataset0 \
    --experiment-name second_run \
    --model-type mlp

# Confronto risultati
python script/core/visualize.py metrics \
    --model-dir experiments/dataset0/model_output/first_run \
    --compare-with experiments/dataset0/model_output/second_run
```

## Risoluzione Problemi

### Errore: "No such file or directory"
- Verificare che la directory dell'esperimento esista
- Usare percorsi assoluti
- Verificare la struttura delle directory

### Errore: "ModuleNotFoundError"
- Verificare l'attivazione dell'ambiente virtuale
- Reinstallare le dipendenze
- Verificare l'installazione in modalità sviluppo
