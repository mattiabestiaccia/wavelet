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
    --balance \
    --num-channels 3  # Specificare per immagini multibanda
```

Parametri principali:
- `--dataset`: Percorso al dataset
- `--output-base`: Directory base dell'esperimento
- `--experiment-name`: Nome dell'esperimento
- `--num-classes`: Numero di classi
- `--epochs`: Numero di epoche
- `--balance`: Bilancia le classi
- `--num-channels`: Numero di canali dell'immagine (default: 3, max: 10)

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

# Predizione su immagine multibanda
python script/core/predict.py \
    --model-path experiments/dataset0/models/first_run/best_model.pth \
    --image-path /path/to/multiband_image.tif \
    --output-base experiments/dataset0 \
    --experiment-name first_run \
    --num-channels 5  # Specificare il numero di canali dell'immagine
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

## Utilizzo con Immagini Multibanda

### Creazione Immagini Multibanda
È possibile creare file TIFF multibanda a partire da immagini monobanda utilizzando lo strumento `multiband_creator.py`:

```bash
python -m wavelet_lib.image_tools.multiband_creator \
    input_dir output_dir \
    --max-bands 10 \
    --compression lzw
```

### Visualizzazione Immagini Multibanda
Per visualizzare immagini multibanda è possibile utilizzare lo strumento di visualizzazione:

```bash
python -m wavelet_lib.image_tools.channel_visualizer \
    path/to/multiband_image.tif \
    --separate \
    --figure-size 15 10
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

### Errore: "ValueError: Number of channels exceeds maximum supported channels"
- Verificare che il numero di canali specificato non superi il massimo supportato (10)
- Assicurarsi che l'immagine di input abbia il numero di canali specificato
