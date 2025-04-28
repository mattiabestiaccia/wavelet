# Modulo di Classificazione Tile con Wavelet Scattering Transform

Questo modulo autonomo fornisce funzionalità complete per la classificazione di immagini utilizzando la trasformata scattering wavelet, con particolare attenzione all'analisi basata su tile.

## Caratteristiche

- **Trasformata Scattering Wavelet**: Utilizza la libreria Kymatio per estrarre caratteristiche robuste dalle immagini
- **Classificazione di immagini intere**: Classifica immagini complete in categorie predefinite
- **Analisi basata su tile**: Suddivide le immagini in tile e classifica ogni tile individualmente
- **Visualizzazione avanzata**: Strumenti per visualizzare i risultati della classificazione e le metriche di addestramento
- **Gestione del dataset**: Supporto per dataset bilanciati e augmentation dei dati
- **Addestramento flessibile**: Configurazione completa dei parametri di addestramento

## Installazione

### Prerequisiti

- Python 3.8+
- PyTorch 1.8+
- Kymatio 0.3+

### Dipendenze

```bash
pip install torch torchvision tqdm matplotlib numpy pillow kymatio
```

## Struttura del modulo

```
tile_classification/
├── __init__.py          # Esporta le funzioni e le classi principali
├── models.py            # Definizioni dei modelli neurali
├── processors.py        # Processori per l'elaborazione delle immagini
├── utils.py             # Funzioni di utilità
├── dataset.py           # Gestione dei dataset
├── training.py          # Funzionalità di addestramento
├── visualization.py     # Funzioni di visualizzazione
├── train.py             # Script di addestramento
├── predict.py           # Script di predizione
└── README.md            # Documentazione
```

## Utilizzo

### Addestramento di un modello

```bash
python train.py --dataset /path/to/dataset --num-classes 4 --epochs 90 --output-dir /path/to/output
```

Opzioni principali:
- `--dataset`: Percorso alla directory del dataset (obbligatorio)
- `--num-classes`: Numero di classi (default: 4)
- `--epochs`: Numero di epoche di addestramento (default: 90)
- `--batch-size`: Dimensione del batch (default: 128)
- `--balance`: Bilancia le classi nel dataset
- `--output-dir`: Directory per salvare i risultati

### Predizione con un modello addestrato

#### Classificazione di un'immagine intera

```bash
python predict.py --model-path /path/to/model.pth --image-path /path/to/image.jpg
```

#### Analisi basata su tile

```bash
python predict.py --model-path /path/to/model.pth --image-path /path/to/image.jpg --tile-mode --tile-size 32
```

Opzioni principali:
- `--model-path`: Percorso del file del modello (obbligatorio)
- `--image-path`: Percorso dell'immagine da classificare (obbligatorio)
- `--tile-mode`: Abilita la modalità tile
- `--tile-size`: Dimensione del tile (default: 32)
- `--confidence-threshold`: Soglia di confidenza per la visualizzazione (default: 0.7)
- `--output-dir`: Directory per salvare i risultati

## Struttura del dataset

Il dataset deve essere organizzato nella seguente struttura:

```
dataset/
├── classe_1/
│   ├── immagine1.jpg
│   ├── immagine2.jpg
│   └── ...
├── classe_2/
│   ├── immagine1.jpg
│   ├── immagine2.jpg
│   └── ...
└── ...
```

Ogni sottodirectory rappresenta una classe e contiene le immagini di quella classe.

## Esempi

### Addestramento di un modello su un dataset a 4 classi

```bash
python train.py --dataset /path/to/dataset --num-classes 4 --epochs 90 --batch-size 64 --balance --output-dir risultati/modello_4_classi
```

### Classificazione di un'immagine con un modello addestrato

```bash
python predict.py --model-path risultati/modello_4_classi/modello.pth --image-path /path/to/image.jpg
```

### Analisi basata su tile di un'immagine

```bash
python predict.py --model-path risultati/modello_4_classi/modello.pth --image-path /path/to/image.jpg --tile-mode --tile-size 32 --confidence-threshold 0.8
```

## Utilizzo programmatico

```python
from tile_classification.models import create_classification_model
from tile_classification.utils import Config, load_model
from tile_classification.processors import ClassificationProcessor

# Crea una configurazione
config = Config(num_classes=4, J=2, scattering_order=2)

# Crea un modello
model, scattering = create_classification_model(config)

# Carica un modello addestrato
checkpoint = load_model('path/to/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Crea un processore
processor = ClassificationProcessor(model, scattering, device, class_names)

# Classifica un'immagine
result = processor.process_image('path/to/image.jpg')
print(f"Classe: {result['class_name']}, Confidenza: {result['confidence']}")

# Analisi basata su tile
results = processor.classify_image_tiles('path/to/image.jpg', tile_size=32)
```

## Licenza

Questo modulo è distribuito con licenza MIT.

## Riconoscimenti

Questo modulo è basato sul lavoro originale del progetto Wavelet e utilizza la libreria Kymatio per la trasformata scattering wavelet.
