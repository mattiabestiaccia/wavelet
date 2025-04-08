# Guida all'Utilizzo della Wavelet Scattering Transform Library

Questa guida fornisce istruzioni dettagliate per l'utilizzo della Wavelet Scattering Transform Library, una libreria completa per l'analisi di immagini utilizzando la trasformata wavelet scattering.

## Indice
1. [Introduzione](#introduzione)
2. [Installazione](#installazione)
3. [Struttura della Libreria](#struttura-della-libreria)
4. [Moduli Principali](#moduli-principali)
   - [Classificazione di Immagini](#classificazione-di-immagini)
   - [Segmentazione di Immagini](#segmentazione-di-immagini)
   - [Classificazione di Oggetti Segmentati](#classificazione-di-oggetti-segmentati)
5. [Workflow Completi](#workflow-completi)
6. [Utilizzo Avanzato](#utilizzo-avanzato)
7. [Risoluzione dei Problemi](#risoluzione-dei-problemi)

## Introduzione

La Wavelet Scattering Transform Library è una libreria Python progettata per sfruttare la potenza della trasformata wavelet scattering per diverse attività di analisi di immagini. La trasformata wavelet scattering è particolarmente efficace per estrarre caratteristiche robuste da immagini, che possono essere utilizzate per classificazione, segmentazione e altre attività di analisi.

La libreria offre tre moduli principali:
- **Single Tile Classification**: Per la classificazione di immagini singole
- **Single Tile Segmentation**: Per la segmentazione di immagini singole
- **Segmented Object Classification**: Per la classificazione di oggetti estratti da immagini segmentate

## Installazione

```bash
# Clona il repository
git clone https://github.com/mattiabestiaccia/wavelet.git
cd wavelet

# Crea un ambiente virtuale e installa le dipendenze
python3 -m venv wavelet_venv
source wavelet_venv/bin/activate
pip install -r requirements.txt

# Installa la libreria in modalità sviluppo
pip install -e .
```

## Struttura della Libreria

La libreria è organizzata nella seguente struttura:

```
wavelet/
├── wavelet_lib/                  # Directory principale della libreria
│   ├── base.py                   # Classi e funzioni di base
│   ├── datasets.py               # Gestione dei dataset
│   ├── training.py               # Funzioni di addestramento generiche
│   ├── visualization.py          # Strumenti di visualizzazione
│   ├── single_tile_classification/  # Modulo per la classificazione di immagini
│   │   ├── models.py             # Modelli di classificazione
│   │   ├── processors.py         # Processori per l'inferenza
│   │   └── usage.md              # Guida specifica per la classificazione
│   ├── single_tile_segmentation/    # Modulo per la segmentazione di immagini
│   │   ├── models.py             # Modelli di segmentazione
│   │   ├── processors.py         # Processori per l'inferenza
│   │   └── usage.md              # Guida specifica per la segmentazione
│   └── segmented_object_classification/  # Modulo per la classificazione di oggetti segmentati
│       ├── models.py             # Modelli per oggetti segmentati
│       ├── processors.py         # Processori per oggetti segmentati
│       ├── rle_utils.py          # Utilità per annotazioni COCO RLE
│       ├── training.py           # Funzioni di addestramento specifiche
│       └── usage.md              # Guida specifica per oggetti segmentati
├── script/                       # Script eseguibili
│   └── core/                     # Script principali
│       ├── classification/       # Script per la classificazione
│       ├── segmentation/         # Script per la segmentazione
│       └── segmented_object_classification/  # Script per oggetti segmentati
└── experiments/                  # Directory per gli esperimenti
```

## Moduli Principali

### Classificazione di Immagini

Il modulo `single_tile_classification` fornisce strumenti per la classificazione di immagini utilizzando la trasformata wavelet scattering.

#### Esempio di Utilizzo Base

```python
import torch
from wavelet_lib.base import Config
from wavelet_lib.single_tile_classification.models import create_classification_model
from wavelet_lib.single_tile_classification.processors import ClassificationProcessor

# Crea configurazione del modello
config = Config(
    num_classes=4,             # Numero di classi di classificazione
    num_channels=3,            # Immagini RGB
    scattering_order=2,        # Ordine massimo della trasformata scattering
    J=2,                       # Parametro di scala wavelet
    shape=(32, 32),            # Dimensione dell'immagine di input
    batch_size=128,            # Dimensione del batch per l'addestramento
    learning_rate=0.1,         # Learning rate iniziale
    weight_decay=5e-4          # Forza della regolarizzazione
)

# Crea modello e trasformata scattering
model, scattering = create_classification_model(config)

# Addestra il modello
from wavelet_lib.training import train_model
train_model(
    model=model,
    scattering=scattering,
    train_dir="/path/to/train_dataset",
    test_dir="/path/to/test_dataset",
    config=config,
    save_path="/path/to/save/model.pth"
)

# Crea processore per l'inferenza
processor = ClassificationProcessor(model_path='/path/to/save/model.pth')

# Classifica un'immagine
result = processor.process_image('image.jpg')
print(f"Predizione: {result['class']} con confidenza {result['confidence']:.2f}")
```

Per istruzioni più dettagliate, consulta la [Guida alla Classificazione di Immagini](wavelet_lib/single_tile_classification/usage.md).

### Segmentazione di Immagini

Il modulo `single_tile_segmentation` fornisce strumenti per la segmentazione di immagini utilizzando un'architettura U-Net arricchita con la trasformata wavelet scattering.

#### Esempio di Utilizzo Base

```python
from wavelet_lib.single_tile_segmentation.models import train_segmentation_model, ScatteringSegmenter
import glob

# Trova tutte le immagini e le maschere
train_images = sorted(glob.glob("/path/to/dataset/images/*.jpg"))
train_masks = sorted(glob.glob("/path/to/dataset/masks/*.png"))

# Addestra il modello
train_segmentation_model(
    train_images=train_images,
    train_masks=train_masks,
    model_path="/path/to/save/model.pth",
    J=2,
    input_shape=(256, 256),
    batch_size=8,
    num_epochs=50
)

# Carica il segmentatore
segmenter = ScatteringSegmenter(
    model_path="/path/to/save/model.pth",
    J=2,
    input_shape=(256, 256),
    apply_morphology=True
)

# Segmenta un'immagine
binary_mask, raw_pred = segmenter.predict(
    image_path="image.jpg",
    threshold=0.5,
    return_raw=True
)
```

Per istruzioni più dettagliate, consulta la [Guida alla Segmentazione di Immagini](wavelet_lib/single_tile_segmentation/usage.md).

### Classificazione di Oggetti Segmentati

Il modulo `segmented_object_classification` fornisce strumenti per la classificazione di oggetti estratti da immagini segmentate, con supporto per annotazioni in formato COCO RLE.

#### Esempio di Utilizzo Base

```python
# Crea un dataset da annotazioni COCO RLE
from wavelet_lib.segmented_object_classification.rle_utils import create_dataset_from_annotations

create_dataset_from_annotations(
    images_dir="/path/to/images",
    annotations_dir="/path/to/annotations",
    output_dir="/path/to/dataset"
)

# Addestra un classificatore
from wavelet_lib.segmented_object_classification.training import train_segmented_object_classifier

train_segmented_object_classifier(
    train_dir="/path/to/dataset",
    model_path="/path/to/save/model.pth",
    input_size=(32, 32),
    batch_size=32,
    num_epochs=50
)

# Classifica oggetti da un'immagine con annotazione
from wavelet_lib.segmented_object_classification.models import load_segmented_object_classifier
from wavelet_lib.segmented_object_classification.processors import SegmentedObjectProcessor
from wavelet_lib.segmented_object_classification.rle_utils import load_coco_rle_annotations, extract_objects_from_coco_annotations
import cv2

# Carica modello
model, class_to_idx = load_segmented_object_classifier(
    model_path="/path/to/save/model.pth",
    num_classes=7
)

# Carica immagine e annotazioni
image = cv2.imread("image.jpg")
annotations = load_coco_rle_annotations("annotation.json")

# Estrai e classifica oggetti
processor = SegmentedObjectProcessor(model=model, class_mapping=class_to_idx)
objects = extract_objects_from_coco_annotations(image, annotations)
for obj_img, bbox, obj_mask, class_name, score in objects:
    result = processor.process_object(obj_img)
    print(f"Classe: {result['class']} con confidenza {result['confidence']:.2f}")
```

Per istruzioni più dettagliate, consulta la [Guida alla Classificazione di Oggetti Segmentati](wavelet_lib/segmented_object_classification/usage.md).

## Workflow Completi

La libreria supporta diversi workflow completi per l'analisi di immagini:

### Workflow 1: Classificazione di Immagini

1. Preparazione del dataset
2. Addestramento del modello di classificazione
3. Valutazione del modello
4. Classificazione di nuove immagini

### Workflow 2: Segmentazione di Immagini

1. Preparazione del dataset con immagini e maschere
2. Addestramento del modello di segmentazione
3. Valutazione del modello
4. Segmentazione di nuove immagini

### Workflow 3: Classificazione di Oggetti Segmentati

1. Segmentazione di immagini (con SAM o altro strumento)
2. Estrazione di oggetti dalle maschere di segmentazione
3. Creazione di un dataset di oggetti segmentati
4. Addestramento di un classificatore per gli oggetti
5. Classificazione di nuovi oggetti segmentati

### Workflow 4: Pipeline Completa

1. Segmentazione di immagini
2. Estrazione di oggetti
3. Classificazione degli oggetti
4. Analisi statistica dei risultati

## Utilizzo Avanzato

### Supporto per Immagini Multibanda

La libreria supporta immagini multibanda (più di 3 canali):

```python
# Configura per immagini multibanda
config = Config(
    num_channels=5,  # Ad esempio, 5 bande
    num_classes=4,
    scattering_order=2,
    J=2,
    shape=(32, 32)
)

# Crea modello per immagini multibanda
model, scattering = create_classification_model(config)
```

### Personalizzazione dei Modelli

È possibile personalizzare i modelli modificando i parametri:

```python
# Personalizza il modello di segmentazione
from wavelet_lib.single_tile_segmentation.models import ScatteringUNet

model = ScatteringUNet(
    J=3,  # Aumenta il numero di scale
    input_shape=(512, 512),  # Aumenta la dimensione di input
    num_classes=3  # Segmentazione multi-classe
)
```

### Integrazione con Segment Anything (SAM)

La libreria può essere integrata con Segment Anything di Meta:

```python
import torch
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import json

# Carica il modello SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# Genera maschere
# ... (codice per generare maschere con SAM)

# Converti in formato COCO RLE
# ... (codice per convertire in formato COCO RLE)

# Classifica oggetti con il nostro modulo
# ... (codice per classificare oggetti)
```

## Risoluzione dei Problemi

### Problemi di Memoria CUDA

Se incontri errori di memoria CUDA, prova a:
- Ridurre la dimensione del batch
- Ridurre la dimensione delle immagini di input
- Utilizzare un modello più leggero

### Prestazioni di Classificazione Scarse

Se le prestazioni di classificazione sono scarse, prova a:
- Aumentare il numero di epoche di addestramento
- Modificare i parametri della trasformata scattering (J, ordine)
- Applicare più data augmentation
- Utilizzare un dataset più bilanciato

### Segmentazione di Bassa Qualità

Se la qualità della segmentazione è bassa, prova a:
- Aumentare la dimensione di input del modello
- Applicare operazioni morfologiche più aggressive
- Utilizzare un modello con più parametri

### Errori con Annotazioni COCO RLE

Se incontri errori con le annotazioni COCO RLE, verifica che:
- Le annotazioni siano nel formato corretto
- Le dimensioni delle immagini corrispondano a quelle nelle annotazioni
- La libreria pycocotools sia installata correttamente
