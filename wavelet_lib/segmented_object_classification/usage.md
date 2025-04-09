# Guida all'utilizzo del modulo Segmented Object Classification

Questo documento fornisce una guida dettagliata all'utilizzo del modulo `segmented_object_classification` della libreria Wavelet Scattering Transform.

## Indice

1. [Introduzione](#introduzione)
2. [Installazione delle dipendenze](#installazione-delle-dipendenze)
3. [Creazione di un dataset da annotazioni COCO RLE](#creazione-di-un-dataset-da-annotazioni-coco-rle)
4. [Addestramento di un modello di classificazione](#addestramento-di-un-modello-di-classificazione)
5. [Classificazione di oggetti segmentati](#classificazione-di-oggetti-segmentati)
6. [Utilizzo avanzato](#utilizzo-avanzato)
7. [Risoluzione dei problemi](#risoluzione-dei-problemi)

## Introduzione

Il modulo `segmented_object_classification` è progettato per classificare oggetti estratti da immagini segmentate utilizzando la trasformata wavelet scattering. È particolarmente utile per classificare oggetti segmentati da strumenti come Segment Anything (SAM) di Meta, che fornisce maschere in formato COCO RLE.

Le classi supportate di default sono:

- vegetazione_bassa
- vegetazione_alta
- strade
- corsi_acqua
- acqua_esterna
- edifici
- mudflat

## Installazione delle dipendenze

Prima di utilizzare il modulo, assicurati di avere installato tutte le dipendenze necessarie:

```bash
pip install torch torchvision kymatio pycocotools opencv-python matplotlib scikit-learn tqdm
```

## Creazione di un dataset da annotazioni COCO RLE

Il primo passo è creare un dataset di oggetti segmentati a partire da immagini e annotazioni in formato COCO RLE.

### Utilizzo dello script da riga di comando

```bash
python script/core/segmented_object_classification/create_dataset_from_annotations.py \
    --images_dir /path/to/images \
    --annotations_dir /path/to/annotations \
    --output_dir /path/to/dataset
```

### Parametri

- `--images_dir`: Directory contenente le immagini originali
- `--annotations_dir`: Directory contenente i file di annotazione JSON in formato COCO RLE
- `--output_dir`: Directory di output per il dataset
- `--no_mask_apply`: (Opzionale) Non applicare la maschera agli oggetti estratti

### Struttura del dataset generato

Il dataset generato avrà la seguente struttura:

```
/path/to/dataset/
├── vegetazione_bassa/
│   ├── image1_obj_001_vegetazione_bassa_0.95.png
│   ├── image2_obj_003_vegetazione_bassa_0.92.png
│   └── ...
├── vegetazione_alta/
│   ├── image1_obj_002_vegetazione_alta_0.98.png
│   └── ...
├── strade/
│   └── ...
└── ...
```

### Utilizzo programmatico

```python
from wavelet_lib.segmented_object_classification.rle_utils import create_dataset_from_annotations

stats = create_dataset_from_annotations(
    images_dir="/path/to/images",
    annotations_dir="/path/to/annotations",
    output_dir="/path/to/dataset",
    apply_mask=True
)

print(f"Immagini processate: {stats['total_images']}")
print(f"Oggetti totali estratti: {stats['total_objects']}")
print(f"Distribuzione degli oggetti per classe: {stats['objects_per_class']}")
```

## Addestramento di un modello di classificazione

Una volta creato il dataset, puoi addestrare un modello di classificazione.

### Utilizzo dello script da riga di comando

```bash
python script/core/segmented_object_classification/train_classifier.py \
    --train_dir /path/to/dataset \
    --model_path /path/to/model.pth \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
```

### Parametri

- `--train_dir`: Directory contenente i dati di training
- `--val_dir`: (Opzionale) Directory contenente i dati di validazione
- `--model_path`: Percorso dove salvare il modello
- `--input_size`: (Opzionale) Dimensione di input per il modello (default: 32,32)
- `--batch_size`: (Opzionale) Dimensione del batch (default: 32)
- `--num_epochs`: (Opzionale) Numero di epoche di training (default: 50)
- `--learning_rate`: (Opzionale) Learning rate per l'ottimizzatore (default: 0.001)
- `--weight_decay`: (Opzionale) Weight decay per l'ottimizzatore (default: 1e-4)
- `--val_split`: (Opzionale) Frazione dei dati da usare per la validazione se val_dir non è specificato (default: 0.2)
- `--j`: (Opzionale) Numero di scale per la trasformata scattering (default: 2)
- `--scattering_order`: (Opzionale) Ordine della trasformata scattering (default: 2)
- `--no_balance`: (Opzionale) Non bilanciare le classi
- `--max_samples`: (Opzionale) Numero massimo di campioni per classe
- `--no_augment`: (Opzionale) Non applicare data augmentation
- `--num_workers`: (Opzionale) Numero di worker per il data loading (default: 4)

### Utilizzo programmatico

```python
from wavelet_lib.segmented_object_classification.training import train_segmented_object_classifier

result = train_segmented_object_classifier(
    train_dir="/path/to/dataset",
    val_split=0.2,
    model_path="/path/to/model.pth",
    input_size=(32, 32),
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    weight_decay=1e-4,
    J=2,
    scattering_order=2,
    balance_classes=True,
    augment=True
)

print(f"Classi: {result['classes']}")
```

## Classificazione di oggetti segmentati

Dopo aver addestrato un modello, puoi utilizzarlo per classificare oggetti segmentati.

### Utilizzo dello script da riga di comando

#### Con annotazioni COCO RLE

```bash
python script/core/segmented_object_classification/run_segmented_classification.py \
    --image /path/to/image.jpg \
    --annotation /path/to/annotation.json \
    --model /path/to/model.pth \
    --output /path/to/output \
    --visualize
```

#### Con maschere binarie

```bash
python script/core/segmented_object_classification/run_segmented_classification.py \
    --image /path/to/image.jpg \
    --mask /path/to/mask.png \
    --model /path/to/model.pth \
    --output /path/to/output \
    --visualize
```

#### Con cartelle di immagini e annotazioni

```bash
python script/core/segmented_object_classification/run_segmented_classification.py \
    --images_dir /path/to/images \
    --annotations_dir /path/to/annotations \
    --model /path/to/model.pth \
    --output /path/to/output
```

### Parametri

- `--image`: Percorso dell'immagine da processare
- `--images_dir`: Directory contenente le immagini da processare
- `--mask`: Percorso della maschera di segmentazione
- `--masks_dir`: Directory contenente le maschere di segmentazione
- `--annotation`: Percorso del file di annotazione COCO RLE
- `--annotations_dir`: Directory contenente i file di annotazione COCO RLE
- `--model`: Percorso del modello di classificazione
- `--output`: Directory di output per i risultati
- `--j`: (Opzionale) Numero di scale per la trasformata scattering (default: 2)
- `--input_size`: (Opzionale) Dimensione di input per il modello (default: 32,32)
- `--min_size`: (Opzionale) Dimensione minima degli oggetti in pixel (default: 100)
- `--no_mask_apply`: (Opzionale) Non applicare la maschera agli oggetti estratti
- `--no_save_objects`: (Opzionale) Non salvare gli oggetti estratti
- `--visualize`: (Opzionale) Visualizza i risultati
- `--num_classes`: (Opzionale) Numero di classi del modello (default: 7)
- `--class_names`: (Opzionale) Nomi delle classi separati da virgola

### Utilizzo programmatico

```python
from wavelet_lib.segmented_object_classification.models import create_scattering_transform, load_segmented_object_classifier
from wavelet_lib.segmented_object_classification.processors import SegmentedObjectProcessor
from wavelet_lib.segmented_object_classification.rle_utils import load_coco_rle_annotations, extract_objects_from_coco_annotations

# Carica modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_to_idx = load_segmented_object_classifier(
    model_path="/path/to/model.pth",
    num_classes=7,
    device=device
)

# Crea trasformata scattering
scattering = create_scattering_transform(
    J=2,
    shape=(32, 32),
    max_order=2,
    device=device
)

# Crea processore
processor = SegmentedObjectProcessor(
    model=model,
    scattering=scattering,
    class_mapping=class_to_idx,
    device=device,
    target_size=(32, 32)
)

# Processa un'immagine con annotazione COCO RLE
import cv2
image = cv2.imread("/path/to/image.jpg")
annotations = load_coco_rle_annotations("/path/to/annotation.json")
objects = extract_objects_from_coco_annotations(image, annotations)

# Classifica oggetti
results = []
for obj_img, bbox, obj_mask, class_name, score in objects:
    result = processor.process_object(obj_img)
    results.append(result)

print(f"Risultati della classificazione: {results}")
```

## Utilizzo avanzato

### Personalizzazione delle classi

Puoi specificare classi personalizzate durante l'addestramento e la classificazione:

```bash
python script/core/segmented_object_classification/run_segmented_classification.py \
    --image /path/to/image.jpg \
    --annotation /path/to/annotation.json \
    --model /path/to/model.pth \
    --output /path/to/output \
    --class_names "classe1,classe2,classe3,classe4,classe5"
```

### Ottimizzazione dei parametri

Per ottenere risultati migliori, puoi ottimizzare i parametri della trasformata wavelet scattering:

```bash
python script/core/segmented_object_classification/train_classifier.py \
    --train_dir /path/to/dataset \
    --model_path /path/to/model.pth \
    --j 3 \
    --scattering_order 2 \
    --input_size 64,64
```

## Risoluzione dei problemi

### Errore: "No module named 'pycocotools'"

Installa pycocotools:

```bash
pip install pycocotools
```

### Errore: "CUDA out of memory"

Riduci la dimensione del batch:

```bash
python script/core/segmented_object_classification/train_classifier.py \
    --train_dir /path/to/dataset \
    --model_path /path/to/model.pth \
    --batch_size 16
```

### Errore: "No objects found in the image"

Assicurati che le annotazioni COCO RLE siano valide e che l'immagine corrisponda alle annotazioni:

```bash
# Verifica le annotazioni
python -c "import json; print(json.load(open('/path/to/annotation.json')))"
```

### Prestazioni di classificazione scarse

Prova ad aumentare la dimensione del dataset o ad applicare data augmentation più aggressiva:

```bash
python script/core/segmented_object_classification/train_classifier.py \
    --train_dir /path/to/dataset \
    --model_path /path/to/model.pth \
    --num_epochs 100 \
    --learning_rate 0.0005
```
