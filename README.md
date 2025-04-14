# Wavelet Scattering Transform Library

Una libreria Python completa per l'analisi di immagini utilizzando la trasformata wavelet scattering (WST), con funzionalità di classificazione, segmentazione e analisi di oggetti segmentati.

## Panoramica

Questa libreria fornisce strumenti per sfruttare le trasformate wavelet scattering per diverse attività di analisi di immagini, tra cui:

- Classificazione avanzata di immagini con caratteristiche wavelet scattering
- Segmentazione di immagini utilizzando architetture U-Net arricchite con trasformate wavelet
- Classificazione di oggetti estratti da immagini segmentate
- Supporto per dataset bilanciati e gestione di immagini multibanda
- Flussi di lavoro completi per addestramento, valutazione e predizione
- Strumenti di analisi wavelet per visualizzazione e ispezione delle caratteristiche
- Utilità per la gestione e l'elaborazione dei dataset

## Funzionalità

### Classificazione di Immagini (Single Tile Classification)

- Estrazione di caratteristiche mediante trasformata wavelet scattering per una classificazione robusta
- Classificatori di reti neurali configurabili ottimizzati per caratteristiche wavelet
- Supporto per classificazione multi-classe con bilanciamento delle classi
- Flusso di lavoro completo per addestramento, valutazione e predizione dei modelli
- Visualizzazione delle metriche di prestazione e analisi del modello

### Segmentazione di Immagini (Single Tile Segmentation)

- Architettura U-Net arricchita con trasformata wavelet scattering
- Supporto per segmentazione binaria e multi-classe
- Operazioni morfologiche post-processing per migliorare la qualità della segmentazione
- Visualizzazione dei risultati con overlay e heatmap
- Metriche di valutazione per la qualità della segmentazione

### Classificazione di Oggetti Segmentati (Segmented Object Classification)

- Estrazione di oggetti da maschere di segmentazione o annotazioni COCO RLE
- Classificazione di oggetti segmentati in categorie predefinite
- Supporto per l'integrazione con strumenti di segmentazione come Segment Anything (SAM)
- Analisi statistica della distribuzione delle classi negli oggetti segmentati
- Visualizzazione dei risultati con etichette e bounding box

### Classificazione Pixel-Wise (Single Pixel Classification)

- Classificazione di ogni pixel in immagini multibanda (3 o più bande)
- Applicazione della trasformata wavelet scattering mantenendo la coerenza tra le bande
- Classificazione in categorie come acqua, vegetazione, strade ed edifici
- Approccio a patch per processare immagini di grandi dimensioni
- Visualizzazione dei risultati con mappe colorate e overlay

### Strumenti di Analisi Wavelet

- Estrazione di caratteristiche mediante trasformata wavelet discreta (DWT)
- Visualizzazione dei coefficienti della trasformata wavelet scattering (WST)
- Supporto per immagini multi-canale e multi-banda
- Analisi statistica dei coefficienti wavelet per l'importanza delle caratteristiche

### Utilità per Dataset

- Ispezione e validazione dei dataset per il controllo della qualità
- Bilanciamento delle classi per gestire dataset sbilanciati
- Analisi delle dimensioni e della distribuzione per la comprensione del dataset
- Preprocessing delle immagini ottimizzato per le trasformate wavelet

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

## Guida Rapida

### Classificazione di Immagini

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

# Carica pesi pre-addestrati
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Crea processore per l'inferenza
processor = ClassificationProcessor(model_path='model.pth', device=config.device)

# Processa un'immagine e ottieni i risultati della classificazione
result = processor.process_image('image.jpg')
print(f"Predizione: {result['class']} con confidenza {result['confidence']:.2f}")
```

### Segmentazione di Immagini

```python
from wavelet_lib.single_tile_segmentation.models import ScatteringSegmenter
import cv2
import matplotlib.pyplot as plt

# Carica il segmentatore
segmenter = ScatteringSegmenter(
    model_path="model_segmentation.pth",
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

# Visualizza i risultati
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB))
plt.title('Immagine Originale')
plt.subplot(1, 3, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Maschera Binaria')
plt.subplot(1, 3, 3)
plt.imshow(raw_pred, cmap='jet')
plt.title('Heatmap di Predizione')
plt.tight_layout()
plt.show()
```

### Classificazione di Oggetti Segmentati

```python
from wavelet_lib.segmented_object_classification.models import load_segmented_object_classifier
from wavelet_lib.segmented_object_classification.processors import SegmentedObjectProcessor
from wavelet_lib.segmented_object_classification.rle_utils import load_coco_rle_annotations, extract_objects_from_coco_annotations
import cv2

# Carica modello
model, class_to_idx = load_segmented_object_classifier(
    model_path="model_segmented_objects.pth",
    num_classes=7,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Carica immagine e annotazioni
image = cv2.imread("image.jpg")
annotations = load_coco_rle_annotations("annotation.json")

# Estrai oggetti
objects = extract_objects_from_coco_annotations(image, annotations)
print(f"Estratti {len(objects)} oggetti dall'immagine")

# Crea processore
processor = SegmentedObjectProcessor(
    model=model,
    scattering=None,  # Verrà creato automaticamente
    class_mapping=class_to_idx,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Classifica oggetti
for i, (obj_img, bbox, obj_mask, class_name, score) in enumerate(objects):
    result = processor.process_object(obj_img)
    print(f"Oggetto {i}: Classe {result['class']} con confidenza {result['confidence']:.2f}")
```

### Classificazione Pixel-Wise

```python
from wavelet_lib.single_pixel_classification.processors import PixelClassificationProcessor
import cv2
import matplotlib.pyplot as plt

# Carica il processore
processor = PixelClassificationProcessor(
    model_path="model_pixel_classification.pth",
    patch_size=32,
    stride=16,
    J=2
)

# Classifica un'immagine pixel per pixel
classification_map = processor.process_image(
    image_path="image.jpg",
    output_path="classification_map.png",
    overlay=True
)

# Visualizza i risultati
processor.visualize_results(
    image_path="image.jpg",
    classification_map=classification_map
)

# Crea legenda
processor.create_legend(output_path="legend.png")
```

## Utilizzo da Riga di Comando

### Classificazione di Immagini

```bash
# Addestra un modello di classificazione
python script/core/classification/train_classification.py --dataset /path/to/dataset --num-classes 4 --epochs 90

# Valuta un modello
python script/core/classification/evaluate_classification.py --model /path/to/model.pth --dataset /path/to/test_dataset

# Effettua predizioni
python script/core/classification/predict_classification.py --model /path/to/model.pth --image /path/to/image.jpg
```

### Segmentazione di Immagini

```bash
# Addestra un modello di segmentazione
python script/core/segmentation/train_segmentation.py --imgs_dir /path/to/images --masks_dir /path/to/masks --model /path/to/model.pth --epochs 50

# Segmenta un'immagine
python script/core/segmentation/run_segmentation.py --image /path/to/image.jpg --model /path/to/model.pth --output /path/to/output --overlay

# Segmenta una cartella di immagini
python script/core/segmentation/run_segmentation.py --folder /path/to/images --model /path/to/model.pth --output /path/to/output
```

### Classificazione di Oggetti Segmentati

```bash
# Crea un dataset da annotazioni COCO RLE
python script/core/segmented_object_classification/create_dataset_from_annotations.py --images_dir /path/to/images --annotations_dir /path/to/annotations --output_dir /path/to/dataset

# Addestra un classificatore di oggetti segmentati
python script/core/segmented_object_classification/train_classifier.py --train_dir /path/to/dataset --model_path /path/to/model.pth --num_epochs 50

# Classifica oggetti da un'immagine con annotazione
python script/core/segmented_object_classification/run_segmented_classification.py --image /path/to/image.jpg --annotation /path/to/annotation.json --model /path/to/model.pth --output /path/to/output --visualize
```

### Classificazione Pixel-Wise

```bash
# Addestra un classificatore pixel-wise
python script/core/pixel_classification/train_pixel_classifier.py --images_dir /path/to/images --masks_dir /path/to/masks --model /path/to/model.pth --patch_size 32 --stride 16 --epochs 50

# Classifica un'immagine pixel per pixel
python script/core/pixel_classification/run_pixel_classification.py --image /path/to/image.jpg --model /path/to/model.pth --output /path/to/output --overlay

# Classifica una cartella di immagini
python script/core/pixel_classification/run_pixel_classification.py --folder /path/to/images --model /path/to/model.pth --output /path/to/output
```

## Struttura degli Esperimenti

Ogni esperimento è organizzato nella seguente struttura:

```
experiments/nome_dataset/
├── classification_result/    # Risultati di classificazione e matrici di confusione
├── segmentation_result/      # Risultati di segmentazione e metriche di valutazione
├── segmented_objects/        # Oggetti segmentati e risultati di classificazione
├── dataset_info/             # Statistiche del dataset e distribuzioni delle classi
│   ├── dataset_report.txt    # Analisi dettagliata del dataset
│   └── dataset_stats.png     # Visualizzazioni delle proprietà del dataset
├── evaluation/               # Metriche di valutazione e analisi delle prestazioni
├── models/                   # Checkpoint dei modelli e pesi salvati
│   ├── class_distribution.png  # Visualizzazione della distribuzione delle classi
│   └── training_metrics.png    # Curve di accuratezza e loss
├── model_output/             # Log di addestramento e output intermedi
└── visualization/            # Visualizzazioni delle caratteristiche e interpretabilità del modello
```

## Utilizzo Avanzato

Per un utilizzo più avanzato e istruzioni dettagliate, consulta la [Guida all'Utilizzo](USAGE.md).

Ogni modulo ha anche una propria guida dettagliata:

- [Guida alla Classificazione di Immagini](wavelet_lib/single_tile_classification/usage.md)
- [Guida alla Segmentazione di Immagini](wavelet_lib/single_tile_segmentation/usage.md)
- [Guida alla Classificazione di Oggetti Segmentati](wavelet_lib/segmented_object_classification/usage.md)
- [Guida alla Classificazione Pixel-Wise](wavelet_lib/single_pixel_classification/usage.md)

## Requisiti

- Python 3.8+
- PyTorch 1.12+
- Kymatio 0.3.0+
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib
- Albumentations (per data augmentation)
- pycocotools (per la gestione di annotazioni COCO RLE)

## Crediti

- Trasformata Wavelet Scattering: [Kymatio](https://github.com/kymatio/kymatio)
- I componenti di segmentazione sono ispirati all'architettura U-Net
- L'integrazione con Segment Anything (SAM) è basata sul lavoro di Meta AI
