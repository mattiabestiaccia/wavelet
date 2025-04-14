# Guida all'utilizzo del modulo Single Pixel Classification

Questo documento fornisce una guida dettagliata all'utilizzo del modulo `single_pixel_classification` della libreria Wavelet Scattering Transform.

## Indice
1. [Introduzione](#introduzione)
2. [Installazione delle dipendenze](#installazione-delle-dipendenze)
3. [Preparazione del dataset](#preparazione-del-dataset)
4. [Addestramento di un modello di classificazione](#addestramento-di-un-modello-di-classificazione)
5. [Classificazione di immagini](#classificazione-di-immagini)
6. [Utilizzo avanzato](#utilizzo-avanzato)
7. [Risoluzione dei problemi](#risoluzione-dei-problemi)

## Introduzione

Il modulo `single_pixel_classification` è progettato per la classificazione pixel-wise di immagini multibanda. A differenza della classificazione di immagini intere, questo modulo classifica ogni pixel dell'immagine in categorie come acqua, vegetazione, strade ed edifici, mantenendo la coerenza tra le bande.

Il modulo supporta due modalità di funzionamento:

1. **Con trasformata wavelet scattering**: Applica la trasformata wavelet scattering a ciascuna patch prima della classificazione, estraendo caratteristiche robuste di texture e scala.

2. **Senza trasformata wavelet scattering**: Classifica direttamente i pixel utilizzando una rete neurale convoluzionale più profonda, senza la fase di trasformazione wavelet.

Entrambe le modalità utilizzano un approccio a patch per processare l'immagine, classificando i pixel centrali di ciascuna patch. Questo approccio è particolarmente efficace per immagini satellitari o aeree multibanda, dove le caratteristiche di texture e scala sono importanti per la classificazione del terreno.

Le classi supportate di default sono:
- background
- water (acqua)
- vegetation (vegetazione)
- streets (strade)
- buildings (edifici)

## Installazione delle dipendenze

Prima di utilizzare il modulo, assicurati di avere installato tutte le dipendenze necessarie:

```bash
pip install torch torchvision kymatio opencv-python matplotlib scikit-learn tqdm albumentations
```

## Preparazione del dataset

Il modulo `single_pixel_classification` richiede un dataset composto da immagini e relative maschere di classe.

### Struttura del dataset

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1_mask.png
    ├── image2_mask.png
    └── ...
```

Le maschere devono essere immagini in scala di grigi dove ogni valore di pixel rappresenta una classe:
- 0: Background
- 1: Acqua
- 2: Vegetazione
- 3: Strade
- 4: Edifici
- ...

### Creazione di maschere di classe

Se non disponi già di maschere di classe, puoi crearle utilizzando strumenti di annotazione come QGIS, LabelMe o altri software di segmentazione semantica.

## Addestramento di un modello di classificazione

Una volta preparato il dataset, puoi addestrare un modello di classificazione pixel-wise.

### Utilizzo dello script da riga di comando

#### Con trasformata wavelet scattering (default)

```bash
python script/core/pixel_classification/train_pixel_classifier.py \
    --images_dir /path/to/dataset/images \
    --masks_dir /path/to/dataset/masks \
    --model /path/to/save/model.pth \
    --patch_size 32 \
    --stride 16 \
    --batch_size 16 \
    --epochs 50 \
    --j 2
```

#### Senza trasformata wavelet scattering

```bash
python script/core/pixel_classification/train_pixel_classifier.py \
    --images_dir /path/to/dataset/images \
    --masks_dir /path/to/dataset/masks \
    --model /path/to/save/model_no_scattering.pth \
    --patch_size 32 \
    --stride 16 \
    --batch_size 16 \
    --epochs 50 \
    --no_scattering
```

### Parametri
- `--images_dir`: Directory contenente le immagini di training
- `--masks_dir`: Directory contenente le maschere di classe
- `--model`: Percorso dove salvare il modello
- `--patch_size`: Dimensione delle patch (default: 32)
- `--stride`: Passo per l'estrazione delle patch (default: 16)
- `--batch_size`: Dimensione del batch (default: 16)
- `--epochs`: Numero di epoche (default: 50)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--val_split`: Frazione dei dati da usare per la validazione (default: 0.2)
- `--j`: Numero di scale per la trasformata scattering (default: 2)
- `--scattering_order`: Ordine della trasformata scattering (default: 2)
- `--no_scattering`: Disabilita la trasformata scattering
- `--no_augment`: Disabilita data augmentation
- `--seed`: Seed per la riproducibilità (default: 42)
- `--num_classes`: Numero di classi (default: 5)
- `--class_names`: Nomi delle classi separati da virgola

### Utilizzo programmatico

#### Con trasformata wavelet scattering

```python
from wavelet_lib.base import Config
from wavelet_lib.single_pixel_classification.models import (
    create_pixel_classifier,
    train_pixel_classifier,
    PixelWiseDataset
)
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# Crea dataset
dataset = PixelWiseDataset(
    images_dir="/path/to/dataset/images",
    masks_dir="/path/to/dataset/masks",
    patch_size=32,
    stride=16,
    augment=True
)

# Dividi in training e validation
train_indices, val_indices = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    random_state=42
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Crea configurazione con trasformata scattering
config = Config(
    num_channels=3,  # RGB
    num_classes=5,
    scattering_order=2,
    J=2,
    shape=(32, 32)
)
config.use_scattering = True  # Abilita la trasformata scattering

# Crea modello e trasformata scattering
model, scattering = create_pixel_classifier(config)

# Addestra il modello
history = train_pixel_classifier(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    model_path="/path/to/save/model.pth",
    batch_size=16,
    num_epochs=50,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    scattering=scattering,
    model=model,
    use_scattering=True
)
```

#### Senza trasformata wavelet scattering

```python
from wavelet_lib.base import Config
from wavelet_lib.single_pixel_classification.models import (
    create_pixel_classifier,
    train_pixel_classifier,
    PixelWiseDataset
)
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# Crea dataset
dataset = PixelWiseDataset(
    images_dir="/path/to/dataset/images",
    masks_dir="/path/to/dataset/masks",
    patch_size=32,
    stride=16,
    augment=True
)

# Dividi in training e validation
train_indices, val_indices = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    random_state=42
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Crea configurazione senza trasformata scattering
config = Config(
    num_channels=3,  # RGB
    num_classes=5,
    shape=(32, 32)
)
config.use_scattering = False  # Disabilita la trasformata scattering

# Crea modello (senza trasformata scattering)
model, _ = create_pixel_classifier(config)

# Addestra il modello
history = train_pixel_classifier(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    model_path="/path/to/save/model_no_scattering.pth",
    batch_size=16,
    num_epochs=50,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model=model,
    use_scattering=False
)
```

## Classificazione di immagini

Dopo aver addestrato un modello, puoi utilizzarlo per classificare nuove immagini.

### Utilizzo dello script da riga di comando

#### Con trasformata wavelet scattering (default)

```bash
python script/core/pixel_classification/run_pixel_classification.py \
    --image /path/to/image.jpg \
    --model /path/to/model.pth \
    --output /path/to/output \
    --overlay \
    --patch_size 32 \
    --stride 16 \
    --j 2
```

#### Senza trasformata wavelet scattering

```bash
python script/core/pixel_classification/run_pixel_classification.py \
    --image /path/to/image.jpg \
    --model /path/to/model_no_scattering.pth \
    --output /path/to/output \
    --overlay \
    --patch_size 32 \
    --stride 16 \
    --no_scattering
```

Per processare una cartella di immagini:

```bash
python script/core/pixel_classification/run_pixel_classification.py \
    --folder /path/to/images \
    --model /path/to/model.pth \
    --output /path/to/output \
    --overlay
```

### Parametri
- `--image`: Percorso dell'immagine da processare
- `--folder`: Directory contenente le immagini da processare
- `--model`: Percorso del modello di classificazione
- `--output`: Directory di output per i risultati
- `--patch_size`: Dimensione delle patch (default: 32)
- `--stride`: Passo per l'inferenza (default: 16)
- `--j`: Numero di scale per la trasformata scattering (default: 2)
- `--no_scattering`: Disabilita la trasformata scattering
- `--overlay`: Crea un overlay con l'immagine originale
- `--alpha`: Opacità dell'overlay (default: 0.5)
- `--no_display`: Non visualizzare i risultati

### Utilizzo programmatico

#### Con trasformata wavelet scattering

```python
from wavelet_lib.single_pixel_classification.processors import PixelClassificationProcessor

# Crea processore con trasformata scattering
processor = PixelClassificationProcessor(
    model_path="/path/to/model.pth",
    patch_size=32,
    stride=16,
    J=2,
    use_scattering=True  # Abilita la trasformata scattering (default)
)

# Classifica un'immagine
classification_map = processor.process_image(
    image_path="/path/to/image.jpg",
    output_path="/path/to/output/classification.png",
    overlay=True,
    alpha=0.5
)

# Visualizza risultati
processor.visualize_results(
    image_path="/path/to/image.jpg",
    classification_map=classification_map,
    output_path="/path/to/output/results.png"
)

# Crea legenda
processor.create_legend(
    output_path="/path/to/output/legend.png"
)
```

#### Senza trasformata wavelet scattering

```python
from wavelet_lib.single_pixel_classification.processors import PixelClassificationProcessor

# Crea processore senza trasformata scattering
processor = PixelClassificationProcessor(
    model_path="/path/to/model_no_scattering.pth",
    patch_size=32,
    stride=16,
    use_scattering=False  # Disabilita la trasformata scattering
)

# Classifica un'immagine
classification_map = processor.process_image(
    image_path="/path/to/image.jpg",
    output_path="/path/to/output/classification_no_scattering.png",
    overlay=True,
    alpha=0.5
)

# Visualizza risultati
processor.visualize_results(
    image_path="/path/to/image.jpg",
    classification_map=classification_map,
    output_path="/path/to/output/results_no_scattering.png"
)
```

## Utilizzo avanzato

### Personalizzazione delle classi

Puoi specificare classi personalizzate durante l'addestramento:

```bash
python script/core/pixel_classification/train_pixel_classifier.py \
    --images_dir /path/to/dataset/images \
    --masks_dir /path/to/dataset/masks \
    --model /path/to/model.pth \
    --num_classes 7 \
    --class_names "background,water,vegetation,streets,buildings,sand,mudflat"
```

### Confronto tra modalità con e senza trasformata scattering

Per valutare il vantaggio di utilizzare la trasformata wavelet scattering, puoi addestrare e confrontare modelli con e senza questa trasformazione:

```python
import matplotlib.pyplot as plt
import numpy as np

# Addestra modello con trasformata scattering
config_with_scattering = Config(
    num_channels=3,
    num_classes=5,
    scattering_order=2,
    J=2,
    shape=(32, 32)
)
config_with_scattering.use_scattering = True
model_with_scattering, scattering = create_pixel_classifier(config_with_scattering)
history_with_scattering = train_pixel_classifier(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    model_path="model_with_scattering.pth",
    use_scattering=True,
    model=model_with_scattering,
    scattering=scattering
)

# Addestra modello senza trasformata scattering
config_without_scattering = Config(
    num_channels=3,
    num_classes=5,
    shape=(32, 32)
)
config_without_scattering.use_scattering = False
model_without_scattering, _ = create_pixel_classifier(config_without_scattering)
history_without_scattering = train_pixel_classifier(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    model_path="model_without_scattering.pth",
    use_scattering=False,
    model=model_without_scattering
)

# Confronta le prestazioni
plt.figure(figsize=(12, 5))

# Confronto accuratezza
plt.subplot(1, 2, 1)
plt.plot(history_with_scattering['val_acc'], label='Con WST')
plt.plot(history_without_scattering['val_acc'], label='Senza WST')
plt.xlabel('Epoca')
plt.ylabel('Accuratezza (%)')
plt.legend()
plt.title('Confronto Accuratezza')

# Confronto loss
plt.subplot(1, 2, 2)
plt.plot(history_with_scattering['val_loss'], label='Con WST')
plt.plot(history_without_scattering['val_loss'], label='Senza WST')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.legend()
plt.title('Confronto Loss')

plt.tight_layout()
plt.savefig('confronto_wst.png')
plt.show()
```

### Supporto per immagini multibanda

Il modulo supporta immagini multibanda (più di 3 canali) in entrambe le modalità:

```python
# Configura per immagini multibanda con trasformata scattering
config = Config(
    num_channels=5,  # Ad esempio, 5 bande
    num_classes=5,
    scattering_order=2,
    J=2,
    shape=(32, 32)
)
config.use_scattering = True  # Abilita la trasformata scattering

# Crea modello per immagini multibanda
model, scattering = create_pixel_classifier(config)

# Configura per immagini multibanda senza trasformata scattering
config_no_scattering = Config(
    num_channels=5,  # Ad esempio, 5 bande
    num_classes=5,
    shape=(32, 32)
)
config_no_scattering.use_scattering = False  # Disabilita la trasformata scattering

# Crea modello per immagini multibanda senza trasformata scattering
model_no_scattering, _ = create_pixel_classifier(config_no_scattering)
```

### Ottimizzazione dei parametri

Per ottenere risultati migliori, puoi ottimizzare i parametri della trasformata wavelet scattering:

```bash
python script/core/pixel_classification/train_pixel_classifier.py \
    --images_dir /path/to/dataset/images \
    --masks_dir /path/to/dataset/masks \
    --model /path/to/model.pth \
    --j 3 \
    --scattering_order 2 \
    --patch_size 64 \
    --stride 32
```

## Risoluzione dei problemi

### Errore: "CUDA out of memory"

Riduci la dimensione del batch o la dimensione delle patch:

```bash
python script/core/pixel_classification/train_pixel_classifier.py \
    --images_dir /path/to/dataset/images \
    --masks_dir /path/to/dataset/masks \
    --model /path/to/model.pth \
    --batch_size 8 \
    --patch_size 16
```

### Classificazione di bassa qualità

Se la qualità della classificazione è bassa, prova a:
- Aumentare la dimensione delle patch
- Ridurre lo stride per un'inferenza più densa
- Aumentare il numero di epoche di addestramento
- Utilizzare un dataset più bilanciato
- Provare entrambe le modalità (con e senza trasformata scattering) per vedere quale funziona meglio per il tuo caso specifico

```bash
# Prova con trasformata scattering
python script/core/pixel_classification/train_pixel_classifier.py \
    --images_dir /path/to/dataset/images \
    --masks_dir /path/to/dataset/masks \
    --model /path/to/model_wst.pth \
    --patch_size 64 \
    --stride 32 \
    --epochs 100 \
    --j 3

# Prova senza trasformata scattering
python script/core/pixel_classification/train_pixel_classifier.py \
    --images_dir /path/to/dataset/images \
    --masks_dir /path/to/dataset/masks \
    --model /path/to/model_no_wst.pth \
    --patch_size 64 \
    --stride 32 \
    --epochs 100 \
    --no_scattering
```

### Artefatti ai bordi

Se noti artefatti ai bordi dell'immagine classificata, prova a:
- Aumentare la dimensione delle patch
- Ridurre lo stride
- Utilizzare padding durante l'inferenza

```python
# Personalizza il processore per ridurre gli artefatti ai bordi
processor = PixelClassificationProcessor(
    model_path="/path/to/model.pth",
    patch_size=64,  # Aumenta la dimensione delle patch
    stride=8,       # Riduci lo stride per un'inferenza più densa
    J=2
)
```

### Maschere non trovate

Se ricevi errori relativi a maschere non trovate, verifica che:
- I nomi delle maschere corrispondano ai nomi delle immagini (image1.jpg -> image1_mask.png)
- Le maschere siano nella directory corretta
- Le maschere siano in formato PNG o altro formato supportato
