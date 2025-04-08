# Guida all'utilizzo del modulo Single Tile Classification

Questo documento fornisce una guida dettagliata all'utilizzo del modulo `single_tile_classification` della libreria Wavelet Scattering Transform.

## Indice
1. [Introduzione](#introduzione)
2. [Installazione delle dipendenze](#installazione-delle-dipendenze)
3. [Preparazione del dataset](#preparazione-del-dataset)
4. [Addestramento di un modello di classificazione](#addestramento-di-un-modello-di-classificazione)
5. [Classificazione di immagini](#classificazione-di-immagini)
6. [Utilizzo avanzato](#utilizzo-avanzato)
7. [Risoluzione dei problemi](#risoluzione-dei-problemi)

## Introduzione

Il modulo `single_tile_classification` è progettato per classificare immagini (tiles) utilizzando la trasformata wavelet scattering. È particolarmente efficace per la classificazione di immagini satellitari o aeree, dove le caratteristiche di texture e scala sono importanti.

Il modulo utilizza la trasformata wavelet scattering per estrarre caratteristiche robuste dalle immagini, che vengono poi utilizzate per addestrare un classificatore. Questo approccio è particolarmente efficace per dataset di piccole dimensioni e per immagini con pattern ripetitivi.

## Installazione delle dipendenze

Prima di utilizzare il modulo, assicurati di avere installato tutte le dipendenze necessarie:

```bash
pip install torch torchvision kymatio opencv-python matplotlib scikit-learn tqdm
```

## Preparazione del dataset

Il modulo `single_tile_classification` si aspetta un dataset organizzato in una struttura di directory specifica, con una cartella per ogni classe.

### Struttura del dataset

```
dataset/
├── classe1/
│   ├── immagine1.jpg
│   ├── immagine2.jpg
│   └── ...
├── classe2/
│   ├── immagine1.jpg
│   └── ...
└── ...
```

### Estrazione di tiles da immagini più grandi

Se hai immagini di grandi dimensioni, puoi utilizzare lo strumento `extract_tiles` per estrarre tiles di dimensioni fisse:

```python
from wavelet_lib.utils.tile_extractor import extract_tiles

# Estrai tiles da un'immagine
num_tiles = extract_tiles(
    input_image_path="/path/to/large_image.jpg",
    output_dir="/path/to/dataset/classe1",
    tile_size=32,
    tiles_per_subwin=30
)

print(f"Estratti {num_tiles} tiles")
```

### Creazione di un dataset bilanciato

Per ottenere risultati migliori, è consigliabile utilizzare un dataset bilanciato, con lo stesso numero di immagini per ogni classe:

```python
from wavelet_lib.dataset_tools.data_utils import extract_balanced_dataset

# Crea un dataset bilanciato
stats = extract_balanced_dataset(
    dataset_path="/path/to/unbalanced_dataset",
    output_path="/path/to/balanced_dataset",
    samples_per_class=100  # Numero di campioni per classe
)

print(f"Dataset bilanciato creato: {stats}")
```

## Addestramento di un modello di classificazione

Una volta preparato il dataset, puoi addestrare un modello di classificazione.

### Utilizzo base

```python
import torch
from wavelet_lib.base import Config
from wavelet_lib.single_tile_classification.models import create_classification_model
from wavelet_lib.training import train_model

# Configura i parametri
config = Config(
    num_channels=3,  # RGB
    num_classes=4,   # Numero di classi nel dataset
    scattering_order=2,
    J=2,
    shape=(32, 32),  # Dimensione delle immagini
    batch_size=64,
    epochs=50,
    learning_rate=0.01
)

# Crea modello e trasformata scattering
model, scattering = create_classification_model(config)

# Addestra il modello
train_model(
    model=model,
    scattering=scattering,
    train_dir="/path/to/train_dataset",
    test_dir="/path/to/test_dataset",
    config=config,
    save_path="/path/to/save/model.pth"
)
```

### Parametri di addestramento

- `num_channels`: Numero di canali nelle immagini (3 per RGB, 1 per scala di grigi)
- `num_classes`: Numero di classi nel dataset
- `scattering_order`: Ordine della trasformata scattering (1 o 2)
- `J`: Numero di scale per la trasformata scattering
- `shape`: Dimensione delle immagini di input
- `batch_size`: Dimensione del batch per l'addestramento
- `epochs`: Numero di epoche di addestramento
- `learning_rate`: Learning rate per l'ottimizzatore

### Utilizzo avanzato

Per un controllo più fine sull'addestramento, puoi utilizzare la classe `Trainer`:

```python
from wavelet_lib.training import Trainer
import torch.optim as optim

# Crea ottimizzatore
optimizer = optim.SGD(
    model.parameters(),
    lr=config.learning_rate,
    momentum=config.momentum,
    weight_decay=config.weight_decay
)

# Crea scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.epochs
)

# Crea trainer
trainer = Trainer(
    model=model,
    scattering=scattering,
    device=config.device,
    optimizer=optimizer,
    scheduler=scheduler
)

# Addestra il modello con più controllo
trainer.train(
    train_dir="/path/to/train_dataset",
    test_dir="/path/to/test_dataset",
    batch_size=config.batch_size,
    epochs=config.epochs,
    save_path="/path/to/save/model.pth",
    class_to_idx=None,  # Verrà determinato automaticamente dal dataset
    reduce_lr_after=20  # Riduce il learning rate ogni 20 epoche
)
```

## Classificazione di immagini

Dopo aver addestrato un modello, puoi utilizzarlo per classificare nuove immagini.

### Classificazione di una singola immagine

```python
from wavelet_lib.single_tile_classification.processors import ClassificationProcessor
from wavelet_lib.single_tile_classification.models import create_scattering_transform

# Carica il modello addestrato
processor = ClassificationProcessor(
    model_path="/path/to/model.pth",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Classifica un'immagine
result = processor.process_image("/path/to/image.jpg")

print(f"Classe predetta: {result['class']}")
print(f"Confidenza: {result['confidence']:.2f}")
```

### Classificazione di un batch di immagini

```python
# Classifica un batch di immagini
results = processor.process_batch([
    "/path/to/image1.jpg",
    "/path/to/image2.jpg",
    "/path/to/image3.jpg"
])

for i, result in enumerate(results):
    print(f"Immagine {i+1}: Classe {result['class']} con confidenza {result['confidence']:.2f}")
```

### Classificazione di tiles da un'immagine più grande

```python
# Classifica tiles estratti da un'immagine più grande
results = processor.process_large_image(
    image_path="/path/to/large_image.jpg",
    tile_size=32,
    overlap=0,  # Nessuna sovrapposizione tra i tiles
    process_30x30_tiles=False  # Imposta a True per immagini 30x30
)

# Visualizza i risultati
processor.visualize_large_image_results(
    results,
    output_path="/path/to/output.png"
)
```

## Utilizzo avanzato

### Personalizzazione del modello

Puoi personalizzare l'architettura del modello modificando il tipo di classificatore:

```python
from wavelet_lib.single_tile_classification.models import ScatteringClassifier

# Crea un classificatore personalizzato
model = ScatteringClassifier(
    in_channels=12,  # Numero di canali dopo la trasformata scattering
    classifier_type='mlp',  # 'cnn', 'mlp', o 'linear'
    num_classes=4
)
```

### Analisi delle prestazioni del modello

```python
from wavelet_lib.single_tile_classification.models import print_classifier_summary

# Stampa un riepilogo delle prestazioni del modello
print_classifier_summary(
    model_path="/path/to/model.pth",
    test_dir="/path/to/test_dataset",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

### Utilizzo con immagini multibanda

Il modulo supporta anche immagini multibanda (più di 3 canali):

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

## Risoluzione dei problemi

### Errore: "CUDA out of memory"

Riduci la dimensione del batch:

```python
config.batch_size = 32  # Riduci da 64 a 32
```

### Errore: "Expected 3-dimensional input for 3-dimensional weight"

Assicurati che le dimensioni delle immagini corrispondano a quelle attese dal modello:

```python
# Verifica le dimensioni delle immagini nel dataset
from wavelet_lib.dataset_tools.data_utils import analyze_image_sizes

stats = analyze_image_sizes("/path/to/dataset")
print(f"Statistiche delle dimensioni: {stats}")
```

### Prestazioni di classificazione scarse

Prova a modificare i parametri della trasformata scattering:

```python
# Aumenta il numero di scale
config.J = 3

# Aumenta l'ordine della trasformata
config.scattering_order = 2

# Ricrea il modello
model, scattering = create_classification_model(config)
```

Oppure prova ad aumentare il numero di epoche di addestramento:

```python
config.epochs = 100
config.learning_rate = 0.005  # Riduci il learning rate per un addestramento più stabile
```

### Overfitting

Se il modello ha buone prestazioni sul training set ma scarse sul test set, prova ad applicare regolarizzazione:

```python
config.weight_decay = 1e-4  # Aumenta il weight decay

# Oppure usa dropout nel modello
model = ScatteringClassifier(
    in_channels=12,
    classifier_type='cnn',
    num_classes=4,
    dropout_rate=0.5  # Aggiungi dropout
)
```
