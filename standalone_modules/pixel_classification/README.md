# Modulo di Classificazione Pixel-wise con Wavelet Scattering Transform

Questo modulo autonomo fornisce funzionalità complete per la classificazione pixel-wise di immagini utilizzando la trasformata scattering wavelet.

## Caratteristiche

- **Trasformata Scattering Wavelet**: Utilizza la libreria Kymatio per estrarre caratteristiche robuste dalle immagini
- **Classificazione pixel-wise**: Classifica ogni pixel dell'immagine in categorie predefinite
- **Gestione efficiente della memoria**: Supporto per lazy loading e caching dei metadati
- **Visualizzazione avanzata**: Strumenti per visualizzare i risultati della classificazione e le metriche di addestramento
- **Addestramento flessibile**: Configurazione completa dei parametri di addestramento
- **Supporto per GPU**: Ottimizzazioni per l'addestramento su GPU, inclusa la precisione mista automatica

## Installazione

### Prerequisiti

- Python 3.8+
- PyTorch 1.8+
- Kymatio 0.3+

### Dipendenze

```bash
pip install torch torchvision tqdm matplotlib numpy pillow kymatio albumentations scikit-learn opencv-python
```

## Struttura del modulo

```
pixel_classification/
├── __init__.py          # Esporta le funzioni e le classi principali
├── models.py            # Definizioni dei modelli neurali
├── dataset.py           # Gestione dei dataset
├── utils.py             # Funzioni di utilità
├── visualization.py     # Funzioni di visualizzazione
├── tools.py             # Strumenti aggiuntivi per dataset e modelli
├── train.py             # Script di addestramento
├── predict.py           # Script di predizione
├── test.py              # Script di test
└── README.md            # Documentazione
```

## Utilizzo

### Addestramento di un modello

```bash
python train.py --images_dir /path/to/images --masks_dir /path/to/masks --model /path/to/model.pth
```

Opzioni principali:
- `--images_dir`: Directory contenente le immagini di training (obbligatorio)
- `--masks_dir`: Directory contenente le maschere di classe (obbligatorio)
- `--model`: Percorso dove salvare il modello (obbligatorio)
- `--patch_size`: Dimensione delle patch (default: 32)
- `--stride`: Passo per l'estrazione delle patch (default: 16)
- `--batch_size`: Dimensione del batch (default: 16)
- `--epochs`: Numero di epoche (default: 50)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--val_split`: Frazione dei dati da usare per la validazione (default: 0.2)
- `--no_scattering`: Disabilita la trasformata scattering
- `--lazy_loading`: Carica le patch solo quando necessario (riduce l'uso di memoria)
- `--metadata_cache`: File per salvare/caricare i metadati delle patch
- `--resume`: Riprendi l'addestramento da un checkpoint esistente

### Predizione con un modello addestrato

```bash
python predict.py --model /path/to/model.pth --image /path/to/image.jpg
```

Opzioni principali:
- `--model`: Percorso del modello addestrato (obbligatorio)
- `--image`: Percorso dell'immagine da classificare (obbligatorio)
- `--mask`: Percorso della maschera di verità (opzionale)
- `--output`: Percorso dove salvare il risultato
- `--patch_size`: Dimensione delle patch (default: 32)
- `--stride`: Passo per l'estrazione delle patch (default: 16)
- `--max_size`: Dimensione massima dell'immagine (ridimensiona se più grande)

### Test di un modello

```bash
python test.py --model /path/to/model.pth --images_dir /path/to/images --masks_dir /path/to/masks
```

Opzioni principali:
- `--model`: Percorso del modello addestrato (obbligatorio)
- `--images_dir`: Directory contenente le immagini di test
- `--masks_dir`: Directory contenente le maschere di verità
- `--image`: Percorso di un'immagine singola da testare
- `--mask`: Percorso della maschera di verità per l'immagine singola
- `--output_dir`: Directory dove salvare i risultati
- `--max_images`: Numero massimo di immagini da testare

## Struttura del dataset

Il dataset deve essere organizzato nella seguente struttura:

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

Le maschere devono essere immagini in scala di grigi dove ogni valore di pixel rappresenta una classe (0 = sfondo, 1 = classe 1, ecc.).

## Ottimizzazioni per la memoria

Il modulo offre diverse ottimizzazioni per ridurre l'uso di memoria:

1. **Lazy loading**: Carica le patch solo quando necessario
   ```bash
   python train.py --lazy_loading --max_patches_in_memory 50000 ...
   ```

2. **Caching dei metadati**: Salva/carica i metadati delle patch per evitare di rielaborare le immagini
   ```bash
   python train.py --metadata_cache /path/to/metadata.pth --save_metadata ...
   ```

3. **Limitazione del numero di immagini**: Elabora solo un sottoinsieme delle immagini
   ```bash
   python train.py --max_images 100 ...
   ```

## Ottimizzazioni per la GPU

Il modulo offre diverse ottimizzazioni per l'addestramento su GPU:

1. **Precisione mista automatica**: Accelera l'addestramento utilizzando la precisione mista
   ```bash
   # Abilitata di default, disabilitala con:
   python train.py --no_amp ...
   ```

2. **Configurazione di cuDNN**: Ottimizza la stabilità dell'addestramento
   ```bash
   # Disabilita completamente cuDNN in caso di errori:
   python train.py --disable_cudnn ...
   ```

## Esempi

### Addestramento di un modello con ottimizzazioni di memoria

```bash
python train.py \
    --images_dir /path/to/images \
    --masks_dir /path/to/masks \
    --model /path/to/model.pth \
    --lazy_loading \
    --max_patches_in_memory 50000 \
    --metadata_cache /path/to/metadata.pth \
    --save_metadata \
    --batch_size 16 \
    --epochs 50
```

### Ripresa dell'addestramento da un checkpoint

```bash
python train.py \
    --images_dir /path/to/images \
    --masks_dir /path/to/masks \
    --model /path/to/model.pth \
    --resume \
    --lazy_loading \
    --metadata_cache /path/to/metadata.pth
```

### Predizione su un'immagine

```bash
python predict.py \
    --model /path/to/model.pth \
    --image /path/to/image.jpg \
    --output /path/to/result.png
```

### Test su un dataset

```bash
python test.py \
    --model /path/to/model.pth \
    --images_dir /path/to/test_images \
    --masks_dir /path/to/test_masks \
    --output_dir /path/to/results
```

## Strumenti aggiuntivi

Il modulo include diversi strumenti utili per l'analisi dei dataset e la gestione dei modelli:

### Analisi del dataset

```bash
# Analisi interattiva di un dataset
python -c "from pixel_classification.tools import analyze_dataset; analyze_dataset('/path/to/images', '/path/to/masks')"
```

### Estrazione di tile

```bash
# Estrazione automatica di tile da un dataset
python -c "from pixel_classification.tools import extract_tiles_batch; extract_tiles_batch('/path/to/images', '/path/to/output', tile_size=32, stride=16)"

# Estrazione interattiva di tile da un'immagine
python -c "from pixel_classification.tools import interactive_tile_selection; interactive_tile_selection('/path/to/image.jpg', '/path/to/output')"
```

### Analisi del modello

```bash
# Analisi di un modello addestrato
python -c "from pixel_classification.tools import analyze_model; analyze_model('/path/to/model.pth')"
```

## Utilizzo programmatico

```python
from pixel_classification.models import create_pixel_classifier
from pixel_classification.utils import Config, load_model
from pixel_classification.dataset import PixelWiseDataset
from pixel_classification.tools import analyze_dataset, extract_tiles, analyze_model

# Crea una configurazione
config = Config(num_classes=5, J=2, scattering_order=2)

# Crea un modello
model, scattering = create_pixel_classifier(config)

# Carica un modello addestrato
checkpoint = load_model('path/to/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Crea un dataset
dataset = PixelWiseDataset(
    images_dir='path/to/images',
    masks_dir='path/to/masks',
    patch_size=32,
    stride=16,
    lazy_loading=True
)

# Addestra il modello
from pixel_classification.models import train_pixel_classifier
history = train_pixel_classifier(
    train_dataset=dataset,
    model_path='path/to/model.pth',
    batch_size=16,
    num_epochs=50
)

# Analizza un dataset
stats = analyze_dataset('path/to/images', 'path/to/masks')

# Estrai tile da un'immagine
tiles = extract_tiles('path/to/image.jpg', 'path/to/output', tile_size=32, stride=16)

# Analizza un modello
model_info = analyze_model('path/to/model.pth')
```

## Licenza

Questo modulo è distribuito con licenza MIT.

## Riconoscimenti

Questo modulo è basato sul lavoro originale del progetto Wavelet e utilizza la libreria Kymatio per la trasformata scattering wavelet.
