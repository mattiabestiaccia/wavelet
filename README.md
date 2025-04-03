# Wavelet Scattering Transform (WST) Image Classification

A modular framework for image classification using Wavelet Scattering Transform representations. The framework provides tools for training, evaluating, and deploying models on various types of image data, including multiband images with up to 10 channels.

## Features

- Wavelet Scattering Transform based feature extraction
- Support for various neural network architectures
- Balanced dataset handling
- Training and evaluation tools
- Tile-based image classification for large images
- Visualization tools for model analysis
- Support for multiband imagery (up to 10 channels)
- Tools for creating and visualizing multiband images

## Installation

1. Clone the repository:

```bash
git clone https://github.com/mattiabestiaccia/wavelet.git
cd wavelet
```

2. Create a virtual environment and install requirements:

```bash
python3 -m venv wavelet_venv
source wavelet_venv/bin/activate
pip install -r wavelet_venv/requirements.txt
```

3. Install the library in development mode:

```bash
# Addestramento con gestione esperimenti
python script/core/train.py \
    --dataset /path/to/dataset \
    --num-classes 4 \
    --epochs 90 \
    --output-base experiments/dataset0 \
    --experiment-name first_run

# Valutazione nell'ambito dell'esperimento
python script/core/evaluate.py \
    --model-path experiments/dataset0/models/first_run/best_model.pth \
    --dataset /path/to/dataset \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

## Struttura degli Esperimenti

Ogni esperimento è organizzato come segue:

```
experiments/dataset0/
├── classification_result/    # Risultati classificazione
├── dataset_info/            # Statistiche dataset
├── evaluation/              # Metriche valutazione
├── models/                  # Checkpoint modelli
├── model_output/           # Output training
├── visualization/          # Visualizzazioni
└── README.md               # Documentazione esperimento
```

Per istruzioni dettagliate sull'utilizzo, consultare il file [USAGE.md](USAGE.md).

## Training a Model

Per addestrare un nuovo modello all'interno di un esperimento, utilizzare lo script `train.py`:

```bash
python script/core/train.py \
    --dataset /path/to/dataset \
    --num-classes 4 \
    --epochs 90 \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

Parametri di training aggiuntivi:

- `--balance`: Bilancia la distribuzione delle classi
- `--num-channels`: Numero di canali in input (default: 3, max: 10)
- `--batch-size`: Dimensione del batch (default: 128)
- `--lr`: Learning rate (default: 0.1)
- `--device`: Device da utilizzare (cuda o cpu)
- `--output-base`: Directory base per l'esperimento
- `--experiment-name`: Nome dell'esperimento

## Making Predictions

Per fare predizioni usando un modello addestrato all'interno di un esperimento:

```bash
python script/core/predict.py \
    --model-path experiments/dataset0/models/first_run/best_model.pth \
    --image-path /path/to/image.jpg \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

Per la classificazione tile-based di immagini grandi:

```bash
python script/core/predict.py \
    --model-path experiments/dataset0/models/first_run/best_model.pth \
    --image-path /path/to/image.jpg \
    --tile-mode \
    --tile-size 32 \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

Parametri di predizione aggiuntivi:

- `--tile-mode`: Abilita la classificazione tile-based
- `--tile-size`: Dimensione dei tile (default: 32)
- `--confidence-threshold`: Soglia di confidenza per la visualizzazione (default: 0.7)
- `--device`: Device da utilizzare (cuda o cpu)
- `--output-base`: Directory base per l'esperimento
- `--experiment-name`: Nome dell'esperimento

## Evaluation

Per valutare un modello su un dataset di test all'interno di un esperimento:

```bash
python script/core/evaluate.py \
    --model-path experiments/dataset0/models/first_run/best_model.pth \
    --dataset /path/to/dataset \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

## Visualization

Per visualizzare le metriche di training da un esperimento:

```bash
python script/core/visualize.py metrics \
    --model-dir experiments/dataset0/model_output/first_run
```

## Dataset Inspection

Per verificare se un dataset è formattato correttamente per il modello:

```bash
python script/utility/dataset_inspector.py --dataset /path/to/dataset --expected-dims 32x32
```

## Example Workflow

1. Prepara il tuo dataset con le sottodirectory delle classi:

```
dataset/
├── class1/
│   ├── image1.jpg           # Immagini con 3 canali (RGB)
│   ├── image2.tif           # Immagini multibanda (fino a 10 canali)
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.tif
│   └── ...
└── ...
```

È possibile utilizzare immagini RGB standard o immagini multibanda in formato TIFF con fino a 10 canali.

2. Verifica se il dataset è adatto al modello:

```bash
python script/utility/dataset_inspector.py --dataset /path/to/dataset --expected-dims 32x32
```

3. Addestra un modello in un nuovo esperimento:

```bash
python script/core/train.py \
    --dataset /path/to/dataset \
    --num-classes 4 \
    --epochs 90 \
    --balance \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

4. Visualizza le metriche di training:

```bash
python script/core/visualize.py metrics \
    --model-dir experiments/dataset0/model_output/first_run
```

5. Valuta il modello sui dati di test:

```bash
python script/core/evaluate.py \
    --model-path experiments/dataset0/models/first_run/best_model.pth \
    --dataset /path/to/dataset \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

6. Fai predizioni su nuove immagini:

```bash
python script/core/predict.py \
    --model-path experiments/dataset0/models/first_run/best_model.pth \
    --image-path /path/to/test_image.jpg \
    --output-base experiments/dataset0 \
    --experiment-name first_run
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- Torchvision 0.8+
- Kymatio 0.2+
- NumPy
- Matplotlib
- scikit-learn
- Pillow
- tqdm
- rasterio (per immagini multibanda)

## Supporto per Immagini Multibanda

Per l'utilizzo con immagini multibanda, consultare il file [MULTIBAND_USAGE.md](MULTIBAND_USAGE.md) per guide dettagliate e esempi.

## Credits

- PyTorch Scattering: https://github.com/kymatio/kymatio
- PyTorch: https://pytorch.org/
- Rasterio: https://rasterio.readthedocs.io/
