# Wavelet Scattering Transform (WST) - Guida all'uso

Questa guida fornisce istruzioni dettagliate per l'uso del framework Wavelet Scattering Transform.

## Prerequisiti

Assicurati di aver installato tutte le dipendenze necessarie:

```bash
# Attiva l'ambiente virtuale
source wavelet_venv/bin/activate

# Installa le dipendenze
pip install -r wavelet_venv/requirements.txt

# Installa il pacchetto in modalità sviluppatore
pip install -e .
```

## Importante: Esecuzione degli script

**NOTA: Tutti gli script devono essere eseguiti dalla directory principale del progetto (wavelet/)!**

```bash
# Posizionati sempre nella directory principale del progetto
cd /home/brus/Projects/wavelet

# Poi esegui gli script dalla directory principale
python script/core/train.py [parametri]
```

## Struttura degli script

I nostri script sono organizzati cosi:

```
script/
├── core/           # Script principali
│   ├── train.py    # Addestramento dei modelli
│   ├── evaluate.py # Valutazione dei modelli
│   ├── predict.py  # Predizione con modelli
│   └── visualize.py # Visualizzazione e analisi
├── utils/          # Utility
│   ├── data_utils.py  # Utilità per dataset
│   └── model_utils.py # Utilità per modelli
└── notebooks/      # Jupyter notebooks di esempio
```

## Addestramento di un modello

Per addestrare un nuovo modello, esegui:

```bash
python script/core/train.py --dataset /path/to/dataset --num-classes N --epochs E --balance
```

Parametri principali:
- `--dataset`: Percorso alla directory del dataset (con sottodirectory per classe)
- `--num-classes`: Numero di classi (default: 4)
- `--epochs`: Numero di epoche di addestramento (default: 90)
- `--balance`: Bilancia le classi nel dataset (opzionale)
- `--output-dir`: Directory per salvare i risultati (opzionale)

Esempio:
```bash
python script/core/train.py --dataset /home/brus/Projects/wavelet/datasets/HPL_images/custom_dataset --num-classes 4 --epochs 40 --balance
```

## Predizione su nuove immagini

Per fare predizioni usando un modello addestrato:

```bash
python script/core/predict.py --model-path /path/to/model.pth --image-path /path/to/image.jpg
```

Per la classificazione a tile di immagini grandi:

```bash
python script/core/predict.py --model-path /path/to/model.pth --image-path /path/to/image.jpg --tile-mode --tile-size 32
```

## Valutazione di un modello

Per valutare completamente le prestazioni di un modello su un dataset:

```bash
python script/core/evaluate.py --model-path /path/to/model.pth --dataset /path/to/dataset
```

Parametri principali:
- `--model-path`: Percorso al file del modello addestrato
- `--dataset`: Percorso alla directory del dataset (con sottodirectory per classe)
- `--balance`: Bilancia le classi nel dataset per la valutazione (opzionale)
- `--output-dir`: Directory per salvare i risultati (opzionale)

Esempio:
```bash
python script/core/evaluate.py --model-path models/model_output_4/best_model.pth --dataset datasets/HPL_images/custom_dataset --balance
```

## Visualizzazione e analisi

Lo script di visualizzazione supporta diversi comandi:

```bash
# Visualizzazione delle metriche di addestramento
python script/core/visualize.py metrics --model-dir models/model_output_4_classes_YYYYMMDD_HHMMSS

# Visualizzazione di campioni del dataset con predizioni
python script/core/visualize.py samples --model-path /path/to/model.pth --dataset /path/to/dataset

# Visualizzazione distribuzione delle classi
python script/core/visualize.py distribution --dataset /path/to/dataset
```

## Utility per dataset e modelli

Per analisi avanzate, abbiamo anche utility specifiche:

```bash
# Analisi di un dataset
python -c "from script.utils.data_utils import analyze_dataset; analyze_dataset('/path/to/dataset')"

# Analisi di un modello
python -c "from script.utils.model_utils import analyze_model; analyze_model('/path/to/model.pth')"
```

## Struttura comune di un dataset

Il dataset deve avere la seguente struttura:

```
dataset/
├── classe1/
│   ├── immagine1.jpg
│   ├── immagine2.jpg
│   └── ...
├── classe2/
│   ├── immagine1.jpg
│   ├── immagine2.jpg
│   └── ...
└── ...
```

## Flusso di lavoro tipico

1. Prepara il dataset nella struttura corretta
2. Addestra il modello:
   ```bash
   python script/core/train.py --dataset /path/to/dataset --num-classes 4 --epochs 40 --balance
   ```

3. Visualizza le metriche di addestramento:
   ```bash
   python script/core/visualize.py metrics --model-dir models/model_output_4_classes_YYYYMMDD_HHMMSS
   ```

4. Valuta il modello sul dataset:
   ```bash
   python script/core/evaluate.py --model-path models/model_output_4_classes_YYYYMMDD_HHMMSS/best_model.pth --dataset /path/to/dataset
   ```

5. Fai predizioni su nuove immagini:
   ```bash
   python script/core/predict.py --model-path models/model_output_4_classes_YYYYMMDD_HHMMSS/best_model.pth --image-path /path/to/new_image.jpg
   ```

## Risoluzione dei problemi

### Errore: "No such file or directory"
Se riscontri errori del tipo "No such file or directory", verifica di:
1. Eseguire gli script dalla directory principale del progetto (wavelet/)
2. Fornire percorsi assoluti corretti ai dataset e ai file di modello
3. Avere attivato l'ambiente virtuale corretto

Esempio di esecuzione corretta:
```bash
cd /home/brus/Projects/wavelet
python script/core/train.py --dataset /home/brus/Projects/wavelet/datasets/HPL_images/custom_dataset --num-classes 4
```

### Errore: "shape '[-1, 12, 8, 8]' is invalid for input of size..."
Questo errore può verificarsi quando c'è una discrepanza tra la dimensione dell'output della trasformata scattering e quella attesa dal modello. Il problema è stato risolto nella versione più recente, ma se lo riscontri ancora:
1. Assicurati di utilizzare gli stessi parametri scattering (J=2, shape=(32, 32), max_order=2) per addestramento e inferenza
2. Verifica che le immagini abbiano la dimensione corretta (32x32 pixel)
3. Utilizza lo script `script/utility/dataset_inspector.py` per verificare la compatibilità del dataset:
```bash
python script/utility/dataset_inspector.py --dataset /path/to/dataset --expected-dims 32x32
```

### Errore: "CUDA out of memory"
Per problemi di memoria, prova a ridurre il batch size:
```bash
python script/core/train.py --dataset /path/to/dataset --batch-size 64
```

### Errore: "ModuleNotFoundError: No module named 'torch'"
Se ricevi questo errore, assicurati di:
1. Aver attivato l'ambiente virtuale: `source wavelet_venv/bin/activate`
2. Aver installato tutte le dipendenze: `pip install -r wavelet_venv/requirements.txt`
3. Aver installato il pacchetto in modalità sviluppatore: `pip install -e .`