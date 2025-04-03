# Utilizzo della Libreria Wavelet con Immagini Multibanda

Questa guida spiega come utilizzare la libreria wavelet con immagini multibanda (fino a 10 canali).

## Configurazione

### Creazione di una configurazione multibanda

```python
from wavelet_lib.base import Config

# Crea una configurazione per immagini con 4 bande
config = Config(
    num_channels=4,  # Numero di canali nell'immagine
    num_classes=5,   # Numero di classi da classificare
    J=2,             # Parametro J per la trasformata wavelet
    shape=(32, 32)   # Dimensione dell'input
)

# La libreria verifica automaticamente che il numero di canali non superi il massimo supportato (10)
# config = Config(num_channels=12)  # Solleverà un errore ValueError
```

## Caricamento di immagini multibanda

La libreria ora supporta nativamente il caricamento di immagini multibanda utilizzando `rasterio`:

```python
from wavelet_lib.processors import ImageProcessor
from wavelet_lib.models import create_model

# Crea modello e scattering transform
model, scattering = create_model(config)

# Crea un processore di immagini
processor = ImageProcessor(model, scattering, config.device)

# Elabora un'immagine multibanda
result = processor.process_image('path/to/multiband_image.tif')
print(f"Classe predetta: {result['class_name']}, Confidenza: {result['confidence']:.2f}")
```

## Creazione di un dataset personalizzato

Per lavorare con un dataset di immagini multibanda:

```python
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders

# Crea una trasformazione adatta per immagini con 4 canali
transform = get_default_transform(
    target_size=(32, 32),
    normalize=True,
    num_channels=4  # Specifica il numero di canali
)

# Crea un dataset bilanciato
dataset = BalancedDataset(
    root='path/to/dataset',
    transform=transform,
    balance=True
)

# Crea i data loader per addestramento e test
train_loader, test_loader = create_data_loaders(
    dataset, 
    test_size=0.2,
    batch_size=32
)
```

## Addestramento di un modello per immagini multibanda

```python
from wavelet_lib.training import train_model

# Addestra il modello sulle immagini multibanda
train_model(
    model=model,
    scattering=scattering,
    train_loader=train_loader,
    test_loader=test_loader,
    config=config
)
```

## Visualizzazione

Per visualizzare le immagini multibanda:

```python
from wavelet_lib.visualization import plot_multiband_image

# Visualizza un'immagine multibanda (selezionando i canali da visualizzare)
plot_multiband_image(
    image_path='path/to/multiband_image.tif',
    channels=[0, 1, 2],  # Mostra i primi tre canali come RGB
    figsize=(10, 10)
)

# Visualizza tutti i canali separatamente
plot_multiband_image(
    image_path='path/to/multiband_image.tif',
    separate=True,  # Mostra ogni canale separatamente
    figsize=(15, 10)
)
```

## Esempio completo

```python
import torch
from wavelet_lib.base import Config, set_seed
from wavelet_lib.models import create_model
from wavelet_lib.datasets import BalancedDataset, get_default_transform, create_data_loaders
from wavelet_lib.training import train_model
from wavelet_lib.processors import ImageProcessor

# Imposta il seed per la riproducibilità
set_seed(42)

# Crea configurazione per 5 canali
config = Config(
    num_channels=5,
    num_classes=4,
    J=2,
    shape=(32, 32),
    batch_size=64,
    epochs=50
)

# Crea modello e scattering transform
model, scattering = create_model(config)

# Crea transform con normalizzazione adeguata per 5 canali
transform = get_default_transform(
    target_size=(32, 32),
    normalize=True,
    num_channels=5
)

# Crea dataset e data loader
dataset = BalancedDataset('path/to/dataset', transform=transform)
train_loader, test_loader = create_data_loaders(dataset, test_size=0.2, batch_size=config.batch_size)

# Addestra il modello
train_model(model, scattering, train_loader, test_loader, config)

# Salva il modello addestrato
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config
}, 'multiband_model.pth')

# Utilizza il modello per predire nuove immagini
processor = ImageProcessor(model, scattering, config.device)
result = processor.process_image('path/to/new_image.tif')
print(f"Classe predetta: {result['class_name']}")
print(f"Confidenza: {result['confidence']:.4f}")
```

## Note tecniche

1. **Normalizzazione**: La libreria ora genera automaticamente parametri di normalizzazione basati sul numero di canali in input.
2. **Compatibilità**: La libreria mantiene la compatibilità retroattiva con immagini RGB standard.
3. **Formato file**: Per immagini multibanda si consiglia di utilizzare il formato TIFF (GeoTIFF).
4. **Performance**: Aumentare il numero di canali incrementa il numero di coefficienti di scattering e quindi la complessità computazionale.