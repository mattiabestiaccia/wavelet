# Guida all'utilizzo del modulo Single Tile Segmentation

Questo documento fornisce una guida dettagliata all'utilizzo del modulo `single_tile_segmentation` della libreria Wavelet Scattering Transform.

## Indice

1. [Introduzione](#introduzione)
2. [Installazione delle dipendenze](#installazione-delle-dipendenze)
3. [Preparazione del dataset](#preparazione-del-dataset)
4. [Addestramento di un modello di segmentazione](#addestramento-di-un-modello-di-segmentazione)
5. [Segmentazione di immagini](#segmentazione-di-immagini)
6. [Utilizzo avanzato](#utilizzo-avanzato)
7. [Risoluzione dei problemi](#risoluzione-dei-problemi)

## Introduzione

Il modulo `single_tile_segmentation` è progettato per segmentare immagini (tiles) utilizzando la trasformata wavelet scattering come preprocessore. È particolarmente efficace per la segmentazione di immagini satellitari o aeree, dove le caratteristiche di texture e scala sono importanti per identificare diverse regioni.

Il modulo implementa un'architettura U-Net arricchita con la trasformata wavelet scattering, che migliora la capacità del modello di catturare caratteristiche a diverse scale. Questo approccio è particolarmente efficace per la segmentazione di oggetti con bordi complessi o texture distintive.

## Installazione delle dipendenze

Prima di utilizzare il modulo, assicurati di avere installato tutte le dipendenze necessarie:

```bash
pip install torch torchvision kymatio opencv-python matplotlib scikit-learn tqdm albumentations
```

## Preparazione del dataset

Il modulo `single_tile_segmentation` richiede un dataset composto da immagini e relative maschere di segmentazione.

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

Le maschere devono essere immagini binarie dove i pixel bianchi (valore 255) rappresentano l'oggetto da segmentare e i pixel neri (valore 0) rappresentano lo sfondo.

### Creazione di maschere di segmentazione

Se non disponi già di maschere di segmentazione, puoi crearle utilizzando lo strumento `create_mask`:

```python
from wavelet_lib.utils.mask_generator import create_mask

# Crea una maschera per un'immagine
mask_path = create_mask(
    input_image_path="/path/to/image.jpg",
    output_dir="/path/to/masks",
    output_filename="image_mask.png",
    tile_size=32,
    tiles_per_subwin=30
)

print(f"Maschera creata: {mask_path}")
```

Questo strumento apre un'interfaccia interattiva che ti permette di selezionare manualmente le aree da includere nella maschera.

## Addestramento di un modello di segmentazione

Una volta preparato il dataset, puoi addestrare un modello di segmentazione.

### Utilizzo dello script da riga di comando

```bash
python script/core/segmentation/train_segmentation.py \
    --imgs_dir /path/to/dataset/images \
    --masks_dir /path/to/dataset/masks \
    --model /path/to/save/model.pth \
    --input_size 256,256 \
    --batch_size 8 \
    --epochs 50 \
    --j 2
```

### Parametri

- `--imgs_dir`: Directory contenente le immagini di training
- `--masks_dir`: Directory contenente le maschere di segmentazione
- `--model`: Percorso dove salvare il modello
- `--input_size`: Dimensione di input per il modello (altezza,larghezza)
- `--batch_size`: Dimensione del batch per l'addestramento
- `--epochs`: Numero di epoche di addestramento
- `--j`: Numero di scale per la trasformata scattering
- `--val_split`: Frazione dei dati da usare per la validazione (default: 0.2)
- `--learning_rate`: Learning rate per l'ottimizzatore (default: 1e-4)
- `--seed`: Seed per la riproducibilità (default: 42)
- `--log_dir`: Directory per i log di training (opzionale)

### Utilizzo programmatico

```python
from wavelet_lib.single_tile_segmentation.models import train_segmentation_model
import glob

# Trova tutte le immagini e le maschere
train_images = sorted(glob.glob("/path/to/dataset/images/*.jpg"))
train_masks = sorted(glob.glob("/path/to/dataset/masks/*.png"))

# Dividi in training e validazione
from sklearn.model_selection import train_test_split
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    train_images, train_masks, test_size=0.2, random_state=42
)

# Addestra il modello
history = train_segmentation_model(
    train_images=train_imgs,
    train_masks=train_masks,
    val_images=val_imgs,
    val_masks=val_masks,
    model_path="/path/to/save/model.pth",
    J=2,
    input_shape=(256, 256),
    batch_size=8,
    num_epochs=50,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Visualizza la curva di loss
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig("/path/to/loss_curve.png")
plt.show()
```

## Segmentazione di immagini

Dopo aver addestrato un modello, puoi utilizzarlo per segmentare nuove immagini.

### Utilizzo dello script da riga di comando

```bash
python script/core/segmentation/run_segmentation.py \
    --image /path/to/image.jpg \
    --model /path/to/model.pth \
    --output /path/to/output_dir \
    --threshold 0.5 \
    --overlay
```

Per processare una cartella di immagini:

```bash
python script/core/segmentation/run_segmentation.py \
    --folder /path/to/images \
    --model /path/to/model.pth \
    --output /path/to/output_dir \
    --threshold 0.5 \
    --overlay
```

### Parametri

- `--image`: Percorso dell'immagine da segmentare
- `--folder`: Directory contenente le immagini da segmentare
- `--model`: Percorso del modello di segmentazione
- `--output`: Directory di output per i risultati
- `--threshold`: Soglia per la segmentazione binaria (default: 0.5)
- `--overlay`: Crea un'overlay della segmentazione sull'immagine originale
- `--input_size`: Dimensione di input per il modello (default: 256,256)
- `--j`: Numero di scale per la trasformata scattering (default: 2)
- `--no_morphology`: Disabilita le operazioni morfologiche post-processing
- `--no_display`: Non visualizza i risultati (utile per batch processing)

### Utilizzo programmatico

```python
from wavelet_lib.single_tile_segmentation.models import ScatteringSegmenter
import cv2
import matplotlib.pyplot as plt

# Carica il segmentatore
segmenter = ScatteringSegmenter(
    model_path="/path/to/model.pth",
    J=2,
    input_shape=(256, 256),
    apply_morphology=True
)

# Segmenta un'immagine
binary_mask, raw_pred = segmenter.predict(
    image_path="/path/to/image.jpg",
    threshold=0.5,
    return_raw=True
)

# Visualizza i risultati
plt.figure(figsize=(15, 5))

# Immagine originale
plt.subplot(1, 3, 1)
original = cv2.imread("/path/to/image.jpg")
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
plt.imshow(original_rgb)
plt.title('Immagine Originale')
plt.axis('off')

# Maschera binaria
plt.subplot(1, 3, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Maschera Binaria')
plt.axis('off')

# Heatmap di predizione
plt.subplot(1, 3, 3)
plt.imshow(raw_pred, cmap='jet', vmin=0, vmax=1)
plt.colorbar(label='Confidenza')
plt.title('Heatmap di Predizione')
plt.axis('off')

plt.tight_layout()
plt.savefig("/path/to/segmentation_result.png")
plt.show()
```

### Segmentazione di un'immagine da un array NumPy

```python
# Segmenta un'immagine da un array NumPy
import numpy as np

# Carica l'immagine
img = cv2.imread("/path/to/image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Segmenta l'immagine
binary_mask = segmenter.segment_image(
    img=img_rgb,
    threshold=0.5
)

# Crea un overlay
overlay_img = img_rgb.copy()
overlay_mask = np.zeros_like(img_rgb)
overlay_mask[binary_mask > 0] = [0, 255, 0]  # Verde
overlay_img = cv2.addWeighted(overlay_img, 0.7, overlay_mask, 0.3, 0)

# Visualizza l'overlay
plt.figure(figsize=(10, 10))
plt.imshow(overlay_img)
plt.title('Segmentazione Overlay')
plt.axis('off')
plt.savefig("/path/to/overlay.png")
plt.show()
```

## Utilizzo avanzato

### Personalizzazione dell'architettura

Puoi personalizzare l'architettura del modello U-Net modificando i parametri:

```python
from wavelet_lib.single_tile_segmentation.models import ScatteringUNet
import torch

# Crea un modello U-Net personalizzato
model = ScatteringUNet(
    J=3,  # Aumenta il numero di scale
    input_shape=(512, 512),  # Aumenta la dimensione di input
    num_classes=1  # 1 per segmentazione binaria, >1 per segmentazione multi-classe
).to("cuda" if torch.cuda.is_available() else "cpu")

# Salva il modello personalizzato
torch.save(model.state_dict(), "/path/to/custom_model.pth")
```

### Segmentazione multi-classe

Il modulo supporta anche la segmentazione multi-classe:

```python
# Crea un modello per segmentazione multi-classe
model = ScatteringUNet(
    J=2,
    input_shape=(256, 256),
    num_classes=3  # 3 classi diverse
).to("cuda" if torch.cuda.is_available() else "cpu")

# Per la predizione multi-classe, non applicare sigmoid ma softmax
def predict_multiclass(model, image, device):
    with torch.no_grad():
        output = model(image.to(device))
        pred = torch.softmax(output, dim=1)
        pred_class = torch.argmax(pred, dim=1)
        return pred_class.cpu().numpy()
```

### Applicazione di data augmentation personalizzata

```python
import albumentations as A

# Crea una pipeline di augmentation personalizzata
transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2),
    A.GaussNoise(p=0.2),
    A.ElasticTransform(p=0.2),
    A.GridDistortion(p=0.2),
    A.OpticalDistortion(p=0.2),
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Usa questa trasformazione nel dataset
from torch.utils.data import Dataset

class CustomSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], 0)
        
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        # Converti a tensori PyTorch
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return {'image': img_tensor, 'mask': mask_tensor}
```

## Risoluzione dei problemi

### Errore: "CUDA out of memory"

Riduci la dimensione del batch o la dimensione di input:

```bash
python script/core/segmentation/train_segmentation.py \
    --imgs_dir /path/to/dataset/images \
    --masks_dir /path/to/dataset/masks \
    --model /path/to/save/model.pth \
    --input_size 128,128 \
    --batch_size 4
```

### Segmentazione di bassa qualità

Prova a modificare i parametri della trasformata scattering o ad applicare operazioni morfologiche più aggressive:

```python
# Aumenta il numero di scale
segmenter = ScatteringSegmenter(
    model_path="/path/to/model.pth",
    J=3,  # Aumenta da 2 a 3
    input_shape=(256, 256),
    apply_morphology=True
)

# Applica operazioni morfologiche personalizzate
import cv2
import numpy as np

def post_process_mask(mask):
    # Applica operazioni morfologiche
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Rimuovi piccoli oggetti
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 100:  # Rimuovi oggetti con area < 100 pixel
            mask[labels == i] = 0
    
    return mask

# Usa la funzione di post-processing
binary_mask, _ = segmenter.predict("/path/to/image.jpg", threshold=0.5, return_raw=True)
processed_mask = post_process_mask(binary_mask)
```

### Bordi imprecisi

Se i bordi della segmentazione sono imprecisi, prova ad utilizzare un modello con input di dimensioni maggiori:

```python
# Addestra un modello con input di dimensioni maggiori
train_segmentation_model(
    train_images=train_imgs,
    train_masks=train_masks,
    model_path="/path/to/save/model.pth",
    J=2,
    input_shape=(512, 512),  # Aumenta la dimensione di input
    batch_size=4,  # Riduci il batch size per compensare
    num_epochs=50
)
```

### Overfitting

Se il modello ha buone prestazioni sul training set ma scarse sul validation set, prova ad applicare più data augmentation o regolarizzazione:

```python
# Aggiungi dropout al modello
class ScatteringUNetWithDropout(ScatteringUNet):
    def __init__(self, J=2, input_shape=(256, 256), num_classes=1, dropout_rate=0.2):
        super().__init__(J=J, input_shape=input_shape, num_classes=num_classes)
        self.dropout = torch.nn.Dropout2d(p=dropout_rate)
    
    def forward(self, x):
        # Applica la trasformata scattering
        x = self.scattering(x)
        
        # Encoder pathway con dropout
        e1 = self.dropout(self.enc1(x))
        e2 = self.dropout(self.enc2(self.pool(e1)))
        e3 = self.dropout(self.enc3(self.pool(e2)))
        e4 = self.dropout(self.enc4(self.pool(e3)))
        
        # Middle
        m = self.dropout(self.middle(self.pool(e4)))
        
        # Decoder pathway con skip connections
        d4 = self.dropout(self.dec4(torch.cat([self.up4(m), e4], dim=1)))
        d3 = self.dropout(self.dec3(torch.cat([self.up3(d4), e3], dim=1)))
        d2 = self.dropout(self.dec2(torch.cat([self.up2(d3), e2], dim=1)))
        d1 = self.dropout(self.dec1(torch.cat([self.up1(d2), e1], dim=1)))
        
        # Final convolution
        return F.interpolate(
            self.final_conv(d1), 
            scale_factor=2**self.J, 
            mode='bilinear', 
            align_corners=False
        )
```
