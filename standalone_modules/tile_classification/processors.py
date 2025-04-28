"""
Modulo di elaborazione per la classificazione di immagini con Wavelet Scattering Transform.
Contiene funzioni per l'elaborazione e la classificazione di immagini utilizzando trasformate scattering.
"""

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class ClassificationProcessor:
    """Classe per l'elaborazione e la classificazione di immagini con trasformata scattering."""
    
    def __init__(self, model, scattering, device, class_names=None, transform=None):
        """
        Inizializza il processore di classificazione.
        
        Args:
            model: Modello addestrato per la classificazione
            scattering: Trasformata scattering
            device: Device da utilizzare per il calcolo
            class_names: Lista dei nomi delle classi
            transform: Pipeline di trasformazione dell'immagine
        """
        self.model = model
        self.scattering = scattering
        self.device = device
        self.class_names = class_names
        
        # Imposta la trasformazione predefinita se non fornita
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
    
    def process_image(self, image_path):
        """
        Elabora una singola immagine.
        
        Args:
            image_path: Percorso dell'immagine
            
        Returns:
            Classe di predizione e confidenza
        """
        # Carica e trasforma l'immagine
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Applica scattering e modello
        with torch.no_grad():
            scattering_coeffs = self.scattering(image_tensor)
            
            # Il modello gestirà il ridimensionamento internamente - passa direttamente i coefficienti scattering
            outputs = self.model(scattering_coeffs)
            
            # Ottieni predizione e confidenza
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        
        # Converti in valori numerici
        prediction = prediction.item()
        confidence = confidence.item()
        
        # Ottieni il nome della classe se disponibile
        class_name = self.class_names[prediction] if self.class_names else None
        
        return {
            'prediction': prediction,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }
    
    def classify_image_tiles(self, image_path, tile_size=32, confidence_threshold=0.7, process_30x30_tiles=False):
        """
        Classifica un'immagine per tile.
        
        Args:
            image_path: Percorso dell'immagine
            tile_size: Dimensione dei tile da elaborare
            confidence_threshold: Soglia di confidenza per la classificazione
            process_30x30_tiles: Se elaborare tile 30x30 (caso speciale)
            
        Returns:
            Dizionario con i risultati della classificazione
        """
        # Carica l'immagine
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Gestisci il caso speciale per i tile 30x30
        if process_30x30_tiles:
            tile_size = 30
            target_size = 32
            h, w, _ = image_array.shape
            center_y, center_x = h // 2, w // 2
            crop_size = 30 * 32
            y_start = max(0, center_y - crop_size // 2)
            x_start = max(0, center_x - crop_size // 2)
            cropped_image = image_array[y_start:y_start + crop_size, x_start:x_start + crop_size, :]
            img_height, img_width, _ = cropped_image.shape
        else:
            img_height, img_width, _ = image_array.shape
            cropped_image = image_array
            target_size = tile_size
        
        # Calcola il numero di tile
        num_tiles_x = img_width // tile_size
        num_tiles_y = img_height // tile_size
        
        # Prepara per la classificazione
        label_matrix = np.full((num_tiles_y, num_tiles_x), -1, dtype=int)
        confidence_matrix = np.zeros((num_tiles_y, num_tiles_x), dtype=float)
        
        # Prepara la trasformazione
        transform_steps = []
        if tile_size != target_size:
            transform_steps.append(transforms.Resize((target_size, target_size)))
        transform_steps += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        transform = transforms.Compose(transform_steps)
        
        # Elabora tutti i tile
        total_tiles = num_tiles_x * num_tiles_y
        print(f"Elaborazione di {total_tiles} tile...")
        
        processed_tiles = 0
        with torch.no_grad():
            for i in range(num_tiles_y):
                for j in range(num_tiles_x):
                    # Estrai il tile
                    tile = cropped_image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size, :]
                    tile_img = Image.fromarray(tile)
                    tile_tensor = transform(tile_img).unsqueeze(0).to(self.device)
                    
                    # Elabora il tile
                    scattering_coeffs = self.scattering(tile_tensor)
                    
                    # Il modello gestirà il ridimensionamento internamente - passa direttamente i coefficienti scattering
                    output = self.model(scattering_coeffs)
                    
                    # Ottieni predizione e confidenza
                    probabilities = torch.softmax(output, dim=1)
                    max_prob, label = torch.max(probabilities, dim=1)
                    
                    # Memorizza la predizione se la confidenza è abbastanza alta
                    if max_prob.item() >= confidence_threshold:
                        label_matrix[i, j] = label.item()
                        confidence_matrix[i, j] = max_prob.item()
                    
                    # Aggiorna il progresso
                    processed_tiles += 1
                    if processed_tiles % 100 == 0 or processed_tiles == total_tiles:
                        progress_percent = (processed_tiles / total_tiles) * 100
                        print(f"Progresso: {processed_tiles}/{total_tiles} tile ({progress_percent:.1f}%)")
        
        print("Classificazione completata.")
        
        # Conta le distribuzioni delle classi
        class_counts = {}
        for class_idx in range(len(self.class_names) if self.class_names else 0):
            class_counts[class_idx] = np.sum(label_matrix == class_idx)
        
        return {
            'original_image': image_array,
            'cropped_image': cropped_image,
            'label_matrix': label_matrix,
            'confidence_matrix': confidence_matrix,
            'class_counts': class_counts,
            'tile_size': tile_size,
            'num_tiles_x': num_tiles_x,
            'num_tiles_y': num_tiles_y,
            'total_tiles': total_tiles,
            'class_names': self.class_names
        }
