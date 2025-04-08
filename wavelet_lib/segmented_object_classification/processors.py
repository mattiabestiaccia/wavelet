"""
Modulo di processori per la classificazione di oggetti segmentati nella Wavelet Scattering Transform Library.
Contiene classi e funzioni per l'estrazione e la classificazione di oggetti da immagini segmentate.
"""

import os
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import json
from pathlib import Path
from kymatio.torch import Scattering2D
from collections import Counter

def extract_objects_from_mask(image, mask, min_size=100, max_size=None, padding=10):
    """
    Estrae oggetti individuali da un'immagine utilizzando una maschera binaria.
    
    Args:
        image: Immagine originale (numpy array)
        mask: Maschera binaria (numpy array)
        min_size: Dimensione minima degli oggetti in pixel
        max_size: Dimensione massima degli oggetti in pixel (None = nessun limite)
        padding: Padding da aggiungere attorno agli oggetti
        
    Returns:
        Lista di tuple (oggetto, bbox, maschera_oggetto)
    """
    # Assicura che la maschera sia binaria
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Trova componenti connesse
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Estrai oggetti
    objects = []
    for i in range(1, num_labels):  # Salta l'etichetta 0 (sfondo)
        # Ottieni statistiche dell'oggetto
        x, y, w, h, area = stats[i]
        
        # Filtra per dimensione
        if area < min_size:
            continue
        if max_size is not None and area > max_size:
            continue
        
        # Crea maschera per questo oggetto
        object_mask = (labels == i).astype(np.uint8)
        
        # Aggiungi padding al bounding box
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(image.shape[1], x + w + padding)
        y_max = min(image.shape[0], y + h + padding)
        
        # Estrai l'oggetto dall'immagine originale
        object_image = image[y_min:y_max, x_min:x_max].copy()
        
        # Estrai la maschera dell'oggetto
        object_mask_cropped = object_mask[y_min:y_max, x_min:x_max]
        
        # Applica la maschera all'immagine (opzionale)
        # object_image[object_mask_cropped == 0] = 0
        
        # Salva l'oggetto, il bounding box e la maschera
        bbox = (x_min, y_min, x_max, y_max)
        objects.append((object_image, bbox, object_mask_cropped))
    
    return objects


def save_extracted_objects(objects, output_dir, base_filename, apply_mask=True):
    """
    Salva gli oggetti estratti come immagini separate.
    
    Args:
        objects: Lista di tuple (oggetto, bbox, maschera_oggetto)
        output_dir: Directory di output
        base_filename: Nome base del file
        apply_mask: Se applicare la maschera all'oggetto
        
    Returns:
        Lista di percorsi dei file salvati
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, (obj_img, bbox, obj_mask) in enumerate(objects):
        # Crea nome file
        filename = f"{base_filename}_obj_{i:03d}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Applica maschera se richiesto
        if apply_mask:
            # Crea una copia dell'immagine
            masked_img = obj_img.copy()
            
            # Crea una maschera RGB per immagini a colori
            if len(obj_img.shape) == 3:
                mask_rgb = np.stack([obj_mask] * 3, axis=2)
                # Applica la maschera (mantieni solo i pixel dell'oggetto)
                masked_img = masked_img * mask_rgb
            else:
                # Per immagini in scala di grigi
                masked_img = masked_img * obj_mask
            
            # Salva l'immagine mascherata
            cv2.imwrite(filepath, masked_img)
        else:
            # Salva l'immagine originale con bounding box
            cv2.imwrite(filepath, obj_img)
        
        saved_paths.append(filepath)
    
    return saved_paths


def get_default_transform(target_size=(32, 32)):
    """
    Ottiene la trasformazione predefinita per le immagini.
    
    Args:
        target_size: Dimensione target per il ridimensionamento
        
    Returns:
        Trasformazione torchvision
    """
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class SegmentedObjectProcessor:
    """Classe per processare e classificare oggetti segmentati."""
    
    def __init__(self, model, scattering, class_mapping, device=None, target_size=(32, 32)):
        """
        Inizializza il processore di oggetti segmentati.
        
        Args:
            model: Modello di classificazione
            scattering: Trasformata scattering
            class_mapping: Mapping delle classi (dict)
            device: Device per l'inferenza
            target_size: Dimensione target per il ridimensionamento
        """
        self.model = model
        self.scattering = scattering
        self.class_mapping = class_mapping
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        
        # Crea mapping inverso
        self.idx_to_class = {v: k for k, v in class_mapping.items()}
        
        # Definisci trasformazione
        self.transform = get_default_transform(target_size)
    
    def process_object(self, object_image):
        """
        Processa un singolo oggetto.
        
        Args:
            object_image: Immagine dell'oggetto (numpy array o percorso)
            
        Returns:
            Dizionario con classe predetta e confidenza
        """
        # Carica immagine se è un percorso
        if isinstance(object_image, str) or isinstance(object_image, Path):
            image = Image.open(object_image).convert('RGB')
        else:
            # Converti da numpy array a PIL Image
            image = Image.fromarray(cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB))
        
        # Applica trasformazione
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Applica scattering e modello
        with torch.no_grad():
            scattering_coeffs = self.scattering(image_tensor)
            outputs = self.model(scattering_coeffs)
            
            # Ottieni predizione e confidenza
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            # Ottieni top-3 predizioni
            top3_probs, top3_indices = torch.topk(probabilities, k=min(3, len(self.class_mapping)))
            top3_classes = [self.idx_to_class[idx.item()] for idx in top3_indices[0]]
            top3_probs = top3_probs[0].cpu().numpy()
        
        # Crea risultato
        result = {
            'class': self.idx_to_class[prediction.item()],
            'confidence': confidence.item(),
            'top3_classes': top3_classes,
            'top3_confidences': top3_probs.tolist()
        }
        
        return result
    
    def process_objects_batch(self, objects_list):
        """
        Processa un batch di oggetti.
        
        Args:
            objects_list: Lista di immagini di oggetti o percorsi
            
        Returns:
            Lista di risultati di classificazione
        """
        results = []
        for obj in tqdm(objects_list, desc="Classificazione oggetti"):
            result = self.process_object(obj)
            results.append(result)
        
        return results
    
    def process_from_segmentation(self, image_path, mask_path, output_dir=None, min_size=100, 
                                 apply_mask=True, save_objects=True, visualize=False):
        """
        Processa oggetti da un'immagine segmentata.
        
        Args:
            image_path: Percorso dell'immagine originale
            mask_path: Percorso della maschera di segmentazione
            output_dir: Directory di output
            min_size: Dimensione minima degli oggetti in pixel
            apply_mask: Se applicare la maschera agli oggetti estratti
            save_objects: Se salvare gli oggetti estratti
            visualize: Se visualizzare i risultati
            
        Returns:
            Dizionario con risultati e statistiche
        """
        # Carica immagine e maschera
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError(f"Impossibile caricare l'immagine o la maschera: {image_path}, {mask_path}")
        
        # Estrai oggetti
        objects = extract_objects_from_mask(image, mask, min_size=min_size)
        print(f"Estratti {len(objects)} oggetti dall'immagine")
        
        # Crea directory di output se necessario
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Salva oggetti se richiesto
        saved_paths = []
        if save_objects and output_dir:
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            objects_dir = os.path.join(output_dir, "objects")
            saved_paths = save_extracted_objects(objects, objects_dir, base_filename, apply_mask)
        
        # Classifica oggetti
        classification_results = []
        for i, (obj_img, bbox, obj_mask) in enumerate(objects):
            # Applica maschera se richiesto
            if apply_mask:
                # Crea una maschera RGB per immagini a colori
                if len(obj_img.shape) == 3:
                    mask_rgb = np.stack([obj_mask] * 3, axis=2)
                    # Applica la maschera (mantieni solo i pixel dell'oggetto)
                    masked_img = obj_img * mask_rgb
                else:
                    # Per immagini in scala di grigi
                    masked_img = obj_img * obj_mask
                
                # Classifica l'immagine mascherata
                result = self.process_object(masked_img)
            else:
                # Classifica l'immagine originale
                result = self.process_object(obj_img)
            
            # Aggiungi informazioni sull'oggetto
            result['object_id'] = i
            result['bbox'] = bbox
            result['size'] = obj_img.shape[:2]
            result['area'] = np.sum(obj_mask)
            
            if save_objects and output_dir:
                result['image_path'] = saved_paths[i]
            
            classification_results.append(result)
        
        # Calcola statistiche
        class_distribution = Counter([r['class'] for r in classification_results])
        
        # Crea risultato finale
        final_result = {
            'image_path': image_path,
            'mask_path': mask_path,
            'num_objects': len(objects),
            'class_distribution': dict(class_distribution),
            'objects': classification_results
        }
        
        # Salva risultati se richiesto
        if output_dir:
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            results_path = os.path.join(output_dir, f"{base_filename}_classification.json")
            with open(results_path, 'w') as f:
                json.dump(final_result, f, indent=2)
        
        # Visualizza risultati se richiesto
        if visualize:
            self.visualize_results(image, mask, classification_results, output_dir)
        
        return final_result
    
    def visualize_results(self, image, mask, results, output_dir=None):
        """
        Visualizza i risultati della classificazione.
        
        Args:
            image: Immagine originale
            mask: Maschera di segmentazione
            results: Risultati della classificazione
            output_dir: Directory di output per salvare la visualizzazione
        """
        # Crea una copia dell'immagine per la visualizzazione
        vis_image = image.copy()
        
        # Disegna bounding box e etichette
        for result in results:
            # Ottieni informazioni
            class_name = result['class']
            confidence = result['confidence']
            bbox = result['bbox']
            
            # Disegna bounding box
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Disegna etichetta
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_image, label, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Visualizza
        plt.figure(figsize=(12, 8))
        
        # Immagine originale
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Immagine Originale')
        plt.axis('off')
        
        # Maschera
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Maschera di Segmentazione')
        plt.axis('off')
        
        # Risultati
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title('Oggetti Classificati')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Salva se richiesto
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "classification_results.png"), dpi=150, bbox_inches='tight')
        
        plt.show()
        
    def process_folder(self, images_dir, masks_dir, output_dir, min_size=100, apply_mask=True, save_objects=True, visualize=False):
        """
        Processa una cartella di immagini e maschere.
        
        Args:
            images_dir: Directory delle immagini
            masks_dir: Directory delle maschere
            output_dir: Directory di output
            min_size: Dimensione minima degli oggetti in pixel
            apply_mask: Se applicare la maschera agli oggetti estratti
            save_objects: Se salvare gli oggetti estratti
            visualize: Se visualizzare i risultati
            
        Returns:
            Dizionario con risultati e statistiche
        """
        # Crea directory di output
        os.makedirs(output_dir, exist_ok=True)
        
        # Trova tutte le immagini
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_paths = []
        for root, _, files in os.walk(images_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            print(f"Nessuna immagine trovata in: {images_dir}")
            return None
        
        print(f"Trovate {len(image_paths)} immagini da processare")
        
        # Processa ogni immagine
        all_results = []
        for image_path in tqdm(image_paths, desc="Processamento immagini"):
            # Trova la maschera corrispondente
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = None
            
            # Cerca la maschera con lo stesso nome base
            for mask_ext in ['.png', '.jpg', '.tif']:
                potential_mask = os.path.join(masks_dir, f"{image_name}_mask{mask_ext}")
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
                    break
            
            if mask_path is None:
                print(f"Maschera non trovata per l'immagine: {image_path}")
                continue
            
            # Crea directory di output per questa immagine
            image_output_dir = os.path.join(output_dir, image_name)
            
            try:
                # Processa l'immagine
                result = self.process_from_segmentation(
                    image_path, mask_path, image_output_dir,
                    min_size=min_size, apply_mask=apply_mask,
                    save_objects=save_objects, visualize=visualize
                )
                all_results.append(result)
            except Exception as e:
                print(f"Errore nel processare {image_path}: {e}")
        
        # Calcola statistiche globali
        if all_results:
            # Conta il numero totale di oggetti
            total_objects = sum(r['num_objects'] for r in all_results)
            
            # Calcola la distribuzione globale delle classi
            global_distribution = Counter()
            for result in all_results:
                global_distribution.update(result['class_distribution'])
            
            # Crea report finale
            final_report = {
                'total_images': len(all_results),
                'total_objects': total_objects,
                'global_class_distribution': dict(global_distribution),
                'per_image_results': all_results
            }
            
            # Salva report
            report_path = os.path.join(output_dir, "classification_report.json")
            with open(report_path, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            # Visualizza distribuzione delle classi
            plt.figure(figsize=(10, 6))
            classes = list(global_distribution.keys())
            counts = list(global_distribution.values())
            
            plt.bar(classes, counts)
            plt.title('Distribuzione Globale delle Classi')
            plt.xlabel('Classe')
            plt.ylabel('Numero di Oggetti')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Salva grafico
            plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150, bbox_inches='tight')
            
            if visualize:
                plt.show()
            else:
                plt.close()
            
            return final_report
        
        return None
