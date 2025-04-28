"""
Modulo per la gestione dei dataset di classificazione pixel-wise.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import albumentations as A

class PixelWiseDataset(Dataset):
    """Dataset per la classificazione pixel-wise."""
    
    def __init__(self, images_dir, masks_dir, patch_size=32, transform=None,
                 augment=True, stride=16, class_mapping=None, lazy_loading=False,
                 max_patches_in_memory=100000, max_images=None, verbose=True,
                 metadata_cache_file=None, save_metadata=False):
        """
        Inizializza il dataset con opzioni per ottimizzare l'uso della memoria.
        
        Args:
            images_dir: Directory contenente le immagini
            masks_dir: Directory contenente le maschere di classe
            patch_size: Dimensione delle patch estratte
            transform: Trasformazioni da applicare
            augment: Se applicare data augmentation
            stride: Passo per l'estrazione delle patch
            class_mapping: Mapping delle classi (dict)
            lazy_loading: Se caricare le patch solo quando richieste (riduce l'uso di memoria)
            max_patches_in_memory: Numero massimo di patch da tenere in memoria (per modalità lazy)
            max_images: Numero massimo di immagini da elaborare (None = tutte)
            verbose: Se stampare messaggi informativi durante l'elaborazione
            metadata_cache_file: File da cui caricare/salvare i metadati delle patch
            save_metadata: Se salvare i metadati delle patch dopo l'estrazione
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.augment = augment
        self.lazy_loading = lazy_loading
        self.max_patches_in_memory = max_patches_in_memory
        self.max_images = max_images
        self.verbose = verbose
        self.metadata_cache_file = metadata_cache_file
        self.save_metadata = save_metadata
        
        # Cache per le patch caricate (usata in modalità lazy)
        self.patch_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Trova tutte le immagini (supporta sia maiuscole che minuscole)
        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")) +
                                 list(self.images_dir.glob("*.JPG")) +
                                 list(self.images_dir.glob("*.png")) +
                                 list(self.images_dir.glob("*.PNG")) +
                                 list(self.images_dir.glob("*.tif")) +
                                 list(self.images_dir.glob("*.TIF")))
        
        # Trova le maschere corrispondenti
        self.mask_paths = []
        for img_path in self.image_paths:
            # Prova diversi formati di nome per le maschere
            possible_mask_paths = [
                self.masks_dir / f"{img_path.stem}_mask.png",
                self.masks_dir / f"{img_path.stem}.png",
                self.masks_dir / f"{img_path.stem}.jpg",
                self.masks_dir / f"{img_path.stem}.tif"
            ]
            
            # Cerca la prima maschera che esiste
            mask_path = None
            for p in possible_mask_paths:
                if p.exists():
                    mask_path = p
                    break
            
            # Se non è stata trovata nessuna maschera, salta questa immagine
            if mask_path is None:
                if self.verbose:
                    print(f"Maschera non trovata per {img_path}")
                continue
            
            # Aggiungi il percorso della maschera senza stampare messaggi verbosi
            self.mask_paths.append(mask_path)
        
        # Filtra le immagini senza maschera
        self.image_paths = [img for i, img in enumerate(self.image_paths)
                           if i < len(self.mask_paths)]
        
        # Crea il mapping delle classi
        if class_mapping is None:
            self.class_mapping = {
                0: "background",
                1: "water",
                2: "vegetation",
                3: "streets",
                4: "buildings",
                5: "other"
            }
        else:
            self.class_mapping = class_mapping
        
        self.num_classes = len(self.class_mapping)
        self.class_to_idx = {v: k for k, v in self.class_mapping.items()}
        
        # Crea trasformazioni di augmentation
        if augment:
            self.aug_transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
            ])
        else:
            self.aug_transform = None
        
        # In modalità lazy, memorizziamo solo i metadati delle patch
        # altrimenti estraiamo tutte le patch in memoria
        if self.lazy_loading:
            self.patch_metadata = []
            
            # Prova a caricare i metadati da file se specificato
            if self.metadata_cache_file and os.path.exists(self.metadata_cache_file):
                if self.verbose:
                    print(f"Caricamento metadati da {self.metadata_cache_file}...")
                self._load_metadata()
            else:
                # Estrai i metadati se non è stato possibile caricarli
                self.extract_patch_metadata()
                
                # Salva i metadati se richiesto
                if self.save_metadata and self.metadata_cache_file:
                    self._save_metadata()
            
            if self.verbose:
                print(f"Modalità lazy loading attivata: {len(self.patch_metadata)} metadati di patch estratti")
        else:
            self.patches = []
            self.extract_patches()
    
    def extract_patch_metadata(self):
        """Estrae solo i metadati delle patch (non le patch stesse) per risparmiare memoria."""
        total_images = len(self.image_paths)
        if self.max_images is not None and self.max_images < total_images:
            total_images = self.max_images
            self.image_paths = self.image_paths[:self.max_images]
            self.mask_paths = self.mask_paths[:self.max_images]
        
        if self.verbose:
            print(f"Estraendo metadati delle patch da {total_images} immagini...")
        
        # Crea una barra di progresso principale per le immagini
        main_pbar = tqdm(total=total_images, desc="Analisi immagini", position=0)
        
        # Contatori per le statistiche
        total_patches = 0
        processed_images = 0
        unique_classes = set()
        
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            # Carica solo la maschera per determinare le posizioni delle patch
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                main_pbar.write(f"⚠️ Impossibile caricare la maschera {mask_path}")
                continue
            
            # Ottieni le dimensioni dell'immagine
            img_info = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img_info is None:
                main_pbar.write(f"⚠️ Impossibile caricare l'immagine {img_path}")
                continue
            
            h, w = img_info.shape[:2]
            
            # Verifica le dimensioni
            if img_info.shape[:2] != mask.shape[:2]:
                main_pbar.write(f"ℹ️ Ridimensionamento maschera per {os.path.basename(str(img_path))}: {img_info.shape[:2]} vs {mask.shape[:2]}")
                # Ridimensiona la maschera per adattarla all'immagine
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Libera memoria
            del img_info
            
            # Aggiorna le classi uniche trovate
            img_unique_values = np.unique(mask)
            unique_classes.update(img_unique_values)
            
            # Estrai metadati delle patch
            patches_count = 0
            
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    # Estrai solo la patch della maschera per verificare se contiene classi
                    mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Verifica che la patch contenga almeno un pixel di classe
                    if np.any(mask_patch > 0):
                        # Salva solo i metadati, non la patch stessa
                        self.patch_metadata.append((str(img_path), str(mask_path), (x, y)))
                        patches_count += 1
            
            # Aggiorna contatori
            total_patches += patches_count
            processed_images += 1
            
            # Aggiorna la barra di progresso con informazioni sintetiche
            main_pbar.set_postfix({
                'metadati': total_patches,
                'img': os.path.basename(str(img_path)),
                'classi': len(unique_classes) - 1  # Sottraiamo 1 per escludere lo sfondo (0)
            })
            main_pbar.update(1)
        
        # Chiudi la barra di progresso
        main_pbar.close()
        
        # Mostra un riepilogo finale
        if self.verbose:
            print(f"✅ Estrazione metadati completata: {total_patches} patch da {processed_images} immagini")
            print(f"📊 Classi trovate: {sorted(unique_classes)}")
            print(f"💾 Memoria risparmiata: patch caricate solo quando necessario")
    
    def extract_patches(self):
        """Estrae patch da tutte le immagini con visualizzazione compatta."""
        total_images = len(self.image_paths)
        if self.max_images is not None and self.max_images < total_images:
            total_images = self.max_images
            self.image_paths = self.image_paths[:self.max_images]
            self.mask_paths = self.mask_paths[:self.max_images]
        
        if self.verbose:
            print(f"Estraendo patch da {total_images} immagini...")
        
        # Crea una barra di progresso principale per le immagini
        main_pbar = tqdm(total=total_images, desc="Elaborazione immagini", position=0)
        
        # Contatori per le statistiche
        total_patches = 0
        processed_images = 0
        unique_classes = set()
        
        # Gestione della memoria
        memory_warning_shown = False
        
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            # Carica immagine e maschera
            img = cv2.imread(str(img_path))
            if img is None:
                main_pbar.write(f"⚠️ Impossibile caricare l'immagine {img_path}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                main_pbar.write(f"⚠️ Impossibile caricare la maschera {mask_path}")
                continue
            
            # Verifica le dimensioni
            if img.shape[:2] != mask.shape[:2]:
                main_pbar.write(f"ℹ️ Ridimensionamento maschera per {os.path.basename(str(img_path))}: {img.shape[:2]} vs {mask.shape[:2]}")
                # Ridimensiona la maschera per adattarla all'immagine
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Aggiorna le classi uniche trovate
            img_unique_values = np.unique(mask)
            unique_classes.update(img_unique_values)
            
            # Estrai patch
            h, w = img.shape[:2]
            patches_count = 0
            
            # Verifica se stiamo per superare il limite di memoria
            if len(self.patches) > self.max_patches_in_memory and not memory_warning_shown:
                main_pbar.write(f"⚠️ Attenzione: Numero di patch ({len(self.patches)}) sta superando il limite consigliato ({self.max_patches_in_memory})")
                main_pbar.write(f"   Considera di usare lazy_loading=True per ridurre l'uso di memoria")
                memory_warning_shown = True
            
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
                    mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Verifica che la patch contenga almeno un pixel di classe
                    if np.any(mask_patch > 0):
                        self.patches.append((img_patch, mask_patch, img_path.stem, (x, y)))
                        patches_count += 1
            
            # Aggiorna contatori
            total_patches += patches_count
            processed_images += 1
            
            # Aggiorna la barra di progresso con informazioni sintetiche
            main_pbar.set_postfix({
                'patches': total_patches,
                'img': os.path.basename(str(img_path)),
                'classi': len(unique_classes) - 1  # Sottraiamo 1 per escludere lo sfondo (0)
            })
            main_pbar.update(1)
        
        # Chiudi la barra di progresso
        main_pbar.close()
        
        # Mostra un riepilogo finale
        if self.verbose:
            print(f"✅ Estrazione completata: {total_patches} patch da {processed_images} immagini")
            print(f"📊 Classi trovate: {sorted(unique_classes)}")
    
    def __len__(self):
        """Restituisce il numero di patch nel dataset."""
        if self.lazy_loading:
            return len(self.patch_metadata)
        else:
            return len(self.patches)
    
    def _load_patch(self, img_path, mask_path, coords):
        """Carica una patch dalle coordinate specificate."""
        x, y = coords
        
        # Carica l'immagine e la maschera
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Impossibile caricare l'immagine {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Impossibile caricare la maschera {mask_path}")
        
        # Verifica le dimensioni
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Estrai la patch
        img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
        
        return img_patch, mask_patch, os.path.basename(img_path).split('.')[0], coords
    
    def _manage_cache(self):
        """Gestisce la cache delle patch per il lazy loading."""
        # Se la cache è piena, rimuovi gli elementi meno recenti
        if len(self.patch_cache) >= self.max_patches_in_memory:
            # Rimuovi il 10% delle patch meno recenti
            num_to_remove = max(1, int(self.max_patches_in_memory * 0.1))
            keys_to_remove = list(self.patch_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self.patch_cache[key]
    
    def _save_metadata(self):
        """Salva i metadati delle patch su file."""
        if not self.metadata_cache_file:
            return
        
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(self.metadata_cache_file), exist_ok=True)
        
        # Prepara i dati da salvare
        metadata = {
            'patch_metadata': self.patch_metadata,
            'patch_size': self.patch_size,
            'stride': self.stride,
            'class_mapping': self.class_mapping,
            'images_dir': str(self.images_dir),
            'masks_dir': str(self.masks_dir)
        }
        
        # Salva i metadati
        try:
            with open(self.metadata_cache_file, 'wb') as f:
                torch.save(metadata, f)
            if self.verbose:
                print(f"✅ Metadati salvati in {self.metadata_cache_file}")
        except Exception as e:
            if self.verbose:
                print(f"❌ Errore nel salvataggio dei metadati: {e}")
    
    def _load_metadata(self):
        """Carica i metadati delle patch da file."""
        if not self.metadata_cache_file or not os.path.exists(self.metadata_cache_file):
            return False
        
        try:
            # Carica i metadati
            metadata = torch.load(self.metadata_cache_file)
            
            # Verifica la compatibilità
            if (metadata['patch_size'] != self.patch_size or
                metadata['stride'] != self.stride or
                str(metadata['images_dir']) != str(self.images_dir) or
                str(metadata['masks_dir']) != str(self.masks_dir)):
                if self.verbose:
                    print(f"⚠️ Metadati non compatibili: parametri diversi")
                    print(f"  Cache: {metadata['patch_size']}x{metadata['patch_size']}, stride={metadata['stride']}")
                    print(f"  Attuale: {self.patch_size}x{self.patch_size}, stride={self.stride}")
                return False
            
            # Carica i metadati
            self.patch_metadata = metadata['patch_metadata']
            self.class_mapping = metadata['class_mapping']
            
            if self.verbose:
                print(f"✅ Metadati caricati: {len(self.patch_metadata)} patch")
            return True
        except Exception as e:
            if self.verbose:
                print(f"❌ Errore nel caricamento dei metadati: {e}")
            return False
    
    def __getitem__(self, idx):
        """
        Restituisce una patch dal dataset.
        
        Args:
            idx: Indice della patch
            
        Returns:
            Dizionario con l'immagine e la maschera
        """
        if self.lazy_loading:
            # Modalità lazy loading: carica la patch on-demand
            cache_key = idx
            
            # Verifica se la patch è già in cache
            if cache_key in self.patch_cache:
                self.cache_hits += 1
                img_patch, mask_patch, img_name, coords = self.patch_cache[cache_key]
            else:
                # Carica la patch
                self.cache_misses += 1
                img_path, mask_path, coords = self.patch_metadata[idx]
                img_patch, mask_patch, img_name, coords = self._load_patch(img_path, mask_path, coords)
                
                # Aggiungi alla cache
                self.patch_cache[cache_key] = (img_patch, mask_patch, img_name, coords)
                
                # Gestisci la dimensione della cache
                self._manage_cache()
        else:
            # Modalità standard: le patch sono già in memoria
            img_patch, mask_patch, img_name, coords = self.patches[idx]
        
        # Applica augmentation se richiesto
        if self.augment and self.aug_transform:
            transformed = self.aug_transform(image=img_patch, mask=mask_patch)
            img_patch = transformed['image']
            mask_patch = transformed['mask']
        
        # Converti in tensori
        img_tensor = torch.from_numpy(img_patch.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask_patch).long()
        
        # Applica trasformazioni aggiuntive se specificate
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'img_name': img_name,
            'coords': coords
        }
