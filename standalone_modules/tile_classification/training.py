"""
Modulo per l'addestramento dei modelli di classificazione.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .utils import save_model

def create_optimizer(model, config):
    """
    Crea l'ottimizzatore per l'addestramento.
    
    Args:
        model: Modello da addestrare
        config: Configurazione con i parametri dell'ottimizzatore
        
    Returns:
        Ottimizzatore
    """
    # Crea l'ottimizzatore SGD con momentum
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    return optimizer

class Trainer:
    """Classe per l'addestramento dei modelli di classificazione."""
    
    def __init__(self, model, scattering, device, optimizer):
        """
        Inizializza il trainer.
        
        Args:
            model: Modello da addestrare
            scattering: Trasformata scattering
            device: Device per l'addestramento
            optimizer: Ottimizzatore
        """
        self.model = model
        self.scattering = scattering
        self.device = device
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader, epoch):
        """
        Addestra il modello per un'epoca.
        
        Args:
            train_loader: Data loader per l'addestramento
            epoch: Numero dell'epoca
            
        Returns:
            Loss e accuratezza medie
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Crea la barra di progresso
        pbar = tqdm(train_loader, desc=f"Epoca {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Ottieni i dati
            inputs = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)
            
            # Azzera i gradienti
            self.optimizer.zero_grad()
            
            # Forward pass
            scattering_coeffs = self.scattering(inputs)
            outputs = self.model(scattering_coeffs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass e ottimizzazione
            loss.backward()
            self.optimizer.step()
            
            # Aggiorna le statistiche
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Aggiorna la barra di progresso
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        # Calcola le metriche finali
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc
    
    def evaluate(self, test_loader):
        """
        Valuta il modello sul set di test.
        
        Args:
            test_loader: Data loader per il test
            
        Returns:
            Loss e accuratezza medie
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Ottieni i dati
                inputs = batch['image'].to(self.device)
                targets = batch['label'].to(self.device)
                
                # Forward pass
                scattering_coeffs = self.scattering(inputs)
                outputs = self.model(scattering_coeffs)
                loss = self.criterion(outputs, targets)
                
                # Aggiorna le statistiche
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calcola le metriche finali
        test_loss = running_loss / len(test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
    
    def train(self, train_loader, test_loader, epochs, save_path=None, reduce_lr_after=20, class_to_idx=None):
        """
        Addestra il modello per un numero specificato di epoche.
        
        Args:
            train_loader: Data loader per l'addestramento
            test_loader: Data loader per il test
            epochs: Numero di epoche
            save_path: Percorso dove salvare il modello
            reduce_lr_after: Numero di epoche dopo cui ridurre il learning rate
            class_to_idx: Mappatura delle classi
            
        Returns:
            Dizionario con i risultati dell'addestramento
        """
        # Inizializza le liste per le metriche
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        
        # Inizializza le variabili per il miglior modello
        best_acc = 0.0
        best_epoch = 0
        
        # Parametri della trasformata scattering
        scattering_params = {
            'J': self.scattering.J,
            'shape': self.scattering.shape,
            'max_order': self.scattering.max_order
        }
        
        # Stampa le informazioni iniziali
        print(f"\nInizio addestramento per {epochs} epoche:")
        print(f"  • Learning rate iniziale: {self.optimizer.param_groups[0]['lr']}")
        print(f"  • Riduzione del learning rate dopo {reduce_lr_after} epoche")
        if save_path:
            print(f"  • Salvataggio del modello in: {save_path}")
        
        # Addestra per il numero specificato di epoche
        for epoch in range(epochs):
            # Riduci il learning rate se necessario
            if epoch > 0 and epoch % reduce_lr_after == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print(f"\nLearning rate ridotto a {self.optimizer.param_groups[0]['lr']}")
            
            # Addestra per un'epoca
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Valuta sul set di test
            test_loss, test_acc = self.evaluate(test_loader)
            
            # Aggiorna le liste delle metriche
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            # Stampa le metriche
            print(f"Epoca {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # Salva il miglior modello
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                
                if save_path:
                    save_model(
                        self.model,
                        scattering_params,
                        self.optimizer,
                        epoch,
                        test_acc,
                        class_to_idx,
                        save_path
                    )
                    print(f"Miglior modello salvato (acc: {test_acc:.2f}%)")
        
        # Stampa le informazioni finali
        print(f"\nAddestramento completato!")
        print(f"Miglior accuratezza: {best_acc:.2f}% all'epoca {best_epoch+1}")
        
        # Restituisci i risultati
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'best_accuracy': best_acc,
            'best_epoch': best_epoch
        }
