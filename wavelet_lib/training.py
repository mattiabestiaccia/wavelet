"""
Training module for the Wavelet Scattering Transform Library.
Contains functions for training and evaluation of models.
"""

import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

class Trainer:
    """Class for training and evaluating models with scattering transform."""
    
    def __init__(self, model, scattering, device, optimizer=None, scheduler=None):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            scattering: Scattering transform
            device: Device to use for training
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
        """
        self.model = model
        self.scattering = scattering
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Initialize metrics storage
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Args:
            train_loader: Data loader for training data
            epoch: Current epoch number
            
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        
        for batch_idx, (data, target) in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with scattering
            scattering_coeffs = self.scattering(data)
            
            # For debugging, print the shape on first batch
            if batch_idx == 0:
                print(f"Scattering output shape: {scattering_coeffs.shape}")
                print(f"Expected model input channels: {self.model.in_channels}")
            
            # Reshape the scattering coefficients properly
            batch_size = data.size(0)
            
            # Need to ensure we match the model's expected input channels
            # The model is expecting 12 channels, but we have [batch, 3, 81, 8, 8]
            # We need to take the first 12 channels from the flattened channel dimension
            flattened = scattering_coeffs.reshape(batch_size, -1, 8, 8)
            
            # Take only the first self.model.in_channels channels
            # If there are fewer than needed, we'll repeat channels
            num_channels_available = flattened.size(1)
            if num_channels_available >= self.model.in_channels:
                scattering_coeffs = flattened[:, :self.model.in_channels, :, :]
            else:
                # Repeat channels if we don't have enough
                repeats_needed = (self.model.in_channels + num_channels_available - 1) // num_channels_available
                repeated = flattened.repeat(1, repeats_needed, 1, 1)
                scattering_coeffs = repeated[:, :self.model.in_channels, :, :]
            
            # For debugging, print the reshaped tensor shape
            if batch_idx == 0:
                print(f"Reshaped tensor shape: {scattering_coeffs.shape}")
            
            output = self.model(scattering_coeffs)
            
            # Compute loss
            loss = F.cross_entropy(output, target)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            current_loss = train_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})
        
        # Compute epoch-level metrics
        avg_loss = train_loss / len(train_loader.dataset)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on a test set.
        
        Args:
            test_loader: Data loader for test data
            
        Returns:
            Average loss and accuracy on the test set
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass with scattering
                scattering_coeffs = self.scattering(data)
                
                # Reshape the scattering coefficients properly
                batch_size = data.size(0)
                
                # Need to ensure we match the model's expected input channels
                # The model is expecting 12 channels, but we have [batch, 3, 81, 8, 8]
                # We need to take the first 12 channels from the flattened channel dimension
                flattened = scattering_coeffs.reshape(batch_size, -1, 8, 8)
                
                # Take only the first self.model.in_channels channels
                # If there are fewer than needed, we'll repeat channels
                num_channels_available = flattened.size(1)
                if num_channels_available >= self.model.in_channels:
                    scattering_coeffs = flattened[:, :self.model.in_channels, :, :]
                else:
                    # Repeat channels if we don't have enough
                    repeats_needed = (self.model.in_channels + num_channels_available - 1) // num_channels_available
                    repeated = flattened.repeat(1, repeats_needed, 1, 1)
                    scattering_coeffs = repeated[:, :self.model.in_channels, :, :]
                
                output = self.model(scattering_coeffs)
                
                # Compute loss
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                
                # Compute accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        # Compute metrics
        avg_loss = test_loss / len(test_loader.dataset)
        accuracy = 100. * correct / total
        
        print(f"Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")
        
        return avg_loss, accuracy
    
    def train(self, train_loader, test_loader, epochs, save_path=None, reduce_lr_after=20, class_to_idx=None):
        """
        Train the model for a specified number of epochs.
        
        Args:
            train_loader: Data loader for training data
            test_loader: Data loader for test data
            epochs: Number of epochs to train for
            save_path: Directory to save models
            reduce_lr_after: Reduce learning rate after this many epochs
            class_to_idx: Dictionary mapping class names to indices
            
        Returns:
            Metrics from training and final model
        """
        print(f"Starting training for {epochs} epochs...")
        
        # Prepare model saving
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            best_model_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
            final_model_path = os.path.join(os.path.dirname(save_path), 'final_model.pth')
            checkpoint_path = os.path.join(os.path.dirname(save_path), 'checkpoint.pth')
        
        # Initialize best accuracy
        best_accuracy = 0
        
        # Training loop
        for epoch in range(1, epochs + 1):
            # Reduce learning rate if needed
            if epoch % reduce_lr_after == 0 and epoch > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.2
                    print(f"Reduced learning rate to {param_group['lr']}")
            
            # Train one epoch
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch)
            
            # Evaluate on test set
            test_loss, test_accuracy = self.evaluate(test_loader)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)
            
            # Update learning rate scheduler if available
            if self.scheduler:
                self.scheduler.step()
            
            # Save models if path is provided
            if save_path:
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'class_to_idx': class_to_idx
                }, checkpoint_path)
                
                # Save best model
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'class_to_idx': class_to_idx,
                        'accuracy': best_accuracy
                    }, best_model_path)
                    print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
        
        # Save final model
        if save_path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'class_to_idx': class_to_idx,
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'test_losses': self.test_losses,
                'test_accuracies': self.test_accuracies
            }, final_model_path)
            print(f"Final model saved to {final_model_path}")
        
        # Save training metrics
        self.plot_training_metrics(epochs, save_path)
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies,
            'best_accuracy': best_accuracy,
            'model': self.model
        }
    
    def plot_training_metrics(self, epochs, save_path=None):
        """
        Plot training metrics.
        
        Args:
            epochs: Number of epochs trained
            save_path: Path to save the plot
        """
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), self.train_accuracies, label='Train Accuracy')
        plt.plot(range(1, epochs + 1), self.test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Testing Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), self.train_losses, label='Train Loss')
        plt.plot(range(1, epochs + 1), self.test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            metrics_path = os.path.join(os.path.dirname(save_path), 'training_metrics.png')
            plt.savefig(metrics_path, dpi=300)
            print(f"Training metrics plot saved to {metrics_path}")
        
        plt.show()

def create_optimizer(model, config):
    """
    Create an optimizer for training.
    
    Args:
        model: Model to optimize
        config: Configuration object with optimization parameters
        
    Returns:
        PyTorch optimizer
    """
    return torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )