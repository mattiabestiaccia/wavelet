"""
Dataset handling module for the Wavelet Scattering Transform Library.
"""

import os
import random
import torch
import numpy as np
from PIL import Image
import rasterio
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

class BalancedDataset(Dataset):
    """Dataset class that loads images and optionally balances class distribution."""
    
    def __init__(self, root, transform=None, balance=True):
        """
        Initialize the balanced dataset.
        
        Args:
            root: Root directory of the dataset
            transform: Torchvision transforms to apply to images
            balance: Whether to balance class distribution
        """
        self.root = root
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect images by class
        class_images = defaultdict(list)
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in allowed_extensions:
                    filepath = os.path.join(cls_dir, fname)
                    class_images[cls].append((filepath, self.class_to_idx[cls]))
        
        # Optional balancing
        if balance:
            min_samples = min(len(images) for images in class_images.values())
            for cls, images in class_images.items():
                if len(images) > min_samples:
                    # Random sampling to balance
                    selected_images = random.sample(images, min_samples)
                    self.samples.extend(selected_images)
                else:
                    self.samples.extend(images)
        else:
            # Use all images without balancing
            for images in class_images.values():
                self.samples.extend(images)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, index):
        """Get a sample from the dataset."""
        filepath, label = self.samples[index]
        
        # Try to load with rasterio first for multiband support
        try:
            with rasterio.open(filepath) as src:
                # Read all bands
                img_array = src.read()
                num_bands = src.count
                
                # Convert to PIL Image format (bands, height, width) -> (height, width, bands)
                img_array = np.transpose(img_array, (1, 2, 0))
                
                if num_bands == 1:
                    # Handle single band case
                    image = Image.fromarray(img_array[:,:,0], mode='L')
                else:
                    # Handle multiband case
                    image = Image.fromarray(img_array.astype(np.uint8))
        except:
            # Fallback to PIL for standard image formats
            image = Image.open(filepath)
        
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_class_names(self):
        """Return the list of class names."""
        return self.classes
    
    def get_class_distribution(self):
        """Return a dictionary with the number of samples per class."""
        class_counts = defaultdict(int)
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        return class_counts

def get_default_transform(target_size=(32, 32), normalize=True, dataset_root=None, num_channels=3):
    """
    Create a default transform pipeline for image preprocessing.
    
    Args:
        target_size: Size to resize images to
        normalize: Whether to normalize images
        dataset_root: Root directory of the dataset for computing statistics
        num_channels: Number of channels in the images (default: 3)
        
    Returns:
        transforms.Compose object with the transform pipeline
    """
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]
    
    if normalize:
        if dataset_root:
            # Compute mean and standard deviation from the dataset
            means, stds = compute_dataset_statistics(dataset_root, num_channels)
        else:
            # Default values if dataset not provided
            means = [0.5] * num_channels
            stds = [0.5] * num_channels
            
        transform_list.append(
            transforms.Normalize(mean=means, std=stds)
        )
    
    return transforms.Compose(transform_list)

def compute_dataset_statistics(dataset_root, num_channels=3):
    """
    Compute mean and standard deviation of the dataset.
    
    Args:
        dataset_root: Root directory containing the dataset
        num_channels: Number of channels expected in the images (default: 3)
        
    Returns:
        means, stds: Lists containing channel-wise means and standard deviations
    """
    # Basic transform to convert to tensor without normalization
    basic_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    # Create a temporary dataset
    temp_dataset = BalancedDataset(dataset_root, transform=basic_transform, balance=False)
    loader = DataLoader(temp_dataset, batch_size=128, num_workers=4, shuffle=False)
    
    # Initialize accumulators for dynamic number of channels
    channels_sum = torch.zeros(num_channels)
    channels_squared_sum = torch.zeros(num_channels)
    num_batches = 0
    
    # Calculate statistics
    for data, _ in loader:
        # Handle the case where the actual number of channels might be different
        actual_channels = data.shape[1]
        if actual_channels != num_channels:
            print(f"Warning: Expected {num_channels} channels but got {actual_channels}. Using the first {min(actual_channels, num_channels)} channels.")
            
        # Sum only over the available channels (up to num_channels)
        channels_to_process = min(actual_channels, num_channels)
        channels_sum[:channels_to_process] += torch.mean(data[:, :channels_to_process], dim=[0, 2, 3])
        channels_squared_sum[:channels_to_process] += torch.mean(data[:, :channels_to_process] ** 2, dim=[0, 2, 3])
        num_batches += 1
    
    # Calculate mean and standard deviation
    means = channels_sum / num_batches
    stds = torch.sqrt(channels_squared_sum / num_batches - means ** 2)
    
    return means.tolist(), stds.tolist()

def create_data_loaders(dataset, test_size=0.2, batch_size=128, num_workers=4, random_state=42):
    """
    Split dataset into train and test sets and create data loaders.
    
    Args:
        dataset: Dataset to split
        test_size: Proportion of the dataset to include in the test split (0.0 to 0.99, or 1.0 for evaluation mode)
        batch_size: Batch size for data loaders
        num_workers: Number of worker threads for data loading
        random_state: Random seed for reproducibility
        
    Returns:
        train_loader, test_loader: DataLoader objects for training and testing
    """
    if test_size == 1.0:
        # Evaluation mode - use the entire dataset for testing
        test_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
        # Return empty train_loader and full test_loader
        return None, test_loader
    
    if test_size > 0.99 or test_size <= 0.0:
        raise ValueError("test_size deve essere un valore tra 0.0 e 0.99, o esattamente 1.0 per modalità di valutazione")
        
    # Extract labels for stratified split
    if hasattr(dataset, 'samples'):
        # Per BalancedDataset
        labels = [label for _, label in dataset.samples]
    else:
        # Per altri tipi di dataset
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
    
    # Split indices
    train_indices, test_indices = train_test_split(
        range(len(dataset)), 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def print_dataset_summary(dataset, train_loader, test_loader):
    """
    Print a summary of the dataset.
    
    Args:
        dataset: The dataset object
        train_loader: Training data loader
        test_loader: Testing data loader
    """
    # Get class distribution
    class_counts = defaultdict(int)
    for subset_name, subset_loader in [("train", train_loader), ("test", test_loader)]:
        subset = subset_loader.dataset
        for idx in range(len(subset)):
            # Handle the case where the dataset is a Subset
            if isinstance(subset, Subset):
                _, label = dataset.samples[subset.indices[idx]]
                class_name = dataset.classes[label]
            else:
                _, label = subset[idx]
                class_name = dataset.classes[label]
            class_counts[f"{subset_name}_{class_name}"] += 1
    
    # Print summary
    print("\n" + "="*80)
    print(" "*30 + "DATASET SUMMARY" + " "*30)
    print("="*80)
    
    print("\nCLASSES:")
    print("-" * 60)
    print(f"{'Class':<25} | {'Train':<10} | {'Test':<10} | {'Total':<10}")
    print("-" * 60)
    
    for class_name in dataset.classes:
        train_count = class_counts.get(f"train_{class_name}", 0)
        test_count = class_counts.get(f"test_{class_name}", 0)
        total_count = train_count + test_count
        print(f"{class_name:<25} | {train_count:<10} | {test_count:<10} | {total_count:<10}")
        
    print("-" * 60)
    print(f"{'Total':<25} | {len(train_loader.dataset):<10} | {len(test_loader.dataset):<10} | {len(train_loader.dataset) + len(test_loader.dataset):<10}")
    
    print("\nDATALOADERS:")
    print(f"  • Train batches: {len(train_loader)}")
    print(f"  • Test batches: {len(test_loader)}")
    print(f"  • Train batch size: {train_loader.batch_size}")
    print(f"  • Test batch size: {test_loader.batch_size}")
    
    print("\n" + "="*80)
