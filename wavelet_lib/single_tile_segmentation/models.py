"""
Segmentation module for the Wavelet Scattering Transform Library.
Contains models and utilities for image segmentation using wavelet scattering transforms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A


class ScatteringPreprocessor(nn.Module):
    """Wavelet Scattering Transform preprocessor for image segmentation."""
    
    def __init__(self, J=2, shape=(256, 256)):
        """
        Initialize the scattering preprocessor.
        
        Args:
            J (int): Number of scales for the scattering transform
            shape (tuple): Shape of the input images (height, width)
        """
        super().__init__()
        self.scattering = Scattering2D(J=J, shape=shape)
        self.J = J
        self.shape = shape
        
    def forward(self, x):
        """
        Apply scattering transform to input images.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Reshaped scattering coefficients
        """
        S = self.scattering(x)
        batch_size, channels, coeffs, h, w = S.shape
        return S.view(batch_size, channels * coeffs, h, w)


class ScatteringUNet(nn.Module):
    """U-Net architecture with Wavelet Scattering Transform for segmentation."""
    
    def __init__(self, J=2, input_shape=(256, 256), num_classes=1):
        """
        Initialize the scattering U-Net model.
        
        Args:
            J (int): Number of scales for the scattering transform
            input_shape (tuple): Shape of the input images (height, width)
            num_classes (int): Number of output classes (1 for binary segmentation)
        """
        super().__init__()
        
        # Calculate scattering dimensions
        self.J = J
        self.scattering = ScatteringPreprocessor(J=J, shape=input_shape)
        dummy_in = torch.randn(1, 3, *input_shape)
        dummy_out = self.scattering(dummy_in)
        scat_channels = dummy_out.shape[1]
        
        # Encoder pathway
        self.enc1 = self._double_conv(scat_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # Middle
        self.middle = self._double_conv(512, 1024)
        
        # Decoder pathway with skip connections
        self.dec4 = self._double_conv(1024 + 512, 512)
        self.dec3 = self._double_conv(512 + 256, 256)
        self.dec2 = self._double_conv(256 + 128, 128)
        self.dec1 = self._double_conv(128 + 64, 64)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Max pooling for encoder
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Upsample operations for decoder
        self.up4 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        
    def _double_conv(self, in_channels, out_channels):
        """
        Create a sequence of double convolution with batch normalization.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            
        Returns:
            nn.Sequential: Double convolution block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Segmentation mask
        """
        # Apply scattering transform
        x = self.scattering(x)
        
        # Encoder pathway with pooling
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Middle
        m = self.middle(self.pool(e4))
        
        # Decoder pathway with skip connections
        d4 = self.dec4(torch.cat([self.up4(m), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Final convolution
        return F.interpolate(
            self.final_conv(d1), 
            scale_factor=2**self.J, 
            mode='bilinear', 
            align_corners=False
        )


class ScatteringSegmenter:
    """Class for segmenting images using a trained Scattering U-Net model."""
    
    def __init__(self, model_path, J=2, input_shape=(256, 256), device=None, apply_morphology=True):
        """
        Initialize the segmenter with a trained model.
        
        Args:
            model_path (str): Path to the trained model weights
            J (int): Number of scales for the scattering transform
            input_shape (tuple): Shape of the input images (height, width)
            device (torch.device): Device to use for inference
            apply_morphology (bool): Whether to apply morphological operations to the output
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Create model
        self.model = ScatteringUNet(J=J, input_shape=input_shape).to(self.device)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Set parameters
        self.J = J
        self.input_shape = input_shape
        self.apply_morphology = apply_morphology
        
        # Define image transform
        self.transform = A.Compose([
            A.Resize(input_shape[0], input_shape[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    
    def predict(self, image_path, threshold=0.5, return_raw=False):
        """
        Segment an image.
        
        Args:
            image_path (str): Path to the input image
            threshold (float): Threshold for binary segmentation
            return_raw (bool): Whether to return the raw prediction as well
            
        Returns:
            numpy.ndarray: Binary segmentation mask
            numpy.ndarray (optional): Raw prediction probabilities if return_raw is True
        """
        # Load and preprocess image
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        original_size = img.shape[:2]
        
        # Apply preprocessing
        transformed = self.transform(image=img)
        img_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float().unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(img_tensor.to(self.device))
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Resize to original size
        pred_resized = cv2.resize(pred, (original_size[1], original_size[0]))
        
        # Apply threshold to get binary mask
        binary_mask = (pred_resized > threshold).astype(np.uint8)
        
        # Apply morphological operations if requested
        if self.apply_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        if return_raw:
            return binary_mask, pred_resized
        return binary_mask
    
    def segment_image(self, img, threshold=0.5, return_raw=False):
        """
        Segment an image from a numpy array.
        
        Args:
            img (numpy.ndarray): Input image as RGB numpy array
            threshold (float): Threshold for binary segmentation
            return_raw (bool): Whether to return the raw prediction as well
            
        Returns:
            numpy.ndarray: Binary segmentation mask
            numpy.ndarray (optional): Raw prediction probabilities if return_raw is True
        """
        # Ensure RGB format
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
        original_size = img.shape[:2]
        
        # Apply preprocessing
        transformed = self.transform(image=img)
        img_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float().unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(img_tensor.to(self.device))
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Resize to original size
        pred_resized = cv2.resize(pred, (original_size[1], original_size[0]))
        
        # Apply threshold to get binary mask
        binary_mask = (pred_resized > threshold).astype(np.uint8)
        
        # Apply morphological operations if requested
        if self.apply_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        if return_raw:
            return binary_mask, pred_resized
        return binary_mask


class SegmentationDataset(Dataset):
    """Dataset class for image segmentation with WST preprocessing."""
    
    def __init__(self, image_paths, mask_paths=None, transform=None, J=2, input_shape=(256, 256)):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): List of paths to input images
            mask_paths (list): List of paths to ground truth masks
            transform (albumentations.Compose): Transforms to apply to images and masks
            J (int): Number of scales for the scattering transform
            input_shape (tuple): Shape of the input images (height, width)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
        # Create scattering transform
        self.scattering = Scattering2D(J=J, shape=input_shape)
        self.J = J
        self.input_shape = input_shape
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = A.Compose([
                A.Resize(input_shape[0], input_shape[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing image and mask
        """
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask if available
        mask = None
        if self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            mask = cv2.imread(mask_path, 0)
            
        # Apply transforms
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
            else:
                transformed = self.transform(image=img)
                img = transformed['image']
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Return image tensor and mask if available
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float()
            if len(mask_tensor.shape) == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            return {'image': img_tensor, 'mask': mask_tensor}
        else:
            return {'image': img_tensor}


def train_segmentation_model(train_images, train_masks,
                             val_images=None, val_masks=None,
                             model_path='wst_unet.pth',
                             J=2,
                             input_shape=(256, 256),
                             batch_size=8,
                             num_epochs=50,
                             learning_rate=1e-4,    
                             device=None,
                             num_workers=4):
    """
    Train a WST-UNet segmentation model.
    
    Args:
        train_images (list): List of paths to training images
        train_masks (list): List of paths to training masks
        val_images (list): List of paths to validation images
        val_masks (list): List of paths to validation masks
        model_path (str): Path to save the trained model
        J (int): Number of scales for the scattering transform
        input_shape (tuple): Shape of the input images (height, width)
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
        device (torch.device): Device to use for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        dict: Training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create augmentations for training
    train_transform = A.Compose([
        A.Resize(input_shape[0], input_shape[1]),
        A.RandomRotate90(),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    # Create augmentations for validation
    val_transform = A.Compose([
        A.Resize(input_shape[0], input_shape[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    train_dataset = SegmentationDataset(
        train_images, train_masks,
        transform=train_transform,
        J=J, input_shape=input_shape
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    
    # Create validation dataset if validation data is provided
    val_loader = None
    if val_images is not None and val_masks is not None:
        val_dataset = SegmentationDataset(
            val_images, val_masks,
            transform=val_transform,
            J=J, input_shape=input_shape
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )
    
    # Create model
    model = ScatteringUNet(J=J, input_shape=input_shape, num_classes=1).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase if validation data is available
        val_loss = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    
                    # Calculate loss
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            
            # Calculate average validation loss
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
                print(f"Epoch {epoch+1}/{num_epochs}, saved new best model with val_loss: {val_loss:.4f}")
        else:
            # Save model at regular intervals without validation
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                torch.save(model.state_dict(), model_path)
                print(f"Epoch {epoch+1}/{num_epochs}, saved model")
        
        # Print progress
        if val_loader is not None:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
    
    # Save final model if using validation
    if val_loader is not None:
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    else:
        # Save final model if not using validation
        torch.save(model.state_dict(), model_path)
        print(f"Training completed. Final model saved to {model_path}")
    
    return history