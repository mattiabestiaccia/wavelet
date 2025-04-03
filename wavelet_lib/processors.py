"""
Image processing module for the Wavelet Scattering Transform Library.
Contains functions for processing and classifying images using scattering transforms.
"""

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import rasterio

class ImageProcessor:
    """Class for processing and classifying images with scattering transform."""
    
    def __init__(self, model, scattering, device, class_names=None, transform=None):
        """
        Initialize the image processor.
        
        Args:
            model: Trained model for classification
            scattering: Scattering transform
            device: Device to use for computation
            class_names: List of class names
            transform: Image transform pipeline
        """
        self.model = model
        self.scattering = scattering
        self.device = device
        self.class_names = class_names
        
        # Set default transform if not provided
        if transform is None:
            # Create dynamic normalization based on model's number of channels
            num_channels = model.in_channels if hasattr(model, 'in_channels') else 3
            mean_values = [0.5] * num_channels
            std_values = [0.5] * num_channels
            
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_values, std=std_values)
            ])
        else:
            self.transform = transform
    
    def process_image(self, image_path):
        """
        Process a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Prediction class and confidence
        """
        # Load and transform image - support both standard RGB and multiband images
        try:
            # Try loading with rasterio first for multiband support
            with rasterio.open(image_path) as src:
                # Read all bands
                img_array = src.read()
                num_bands = src.count
                
                # Convert to PIL Image format (bands, height, width) -> (height, width, bands)
                img_array = np.transpose(img_array, (1, 2, 0))
                
                if num_bands > 3:
                    # Handle multiband case
                    image = Image.fromarray(img_array.astype(np.uint8))
                else:
                    # For 1-3 bands, we can use PIL directly
                    image = Image.fromarray(img_array.astype(np.uint8))
        except:
            # Fallback to PIL for standard image formats
            image = Image.open(image_path)
            
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Apply scattering and model
        with torch.no_grad():
            scattering_coeffs = self.scattering(image_tensor)
            
            # The model will handle reshaping internally - pass the scattering coefficients directly
            outputs = self.model(scattering_coeffs)
            
            # Get prediction and confidence
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        
        # Convert to numeric values
        prediction = prediction.item()
        confidence = confidence.item()
        
        # Get class name if available
        class_name = self.class_names[prediction] if self.class_names else None
        
        return {
            'prediction': prediction,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }
    
    def classify_image_tiles(self, image_path, tile_size=32, confidence_threshold=0.7, process_30x30_tiles=False):
        """
        Classify an image by tiles.
        
        Args:
            image_path: Path to the image
            tile_size: Size of tiles to process
            confidence_threshold: Threshold for confidence in classification
            process_30x30_tiles: Whether to process 30x30 tiles (special case)
            
        Returns:
            Dictionary with classification results
        """
        # Load image with support for multiband
        try:
            # Try loading with rasterio first for multiband support
            with rasterio.open(image_path) as src:
                # Read all bands
                img_array = src.read()
                num_bands = src.count
                
                # Convert to format (height, width, bands)
                image_array = np.transpose(img_array, (1, 2, 0))
        except:
            # Fallback to PIL for standard image formats
            image = Image.open(image_path)
            image_array = np.array(image)
        
        # Handle special case for 30x30 tiles
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
        
        # Calculate number of tiles
        num_tiles_x = img_width // tile_size
        num_tiles_y = img_height // tile_size
        
        # Prepare for classification
        label_matrix = np.full((num_tiles_y, num_tiles_x), -1, dtype=int)
        confidence_matrix = np.zeros((num_tiles_y, num_tiles_x), dtype=float)
        
        # Prepare transform
        transform_steps = []
        if tile_size != target_size:
            transform_steps.append(transforms.Resize((target_size, target_size)))
        # Determine number of channels from model or image
        if hasattr(self.model, 'in_channels'):
            num_channels = self.model.in_channels
        else:
            # Try to infer from image array
            if len(image_array.shape) == 3:
                num_channels = image_array.shape[2]
            else:
                num_channels = 3  # Default fallback
                
        mean_values = [0.5] * num_channels
        std_values = [0.5] * num_channels
        
        transform_steps += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values)
        ]
        transform = transforms.Compose(transform_steps)
        
        # Process all tiles
        total_tiles = num_tiles_x * num_tiles_y
        print(f"Processing {total_tiles} tiles...")
        
        processed_tiles = 0
        with torch.no_grad():
            for i in range(num_tiles_y):
                for j in range(num_tiles_x):
                    # Extract tile
                    tile = cropped_image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size, :]
                    tile_img = Image.fromarray(tile)
                    tile_tensor = transform(tile_img).unsqueeze(0).to(self.device)
                    
                    # Process tile
                    scattering_coeffs = self.scattering(tile_tensor)
                    
                    # The model will handle reshaping internally - pass the scattering coefficients directly
                    output = self.model(scattering_coeffs)
                    
                    # Get prediction and confidence
                    probabilities = torch.softmax(output, dim=1)
                    max_prob, label = torch.max(probabilities, dim=1)
                    
                    # Store prediction if confidence is high enough
                    if max_prob.item() >= confidence_threshold:
                        label_matrix[i, j] = label.item()
                        confidence_matrix[i, j] = max_prob.item()
                    
                    # Update progress
                    processed_tiles += 1
                    if processed_tiles % 100 == 0 or processed_tiles == total_tiles:
                        progress_percent = (processed_tiles / total_tiles) * 100
                        print(f"Progress: {processed_tiles}/{total_tiles} tiles ({progress_percent:.1f}%)")
        
        print("Classification complete.")
        
        # Count class distributions
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
            'total_tiles': total_tiles
        }