#!/usr/bin/env python3
"""
Script to define the correct class mapping for VDD dataset.
This provides consistent class names across training and inference.
"""

# Standard VDD class mapping based on the official dataset
VDD_CLASS_MAPPING = {
    0: 'background',
    1: 'wall',
    2: 'roads',
    3: 'vegetation',
    4: 'vehicles',
    5: 'roof',
    6: 'others'
}

# Function to save this mapping into a model checkpoint
def update_model_class_mapping(model_path, output_path=None):
    import torch
    import os
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Update the class mapping
    checkpoint['class_mapping'] = VDD_CLASS_MAPPING
    
    # Save the updated checkpoint
    if output_path is None:
        output_path = model_path
    
    torch.save(checkpoint, output_path)
    print(f"Updated class mapping in {output_path}")
    print(f"New class mapping: {VDD_CLASS_MAPPING}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update class mapping in a model checkpoint")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output", type=str, help="Output path for the updated model (defaults to overwriting input)")
    
    args = parser.parse_args()
    update_model_class_mapping(args.model, args.output)