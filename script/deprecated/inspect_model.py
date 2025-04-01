#!/usr/bin/env python3
"""
Script to inspect the structure of saved model files.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def inspect_model(model_path):
    """
    Load and inspect a saved model file structure.
    
    Args:
        model_path: Path to the model file
    """
    print(f"Inspecting model file: {model_path}")
    
    try:
        # Load the model file
        model_data = torch.load(model_path, map_location='cpu')
        
        # Print top level keys
        print("\nTop-level keys in model file:")
        for key in model_data.keys():
            print(f"- {key}")
        
        # Check for model state
        if 'model_state_dict' in model_data:
            print("\nModel state dictionary keys (model_state_dict):")
            for key in model_data['model_state_dict'].keys():
                tensor = model_data['model_state_dict'][key]
                print(f"- {key}: {tensor.shape if hasattr(tensor, 'shape') else type(tensor)}")
        elif 'model_state' in model_data:
            print("\nModel state dictionary keys (model_state):")
            for key in model_data['model_state'].keys():
                tensor = model_data['model_state'][key]
                print(f"- {key}: {tensor.shape if hasattr(tensor, 'shape') else type(tensor)}")
                
        # Check for class information
        if 'class_to_idx' in model_data:
            print("\nClass mapping:")
            for cls, idx in model_data['class_to_idx'].items():
                print(f"- {cls}: {idx}")
        
        # Check for metrics data
        metrics_keys = ['train_losses', 'train_accuracies', 'test_losses', 'test_accuracies', 'metrics']
        for key in metrics_keys:
            if key in model_data:
                if key == 'metrics':
                    print("\nMetrics dictionary content:")
                    for m_key, value in model_data[key].items():
                        value_info = f"list with {len(value)} items" if isinstance(value, list) else value
                        print(f"- {m_key}: {value_info}")
                else:
                    print(f"\n{key} data: {len(model_data[key])} epochs")
    
    except Exception as e:
        print(f"Error inspecting model file: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <model_dir>")
        
        # List available models
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_dirs = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d)) and 
                     d.startswith('model_output')]
        
        print("\nAvailable model directories:")
        for model_dir in model_dirs:
            print(f"- {model_dir}")
        return
    
    # Get model path
    model_dir = sys.argv[1]
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    # Try to find model files
    model_path = os.path.join(models_dir, model_dir)
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found: {model_path}")
        return
    
    # Check for best model
    best_model_path = os.path.join(model_path, 'best_model.pth')
    if os.path.exists(best_model_path):
        inspect_model(best_model_path)
    else:
        print(f"Warning: best_model.pth not found in {model_path}")
    
    # Check for final model
    final_model_path = os.path.join(model_path, 'final_model.pth')
    if os.path.exists(final_model_path):
        print("\n" + "="*80)
        inspect_model(final_model_path)
    else:
        print(f"Warning: final_model.pth not found in {model_path}")

if __name__ == "__main__":
    main()