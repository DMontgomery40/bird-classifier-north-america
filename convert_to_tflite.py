#!/usr/bin/env python3
"""
Convert NABirds PyTorch model to TFLite format for use with Scrypted.

This script provides a template for converting the PyTorch model from
npatta01/Bird-Classifier to TFLite format.

Requirements:
- torch
- tensorflow
- onnx
- onnx-tf

Usage:
    python convert_to_tflite.py --pytorch-model path/to/model.pkl --output nabirds.tflite
"""

import argparse
import json
import numpy as np
import sys

# Note: This is a template script. The actual conversion process may require
# adjustments based on the specific PyTorch model architecture.

def convert_pytorch_to_tflite(pytorch_model_path, output_path):
    """
    Convert PyTorch model to TFLite format.
    
    This is a general approach - specific models may require adjustments.
    """
    try:
        import torch
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare
        
        print("Step 1: Loading PyTorch model...")
        # Load the PyTorch model
        # Note: The actual loading method depends on how the model was saved
        # This is just an example
        model = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
        model.eval()
        
        print("Step 2: Converting to ONNX...")
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX
        onnx_path = output_path.replace('.tflite', '.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print("Step 3: Converting ONNX to TensorFlow...")
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph("tf_model")
        
        print("Step 4: Converting TensorFlow to TFLite...")
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ Successfully converted model to {output_path}")
        
        # Generate metadata (optional but recommended)
        print("\nStep 5: Generating model information...")
        print("Input shape: [1, 224, 224, 3]")
        print("Output shape: [1, 400] (400 bird classes)")
        print("\nRemember to:")
        print("1. Upload the .tflite file to a public location")
        print("2. Update config.json with the model URL")
        print("3. Verify the labels match your model's output classes")
        
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("\nPlease install required packages:")
        print("pip install torch tensorflow onnx onnx-tf")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("\nThis is a template script that may need adjustments for your specific model.")
        sys.exit(1)

def create_labels_file(classes_json_path, output_path):
    """
    Convert classes.json to labels.txt format if needed.
    """
    try:
        with open(classes_json_path, 'r') as f:
            classes = json.load(f)
        
        # Convert to labels.txt format
        with open(output_path, 'w') as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
        
        print(f"✓ Created labels file: {output_path}")
        
    except Exception as e:
        print(f"Error creating labels file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert NABirds PyTorch model to TFLite')
    parser.add_argument('--pytorch-model', required=True, help='Path to PyTorch model file')
    parser.add_argument('--output', default='nabirds_model.tflite', help='Output TFLite model path')
    parser.add_argument('--classes-json', help='Path to classes.json file')
    parser.add_argument('--create-labels', help='Create labels.txt file at this path')
    
    args = parser.parse_args()
    
    # Convert model
    convert_pytorch_to_tflite(args.pytorch_model, args.output)
    
    # Create labels file if requested
    if args.classes_json and args.create_labels:
        create_labels_file(args.classes_json, args.create_labels)

if __name__ == '__main__':
    main()
