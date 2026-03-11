import tensorflow as tf
import os
import argparse
from pathlib import Path

"""
Generalized TFLite Conversion Script for Set Finder.
Usage: python3 convert_to_tflite.py --input model_v14.keras [--output model_v14.tflite]
"""

def convert_to_tflite(keras_path, tflite_path):
    if not os.path.exists(keras_path):
        print(f"Error: {keras_path} not found.")
        return
        
    print(f"Loading model {keras_path}...")
    model = tf.keras.models.load_model(keras_path)
    
    # We use the functional API order to ensure TFLite outputs match Keras
    # Passing training=False is critical to disable dropout/batchnorm updates
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations by default for mobile efficiency
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("Converting to TFLite (with DEFAULT optimizations)...")
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Successfully saved to {tflite_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .keras model")
    parser.add_argument("--output", help="Path to save .tflite model (default: same as input)")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = args.output or input_path.with_suffix(".tflite")
    
    convert_to_tflite(str(input_path), str(output_path))
