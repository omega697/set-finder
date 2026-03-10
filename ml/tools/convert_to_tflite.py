import tensorflow as tf
import os
import argparse

"""
Simplified TFLite Conversion Script for Set Finder.

This script uses the standard Keras conversion path while ensuring 
that training-only layers (like RandomRotation) are disabled for inference.
"""

def convert_to_tflite(keras_path, tflite_path):
    if not os.path.exists(keras_path):
        print(f"Error: {keras_path} not found.")
        return
        
    print(f"Loading model {keras_path}...")
    # Load the trained model
    model = tf.keras.models.load_model(keras_path)
    
    # Create a clean inference-only wrapper.
    # Passing training=False ensures that augmentation and dropout layers 
    # are bypassed, which prevents conversion errors and runtime crashes.
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input_data')
    outputs = model(inputs, training=False)
    inference_model = tf.keras.Model(inputs, outputs)
    
    print("Converting to TFLite...")
    # Use the standard converter. The output order is guaranteed to match 
    # the order of outputs in the original Keras model.
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Successfully saved to {tflite_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_to_tflite(args.input, args.output)
