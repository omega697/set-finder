import tensorflow as tf
import os
import argparse

def convert_to_tflite(keras_path, tflite_path):
    if not os.path.exists(keras_path):
        print(f"Error: {keras_path} not found.")
        return
        
    print(f"Converting {keras_path} to {tflite_path}...")
    model = tf.keras.models.load_model(keras_path)
    
    # Use standard Keras-to-TFLite path
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Successfully converted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input .keras model")
    parser.add_argument("--output", required=True, help="Path to output .tflite model")
    args = parser.parse_args()
    
    convert_to_tflite(args.input, args.output)
