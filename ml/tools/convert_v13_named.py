import tensorflow as tf
from pathlib import Path
import os

def convert_with_signatures(keras_path, tflite_path):
    if not os.path.exists(keras_path):
        print(f"Error: {keras_path} not found.")
        return
        
    print(f"Converting {keras_path} to {tflite_path} with Signatures...")
    model = tf.keras.models.load_model(keras_path)
    
    # Define a concrete function for the signature
    run_model = tf.function(lambda x: model(x))
    
    # Get the concrete function for a specific input shape
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, 224, 224, 3], model.inputs[0].dtype)
    )
    
    # Use from_concrete_functions and provide a signature key
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_func], model
    )
    
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Successfully converted with Signatures.")

if __name__ == "__main__":
    assets_dir = Path("set-finder/app/src/main/assets")
    
    # Convert v13 models with signatures to preserve output names
    # Note: Using the actual .keras paths from the v13 run
    convert_with_signatures("set-finder/ml/attribute_expert_v13.keras", assets_dir / "attribute_expert_v13.tflite")
    convert_with_signatures("set-finder/ml/card_filter_v13.keras", assets_dir / "card_filter_v13.tflite")
