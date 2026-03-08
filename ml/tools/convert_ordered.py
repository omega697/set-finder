import tensorflow as tf
from pathlib import Path
import os

def convert_to_tflite(keras_path, tflite_path):
    if not os.path.exists(keras_path):
        print(f"Skipping {keras_path} (not found)")
        return
        
    print(f"Converting {keras_path} to {tflite_path}...")
    model = tf.keras.models.load_model(keras_path)
    
    # Use the standard converter (more reliable than concrete functions for MobileNetV2)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Done.")

if __name__ == "__main__":
    assets_dir = Path("set-finder/app/src/main/assets")
    
    # Convert v12 models
    convert_to_tflite("set-finder/ml/attribute_expert_v12.keras", assets_dir / "set_card_model_final.tflite")
    convert_to_tflite("set-finder/ml/card_filter_v12.keras", assets_dir / "card_filter.tflite")
