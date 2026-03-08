import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras import applications

IMG_SIZE = (224, 224)

def load_and_preprocess(path):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = applications.mobilenet_v2.preprocess_input(img)
    return img

def test_tflite(tflite_path):
    print(f"\n--- Investigating {tflite_path} ---")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()
    
    # Sort by index to see the order
    output_details = sorted(output_details, key=lambda x: x['index'])
    
    for i, detail in enumerate(output_details):
        print(f"Index {i}: Name='{detail['name']}', Index={detail['index']}, Shape={detail['shape']}")

if __name__ == "__main__":
    test_tflite("set-finder/app/src/main/assets/set_card_model_final.tflite")
