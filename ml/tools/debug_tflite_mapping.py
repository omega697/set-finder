import tensorflow as tf
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path

"""
Generalized TFLite Mapping Debugger.
Runs inference on one or more images and prints the output of every model head.
Usage: python3 debug_tflite_mapping.py --model path/to/model.tflite --images path/to/chips/
"""

def run_inference(interpreter, image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    # Standard MobileNetV2 preprocessing
    img_data = (img_data - 127.5) / 127.5
    
    interpreter.set_tensor(input_details[0]['index'], img_data)
    interpreter.invoke()
    
    results = []
    # The order in output_details is usually the order of heads in the TFLite file
    for detail in output_details:
        res = interpreter.get_tensor(detail['index'])[0]
        idx = np.argmax(res)
        results.append((idx, res[idx]))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--images", required=True, help="Path to image file or directory of images")
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    image_paths = []
    p = Path(args.images)
    if p.is_dir():
        image_paths = list(p.glob("*.jpg"))[:5] # Sample up to 5
    else:
        image_paths = [p]

    print(f"\n--- TFLite Output Mapping Debug: {args.model} ---")
    for img_path in image_paths:
        if img_path.exists():
            res = run_inference(interpreter, str(img_path))
            print(f"\nImage: {img_path.name}")
            for i, (idx, conf) in enumerate(res):
                print(f"  Head {i}: idx={idx} (conf={conf:.4f})")
        else:
            print(f"\nImage {img_path} not found.")
