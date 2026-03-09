import tensorflow as tf
import numpy as np
from PIL import Image
import os

def run_inference(interpreter, image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    img_data = (img_data - 127.5) / 127.5
    
    interpreter.set_tensor(input_details[0]['index'], img_data)
    interpreter.invoke()
    
    results = []
    for i, detail in enumerate(output_details):
        res = interpreter.get_tensor(detail['index'])[0]
        idx = np.argmax(res)
        results.append((idx, res[idx]))
    return results

if __name__ == "__main__":
    model_path = "set-finder/app/src/main/assets/attribute_expert_v13.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Cards chosen to have distinct attributes
    test_cases = [
        # Expected: ONE (1), RED (1), EMPTY (3), DIAMOND (2)
        ("set-finder/app/src/androidTest/assets/test_ONE_RED_EMPTY_DIAMOND.jpg", "ONE RED EMPTY DIAMOND"),
        # Expected: THREE (3), PURPLE (3), SOLID (1), SQUIGGLE (3)
        ("set-finder/app/src/androidTest/assets/test_ONE_PURPLE_SOLID_SQUIGGLE.jpg", "ONE PURPLE SOLID SQUIGGLE"),
        # Expected: ONE (1), GREEN (2), SHADED (2), OVAL (1)
        ("set-finder/app/src/androidTest/assets/test_ONE_GREEN_SHADED_OVAL.jpg", "ONE GREEN SHADED OVAL")
    ]

    print("\n--- TFLite Output Disambiguation ---")
    for img_path, label in test_cases:
        if os.path.exists(img_path):
            res = run_inference(interpreter, img_path)
            print(f"\nImage: {label}")
            for i, (idx, conf) in enumerate(res):
                print(f"  Output {i}: idx={idx} (conf={conf:.4f})")
        else:
            print(f"\nImage {img_path} not found.")
