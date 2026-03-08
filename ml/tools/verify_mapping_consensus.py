import tensorflow as tf
import numpy as np
from tensorflow.keras import applications

IMG_SIZE = (224, 224)
MAPS = {
    'count':   ['ZERO', 'ONE', 'TWO', 'THREE'],
    'color':   ['NONE', 'RED', 'GREEN', 'PURPLE'],
    'pattern': ['NONE', 'SOLID', 'SHADED', 'EMPTY'],
    'shape':   ['NONE', 'OVAL', 'DIAMOND', 'SQUIGGLE']
}

def load_and_preprocess(path):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = applications.mobilenet_v2.preprocess_input(img)
    return img

def test_mapping(tflite_path, test_data):
    print(f"\n--- Testing {tflite_path} with {len(test_data)} samples ---")
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output_details = sorted(output_details, key=lambda x: x['index'])

    # Matrix to track votes: OutputIndex x AttributeName
    votes = {i: {attr: 0 for attr in MAPS.keys()} for i in range(len(output_details))}

    for path, expected in test_data:
        print(f"Sample: {path}")
        img = load_and_preprocess(path)
        input_data = np.expand_dims(img, 0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        for i, detail in enumerate(output_details):
            t_pred = interpreter.get_tensor(detail['index'])[0]
            t_idx = np.argmax(t_pred)
            t_conf = t_pred[t_idx]
            
            for attr, labels in MAPS.items():
                if labels[t_idx] == expected[attr]:
                    votes[i][attr] += 1
                    print(f"  Output {i} predicts {labels[t_idx]} (conf={t_conf:.3f}), matches {attr}")

    print("\n--- Final Vote Summary ---")
    for i in range(len(output_details)):
        best_attr = max(votes[i], key=votes[i].get)
        print(f"Output {i}: {best_attr} (Votes: {votes[i]})")

if __name__ == "__main__":
    test_samples = [
        ("set-finder/ml/dataset/THREE/PURPLE/SOLID/SQUIGGLE/ok_chip_PXL_20260308_003730588_f002366_00.jpg", 
         {'count': 'THREE', 'color': 'PURPLE', 'pattern': 'SOLID', 'shape': 'SQUIGGLE'}),
        ("set-finder/ml/dataset/ONE/RED/SOLID/SQUIGGLE/ok_chip_PXL_20260308_003730588_f000024_07.jpg", 
         {'count': 'ONE', 'color': 'RED', 'pattern': 'SOLID', 'shape': 'SQUIGGLE'}),
        ("set-finder/ml/dataset/THREE/RED/SOLID/DIAMOND/ok_chip_PXL_20260308_003730588_f000970_03.jpg", 
         {'count': 'THREE', 'color': 'RED', 'pattern': 'SOLID', 'shape': 'DIAMOND'}),
        ("set-finder/ml/dataset/TWO/PURPLE/SOLID/DIAMOND/ok_chip_PXL_20260306_223532288_1844.jpg", 
         {'count': 'TWO', 'color': 'PURPLE', 'pattern': 'SOLID', 'shape': 'DIAMOND'}),
    ]
    test_mapping("set-finder/app/src/main/assets/set_card_model_final.tflite", test_samples)
