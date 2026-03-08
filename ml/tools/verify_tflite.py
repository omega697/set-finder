import tensorflow as tf
import numpy as np
from pathlib import Path
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

def test_tflite(tflite_path, keras_path, image_paths):
    print(f"\n--- Testing {tflite_path} ---")
    model = tf.keras.models.load_model(keras_path)
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    keras_names = ['count_out', 'color_out', 'pattern_out', 'shape_out']
    
    for image_path in image_paths:
        print(f"\nImage: {image_path}")
        img = load_and_preprocess(image_path)
        input_data = np.expand_dims(img, 0).astype(np.float32)

        k_preds = model.predict(input_data, verbose=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        for i, detail in enumerate(output_details):
            t_pred = interpreter.get_tensor(detail['index'])[0]
            t_idx = np.argmax(t_pred)
            
            print(f"  TFLite Index {i} ({detail['name']}): Predicted {t_idx} (conf={t_pred[t_idx]:.3f})")
            for j, k_name in enumerate(keras_names):
                k_pred = k_preds[j][0]
                k_idx = np.argmax(k_pred)
                if k_idx == t_idx and k_pred[k_idx] > 0.9:
                    print(f"    Possible Match: Keras {k_name}")

if __name__ == "__main__":
    images = [
        "set-finder/ml/dataset/THREE/RED/EMPTY/DIAMOND/ok_chip_PXL_20260308_003730588_f000202_01.jpg",
        "set-finder/ml/dataset/ONE/GREEN/SHADED/SQUIGGLE/ok_chip_PXL_20260306_223532288_0256.jpg",
        "set-finder/ml/dataset/TWO/PURPLE/SOLID/OVAL/ok_chip_PXL_20260306_223532288_2407.jpg"
    ]
    test_tflite("set-finder/app/src/main/assets/set_card_model_final.tflite", 
                "set-finder/ml/attribute_expert_v12.keras", 
                images)
