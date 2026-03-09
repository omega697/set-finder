import tensorflow as tf
from pathlib import Path

keras_path = "set-finder/ml/attribute_expert_v13.keras"
if not Path(keras_path).exists():
    print(f"Error: {keras_path} not found.")
    exit(1)

model = tf.keras.models.load_model(keras_path)
print("\n--- v13 Model Outputs ---")
for i, out in enumerate(model.outputs):
    print(f"Output {i}: {out.name}")

print("\n--- v13 Model Layers (Last 15) ---")
for layer in model.layers[-15:]:
    print(f"Layer: {layer.name}, Type: {type(layer)}")
    if hasattr(layer, 'activation'):
        print(f"  Activation: {layer.activation.__name__ if hasattr(layer.activation, '__name__') else layer.activation}")
