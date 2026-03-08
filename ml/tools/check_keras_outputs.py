import tensorflow as tf
model = tf.keras.models.load_model("set-finder/ml/attribute_expert_v12.keras")
print("\n--- Model Outputs ---")
for i, out in enumerate(model.outputs):
    print(f"Output {i}: {out.name}")

print("\n--- Model Layers (Last 10) ---")
for layer in model.layers[-10:]:
    print(f"Layer: {layer.name}, Type: {type(layer)}")
