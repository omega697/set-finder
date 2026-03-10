import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import os
from pathlib import Path
import argparse
import shutil

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

MAPS = {
    'count':   {'ONE': 1, 'TWO': 2, 'THREE': 3},
    'color':   {'RED': 1, 'GREEN': 2, 'PURPLE': 3},
    'pattern': {'SOLID': 1, 'SHADED': 2, 'EMPTY': 3},
    'shape':   {'OVAL': 1, 'DIAMOND': 2, 'SQUIGGLE': 3}
}

REV_MAPS = {k: {v: k2 for k2, v in v.items()} for k, v in MAPS.items()}

def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = applications.mobilenet_v2.preprocess_input(img)
    return img

def prepare_datasets(dataset_root, oversample_factor):
    DATASET_ROOT = Path(dataset_root).resolve()
    print(f"Loading data from {DATASET_ROOT}...")
    
    # New Structure:
    # ml/dataset/cards/COUNT/COLOR/PATTERN/SHAPE/*.jpg
    # ml/dataset/non_cards/*.jpg
    
    card_files = list((DATASET_ROOT / "cards").rglob("*.jpg"))
    non_card_files = list((DATASET_ROOT / "non_cards").rglob("*.jpg"))
    
    print(f"Found {len(card_files)} card files and {len(non_card_files)} non-card files.")
    
    # 1. Dataset for Card Filter (Binary classification)
    # We want a balanced dataset or at least enough of both.
    # Oversample cards if they are verified.
    
    ok_cards = [f for f in card_files if f.name.startswith("ok_")]
    ext_cards = [f for f in card_files if f.name.startswith("ext_")]
    
    print(f"  (Cards: {len(ok_cards)} verified, {len(ext_cards)} external)")
    
    balanced_cards = ext_cards + (ok_cards * oversample_factor)
    
    filter_paths = []
    filter_labels = []
    
    for p in balanced_cards:
        filter_paths.append(str(p))
        filter_labels.append(1) # Is a card
        
    for p in non_card_files:
        filter_paths.append(str(p))
        filter_labels.append(0) # Not a card
        
    # Shuffle for filter dataset
    c = list(zip(filter_paths, filter_labels))
    np.random.shuffle(c)
    filter_paths, filter_labels = zip(*c)

    def gen1():
        for p, l in zip(filter_paths, filter_labels):
            yield p, l

    ds1 = tf.data.Dataset.from_generator(
        gen1, 
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    ds1 = ds1.map(lambda p, l: (load_and_preprocess_image(p), l), num_parallel_calls=tf.data.AUTOTUNE)
    ds1 = ds1.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # 2. Dataset for Attribute Expert (Multi-output, Cards Only)
    expert_paths = []
    l_count = []; l_color = []; l_pattern = []; l_shape = []
    
    CARDS_ROOT = DATASET_ROOT / "cards"
    
    for p in balanced_cards:
        rel = p.relative_to(CARDS_ROOT).parts
        if len(rel) < 4: 
            # Skip misorganized files that don't have all attributes
            continue
        
        expert_paths.append(str(p))
        l_count.append(MAPS['count'][rel[0]])
        l_color.append(MAPS['color'][rel[1]])
        l_pattern.append(MAPS['pattern'][rel[2]])
        l_shape.append(MAPS['shape'][rel[3]])
        
    def gen2():
        for i in range(len(expert_paths)):
            yield expert_paths[i], {
                'count_out': l_count[i],
                'color_out': l_color[i],
                'pattern_out': l_pattern[i],
                'shape_out': l_shape[i]
            }

    ds2 = tf.data.Dataset.from_generator(
        gen2,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            {
                'count_out': tf.TensorSpec(shape=(), dtype=tf.int32),
                'color_out': tf.TensorSpec(shape=(), dtype=tf.int32),
                'pattern_out': tf.TensorSpec(shape=(), dtype=tf.int32),
                'shape_out': tf.TensorSpec(shape=(), dtype=tf.int32)
            }
        )
    )
    ds2 = ds2.map(lambda p, l: (load_and_preprocess_image(p), l), num_parallel_calls=tf.data.AUTOTUNE)
    ds2 = ds2.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return ds1, ds2

def build_model(output_type='filter'):
    base = applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
        
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.RandomRotation(0.5)(inputs)
    
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    
    if output_type == 'filter':
        outputs = layers.Dense(1, activation='sigmoid', name='filter_out')(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        x = layers.Dense(512, activation='relu')(x)
        
        # Color: NONE=0, RED=1, GREEN=2, PURPLE=3
        out_color = layers.Dense(4, name='color_logits')(x)
        out_color = layers.Activation('softmax', name='color_out')(out_color)

        # Count: ZERO=0, ONE=1, TWO=2, THREE=3 (Note: ZERO still in MAPS index 0 for consistency if needed, but not used in cards/)
        out_count = layers.Dense(4, name='count_logits')(x)
        out_count = layers.Activation('softmax', name='count_out')(out_count)
        
        # Pattern: NONE=0, SOLID=1, SHADED=2, EMPTY=3
        out_pattern = layers.Dense(4, name='pattern_logits')(x)
        out_pattern = layers.Activation('softmax', name='pattern_out')(out_pattern)
        
        # Shape: NONE=0, OVAL=1, DIAMOND=2, SQUIGGLE=3
        out_shape = layers.Dense(4, name='shape_logits')(x)
        out_shape = layers.Activation('softmax', name='shape_out')(out_shape)
        
        model = models.Model(inputs, [out_color, out_count, out_pattern, out_shape])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                      loss='sparse_categorical_crossentropy', 
                      metrics={'color_out': 'accuracy', 'count_out': 'accuracy', 'pattern_out': 'accuracy', 'shape_out': 'accuracy'})
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_dataset = Path(__file__).parent.parent / "dataset"
    parser.add_argument("--dataset", default=str(default_dataset))
    parser.add_argument("--filter_model", default="card_filter_latest.keras")
    parser.add_argument("--expert_model", default="attribute_expert_latest.keras")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--oversample", type=int, default=4)
    args = parser.parse_args()

    print(f"Preparing datasets with {args.oversample}x oversampling...")
    ds_filter, ds_expert = prepare_datasets(args.dataset, args.oversample)

    print("\n--- Training Card Filter ---")
    m_f = build_model('filter')
    m_f.fit(ds_filter, epochs=args.epochs)
    m_f.save(args.filter_model)

    print("\n--- Training Attribute Expert ---")
    m_e = build_model('expert')
    m_e.fit(ds_expert, epochs=args.epochs)
    m_e.save(args.expert_model)
