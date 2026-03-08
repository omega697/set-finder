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
    'count':   {'ZERO': 0, 'ONE': 1, 'TWO': 2, 'THREE': 3},
    'color':   {'NONE': 0, 'RED': 1, 'GREEN': 2, 'PURPLE': 3},
    'pattern': {'NONE': 0, 'SOLID': 1, 'SHADED': 2, 'EMPTY': 3},
    'shape':   {'NONE': 0, 'OVAL': 1, 'DIAMOND': 2, 'SQUIGGLE': 3}
}

REV_MAPS = {k: {v: k2 for k2, v in v.items()} for k, v in MAPS.items()}

def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = applications.mobilenet_v2.preprocess_input(img)
    return img

def prepare_datasets(dataset_root, oversample_factor):
    DATASET_ROOT = Path(dataset_root)
    all_files = list(DATASET_ROOT.rglob("*.jpg"))
    
    ok_files = [f for f in all_files if f.name.startswith("ok_")]
    ext_files = [f for f in all_files if not f.name.startswith("ok_")]
    
    print(f"Found {len(ok_files)} verified files and {len(ext_files)} bootstrapped files.")
    
    # Combined paths for oversampling
    all_paths = ext_files + (ok_files * oversample_factor)
    np.random.shuffle(all_paths)
    
    # Dataset 1: Card Filter (Binary)
    paths1 = []; labels1 = []
    for p in all_paths:
        rel = p.relative_to(DATASET_ROOT).parts
        if len(rel) < 2: continue
        paths1.append(str(p))
        labels1.append(0 if rel[0] == "ZERO" else 1)
        
    ds1 = tf.data.Dataset.from_tensor_slices((paths1, labels1))
    ds1 = ds1.map(lambda x, y: (load_and_preprocess_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds1 = ds1.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Dataset 2: Attribute Expert (Multi-output, Cards Only)
    paths2 = []; l_count = []; l_color = []; l_pattern = []; l_shape = []
    for p in all_paths:
        rel = p.relative_to(DATASET_ROOT).parts
        if len(rel) < 4 or rel[0] == "ZERO": continue
        
        paths2.append(str(p))
        l_count.append(MAPS['count'][rel[0]])
        l_color.append(MAPS['color'][rel[1]])
        l_pattern.append(MAPS['pattern'][rel[2]])
        l_shape.append(MAPS['shape'][rel[3]])
        
    img_ds2 = tf.data.Dataset.from_tensor_slices(paths2).map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    lbl_ds2 = tf.data.Dataset.from_tensor_slices({
        'count_out': l_count, 'color_out': l_color, 
        'pattern_out': l_pattern, 'shape_out': l_shape
    })
    ds2 = tf.data.Dataset.zip((img_ds2, lbl_ds2)).shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return ds1, ds2

def build_model(output_type='filter'):
    base = applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
        
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.1)(x)
    
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    
    if output_type == 'filter':
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        x = layers.Dense(512, activation='relu')(x)
        
        # Explicitly named output layers for TFLite robustness
        # Order: Color, Count, Pattern, Shape (to match common logic)
        out_color = layers.Dense(4, name='color_logits')(x)
        out_color = layers.Activation('softmax', name='color_out')(out_color)

        out_count = layers.Dense(4, name='count_logits')(x)
        out_count = layers.Activation('softmax', name='count_out')(out_count)
        
        out_pattern = layers.Dense(4, name='pattern_logits')(x)
        out_pattern = layers.Activation('softmax', name='pattern_out')(out_pattern)
        
        out_shape = layers.Dense(4, name='shape_logits')(x)
        out_shape = layers.Activation('softmax', name='shape_out')(out_shape)
        
        model = models.Model(inputs, [out_color, out_count, out_pattern, out_shape])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                      loss='sparse_categorical_crossentropy', 
                      metrics={'color_out': 'accuracy', 'count_out': 'accuracy', 'pattern_out': 'accuracy', 'shape_out': 'accuracy'})
    return model

def rescue_pile(m_filter, m_expert, predictions_root):
    PRED_ROOT = Path(predictions_root)
    PRED_ZERO_PILE = PRED_ROOT / "ZERO" / "NONE" / "NONE" / "NONE"
    
    print("\n--- Rescuing Cards from ZERO Pile ---")
    if not PRED_ZERO_PILE.exists():
        print(f"Pile {PRED_ZERO_PILE} does not exist.")
        return
        
    pile = list(PRED_ZERO_PILE.glob("*.jpg"))
    if not pile: 
        print("ZERO pile is empty.")
        return
        
    print(f"Analyzing {len(pile)} images in rescue pile...")
    rescued = 0
    batch_size = 32
    for i in range(0, len(pile), batch_size):
        batch = pile[i:i+batch_size]
        tensors = tf.stack([load_and_preprocess_image(str(p)) for p in batch])
        
        is_card_probs = m_filter.predict(tensors, verbose=0)
        expert_preds = m_expert.predict(tensors, verbose=0)
        
        for j, prob in enumerate(is_card_probs):
            if prob > 0.5:
                # Expert returns [color, count, pattern, shape]
                p_color = REV_MAPS['color'][np.argmax(expert_preds[0][j])]
                p_count = REV_MAPS['count'][np.argmax(expert_preds[1][j])]
                p_pattern = REV_MAPS['pattern'][np.argmax(expert_preds[2][j])]
                p_shape = REV_MAPS['shape'][np.argmax(expert_preds[3][j])]
                
                label = [p_count, p_color, p_pattern, p_shape]
                if "ZERO" not in label and "NONE" not in label:
                    target_dir = PRED_ROOT.joinpath(*label)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(batch[j], target_dir / batch[j].name)
                    rescued += 1
    
    print(f"Done! Rescued {rescued} images from the ZERO pile.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="set-finder/ml/dataset")
    parser.add_argument("--predictions", default="set-finder/ml/predictions")
    parser.add_argument("--filter_model", default="set-finder/ml/card_filter_latest.keras")
    parser.add_argument("--expert_model", default="set-finder/ml/attribute_expert_latest.keras")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--oversample", type=int, default=4)
    args = parser.parse_args()

    print(f"Preparing datasets with {args.oversample}x oversampling...")
    ds_filter, ds_expert = prepare_datasets(args.dataset, args.oversample)
    
    # 1. Card Filter
    print("\n--- Training Card Filter ---")
    m_f = build_model('filter')
    m_f.fit(ds_filter, epochs=5)
    m_f.save(args.filter_model)
    
    # 2. Attribute Expert
    print("\n--- Training Attribute Expert ---")
    m_e = build_model('expert')
    m_e.fit(ds_expert, epochs=args.epochs)
    m_e.save(args.expert_model)
    
    rescue_pile(m_f, m_e, args.predictions)
