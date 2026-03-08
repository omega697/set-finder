import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil

# Configuration
DATASET_ROOT = Path("set-finder/ml/dataset")
CLEANUP_ROOT = Path("set-finder/ml/cleanup")
FILTER_MODEL_PATH = "set-finder/ml/card_filter.keras"
EXPERT_MODEL_PATH = "set-finder/ml/attribute_expert.keras"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.95

MAPS = {
    'count':   {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE'},
    'color':   {0: 'NONE', 1: 'RED', 2: 'GREEN', 3: 'PURPLE'},
    'pattern': {0: 'NONE', 1: 'SOLID', 2: 'SHADED', 3: 'EMPTY'},
    'shape':   {0: 'NONE', 1: 'OVAL', 2: 'DIAMOND', 3: 'SQUIGGLE'}
}

def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)

def clean():
    print("Loading models for verification...")
    m_filter = tf.keras.models.load_model(FILTER_MODEL_PATH)
    m_expert = tf.keras.models.load_model(EXPERT_MODEL_PATH)
    
    if CLEANUP_ROOT.exists(): shutil.rmtree(CLEANUP_ROOT)
    CLEANUP_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("Scanning your confirmed 'ok_' labels...")
    to_verify = []
    for path in DATASET_ROOT.rglob("ok_*.jpg"):
        to_verify.append(path)

    print(f"Verifying {len(to_verify)} images (confidence > {CONFIDENCE_THRESHOLD})...")
    flagged_count = 0
    
    for path in to_verify:
        try:
            rel_parts = list(path.relative_to(DATASET_ROOT).parts)
            # Actual label is determined by folder structure
            actual_parts = rel_parts[:4] # [COUNT, COLOR, PATTERN, SHAPE]
            
            img_tensor = load_and_preprocess_image(str(path))
            img_tensor = tf.expand_dims(img_tensor, 0)
            
            # --- Stage 1: Filter check ---
            is_card_prob = m_filter.predict(img_tensor, verbose=0)[0][0]
            pred_is_card = is_card_prob > 0.5
            actual_is_card = actual_parts[0] != "ZERO"
            
            flag = False
            pred_label_parts = []

            if pred_is_card != actual_is_card and (is_card_prob > CONFIDENCE_THRESHOLD or is_card_prob < (1-CONFIDENCE_THRESHOLD)):
                flag = True
                pred_label_parts = ["ZERO", "NONE", "NONE", "NONE"] if not pred_is_card else ["CARD?", "?", "?", "?"]
            
            if not flag and actual_is_card:
                # --- Stage 2: Expert check ---
                preds = m_expert.predict(img_tensor, verbose=0)
                # preds: [count, color, pattern, shape]
                
                for i, attr in enumerate(['count', 'color', 'pattern', 'shape']):
                    probs = preds[i][0]
                    best_idx = np.argmax(probs)
                    conf = probs[best_idx]
                    pred_val = MAPS[attr][best_idx]
                    pred_label_parts.append(pred_val)
                    
                    if conf > CONFIDENCE_THRESHOLD and pred_val != actual_parts[i]:
                        flag = True

            if flag:
                # Move to cleanup under its PREDICTED label so user can bulk-confirm or reject
                target_dir = CLEANUP_ROOT.joinpath(*pred_label_parts)
                target_dir.mkdir(parents=True, exist_ok=True)
                # Keep record of original label in name
                original_str = "_".join(actual_parts)
                shutil.move(path, target_dir / f"from_{original_str}_{path.name}")
                flagged_count += 1
                
            if (to_verify.index(path) + 1) % 100 == 0:
                print(f"Verified {to_verify.index(path) + 1}/{len(to_verify)}... (Flagged: {flagged_count})")
                
        except Exception as e:
            print(f"Error checking {path}: {e}")

    print(f"Verification complete. {flagged_count} suspicious images moved to {CLEANUP_ROOT}")

if __name__ == "__main__":
    clean()
