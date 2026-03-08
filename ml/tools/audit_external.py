import os
import numpy as np
import tensorflow as tf
from pathlib import Path

DATASET_ROOT = Path("set-finder/ml/dataset")
FILTER_MODEL_PATH = "set-finder/ml/card_filter.keras"
EXPERT_MODEL_PATH = "set-finder/ml/attribute_expert.keras"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.98 # Very high threshold to find definite errors

MAPS = {
    'count':   {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE'},
    'color':   {0: 'NONE', 1: 'RED', 2: 'GREEN', 3: 'PURPLE'},
    'pattern': {0: 'NONE', 1: 'SOLID', 2: 'SHADED', 3: 'EMPTY'},
    'shape':   {0: 'NONE', 1: 'OVAL', 2: 'DIAMOND', 3: 'SQUIGGLE'}
}

def load_and_preprocess_image(path):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)

def check_external():
    print("Loading models...")
    m_expert = tf.keras.models.load_model(EXPERT_MODEL_PATH)
    
    print("Scanning external dataset for count disagreements...")
    disagreements = []
    ext_paths = list(DATASET_ROOT.rglob("ext_*.jpg"))
    
    # We'll batch this to make it faster
    batch_size = 32
    for i in range(0, len(ext_paths), batch_size):
        batch_paths = ext_paths[i:i+batch_size]
        batch_imgs = []
        batch_labels = []
        
        for p in batch_paths:
            batch_imgs.append(load_and_preprocess_image(p))
            batch_labels.append(p.relative_to(DATASET_ROOT).parts[0]) # COUNT
            
        imgs_tensor = tf.stack(batch_imgs)
        preds = m_expert.predict(imgs_tensor, verbose=0)
        
        count_preds = preds[0] # [batch, 4]
        
        for j, prob_dist in enumerate(count_preds):
            pred_idx = np.argmax(prob_dist)
            conf = prob_dist[pred_idx]
            pred_val = MAPS['count'][pred_idx]
            actual_val = batch_labels[j]
            
            if conf > CONFIDENCE_THRESHOLD and pred_val != actual_val:
                disagreements.append({
                    "path": batch_paths[j],
                    "actual": actual_val,
                    "pred": pred_val,
                    "conf": float(conf)
                })
        
        if (i + batch_size) % 1024 == 0:
            print(f"Checked {i + batch_size}/{len(ext_paths)}... (Found: {len(disagreements)})")

    print(f"\nResults: Found {len(disagreements)} high-confidence count disagreements in 13k images.")
    # Show breakdown
    if disagreements:
        print("Sample of disagreements:")
        for d in disagreements[:10]:
            print(f"  {d['path'].name}: Dataset said {d['actual']}, Model says {d['pred']} ({d['conf']:.1%})")

if __name__ == "__main__":
    check_external()
