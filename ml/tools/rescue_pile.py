import tensorflow as tf
from tensorflow.keras import models, applications
import numpy as np
import os
from pathlib import Path
import argparse
import shutil

# Constants
IMG_SIZE = (224, 224)

MAPS = {
    'count':   ['ZERO', 'ONE', 'TWO', 'THREE'],
    'color':   ['NONE', 'RED', 'GREEN', 'PURPLE'],
    'pattern': ['NONE', 'SOLID', 'SHADED', 'EMPTY'],
    'shape':   ['NONE', 'OVAL', 'DIAMOND', 'SQUIGGLE']
}

def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = applications.mobilenet_v2.preprocess_input(img)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_model", required=True, help="Path to the .keras card filter model")
    parser.add_argument("--expert_model", required=True, help="Path to the .keras attribute expert model")
    parser.add_argument("--predictions", default="set-finder/ml/predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    m_filter = models.load_model(args.filter_model)
    m_expert = models.load_model(args.expert_model)
    
    PRED_ROOT = Path(args.predictions)
    ZERO_PILE = PRED_ROOT / "ZERO" / "NONE" / "NONE" / "NONE"
    
    if not ZERO_PILE.exists():
        print(f"Pile {ZERO_PILE} not found.")
        return

    pile = list(ZERO_PILE.glob("*.jpg"))
    print(f"Analyzing {len(pile)} images in ZERO pile...")
    
    rescued = 0
    batch_size = 32
    for i in range(0, len(pile), batch_size):
        batch = pile[i:i+batch_size]
        tensors = tf.stack([load_and_preprocess_image(str(p)) for p in batch])
        
        is_card_probs = m_filter.predict(tensors, verbose=0)
        expert_preds = m_expert.predict(tensors, verbose=0)
        
        for j, prob in enumerate(is_card_probs):
            if prob > args.threshold:
                # Output order: color, count, pattern, shape
                p_color = MAPS['color'][np.argmax(expert_preds[0][j])]
                p_count = MAPS['count'][np.argmax(expert_preds[1][j])]
                p_pattern = MAPS['pattern'][np.argmax(expert_preds[2][j])]
                p_shape = MAPS['shape'][np.argmax(expert_preds[3][j])]
                
                label = [p_count, p_color, p_pattern, p_shape]
                if "ZERO" not in label and "NONE" not in label:
                    target_dir = PRED_ROOT.joinpath(*label)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(batch[j], target_dir / batch[j].name)
                    rescued += 1

    print(f"Finished. Rescued {rescued} cards.")

if __name__ == "__main__":
    main()
