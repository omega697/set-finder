import json
import numpy as np
from pathlib import Path
import shutil
from feature_extractor import extract_features

DATASET_ROOT = Path("set-finder/ml/dataset")
PRED_ROOT = Path("set-finder/ml/predictions")

def apply():
    with open("set-finder/ml/bootstrap_model.json", "r") as f:
        centroids = json.load(f)
    
    labels = list(centroids.keys())
    centroid_matrix = np.array([centroids[l] for l in labels])
    
    print("Predicting labels for unreviewed chips...")
    count = 0
    # Process original chips (not starting with ok_ or ext_)
    for path in DATASET_ROOT.rglob("chip_*.jpg"):
        if path.name.startswith("ok_") or path.name.startswith("ext_"):
            continue
        
        feats = extract_features(path)
        if feats is None: continue
        
        # Simple nearest neighbor (Euclidean distance)
        # We should normalize features for better results, but let's see if this works first
        distances = np.linalg.norm(centroid_matrix - feats, axis=1)
        best_idx = np.argmin(distances)
        pred_label = labels[best_idx]
        
        # Move to predictions directory
        target_dir = PRED_ROOT / pred_label.replace(" ", "/")
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target_dir / path.name)
        
        count += 1
        if count % 100 == 0:
            print(f"Predicted {count} images...")

    print(f"Prediction complete. {count} images moved to {PRED_ROOT}")

if __name__ == "__main__":
    apply()
