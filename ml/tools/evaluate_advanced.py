import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import random
from train_advanced import build_feature_extractor, get_image_embedding

DATASET_ROOT = Path("set-finder/ml/dataset")

def evaluate():
    print("Initializing MobileNetV2 feature extractor...")
    model = build_feature_extractor()
    
    # 1. Collect all labeled images
    labeled_images = []
    for path in DATASET_ROOT.rglob("*.jpg"):
        if path.name.startswith("ext_") or path.name.startswith("ok_"):
            rel_parts = path.relative_to(DATASET_ROOT).parts
            if len(rel_parts) >= 5:
                label = " ".join(rel_parts[:4])
                labeled_images.append((path, label))
    
    random.shuffle(labeled_images)
    
    # 2. Split into train (for centroids) and test (for evaluation)
    # We'll use a smaller sample for speed in this environment, e.g., 2000 total
    sample_size = min(len(labeled_images), 2000)
    eval_set = labeled_images[:sample_size]
    
    # Split the sample 80/20
    split_idx = int(0.8 * len(eval_set))
    train_data = eval_set[:split_idx]
    test_data = eval_set[split_idx:]
    
    # 3. Build centroids from train_data
    print(f"Extracting features for {len(train_data)} training images...")
    embeddings_by_label = {}
    for path, label in train_data:
        emb = get_image_embedding(model, path)
        if emb is not None:
            if label not in embeddings_by_label:
                embeddings_by_label[label] = []
            embeddings_by_label[label].append(emb)
            
    centroids = {}
    for label, emb_list in embeddings_by_label.items():
        centroids[label] = np.mean(emb_list, axis=0)
        
    # 4. Evaluate on test_data
    print(f"Evaluating on {len(test_data)} test images...")
    correct = 0
    total = 0
    
    centroid_labels = list(centroids.keys())
    centroid_matrix = np.array([centroids[l] for l in centroid_labels])
    
    for path, true_label in test_data:
        emb = get_image_embedding(model, path)
        if emb is None: continue
        
        distances = np.linalg.norm(centroid_matrix - emb, axis=1)
        pred_label = centroid_labels[np.argmin(distances)]
        
        if pred_label == true_label:
            correct += 1
        total += 1
        
        if total % 50 == 0:
            print(f"Processed {total}/{len(test_data)}...")

    accuracy = correct / total if total > 0 else 0
    print(f"\\nEvaluation Results:")
    print(f"Total evaluated: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    evaluate()
