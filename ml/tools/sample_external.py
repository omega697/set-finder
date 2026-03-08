import random
from pathlib import Path

DATASET_ROOT = Path("set-finder/ml/dataset")

def sample():
    # Group external images by label
    labeled_groups = {}
    for path in DATASET_ROOT.rglob("ext_*.jpg"):
        rel_parts = path.relative_to(DATASET_ROOT).parts
        if len(rel_parts) < 5: continue
        label = " ".join(rel_parts[:4])
        if label not in labeled_groups:
            labeled_groups[label] = []
        labeled_groups[label].append(path)

    # Focus on TWO/THREE count issues
    problem_labels = [l for l in labeled_groups.keys() if "TWO" in l or "THREE" in l]
    
    # Sample a few categories
    sampled_categories = random.sample(problem_labels, min(len(problem_labels), 6))
    
    results = []
    for cat in sampled_categories:
        imgs = random.sample(labeled_groups[cat], min(len(labeled_groups[cat]), 3))
        results.append((cat, imgs))
    
    return results

if __name__ == "__main__":
    samples = sample()
    for cat, imgs in samples:
        print(f"CATEGORY: {cat}")
        for img in imgs:
            print(f"  {img}")
