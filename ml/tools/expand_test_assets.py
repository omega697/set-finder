import os
import shutil
import random
from pathlib import Path

def sample_dataset(source_dir, cards_dest, non_cards_dest):
    # Valid attributes for flattening names
    counts = ["ONE", "TWO", "THREE"]
    
    # Traverse all leaf directories under cards/
    cards_root = Path(source_dir) / "cards"
    leaf_dirs = []
    for root, dirs, files in os.walk(cards_root):
        if not dirs and files:
            leaf_dirs.append(Path(root))

    card_count = 0
    # 1. Sample Cards
    for leaf in leaf_dirs:
        try:
            attr_parts = leaf.relative_to(cards_root).parts
        except ValueError:
            continue
            
        if not attr_parts or attr_parts[0] not in counts: continue

        files = [f for f in os.listdir(leaf) if f.endswith(".jpg")]
        if not files: continue

        # Pick up to 2 random files from each leaf for robustness
        sampled = random.sample(files, min(len(files), 2))

        for i, filename in enumerate(sampled):
            src = leaf / filename
            # New format: COUNT_COLOR_PATTERN_SHAPE_0.jpg
            if len(attr_parts) >= 4:
                new_name = f"{attr_parts[0]}_{attr_parts[1]}_{attr_parts[2]}_{attr_parts[3]}_{i}.jpg"
            else:
                new_name = f"{'_'.join(attr_parts)}_{i}.jpg"
            
            shutil.copy(src, Path(cards_dest) / new_name)
            card_count += 1

    # 2. Sample Non-Cards
    non_cards_root = Path(source_dir) / "non_cards"
    non_card_count = 0
    if non_cards_root.exists():
        files = [f for f in os.listdir(non_cards_root) if f.endswith(".jpg")]
        sampled = random.sample(files, min(len(files), 20))
        for i, filename in enumerate(sampled):
            src = non_cards_root / filename
            new_name = f"background_{i:02d}.jpg"
            shutil.copy(src, Path(non_cards_dest) / new_name)
            non_card_count += 1

    print(f"Copied {card_count} card chips and {non_card_count} non-card chips.")

if __name__ == "__main__":
    cards_path = "set-finder/app/src/androidTest/assets/chips/cards"
    non_cards_path = "set-finder/app/src/androidTest/assets/chips/non_cards"
    
    # Cleanup old assets before fresh sampling
    if os.path.exists(cards_path): shutil.rmtree(cards_path)
    if os.path.exists(non_cards_path): shutil.rmtree(non_cards_path)
    
    os.makedirs(cards_path, exist_ok=True)
    os.makedirs(non_cards_path, exist_ok=True)
    
    sample_dataset(
        "set-finder/ml/dataset",
        cards_path,
        non_cards_path
    )
