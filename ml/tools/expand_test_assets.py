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

        # Pick 1 random file from each leaf to ensure maximum diversity
        sampled = random.sample(files, 1)

        for i, filename in enumerate(sampled):
            src = leaf / filename
            # Standardize name: test_COUNT_COLOR_PATTERN_SHAPE_orig.jpg
            if len(attr_parts) >= 4:
                new_name = f"test_{attr_parts[0]}_{attr_parts[1]}_{attr_parts[2]}_{attr_parts[3]}_{i}.jpg"
            else:
                new_name = f"test_{'_'.join(attr_parts)}_{i}.jpg"
            
            shutil.copy(src, Path(cards_dest) / new_name)
            card_count += 1

    # 2. Sample Non-Cards
    non_cards_root = Path(source_dir) / "non_cards"
    non_card_count = 0
    if non_cards_root.exists():
        files = [f for f in os.listdir(non_cards_root) if f.endswith(".jpg")]
        # Sample up to 20 background images
        sampled = random.sample(files, min(len(files), 20))
        for i, filename in enumerate(sampled):
            src = non_cards_root / filename
            new_name = f"background_{i:02d}.jpg"
            shutil.copy(src, Path(non_cards_dest) / new_name)
            non_card_count += 1

    print(f"Copied {card_count} card chips and {non_card_count} non-card chips.")

if __name__ == "__main__":
    # Ensure dest dirs exist
    os.makedirs("set-finder/app/src/androidTest/assets/chips", exist_ok=True)
    os.makedirs("set-finder/app/src/androidTest/assets/non_cards", exist_ok=True)
    
    sample_dataset(
        "set-finder/ml/dataset",
        "set-finder/app/src/androidTest/assets/chips",
        "set-finder/app/src/androidTest/assets/non_cards"
    )
