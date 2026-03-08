import os
import shutil
import re
from pathlib import Path

CLEANUP_ROOT = Path("set-finder/ml/cleanup")
DATASET_ROOT = Path("set-finder/ml/dataset")

def finalize():
    print("Moving all cleanup images back to original labels...")
    count = 0
    for path in CLEANUP_ROOT.rglob("*.jpg"):
        # Format: from_COUNT_COLOR_PATTERN_SHAPE_ok_chip_...
        match = re.match(r"from_([^_]+)_([^_]+)_([^_]+)_([^_]+)_(ok_chip_.*)", path.name)
        if match:
            c, cl, p, s, orig_name = match.groups()
            target_dir = DATASET_ROOT / c / cl / p / s
            target_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.move(path, target_dir / orig_name)
            count += 1
        else:
            print(f"Skipping unexpected filename: {path.name}")

    print(f"Restored {count} images to their original labels.")
    if CLEANUP_ROOT.exists():
        shutil.rmtree(CLEANUP_ROOT)
        print("Cleanup directory removed.")

if __name__ == "__main__":
    finalize()
