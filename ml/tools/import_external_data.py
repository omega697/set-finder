import os
import shutil
from pathlib import Path

SOURCE_DIR = Path("/usr/local/google/home/kdresner/.gemini/tmp/androidstudioprojects/set-game/data/train-v2/labelled")
TARGET_ROOT = Path("set-finder/ml/dataset")

# Mapping from set-game naming to our naming
MAP = {
    "1": "ONE", "2": "TWO", "3": "THREE",
    "red": "RED", "green": "GREEN", "purple": "PURPLE",
    "empty": "EMPTY", "solid": "SOLID", "striped": "SHADED",
    "diamond": "DIAMOND", "oval": "OVAL", "squiggle": "SQUIGGLE",
    "diamonds": "DIAMOND", "ovals": "OVAL", "squiggles": "SQUIGGLE"
}

def import_data():
    if not SOURCE_DIR.exists():
        print(f"Source {SOURCE_DIR} not found.")
        return

    count = 0
    for subdir in SOURCE_DIR.iterdir():
        if not subdir.is_dir():
            continue
        
        # Original format: 1-green-empty-diamond
        parts = subdir.name.split('-')
        if len(parts) != 4:
            print(f"Skipping unexpected directory: {subdir.name}")
            continue
            
        try:
            target_parts = [MAP[p] for p in parts]
            target_dir = TARGET_ROOT.joinpath(*target_parts)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for img in subdir.glob("*.jpg"):
                # Prefix with ext_ so we know it came from the external set
                target_file = target_dir / f"ext_{img.name}"
                shutil.copy2(img, target_file)
                count += 1
        except KeyError as e:
            print(f"Mapping error for {subdir.name}: {e}")

    print(f"Imported {count} images.")

if __name__ == "__main__":
    import_data()
