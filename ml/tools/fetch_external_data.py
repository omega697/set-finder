import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# External data source from tomwhite/set-game
REPO_URL = "https://github.com/tomwhite/set-game.git"
SUBDIR_PATH = "data/train-v2/labelled"

# Local target
ML_ROOT = Path(__file__).parent.parent.resolve()
TARGET_ROOT = ML_ROOT / "dataset"

# Mapping from set-game naming to our naming
MAP = {
    "1": "ONE", "2": "TWO", "3": "THREE",
    "red": "RED", "green": "GREEN", "purple": "PURPLE",
    "empty": "EMPTY", "solid": "SOLID", "striped": "SHADED",
    "diamond": "DIAMOND", "oval": "OVAL", "squiggle": "SQUIGGLE",
    "diamonds": "DIAMOND", "ovals": "OVAL", "squiggles": "SQUIGGLE"
}

def fetch_and_import():
    print(f"Fetching external data from {REPO_URL}...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # We use a sparse checkout to only get the labelled data we need
        try:
            subprocess.run([
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                REPO_URL, str(tmp_path)
            ], check=True)
            
            subprocess.run([
                "git", "-C", str(tmp_path), "sparse-checkout", "set", SUBDIR_PATH
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            return

        source_dir = tmp_path / SUBDIR_PATH
        if not source_dir.exists():
            print(f"Error: Source directory {SUBDIR_PATH} not found in repository.")
            return

        print("Importing images...")
        count = 0
        for subdir in source_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            # Original format: 1-green-empty-diamond
            parts = subdir.name.split('-')
            if len(parts) != 4:
                continue
                
            try:
                target_parts = [MAP[p] for p in parts]
                target_dir = TARGET_ROOT.joinpath(*target_parts)
                target_dir.mkdir(parents=True, exist_ok=True)
                
                for img in subdir.glob("*.jpg"):
                    # Prefix with ext_ so it's ignored by Git
                    target_file = target_dir / f"ext_{img.name}"
                    shutil.copy2(img, target_file)
                    count += 1
            except KeyError as e:
                # Some subdirs might not match our 4-attribute pattern, skip them
                pass

        print(f"Successfully imported {count} external images to {TARGET_ROOT}")
        print("These files are prefixed with 'ext_' and are ignored by Git.")

if __name__ == "__main__":
    fetch_and_import()
