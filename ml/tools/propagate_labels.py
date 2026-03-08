import os
import shutil
from pathlib import Path

def propagate_labels(labels_file, mapping_file, chips_dir, output_base):
    # 1. Load labels for representative images
    labels = {}
    with open(labels_file, 'r') as f:
        for line in f:
            if ':' not in line: continue
            name, label = line.strip().split(': ', 1)
            labels[name] = label

    # 2. Process mapping and copy chips
    with open(mapping_file, 'r') as f:
        for line in f:
            if ' -> ' not in line: continue
            repr_name, members_str = line.strip().split(' -> ')
            
            label = labels.get(repr_name)
            if not label:
                continue
            
            if label == "NONE":
                # For negative examples, we use a "ZERO" count directory
                label_path = "ZERO/NONE/NONE/NONE"
            else:
                # Label format: COUNT COLOR PATTERN SHAPE (e.g., ONE RED SOLID OVAL)
                label_path = label.replace(" ", "/")
                
            target_dir = Path(output_base) / label_path
            target_dir.mkdir(parents=True, exist_ok=True)
            
            members = members_str.split(',')
            for member_path in members:
                # members in mapping.txt are like "chips/chip_..."
                # but we moved chips/ to set-finder/ml/raw_data/chips/
                source_file = Path("set-finder/ml/raw_data") / member_path
                if source_file.exists():
                    target_file = target_dir / source_file.name
                    shutil.copy2(source_file, target_file)
                    # print(f"Copied {source_file.name} to {label_path}")

if __name__ == "__main__":
    propagate_labels(
        "set-finder/ml/raw_data/labeling_pool/labels.txt",
        "set-finder/ml/raw_data/labeling_pool/mapping.txt",
        "set-finder/ml/raw_data/chips",
        "set-finder/ml/dataset"
    )
