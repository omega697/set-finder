import cv2
import numpy as np
from pathlib import Path
import os
import concurrent.futures

def white_balance(img):
    if img is None: return None
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Use the top 10% brightest pixels as the reference for "white"
    mask = l > np.percentile(l, 90)
    if not np.any(mask):
        return img
        
    avg_a = np.mean(a[mask])
    avg_b = np.mean(b[mask])
    
    # Shift a and b to be centered around 128 (neutral in LAB)
    a_shifted = cv2.add(a, int(128 - avg_a))
    b_shifted = cv2.add(b, int(128 - avg_b))
    
    normalized_lab = cv2.merge([l, a_shifted, b_shifted])
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

def process_file(path):
    img = cv2.imread(str(path))
    balanced = white_balance(img)
    if balanced is not None:
        cv2.imwrite(str(path), balanced)

def process_directory(root_dir):
    print(f"Normalizing all images in {root_dir}...")
    paths = list(Path(root_dir).rglob("*.jpg"))
    
    # Use ThreadPoolExecutor for faster processing of 15k images
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for i, _ in enumerate(executor.map(process_file, paths)):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(paths)} images...")

if __name__ == "__main__":
    process_directory("set-finder/ml/dataset")
    # Also normalize predictions and raw chips just in case
    if Path("set-finder/ml/predictions").exists():
        process_directory("set-finder/ml/predictions")
    if Path("set-finder/ml/raw_data/chips").exists():
        process_directory("set-finder/ml/raw_data/chips")
