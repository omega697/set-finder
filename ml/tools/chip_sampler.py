import cv2
import numpy as np
import os
import shutil
from pathlib import Path

def get_image_signature(img):
    """Computes a signature for the image based on color histograms and a thumbnail."""
    # 1. Color signature (HSV histogram)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    
    # 2. Structural signature (Tiny thumbnail)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    
    return hist.flatten(), thumb.flatten()

def compare_signatures(sig1, sig2):
    """Returns a similarity score between 0 and 1."""
    hist1, thumb1 = sig1
    hist2, thumb2 = sig2
    
    # Histogram correlation (high is good)
    h_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Thumbnail MSE (low is good)
    t_sim = 1.0 - (np.mean((thumb1.astype(float) - thumb2.astype(float))**2) / 65025.0)
    
    # Weighted average
    return (h_sim * 0.6) + (t_sim * 0.4)

def sample_chips(input_dir, output_dir, threshold=0.92):
    """Clusters similar images and saves representatives for labeling."""
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = sorted(list(Path(input_dir).glob("*.jpg")))
    print(f"Analyzing {len(image_paths)} chips...")
    
    representatives = [] # List of (path, signature)
    mapping = {} # repr_path -> [member_paths]

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        sig = get_image_signature(img)
        
        # Check against existing representatives
        found_cluster = False
        # We only check the last 20 clusters to keep it fast (assuming video continuity)
        # and because most duplicates are temporal.
        for r_path, r_sig in representatives[-30:]:
            if compare_signatures(sig, r_sig) > threshold:
                mapping[r_path].append(str(img_path))
                found_cluster = True
                break
        
        if not found_cluster:
            representatives.append((str(img_path), sig))
            mapping[str(img_path)] = [str(img_path)]
            
        if i % 100 == 0:
            print(f"Processed {i}/{len(image_paths)} chips. Current unique pool: {len(representatives)}")

    # Copy representatives to the pool
    for i, (r_path, _) in enumerate(representatives):
        new_name = f"repr_{i:04d}.jpg"
        shutil.copy(r_path, os.path.join(output_dir, new_name))
        
    print(f"\nSampling Complete!")
    print(f"Original: {len(image_paths)} chips")
    print(f"Unique Pool: {len(representatives)} chips")
    print(f"Represented images saved to: {output_dir}")
    
    # Save a mapping file so we can propagate labels later
    with open(os.path.join(output_dir, "mapping.txt"), "w") as f:
        for i, (r_path, _) in enumerate(representatives):
            members = ",".join(mapping[r_path])
            f.write(f"repr_{i:04d}.jpg -> {members}\n")

if __name__ == "__main__":
    sample_chips("chips", "labeling_pool")
