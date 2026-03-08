import cv2
import numpy as np
import os
from pathlib import Path

def extract_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None: return None
    
    # 1. Color Features (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_h = np.mean(hsv[:,:,0])
    avg_s = np.mean(hsv[:,:,1])
    avg_v = np.mean(hsv[:,:,2])
    
    # 2. Count/Shape Features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold to find shapes
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small noise contours
    valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
    num_shapes = len(valid_contours)
    
    # Shape description (using the largest contour)
    shape_ratio = 0
    if valid_contours:
        c = max(valid_contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        if perimeter > 0:
            # Compactness/Circularity-like measure
            shape_ratio = (perimeter * perimeter) / area
            
    # 3. Pattern/Texture (Internal variance)
    # Mask out everything except the shapes
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, valid_contours, -1, 255, -1)
    # Calculate standard deviation of pixels inside shapes
    std_dev = 0
    if np.any(mask):
        pixels = gray[mask == 255]
        std_dev = np.std(pixels)

    return np.array([avg_h, avg_s, avg_v, num_shapes, shape_ratio, std_dev])

if __name__ == "__main__":
    # Test on a few files
    for path in list(Path("set-finder/ml/dataset").rglob("ext_*.jpg"))[:5]:
        feats = extract_features(path)
        print(f"{path.parent.name}: {feats}")
