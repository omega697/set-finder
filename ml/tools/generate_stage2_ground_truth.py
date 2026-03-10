import cv2
import numpy as np
import os
import json
from pathlib import Path

# --- Logic copied exactly from chip_extractor.py ---

def rectify(pts):
    pts = pts.reshape((4, 2))
    new_pts = np.zeros((4, 2), dtype=np.float32)
    add = pts.sum(1)
    new_pts[0] = pts[np.argmin(add)]
    new_pts[2] = pts[np.argmax(add)]
    diff = np.diff(pts, axis=1)
    new_pts[1] = pts[np.argmin(diff)]
    new_pts[3] = pts[np.argmax(diff)]
    return new_pts

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]); yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interWidth = max(0, xB - xA); interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

class SetChipExtractor:
    def __init__(self, target_width=144, target_height=224):
        self.target_width = target_width
        self.target_height = target_height

    def find_candidates(self, img):
        # chip_extractor.py uses BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        frame_area = img.shape[0] * img.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < frame_area / 500 or area > frame_area / 2: continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx): candidates.append(approx)
        candidates = sorted(candidates, key=cv2.contourArea, reverse=True)
        unique = []
        for cand in candidates:
            box = cv2.boundingRect(cand); is_dup = False
            for u in unique:
                if get_iou(box, cv2.boundingRect(u)) > 0.7: is_dup = True; break
            if not is_dup: unique.append(cand)
        return unique

    def unwarp(self, img, contour):
        pts = contour.reshape(4, 2).astype(np.float32); rect = rectify(pts)
        (tl, tr, br, bl) = rect
        width = np.linalg.norm(tr - tl); height = np.linalg.norm(bl - tl)
        if width > height: rect = np.array([bl, tl, tr, br], dtype="float32")
        dst = np.array([[0, 0], [self.target_width - 1, 0], [self.target_width - 1, self.target_height - 1], [0, self.target_height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (self.target_width, self.target_height))

    def apply_white_balance(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        a_mean = np.mean(a); b_mean = np.mean(b)
        a = cv2.add(a, int(128 - a_mean))
        b = cv2.add(b, int(128 - b_mean))
        balanced = cv2.merge([l, a, b])
        return cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)

# --- End logic copy ---

def generate_truth():
    SCENES_DIR = Path("set-finder/app/src/androidTest/assets/scenes")
    OUTPUT_ROOT = Path("set-finder/app/src/androidTest/assets/references/extracted_chips")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    extractor = SetChipExtractor()
    manifest = {}

    for scene_path in SCENES_DIR.glob("*.jpg"):
        # Load as BGR (Standard Python OpenCV)
        img = cv2.imread(str(scene_path))
        if img is None: continue
        
        # chip_extractor.py uses 1000px max dim for candidate search
        scale = 1000.0 / max(img.shape[:2])
        small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        candidates = extractor.find_candidates(small)
        
        scene_manifest = []
        for i, quad in enumerate(candidates):
            # Scale quad back to full-res
            quad_full = (quad / scale).reshape(4, 2).tolist()
            
            # Extract chip using canonical Python logic
            # This is BGR input -> BGR output
            chip = extractor.unwarp(img, np.array(quad_full, dtype=np.float32))
            chip = extractor.apply_white_balance(chip)
            
            ref_name = f"{scene_path.stem}_ref_{i}.jpg"
            # Save reference chip as BGR
            cv2.imwrite(str(OUTPUT_ROOT / ref_name), chip)
            
            scene_manifest.append({
                "quad": quad_full,
                "reference": ref_name
            })
        
        manifest[scene_path.name] = scene_manifest
        print(f"Processed {scene_path.name}: {len(scene_manifest)} cards.")

    with open(OUTPUT_ROOT / "extracted_chips_ground_truth.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    generate_truth()
