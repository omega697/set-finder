import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from segment_anything import sam_model_registry, SamPredictor
import time
from pathlib import Path

import sys

# --- CONFIG ---
CHECKPOINT = "ml/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cpu" 
VIDEOS = [
    "/usr/local/google/home/kdresner/Downloads/PXL_20260308_003730588.mp4",
    "/usr/local/google/home/kdresner/Downloads/PXL_20260306_221751495.mp4",
    "/usr/local/google/home/kdresner/Downloads/PXL_20260306_223532288.mp4"
]
FILTER_MODEL_PATH = "ml/models/card_filter_v14.keras"
OUTPUT_DIR = "ml/dataset/yolo_pose"
LOG_DIR = "ml/tools/logs"
THRESHOLD = 0.90
FRAME_STEP = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Redirect output to log file
log_file = os.path.join(LOG_DIR, "bulk_label.log")
print(f"Logging to {log_file}")
f = open(log_file, 'w')
sys.stdout = f
sys.stderr = f

# --- MODELS ---
print("Initializing SAM Predictor...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

print("Loading Card Filter...")
card_filter = tf.keras.models.load_model(FILTER_MODEL_PATH)

def get_card_crop(img, quad, size=(224, 224)):
    pts = np.array(quad, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    dst = np.array([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, size)

def intersect_lines(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    return np.array([x1 + ua*(x2-x1), y1 + ua*(y2-y1)])

def fit_outer_quad(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(float)
    while len(hull) > 4:
        best_idx = -1; min_added_area = float('inf'); best_new_pt = None
        for i in range(len(hull)):
            p_m1, p_0, p_p1, p_p2 = hull[i-1], hull[i], hull[(i+1)%len(hull)], hull[(i+2)%len(hull)]
            new_pt = intersect_lines(p_m1, p_0, p_p1, p_p2)
            if new_pt is not None:
                area = 0.5 * abs(p_0[0]*(p_p1[1]-new_pt[1]) + p_p1[0]*(new_pt[1]-p_0[1]) + new_pt[0]*(p_0[1]-p_p1[1]))
                if area < min_added_area:
                    min_added_area = area; best_idx = i; best_new_pt = new_pt
        if best_idx == -1: break
        next_idx = (best_idx + 1) % len(hull)
        if next_idx > best_idx:
            hull = np.delete(hull, [best_idx, next_idx], axis=0)
            hull = np.insert(hull, best_idx, best_new_pt, axis=0)
        else:
            hull = np.delete(hull, [best_idx, next_idx], axis=0)
            hull = np.append(hull, [best_new_pt], axis=0)
    return hull.astype(int)

def auto_detect(frame):
    h, w = frame.shape[:2]
    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Broaden further: lower value threshold, higher saturation allowance
    seed_mask = cv2.bitwise_and(cv2.threshold(hsv[:,:,2], 120, 255, cv2.THRESH_BINARY)[1], 
                               cv2.threshold(hsv[:,:,1], 80, 255, cv2.THRESH_BINARY_INV)[1])
    
    # Exclude edges (avoid table edges/glare)
    margin = 50
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(edge_mask, (margin, margin), (w-margin, h-margin), 255, -1)
    seed_mask = cv2.bitwise_and(seed_mask, edge_mask)

    exclusion_mask = np.zeros((h, w), dtype=np.uint8)
    found_quads = []
    
    for i in range(15): 
        active_seeds = cv2.bitwise_and(seed_mask, cv2.bitwise_not(exclusion_mask))
        dist = cv2.distanceTransform(active_seeds, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        if max_val < 15: break
            
        # SAM from this peak with a negative point to discourage background
        neg_pt = [max_loc[0] + 250, max_loc[1] + 250]
        if neg_pt[0] >= w: neg_pt[0] = max_loc[0] - 250
        if neg_pt[1] >= h: neg_pt[1] = max_loc[1] - 250
        
        masks, scores, _ = predictor.predict(
            point_coords=np.array([max_loc, neg_pt]),
            point_labels=np.array([1, 0]),
            multimask_output=True
        )
        
        # Pick the SMALLEST mask that is within our area bounds
        best_mask = None
        for m in masks:
            a = np.sum(m) / (h * w)
            if 0.002 < a < 0.25:
                if best_mask is None or a < (np.sum(best_mask) / (h * w)):
                    best_mask = m
        
        if best_mask is None:
            # If all multimasks are out of bounds, check the highest score one just to log why
            top_idx = np.argmax(scores)
            top_a = np.sum(masks[top_idx]) / (h * w)
            print(f"      Iter {i}: Rejected (all masks out of bounds, top area {top_a:.4f})")
            cv2.circle(exclusion_mask, max_loc, int(max_val) + 5, 255, -1)
            continue

        mask = best_mask
        area_ratio = np.sum(mask) / (h * w)
        
        quad = fit_outer_quad(mask.astype(np.uint8)*255)
        if quad is not None:
            crop = get_card_crop(frame, quad)
            crop_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(crop.astype(np.float32))
            prob = card_filter.predict(np.expand_dims(crop_preprocessed, 0), verbose=0)[0][0]
            if prob > THRESHOLD:
                found_quads.append(quad.tolist())
                cv2.fillPoly(exclusion_mask, [np.array(quad, dtype=np.int32)], 255)
                print(f"      Iter {i}: Found card (prob={prob:.4f}, area={area_ratio:.4f})")
                continue
            else:
                print(f"      Iter {i}: Rejected (prob={prob:.4f}, area={area_ratio:.4f})")
        else:
            print(f"      Iter {i}: Rejected (no quad fitted, area={area_ratio:.4f})")

        cv2.circle(exclusion_mask, max_loc, int(max_val) + 5, 255, -1)
    return found_quads

def save_labels(frame, quads, filename):
    img_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(img_path, frame)
    
    label_path = os.path.join(OUTPUT_DIR, filename.replace('.jpg', '.txt'))
    h, w = frame.shape[:2]
    
    with open(label_path, 'w') as f:
        for q in quads:
            q = np.array(q)
            x_min, y_min = np.min(q, axis=0)
            x_max, y_max = np.max(q, axis=0)
            bw, bh = (x_max - x_min), (y_max - y_min)
            cx, cy = x_min + bw/2, y_min + bh/2
            
            line = f"0 {cx/w} {cy/h} {bw/w} {bh/h}"
            for pt in q:
                line += f" {pt[0]/w} {pt[1]/h} 2"
            f.write(line + "\n")

for video_path in VIDEOS:
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    video_name = Path(video_path).stem
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % FRAME_STEP == 0:
            print(f"  Frame {frame_idx}...")
            quads = auto_detect(frame)
            if quads:
                filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
                save_labels(frame, quads, filename)
                print(f"    Found {len(quads)} cards.")
            else:
                print(f"    No cards found.")
        
        frame_idx += 1
    
    cap.release()

print("Bulk processing complete!")
