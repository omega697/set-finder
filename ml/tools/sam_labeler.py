import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from segment_anything import sam_model_registry, SamPredictor
import io

app = Flask(__name__, static_folder='sam_labeler_web', static_url_path='')

@app.route('/')
def index():
    return send_file('sam_labeler_web/index.html')

# --- CONFIG ---
CHECKPOINT = "ml/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cpu" 
VIDEO_DIR = "ml/raw_data"
OUTPUT_DIR = "ml/dataset/yolo_pose"
FILTER_MODEL_PATH = "ml/models/card_filter_v14.keras"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODELS ---
print("Initializing SAM Predictor...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

print("Loading Card Filter...")
try:
    card_filter = tf.keras.models.load_model(FILTER_MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load card filter: {e}")
    card_filter = None

# --- GEOMETRY UTILS ---
def get_card_crop(img, quad, size=(224, 224)):
    # quad: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    # Ensure quad points are in consistent order (top-left, top-right, bottom-right, bottom-left)
    # For now, we'll use a simple sort, but a more robust corner sorter is better.
    pts = np.array(quad, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    
    dst = np.array([
        [0, 0],
        [size[0] - 1, 0],
        [size[0] - 1, size[1] - 1],
        [0, size[1] - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, size)
    return warped

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
        best_idx = -1
        min_added_area = float('inf')
        best_new_pt = None
        for i in range(len(hull)):
            p_m1, p_0, p_p1, p_p2 = hull[i-1], hull[i], hull[(i+1)%len(hull)], hull[(i+2)%len(hull)]
            new_pt = intersect_lines(p_m1, p_0, p_p1, p_p2)
            if new_pt is not None:
                area = 0.5 * abs(p_0[0]*(p_p1[1]-new_pt[1]) + p_p1[0]*(new_pt[1]-p_0[1]) + new_pt[0]*(p_0[1]-p_p1[1]))
                if area < min_added_area:
                    min_added_area = area
                    best_idx = i
                    best_new_pt = new_pt
        if best_idx == -1: break
        next_idx = (best_idx + 1) % len(hull)
        if next_idx > best_idx:
            hull = np.delete(hull, [best_idx, next_idx], axis=0)
            hull = np.insert(hull, best_idx, best_new_pt, axis=0)
        else:
            hull = np.delete(hull, [best_idx, next_idx], axis=0)
            hull = np.append(hull, [best_new_pt], axis=0)
    return hull.astype(int)

def get_mask_contrast(image_gray, mask):
    m_uint8 = mask.astype(np.uint8) * 255
    inner_mean = cv2.mean(image_gray, mask=m_uint8)[0]
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(m_uint8, kernel, iterations=2)
    buffer_mask = cv2.subtract(dilated, m_uint8)
    outer_mean = cv2.mean(image_gray, mask=buffer_mask)[0]
    return inner_mean - outer_mean

# --- STATE ---
cap = None
current_frame = None
current_gray = None
current_frame_idx = -1

@app.route('/api/load_video', methods=['POST'])
def load_video():
    global cap
    path = request.json.get('path')
    full_path = os.path.join(VIDEO_DIR, path)
    if not os.path.exists(full_path):
        return jsonify({"error": f"File not found: {path}"}), 404
    
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(full_path)
    return jsonify({"status": "ok", "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})

@app.route('/api/get_frame/<int:index>')
def get_frame(index):
    global current_frame, current_gray, current_frame_idx, cap
    
    if cap is None:
        # Fallback to test image for easy initial verification
        img_path = "app/src/androidTest/assets/scenes/card_3_red_solid_oval.jpg"
        current_frame = cv2.imread(img_path)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, current_frame = cap.read()
        if not ret:
            return jsonify({"error": "Failed to read frame"}), 400
            
    current_frame_idx = index
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    predictor.set_image(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
    
    _, buffer = cv2.imencode('.jpg', current_frame)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

@app.route('/api/save', methods=['POST'])
def save_labels():
    data = request.json # { filename: "...", quads: [[ [x1,y1], ... ], ...] }
    filename = data.get('filename', f"frame_{current_frame_idx}.jpg")
    quads = data.get('quads', [])
    
    # Save image
    img_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(img_path, current_frame)
    
    # Save YOLO Pose label
    # Format: class_id x_center y_center width height x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
    # (Assuming class_id 0 for Card)
    label_path = os.path.join(OUTPUT_DIR, filename.replace('.jpg', '.txt'))
    h, w = current_frame.shape[:2]
    
    with open(label_path, 'w') as f:
        for q in quads:
            q = np.array(q)
            # Bounding box
            x_min, y_min = np.min(q, axis=0)
            x_max, y_max = np.max(q, axis=0)
            bw, bh = (x_max - x_min), (y_max - y_min)
            cx, cy = x_min + bw/2, y_min + bh/2
            
            # Normalize
            line = f"0 {cx/w} {cy/h} {bw/w} {bh/h}"
            # Keypoints (v=2 means visible and labeled)
            for pt in q:
                line += f" {pt[0]/w} {pt[1]/h} 2"
            f.write(line + "\n")
            
    return jsonify({"status": "saved", "path": label_path})

@app.route('/api/click', methods=['POST'])
def handle_click():
    data = request.json
    pt = [data['x'], data['y']]
    
    # 1. Primary SAM pass
    masks, scores, _ = predictor.predict(np.array([pt]), np.array([1]), multimask_output=True)
    
    # 2. Select best mask (Expand if symbol)
    best_mask_idx = np.argmax(scores)
    contrast = get_mask_contrast(current_gray, masks[best_mask_idx])
    
    if contrast < -20: # Clicked on a symbol
        print("Expanding from symbol to card...")
        offsets = [[150,0], [-150,0], [0,150], [0,-150]]
        best_expanded_mask = None
        best_s = -1
        for off in offsets:
            new_pt = [pt[0] + off[0], pt[1] + off[1]]
            m, s, _ = predictor.predict(np.array([pt, new_pt]), np.array([0, 1]), multimask_output=True)
            idx = np.argmax(s)
            if s[idx] > best_s:
                best_s = s[idx]
                best_expanded_mask = m[idx]
        final_mask = best_expanded_mask
    else:
        final_mask = masks[best_mask_idx]
        
    # 3. Fit Outer Quad
    quad = fit_outer_quad(final_mask.astype(np.uint8)*255)
    
    if quad is not None:
        return jsonify({"quad": quad.tolist()})
    return jsonify({"error": "No quad found"}), 400

@app.route('/api/auto_detect', methods=['POST'])
def auto_detect():
    import time
    start_all = time.time()
    global current_frame, current_gray
    if current_frame is None:
        return jsonify({"error": "No frame loaded"}), 400
    
    if card_filter is None:
        return jsonify({"error": "Card filter model not loaded"}), 500

    h, w = current_frame.shape[:2]
    hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    s_channel = hsv[:,:,1]
    
    # Whitish pixels: High value, low saturation
    white_mask = cv2.threshold(v_channel, 180, 255, cv2.THRESH_BINARY)[1]
    low_sat = cv2.threshold(s_channel, 50, 255, cv2.THRESH_BINARY_INV)[1]
    seed_mask = cv2.bitwise_and(white_mask, low_sat)
    
    exclusion_mask = np.zeros((h, w), dtype=np.uint8)
    found_quads = []
    
    # Max iterations to find cards
    for i in range(12): 
        iter_start = time.time()
        active_seeds = cv2.bitwise_and(seed_mask, cv2.bitwise_not(exclusion_mask))
        dist = cv2.distanceTransform(active_seeds, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        
        if max_val < 15: 
            break
            
        # SAM from this peak
        masks, scores, _ = predictor.predict(np.array([max_loc]), np.array([1]), multimask_output=True)
        idx = np.argmax(scores)
        mask = masks[idx]
        
        # Area constraint
        area_ratio = np.sum(mask) / (h * w)
        if 0.005 < area_ratio < 0.25:
            quad = fit_outer_quad(mask.astype(np.uint8)*255)
            if quad is not None:
                crop = get_card_crop(current_frame, quad)
                crop_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(crop.astype(np.float32))
                prob = card_filter.predict(np.expand_dims(crop_preprocessed, 0), verbose=0)[0][0]
                
                if prob > 0.90:
                    found_quads.append(quad.tolist())
                    cv2.fillPoly(exclusion_mask, [np.array(quad, dtype=np.int32)], 255)
                    print(f"Iter {i}: Found card (prob={prob:.4f}) in {time.time()-iter_start:.2f}s")
                    continue

        cv2.circle(exclusion_mask, max_loc, int(max_val) + 5, 255, -1)
        print(f"Iter {i}: No card in {time.time()-iter_start:.2f}s")

    print(f"Auto-detect total: {time.time()-start_all:.2f}s")
    return jsonify({"quads": found_quads})

if __name__ == '__main__':
    app.run(port=5000, debug=False)
