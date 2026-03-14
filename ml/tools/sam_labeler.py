import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from segment_anything import sam_model_registry, SamPredictor
import io
import json
import base64
from pathlib import Path

# Import shared standardized logic from cv_library.py
from cv_library import unwarp, apply_white_balance, rectify
from model_downloader import ensure_checkpoint

app = Flask(__name__, static_folder='sam_labeler_web', static_url_path='')

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CHECKPOINT = os.path.join(SCRIPT_DIR, "temp", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
DEVICE = "cpu" 
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset/yolo_pose")
EXPORT_DIR = os.path.join(PROJECT_ROOT, "dataset/yolo_pose_cleaned")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")
FILTER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/card_filter_v14.keras")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# --- MODELS ---
print("Initializing SAM Predictor...")
checkpoint_path = ensure_checkpoint(CHECKPOINT, MODEL_TYPE)
sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

print("Loading Card Filter...")
try:
    card_filter = tf.keras.models.load_model(FILTER_MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load card filter: {e}")
    card_filter = None

# --- STATE MANAGEMENT ---
def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_manifest(manifest):
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

def get_frame_list():
    frames = sorted(list(Path(DATASET_DIR).glob("*.jpg")))
    return [f.name for f in frames]

# Cache
current_frame_name = None
current_frame_img_bgr = None
current_frame_img_rgb = None
current_predictor_set = False

def intersect_lines(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
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
            hull = np.delete(hull, [best_idx, next_idx], axis=0); hull = np.insert(hull, best_idx, best_new_pt, axis=0)
        else:
            hull = np.delete(hull, [best_idx, next_idx], axis=0); hull = np.append(hull, [best_new_pt], axis=0)
    return hull.astype(int)

# --- ROUTES ---

@app.route('/')
def index():
    return send_file('sam_labeler_web/index.html')

@app.route('/api/init')
def init():
    manifest = load_manifest()
    frames = get_frame_list()
    return jsonify({
        "frames": frames,
        "manifest": manifest
    })

@app.route('/api/frame_image/<filename>')
def frame_image(filename):
    # This serves the image to the browser.
    path = os.path.join(DATASET_DIR, filename)
    img = cv2.imread(path)
    if img is None: return "Not found", 404
    
    _, buffer = cv2.imencode('.jpg', img)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

@app.route('/api/frame_data/<filename>')
def frame_data(filename):
    global current_frame_name, current_frame_img_bgr, current_frame_img_rgb, current_predictor_set
    path = os.path.join(DATASET_DIR, filename)
    img_bgr = cv2.imread(path)
    if img_bgr is None: return jsonify({"error": "File not found"}), 404
    
    current_frame_name = filename
    # Internal representation: BGR for unwarping, and RGB for SAM/TF
    current_frame_img_bgr = img_bgr.copy()
    current_frame_img_rgb = cv2.cvtColor(current_frame_img_bgr, cv2.COLOR_BGR2RGB)
    current_predictor_set = False
    
    quads = []
    txt_path = path.replace(".jpg", ".txt")
    if os.path.exists(txt_path):
        h, w = img_bgr.shape[:2]
        with open(txt_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                # Format: 0 cx cy w h p1x p1y 2 p2x p2y 2 p3x p3y 2 p4x p4y 2
                if len(parts) >= 17:
                    pts = []
                    for i in range(5, 17, 3):
                        pts.append([parts[i] * w, parts[i+1] * h])
                    quads.append(pts)
                    
    return jsonify({
        "quads": quads,
        "is_cleaned": load_manifest().get(filename, False)
    })

@app.route('/api/inspect_quad', methods=['POST'])
def inspect_quad():
    data = request.json
    quad = data['quad']
    if current_frame_img_rgb is None:
        return jsonify({"error": "No frame loaded"}), 400
        
    # Standardized extraction (RGB)
    chip_rgb = unwarp(current_frame_img_rgb, np.array(quad))
    chip_rgb = apply_white_balance(chip_rgb)
    
    prob = -1.0
    if card_filter:
        crop_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(cv2.resize(chip_rgb, (224, 224)).astype(np.float32))
        prob = float(card_filter.predict(np.expand_dims(crop_preprocessed, 0), verbose=0)[0][0])
        
    # Convert to BGR for browser display via base64
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(chip_rgb, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "chip": f"data:image/jpeg;base64,{img_base64}",
        "prob": prob
    })

@app.route('/api/sam_click', methods=['POST'])
def sam_click():
    global current_predictor_set
    data = request.json
    x, y = data['x'], data['y']
    
    if not current_predictor_set:
        predictor.set_image(current_frame_img_rgb)
        current_predictor_set = True
        
    masks, scores, _ = predictor.predict(np.array([[x, y]]), np.array([1]), multimask_output=True)
    best_mask = masks[np.argmax(scores)]
    quad = fit_outer_quad(best_mask.astype(np.uint8)*255)
    
    if quad is not None:
        return jsonify({"quad": quad.tolist()})
    return jsonify({"error": "No quad found"}), 400

@app.route('/api/rectify', methods=['POST'])
def rectify_points():
    pts = np.array(request.json['points'])
    rect = rectify(pts)
    return jsonify({"quad": rect.tolist()})

@app.route('/api/save_frame', methods=['POST'])
def save_frame():
    data = request.json
    filename = data['filename']
    quads = data['quads']
    mark_clean = data.get('mark_clean', False)
    
    path = os.path.join(DATASET_DIR, filename)
    txt_path = path.replace(".jpg", ".txt")
    img = cv2.imread(path)
    h, w = img.shape[:2]
    
    with open(txt_path, 'w') as f:
        for q in quads:
            q = np.array(q)
            x_min, y_min = np.min(q, axis=0); x_max, y_max = np.max(q, axis=0)
            bw, bh = (x_max - x_min), (y_max - y_min); cx, cy = x_min + bw/2, y_min + bh/2
            pts_str = " ".join([f"{pt[0]/w:.6f} {pt[1]/h:.6f} 2" for pt in q])
            f.write(f"0 {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f} {pts_str}\n")
            
    if mark_clean:
        manifest = load_manifest()
        manifest[filename] = True
        save_manifest(manifest)
        
    return jsonify({"status": "ok"})

@app.route('/api/export', methods=['POST'])
def export_cleaned():
    manifest = load_manifest()
    cleaned_count = 0
    for filename, is_cleaned in manifest.items():
        if is_cleaned:
            src_jpg = os.path.join(DATASET_DIR, filename)
            src_txt = src_jpg.replace(".jpg", ".txt")
            if os.path.exists(src_jpg) and os.path.exists(src_txt):
                import shutil
                shutil.copy2(src_jpg, os.path.join(EXPORT_DIR, filename))
                shutil.copy2(src_txt, os.path.join(EXPORT_DIR, filename.replace(".jpg", ".txt")))
                cleaned_count += 1
    return jsonify({"status": "ok", "exported": cleaned_count})

if __name__ == '__main__':
    app.run(port=5000, debug=False)
