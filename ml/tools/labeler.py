import http.server
import json
import os
import shutil
import urllib.parse
from pathlib import Path
import re
import numpy as np
import tensorflow as tf

# Configuration
TOOLS_ROOT = Path(__file__).parent
DATASET_ROOT = Path("set-finder/ml/dataset")
PRED_ROOT = Path("set-finder/ml/predictions") 
EXPERT_MODEL_PATH = "set-finder/ml/attribute_expert_final.keras"
WEB_ROOT = TOOLS_ROOT / "labeler_web"
IMG_SIZE = (224, 224)
PORT = 5000

MAPS = {
    'count':   {0: 'ONE', 1: 'TWO', 2: 'THREE'},
    'color':   {0: 'RED', 1: 'GREEN', 2: 'PURPLE'},
    'pattern': {0: 'SOLID', 1: 'SHADED', 2: 'EMPTY'},
    'shape':   {0: 'OVAL', 1: 'DIAMOND', 2: 'SQUIGGLE'}
}

EXPERT_MAPS = {
    'count':   {1: 'ONE', 2: 'TWO', 3: 'THREE'},
    'color':   {1: 'RED', 2: 'GREEN', 3: 'PURPLE'},
    'pattern': {1: 'SOLID', 2: 'SHADED', 3: 'EMPTY'},
    'shape':   {1: 'OVAL', 2: 'DIAMOND', 3: 'SQUIGGLE'}
}

# Global state
m_expert = None
SKIPPED_FOLDERS = set()

def load_expert():
    global m_expert
    if m_expert is None:
        print(f"Loading expert model from {EXPERT_MODEL_PATH}...")
        try: m_expert = tf.keras.models.load_model(EXPERT_MODEL_PATH)
        except: print("Warning: Could not load expert model.")

def parse_original_label(filename):
    match = re.match(r"from_([^_]+)_([^_]+)_([^_]+)_([^_]+)_", filename)
    if match: return " ".join(match.groups())
    return None

def get_stats():
    reviewed = 0; total = 0
    for path in DATASET_ROOT.rglob("*.jpg"):
        total += 1
        if path.name.startswith("ok_") or path.name.startswith("ext_"): reviewed += 1
    if PRED_ROOT.exists():
        for path in PRED_ROOT.rglob("*.jpg"): total += 1
    return {"reviewed": reviewed, "total": total}

class LabelerHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == "/":
                self.send_response(200); self.send_header("Content-type", "text/html"); self.end_headers()
                with open(WEB_ROOT / "index.html", "r") as f: self.wfile.write(f.read().encode())
            elif self.path.startswith("/static/"):
                filename = self.path[8:]
                file_path = WEB_ROOT / filename
                if file_path.exists():
                    self.send_response(200)
                    if filename.endswith(".js"): self.send_header("Content-type", "application/javascript")
                    elif filename.endswith(".css"): self.send_header("Content-type", "text/css")
                    self.end_headers()
                    with open(file_path, "rb") as f: self.wfile.write(f.read())
                else: self.send_error(404)
            elif self.path == "/next":
                data = self.get_next_image_data(); data["stats"] = get_stats()
                self.send_response(200); self.send_header("Content-type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            elif self.path == "/bulk_list":
                data = self.get_bulk_list()
                if data: data["stats"] = get_stats()
                self.send_response(200); self.send_header("Content-type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            elif self.path.startswith("/images/"):
                image_path = urllib.parse.unquote(self.path[8:])
                full_path = PRED_ROOT / image_path
                if not full_path.exists(): full_path = DATASET_ROOT / image_path
                if full_path.exists():
                    self.send_response(200); self.send_header("Content-type", "image/jpeg"); self.end_headers()
                    with open(full_path, "rb") as f: self.wfile.write(f.read())
                else: self.send_error(404)
            else: self.send_error(404)
        except Exception as e:
            print(f"Error GET: {e}"); self.send_error(500)

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(content_length).decode())
            if self.path == "/move":
                success = self.move_image(data)
                self.send_response(200); self.send_header("Content-type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"success": success}).encode())
            elif self.path == "/confirm_bulk":
                success = self.confirm_bulk(data)
                self.send_response(200); self.send_header("Content-type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"success": success}).encode())
            elif self.path == "/rescue":
                success = self.rescue_images(data)
                self.send_response(200); self.send_header("Content-type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"success": success}).encode())
            elif self.path == "/skip":
                SKIPPED_FOLDERS.add(data["folder"])
                self.send_response(200); self.send_header("Content-type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode())
        except Exception as e:
            print(f"Error POST: {e}"); self.send_error(500)

    def get_bulk_list(self):
        if not PRED_ROOT.exists(): return None
        valid_dirs = [p for p in PRED_ROOT.rglob("*") if p.is_dir() and any(p.glob("*.jpg"))]
        if not valid_dirs: return None
        valid_dirs.sort(key=lambda p: str(p.relative_to(PRED_ROOT)))
        for path in valid_dirs:
            rel_path = str(path.relative_to(PRED_ROOT))
            if rel_path in SKIPPED_FOLDERS: continue
            images = list(path.glob("*.jpg"))
            parts = list(path.relative_to(PRED_ROOT).parts)
            while len(parts) < 4: parts.append("NONE")
            img_data = [{"path": str(img.relative_to(PRED_ROOT)), "original": parse_original_label(img.name)} for img in images[:50]]
            return { "label": " ".join(parts[:4]), "label_parts": parts[:4], "images": img_data, "folder": rel_path }
        return None

    def confirm_bulk(self, data):
        for item in data["items"]:
            src = PRED_ROOT / item["path"]
            if src.exists():
                if item["count"] == "ZERO" or "NONE" in [item["color"], item["pattern"], item["shape"]]:
                    dest_dir = DATASET_ROOT / "non_cards"
                else:
                    dest_dir = DATASET_ROOT / "cards" / item["count"] / item["color"] / item["pattern"] / item["shape"]
                dest_dir.mkdir(parents=True, exist_ok=True)
                name = src.name
                if "ok_chip_" in name: name = name[name.find("ok_chip_"):]
                shutil.move(src, dest_dir / (f"ok_{name}" if not name.startswith("ok_") else name))
        return True

    def rescue_images(self, data):
        load_expert()
        if m_expert is None: return False
        selected_items = [i for i in data["items"] if i.get("selected")]
        tensors = []
        valid_paths = []
        for item in selected_items:
            p = PRED_ROOT / item["path"]
            if p.exists():
                img = tf.io.read_file(str(p))
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, IMG_SIZE)
                tensors.append(tf.keras.applications.mobilenet_v2.preprocess_input(img))
                valid_paths.append(p)
        if not tensors: return True
        preds = m_expert.predict(tf.stack(tensors), verbose=0)
        for i, path in enumerate(valid_paths):
            lbl = [EXPERT_MAPS['count'].get(np.argmax(preds[1][i]), "NONE"),
                   EXPERT_MAPS['color'].get(np.argmax(preds[0][i]), "NONE"),
                   EXPERT_MAPS['pattern'].get(np.argmax(preds[2][i]), "NONE"),
                   EXPERT_MAPS['shape'].get(np.argmax(preds[3][i]), "NONE")]
            target = PRED_ROOT.joinpath(*lbl) if "NONE" not in lbl else PRED_ROOT / "RESCUE_FAILED"
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(path, target / path.name)
        return True

    def get_next_image_data(self):
        for path in DATASET_ROOT.rglob("*.jpg"):
            if not (path.name.startswith("ok_") or path.name.startswith("ext_")):
                rel = path.relative_to(DATASET_ROOT); parts = list(rel.parts); fname = parts.pop()
                if "cards" in parts: parts.remove("cards")
                attrs = parts + ["NONE"] * (4 - len(parts))
                return {"path": str(rel), "count": attrs[0], "color": attrs[1], "pattern": attrs[2], "shape": attrs[3], "filename": fname, "original": parse_original_label(fname)}
        return {"image": None}

    def move_image(self, data):
        src = DATASET_ROOT / data["old_path"]
        if not src.exists(): src = PRED_ROOT / data["old_path"]
        if data["count"] == "ZERO" or "NONE" in [data["color"], data["pattern"], data["shape"]]:
            dest = DATASET_ROOT / "non_cards"
        else:
            dest = DATASET_ROOT / "cards" / data["count"] / data["color"] / data["pattern"] / data["shape"]
        dest.mkdir(parents=True, exist_ok=True)
        name = data['filename']
        if "ok_chip_" in name: name = name[name.find("ok_chip_"):]
        if src.exists():
            shutil.move(src, dest / (f"ok_{name}" if not name.startswith("ok_") else name))
            return True
        return False

if __name__ == "__main__":
    server_address = ('', PORT)
    httpd = http.server.HTTPServer(server_address, LabelerHandler)
    print(f"Starting Dataset Cleaner v12.0 at http://localhost:{PORT}")
    httpd.serve_forever()
