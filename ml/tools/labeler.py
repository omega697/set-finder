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
DATASET_ROOT = Path("set-finder/ml/dataset")
PRED_ROOT = Path("set-finder/ml/predictions") 
EXPERT_MODEL_PATH = "set-finder/ml/attribute_expert_final.keras"
IMG_SIZE = (224, 224)
PORT = 5000

MAPS = {
    'count':   {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE'},
    'color':   {0: 'NONE', 1: 'RED', 2: 'GREEN', 3: 'PURPLE'},
    'pattern': {0: 'NONE', 1: 'SOLID', 2: 'SHADED', 3: 'EMPTY'},
    'shape':   {0: 'NONE', 1: 'OVAL', 2: 'DIAMOND', 3: 'SQUIGGLE'}
}

# Global state for session
m_expert = None
SKIPPED_FOLDERS = set()

def load_expert():
    global m_expert
    if m_expert is None:
        print(f"Loading expert model from {EXPERT_MODEL_PATH}...")
        m_expert = tf.keras.models.load_model(EXPERT_MODEL_PATH)

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
                self.wfile.write(self.get_ui_html().encode())
            elif self.path == "/next":
                data = self.get_next_image_data(); data["stats"] = get_stats()
                self.send_response(200); self.send_header("Content-type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            elif self.path == "/bulk_list":
                data = self.get_bulk_list(); 
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
            print(f"Error GET: {e}")
            try: self.send_error(500)
            except: pass

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length); data = json.loads(post_data.decode())
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
            print(f"Error POST: {e}")
            try: self.send_error(500)
            except: pass

    def get_bulk_list(self):
        if not PRED_ROOT.exists(): return None
        
        # Get all subdirectories that contain at least one .jpg
        valid_dirs = []
        for p in PRED_ROOT.rglob("*"):
            if p.is_dir() and any(p.glob("*.jpg")):
                valid_dirs.append(p)
        
        if not valid_dirs: return None

        # Sort: 1. ZERO folders first, 2. Alphabetical by relative path
        valid_dirs.sort(key=lambda p: (
            0 if "ZERO" in str(p.relative_to(PRED_ROOT)) else 1,
            str(p.relative_to(PRED_ROOT))
        ))
        
        for path in valid_dirs:
            rel_path = str(path.relative_to(PRED_ROOT))
            if rel_path in SKIPPED_FOLDERS: continue
            
            images = list(path.glob("*.jpg"))
            rel_dir = path.relative_to(PRED_ROOT); parts = list(rel_dir.parts)
            while len(parts) < 4: parts.append("NONE")
            img_data = [{"path": str(img.relative_to(PRED_ROOT)), "original": parse_original_label(img.name)} for img in images[:50]]
            return { "label": " ".join(parts[:4]), "label_parts": parts[:4], "images": img_data, "folder": rel_path }
        
        return None

    def confirm_bulk(self, data):
        for item in data["items"]:
            src = PRED_ROOT / item["path"]
            if src.exists():
                dest_dir = DATASET_ROOT / item["count"] / item["color"] / item["pattern"] / item["shape"]
                dest_dir.mkdir(parents=True, exist_ok=True)
                name = src.name
                if "ok_chip_" in name: name = name[name.find("ok_chip_"):]
                shutil.move(src, dest_dir / (f"ok_{name}" if not name.startswith("ok_") else name))
        return True

    def rescue_images(self, data):
        load_expert()
        paths = []
        selected_items = [i for i in data["items"] if i.get("selected")]
        if not selected_items: return True

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
        
        expert_preds = m_expert.predict(tf.stack(tensors), verbose=0)
        
        for i, path in enumerate(valid_paths):
            p_count = MAPS['count'][np.argmax(expert_preds[0][i])]
            p_color = MAPS['color'][np.argmax(expert_preds[1][i])]
            p_pattern = MAPS['pattern'][np.argmax(expert_preds[2][i])]
            p_shape = MAPS['shape'][np.argmax(expert_preds[3][i])]
            label_parts = [p_count, p_color, p_pattern, p_shape]
            
            target_dir = PRED_ROOT.joinpath(*label_parts)
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(path, target_dir / path.name)
        return True

    def get_next_image_data(self):
        for ext in ["*.jpg", "*.jpeg"]:
            for path in DATASET_ROOT.rglob(ext):
                if not (path.name.startswith("ok_") or path.name.startswith("ext_")):
                    rel_path = path.relative_to(DATASET_ROOT); parts = list(rel_path.parts); filename = parts.pop()
                    attrs = parts + ["NONE"] * (4 - len(parts))
                    return {"path": str(rel_path), "count": attrs[0], "color": attrs[1], "pattern": attrs[2], "shape": attrs[3], "filename": filename, "original": parse_original_label(filename)}
        return {"image": None}

    def move_image(self, data):
        src_path = DATASET_ROOT / data["old_path"]
        if not src_path.exists(): src_path = PRED_ROOT / data["old_path"]
        dest_dir = DATASET_ROOT / data["count"] / data["color"] / data["pattern"] / data["shape"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = data['filename']
        if "ok_chip_" in name: name = name[name.find("ok_chip_"):]
        if src_path.exists():
            shutil.move(src_path, dest_dir / (f"ok_{name}" if not name.startswith("ok_") else name))
            return True
        return False

    def get_ui_html(self):
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Cleaner v11.5</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .selected { background-color: #3b82f6 !important; color: white; border-color: #1d4ed8; }
        .bulk-selected { border: 4px solid #3b82f6 !important; }
        .chip-card { border: 4px solid transparent; cursor: pointer; transition: all 0.1s; display: flex; flex-direction: column; height: auto; }
        .label-container { padding: 4px; display: flex; flex-direction: column; gap: 2px; }
        .label-text { font-size: 0.65rem; line-height: 1.2; text-align: center; word-break: break-all; }
        .key-hint { font-size: 0.7rem; color: #666; margin-left: 4px; border: 1px solid #ddd; padding: 1px 3px; border-radius: 3px; background: #eee; }
        .selected .key-hint { color: #eee; border-color: #60a5fa; background: #2563eb; }
        #rescue-loading { display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 100; background: white; padding: 2rem; border-radius: 1rem; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25); border: 2px solid #3b82f6; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div id="rescue-loading" class="text-center">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <div class="text-xl font-black text-blue-600 uppercase tracking-widest">Rescuing Cards...</div>
        <div class="text-sm text-gray-500 mt-2">Expert model is classifying selected chips.</div>
    </div>

    <div class="max-w-[1600px] mx-auto py-8 px-4">
        <div class="flex justify-between items-center mb-6">
            <h1 class="text-3xl font-black text-gray-800 tracking-tighter">Set Labeler <span id="mode-badge" class="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded ml-2">v11.5 Bulk Rescue & Skip</span></h1>
            <div class="flex gap-4 items-center">
                <div id="selection-stats" class="text-xl font-bold text-orange-500">Selected: 0</div>
                <div id="stats" class="text-2xl font-mono font-black text-blue-600">-- / --</div>
            </div>
        </div>

        <div id="bulk-ui" class="bg-white p-6 rounded-xl shadow-lg">
            <div class="mb-6 flex justify-between items-center bg-blue-50 p-4 rounded-lg">
                <div>
                    <div class="text-xs font-bold text-blue-400 uppercase tracking-widest">Currently Reviewing:</div>
                    <h2 id="bulk-label" class="text-2xl font-black text-blue-700 uppercase">...</h2>
                </div>
                <div class="flex gap-2">
                    <button onclick="skipCategory()" class="px-4 py-2 bg-gray-400 text-white rounded font-bold hover:bg-gray-500 transition">Skip <span class="key-hint">z</span></button>
                    <button onclick="rescueSelected()" class="px-4 py-2 bg-blue-600 text-white rounded font-bold hover:bg-blue-700 transition shadow-md">Rescue Cards <span class="key-hint bg-blue-700 border-blue-400 text-white">x</span></button>
                    <button onclick="invertSelection()" class="px-4 py-2 bg-gray-200 text-gray-700 rounded font-bold hover:bg-gray-300 transition">Invert <span class="key-hint">i</span></button>
                    <button onclick="selectAll()" class="px-4 py-2 bg-gray-200 text-gray-700 rounded font-bold hover:bg-gray-300 transition">All <span class="key-hint">a</span></button>
                </div>
            </div>
            <div id="bulk-grid" class="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-4 mb-8"></div>
            <button id="confirm-btn" onclick="submitBulk()" class="w-full py-4 bg-green-600 text-white text-xl font-black rounded-xl uppercase tracking-widest hover:bg-green-700 shadow-lg transition">Confirm Batch <span class="text-sm font-normal opacity-70 ml-2">Space / Enter</span></button>
        </div>
    </div>

    <script>
        let bulkData = null; let bulkItems = []; 
        
        async function loadBulk(){ 
            const res=await fetch('/bulk_list'); bulkData=await res.json(); 
            if(!bulkData){alert("Cleanup Complete!"); return;} 
            document.getElementById('bulk-label').innerText=bulkData.label; 
            if (bulkData.stats) {
                document.getElementById('stats').innerText = `${bulkData.stats.reviewed} / ${bulkData.stats.total}`;
            }
            bulkItems=bulkData.images.map(img => ({ path: img.path, original: img.original, count:bulkData.label_parts[0], color:bulkData.label_parts[1], pattern:bulkData.label_parts[2], shape:bulkData.label_parts[3], selected:false })); 
            renderBulkGrid(); 
        }

        function renderBulkGrid(){ 
            console.log("Rendering grid with", bulkItems.length, "items. bulkData:", bulkData);
            const grid = document.getElementById('bulk-grid'); grid.innerHTML=''; 
            let selectedCount = 0;
            bulkItems.forEach((item,idx)=>{ 
                if (item.selected) selectedCount++;
                const div = document.createElement('div'); div.className=`chip-card bg-gray-50 rounded-lg shadow-sm border-2 ${item.selected?'bulk-selected':'border-transparent'}`; 
                const imgWrap = document.createElement('div'); imgWrap.className = "bg-gray-200 rounded-t-lg overflow-hidden aspect-square w-full";
                const img = document.createElement('img'); img.src='/images/'+encodeURIComponent(item.path); img.className="w-full h-full object-contain"; 
                imgWrap.appendChild(img); div.appendChild(imgWrap);
                
                const labelCont = document.createElement('div'); labelCont.className = "label-container";
                const currentLabel = document.createElement('div'); 
                currentLabel.className = "label-text font-black uppercase flex flex-wrap justify-center gap-x-1";
                
                ['count', 'color', 'pattern', 'shape'].forEach((attr, i) => {
                    const span = document.createElement('span');
                    const val = item[attr];
                    const isDiff = val !== bulkData.label_parts[i];
                    span.innerText = val;
                    span.className = isDiff ? 'text-blue-600' : 'text-gray-400';
                    currentLabel.appendChild(span);
                });
                
                labelCont.appendChild(currentLabel);
                const originalLabel = document.createElement('div');
                originalLabel.className = "label-text text-orange-600 font-bold border-t border-gray-100 mt-1 pt-1";
                originalLabel.innerText = (item.original || "???");
                labelCont.appendChild(originalLabel);
                div.appendChild(labelCont);
                div.onclick=()=>{
                    console.log("Item clicked:", idx, item.path);
                    item.selected=!item.selected;
                    renderBulkGrid();
                }; 
                grid.appendChild(div); 
            }); 
            document.getElementById('selection-stats').innerText = `Selected: ${selectedCount}`;
        }

        function selectAll(){ bulkItems.forEach(i=>i.selected=true); renderBulkGrid(); }
        function clearSelection(){ bulkItems.forEach(i=>i.selected=false); renderBulkGrid(); }
        function invertSelection(){ bulkItems.forEach(i=>i.selected=!i.selected); renderBulkGrid(); }
        function applyOverride(t,v){ 
            console.log("Applying override:", t, v);
            let count = 0;
            bulkItems.forEach(i=>{
                if(i.selected){ 
                    if(t==='all_none'){
                        i.count='ZERO';i.color='NONE';i.pattern='NONE';i.shape='NONE';
                    } else {
                        i[t]=v;
                    } 
                    count++;
                }
            }); 
            console.log("Updated", count, "items");
            renderBulkGrid(); 
        }
        
        async function rescueSelected() {
            const hasSelection = bulkItems.some(i => i.selected);
            if (!hasSelection) return;
            
            document.getElementById('rescue-loading').style.display = 'block';
            await fetch('/rescue', {method:'POST', body:JSON.stringify({items:bulkItems})});
            document.getElementById('rescue-loading').style.display = 'none';
            loadBulk();
        }

        async function skipCategory() {
            if (!bulkData || !bulkData.folder) return;
            await fetch('/skip', {method:'POST', body:JSON.stringify({folder:bulkData.folder})});
            loadBulk();
        }

        async function submitBulk(){ await fetch('/confirm_bulk',{method:'POST',body:JSON.stringify({items:bulkItems})}); loadBulk(); }

        window.addEventListener('keydown', (e) => { 
            const key = e.key.toLowerCase(); 
            if (key === 'a') { e.preventDefault(); selectAll(); return; } 
            if (key === 'c') { e.preventDefault(); clearSelection(); return; } 
            if (key === 'i') { e.preventDefault(); invertSelection(); return; } 
            if (key === 'x') { e.preventDefault(); rescueSelected(); return; } 
            if (key === 'z') { e.preventDefault(); skipCategory(); return; } 
            if (key === '0' || key === 'n') { e.preventDefault(); applyOverride('all_none'); return; } 
            
            if (key === '1') { e.preventDefault(); applyOverride('count', 'ONE'); return; }
            if (key === '2') { e.preventDefault(); applyOverride('count', 'TWO'); return; }
            if (key === '3') { e.preventDefault(); applyOverride('count', 'THREE'); return; }
            
            if (key === 'r') { e.preventDefault(); applyOverride('color', 'RED'); return; }
            if (key === 'g') { e.preventDefault(); applyOverride('color', 'GREEN'); return; }
            if (key === 'p') { e.preventDefault(); applyOverride('color', 'PURPLE'); return; }
            
            if (key === 's') { e.preventDefault(); applyOverride('pattern', 'SOLID'); return; }
            if (key === 'h') { e.preventDefault(); applyOverride('pattern', 'SHADED'); return; }
            if (key === 'e') { e.preventDefault(); applyOverride('pattern', 'EMPTY'); return; }
            
            if (key === 'o') { e.preventDefault(); applyOverride('shape', 'OVAL'); return; }
            if (key === 'd') { e.preventDefault(); applyOverride('shape', 'DIAMOND'); return; }
            if (key === 'q') { e.preventDefault(); applyOverride('shape', 'SQUIGGLE'); return; }
            
            if (key === 'enter' || key === ' ') { e.preventDefault(); submitBulk(); return; } 
        });
        loadBulk();
    </script>
</body>
</html>
        """

if __name__ == "__main__":
    server_address = ('', PORT)
    httpd = http.server.HTTPServer(server_address, LabelerHandler)
    print(f"Starting Dataset Cleaner v11.5 at http://localhost:{PORT}")
    httpd.serve_forever()
