const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let img = new Image();
let quads = [];
let manualPoints = [];
let currentFrameIdx = 0;
let frames = [];
let manifest = {};
let isCleaned = false;

// Dragging state
let draggingCorner = null; // { quadIdx, ptIdx }
let hoverCorner = null;
let lastInspectQuad = null;
let inspectThrottle = null;

// --- INITIALIZATION ---
async function init() {
    const resp = await fetch('/api/init');
    const data = await resp.json();
    frames = data.frames;
    manifest = data.manifest;
    
    if (frames.length > 0) {
        await loadFrame(0);
    } else {
        document.getElementById('filename').innerText = "No frames found in ml/dataset/yolo_pose";
    }
}

// --- LOADING ---
async function loadFrame(index) {
    if (index < 0 || index >= frames.length) return;
    
    currentFrameIdx = index;
    const filename = frames[index];
    document.getElementById('filename').innerText = `${index + 1}/${frames.length}: ${filename}`;
    
    // Load data first
    const dataResp = await fetch(`/api/frame_data/${filename}`);
    const data = await dataResp.json();
    quads = data.quads;
    isCleaned = data.is_cleaned;
    manualPoints = [];
    draggingCorner = null;
    lastInspectQuad = null;
    hideInspect();
    
    updateUI();
    
    // Load image
    img.src = `/api/frame_image/${filename}?t=${Date.now()}`;
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        render();
    };
}

function updateUI() {
    const badge = document.getElementById('clean-status');
    badge.innerText = isCleaned ? "CLEAN" : "DIRTY";
    badge.className = `clean-badge ${isCleaned ? 'is-clean' : 'is-dirty'}`;
    
    const cleanedCount = Object.values(manifest).filter(v => v).length;
    const total = frames.length;
    const pct = total > 0 ? Math.round((cleanedCount / total) * 100) : 0;
    document.getElementById('progress').innerText = `Cleaned: ${cleanedCount}/${total} (${pct}%)`;
}

// --- INTERACTION ---

function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / (rect.right - rect.left) * canvas.width;
    const y = (e.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height;
    return {x, y};
}

function findNearCorner(x, y) {
    const threshold = 15 / (canvas.clientWidth / canvas.width);
    for (let qi = 0; qi < quads.length; qi++) {
        for (let pi = 0; pi < 4; pi++) {
            const dx = quads[qi][pi][0] - x;
            const dy = quads[qi][pi][1] - y;
            if (Math.sqrt(dx*dx + dy*dy) < threshold) {
                return { quadIdx: qi, ptIdx: pi };
            }
        }
    }
    return null;
}

canvas.onmousedown = async (e) => {
    const {x, y} = getMousePos(e);
    
    if (e.button === 0) { // Left Click
        // 1. Check for corner drag
        const corner = findNearCorner(x, y);
        if (corner) {
            draggingCorner = corner;
            return;
        }

        // 2. Hit test for deletion
        const hitIdx = quads.findIndex(q => isPointInQuad(x, y, q));
        if (hitIdx !== -1) {
            quads.splice(hitIdx, 1);
            await saveCurrentFrame(false);
            lastInspectQuad = null;
            hideInspect();
            render();
            return;
        }
        
        // 3. SAM Add
        setHint("SAM is thinking...");
        const resp = await fetch('/api/sam_click', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ x: Math.round(x), y: Math.round(y) })
        });
        const data = await resp.json();
        if (data.quad) {
            quads.push(data.quad);
            await saveCurrentFrame(false);
            render();
            setHint("Quad added!");
        } else {
            setHint("No card found at click location.");
        }
    } else if (e.button === 2) { // Right Click
        e.preventDefault();
        manualPoints.push([x, y]);
        
        if (manualPoints.length === 4) {
            setHint("Rectifying manual quad...");
            const resp = await fetch('/api/rectify', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ points: manualPoints })
            });
            const data = await resp.json();
            quads.push(data.quad);
            manualPoints = [];
            await saveCurrentFrame(false);
            render();
            setHint("Manual quad added!");
        } else {
            setHint(`Placed corner ${manualPoints.length}/4...`);
            render();
        }
    }
};

window.onmousemove = (e) => {
    const {x, y} = getMousePos(e);
    
    const box = document.getElementById('inspect-box');
    box.style.left = (e.clientX + 20) + 'px';
    box.style.top = (e.clientY + 20) + 'px';
    // Boundary check
    if (e.clientX + 200 > window.innerWidth) box.style.left = (e.clientX - 180) + 'px';
    if (e.clientY + 300 > window.innerHeight) box.style.top = (e.clientY - 280) + 'px';

    if (draggingCorner) {
        quads[draggingCorner.quadIdx][draggingCorner.ptIdx] = [x, y];
        render();
        hideInspect();
    } else {
        const prevHover = hoverCorner;
        hoverCorner = findNearCorner(x, y);
        if (!!prevHover !== !!hoverCorner) render();

        // INSPECT LOGIC
        const hitQuad = quads.find(q => isPointInQuad(x, y, q));
        if (hitQuad) {
            const quadStr = JSON.stringify(hitQuad);
            if (quadStr !== lastInspectQuad) {
                lastInspectQuad = quadStr;
                triggerInspect(hitQuad);
            }
            box.style.display = 'block';
        } else {
            if (lastInspectQuad !== null) {
                lastInspectQuad = null;
                hideInspect();
            }
        }
    }
};

async function triggerInspect(quad) {
    document.getElementById('inspect-loading').style.display = 'block';
    if (inspectThrottle) clearTimeout(inspectThrottle);
    inspectThrottle = setTimeout(async () => {
        const resp = await fetch('/api/inspect_quad', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ quad: quad })
        });
        if (!resp.ok) {
            document.getElementById('inspect-loading').innerText = "Error";
            return;
        }
        const data = await resp.json();
        if (data.chip && JSON.stringify(quad) === lastInspectQuad) {
            showInspect(data.chip, data.prob);
        }
    }, 50);
}

function showInspect(chip, prob) {
    document.getElementById('inspect-loading').style.display = 'none';
    const img = document.getElementById('inspect-img');
    const probDiv = document.getElementById('inspect-prob');
    img.src = chip;
    probDiv.innerText = `Prob: ${prob.toFixed(4)}`;
    probDiv.style.color = prob > 0.4 ? '#28a745' : '#dc3545';
}

function hideInspect() {
    document.getElementById('inspect-box').style.display = 'none';
}

window.onmouseup = async (e) => {
    if (draggingCorner) {
        draggingCorner = null;
        await saveCurrentFrame(false);
        render();
    }
};

canvas.oncontextmenu = (e) => e.preventDefault();

function isPointInQuad(x, y, quad) {
    let inside = false;
    for (let i = 0, j = quad.length - 1; i < quad.length; j = i++) {
        const xi = quad[i][0], yi = quad[i][1];
        const xj = quad[j][0], yj = quad[j][1];
        const intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// --- PERSISTENCE ---

async function saveCurrentFrame(markClean = false) {
    const filename = frames[currentFrameIdx];
    await fetch('/api/save_frame', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            filename: filename,
            quads: quads,
            mark_clean: markClean
        })
    });
    if (markClean) {
        manifest[filename] = true;
        isCleaned = true;
        updateUI();
    }
}

async function markCleanAndNext() {
    await saveCurrentFrame(true);
    changeFrame(1);
}

function changeFrame(delta) {
    const next = Math.max(0, Math.min(frames.length - 1, currentFrameIdx + delta));
    loadFrame(next);
}

function goToNextUncleaned() {
    for (let i = currentFrameIdx + 1; i < frames.length; i++) {
        if (!manifest[frames[i]]) {
            loadFrame(i);
            return;
        }
    }
    alert("No more uncleaned frames found!");
}

async function exportCleaned() {
    setHint("Exporting cleaned dataset...");
    const resp = await fetch('/api/export', {method: 'POST'});
    const data = await resp.json();
    alert(`Export complete! ${data.exported} frames copied to ml/dataset/yolo_pose_cleaned`);
    setHint("Exported.");
}

// --- RENDERING ---

function render() {
    ctx.drawImage(img, 0, 0);
    
    const scaleFactor = canvas.clientWidth / canvas.width;

    // Draw Quads
    quads.forEach((quad, qi) => {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = Math.max(2, 4 / scaleFactor);
        ctx.beginPath();
        ctx.moveTo(quad[0][0], quad[0][1]);
        for (let i = 1; i < 4; i++) ctx.lineTo(quad[i][0], quad[i][1]);
        ctx.closePath();
        ctx.stroke();
        
        ctx.fillStyle = 'rgba(0, 255, 0, 0.15)';
        ctx.fill();

        // Draw corners as drag handles
        quad.forEach((pt, pi) => {
            const isHovered = hoverCorner && hoverCorner.quadIdx === qi && hoverCorner.ptIdx === pi;
            const isDragging = draggingCorner && draggingCorner.quadIdx === qi && draggingCorner.ptIdx === pi;
            
            ctx.fillStyle = isDragging ? '#ff00ff' : (isHovered ? '#00ffff' : '#00ff00');
            const r = (isHovered || isDragging ? 10 : 5) / scaleFactor;
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2);
            ctx.fill();
        });
    });
    
    // Draw Manual points
    const ptSize = Math.max(5, 8 / scaleFactor);
    manualPoints.forEach((pt, i) => {
        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], ptSize, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.font = `${ptSize * 2}px Arial`;
        ctx.fillText(i + 1, pt[0] + ptSize, pt[1] - ptSize);
    });
}

function setHint(text) {
    document.getElementById('hint').innerText = text;
}

// --- KEYBOARD ---
window.onkeydown = (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key.toLowerCase() === 'd') changeFrame(1);
    if (e.key.toLowerCase() === 'a') changeFrame(-1);
    if (e.key.toLowerCase() === 'w') markCleanAndNext();
};

init();
