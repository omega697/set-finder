let bulkData = null; 
let bulkItems = []; 

async function loadBulk(){ 
    const res = await fetch('/bulk_list'); 
    bulkData = await res.json(); 
    if(!bulkData){
        alert("Cleanup Complete!"); 
        return;
    } 
    document.getElementById('bulk-label').innerText = bulkData.label; 
    if (bulkData.stats) {
        document.getElementById('stats').innerText = `${bulkData.stats.reviewed} / ${bulkData.stats.total}`;
    }
    bulkItems = bulkData.images.map(img => ({ 
        path: img.path, 
        original: img.original, 
        count: bulkData.label_parts[0], 
        color: bulkData.label_parts[1], 
        pattern: bulkData.label_parts[2], 
        shape: bulkData.label_parts[3], 
        selected: false 
    })); 
    renderBulkGrid(); 
}

function renderBulkGrid(){ 
    const grid = document.getElementById('bulk-grid'); 
    grid.innerHTML = ''; 
    bulkItems.forEach((item, idx) => { 
        const div = document.createElement('div'); 
        div.className = `chip-card bg-gray-50 rounded-lg shadow-sm border-2 ${item.selected ? 'bulk-selected' : 'border-transparent'}`; 
        
        const imgWrap = document.createElement('div'); 
        imgWrap.className = "bg-gray-200 rounded-t-lg overflow-hidden aspect-square w-full";
        
        const img = document.createElement('img'); 
        img.src = '/images/' + encodeURIComponent(item.path); 
        img.className = "w-full h-full object-contain"; 
        
        imgWrap.appendChild(img); 
        div.appendChild(imgWrap);
        
        const labelCont = document.createElement('div'); 
        labelCont.className = "label-container";
        
        const currentLabel = document.createElement('div'); 
        currentLabel.className = "label-text font-black uppercase flex flex-wrap justify-center gap-x-1";
        
        ['count', 'color', 'pattern', 'shape'].forEach((attr, i) => {
            const span = document.createElement('span'); 
            span.innerText = item[attr];
            const isDiff = item[attr] !== bulkData.label_parts[i];
            span.className = isDiff ? 'text-blue-600' : 'text-gray-400';
            currentLabel.appendChild(span);
        });
        
        labelCont.appendChild(currentLabel);
        
        const orig = document.createElement('div'); 
        orig.className = "label-text text-orange-600 font-bold border-t border-gray-100 mt-1 pt-1"; 
        orig.innerText = item.original || "???";
        
        labelCont.appendChild(orig); 
        div.appendChild(labelCont);
        
        div.onclick = () => {
            item.selected = !item.selected; 
            renderBulkGrid();
        }; 
        grid.appendChild(div); 
    }); 
}

function selectAll(){ 
    bulkItems.forEach(i => i.selected = true); 
    renderBulkGrid(); 
}

function invertSelection(){ 
    bulkItems.forEach(i => i.selected = !i.selected); 
    renderBulkGrid(); 
}

function applyOverride(t, v){ 
    bulkItems.forEach(i => { 
        if(i.selected){ 
            if(t === 'all_none'){ 
                i.count = 'ZERO'; i.color = 'NONE'; i.pattern = 'NONE'; i.shape = 'NONE'; 
            } else { 
                i[t] = v; 
            } 
        } 
    }); 
    renderBulkGrid(); 
}

async function rescueSelected() {
    if (!bulkItems.some(i => i.selected)) return;
    document.getElementById('rescue-loading').style.display = 'block';
    await fetch('/rescue', {method: 'POST', body: JSON.stringify({items: bulkItems})});
    document.getElementById('rescue-loading').style.display = 'none';
    loadBulk();
}

async function skipCategory() { 
    if (bulkData?.folder) { 
        await fetch('/skip', {method: 'POST', body: JSON.stringify({folder: bulkData.folder})}); 
        loadBulk(); 
    } 
}

async function submitBulk(){ 
    await fetch('/confirm_bulk', {method: 'POST', body: JSON.stringify({items: bulkItems})}); 
    loadBulk(); 
}

window.addEventListener('keydown', (e) => { 
    const key = e.key.toLowerCase(); 
    if (key === 'a') { e.preventDefault(); selectAll(); return; } 
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

// Initial load
loadBulk();
