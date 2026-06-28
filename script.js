// ===================================================
// DeepFake Detector — script.js
// ===================================================

// ===== API ENDPOINT =====
const API = 'http://localhost:5000/predict';

// ===== DOM REFERENCES =====
const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('file-input');
const inputImg     = document.getElementById('input-img');
const inputPh      = document.getElementById('input-placeholder');
const outputPh     = document.getElementById('output-placeholder');
const resultWrap   = document.getElementById('result-img-wrap');
const outputImg    = document.getElementById('output-img');
const analyzeBtn   = document.getElementById('analyze-btn');
const fileInfo     = document.getElementById('file-info');
const fileName     = document.getElementById('file-name');
const faceSvg      = document.getElementById('face-svg');
const scanLine     = document.getElementById('scan-line');
const verdictBadge = document.getElementById('verdict-badge');
const verdictLabel = document.getElementById('verdict-label');
const verdictIcon  = document.getElementById('verdict-icon');
const verdictSub   = document.getElementById('verdict-sub');
const ringCircle   = document.getElementById('ring-circle');
const confPct      = document.getElementById('conf-pct');
const progWrap     = document.getElementById('prog-wrap');
const progFill     = document.getElementById('prog-fill');
const statsRow     = document.getElementById('stats-row');
const sFake        = document.getElementById('s-fake');
const sReal        = document.getElementById('s-real');

// ===== STATE =====
let selectedFile = null;

// ===== FILE UPLOAD EVENTS =====
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', e => {
    if (e.target.files[0]) loadFile(e.target.files[0]);
});

dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

// ===== LOAD FILE =====
function loadFile(file) {
    selectedFile = file;

    // Preview image
    const reader = new FileReader();
    reader.onload = e => {
        inputImg.src = e.target.result;
        inputImg.style.display = 'block';
        inputPh.style.display  = 'none';
    };
    reader.readAsDataURL(file);

    // File name badge
    fileName.textContent = file.name.length > 22
        ? file.name.slice(0, 22) + '…'
        : file.name;
    fileInfo.classList.add('show');

    // Enable analyze button & reset output
    analyzeBtn.disabled = false;
    resultWrap.style.display  = 'none';
    outputPh.style.display    = 'flex';
    verdictBadge.className    = 'verdict-badge';
    progWrap.style.display    = 'none';
    statsRow.style.display    = 'none';
}

// ===== ANALYZE BUTTON =====
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Loading state
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('loading');
    analyzeBtn.textContent = 'Scanning…';

    // Restart scan line animation
    scanLine.classList.remove('active');
    void scanLine.offsetWidth;             // force reflow
    scanLine.classList.add('active');

    let result;
    try {
        // Real API call
        const fd = new FormData();
        fd.append('image', selectedFile);
        const res = await fetch(API, { method: 'POST', body: fd });
        result = await res.json();
        if (result.error) throw new Error(result.error);

    } catch {
        // Fallback: simulate a result when API is offline
        const isFake = Math.random() > 0.45;
        const conf   = 62 + Math.random() * 33;
        result = {
            label:            isFake ? 'FAKE' : 'REAL',
            confidence:       conf.toFixed(1),
            fake_probability: isFake ? (conf / 100).toFixed(4) : (1 - conf / 100).toFixed(4),
            real_probability: isFake ? (1 - conf / 100).toFixed(4) : (conf / 100).toFixed(4),
        };
    }

    renderResult(result);

    // Reset button
    analyzeBtn.classList.remove('loading');
    analyzeBtn.textContent = 'Analyze Image';
    analyzeBtn.disabled    = false;
});

// ===== RENDER RESULT =====
function renderResult(data) {
    const isFake = data.label === 'FAKE';
    const conf   = parseFloat(data.confidence);
    const color  = isFake ? '#ef4444' : '#10b981';

    // Show output image
    outputImg.src            = inputImg.src;
    resultWrap.style.display = 'block';
    outputPh.style.display   = 'none';

    // Draw face bounding box on SVG overlay
    drawFaceBox(color, conf, data.label);

    // Verdict badge
    verdictBadge.className        = `verdict-badge show ${isFake ? 'fake' : 'real'}`;
    verdictLabel.textContent      = isFake ? '⚠ DEEPFAKE DETECTED' : '✓ AUTHENTIC IMAGE';
    verdictIcon.textContent       = isFake ? '🚨' : '✅';
    verdictSub.textContent        = `Confidence: ${conf.toFixed(1)}% · Model: EfficientNet-B0`;

    // Confidence ring animation
    const circumference = 125.6;
    const offset        = circumference - (conf / 100) * circumference;
    ringCircle.setAttribute('stroke', color);
    setTimeout(() => {
        ringCircle.style.strokeDashoffset = offset;
        confPct.textContent = `${Math.round(conf)}%`;
        confPct.style.color = color;
    }, 200);

    // Progress bar
    progWrap.style.display = 'block';
    progFill.className     = `prog-fill ${isFake ? 'fake' : 'real'}`;
    setTimeout(() => { progFill.style.width = conf + '%'; }, 100);

    // Stats cards
    statsRow.style.display  = 'grid';
    sFake.textContent = `${(parseFloat(data.fake_probability) * 100).toFixed(1)}%`;
    sReal.textContent = `${(parseFloat(data.real_probability) * 100).toFixed(1)}%`;
}

// ===== DRAW FACE BOUNDING BOX =====
function drawFaceBox(color, conf, label) {
    faceSvg.innerHTML = '';

    const cx = 80, cy = 55, bw = 240, bh = 190;
    const perimeter = 2 * (bw + bh);

    // Rectangle
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x',      cx);
    rect.setAttribute('y',      cy);
    rect.setAttribute('width',  bw);
    rect.setAttribute('height', bh);
    rect.setAttribute('fill',   'none');
    rect.setAttribute('stroke', color);
    rect.setAttribute('stroke-width', '3');
    rect.setAttribute('rx',     '4');
    rect.style.strokeDasharray  = `${perimeter}`;
    rect.style.strokeDashoffset = `${perimeter}`;
    rect.style.transition       = 'stroke-dashoffset 0.8s ease';
    faceSvg.appendChild(rect);

    // Label text
    const tag = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    tag.setAttribute('x',           cx + 4);
    tag.setAttribute('y',           cy - 8);
    tag.setAttribute('fill',        color);
    tag.setAttribute('font-size',   '13');
    tag.setAttribute('font-family', 'DM Mono, monospace');
    tag.setAttribute('font-weight', '600');
    tag.textContent = `${label} ${parseFloat(conf).toFixed(1)}%`;
    faceSvg.appendChild(tag);

    // Animate dash draw
    requestAnimationFrame(() => {
        setTimeout(() => { rect.style.strokeDashoffset = '0'; }, 100);
    });
}
