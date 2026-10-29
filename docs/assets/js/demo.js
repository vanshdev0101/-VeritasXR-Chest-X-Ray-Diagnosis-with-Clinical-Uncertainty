/* VeritasXR in-browser demo.
 *
 * Preprocessing here must match the Python eval transform exactly:
 *   greyscale -> resize 224x224 bilinear -> /255 -> (x - 0.5) / 0.5
 * A silent mismatch produces confident nonsense, so scripts/check_web_parity.py
 * asserts Python and this pipeline agree on the same image before release.
 */
const MODEL_URL = 'model/veritasxr.onnx';
const SIZE = 224;
const CLASSES = ['NORMAL', 'PNEUMONIA'];

const GLYPH = {
  high: '<path d="M20 6 9 17l-5-5"/>',
  moderate: '<circle cx="12" cy="12" r="9"/><path d="M12 8v4m0 4h.01"/>',
  low: '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4m0 4h.01"/>',
};
// Thresholds are the 60th and 85th percentile of predictive entropy on the
// VALIDATION split, never the test set. Arbitrary round numbers put every case
// in one band: the original 0.4/0.7 cutoffs left 100% of the test set in "high
// confidence", which made the triage decorative.
const BANDS = [
  [0.829, 'high', 'Higher confidence',
   'Among the most confident ~60% of cases. On validation data this band is right about 90% of the time.'],
  [0.981, 'moderate', 'Moderate confidence',
   'The prediction is closer to the decision boundary. Worth a second look.'],
  [Infinity, 'low', 'Low confidence',
   'The model is near-undecided. Treat this prediction as unreliable.'],
];

const $ = (id) => document.getElementById(id);
let session = null;

/* ---------- theme ---------- */
(function theme() {
  const saved = (() => { try { return localStorage.getItem('vxr-theme'); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  $('theme').addEventListener('click', () => {
    const dark = matchMedia('(prefers-color-scheme: dark)').matches;
    const cur = document.documentElement.dataset.theme || (dark ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('vxr-theme', next); } catch { /* private mode */ }
  });
})();

/* ---------- model ---------- */
async function loadModel() {
  const badge = $('model-badge');
  try {
    const head = await fetch(MODEL_URL, { method: 'HEAD' });
    if (!head.ok) throw new Error(`HTTP ${head.status}`);
    badge.textContent = 'loading model…';
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all',
    });
    const mb = (Number(head.headers.get('content-length') || 0) / 1e6).toFixed(1);
    badge.className = 'badge';
    badge.textContent = `model ready · ${mb} MB · runs locally`;
  } catch (err) {
    badge.className = 'badge badge-pending';
    badge.textContent = 'model export pending — UI is live, inference is not';
    $('model-state').insertAdjacentHTML('beforeend',
      ' <span class="hint">The model file is not published yet. Upload and ' +
      'preprocessing work; prediction activates once it lands.</span>');
    console.info('[VeritasXR] model unavailable:', err.message);
  }
}

/* ---------- preprocessing ---------- */
function toTensor(img) {
  const c = $('preview'), ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.drawImage(img, 0, 0, SIZE, SIZE);           // matches Resize((224,224))
  const px = ctx.getImageData(0, 0, SIZE, SIZE);
  const d = px.data, out = new Float32Array(SIZE * SIZE);
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    // ITU-R 601-2 luma, the same conversion PIL's "L" mode applies.
    const g = (d[i] * 299 + d[i + 1] * 587 + d[i + 2] * 114) / 1000;
    out[p] = (g / 255 - 0.5) / 0.5;
    d[i] = d[i + 1] = d[i + 2] = g;               // show what the model sees
  }
  ctx.putImageData(px, 0, 0);
  return new ort.Tensor('float32', out, [1, 1, SIZE, SIZE]);
}

/* ---------- CAM rendering ---------- */
function drawCam(canvasId, cam, h, w) {
  const c = $(canvasId), ctx = c.getContext('2d');
  let lo = Infinity, hi = -Infinity;
  for (const v of cam) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const span = hi - lo || 1;

  ctx.drawImage($('preview'), 0, 0, SIZE, SIZE);
  const layer = ctx.getImageData(0, 0, SIZE, SIZE), ld = layer.data;
  for (let y = 0; y < SIZE; y++) {
    const sy = Math.min(h - 1, (y * h / SIZE) | 0);
    for (let x = 0; x < SIZE; x++) {
      const sx = Math.min(w - 1, (x * w / SIZE) | 0);
      const t = Math.max(0, Math.min(1, (cam[sy * w + sx] - lo) / span));
      const i = (y * SIZE + x) * 4;
      // Warm overlay weighted by activation; the base image stays visible so
      // the heatmap is not the only thing carrying information.
      ld[i]     = ld[i] * (1 - t * 0.75) + 255 * t * 0.75;
      ld[i + 1] = ld[i + 1] * (1 - t * 0.75) + 140 * t * 0.75;
      ld[i + 2] = ld[i + 2] * (1 - t * 0.75) + 20 * t * 0.75;
    }
  }
  ctx.putImageData(layer, 0, 0);
}

/* ---------- run ---------- */
function softmax(a) {
  const m = Math.max(...a), e = a.map((v) => Math.exp(v - m)), s = e.reduce((p, c) => p + c, 0);
  return e.map((v) => v / s);
}

async function predict(img) {
  const tensor = toTensor(img);
  if (!session) return;

  const t0 = performance.now();
  const res = await session.run({ input: tensor });
  const ms = performance.now() - t0;

  const names = Object.keys(res);
  const get = (want) => res[names.find((n) => n.toLowerCase().includes(want))];
  const verdict = get('verdict') || res[names[0]];
  const p = softmax(Array.from(verdict.data));
  const idx = p[1] >= p[0] ? 1 : 0;

  $('verdict').textContent = CLASSES[idx];
  $('probs').textContent = `NORMAL ${(p[0] * 100).toFixed(1)}% · PNEUMONIA ${(p[1] * 100).toFixed(1)}%`;
  $('latency').textContent = `${ms.toFixed(0)} ms`;

  const disT = get('disagree');
  if (disT) {
    const d = Number(disT.data[0]);
    $('disagree').textContent = d.toFixed(4);
  }

  const uncT = get('uncertain');
  if (uncT) {
    const u = Number(uncT.data[0]);
    $('unc').textContent = u.toFixed(3);
    $('unc-bar').style.width = `${Math.round(u * 100)}%`;
    const [, band, label, note] = BANDS.find(([t]) => u < t);
    const tri = $('triage');
    tri.dataset.band = band;
    $('triage-glyph').innerHTML = GLYPH[band];
    $('triage-label').textContent = label;
    $('triage-note').textContent = note;
  }
  $('out').hidden = false;

  const cl = get('cam_local'), cg = get('cam_global');
  if (cl && cg) {
    const [, , h, w] = cl.dims.length === 4 ? cl.dims : [1, 1, ...cl.dims.slice(-2)];
    drawCam('cam-local', cl.data, h, w);
    drawCam('cam-global', cg.data, h, w);
    $('cams').hidden = false;
  }
}

/* ---------- input wiring ---------- */
function handleFile(file) {
  if (!file || !/^image\/(png|jpeg)$/.test(file.type)) {
    alert('Please choose a PNG or JPEG image.');
    return;
  }
  const img = new Image();
  img.onload = () => { predict(img); URL.revokeObjectURL(img.src); };
  img.onerror = () => alert('That file could not be read as an image.');
  img.src = URL.createObjectURL(file);
}

$('browse').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => handleFile(e.target.files[0]));

const drop = $('drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.add('is-over');
}));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.remove('is-over');
}));
drop.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));

loadModel();
