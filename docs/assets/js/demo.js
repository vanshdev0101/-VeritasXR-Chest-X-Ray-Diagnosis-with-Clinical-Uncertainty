/* VeritasXR in-browser demo.
 *
 * Preprocessing here must match the Python eval transform exactly:
 *   greyscale -> resize 224x224 bilinear -> /255 -> (x - 0.5) / 0.5
 * A silent mismatch produces confident nonsense, so scripts/check_web_parity.py
 * asserts Python and this pipeline agree on the same image before release.
 */
// Versioned so a cached model can never be paired with newer code. The two
// must agree on output names; a stale 16.9 MB model behind fresh JavaScript
// fails silently rather than loudly, which is the worst way for it to fail.
const MODEL_VERSION = 'b7d6213f';
const MODEL_URL = `model/veritasxr.onnx?v=${MODEL_VERSION}`;
const SIZE = 224;
const GATE_URL = 'data/input_gate.json';
let gate = null;
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
//
// They belong to the seed-0 checkpoint this page serves and to no other. The
// same percentiles on seeds 1 and 2 are 0.7473/0.9619 and 0.65/0.9471, so these
// cuts are a property of one trained model, not of the architecture.
// Recomputed by scripts/eval_uncertainty.py into docs/data/uncertainty.json.
const BANDS = [
  [0.8271, 'high', 'Higher confidence',
   'Among the most confident ~60% of cases. On validation data this band is right about 90% of the time.'],
  [0.9789, 'moderate', 'Moderate confidence',
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
    gate = await (await fetch(GATE_URL)).json();
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

/* ---------- input gate ---------- */
// The classifier has no way to say "not my kind of image". Run a photograph
// through it and it answers NORMAL with high confidence, because entropy
// measures distance from the decision boundary and a photograph is nowhere
// near it. So the refusal lives here, ahead of inference, on the pixels.
//
// Three checks, each cut at the 2nd percentile of the validation split:
// contrast (a blank field z-scores to the same vector as a faint one),
// squared distance from the mean chest layout, and correlation with it.
// Fitted and measured by scripts/fit_input_gate.py.
function gateStats(t) {
  const g = gate.grid, b = SIZE / g, cells = new Float32Array(g * g);
  for (let y = 0; y < SIZE; y++) {
    const row = (y / b) | 0;
    for (let x = 0; x < SIZE; x++) cells[row * g + ((x / b) | 0)] += t[y * SIZE + x];
  }
  for (let i = 0; i < cells.length; i++) cells[i] /= b * b;

  let m = 0; for (const v of cells) m += v; m /= cells.length;
  let s2 = 0; for (const v of cells) s2 += (v - m) ** 2;
  const sd = Math.sqrt(s2 / cells.length) || 1e-6;
  const z = Array.from(cells, (v) => (v - m) / sd);

  // The tensor is (grey/255 - 0.5) / 0.5, so a standard deviation in 0-255
  // units is the tensor's own standard deviation times 127.5.
  let tm = 0; for (let i = 0; i < t.length; i++) tm += t[i]; tm /= t.length;
  let tv = 0; for (let i = 0; i < t.length; i++) tv += (t[i] - tm) ** 2;
  const contrast = Math.sqrt(tv / t.length) * 127.5;

  let dist = 0, dot = 0, nz = 0, nm = 0;
  for (let i = 0; i < z.length; i++) {
    dist += ((z[i] - gate.mean[i]) / gate.std[i]) ** 2;
    dot += z[i] * gate.mean[i];
    nz += z[i] * z[i];
    nm += gate.mean[i] * gate.mean[i];
  }
  return { contrast, distance: dist / z.length,
           correlation: dot / (Math.sqrt(nz) * Math.sqrt(nm) + 1e-9) };
}

// Colour has no counterpart in the training split -- every radiograph there is
// greyscale -- so this cut is fitted on nothing and is deliberately generous.
// It exists because a colour photograph is the input a visitor actually tries.
function colourCast(img) {
  const c = document.createElement('canvas');
  c.width = c.height = 32;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, 32, 32);
  const d = ctx.getImageData(0, 0, 32, 32).data;
  let worst = 0;
  for (let i = 0; i < d.length; i += 4) {
    worst = Math.max(worst, Math.max(d[i], d[i + 1], d[i + 2])
                          - Math.min(d[i], d[i + 1], d[i + 2]));
  }
  return worst;
}

function refuse(reason) {
  $('refused-why').textContent = reason;
  $('refused').hidden = false;
  $('out').hidden = true;
  $('cams').hidden = true;
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
// The exported graph already applies softmax, so `verdict` carries
// probabilities. Applying softmax again here compressed every displayed
// confidence toward 50% -- a case scored at 0.9999 NORMAL was shown as 0.73 --
// while leaving the label correct, because softmax is monotone. The label
// being right is what let it go unnoticed.
function readProbs(t) {
  const p = Array.from(t.data, Number);
  const sum = p.reduce((a, b) => a + b, 0);
  if (!(Math.abs(sum - 1) < 1e-3) || p.some((v) => v < 0 || v > 1)) {
    console.warn('[VeritasXR] verdict output is not a probability vector', p);
    const m = Math.max(...p), e = p.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }
  return p;
}

async function predict(img) {
  const tensor = toTensor(img);
  if (!session) return;

  // Refuse before inference, never after: a verdict that is shown and then
  // retracted has already been read.
  if (gate) {
    const cast = colourCast(img);
    const g = gateStats(tensor.data);
    if (cast > 40) {
      refuse(`The image carries colour (channel spread ${cast}); every radiograph ` +
             'this model was trained on is greyscale.');
      return;
    }
    if (g.contrast < gate.min_contrast) {
      refuse(`Too little contrast (${g.contrast.toFixed(1)}, the gate needs ` +
             `${gate.min_contrast}) — a blank or near-blank field.`);
      return;
    }
    if (g.correlation < gate.min_correlation || g.distance > gate.max_distance) {
      refuse(`The layout does not match a frontal chest radiograph ` +
             `(correlation ${g.correlation.toFixed(2)} against the required ` +
             `${gate.min_correlation}, distance ${g.distance.toFixed(2)} against ` +
             `${gate.max_distance}). A radiograph of another body part, or a ` +
             'rotated or cropped chest, lands here.');
      return;
    }
  }
  $('refused').hidden = true;

  const t0 = performance.now();
  const res = await session.run({ input: tensor });
  const ms = performance.now() - t0;

  const names = Object.keys(res);
  const get = (want) => res[names.find((n) => n.toLowerCase().includes(want))];
  const verdict = get('verdict') || res[names[0]];
  const p = readProbs(verdict);
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
