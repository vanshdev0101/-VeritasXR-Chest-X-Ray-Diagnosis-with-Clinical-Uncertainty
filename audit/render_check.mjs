/* Executes a page's JS against a stub DOM and the real JSON, so
 * a renderer that throws on a key the data no longer has fails here instead of
 * on the page. Not a browser: it checks the code paths, not the pixels.
 *
 *   cd docs && node ../audit/render_check.mjs . results.html assets/js/results.js
 *                 node ../audit/render_check.mjs . index.html   assets/js/demo.js
 *
 * Catches: an id the HTML no longer carries, a JSON key the exporter stopped
 * writing, and any renderer that silently fills nothing. */
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';

const root = process.argv[2];
const htmlFile = process.argv[3];
const jsFile = process.argv[4];
const html = readFileSync(path.join(root, htmlFile), 'utf8');
const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));

const touched = new Map();
const el = (id) => {
  if (!touched.has(id)) {
    touched.set(id, {
      id, dataset: {}, style: {}, classList: { add() {}, remove() {} },
      addEventListener() {}, setAttribute() {}, removeAttribute() {},
      appendChild() {}, getContext: () => ({}), insertAdjacentHTML(_, v) { this._html = (this._html || '') + v; },
      querySelector: () => null, querySelectorAll: () => [], focus() {}, click() {},
      set innerHTML(v) { this._html = String(v); }, get innerHTML() { return this._html || ''; },
      set textContent(v) { this._text = String(v); }, get textContent() { return this._text || ''; },
      hidden: false,
    });
  }
  return touched.get(id);
};

const errors = [];
const ctx = {
  console,
  document: {
    documentElement: el('html'),
    getElementById: (id) => (ids.has(id) ? el(id) : (errors.push(`missing #${id}`), null)),
    querySelectorAll: () => [],
    addEventListener() {},
  },
  localStorage: { getItem: () => null, setItem() {} },
  matchMedia: () => ({ matches: false }),
  getComputedStyle: () => ({ getPropertyValue: () => '#000' }),
  requestAnimationFrame: (f) => f(),
  location: { reload() {} },
  Chart: class { constructor() {} destroy() {} },
  ort: { env: { wasm: {} }, InferenceSession: { create: async () => { throw new Error('no model in stub'); } } },
  Image: class { set src(_) {} },
  performance: { now: () => 0 },
  setTimeout, clearTimeout,
  fetch: async (p) => {
    const full = path.join(root, String(p).split("?")[0]);
    try {
      const text = readFileSync(full, 'utf8');
      return { ok: true, status: 200, headers: { get: () => null },
               json: async () => JSON.parse(text), text: async () => text };
    } catch { return { ok: false, status: 404, headers: { get: () => null },
                       json: async () => null }; }
  },
};
ctx.window = ctx;
ctx.globalThis = ctx;
vm.createContext(ctx);

try {
  vm.runInContext(readFileSync(path.join(root, jsFile), 'utf8'), ctx,
                 { filename: jsFile });
} catch (e) { errors.push(`threw synchronously: ${e.message}`); }

await new Promise((r) => setTimeout(r, 400));
process.on('unhandledRejection', (e) => errors.push(`unhandled: ${e && e.message}`));
await new Promise((r) => setTimeout(r, 200));

const empty = [...touched.values()].filter(
  (e) => e.id !== 'html' && !e._html && !e._text && /table|note|headline/.test(e.id));
if (empty.length) errors.push(`rendered nothing into: ${empty.map((e) => e.id).join(', ')}`);

if (errors.length) { console.log('FAIL\n  ' + errors.join('\n  ')); process.exit(1); }
console.log(`OK  ${jsFile}: ${touched.size - 1} elements filled`);
for (const e of touched.values()) {
  if (/note$/.test(e.id) && e._html) {
    console.log(`  #${e.id}: ${e._html.replace(/\s+/g, ' ').trim().slice(0, 120)}...`);
  }
}
