/* Scores tensors with the browser's own gate code, so the Python fit and the
 * JavaScript that enforces it cannot drift apart unnoticed. Reads
 * {"tensors": [[...224*224 floats...], ...]} on stdin, writes one scored
 * object per tensor. Driven by audit/check_gate_parity.py.
 *
 *   python audit/check_gate_parity.py
 */
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';

const root = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'docs');
const stub = () => ({
  dataset: {}, style: {}, classList: { add() {}, remove() {} },
  addEventListener() {}, setAttribute() {}, insertAdjacentHTML() {},
  getContext: () => ({
    drawImage() {}, clearRect() {}, putImageData() {},
    getImageData: () => ({ data: new Uint8ClampedArray(32 * 32 * 4) }),
  }),
  set innerHTML(v) {}, set textContent(v) {}, hidden: false,
});

const ctx = {
  console: { ...console, info() {}, warn() {} },
  document: {
    documentElement: stub(), getElementById: stub, createElement: stub,
    querySelectorAll: () => [], addEventListener() {},
  },
  localStorage: { getItem: () => null, setItem() {} },
  matchMedia: () => ({ matches: false }),
  getComputedStyle: () => ({ getPropertyValue: () => '#000' }),
  requestAnimationFrame: (f) => f(), location: { reload() {} },
  Image: class { set src(_) {} }, performance: { now: () => 0 },
  setTimeout, clearTimeout,
  ort: { env: { wasm: {} }, InferenceSession: { create: async () => { throw new Error('no model'); } } },
  fetch: async (p) => {
    const text = readFileSync(path.join(root, String(p).split('?')[0]), 'utf8');
    return { ok: true, status: 200, headers: { get: () => null }, json: async () => JSON.parse(text) };
  },
};
ctx.window = ctx; ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(readFileSync(path.join(root, 'assets/js/demo.js'), 'utf8'), ctx);

const input = JSON.parse(readFileSync(0, 'utf8'));
await ctx.loadModel();                       // fetches input_gate.json, then fails on ort
const out = input.tensors.map((t) => ctx.gateStats(Float32Array.from(t)));
process.stdout.write(JSON.stringify(out));
