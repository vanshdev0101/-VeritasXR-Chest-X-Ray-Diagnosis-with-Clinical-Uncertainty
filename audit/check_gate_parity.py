"""The gate is fitted in Python and enforced in JavaScript. This asserts the
two agree on the same pixels, because a silent drift between them either lets
photographs through or turns real radiographs away, and neither shows up as an
error anywhere else.

    python audit/check_gate_parity.py
"""
import csv, json, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from fit_input_gate import SIZE, describe, pool, zscore  # noqa: E402

gate = json.loads((REPO / "docs/data/input_gate.json").read_text())
mu, sd = np.array(gate["mean"]), np.array(gate["std"])

rows = list(csv.DictReader(open(REPO / "data/processed/test.csv")))[:12]
tensors, expect = [], []
for r in rows:
    a = np.asarray(Image.open(REPO / "data/processed" / r["path"]).convert("L")
                   .resize((SIZE, SIZE), Image.BILINEAR), np.float32)
    t = (a / 255 - 0.5) / 0.5
    tensors.append(t.ravel().tolist())
    z = zscore(pool(a)).ravel()
    expect.append({"contrast": float(a.std()),
                   "distance": float((((z - mu) / sd) ** 2).mean()),
                   "correlation": float(z @ mu / (np.linalg.norm(z) * np.linalg.norm(mu) + 1e-9))})

got = json.loads(subprocess.run(
    ["node", str(REPO / "audit/gate_parity.mjs")], input=json.dumps({"tensors": tensors}),
    capture_output=True, text=True, check=True).stdout)

worst = {k: 0.0 for k in ("contrast", "distance", "correlation")}
for e, g in zip(expect, got):
    for k in worst:
        worst[k] = max(worst[k], abs(e[k] - g[k]))

tol = {"contrast": 1e-2, "distance": 1e-4, "correlation": 1e-6}
print(f"  {len(rows)} radiographs scored in both languages")
for k, v in worst.items():
    print(f"  max |Δ {k}|{'':<4} {v:.3e}   (tolerance {tol[k]:.0e})")
bad = [k for k, v in worst.items() if v > tol[k]]
if bad:
    raise SystemExit(f"\n  FAIL: Python and JavaScript disagree on {', '.join(bad)}")

# The decision, not just the numbers: both sides must accept the same images.
def rejects(s):
    return (s["contrast"] < gate["min_contrast"] or s["distance"] > gate["max_distance"]
            or s["correlation"] < gate["min_correlation"])
assert [rejects(e) for e in expect] == [rejects(g) for g in got], "verdicts differ"
print(f"  same accept/reject decision on all {len(rows)}")
print("\n  PASS: the browser enforces the gate that Python fitted")
