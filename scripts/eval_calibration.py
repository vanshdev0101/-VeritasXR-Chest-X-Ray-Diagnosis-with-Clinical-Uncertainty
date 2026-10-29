"""Are the predicted probabilities believable as probabilities?

AUPRC only cares about ranking. A model can rank perfectly while being
systematically overconfident, which matters here: the triage bands are read off
the predicted distribution, so if the numbers are inflated the bands inherit
that inflation.

Reported per model:

  ECE    expected calibration error -- mean gap between confidence and accuracy,
         weighted by bin occupancy. 0 is perfect.
  MCE    the worst single bin, which ECE can hide.
  Brier  squared error on the probability itself; proper scoring rule, so it
         penalises confident mistakes rather than just ranking them late.

    python scripts/eval_calibration.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

BINS = 15


def reliability(y, p, bins=BINS):
    """Per-bin confidence vs observed accuracy, as the diagram plots it."""
    edges = np.linspace(0, 1, bins + 1)
    conf = np.maximum(p, 1 - p)                 # confidence in the predicted class
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)

    rows, ece, mce = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            rows.append({"lo": round(float(lo), 3), "hi": round(float(hi), 3),
                         "n": 0, "confidence": None, "accuracy": None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        gap = abs(c - a)
        ece += m.mean() * gap
        mce = max(mce, gap)
        rows.append({"lo": round(float(lo), 3), "hi": round(float(hi), 3),
                     "n": int(m.sum()), "confidence": round(c, 4),
                     "accuracy": round(a, 4)})
    return rows, float(ece), float(mce)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("docs/data/calibration.json"))
    args = ap.parse_args()

    out = {}
    for d in sorted(args.results.glob("*_seed0")):
        f, z = d / "metrics.json", d / "test_scores.npz"
        if not (f.is_file() and z.is_file()):
            continue
        m = json.loads(f.read_text())
        if m.get("ablation"):
            continue
        a = np.load(z, allow_pickle=True)
        y, p = a["y_true"].astype(int), a["y_score"].astype(float)
        rows, ece, mce = reliability(y, p)
        out[m["model"]] = {
            "bins": rows,
            "ece": round(ece, 4),
            "mce": round(mce, 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "mean_confidence": round(float(np.maximum(p, 1 - p).mean()), 4),
            "accuracy": round(float(((p >= 0.5).astype(int) == y).mean()), 4),
            "n": int(y.size),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"  {'model':<18}{'ECE':>8}{'MCE':>8}{'Brier':>8}"
          f"{'mean conf':>11}{'accuracy':>10}{'gap':>8}")
    print("  " + "-" * 71)
    for k, v in sorted(out.items(), key=lambda kv: kv[1]["ece"]):
        gap = v["mean_confidence"] - v["accuracy"]
        print(f"  {k:<18}{v['ece']:>8.4f}{v['mce']:>8.4f}{v['brier']:>8.4f}"
              f"{v['mean_confidence']:>11.4f}{v['accuracy']:>10.4f}{gap:>+8.4f}")
    print("\n  gap > 0 means the model is overconfident: it claims more certainty"
          "\n  than its accuracy supports.")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
