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
from collections import defaultdict
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

    runs = defaultdict(list)
    for d in sorted(args.results.iterdir()):
        f, z = d / "metrics.json", d / "test_scores.npz"
        if not (d.is_dir() and f.is_file() and z.is_file()):
            continue
        m = json.loads(f.read_text())
        if m.get("ablation") or m.get("lr") not in (None, m["config"]["train"]["lr"]):
            continue
        a = np.load(z, allow_pickle=True)
        runs[m["model"]].append(
            (m["seed"], a["y_true"].astype(int), a["y_score"].astype(float)))

    out = {}
    for model, entries in runs.items():
        entries.sort()
        per_seed, all_bins = [], []
        for seed, y, p in entries:
            rows, ece, mce = reliability(y, p)
            all_bins.append(rows)
            per_seed.append({
                "seed": seed,
                "ece": round(ece, 4),
                "mce": round(mce, 4),
                "brier": round(float(np.mean((p - y) ** 2)), 4),
                "mean_confidence": round(float(np.maximum(p, 1 - p).mean()), 4),
                "accuracy": round(float(((p >= 0.5).astype(int) == y).mean()), 4),
            })

        entry = {"per_seed": per_seed, "seeds": [r["seed"] for r in per_seed],
                 "n": int(entries[0][1].size)}
        for key in ("ece", "mce", "brier", "mean_confidence", "accuracy"):
            vals = [r[key] for r in per_seed]
            entry[key] = round(float(np.mean(vals)), 4)
            entry[key + "_std"] = round(float(np.std(vals)), 4)
            entry[key + "_per_seed"] = vals

        # One reliability curve per model, occupancy-weighted across seeds, with
        # the seed-to-seed spread kept so the diagram cannot imply a precision
        # a single run does not have.
        merged = []
        for i in range(len(all_bins[0])):
            cells = [b[i] for b in all_bins if b[i]["n"]]
            if not cells:
                merged.append(dict(all_bins[0][i], n=0,
                                   confidence=None, accuracy=None))
                continue
            w = np.array([c["n"] for c in cells], dtype=float)
            accs = [c["accuracy"] for c in cells]
            merged.append({
                "lo": cells[0]["lo"], "hi": cells[0]["hi"],
                "n": int(w.sum()),
                "confidence": round(float(np.average(
                    [c["confidence"] for c in cells], weights=w)), 4),
                "accuracy": round(float(np.average(accs, weights=w)), 4),
                "accuracy_lo": round(float(min(accs)), 4),
                "accuracy_hi": round(float(max(accs)), 4),
                "seeds": len(cells),
            })
        entry["bins"] = merged
        out[model] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"  {'model':<18}{'ECE':>18}{'MCE':>8}{'Brier':>8}"
          f"{'mean conf':>11}{'accuracy':>10}{'gap':>8}")
    print("  " + "-" * 81)
    for k, v in sorted(out.items(), key=lambda kv: kv[1]["ece"]):
        gap = v["mean_confidence"] - v["accuracy"]
        ece = f"{v['ece']:.4f} +/-{v['ece_std']:.4f}"
        print(f"  {k:<18}{ece:>18}{v['mce']:>8.4f}{v['brier']:>8.4f}"
              f"{v['mean_confidence']:>11.4f}{v['accuracy']:>10.4f}{gap:>+8.4f}")
    worst = max(out, key=lambda k: out[k]["ece_std"])
    print(f"\n  ECE is seed-sensitive: {worst} ranges "
          f"{min(out[worst]['ece_per_seed']):.4f}-{max(out[worst]['ece_per_seed']):.4f}"
          f" across seeds. A single-seed calibration table is not a result.")
    print("\n  gap > 0 means the model is overconfident: it claims more certainty"
          "\n  than its accuracy supports.")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
