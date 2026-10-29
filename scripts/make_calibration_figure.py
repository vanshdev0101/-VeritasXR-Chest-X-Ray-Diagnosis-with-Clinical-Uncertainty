"""Reliability diagram: confidence against observed accuracy.

A perfectly calibrated model sits on the diagonal. Above it the model is
underconfident, below it overconfident -- which is the usual direction and the
one that matters when the number is read as a probability.

    python scripts/make_calibration_figure.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABEL = {"veritasxr": "VeritasXR", "efficientnet_b0": "EfficientNet-B0",
         "resnet50": "ResNet50", "densenet121": "DenseNet121",
         "hog_svm": "HOG + SVM"}
STYLE = {"veritasxr": ("#1E40AF", "-", "o"), "efficientnet_b0": ("#0F766E", "--", "^"),
         "resnet50": ("#B45309", ":", "s"), "densenet121": ("#7E22CE", "-.", "D"),
         "hog_svm": ("#475569", (0, (1, 2)), "x")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("docs/data/calibration.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/assets/img/calibration.png"))
    args = ap.parse_args()
    d = json.loads(args.data.read_text())

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax.plot([0.5, 1], [0.5, 1], color="#94A3B8", lw=1, ls=(0, (4, 4)),
            label="perfect calibration", zorder=1)

    for k, v in sorted(d.items(), key=lambda kv: kv[1]["ece"]):
        col, ls, mk = STYLE.get(k, ("#333", "-", "o"))
        pts = [(b["confidence"], b["accuracy"]) for b in v["bins"]
               if b["n"] and b["confidence"] is not None and b["n"] >= 15]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, ls=ls, marker=mk, ms=5, lw=1.8, color=col,
                label=f"{LABEL.get(k, k)}  (ECE {v['ece']:.3f})", zorder=2)

    ax.set_xlabel("predicted confidence"); ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability diagram", fontsize=11)
    ax.set_xlim(0.45, 1.02); ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(alpha=.25)
    ax.text(.98, .06, "below the line = overconfident", transform=ax.transAxes,
            ha="right", fontsize=8, color="#64748B", style="italic")

    names = sorted(d, key=lambda k: d[k]["ece"])
    y = np.arange(len(names))
    bx.barh(y, [d[k]["ece"] for k in names],
            color=[STYLE.get(k, ("#333",))[0] for k in names], height=.6)
    bx.set_yticks(y); bx.set_yticklabels([LABEL.get(k, k) for k in names], fontsize=9)
    bx.invert_yaxis()
    bx.set_xlabel("expected calibration error (lower is better)")
    bx.set_title("Calibration error", fontsize=11)
    for i, k in enumerate(names):
        bx.text(d[k]["ece"] + .004, i, f"{d[k]['ece']:.4f}", va="center", fontsize=8)
    bx.set_xlim(0, max(d[k]["ece"] for k in names) * 1.25)
    bx.grid(axis="x", alpha=.25)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight", facecolor="white")
    try:
        from PIL import Image as _I
        _I.open(args.out).convert("P", palette=_I.ADAPTIVE, colors=128).save(
            args.out, optimize=True)
    except Exception as exc:
        print(f"  (palette optimisation skipped: {exc})")
    print(f"wrote {args.out} ({args.out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
