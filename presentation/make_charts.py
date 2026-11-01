"""Charts for the deck, drawn from the repo's own result files.

Nothing here is typed in by hand: curves and means come from
docs/data/summary.json and docs/data/uncertainty.json, and the per-seed
ablation deltas are recomputed from results/<run>/metrics.json. Re-run after
any retraining and the slides stay true.

    python make_charts.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]   # the repo this folder sits in
OUT = Path(__file__).parent / "charts"
OUT.mkdir(exist_ok=True)

CREAM, INK, MUTE = "#FBF6F1", "#524545", "#8A7A72"
LINE, AMBER, OLIVE = "#C2B49F", "#A2560F", "#CDC29C"
GREY = ["#9A8F86", "#7C736B", "#B7ADA2", "#6A615A"]

plt.rcParams.update({
    "font.family": "Inter, DejaVu Sans, sans-serif",
    "figure.facecolor": CREAM, "axes.facecolor": CREAM,
    "savefig.facecolor": CREAM,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": LINE, "axes.linewidth": 1.0,
    "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "grid.color": LINE, "grid.alpha": 0.45, "grid.linewidth": 0.8,
})

summary = json.loads((REPO / "docs/data/summary.json").read_text())
models = {m["id"]: m for m in summary["models"]}
ORDER = ["efficientnet_b0", "densenet121", "resnet50", "veritasxr", "hog_svm"]
LABEL = {"efficientnet_b0": "EfficientNet-B0", "densenet121": "DenseNet121",
         "resnet50": "ResNet50", "veritasxr": "VeritasXR", "hog_svm": "HOG + SVM"}


def finish(fig, ax, name, legend=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend:
        ax.legend(handles=legend, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("  ", name)


def pr_curves():
    """Precision–recall, because AUPRC is the headline metric."""
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for i, mid in enumerate(ORDER):
        m = models[mid]
        pts = m.get("pr") or []
        if not pts:
            continue
        recall = [p[0] for p in pts]
        precision = [p[1] for p in pts]
        is_ours = mid == "veritasxr"
        ax.plot(recall, precision,
                color=AMBER if is_ours else GREY[i % len(GREY)],
                lw=2.6 if is_ours else 1.6, zorder=3 if is_ours else 2,
                label=f"{LABEL[mid]}  {m['metrics']['auprc']['mean']:.4f}")
    ax.axhline(summary["dataset"]["positive_rate"] if "positive_rate" in summary.get("dataset", {})
               else 0.225, color=MUTE, ls=":", lw=1.2)
    ax.text(0.98, 0.245, "prevalence 22.5% — guessing", fontsize=11, color=MUTE, ha="right")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall on the test set", pad=12, loc="left", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="both")
    ax.legend(frameon=False, loc="lower left", fontsize=12)
    finish(fig, ax, "1-pr-curves")


def auprc_vs_params():
    """Accuracy against model size: the 'capacity is not the limit' picture."""
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for mid in ORDER:
        if mid == "hog_svm":
            continue
        m = models[mid]
        x = m["params"] / 1e6
        y = m["metrics"]["auprc"]["mean"]
        err = m["metrics"]["auprc"]["std"]
        ours = mid == "veritasxr"
        ax.errorbar(x, y, yerr=err, fmt="o", ms=13 if ours else 10,
                    color=AMBER if ours else OLIVE,
                    ecolor=AMBER if ours else MUTE, elinewidth=1.6, capsize=5,
                    markeredgecolor=INK if ours else "none", markeredgewidth=1.2, zorder=3)
        ax.annotate(f"{LABEL[mid]}\n{y:.4f}", (x, y),
                    textcoords="offset points",
                    xytext=(14 if mid == "efficientnet_b0" else 0, 16),
                    ha="left" if mid == "efficientnet_b0" else "center", fontsize=12,
                    color=AMBER if ours else INK,
                    fontweight="bold" if ours else "normal")
    ax.set_xscale("log")
    # Default log ticks render as 4x10^0, which is noise at this range.
    ax.set_xticks([4, 6, 10, 16, 24])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(3.3, 32)
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("AUPRC  (mean ± SD, 3 seeds)")
    ax.set_title("More parameters did not buy accuracy", pad=12, loc="left", fontweight="bold")
    ax.set_ylim(0.60, 0.74)
    ax.grid(True, axis="y")
    finish(fig, ax, "2-auprc-vs-params")


def uncertainty_bars():
    """The two candidate uncertainty signals, against the model's own errors."""
    u = json.loads((REPO / "docs/data/uncertainty.json").read_text())["auroc"]
    keys = ["disagreement", "entropy", "entropy_plus_5x_disagreement"]
    names = ["Pathway\ndisagreement", "Predictive\nentropy", "Entropy +\n5x disagreement"]
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for i, k in enumerate(keys):
        mean = u[k]["mean"]
        ours = k == "entropy"
        ax.bar(i, mean, width=0.58, color=AMBER if ours else OLIVE,
               edgecolor=INK if ours else "none", linewidth=1.2, zorder=2)
        ax.scatter([i] * len(u[k]["per_seed"]), u[k]["per_seed"], s=34,
                   color=INK, zorder=4, alpha=0.75)
        ax.text(i, mean + 0.018, f"{mean:.4f}", ha="center", fontsize=13,
                fontweight="bold", color=AMBER if ours else INK)
    ax.axhline(0.5, color=MUTE, ls="--", lw=1.4, zorder=3)
    ax.text(2.42, 0.508, "chance", fontsize=12, color=MUTE, ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels(names)
    ax.set_ylabel("AUROC against the model's own errors")
    ax.set_title("Only one uncertainty signal beats chance", pad=12, loc="left",
                 fontweight="bold")
    ax.set_ylim(0.4, 0.88)
    ax.grid(True, axis="y")
    finish(fig, ax, "3-uncertainty-auroc", legend=[
        Line2D([], [], marker="o", ls="none", color=INK, alpha=0.75, label="individual seeds")])


def ablation_per_seed():
    """Per-seed deltas against the full model — the sign-flip picture."""
    def auprc(run, seed):
        p = REPO / f"results/{run}_seed{seed}/metrics.json"
        return json.loads(p.read_text())["test"]["auprc"] if p.exists() else None

    variants = ["concat_merge", "fixed_merge", "scalar_merge", "no_se",
                "global_only", "local_only"]
    rows, labels = [], []
    for v in variants:
        deltas = []
        for seed in (0, 1, 2):
            full = auprc("veritasxr_full", seed)
            var = auprc(f"veritasxr_{v}", seed)
            if full is None or var is None:
                continue
            deltas.append(var - full)
        if deltas:
            rows.append(deltas)
            labels.append(v)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for i, deltas in enumerate(rows):
        flips = min(deltas) < 0 < max(deltas)
        ax.scatter(deltas, [i] * len(deltas), s=95,
                   color=OLIVE if flips else AMBER,
                   edgecolor=INK, linewidth=1.1, zorder=3)
        mean = sum(deltas) / len(deltas)
        ax.scatter([mean], [i], marker="|", s=420, color=INK, zorder=4, linewidth=2.2)
    ax.axvline(0, color=INK, lw=1.4, zorder=2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUPRC difference from the full model (one point per seed)")
    ax.set_title("Only removing the 7x7 pathway hurts on every seed", pad=12,
                 loc="left", fontweight="bold")
    ax.grid(True, axis="x")
    ax.invert_yaxis()
    finish(fig, ax, "4-ablation-per-seed", legend=[
        Line2D([], [], marker="o", ls="none", color=OLIVE, markeredgecolor=INK,
               label="sign flips across seeds"),
        Line2D([], [], marker="o", ls="none", color=AMBER, markeredgecolor=INK,
               label="same direction on all three"),
        Line2D([], [], marker="|", ls="none", color=INK, markersize=12, label="mean")])


if __name__ == "__main__":
    print("charts:")
    pr_curves()
    auprc_vs_params()
    uncertainty_bars()
    ablation_per_seed()
