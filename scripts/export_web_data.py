"""Export run results as one JSON the static site can read.

The browser cannot open test_scores.npz, and recomputing curves in JavaScript
would duplicate logic that already exists in Python. So curve points, bootstrap
intervals and summary metrics are computed once here and written to
docs/data/summary.json.

S5 extends this with significance tests; the shape is designed to absorb that
without the frontend changing.

    python scripts/export_web_data.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

LABELS = {
    "veritasxr": ("VeritasXR", "custom dual-pathway CNN, from scratch"),
    "resnet50": ("ResNet50", "residual CNN, ImageNet pretrained"),
    "densenet121": ("DenseNet121", "dense CNN, ImageNet pretrained (CheXNet backbone)"),
    "efficientnet_b0": ("EfficientNet-B0", "efficient CNN, ImageNet pretrained"),
    "hog_svm": ("HOG + linear SVM", "classical ML, no deep learning"),
}
MAX_CURVE_POINTS = 220


def thin(xs, ys, n=MAX_CURVE_POINTS):
    """Even subsample so curves stay faithful but small on the wire."""
    if len(xs) <= n:
        idx = np.arange(len(xs))
    else:
        idx = np.unique(np.linspace(0, len(xs) - 1, n).astype(int))
    return [[round(float(xs[i]), 4), round(float(ys[i]), 4)] for i in idx]


def paired_diff(y, a, b, fn, n=2000, seed=0):
    """Bootstrap the difference between two models on the SAME resampled cases.

    Comparing two marginal confidence intervals and checking whether they
    overlap is a weaker and misleading test: intervals can overlap while the
    paired difference is clearly non-zero, because resampling the same cases
    for both models cancels the shared difficulty of those cases.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d = np.empty(n)
    for i in range(n):
        yi = y[idx[i]]
        d[i] = (fn(yi, a[idx[i]]) - fn(yi, b[idx[i]])
                if yi.min() != yi.max() else np.nan)
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4),
            "separated": bool(lo > 0 or hi < 0)}


def bootstrap_ci(y, s, fn, n=2000, seed=0):
    """Percentile bootstrap over the test set; the reason scores are saved."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    for i in range(n):
        yi = y[idx[i]]
        vals[i] = fn(yi, s[idx[i]]) if yi.min() != yi.max() else np.nan
    vals = vals[~np.isnan(vals)]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def collect(results: Path):
    runs = defaultdict(list)
    for d in sorted(results.iterdir()):
        f = d / "metrics.json"
        if d.is_dir() and f.is_file():
            m = json.loads(f.read_text())
            runs[m.get("run") or m["model"]].append((d, m))
    return runs


def learning_rate_probe(runs):
    """Seed-0 AUPRC per model per learning rate, including the shared rate.

    Tests whether one shared lr for every model is what holds the custom model
    back. Ablation variants are excluded: they share the rate but not the
    architecture, so they would overwrite the shared-rate point.
    """
    shared, pts = None, defaultdict(dict)
    for entries in runs.values():
        for _, m in entries:
            if m["seed"] != 0 or m.get("ablation") is not None:
                continue
            shared = m["config"]["train"]["lr"]
            pts[m["model"]][m.get("lr") or shared] = round(m["test"]["auprc"], 4)

    rows = []
    for model, by_lr in pts.items():
        if len(by_lr) < 2 or shared not in by_lr:
            continue                      # no probe was run for this model
        best = max(by_lr, key=by_lr.get)
        rows.append({
            "model": model,
            "label": LABELS.get(model, (model, ""))[0],
            "points": sorted(by_lr.items()),
            "best_lr": best,
            "gain": round(by_lr[best] - by_lr[shared], 4),
        })
    return {"shared_lr": shared, "seed": 0,
            "models": sorted(rows, key=lambda r: -r["gain"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("docs/data/summary.json"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    runs = collect(args.results)
    models, ablations = [], []

    for run, entries in sorted(runs.items()):
        base = entries[0][1]["model"]
        m_any = entries[0][1]
        is_abl = m_any.get("ablation") is not None
        # Learning-rate probe runs are reported on their own, not as models.
        if m_any.get("lr") not in (None, m_any["config"]["train"]["lr"]):
            continue
        seeds = sorted(m["seed"] for _, m in entries)

        # mean +/- std across seeds for every headline metric
        agg = {}
        for key in ("auprc", "auroc", "f1", "balanced_accuracy",
                    "sensitivity", "specificity", "accuracy"):
            vals = [m["test"][key] for _, m in entries]
            agg[key] = {"mean": round(float(np.mean(vals)), 4),
                        "std": round(float(np.std(vals)), 4),
                        "n": len(vals)}

        # Curves and intervals come from the first seed, so every model is
        # represented by the same seed rather than a cherry-picked best.
        d0, m0 = sorted(entries, key=lambda e: e[1]["seed"])[0]
        entry = {
            "id": run,
            "label": LABELS.get(base, (run, ""))[0] if not is_abl else run,
            "note": LABELS.get(base, ("", ""))[1] if not is_abl else "",
            "ablation": entries[0][1].get("ablation"),
            "kind": m0.get("kind"),
            "params": m0.get("params_total"),
            "seeds": seeds,
            "epochs": [m.get("epochs_run") for _, m in entries],
            "train_minutes": round(float(np.mean(
                [m.get("train_seconds", 0) for _, m in entries])) / 60, 1),
            "metrics": agg,
            "confusion": m0["test"]["confusion"],
            "history": [
                {"epoch": h["epoch"], "loss": round(h["train_loss"], 4),
                 "val_auprc": round(h["val_auprc"], 4),
                 "val_auroc": round(h["val_auroc"], 4)}
                for h in m0.get("history", [])
            ],
        }

        # Every seed's scores, not just the first: the paired significance test
        # below was running on seed 0 alone, which is the same single-seed
        # mistake that reversed the ablation and the calibration results.
        entry["_all_scores"] = {}
        for d, m in entries:
            f = d / "test_scores.npz"
            if f.is_file():
                z = np.load(f, allow_pickle=True)
                entry["_all_scores"][m["seed"]] = (z["y_true"].astype(int),
                                                   z["y_score"].astype(float))

        npz = d0 / "test_scores.npz"
        if npz.is_file():
            z = np.load(npz, allow_pickle=True)
            y, s = z["y_true"].astype(int), z["y_score"].astype(float)
            fpr, tpr, _ = roc_curve(y, s)
            prec, rec, _ = precision_recall_curve(y, s)
            entry["roc"] = thin(fpr, tpr)
            entry["pr"] = thin(rec, prec)
            # Averaged over seeds, like the paired test: a CI from one run
            # beside a three-seed mean invites reading one as the other.
            cis = [bootstrap_ci(yy, ss, average_precision_score, args.bootstrap)
                   for yy, ss in entry["_all_scores"].values()]
            entry["ci"] = {
                "auprc": [round(float(np.mean([c[0] for c in cis])), 4),
                          round(float(np.mean([c[1] for c in cis])), 4)],
                "seeds": len(cis),
            }
            entry["_scores"] = (y, s)
            # Risk-coverage: accuracy over the most confident x% of cases.
            conf = np.abs(s - 0.5)
            order = np.argsort(-conf)
            correct = ((s >= 0.5).astype(int) == y)[order]
            cov = np.arange(0.05, 1.0001, 0.05)
            entry["risk_coverage"] = [
                [round(float(c), 2),
                 round(float(correct[:max(int(len(correct) * c), 1)].mean()), 4)]
                for c in cov
            ]
        (ablations if is_abl else models).append(entry)

    models.sort(key=lambda m: -m["metrics"]["auprc"]["mean"])

    # Paired significance against the custom model, which is the comparison the
    # study is actually making.
    ref = next((m for m in models if m["id"] == "veritasxr"), None)
    if ref is not None and ref.get("_all_scores"):
        for m in models:
            if m is ref or not m.get("_all_scores"):
                continue
            per_seed = []
            for seed, (yr, sr) in sorted(ref["_all_scores"].items()):
                if seed not in m["_all_scores"]:
                    continue
                r = paired_diff(yr, m["_all_scores"][seed][1], sr,
                                average_precision_score, args.bootstrap)
                per_seed.append(dict(r, seed=seed))
            if not per_seed:
                continue
            signs = {r["diff"] > 0 for r in per_seed}
            m["vs_veritasxr"] = {
                "diff": round(float(np.mean([r["diff"] for r in per_seed])), 4),
                "ci": [round(float(np.mean([r["ci"][0] for r in per_seed])), 4),
                       round(float(np.mean([r["ci"][1] for r in per_seed])), 4)],
                "p": max(r["p"] for r in per_seed),
                "separated": all(r["separated"] for r in per_seed),
                "same_sign": len(signs) == 1,
                "per_seed": per_seed,
                "seeds": len(per_seed),
            }
    for m in models + ablations:
        m.pop("_scores", None)
        m.pop("_all_scores", None)
    test_n = models[0]["confusion"]
    pos = test_n["tp"] + test_n["fn"]
    total = sum(test_n.values())

    payload = {
        "dataset": {
            "name": "RSNA Pneumonia Detection Challenge",
            "images": 26684,
            "split": {"train": 18677, "val": 4001, "test": 4006},
            "prevalence": round(pos / total, 4),
            "population": "adults, multi-site (from NIH ChestX-ray14)",
            "labels": "1-3 board-certified radiologists, with bounding boxes",
        },
        "baselines": {
            "always_negative_accuracy": round(1 - pos / total, 4),
            "prevalence_auprc": round(pos / total, 4),
        },
        "models": models,
        "ablations": sorted(ablations, key=lambda m: -m["metrics"]["auprc"]["mean"]),
        "lr_probe": learning_rate_probe(runs),
        "pending": [k for k in ("significance", "localisation") ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({kb:.0f} KB): {len(models)} models, "
          f"{len(ablations)} ablations")
    for m in models:
        ci = m.get("ci", {}).get("auprc")
        print(f"  {m['label']:<18} AUPRC {m['metrics']['auprc']['mean']:.4f}"
              f" +/- {m['metrics']['auprc']['std']:.4f}"
              f"  95% CI {ci}  seeds {m['seeds']}")


if __name__ == "__main__":
    main()
