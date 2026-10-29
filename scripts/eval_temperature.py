"""Does VeritasXR's calibration advantage survive temperature scaling?

VeritasXR's one clear win over the pretrained baselines is calibration: an ECE
of 0.0059 against 0.0210 for the next best. The obvious objection is that
calibration is cheap to buy after the fact -- fit one temperature on the
validation split and any overconfident model moves onto the diagonal without
its ranking changing at all. If the baselines close the gap for the cost of a
single scalar, then "best calibrated" is a statement about VeritasXR's default
output scale, not about the architecture.

So it is measured. One temperature per run, fitted by minimising validation
negative log-likelihood, applied to the test probabilities. Scaling a binary
logit is monotone, so AUPRC and AUROC are unchanged by construction -- the
script asserts that, which is also the check that the transform was applied
correctly.

Classical HOG + SVM is excluded: its probabilities come from squashing an
unbounded margin through a logistic, so a temperature is fitting the squash
rather than correcting a trained model.

    python scripts/eval_temperature.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_calibration import reliability  # noqa: E402
from scripts.eval_thresholds import infer  # noqa: E402
from src.models.registry import build_model  # noqa: E402

MODELS = ("efficientnet_b0", "densenet121", "resnet50", "veritasxr")
SEEDS = (0, 1, 2)
EPS = 1e-6


def fit_temperature(y, p):
    """Temperature on the binary logit, by validation NLL. T > 1 cools."""
    z = logit(np.clip(p, EPS, 1 - EPS))

    def nll(log_t):
        q = np.clip(expit(z / np.exp(log_t)), EPS, 1 - EPS)
        return -float(np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    r = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(r.x))


def apply_temperature(p, t):
    return expit(logit(np.clip(p, EPS, 1 - EPS)) / t)


def scores(y, p):
    rows, ece, mce = reliability(y, p)
    return {"ece": round(ece, 4), "mce": round(mce, 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "mean_confidence": round(float(np.maximum(p, 1 - p).mean()), 4),
            "accuracy": round(float(((p >= 0.5).astype(int) == y).mean()), 4)}


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    root = Path(cfg["data"]["root"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}

    for name in MODELS:
        spec = cfg["per_model"][name]
        per_seed = []
        for seed in SEEDS:
            ckpt = Path("models") / f"{name}_seed{seed}.pt"
            npz = Path("results") / f"{name}_seed{seed}" / "test_scores.npz"
            if not (ckpt.is_file() and npz.is_file()):
                continue
            model = build_model(name, spec).to(dev)
            model.load_state_dict(torch.load(ckpt, map_location=dev))
            model.eval()
            yv, pv = infer(model, "val", spec, cfg, root, dev)
            del model
            torch.cuda.empty_cache()

            a = np.load(npz, allow_pickle=True)
            yt, pt = a["y_true"].astype(int), a["y_score"].astype(float)
            t = fit_temperature(yv, pv)
            qt = apply_temperature(pt, t)

            # Monotone in the logit, so ranking metrics must not move. This is
            # the correctness check on the transform, not a result.
            for fn in (average_precision_score, roc_auc_score):
                assert abs(fn(yt, pt) - fn(yt, qt)) < 1e-6, (name, seed, fn)

            per_seed.append({"seed": seed, "temperature": round(t, 4),
                             "before": scores(yt, pt), "after": scores(yt, qt)})
            print(f"  {name:<18} seed {seed}  T={t:.3f}  "
                  f"ECE {per_seed[-1]['before']['ece']:.4f} -> "
                  f"{per_seed[-1]['after']['ece']:.4f}")

        if not per_seed:
            continue
        agg = {"temperature": round(float(np.mean(
            [r["temperature"] for r in per_seed])), 4)}
        for when in ("before", "after"):
            agg[when] = {k: round(float(np.mean([r[when][k] for r in per_seed])), 4)
                         for k in per_seed[0][when]}
        out[name] = {"per_seed": per_seed, "agg": agg, "seeds": len(per_seed)}

    p = Path("docs/data/temperature.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"\n  {'model':<18}{'T':>7}{'ECE before':>12}{'ECE after':>11}"
          f"{'Brier before':>14}{'Brier after':>13}")
    print("  " + "-" * 75)
    for k, v in sorted(out.items(), key=lambda kv: kv[1]["agg"]["after"]["ece"]):
        a = v["agg"]
        print(f"  {k:<18}{a['temperature']:>7.3f}{a['before']['ece']:>12.4f}"
              f"{a['after']['ece']:>11.4f}{a['before']['brier']:>14.4f}"
              f"{a['after']['brier']:>13.4f}")
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
