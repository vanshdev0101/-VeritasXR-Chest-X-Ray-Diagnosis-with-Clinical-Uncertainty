"""Validation-tuned operating points for every completed deep run.

The harness now records this for new runs. These runs predate that, so the
validation scores are recomputed here from the saved checkpoints.

Why it matters: F1, sensitivity and specificity at a fixed 0.5 compare models
at the same raw cut even though they differ in calibration by more than an
order of magnitude. In this study that reversed the F1 ranking -- at 0.5 the
table put ResNet50 first, contradicting AUPRC and AUROC, and a validation-tuned
cut removes the contradiction.

    python scripts/eval_thresholds.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import RSNADataset  # noqa: E402
from src.models.registry import build_model, build_transforms  # noqa: E402
from src.training.harness import metrics_from, tune_threshold  # noqa: E402

MODELS = ("efficientnet_b0", "densenet121", "resnet50", "veritasxr")
SEEDS = (0, 1, 2)


@torch.no_grad()
def infer(model, split, spec, cfg, root, dev):
    ds = RSNADataset(root / cfg["data"][split], root,
                     transform=build_transforms(spec, cfg, train=False))
    dl = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)
    S, Y = [], []
    for x, y in dl:
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model(x)
        lg = out[0] if isinstance(out, (tuple, list)) else out
        S.append(torch.softmax(lg.float(), 1)[:, 1].cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(S)


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    root = Path(cfg["data"]["root"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}

    for name in MODELS:
        spec = cfg["per_model"][name]
        per_seed = []
        for seed in SEEDS:
            ck = Path(f"models/{name}_seed{seed}.pt")
            if not ck.is_file():
                continue
            m = build_model(name, spec)
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            m.eval().to(dev)
            yv, sv = infer(m, "val", spec, cfg, root, dev)
            yt, st = infer(m, "test", spec, cfg, root, dev)
            t = tune_threshold(yv, sv)
            per_seed.append({
                "seed": seed, "threshold": t,
                "at_half": metrics_from(yt, st, 0.5),
                "at_tuned": metrics_from(yt, st, t),
            })
            del m
            torch.cuda.empty_cache()
        if not per_seed:
            continue
        agg = {}
        for key in ("f1", "sensitivity", "specificity", "balanced_accuracy"):
            for where in ("at_half", "at_tuned"):
                v = [p[where][key] for p in per_seed]
                agg[f"{key}_{where}"] = {"mean": round(float(np.mean(v)), 4),
                                         "std": round(float(np.std(v)), 4)}
        out[name] = {"per_seed": per_seed, "agg": agg,
                     "threshold_mean": round(float(np.mean(
                         [p["threshold"] for p in per_seed])), 3),
                     "thresholds": [p["threshold"] for p in per_seed]}
        a = out[name]["agg"]
        print(f"  {name:<18} t={out[name]['threshold_mean']:.2f}  "
              f"F1@0.5 {a['f1_at_half']['mean']:.4f} -> "
              f"F1@tuned {a['f1_at_tuned']['mean']:.4f} "
              f"({a['f1_at_tuned']['mean']-a['f1_at_half']['mean']:+.4f})")

    p = Path("docs/data/thresholds.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    r1 = sorted(out, key=lambda k: -out[k]["agg"]["f1_at_half"]["mean"])
    r2 = sorted(out, key=lambda k: -out[k]["agg"]["f1_at_tuned"]["mean"])
    print(f"\n  F1 ranking @0.5   : {' > '.join(r1)}")
    print(f"  F1 ranking @tuned : {' > '.join(r2)}")
    print(f"  ranking changes   : {'YES' if r1 != r2 else 'no'}")
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
