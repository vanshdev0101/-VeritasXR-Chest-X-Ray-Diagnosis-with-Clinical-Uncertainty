"""The uncertainty signal, on the whole test set and on every seed.

This table was published from 800 test images and one seed. It is the table
the demo's triage bands rest on, so it is the last place in the study that
should be a single draw -- four other single-seed results in this project
reversed on replication.

Three signals are scored against whether the model was actually wrong:

  Jensen-Shannon disagreement between the two pathway heads -- the original
    design, which the architecture was built around;
  predictive entropy of the merged verdict -- what the model reports instead;
  entropy + 5x disagreement -- whether the rejected signal adds anything to
    the one that works.

    python scripts/eval_uncertainty.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import RSNADataset  # noqa: E402
from src.models.registry import build_transforms  # noqa: E402
from src.models.veritasxr import get_model  # noqa: E402

SEEDS = (0, 1, 2)
COVERAGE = (0.25, 0.50, 0.75, 1.00)


@torch.no_grad()
def run(model, split, spec, cfg, root, dev):
    """Verdict probability, entropy and disagreement for every image."""
    ds = RSNADataset(root / cfg["data"][split], root,
                     transform=build_transforms(spec, cfg, train=False))
    dl = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)
    P, H, D, Y = [], [], [], []
    for x, y in dl:
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            verdict, unc, _, _, dis = model(x, aux=True)
        P.append(torch.softmax(verdict.float(), 1)[:, 1].cpu().numpy())
        H.append(unc.float().squeeze(1).cpu().numpy())
        D.append(dis.float().squeeze(1).cpu().numpy())
        Y.append(y.numpy())
    return (np.concatenate(Y), np.concatenate(P),
            np.concatenate(H), np.concatenate(D))


def selective(err, signal):
    """Accuracy over the most confident fraction, ordered by the signal."""
    order = np.argsort(signal)
    return {f"{c:.2f}": round(float(1 - err[order[:max(int(len(err) * c), 1)]].mean()), 4)
            for c in COVERAGE}


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    spec = cfg["per_model"]["veritasxr"]
    root = Path(cfg["data"]["root"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed, bands = [], []
    for seed in SEEDS:
        ckpt = Path("models") / f"veritasxr_seed{seed}.pt"
        if not ckpt.is_file():
            continue
        model = get_model().to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.eval()

        y, p, h, d = run(model, "test", spec, cfg, root, dev)
        # Bands are set on validation, never on test, so they are recorded here
        # from the same checkpoint rather than inherited from a previous run.
        _, _, hv, _ = run(model, "val", spec, cfg, root, dev)
        del model
        torch.cuda.empty_cache()

        err = ((p >= 0.5).astype(int) != y).astype(int)
        row = {
            "seed": seed,
            "n": int(y.size),
            "error_rate": round(float(err.mean()), 4),
            "auroc": {
                "disagreement": round(float(roc_auc_score(err, d)), 4),
                "entropy": round(float(roc_auc_score(err, h)), 4),
                "entropy_plus_5x_disagreement":
                    round(float(roc_auc_score(err, h + 5 * d)), 4),
            },
            "disagreement": {"max": round(float(d.max()), 4),
                             "p99": round(float(np.percentile(d, 99)), 4),
                             "mean": round(float(d.mean()), 4)},
            "selective_entropy": selective(err, h),
            "bands": {"p60": round(float(np.percentile(hv, 60)), 4),
                      "p85": round(float(np.percentile(hv, 85)), 4)},
        }
        per_seed.append(row)
        bands.append(row["bands"])
        print(f"  seed {seed}  disagreement {row['auroc']['disagreement']:.4f}"
              f"  entropy {row['auroc']['entropy']:.4f}"
              f"  combined {row['auroc']['entropy_plus_5x_disagreement']:.4f}"
              f"  bands {row['bands']['p60']:.3f}/{row['bands']['p85']:.3f}")

    if not per_seed:
        print("  no veritasxr checkpoints found")
        return

    def agg(path):
        vals = [eval_path(r, path) for r in per_seed]
        return {"mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "per_seed": vals}

    def eval_path(row, path):
        v = row
        for k in path:
            v = v[k]
        return v

    out = {
        "seeds": [r["seed"] for r in per_seed],
        "n": per_seed[0]["n"],
        "auroc": {k: agg(("auroc", k)) for k in per_seed[0]["auroc"]},
        "disagreement_max": agg(("disagreement", "max")),
        "selective_entropy": {c: agg(("selective_entropy", c))
                              for c in per_seed[0]["selective_entropy"]},
        "bands": {k: agg(("bands", k)) for k in per_seed[0]["bands"]},
        "error_rate": agg(("error_rate",)),
        "per_seed": per_seed,
    }
    p = Path("docs/data/uncertainty.json")
    p.write_text(json.dumps(out, indent=1))

    print(f"\n  {'signal':<34}{'AUROC vs error':>18}")
    print("  " + "-" * 52)
    for k, v in sorted(out["auroc"].items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {k:<34}{v['mean']:>10.4f} +/-{v['std']:.4f}")
    print(f"\n  full test set ({out['n']} images), {len(per_seed)} seeds."
          f"\n  wrote {p}")


if __name__ == "__main__":
    main()
