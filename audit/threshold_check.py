"""Does the fixed 0.5 operating point misrepresent the models?

Every threshold-dependent metric in the study (F1, sensitivity, specificity,
balanced accuracy) is computed at 0.5. These models differ in calibration by
more than an order of magnitude, so 0.5 sits at a different place on each
model's distribution. A model that is well calibrated and one that is
overconfident should not be compared at the same raw cut.

Tunes the threshold on the VALIDATION split, then applies it to test. Val
scores are not stored by the harness, so they are recomputed here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import RSNADataset  # noqa: E402
from src.models.registry import build_model, build_transforms  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
root = Path(cfg["data"]["root"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def infer(model, split, spec):
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


def best_threshold(y, s):
    ts = np.unique(np.round(np.linspace(0.05, 0.95, 91), 3))
    f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f1s))]), float(max(f1s))


def main():
    print(f"{'model':<18}{'tuned t':>9}{'F1@0.5':>9}{'F1@tuned':>10}"
          f"{'ΔF1':>8}{'sens@t':>9}{'spec@t':>9}")
    print("-" * 72)
    rows = []
    for name in ("efficientnet_b0", "densenet121", "resnet50", "veritasxr"):
        ck = Path(f"models/{name}_seed0.pt")
        if not ck.is_file():
            continue
        spec = cfg["per_model"][name]
        m = build_model(name, spec)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval().to(dev)
        yv, sv = infer(m, "val", spec)
        yt, st = infer(m, "test", spec)
        t, _ = best_threshold(yv, sv)
        f05 = f1_score(yt, (st >= 0.5).astype(int), zero_division=0)
        ft = f1_score(yt, (st >= t).astype(int), zero_division=0)
        pred = (st >= t).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
        tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
        print(f"{name:<18}{t:>9.2f}{f05:>9.4f}{ft:>10.4f}{ft-f05:>+8.4f}"
              f"{tp/(tp+fn):>9.4f}{tn/(tn+fp):>9.4f}")
        rows.append((name, f05, ft))
        del m
        torch.cuda.empty_cache()

    print()
    r05 = [n for n, a, _ in sorted(rows, key=lambda r: -r[1])]
    rt = [n for n, _, b in sorted(rows, key=lambda r: -r[2])]
    print(f"  F1 ranking @0.5   : {' > '.join(r05)}")
    print(f"  F1 ranking @tuned : {' > '.join(rt)}")
    print(f"  ranking changes: {'YES' if r05 != rt else 'no'}")


if __name__ == "__main__":
    main()
