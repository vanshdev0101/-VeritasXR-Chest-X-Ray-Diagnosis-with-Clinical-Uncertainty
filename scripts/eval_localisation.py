"""Score every model's attention against the radiologist boxes.

    python scripts/eval_localisation.py

Runs over the positive test images -- the ones that carry boxes -- and reports,
per model, how often the attention peak lands inside an annotated opacity and
how much of the activation mass falls there.

For VeritasXR the two pathways are scored separately as well as merged, which
is the direct test of whether local and global attend to different places.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.localisation import (GradCAM, cam_iou, mass_in_box,  # noqa: E402
                                         peak_in_box, summarise, veritasxr_cams)
from src.models.registry import NORMALIZE, build_model  # noqa: E402
from src.models.veritasxr import get_model  # noqa: E402

# The last convolutional block of each backbone. DenseNet is hooked at
# denseblock4 rather than at `features`, because its forward applies an
# in-place ReLU to the output of `features` and autograd refuses to let a
# backward hook's view be modified in place.
GRAD_LAYER = {"resnet50": lambda m: m.layer4,
              "densenet121": lambda m: m.features.denseblock4,
              "efficientnet_b0": lambda m: m.features[-1]}


def load_image(path, channels, mean, std, size=224):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR),
                       dtype=np.float32) / 255.0
    x = torch.from_numpy(a)[None, None]
    if channels == 3:
        x = x.repeat(1, 3, 1, 1)
    m = torch.tensor(mean).view(1, -1, 1, 1)
    s = torch.tensor(std).view(1, -1, 1, 1)
    return (x - m) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out", type=Path, default=Path("docs/data/localisation.json"))
    ap.add_argument("-n", type=int, default=0, help="limit images (0 = all positives)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    dev = torch.device(args.device)
    df = pd.read_csv(args.data / "test.csv")
    df = df[df.target == 1].reset_index(drop=True)
    if args.n:
        df = df.head(args.n)
    print(f"  {len(df)} positive test images, device {dev}")

    results = {}

    # ---- VeritasXR: per-pathway CAM, no gradients needed ----
    vx = get_model()
    vx.load_state_dict(torch.load("models/veritasxr_seed0.pt", map_location="cpu"))
    vx.eval().to(dev)
    spec = cfg["per_model"]["veritasxr"]
    mean, std = NORMALIZE[spec["normalize"]]
    rec = {f"{k}_{m}": [] for k in ("local", "global", "merged")
           for m in ("point", "mass", "iou")}
    agree = []
    for row in df.itertuples(index=False):
        boxes = json.loads(row.boxes)
        x = load_image(args.data / row.path, 1, mean, std).to(dev)
        cams = veritasxr_cams(vx, x)
        cl, cg = cams["local"][0], cams["global"][0]
        merged = (cl - cl.min()) / (np.ptp(cl) or 1) + (cg - cg.min()) / (np.ptp(cg) or 1)
        for key, cam in (("local", cl), ("global", cg), ("merged", merged)):
            rec[f"{key}_point"].append(peak_in_box(cam, boxes))
            rec[f"{key}_mass"].append(mass_in_box(cam, boxes))
            rec[f"{key}_iou"].append(cam_iou(cam, boxes))
        agree.append(int(np.argmax(cl)) == int(np.argmax(cg)))
    results["veritasxr"] = summarise(rec)
    results["veritasxr"]["pathway_peak_agreement"] = {
        "mean": round(float(np.mean(agree)), 4), "n": len(agree)}
    print(f"  veritasxr   local point {results['veritasxr']['local_point']['mean']:.3f} | "
          f"global point {results['veritasxr']['global_point']['mean']:.3f}")

    # ---- baselines: Grad-CAM ----
    for name in ("resnet50", "densenet121", "efficientnet_b0"):
        ckpt = Path(f"models/{name}_seed0.pt")
        if not ckpt.is_file():
            print(f"  {name}: no checkpoint, skipped")
            continue
        spec = cfg["per_model"][name]
        m, s = NORMALIZE[spec["normalize"]]
        model = build_model(name, spec)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval().to(dev)
        gc = GradCAM(model, GRAD_LAYER[name](model))
        rec = {"point": [], "mass": [], "iou": []}
        for row in df.itertuples(index=False):
            boxes = json.loads(row.boxes)
            x = load_image(args.data / row.path, 3, m, s).to(dev)
            cam = gc(x)[0]
            rec["point"].append(peak_in_box(cam, boxes))
            rec["mass"].append(mass_in_box(cam, boxes))
            rec["iou"].append(cam_iou(cam, boxes))
        results[name] = summarise(rec)
        print(f"  {name:<16} point {results[name]['point']['mean']:.3f}")
        del model, gc
        torch.cuda.empty_cache()

    # ---- chance reference: how often a random point lands in a box ----
    rng = np.random.default_rng(0)
    hits, cover = [], []
    for row in df.itertuples(index=False):
        boxes = json.loads(row.boxes)
        px, py = rng.random(2)
        hits.append(any(b[0] <= px <= b[0]+b[2] and b[1] <= py <= b[1]+b[3] for b in boxes))
        cover.append(sum(b[2]*b[3] for b in boxes))
    results["_chance"] = {"random_point_in_box": round(float(np.mean(hits)), 4),
                          "mean_box_area_fraction": round(float(np.mean(cover)), 4),
                          "n": len(hits)}
    print(f"  chance      random point in box {results['_chance']['random_point_in_box']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
