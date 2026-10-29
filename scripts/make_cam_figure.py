"""Render CAM overlays against the radiologist boxes, for the site and report.

Picks cases where the two pathways disagree about where to look, since that is
the claim being illustrated: the branches attend to different regions, and the
3x3 branch is the one that finds the annotated opacity.

    python scripts/make_cam_figure.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.localisation import peak_in_box, veritasxr_cams  # noqa: E402
from src.models.veritasxr import get_model  # noqa: E402

SIZE = 224


def up(cam):
    a = F.interpolate(torch.tensor(cam)[None, None].float(), size=(SIZE, SIZE),
                      mode="bilinear", align_corners=False)[0, 0].numpy()
    a = a - a.min()
    return a / (a.max() or 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("docs/assets/img/localisation.png"))
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()

    model = get_model()
    model.load_state_dict(torch.load("models/veritasxr_seed0.pt", map_location="cpu"))
    model.eval()

    df = pd.read_csv(args.data / "test.csv")
    df = df[df.target == 1].reset_index(drop=True)

    # Prefer cases the local pathway gets right and the global one does not:
    # that is the finding, shown rather than asserted.
    picks = []
    for row in df.itertuples(index=False):
        boxes = json.loads(row.boxes)
        with Image.open(args.data / row.path) as im:
            g = np.asarray(im.convert("L").resize((SIZE, SIZE), Image.BILINEAR),
                           dtype=np.float32) / 255.0
        x = torch.from_numpy((g - 0.5) / 0.5)[None, None]
        cams = veritasxr_cams(model, x)
        cl, cg = cams["local"][0], cams["global"][0]
        if peak_in_box(cl, boxes) and not peak_in_box(cg, boxes):
            picks.append((row.patientId, g, boxes, up(cl), up(cg)))
        if len(picks) >= args.rows:
            break

    if not picks:
        sys.exit("no qualifying cases found")

    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    axes = np.atleast_2d(axes)
    titles = ["radiograph + radiologist boxes", "local pathway (3x3)", "global pathway (7x7)"]
    for r, (pid, g, boxes, cl, cg) in enumerate(picks):
        for c, (img, cam) in enumerate(((g, None), (g, cl), (g, cg))):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            if cam is not None:
                ax.imshow(cam, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
                y, x = np.unravel_index(cam.argmax(), cam.shape)
                ax.plot(x, y, marker="+", ms=14, mew=2.2, color="#39FF14")
            for bx, by, bw, bh in boxes:
                ax.add_patch(mpatches.Rectangle(
                    (bx * SIZE, by * SIZE), bw * SIZE, bh * SIZE,
                    fill=False, ec="#00E5FF", lw=1.8))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
        axes[r, 0].set_ylabel(pid[:8], fontsize=7)

    fig.suptitle("Where each pathway looks, against radiologist annotations\n"
                 "cyan = radiologist box   ·   green + = attention peak",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 130 dpi produced a 1.1 MB PNG, too heavy for a page that already ships
    # a 17 MB model. 96 dpi is still legible at the size it renders.
    fig.savefig(args.out, dpi=96, bbox_inches="tight", facecolor="white")
    try:
        from PIL import Image as _I
        im = _I.open(args.out).convert("P", palette=_I.ADAPTIVE, colors=192)
        im.save(args.out, optimize=True)
    except Exception as exc:
        print(f"  (palette optimisation skipped: {exc})")
    print(f"wrote {args.out} ({args.out.stat().st_size/1024:.0f} KB), {len(picks)} cases")


if __name__ == "__main__":
    main()
