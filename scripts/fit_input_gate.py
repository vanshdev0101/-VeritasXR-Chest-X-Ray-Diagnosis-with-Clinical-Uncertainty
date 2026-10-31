"""Fit and validate the input gate that stands in front of the classifier.

The classifier has no way to refuse. Run a photograph through it and it returns
a confident NORMAL, because predictive entropy measures distance from the
decision boundary and a photograph is nowhere near it. Measured on the
deployed checkpoint: 4 of 7 ordinary photographs landed in the "higher
confidence" band, and a flat white field was reported as 68% pneumonia.

So the refusal has to happen before inference, on the pixels. A chest
radiograph has a layout: bright mediastinal column down the centre, dark lung
fields either side, bright chest wall at the edges. This fits a per-pixel mean
and standard deviation over a 16x16 average-pool of the training split, after
z-scoring each image so exposure and contrast drop out, and scores an input by
the mean squared z-distance from that template.

What this is: a layout check. It rejects photographs, flat fields, noise, and
radiographs of the wrong body part, because none of them put a bright column
down the middle. What it is not: a judgement about image quality, or proof an
accepted image is a chest radiograph. It is one cheap statistic, and the
threshold is a percentile of the validation split, not a clinical standard.

    python scripts/fit_input_gate.py            # fit, validate, write JSON
"""
import csv, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
PROC = REPO / "data/processed"
OUT = REPO / "docs/data/input_gate.json"
GRID = 16          # 224 average-pools to 16x16 in blocks of 14
SIZE = 224


def pool(arr224):
    """224x224 float -> GRID x GRID means. demo.js does the same block mean."""
    b = SIZE // GRID
    return arr224.reshape(GRID, b, GRID, b).mean((1, 3))


def zscore(v):
    """Per-image standardisation, so exposure and contrast drop out."""
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-6 else 1e-6)


def features(path):
    im = Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    return describe(np.asarray(im, np.float32))


def describe(a224):
    """(z-scored 16x16 layout, raw contrast). Contrast is kept because
    z-scoring destroys it, and a blank field z-scores to the same vector as a
    faint one -- which is how a pure white image reached 68% pneumonia."""
    return np.append(zscore(pool(a224)).ravel(), a224.std())


def load_split(name):
    rows = list(csv.DictReader(open(PROC / f"{name}.csv")))
    paths = [str(PROC / r["path"]) for r in rows]
    with ProcessPoolExecutor() as ex:
        return np.stack(list(ex.map(features, paths, chunksize=64)))


def distance(F, mu, sd):
    return (((F[..., :-1] - mu) / sd) ** 2).mean(axis=-1)


def correlation(F, mu):
    """Cosine similarity to the mean chest layout. Mean squared distance is
    dominated by how far each cell is; this asks whether the bright column and
    the dark fields are where they belong at all."""
    z, m = F[..., :-1], mu
    return (z @ m) / (np.linalg.norm(z, axis=-1) * np.linalg.norm(m) + 1e-9)


def reject(F, g):
    """The shipped rule. Any one of three checks turns an input away."""
    return ((F[..., -1] < g["min_contrast"])
            | (distance(F, np.array(g["mean"]), np.array(g["std"])) > g["max_distance"])
            | (correlation(F, np.array(g["mean"])) < g["min_correlation"]))


# ---------- out-of-distribution probes, all built from held-out radiographs ----------
# No licensed non-chest radiograph corpus ships with this repo, so "wrong body
# part" is stood in for by crops that keep radiographic texture and destroy the
# chest layout: a quarter of a chest is not a chest, the same way a hand is not.
def probes(paths, rng):
    out = {k: [] for k in ("rot90", "quarter_crop", "inverted", "half", "noise", "flat")}
    for p in paths:
        im = Image.open(p).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
        a = np.asarray(im, np.float32)
        out["rot90"].append(describe(np.rot90(a).copy()))
        c = np.asarray(im.crop((0, 0, SIZE // 2, SIZE // 2)).resize((SIZE, SIZE), Image.BILINEAR), np.float32)
        out["quarter_crop"].append(describe(c))
        out["inverted"].append(describe(255 - a))
        h = np.asarray(im.crop((0, 0, SIZE // 2, SIZE)).resize((SIZE, SIZE), Image.BILINEAR), np.float32)
        out["half"].append(describe(h))
        out["noise"].append(describe(rng.uniform(0, 255, (SIZE, SIZE))))
        out["flat"].append(describe(np.full((SIZE, SIZE), rng.uniform(20, 235))))
    return {k: np.stack(v) for k, v in out.items()}


def auroc(pos, neg):
    """P(a random OOD scores higher than a random radiograph). No sklearn here."""
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    rp = r[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    rng = np.random.default_rng(0)
    print("fitting on train split...")
    train = load_split("train")
    mu, sd = train[:, :-1].mean(0), train[:, :-1].std(0).clip(min=1e-3)

    val, test = load_split("val"), load_split("test")

    # Each cut sits at the 2nd percentile of validation, for a combined
    # false-reject rate near 4%. Looser was tried first and does not hold: at a
    # 0.5% budget the gate let 33% of rotated radiographs through. Headroom is
    # not free either -- every point of it is a real chest film the page
    # refuses to read -- so 4% is the price of turning away 95% of radiographs
    # of the wrong body part (21 of 22, measured; see audit/ood_xray_urls.txt).
    gate = {
        "grid": GRID, "size": SIZE,
        "mean": [round(v, 4) for v in mu.tolist()],
        "std": [round(v, 4) for v in sd.tolist()],
        "max_distance": round(float(np.percentile(distance(val, mu, sd), 98)), 4),
        "min_correlation": round(float(np.percentile(correlation(val, mu), 2)), 4),
        "min_contrast": round(float(np.percentile(val[:, -1], 2)), 3),
    }

    probe_paths = [str(PROC / r["path"]) for r in
                   list(csv.DictReader(open(PROC / "test.csv")))[:400]]
    pr = probes(probe_paths, rng)
    d_test = distance(test, mu, sd)

    print(f"\n  grid {GRID}x{GRID}  distance>{gate['max_distance']}  "
          f"correlation<{gate['min_correlation']}  contrast<{gate['min_contrast']}")
    print(f"\n  {'split / probe':16} {'n':>6} {'median d':>9} {'median r':>9} "
          f"{'turned away':>12}   AUROC vs test")
    for name, F in [("val (in-dist)", val), ("test (in-dist)", test)]:
        print(f"  {name:16} {len(F):6d} {np.median(distance(F, mu, sd)):9.3f} "
              f"{np.median(correlation(F, mu)):9.3f} {100*reject(F, gate).mean():11.2f}%")
    rows = {}
    for name, F in pr.items():
        d, a = distance(F, mu, sd), None
        a = auroc(d, d_test)
        rows[name] = (len(F), float(np.median(d)), float(reject(F, gate).mean()), float(a))
        print(f"  {name:16} {len(F):6d} {np.median(d):9.3f} "
              f"{np.median(correlation(F, mu)):9.3f} {100*reject(F, gate).mean():11.2f}%"
              f"        {a:.4f}")

    gate.update({
        "fitted_on": "train split; 16x16 average pool of the 224x224 grayscale "
                     "tensor, per-image z-scored, plus raw contrast",
        "threshold_rule": "each cut at the 2nd percentile of the validation split",
        "measured": {
            "non_chest_radiographs_turned_away": "21/22 public hand, knee, pelvis, "
                "foot, skull and ankle radiographs (audit/ood_xray_urls.txt). The "
                "one that passes is a horse hoof.",
            "photographs_turned_away": "11/11 ordinary photographs",
            "scanned_chest_film": "accepted -- it is a chest radiograph, which is "
                "the gate working, not failing",
        },
        "test_reject_rate": round(float(reject(test, gate).mean()), 5),
        "probes": {k: {"n": v[0], "median_distance": round(v[1], 3),
                       "turned_away": round(v[2], 4), "auroc_vs_test": round(v[3], 4)}
                   for k, v in rows.items()},
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gate, indent=1) + "\n")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
