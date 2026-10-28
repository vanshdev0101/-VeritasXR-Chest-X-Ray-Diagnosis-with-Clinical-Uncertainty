"""Turn the RSNA Pneumonia competition download into PNGs plus split manifests.

Reads stage_2_train_labels.csv (one row per bounding box, so several rows per
image) and stage_2_detailed_class_info.csv, caches each DICOM as a PNG, and
writes train/val/test manifests split by patientId.

Boxes are stored normalised to [0, 1] against the original DICOM dimensions, so
the manifests stay valid whatever resolution the cache is built at.

Usage:
    python scripts/prepare_rsna.py --raw data/raw --out data/processed
"""
import argparse, json, random, zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom
from PIL import Image

# Cached larger than the 224 the models train at, so augmentation has room to
# crop without a second pass over 26k DICOMs.
CACHE_PX = 256
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


# stage_2_test_images/ is the competition's held-out set and carries no labels,
# so it is skipped: 3,000 DICOMs of no use to a supervised study.
WANTED = ("stage_2_train_labels.csv", "stage_2_detailed_class_info.csv",
          "stage_2_train_images/")


def unzip(raw: Path) -> Path:
    """Unpack the labelled members once; returns the directory holding them."""
    archives = list(raw.glob("*.zip"))
    if not archives:
        return raw
    target = raw / "extracted"
    if (target / "stage_2_train_labels.csv").is_file():
        print(f"already extracted at {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    for a in archives:
        with zipfile.ZipFile(a) as z:
            members = [m for m in z.namelist() if m.startswith(WANTED)]
            print(f"extracting {len(members):,} of {len(z.namelist()):,} "
                  f"members from {a.name} (skipping the unlabelled test set) ...")
            z.extractall(target, members=members)
    return target


def convert(task):
    """DICOM -> PNG. Returns (patientId, width, height) or None on failure."""
    dcm_path, png_path = task
    pid = Path(dcm_path).stem
    try:
        if Path(png_path).exists():
            with Image.open(png_path) as im:
                im.verify()
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            return pid, int(ds.Columns), int(ds.Rows)
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array
        h, w = arr.shape[:2]
        Image.fromarray(arr).convert("L").resize(
            (CACHE_PX, CACHE_PX), Image.LANCZOS
        ).save(png_path, optimize=True)
        return pid, w, h
    except Exception as exc:  # a corrupt DICOM should not kill the whole run
        print(f"  !! {pid}: {exc}")
        return None


def build_records(labels: pd.DataFrame, classes: pd.DataFrame) -> dict:
    """Collapse the one-row-per-box CSV into one record per image."""
    cls = dict(zip(classes.patientId, classes["class"]))
    recs = {}
    for row in labels.itertuples(index=False):
        r = recs.setdefault(
            row.patientId,
            {"patientId": row.patientId, "target": 0,
             "class": cls.get(row.patientId, "unknown"), "boxes": []},
        )
        if int(row.Target) == 1 and pd.notna(row.x):
            r["target"] = 1
            r["boxes"].append([float(row.x), float(row.y),
                               float(row.width), float(row.height)])
    return recs


def split_by_patient(recs: dict, seed: int = SEED) -> dict:
    """Proportional split keyed on patientId, stratified on the 3-class label.

    Every row belonging to one patientId lands in exactly one split, which is
    what tests/test_no_leakage.py asserts.
    """
    by_class = defaultdict(list)
    for pid, r in recs.items():
        by_class[r["class"]].append(pid)

    rng = random.Random(seed)
    out = {k: [] for k in SPLITS}
    for pids in by_class.values():
        pids = sorted(pids)          # sort first so the shuffle is reproducible
        rng.shuffle(pids)
        n = len(pids)
        n_tr = int(n * SPLITS["train"])
        n_va = int(n * SPLITS["val"])
        out["train"] += pids[:n_tr]
        out["val"] += pids[n_tr:n_tr + n_va]
        out["test"] += pids[n_tr + n_va:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = unzip(args.raw)
    labels = pd.read_csv(root / "stage_2_train_labels.csv")
    classes = pd.read_csv(root / "stage_2_detailed_class_info.csv")
    print(f"labels: {len(labels):,} rows | classes: {len(classes):,} rows")

    recs = build_records(labels, classes)
    print(f"unique images: {len(recs):,}")

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    dcm_dir = root / "stage_2_train_images"
    tasks = [(str(dcm_dir / f"{p}.dcm"), str(img_dir / f"{p}.png")) for p in recs]

    print(f"converting {len(tasks):,} DICOMs at {CACHE_PX}px with {args.workers} workers ...")
    dims, failed = {}, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(convert, tasks, chunksize=32), 1):
            if res is None:
                failed += 1
            else:
                pid, w, h = res
                dims[pid] = (w, h)
            if i % 2500 == 0:
                print(f"  {i:,}/{len(tasks):,}")
    print(f"converted {len(dims):,} | failed {failed}")

    splits = split_by_patient({k: v for k, v in recs.items() if k in dims})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, pids in splits.items():
        rows = []
        for pid in pids:
            r = recs[pid]
            w, h = dims[pid]
            rows.append({
                "patientId": pid,
                "path": f"images/{pid}.png",
                "target": r["target"],
                "class": r["class"],
                # normalised so the manifest survives a change of CACHE_PX
                "boxes": json.dumps([[x / w, y / h, bw / w, bh / h]
                                     for x, y, bw, bh in r["boxes"]]),
                "orig_w": w,
                "orig_h": h,
            })
        df = pd.DataFrame(rows).sort_values("patientId")
        df.to_csv(args.out / f"{name}.csv", index=False)
        pos = int(df.target.sum())
        print(f"{name:5} {len(df):6,} images | positive {pos:6,} ({pos/len(df):.1%})")


if __name__ == "__main__":
    main()
