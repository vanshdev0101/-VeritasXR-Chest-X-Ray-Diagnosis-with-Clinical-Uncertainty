"""HOG + linear SVM: the control for whether deep learning is needed at all.

This path cannot honour the shared recipe, and pretending otherwise would be
dishonest: an SVM has no epochs, no learning rate and no schedule, so "same
epochs, same scheduler, same early stopping" is undefined for it. What it does
share is the data, the splits, the class-imbalance handling (balanced class
weights), the selection metric and the results contract -- so it sits in the
same results table and the same significance tests as everything else.

It also sees no augmentation. Augmenting would mean recomputing HOG descriptors
over augmented copies, multiplying a already-expensive feature extraction for a
baseline whose purpose is to establish a floor.

Descriptor: 9 orientations, 16x16 px cells, 2x2 cell blocks, L2-Hys, on the
224px grayscale image -> 13x13x2x2x9 = 6,084 features.
"""
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.training.harness import metrics_from

HOG_KW = dict(orientations=9, pixels_per_cell=(16, 16),
              cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
SIZE = 224


def _descriptor(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize((SIZE, SIZE), Image.BILINEAR),
                         dtype=np.float32) / 255.0
    return hog(arr, **HOG_KW).astype(np.float32)


def features_for(manifest, root, workers=8, cache=None):
    """Descriptors for one split, cached to disk.

    HOG is a fixed transform with no learned parameters, so the descriptors do
    not depend on the seed -- only the SVM fit does. Without this cache the
    extraction ran once per seed for no benefit.
    """
    df = pd.read_csv(manifest)
    if cache is not None and Path(cache).is_file():
        z = np.load(cache, allow_pickle=True)
        if len(z["X"]) == len(df):
            print(f"  reusing cached descriptors {Path(cache).name}")
            return z["X"], df.target.to_numpy(), df.patientId.tolist()
    paths = [str(Path(root) / p) for p in df.path]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        feats = list(pool.map(_descriptor, paths, chunksize=64))
    X = np.stack(feats)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X)
    return X, df.target.to_numpy(), df.patientId.tolist()


def run(cfg, seed, workers=8, limit=None):
    root = Path(cfg["data"]["root"])
    t0 = time.time()

    data = {}
    for split in ("train", "val", "test"):
        m = root / cfg["data"][split]
        # Not cached for smoke runs, whose truncated splits would poison it.
        cache = None if limit else root / f"hog_cache_{split}.npz"
        X, y, ids = features_for(m, root, workers, cache=cache)
        if limit:
            X, y, ids = X[:limit], y[:limit], ids[:limit]
        data[split] = (X, y, ids)
        print(f"  {split}: {X.shape[0]:,} x {X.shape[1]:,} features")
    feat_seconds = round(time.time() - t0, 1)

    Xtr, ytr, _ = data["train"]
    clf = make_pipeline(
        StandardScaler(),
        # balanced mirrors the inverse-frequency weighting the deep models get
        LinearSVC(C=1.0, class_weight="balanced", random_state=seed,
                  dual="auto", max_iter=5000),
    )
    print("  fitting LinearSVC ...")
    clf.fit(Xtr, ytr)

    # decision_function gives an unbounded margin. Ranking metrics (AUROC,
    # AUPRC) only need the ordering, but metrics_from thresholds at 0.5, so the
    # margin is squashed to [0,1] -- a monotone map that leaves ranking intact.
    def score(X):
        d = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))

    yv, sv = data["val"][1], score(data["val"][0])
    yt, st = data["test"][1], score(data["test"][0])

    payload = {
        "model": "hog_svm", "seed": seed, "kind": "classical",
        "params_total": int(Xtr.shape[1]) + 1, "params_trainable": int(Xtr.shape[1]) + 1,
        "feature_dim": int(Xtr.shape[1]),
        "epochs_run": None, "best_epoch": None,
        "best_val_auprc": float(average_precision_score(yv, sv)),
        "feature_seconds": feat_seconds,
        "train_seconds": round(time.time() - t0, 1),
        "history": [],
        "test": metrics_from(yt, st),
        "device": "cpu",
        "config": {k: cfg[k] for k in ("data", "train")},
        "per_model": cfg["per_model"]["hog_svm"],
        "deviations": [
            "no epochs, learning rate or schedule: not defined for an SVM",
            "no augmentation: HOG descriptors are precomputed once per image",
            "scores are a sigmoid of the SVM margin, monotone so ranking is unaffected",
        ],
    }
    return payload, yt, st, data["test"][2]
