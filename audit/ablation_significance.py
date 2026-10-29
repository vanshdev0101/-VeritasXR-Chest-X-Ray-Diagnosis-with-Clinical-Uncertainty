"""Paired bootstrap for the ablation, the same test used for the main models.

Comparing three seed means with a 2x-pooled-std rule of thumb is weaker and
less principled than resampling the test set with both models scored on the
same cases. Done per seed, so a result that depends on one seed is visible
rather than averaged away -- which is how this study was misled once already.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as ap

R = Path("results")
N = 2000
SEEDS = (0, 1, 2)


def scores(run, seed):
    p = R / f"{run}_seed{seed}" / "test_scores.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    return z["y_true"].astype(int), z["y_score"].astype(float)


def paired(y, a, b, rng):
    idx = rng.integers(0, len(y), size=(N, len(y)))
    d = np.empty(N)
    for i in range(N):
        yi = y[idx[i]]
        d[i] = ap(yi, a[idx[i]]) - ap(yi, b[idx[i]]) if yi.min() != yi.max() else np.nan
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return d.mean(), lo, hi, bool(lo > 0 or hi < 0)


def main():
    variants = sorted({d.name.rsplit("_seed", 1)[0] for d in R.iterdir()
                       if d.name.startswith("veritasxr_") and "_seed" in d.name
                       and (d / "test_scores.npz").is_file()})
    variants = [v for v in variants if v != "veritasxr_full"]

    print(f"{'variant':<16}" + "".join(f"{'seed'+str(s):>22}" for s in SEEDS)
          + f"{'signs agree':>13}")
    print("-" * (16 + 22 * len(SEEDS) + 13))
    out = {}
    for v in variants:
        cells, seps, signs = [], [], []
        for s in SEEDS:
            f, g = scores("veritasxr_full", s), scores(v, s)
            if f is None or g is None:
                cells.append(f"{'n/a':>22}")
                continue
            rng = np.random.default_rng(s)
            m, lo, hi, sep = paired(f[0], g[1], f[1], rng)
            cells.append(f"{m:>+9.4f} {'sep' if sep else '  —':>4}      ")
            seps.append(sep)
            signs.append(np.sign(m))
        agree = len(set(signs)) == 1 if signs else False
        verdict = ("consistent" if agree and all(seps)
                   else "mixed" if agree else "SIGN FLIPS")
        print(f"{v.replace('veritasxr_',''):<16}" + "".join(cells) + f"{verdict:>13}")
        out[v] = {"separated_in": sum(seps), "signs_agree": agree}
    print("\n  'sep' = the paired 95% interval excludes zero for that seed.")
    print("  A variant is only reportable as a finding if the sign is stable")
    print("  across seeds; separation on one seed alone is what misled this")
    print("  study before.")
    Path("audit/ablation_significance.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
