"""Gate test for the RSNA splits. Run it after scripts/prepare_rsna.py.

    python tests/test_no_leakage.py [--data data/processed]

Plain asserts, no test framework. The final check deliberately corrupts an
in-memory copy of the splits to prove the leakage assertion actually fires --
a check that cannot fail is not a check.
"""
import argparse, json, sys
from itertools import combinations
from pathlib import Path

import pandas as pd

NAMES = ("train", "val", "test")
TOL = 1e-6


def load(data: Path):
    return {n: pd.read_csv(data / f"{n}.csv") for n in NAMES}


def check_disjoint(splits):
    """No patientId may appear in more than one split."""
    for a, b in combinations(NAMES, 2):
        overlap = set(splits[a].patientId) & set(splits[b].patientId)
        assert not overlap, (
            f"LEAKAGE: {len(overlap)} patientId(s) in both {a} and {b}, "
            f"e.g. {sorted(overlap)[:3]}"
        )
    print(f"  ok  all {len(list(combinations(NAMES, 2)))} split pairs are disjoint")


def check_unique_within(splits):
    for n, df in splits.items():
        dupes = df.patientId.duplicated().sum()
        assert dupes == 0, f"{n}: {dupes} duplicated patientId rows"
    print("  ok  no duplicate patientId inside any split")


def check_files_exist(splits, data: Path):
    missing = [p for df in splits.values() for p in df.path
               if not (data / p).is_file()]
    assert not missing, f"{len(missing)} manifest rows point at missing images, e.g. {missing[:3]}"
    print(f"  ok  all {sum(len(d) for d in splits.values()):,} referenced images exist")


def check_boxes(splits):
    """Boxes present exactly when target == 1, and normalised inside [0, 1]."""
    for n, df in splits.items():
        for row in df.itertuples(index=False):
            boxes = json.loads(row.boxes)
            if row.target == 1:
                assert boxes, f"{n}/{row.patientId}: target=1 but no boxes"
            else:
                assert not boxes, f"{n}/{row.patientId}: target=0 but {len(boxes)} boxes"
            for x, y, w, h in boxes:
                assert w > 0 and h > 0, f"{n}/{row.patientId}: degenerate box {(x,y,w,h)}"
                assert -TOL <= x <= 1 + TOL and -TOL <= y <= 1 + TOL, \
                    f"{n}/{row.patientId}: box origin outside [0,1]: {(x, y)}"
                assert x + w <= 1 + TOL and y + h <= 1 + TOL, \
                    f"{n}/{row.patientId}: box extends past the image: {(x, y, w, h)}"
    print("  ok  boxes present iff target==1, all normalised within [0,1]")


def check_stratification(splits):
    """Positive rate should not drift much between splits."""
    rates = {n: df.target.mean() for n, df in splits.items()}
    spread = max(rates.values()) - min(rates.values())
    assert spread < 0.02, f"positive rate drifts {spread:.3f} across splits: {rates}"
    print("  ok  positive rate within 2pp across splits: "
          + ", ".join(f"{n} {r:.1%}" for n, r in rates.items()))


def check_proportions(splits):
    total = sum(len(d) for d in splits.values())
    fracs = {n: len(d) / total for n, d in splits.items()}
    for n, want in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        assert abs(fracs[n] - want) < 0.02, f"{n} is {fracs[n]:.1%}, expected ~{want:.0%}"
    print("  ok  split sizes ~70/15/15: "
          + ", ".join(f"{n} {f:.1%}" for n, f in fracs.items()))


def check_detects_leakage(splits):
    """Negative control: inject a leak and confirm check_disjoint catches it."""
    poisoned = {n: df.copy() for n, df in splits.items()}
    stolen = poisoned["train"].iloc[[0]]
    poisoned["test"] = pd.concat([poisoned["test"], stolen], ignore_index=True)
    try:
        check_disjoint(poisoned)
    except AssertionError:
        print("  ok  negative control: injected leak was detected")
        return
    raise AssertionError(
        "check_disjoint passed on deliberately leaked splits -- the gate is broken"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    missing = [n for n in NAMES if not (args.data / f"{n}.csv").is_file()]
    if missing:
        sys.exit(f"missing manifest(s) {missing} in {args.data} -- run scripts/prepare_rsna.py first")

    splits = load(args.data)
    print(f"checking {args.data}: "
          + ", ".join(f"{n}={len(d):,}" for n, d in splits.items()))

    check_disjoint(splits)
    check_unique_within(splits)
    check_files_exist(splits, args.data)
    check_boxes(splits)
    check_stratification(splits)
    check_proportions(splits)
    check_detects_leakage(splits)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
