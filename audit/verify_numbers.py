"""Recompute every headline number from the run artifacts.

Nothing here reads the README or the site. The point is to produce the truth
independently, so published values can be checked against it rather than
against themselves.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("results")


def load_runs():
    out = defaultdict(list)
    for d in sorted(R.iterdir()):
        f = d / "metrics.json"
        if f.is_file():
            m = json.loads(f.read_text())
            out[m.get("run") or m["model"]].append(m)
    return out


def main():
    runs = load_runs()
    main_runs = {k: v for k, v in runs.items() if v[0].get("ablation") is None}
    abl = {k: v for k, v in runs.items() if v[0].get("ablation")}

    print("=" * 78)
    print("A1. MAIN COMPARISON  (recomputed from results/*/metrics.json)")
    print("=" * 78)
    print(f"{'model':<18}{'n':>3}{'params':>11}{'AUPRC':>18}{'AUROC':>18}")
    for k, e in sorted(main_runs.items(),
                       key=lambda kv: -np.mean([m["test"]["auprc"] for m in kv[1]])):
        a = np.array([m["test"]["auprc"] for m in e])
        r = np.array([m["test"]["auroc"] for m in e])
        print(f"{k:<18}{len(e):>3}{e[0]['params_total']:>11,}"
              f"{a.mean():>11.4f} ±{a.std():.4f}{r.mean():>11.4f} ±{r.std():.4f}")

    print()
    print("=" * 78)
    print("A1. ABLATION  (3 seeds where available)")
    print("=" * 78)
    full = np.mean([m["test"]["auprc"] for m in abl["veritasxr_full"]])
    full_sd = np.std([m["test"]["auprc"] for m in abl["veritasxr_full"]])
    print(f"{'variant':<16}{'n':>3}{'params':>11}{'AUPRC':>18}{'delta':>10}{'|d|/sd':>9}")
    for k, e in sorted(abl.items(),
                       key=lambda kv: -np.mean([m["test"]["auprc"] for m in kv[1]])):
        a = np.array([m["test"]["auprc"] for m in e])
        d = a.mean() - full
        pooled = np.sqrt((a.std() ** 2 + full_sd ** 2) / 2) or 1e-9
        print(f"{e[0]['ablation']:<16}{len(e):>3}{e[0]['params_total']:>11,}"
              f"{a.mean():>11.4f} ±{a.std():.4f}{d:>+10.4f}{abs(d)/pooled:>9.1f}")

    print()
    print("=" * 78)
    print("A3. PUBLISHED JSON vs RECOMPUTED")
    print("=" * 78)
    p = Path("docs/data/summary.json")
    if p.is_file():
        d = json.loads(p.read_text())
        ok = True
        for m in d["models"]:
            live = np.mean([x["test"]["auprc"] for x in main_runs[m["id"]]])
            diff = abs(live - m["metrics"]["auprc"]["mean"])
            flag = "OK" if diff < 5e-5 else f"MISMATCH {diff:.5f}"
            if diff >= 5e-5:
                ok = False
            print(f"  summary.json {m['id']:<18} published {m['metrics']['auprc']['mean']:.4f}"
                  f"  recomputed {live:.4f}  {flag}")
        print(f"  ablations in summary.json: {len(d.get('ablations', []))}"
              f"  (on disk: {len(abl)})")
        print(f"  -> {'consistent' if ok else 'INCONSISTENT'}")
    else:
        print("  summary.json missing")

    p = Path("docs/data/localisation.json")
    if p.is_file():
        d = json.loads(p.read_text())
        v = d.get("veritasxr", {})
        has_sd = "seed_std" in v.get("local_point", {})
        print(f"  localisation.json: multi-seed = {has_sd}, "
              f"seeds = {v.get('seeds', 'n/a')}")
    # Every published file that aggregates over runs must say how many runs,
    # and must not be one. A single-seed calibration table shipped here once
    # and overstated the custom model's advantage by an order of magnitude.
    for name, seeds_of in (("calibration.json", lambda v: len(v.get("seeds", []))),
                           ("temperature.json", lambda v: v.get("seeds", 0))):
        p = Path("docs/data") / name
        if not p.is_file():
            print(f"  {name}: not generated")
            continue
        d = json.loads(p.read_text())
        counts = {k: seeds_of(v) for k, v in d.items()}
        bad = [k for k, n in counts.items() if n < 3]
        best = min(d, key=lambda k: d[k].get("ece", d[k].get("agg", {})
                                             .get("after", {}).get("ece", 9)))
        print(f"  {name}: {len(d)} models, seeds {sorted(set(counts.values()))},"
              f" best = {best}")
        if bad:
            print(f"    !! single-seed or incomplete entries: {bad}")


if __name__ == "__main__":
    main()
