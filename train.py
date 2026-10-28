#!/usr/bin/env python3
"""Train one model under the shared recipe.

    python train.py --model resnet50 --seed 0
    python train.py --model hog_svm --seed 0
    python train.py --smoke                  # 1 epoch, few batches, all models

Every model reads configs/default.yaml, so the recipe cannot drift between
them. Results land in results/<model>_seed<seed>/.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.training.harness import train_deep, write_results  # noqa: E402

MODELS = ["veritasxr", "resnet50", "densenet121", "efficientnet_b0", "hog_svm"]


def run_one(name, cfg, seed, device, args):
    print(f"\n=== {name} | seed {seed} ===")
    if name == "hog_svm":
        from src.training.hog_svm import run as run_svm
        payload, yt, st, ids = run_svm(
            cfg, seed, workers=args.workers if args.workers is not None else 8,
            limit=args.limit_samples)
    else:
        payload, yt, st, ids = train_deep(
            name, cfg, seed, device,
            max_epochs=args.max_epochs, limit_batches=args.limit_batches,
            batch_size=args.batch_size, workers=args.workers)
    out = Path(cfg["output_dir"]) / f"{name}_seed{seed}"
    write_results(out, payload, yt, st, ids)
    t = payload["test"]
    print(f"  -> AUPRC {t['auprc']:.4f} | AUROC {t['auroc']:.4f} | "
          f"F1 {t['f1']:.4f} | balanced acc {t['balanced_accuracy']:.4f}")
    print(f"  -> written to {out}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS + ["all"], default="all")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--limit-batches", type=int,
                    help="cap batches per epoch (smoke testing)")
    ap.add_argument("--limit-samples", type=int,
                    help="cap samples for the classical path (smoke testing)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch over a handful of batches for every model")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        args.max_epochs = args.max_epochs or 1
        args.limit_batches = args.limit_batches or 4
        args.limit_samples = args.limit_samples or 256
        args.workers = 2 if args.workers is None else args.workers
        cfg["output_dir"] = "results/smoke"
        cfg["train"]["early_stop_patience"] = 10**6  # never stop in a smoke run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | config: {args.config}")
    if device.type == "cpu":
        print("WARNING: no CUDA; the deep runs will be very slow")

    names = MODELS if args.model == "all" else [args.model]
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    if args.smoke:
        seeds = [seeds[0]]

    failures = []
    for name in names:
        for seed in seeds:
            try:
                run_one(name, cfg, seed, device, args)
            except Exception as exc:
                failures.append((name, seed, repr(exc)))
                print(f"  !! FAILED {name} seed {seed}: {exc!r}", file=sys.stderr)
                if not args.smoke:
                    raise

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for n, s, e in failures:
            print(f"  {n} seed {s}: {e}")
        sys.exit(1)
    print("\nall runs completed")


if __name__ == "__main__":
    main()
