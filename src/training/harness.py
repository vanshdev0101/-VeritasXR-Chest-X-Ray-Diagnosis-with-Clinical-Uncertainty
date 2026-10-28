"""Shared training and evaluation loop.

Every deep model in the comparison goes through this one function, so loss,
schedule, augmentation, stopping rule and checkpoint-selection metric are
identical by construction rather than by discipline.

Results are written in a fixed shape (see `write_results`) that both the
classical and deep paths honour, so S5 can consume them without caring how a
model was fitted. Per-sample test scores are saved alongside the metrics, which
is what lets S5 bootstrap confidence intervals and run significance tests
without retraining anything.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import RSNADataset
from src.models.registry import build_model, build_transforms, count_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(targets):
    """Inverse-frequency weights, normalised to mean 1 so the loss scale is
    comparable to the unweighted case."""
    counts = np.bincount(targets, minlength=2).astype(np.float64)
    w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def logits_of(out):
    """VeritasXR returns (verdict, uncertainty); the baselines return a tensor."""
    return out[0] if isinstance(out, (tuple, list)) else out


def metrics_from(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "accuracy": float((tp + tn) / len(y_true)),  # reported, never selected on
        "threshold": threshold,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }


def make_loaders(cfg, spec, batch_size=None, workers=None):
    root = Path(cfg["data"]["root"])
    bs = batch_size or cfg["train"]["batch_size"]
    nw = workers if workers is not None else cfg["train"]["num_workers"]
    out = {}
    for split in ("train", "val", "test"):
        is_train = split == "train"
        ds = RSNADataset(root / cfg["data"][split], root,
                         transform=build_transforms(spec, cfg, train=is_train))
        out[split] = DataLoader(
            ds, batch_size=bs, shuffle=is_train, num_workers=nw,
            pin_memory=True,
            # drop_last on train only: VeritasXR's classifier contains a
            # BatchNorm1d, which cannot compute variance over a batch of 1.
            drop_last=is_train,
            persistent_workers=nw > 0,
        )
    return out


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, limit=None):
    model.eval()
    scores, labels = [], []
    for i, (x, y) in enumerate(loader):
        if limit and i >= limit:
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            logits = logits_of(model(x))
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        scores.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def write_results(out_dir, payload, y_true=None, y_score=None, ids=None):
    """The single results contract shared by the deep and classical paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    if y_true is not None:
        np.savez_compressed(out_dir / "test_scores.npz",
                            y_true=y_true, y_score=y_score,
                            patient_id=np.array(ids if ids is not None else [],
                                                dtype=object))
    return out_dir


def train_deep(name, cfg, seed, device, max_epochs=None, limit_batches=None,
               batch_size=None, workers=None, out_root=None):
    spec = cfg["per_model"][name]
    set_seed(seed)

    loaders = make_loaders(cfg, spec, batch_size=batch_size, workers=workers)
    model = build_model(name, spec).to(device)
    total, trainable = count_params(model)

    tcfg = cfg["train"]
    lr = float(cfg.get("lr_overrides", {}).get(name, tcfg["lr"]))
    epochs = max_epochs or tcfg["max_epochs"]
    amp_dtype = getattr(torch, tcfg["amp_dtype"])

    weights = (class_weights(loaders["train"].dataset.targets).to(device)
               if tcfg["class_weighting"] == "balanced" else None)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=float(tcfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history, best, best_epoch, patience = [], -1.0, -1, 0
    ckpt = Path(out_root or tcfg.get("ckpt_dir", "models")) / f"{name}_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        bar = tqdm(loaders["train"], desc=f"{name} s{seed} ep{epoch}/{epochs}",
                   ncols=96, leave=False)
        for i, (x, y) in enumerate(bar):
            if limit_batches and i >= limit_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                loss = criterion(logits_of(model(x)), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
            # divide by samples seen so far, not the epoch total
            bar.set_postfix(loss=f"{running/max(seen,1):.4f}")
        sched.step()

        yv, sv = evaluate(model, loaders["val"], device, amp_dtype,
                          limit=limit_batches)
        m = metrics_from(yv, sv)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1),
               "lr": optimizer.param_groups[0]["lr"],
               "val_auprc": m["auprc"], "val_auroc": m["auroc"], "val_f1": m["f1"]}
        history.append(row)
        print(f"  ep{epoch:02d} loss {row['train_loss']:.4f} | "
              f"val AUPRC {m['auprc']:.4f} AUROC {m['auroc']:.4f}")

        if m["auprc"] > best:
            best, best_epoch, patience = m["auprc"], epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= tcfg["early_stop_patience"]:
                print(f"  early stop: no val AUPRC gain for {patience} epochs")
                break

    model.load_state_dict(torch.load(ckpt))
    yt, st = evaluate(model, loaders["test"], device, amp_dtype, limit=limit_batches)
    test = metrics_from(yt, st)
    ids = loaders["test"].dataset.df.patientId.tolist()[:len(yt)]

    payload = {
        "model": name, "seed": seed, "kind": "deep",
        "params_total": total, "params_trainable": trainable,
        "lr": lr, "epochs_run": len(history), "best_epoch": best_epoch,
        "best_val_auprc": best, "train_seconds": round(time.time() - t0, 1),
        "history": history, "test": test,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "config": {k: cfg[k] for k in ("data", "train", "augment")},
        "per_model": spec,
    }
    return payload, yt, st, ids
