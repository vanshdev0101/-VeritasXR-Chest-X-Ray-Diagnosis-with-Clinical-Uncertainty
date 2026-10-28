import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm
import json, time, os

# ─── Config ───────────────────────────────────────────
DATA_DIR   = "data/chest_xray"
SAVE_DIR   = "models/resnet_ablation"
RESULTS    = "results/resnet_ablation.json"
EPOCHS     = 5
BATCH_SIZE = 32
LR         = 3e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

# ─── Transforms ───────────────────────────────────────
# ResNet50 expects 3 channel RGB
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── Data ─────────────────────────────────────────────
def get_loaders():
    train = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
    val   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=val_transform)
    test  = datasets.ImageFolder(f"{DATA_DIR}/test",  transform=val_transform)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train_loader, val_loader, test_loader

# ─── Build ResNet50 with N layers unfrozen ────────────
def build_resnet(unfreeze_layers=0):
    """
    unfreeze_layers:
        0 = freeze all backbone, only train FC  (Level 1)
        1 = unfreeze layer4 + FC               (Level 2)
        2 = unfreeze layer3 + layer4 + FC      (Level 3)
        3 = unfreeze layer2 + layer3 + layer4  (Level 4)
        4 = unfreeze layer1 + ... + layer4     (Level 5 — almost full)
        5 = unfreeze everything                (Level 6 — full fine-tune)
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layers from the end based on level
    layers_to_unfreeze = [model.layer4, model.layer3, model.layer2, model.layer1, model.bn1, model.conv1]
    for i in range(min(unfreeze_layers, len(layers_to_unfreeze))):
        for param in layers_to_unfreeze[i].parameters():
            param.requires_grad = True

    # Always replace and unfreeze the final FC layer
    model.fc = nn.Linear(2048, 2)
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params ({trainable/total*100:.1f}%)")

    return model.to(DEVICE)

# ─── Train one epoch ──────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch}/{total_epochs} [Train]",
                ncols=90, unit="batch", colour="green")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        pbar.set_postfix({
            "loss": f"{total_loss/len(pbar):.3f}",
            "acc":  f"{correct/total*100:.1f}%"
        })

    return correct / total

# ─── Evaluate ─────────────────────────────────────────
def evaluate(model, loader, split="Val"):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"  [{split}]", ncols=90, unit="batch", colour="blue")

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds   = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            pbar.set_postfix({"acc": f"{correct/total*100:.1f}%"})

    acc = correct / total

    # F1 score (manual — no sklearn needed)
    tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return acc, f1, precision, recall

# ─── Measure latency ──────────────────────────────────
def measure_latency(model, runs=100):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            model(dummy)
    torch.cuda.synchronize()

    return (time.perf_counter() - t0) / runs * 1000  # ms

# ─── Main ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")
    train_loader, val_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()

    experiments = [
        {"level": 1, "unfreeze": 0, "name": "FC only"},
        {"level": 2, "unfreeze": 1, "name": "Layer4 + FC"},
        {"level": 3, "unfreeze": 2, "name": "Layer3-4 + FC"},
        {"level": 4, "unfreeze": 3, "name": "Layer2-4 + FC"},
        {"level": 5, "unfreeze": 4, "name": "Layer1-4 + FC"},
        {"level": 6, "unfreeze": 5, "name": "Full fine-tune"},
    ]

    all_results = []

    # Overall progress across all experiments
    overall = tqdm(experiments, desc="Overall Progress", ncols=90,
                   unit="experiment", colour="yellow")

    for exp in overall:
        overall.set_description(f"Experiment {exp['level']}/6 — {exp['name']}")
        print(f"\n{'='*55}")
        print(f"Experiment {exp['level']}/6 — ResNet50 {exp['name']}")
        print(f"{'='*55}")

        model     = build_resnet(unfreeze_layers=exp["unfreeze"])
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        best_val_acc = 0
        model_path   = f"{SAVE_DIR}/resnet50_level{exp['level']}.pth"

        for epoch in range(1, EPOCHS + 1):
            train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch, EPOCHS)
            val_acc, val_f1, _, _ = evaluate(model, val_loader, "Val")
            scheduler.step()

            print(f"  → Train Acc: {train_acc*100:.1f}% | "
                  f"Val Acc: {val_acc*100:.1f}% | "
                  f"Val F1: {val_f1:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print(f"  ✓ Saved best model (val acc: {val_acc*100:.1f}%)")

        # Final test evaluation
        model.load_state_dict(torch.load(model_path))
        test_acc, test_f1, precision, recall = evaluate(model, test_loader, "Test")
        latency_ms = measure_latency(model)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        result = {
            "level":             exp["level"],
            "name":              f"ResNet50 — {exp['name']}",
            "test_accuracy":     round(test_acc * 100, 2),
            "test_f1":           round(test_f1, 4),
            "precision":         round(precision, 4),
            "recall":            round(recall, 4),
            "latency_ms":        round(latency_ms, 2),
            "trainable_params":  trainable_params,
            "model_path":        model_path
        }

        all_results.append(result)

        print(f"\n  ── Result ──────────────────────────")
        print(f"  Test Acc  : {test_acc*100:.2f}%")
        print(f"  Test F1   : {test_f1:.4f}")
        print(f"  Latency   : {latency_ms:.2f}ms")

    # Save all results
    with open(RESULTS, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final comparison table
    print(f"\n\n{'='*70}")
    print(f"{'Model':<30} {'Acc':>8} {'F1':>8} {'Latency':>10} {'Params':>12}")
    print(f"{'='*70}")
    for r in all_results:
        print(f"{r['name']:<30} {r['test_accuracy']:>7}% {r['test_f1']:>8.4f} "
              f"{r['latency_ms']:>9.2f}ms {r['trainable_params']:>12,}")
    print(f"{'='*70}")
    print(f"\nResults saved to {RESULTS}")

if __name__ == "__main__":
    main()
