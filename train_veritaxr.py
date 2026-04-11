import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from veritasxr_model import get_model
import time

# ─── Config ───────────────────────────────────────────
DATA_DIR   = "data/chest_xray/chest_xray"
MODEL_PATH = "models/veritasxr.pth"
EPOCHS     = 10
BATCH_SIZE = 32
LR         = 3e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Transforms ───────────────────────────────────────
# Grayscale this time — our model takes 1 channel
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
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

# ─── Train one epoch ──────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        verdicts, uncertainty = model(imgs)
        loss = criterion(verdicts, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (verdicts.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{len(loader)} | "
                  f"Loss: {total_loss/(step+1):.3f} | "
                  f"Acc: {correct/total*100:.1f}%")

    return correct / total

# ─── Evaluate ─────────────────────────────────────────
def evaluate(model, loader, label="Val"):
    model.eval()
    correct, total = 0, 0
    avg_uncertainty = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            verdicts, uncertainty = model(imgs)
            correct += (verdicts.argmax(1) == labels).sum().item()
            total   += labels.size(0)
            avg_uncertainty += uncertainty.mean().item()

    acc = correct / total
    avg_uncertainty /= len(loader)
    print(f"{label} Acc: {acc*100:.1f}% | Avg Uncertainty: {avg_uncertainty:.3f}")
    return acc

# ─── Main ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Training VeritasXR from scratch...\n")

    train_loader, val_loader, test_loader = get_loaders()

    model     = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Print parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"VeritasXR Parameters: {params:,}\n")

    best_acc = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        t0 = time.time()

        train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_acc   = evaluate(model, val_loader, "Val")
        scheduler.step()

        print(f"Epoch {epoch} done in {time.time()-t0:.0f}s | "
              f"Train Acc: {train_acc*100:.1f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best model (val acc: {val_acc*100:.1f}%)")

    # Final test evaluation
    print("\n─── Final Test Evaluation ───")
    model.load_state_dict(torch.load(MODEL_PATH))

    # Print learned path weights — this is your novelty output
    w = torch.softmax(model.path_weights, dim=0)
    print(f"\nLearned Path Weights:")
    print(f"  Local  (patches): {w[0].item():.3f}")
    print(f"  Global (lung):    {w[1].item():.3f}")

    evaluate(model, test_loader, "Test")

if __name__ == "__main__":
    main()