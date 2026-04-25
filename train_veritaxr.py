import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from veritasxr_model import get_model
from tqdm import tqdm
import time

DATA_DIR   = "data/chest_xray"
MODEL_PATH = "models/veritasxr.pth"
EPOCHS     = 20
BATCH_SIZE = 32
LR         = 1e-4        # lower than before
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_loaders():
    train = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
    val   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=val_transform)
    test  = datasets.ImageFolder(f"{DATA_DIR}/test",  transform=val_transform)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
        DataLoader(val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        DataLoader(test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )

def evaluate(model, loader, label="Val"):
    model.eval()
    correct, total = 0, 0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            verdicts, _ = model(imgs)
            preds = verdicts.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            tp += ((preds==1) & (labels==1)).sum().item()
            fp += ((preds==1) & (labels==0)).sum().item()
            fn += ((preds==0) & (labels==1)).sum().item()
            tn += ((preds==0) & (labels==0)).sum().item()

    acc  = correct / total
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    spec = tn / (tn + fp + 1e-8)   # specificity — how often NORMAL is correct
    print(f"{label} | Acc: {acc*100:.1f}% | F1: {f1:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | Specificity: {spec:.3f}")
    return acc, f1

def main():
    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader = get_loaders()

    model = get_model().to(DEVICE)
    
    # Class weights — normal gets 2.9x more weight to fix bias
    class_weights = torch.tensor([2.9, 1.0]).to(DEVICE)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}", ncols=90, colour="green")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            verdicts, _ = model(imgs)
            loss = criterion(verdicts, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (verdicts.argmax(1) == labels).sum().item()
            total      += labels.size(0)
            pbar.set_postfix({"loss": f"{total_loss/len(pbar):.3f}", "acc": f"{correct/total*100:.1f}%"})

        val_acc, val_f1 = evaluate(model, val_loader, "Val")
        scheduler.step()

        # Save on best F1 not just accuracy — F1 is fairer with imbalanced data
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved (F1: {val_f1:.3f})")

    print("\n── Final Test ──")
    model.load_state_dict(torch.load(MODEL_PATH))
    w = torch.softmax(model.path_weights, dim=0)
    print(f"Learned path weights → Local: {w[0].item():.3f} | Global: {w[1].item():.3f}")
    evaluate(model, test_loader, "Test")

if __name__ == "__main__":
    main()