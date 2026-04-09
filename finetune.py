import sys
print("Python:", sys.version)
print("Starting...")

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os, time

def main():
    DATA_DIR   = "data/chest_xray/chest_xray"
    MODEL_PATH = "models/resnet50_xray.pth"
    EPOCHS     = 5
    BATCH_SIZE = 32
    LR         = 0.001

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    train_data = ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_data   = ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_transforms)
    test_data  = ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_transforms)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Classes: {train_data.classes}")

    print("\nLoading ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / total

    print("\nStarting training...\n")
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        t0 = time.time()

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 20 == 0:
                print(f"  Epoch {epoch+1} | Step {i+1}/{len(train_loader)} "
                      f"| Loss: {running_loss/(i+1):.3f} "
                      f"| Acc: {100.*correct/total:.1f}%")

        val_acc = evaluate(val_loader)
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+1}/{EPOCHS} done in {elapsed:.0f}s "
              f"| Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved best model (val acc: {val_acc:.1f}%)")

        scheduler.step()
        print()

    print("Loading best model for final test evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH))
    test_acc = evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.1f}%")
    print(f"Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()