import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from PIL import Image

os.makedirs("results/analysis", exist_ok=True)

DATA_DIR   = "data/chest_xray/chest_xray"
MODEL_PATH = "models/resnet50_xray.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

raw_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("Loading datasets...")
train_data = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
test_data  = ImageFolder(os.path.join(DATA_DIR, "test"),  transform=transform)
val_data   = ImageFolder(os.path.join(DATA_DIR, "val"),   transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False, num_workers=0)

CLASSES = train_data.classes
print(f"Classes: {CLASSES}")

# ── Load model ─────────────────────────────────────────────
print("Loading model...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cuda"))
model.eval().cuda()

# ══════════════════════════════════════════════════════════
# FIGURE 1 — EDA
# ══════════════════════════════════════════════════════════
print("\n[1/5] Generating EDA plots...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Exploratory Data Analysis — Chest X-Ray Dataset", 
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1a — class distribution
ax1 = fig.add_subplot(gs[0, 0])
splits = {"Train": train_data, "Val": val_data, "Test": test_data}
x = np.arange(len(CLASSES))
width = 0.25
colors = ["#6c63ff", "#4ecca3", "#ffa94d"]
for i, (split_name, dataset) in enumerate(splits.items()):
    counts = [sum(1 for _, l in dataset.samples if l == c) for c in range(len(CLASSES))]
    bars = ax1.bar(x + i*width, counts, width, label=split_name, 
                   color=colors[i], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha="center", va="bottom", fontsize=8, fontweight="bold")
ax1.set_xticks(x + width)
ax1.set_xticklabels(CLASSES)
ax1.set_title("Class distribution per split", fontsize=11, fontweight="bold")
ax1.set_ylabel("Number of images")
ax1.legend(fontsize=9)
ax1.spines[["top","right"]].set_visible(False)

# 1b — class imbalance pie
ax2 = fig.add_subplot(gs[0, 1])
train_counts = [sum(1 for _, l in train_data.samples if l == c) for c in range(len(CLASSES))]
pie_colors = ["#6c63ff", "#ff6b6b"]
wedges, texts, autotexts = ax2.pie(
    train_counts, labels=CLASSES, autopct="%1.1f%%",
    colors=pie_colors, startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
ax2.set_title("Training set class balance", fontsize=11, fontweight="bold")

# 1c — image size distribution
ax3 = fig.add_subplot(gs[0, 2])
widths, heights = [], []
sample_paths = [s[0] for s in train_data.samples[:200]]
for p in sample_paths:
    try:
        img = Image.open(p)
        widths.append(img.size[0])
        heights.append(img.size[1])
    except:
        pass
ax3.scatter(widths, heights, alpha=0.4, color="#6c63ff", s=15, edgecolors="none")
ax3.set_xlabel("Width (px)")
ax3.set_ylabel("Height (px)")
ax3.set_title("Original image dimensions (200 samples)", fontsize=11, fontweight="bold")
ax3.spines[["top","right"]].set_visible(False)
ax3.axvline(np.median(widths), color="#ffa94d", linestyle="--", linewidth=1, label=f"Median W: {int(np.median(widths))}px")
ax3.axhline(np.median(heights), color="#4ecca3", linestyle="--", linewidth=1, label=f"Median H: {int(np.median(heights))}px")
ax3.legend(fontsize=8)

# 1d — sample images grid
ax4 = fig.add_subplot(gs[1, :])
ax4.axis("off")
ax4.set_title("Sample images from each class (after 224×224 resize)", 
              fontsize=11, fontweight="bold", pad=10)

n_samples = 8
samples_per_class = {c: [] for c in range(len(CLASSES))}
for path, label in train_data.samples:
    if len(samples_per_class[label]) < n_samples:
        samples_per_class[label].append(path)
    if all(len(v) >= n_samples for v in samples_per_class.values()):
        break

inner_gs = gridspec.GridSpecFromSubplotSpec(
    len(CLASSES), n_samples, subplot_spec=gs[1, :], 
    hspace=0.05, wspace=0.05
)

for row, cls_idx in enumerate(range(len(CLASSES))):
    for col, path in enumerate(samples_per_class[cls_idx]):
        ax = fig.add_subplot(inner_gs[row, col])
        img = Image.open(path).convert("RGB").resize((224, 224))
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(CLASSES[cls_idx], fontsize=9, 
                         rotation=90, labelpad=2, fontweight="bold")

plt.savefig("results/analysis/1_eda.png", dpi=150, bbox_inches="tight", 
            facecolor="#0a0a0f", edgecolor="none")
plt.close()
print("  Saved: results/analysis/1_eda.png")

# ══════════════════════════════════════════════════════════
# FIGURE 2 — Model evaluation metrics
# ══════════════════════════════════════════════════════════
print("[2/5] Running model evaluation...")

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.cuda()
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1])

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

tp = np.sum((all_preds == 1) & (all_labels == 1))
tn = np.sum((all_preds == 0) & (all_labels == 0))
fp = np.sum((all_preds == 1) & (all_labels == 0))
fn = np.sum((all_preds == 0) & (all_labels == 1))

accuracy  = (tp + tn) / len(all_labels) * 100
precision = tp / (tp + fp) * 100 if (tp+fp) > 0 else 0
recall    = tp / (tp + fn) * 100 if (tp+fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0
specificity = tn / (tn + fp) * 100 if (tn+fp) > 0 else 0

print(f"\n  Accuracy:    {accuracy:.1f}%")
print(f"  Precision:   {precision:.1f}%")
print(f"  Recall:      {recall:.1f}%")
print(f"  F1 Score:    {f1:.1f}%")
print(f"  Specificity: {specificity:.1f}%")
print(f"\n  TP={tp} TN={tn} FP={fp} FN={fn}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Performance Analysis", fontsize=15, fontweight="bold")
fig.patch.set_facecolor("#0a0a0f")
for ax in axes:
    ax.set_facecolor("#111118")

# 2a — confusion matrix
cm = np.array([[tn, fp], [fn, tp]])
im = axes[0].imshow(cm, cmap="Blues", aspect="auto")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["Pred Normal", "Pred Pneumonia"], color="white")
axes[0].set_yticklabels(["Actual Normal", "Actual Pneumonia"], color="white")
axes[0].set_title("Confusion Matrix", fontsize=12, fontweight="bold", color="white")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    fontsize=22, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else "#111118")
plt.colorbar(im, ax=axes[0])

# 2b — metrics bar
metrics = {"Accuracy": accuracy, "Precision": precision, 
           "Recall": recall, "F1 Score": f1, "Specificity": specificity}
bar_colors = ["#6c63ff", "#4ecca3", "#ff6b6b", "#ffa94d", "#74b9ff"]
bars = axes[1].barh(list(metrics.keys()), list(metrics.values()),
                    color=bar_colors, edgecolor="none", height=0.5)
for bar, val in zip(bars, metrics.values()):
    axes[1].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10, 
                fontweight="bold", color="white")
axes[1].set_xlim(0, 110)
axes[1].set_title("Performance Metrics", fontsize=12, fontweight="bold", color="white")
axes[1].tick_params(colors="white")
axes[1].spines[["top","right","bottom","left"]].set_visible(False)
axes[1].set_xlabel("Score (%)", color="white")

# 2c — confidence distribution
normal_probs  = all_probs[all_labels == 0]
pneum_probs   = all_probs[all_labels == 1]
axes[2].hist(normal_probs,  bins=30, alpha=0.7, color="#6c63ff", 
             label="Normal", edgecolor="none")
axes[2].hist(pneum_probs,   bins=30, alpha=0.7, color="#ff6b6b", 
             label="Pneumonia", edgecolor="none")
axes[2].axvline(0.5, color="white", linestyle="--", linewidth=1.5, label="Decision boundary")
axes[2].set_xlabel("Predicted probability of Pneumonia", color="white")
axes[2].set_ylabel("Count", color="white")
axes[2].set_title("Confidence Distribution", fontsize=12, fontweight="bold", color="white")
axes[2].legend(fontsize=9, facecolor="#1a1a24", labelcolor="white")
axes[2].tick_params(colors="white")
axes[2].spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("results/analysis/2_model_metrics.png", dpi=150, bbox_inches="tight",
            facecolor="#0a0a0f")
plt.close()
print("  Saved: results/analysis/2_model_metrics.png")

# ══════════════════════════════════════════════════════════
# FIGURE 3 — ROC curve + Precision-Recall
# ══════════════════════════════════════════════════════════
print("[3/5] Generating ROC and PR curves...")

thresholds = np.linspace(0, 1, 200)
tprs, fprs, precs, recs = [], [], [], []

for thresh in thresholds:
    preds_t = (all_probs >= thresh).astype(int)
    tp_t = np.sum((preds_t == 1) & (all_labels == 1))
    tn_t = np.sum((preds_t == 0) & (all_labels == 0))
    fp_t = np.sum((preds_t == 1) & (all_labels == 0))
    fn_t = np.sum((preds_t == 0) & (all_labels == 1))
    tprs.append(tp_t / (tp_t + fn_t) if (tp_t+fn_t) > 0 else 0)
    fprs.append(fp_t / (fp_t + tn_t) if (fp_t+tn_t) > 0 else 0)
    precs.append(tp_t / (tp_t + fp_t) if (tp_t+fp_t) > 0 else 1)
    recs.append(tp_t / (tp_t + fn_t) if (tp_t+fn_t) > 0 else 0)

auc = np.trapz(tprs[::-1], fprs[::-1])
ap  = np.trapz(precs[::-1], recs[::-1])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("ROC Curve & Precision-Recall Curve", fontsize=14, fontweight="bold")
fig.patch.set_facecolor("#0a0a0f")
for ax in axes:
    ax.set_facecolor("#111118")

axes[0].plot(fprs, tprs, color="#6c63ff", linewidth=2.5, label=f"ResNet-50 (AUC = {auc:.3f})")
axes[0].plot([0,1],[0,1], color="#444", linestyle="--", linewidth=1, label="Random classifier")
axes[0].fill_between(fprs, tprs, alpha=0.1, color="#6c63ff")
axes[0].set_xlabel("False Positive Rate", color="white")
axes[0].set_ylabel("True Positive Rate", color="white")
axes[0].set_title("ROC Curve", fontsize=12, fontweight="bold", color="white")
axes[0].legend(fontsize=10, facecolor="#1a1a24", labelcolor="white")
axes[0].tick_params(colors="white")
axes[0].spines[["top","right"]].set_color("#333")

axes[1].plot(recs, precs, color="#4ecca3", linewidth=2.5, label=f"ResNet-50 (AP = {ap:.3f})")
axes[1].fill_between(recs, precs, alpha=0.1, color="#4ecca3")
base = sum(all_labels) / len(all_labels)
axes[1].axhline(base, color="#444", linestyle="--", linewidth=1, label=f"Baseline ({base:.2f})")
axes[1].set_xlabel("Recall", color="white")
axes[1].set_ylabel("Precision", color="white")
axes[1].set_title("Precision-Recall Curve", fontsize=12, fontweight="bold", color="white")
axes[1].legend(fontsize=10, facecolor="#1a1a24", labelcolor="white")
axes[1].tick_params(colors="white")
axes[1].spines[["top","right"]].set_color("#333")

plt.tight_layout()
plt.savefig("results/analysis/3_roc_pr.png", dpi=150, bbox_inches="tight",
            facecolor="#0a0a0f")
plt.close()
print(f"  AUC: {auc:.3f}  |  AP: {ap:.3f}")
print("  Saved: results/analysis/3_roc_pr.png")

# ══════════════════════════════════════════════════════════
# FIGURE 4 — Pixel intensity / brightness EDA
# ══════════════════════════════════════════════════════════
print("[4/5] Analyzing pixel intensities...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Pixel Intensity Analysis", fontsize=14, fontweight="bold")
fig.patch.set_facecolor("#0a0a0f")
for ax in axes:
    ax.set_facecolor("#111118")

class_pixels = {0: [], 1: []}
for path, label in train_data.samples[:400]:
    try:
        img = np.array(Image.open(path).convert("L").resize((224, 224)), dtype=np.float32) / 255.0
        class_pixels[label].extend(img.flatten().tolist())
    except:
        pass

cls_colors = ["#6c63ff", "#ff6b6b"]
for cls_idx in range(len(CLASSES)):
    pixels = np.array(class_pixels[cls_idx])
    axes[0].hist(pixels, bins=60, alpha=0.65, color=cls_colors[cls_idx],
                label=f"{CLASSES[cls_idx]} (μ={pixels.mean():.3f})", edgecolor="none")
axes[0].set_xlabel("Pixel intensity (normalised)", color="white")
axes[0].set_ylabel("Frequency", color="white")
axes[0].set_title("Pixel intensity distribution", fontsize=11, fontweight="bold", color="white")
axes[0].legend(fontsize=9, facecolor="#1a1a24", labelcolor="white")
axes[0].tick_params(colors="white")
axes[0].spines[["top","right"]].set_visible(False)

means_n = [np.mean(np.array(Image.open(p).convert("L").resize((224,224)), dtype=np.float32)/255.0)
           for p, l in train_data.samples[:300] if l == 0]
means_p = [np.mean(np.array(Image.open(p).convert("L").resize((224,224)), dtype=np.float32)/255.0)
           for p, l in train_data.samples[:300] if l == 1]
bp = axes[1].boxplot([means_n, means_p], labels=CLASSES, patch_artist=True,
                     medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], cls_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_ylabel("Mean brightness per image", color="white")
axes[1].set_title("Brightness by class", fontsize=11, fontweight="bold", color="white")
axes[1].tick_params(colors="white")
axes[1].spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("results/analysis/4_pixel_analysis.png", dpi=150, bbox_inches="tight",
            facecolor="#0a0a0f")
plt.close()
print("  Saved: results/analysis/4_pixel_analysis.png")

# ══════════════════════════════════════════════════════════
# FIGURE 5 — Inference speed full comparison
# ══════════════════════════════════════════════════════════
print("[5/5] Generating final speed comparison...")

import json
with open("results/benchmark_results.json") as f:
    bench = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Inference Acceleration: Full Pipeline Comparison\nResNet-50 · Chest X-Ray · RTX 3070",
             fontsize=13, fontweight="bold")
fig.patch.set_facecolor("#0a0a0f")
for ax in axes:
    ax.set_facecolor("#111118")

labels     = ["PyTorch\n(FP32)", "TensorRT\n(FP16)", "Triton +\nTensorRT"]
latencies  = [bench["pytorch"]["latency_ms"], bench["tensorrt"]["latency_ms"], bench["triton"]["latency_ms"]]
throughputs= [bench["pytorch"]["throughput"],  bench["tensorrt"]["throughput"],  bench["triton"]["throughput"]]
bar_colors = ["#6c63ff", "#4ecca3", "#ffa94d"]
speedups   = [1.0, bench["pytorch"]["latency_ms"]/bench["tensorrt"]["latency_ms"],
              bench["pytorch"]["latency_ms"]/bench["triton"]["latency_ms"]]

bars = axes[0].bar(labels, latencies, color=bar_colors, width=0.45, edgecolor="none")
for bar, val, sp in zip(bars, latencies, speedups):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"{val:.1f}ms", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="white")
    if sp > 1:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                    f"{sp:.1f}x", ha="center", va="center", fontsize=11,
                    fontweight="bold", color="#0a0a0f")
axes[0].set_title("Latency (ms) — lower is better", fontsize=11, fontweight="bold", color="white")
axes[0].set_ylabel("Milliseconds", color="white")
axes[0].tick_params(colors="white")
axes[0].spines[["top","right"]].set_visible(False)
axes[0].set_ylim(0, max(latencies)*1.3)

bars2 = axes[1].bar(labels, throughputs, color=bar_colors, width=0.45, edgecolor="none")
for bar, val in zip(bars2, throughputs):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="white")
axes[1].set_title("Throughput (img/s) — higher is better", fontsize=11, fontweight="bold", color="white")
axes[1].set_ylabel("Images per second", color="white")
axes[1].tick_params(colors="white")
axes[1].spines[["top","right"]].set_visible(False)
axes[1].set_ylim(0, max(throughputs)*1.3)

plt.tight_layout()
plt.savefig("results/analysis/5_speed_comparison.png", dpi=150, bbox_inches="tight",
            facecolor="#0a0a0f")
plt.close()
print("  Saved: results/analysis/5_speed_comparison.png")

# ── Print summary ──────────────────────────────────────────
print(f"""
{'='*45}
ANALYSIS COMPLETE
{'='*45}
Model Performance:
  Accuracy:    {accuracy:.1f}%
  Precision:   {precision:.1f}%
  Recall:      {recall:.1f}%
  F1 Score:    {f1:.1f}%
  AUC-ROC:     {auc:.3f}

Inference Speed:
  PyTorch:     {bench['pytorch']['latency_ms']}ms  ({bench['pytorch']['throughput']} img/s)
  TensorRT:    {bench['tensorrt']['latency_ms']}ms  ({bench['tensorrt']['throughput']} img/s)
  Triton+TRT:  {bench['triton']['latency_ms']}ms  ({bench['triton']['throughput']} img/s)
  Total speedup: {bench['pytorch']['latency_ms']/bench['triton']['latency_ms']:.1f}x

Saved to: results/analysis/
{'='*45}
""")