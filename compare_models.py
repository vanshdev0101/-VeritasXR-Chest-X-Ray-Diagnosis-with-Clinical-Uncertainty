import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json, os

os.makedirs("results", exist_ok=True)

# ─── Data ─────────────────────────────────────────────
models = {
    "ResNet-50\nPyTorch",
    "VeritasXR\nPyTorch",
    "VeritasXR\nTensorRT"
}

latencies    = [18.1,  8.26, 2.53]
throughputs  = [55,    121,  395.8]
parameters   = [25.5,  17.0, 17.0]
accuracies   = [94.0,  75.6, 75.6]
labels       = ["ResNet-50\nPyTorch", "VeritasXR\nPyTorch", "VeritasXR\nTensorRT"]

colors = ["#e74c3c", "#3498db", "#2ecc71"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("VeritasXR vs ResNet-50: Full Comparison", fontsize=16, fontweight="bold", y=1.01)

# ─── Plot 1: Latency ──────────────────────────────────
ax = axes[0, 0]
bars = ax.bar(labels, latencies, color=colors, edgecolor="black", linewidth=0.5)
ax.set_title("Inference Latency (lower is better)", fontweight="bold")
ax.set_ylabel("Latency (ms)")
for bar, val in zip(bars, latencies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{val}ms", ha="center", va="bottom", fontweight="bold")
ax.set_ylim(0, 22)

# ─── Plot 2: Throughput ───────────────────────────────
ax = axes[0, 1]
bars = ax.bar(labels, throughputs, color=colors, edgecolor="black", linewidth=0.5)
ax.set_title("Throughput (higher is better)", fontweight="bold")
ax.set_ylabel("Images / second")
for bar, val in zip(bars, throughputs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{val}", ha="center", va="bottom", fontweight="bold")
ax.set_ylim(0, 450)

# ─── Plot 3: Parameters ───────────────────────────────
ax = axes[1, 0]
bars = ax.bar(labels, parameters, color=colors, edgecolor="black", linewidth=0.5)
ax.set_title("Model Size — Parameters (lower is better)", fontweight="bold")
ax.set_ylabel("Parameters (millions)")
for bar, val in zip(bars, parameters):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{val}M", ha="center", va="bottom", fontweight="bold")
ax.set_ylim(0, 30)

# ─── Plot 4: Accuracy ────────────────────────────────
ax = axes[1, 1]
bars = ax.bar(labels, accuracies, color=colors, edgecolor="black", linewidth=0.5)
ax.set_title("Test Accuracy", fontweight="bold")
ax.set_ylabel("Accuracy (%)")
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val}%", ha="center", va="bottom", fontweight="bold")
ax.set_ylim(60, 100)

# ─── Unique VeritasXR features box ───────────────────
fig.text(0.5, -0.02,
    "★ VeritasXR exclusive features: Uncertainty Estimation · Dual-Pathway Architecture · "
    "Learned Local/Global Weighting (0.510 / 0.490) · Grayscale-Native Input",
    ha="center", fontsize=10, style="italic",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig("results/veritasxr_comparison.png", dpi=150, bbox_inches="tight")
print("Chart saved to results/veritasxr_comparison.png")
plt.show()