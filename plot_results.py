import json, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("results/benchmark_results.json") as f:
    data = json.load(f)

labels    = ["PyTorch (FP32)", "TensorRT (FP16)"]
latencies = [data["pytorch"]["latency_ms"], data["tensorrt"]["latency_ms"]]
throughputs = [data["pytorch"]["throughput"], data["tensorrt"]["throughput"]]
speedup   = data["speedup"]

colors = ["#7F77DD", "#1D9E75"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(f"ResNet-50 Inference: PyTorch vs TensorRT  |  {speedup}× speedup", 
             fontsize=14, fontweight="bold", y=1.02)

# Latency chart (lower is better)
bars = axes[0].bar(labels, latencies, color=colors, width=0.45, edgecolor="white", linewidth=0.8)
axes[0].set_title("Latency (ms) — lower is better", fontsize=11)
axes[0].set_ylabel("milliseconds")
for bar, val in zip(bars, latencies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[0].set_ylim(0, max(latencies) * 1.25)
axes[0].spines[["top","right"]].set_visible(False)

# Throughput chart (higher is better)
bars2 = axes[1].bar(labels, throughputs, color=colors, width=0.45, edgecolor="white", linewidth=0.8)
axes[1].set_title("Throughput (img/s) — higher is better", fontsize=11)
axes[1].set_ylabel("images per second")
for bar, val in zip(bars2, throughputs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[1].set_ylim(0, max(throughputs) * 1.25)
axes[1].spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("results/charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to results/charts.png")