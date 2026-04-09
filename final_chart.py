import json
import matplotlib.pyplot as plt
import numpy as np

with open("results/benchmark_results.json") as f:
    data = json.load(f)

labels = ["PyTorch\n(FP32)", "TensorRT\n(FP16)", "Triton +\nTensorRT"]
latencies = [
    data["pytorch"]["latency_ms"],
    data["tensorrt"]["latency_ms"],
    data["triton"]["latency_ms"]
]
throughputs = [
    data["pytorch"]["throughput"],
    data["tensorrt"]["throughput"],
    data["triton"]["throughput"]
]

colors = ["#7F77DD", "#1D9E75", "#D85A30"]
speedup_trt    = round(data["pytorch"]["latency_ms"] / data["tensorrt"]["latency_ms"], 1)
speedup_triton = round(data["pytorch"]["latency_ms"] / data["triton"]["latency_ms"], 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Chest X-Ray Inference: PyTorch vs TensorRT vs Triton\n"
    f"ResNet-50 | RTX 3070 | TensorRT {speedup_trt}x faster | Triton {speedup_triton}x faster",
    fontsize=13, fontweight="bold"
)

# Latency
bars = axes[0].bar(labels, latencies, color=colors, width=0.45,
                   edgecolor="white", linewidth=0.8)
axes[0].set_title("Latency (ms) — lower is better", fontsize=11)
axes[0].set_ylabel("milliseconds")
for bar, val in zip(bars, latencies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}ms", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
axes[0].set_ylim(0, max(latencies) * 1.3)
axes[0].spines[["top", "right"]].set_visible(False)

# Throughput
bars2 = axes[1].bar(labels, throughputs, color=colors, width=0.45,
                    edgecolor="white", linewidth=0.8)
axes[1].set_title("Throughput (img/s) — higher is better", fontsize=11)
axes[1].set_ylabel("images per second")
for bar, val in zip(bars2, throughputs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.0f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
axes[1].set_ylim(0, max(throughputs) * 1.3)
axes[1].spines[["top", "right"]].set_visible(False)

# Speedup annotations on latency chart
axes[0].annotate(f"{speedup_trt}x faster",
                 xy=(1, latencies[1]), xytext=(1, latencies[1] + max(latencies)*0.15),
                 ha="center", fontsize=9, color="#1D9E75", fontweight="bold")
axes[0].annotate(f"{speedup_triton}x faster",
                 xy=(2, latencies[2]), xytext=(2, latencies[2] + max(latencies)*0.15),
                 ha="center", fontsize=9, color="#D85A30", fontweight="bold")

plt.tight_layout()
plt.savefig("results/final_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to results/final_chart.png")