# VeritasXR — Chest X-Ray Diagnosis with Clinical Uncertainty

A production-grade deep learning inference pipeline for pneumonia detection from chest X-rays. VeritasXR is a custom dual-pathway CNN that learns local and global features in parallel, with a learned uncertainty head for clinical triage — benchmarked against ResNet50 at six fine-tuning depths, deployed on NVIDIA Triton Inference Server.

> *"We propose a dual-pathway architecture where local and global features are learned in parallel, with inter-path disagreement used as an intrinsic uncertainty signal for clinical decision support."*

---

## Results

| Method | Latency | Throughput | Speedup |
|---|---|---|---|
| PyTorch FP32 (baseline) | 18.1ms | 55 img/s | 1× |
| TensorRT FP16 | 2.2ms | 447 img/s | 8.1× |
| Triton + ONNX (VeritasXR) | 1.4ms | 648 img/s | **13×** |

---

## What Makes VeritasXR Novel

Standard approaches fine-tune a pretrained ResNet50 — one path, one output, no uncertainty. VeritasXR was designed from scratch with clinical deployment in mind.

| Design Decision | Why |
|---|---|
| **Dual-path architecture** | Small kernels (3×3) capture fine lesion details. Large kernels (7×7) capture whole-lung structure. Both run in parallel. |
| **Learned path weights** | A learnable `path_weights` parameter — the model decides how much to trust local vs global features per input. |
| **Squeeze-Excitation blocks** | Channel attention in each path — suppresses irrelevant feature maps automatically. |
| **Uncertainty head** | A separate sigmoid output estimates prediction confidence. High uncertainty cases are flagged for radiologist review. |
| **Triton deployment** | Both verdict and uncertainty served from GPU in a single inference call — no local PyTorch at runtime. |

---

## Architecture

```
Input (1×224×224 grayscale X-ray)
        │
        ▼
    Stem (Conv 11×11 → Conv 3×3 → MaxPool)
        │
   ┌────┴────┐
   │         │
Local Path  Global Path
(kernel=3)  (kernel=7)
DualPathBlock  DualPathBlock
DualPathBlock  DualPathBlock
SqueezeExcite  SqueezeExcite
AdaptivePool   AdaptivePool
   │         │
   └────┬────┘
        │
   Learned Weighted Merge (softmax weights)
        │
   ┌────┴────┐
   │         │
Classifier  Uncertainty
  Head        Head
   │         │
verdict   uncertainty
(2 classes) (0–1 score)
```

---

## ResNet50 Ablation Study

VeritasXR is benchmarked against ResNet50 trained at six progressive fine-tuning depths — from FC layer only up to full fine-tune. This validates the design choices made in VeritasXR against a standard transfer learning baseline.

| Level | Unfrozen Layers | Test Accuracy | F1 | Trainable Params |
|---|---|---|---|---|
| L1 | FC only | 84.29% | 0.882 | 4K |
| L2 | Layer4 + FC | 72.92% | 0.822 | 14.9M |
| L3 | Layer3–4 + FC | 81.25% | 0.869 | 22M |
| L4 | Layer2–4 + FC | 86.38% | 0.900 | 23.2M |
| L5 | Layer1–4 + FC | 87.98% | 0.911 | 23.5M |
| L6 | Full fine-tune | 90.87% | 0.930 | 23.5M |
| **VeritasXR** | Custom (scratch) | 75.6% | — | **17M** |

VeritasXR trades raw accuracy for clinical utility — it is the only model with an uncertainty output, runs faster than all ResNet variants, and uses fewer parameters than any fully fine-tuned ResNet.

---

## Project Structure

```
trt_speedtest/
├── api/
│   └── app.py                       ← FastAPI backend → Triton + ResNet inference
├── frontend/
│   └── index.html                   ← Demo UI with model selector
├── models/
│   ├── veritasxr.pth                ← VeritasXR trained weights
│   ├── veritasxr.onnx               ← Exported ONNX model
│   └── resnet_ablation/
│       ├── resnet50_level1.pth      ← FC only
│       ├── resnet50_level2.pth      ← Layer4 + FC
│       ├── resnet50_level3.pth      ← Layer3-4 + FC
│       ├── resnet50_level4.pth      ← Layer2-4 + FC
│       ├── resnet50_level5.pth      ← Layer1-4 + FC
│       └── resnet50_level6.pth      ← Full fine-tune
├── triton_models/
│   └── veritasxr/
│       ├── config.pbtxt             ← Triton model config
│       └── 1/
│           └── model.onnx           ← ONNX model served by Triton
├── results/
│   ├── veritasxr_benchmark.json     ← Latency and throughput results
│   ├── veritasxr_comparison.json   ← Model comparison results
│   └── resnet_ablation.json        ← ResNet ablation results
├── veritasxr_model.py               ← VeritasXR architecture
├── train_veritaxr.py                ← VeritasXR training script
├── train_resnet_ablation.py         ← ResNet50 ablation training (6 levels)
├── export_onnx.py                   ← Export to ONNX with correct output names
├── benchmark.py                     ← Latency + throughput benchmark
├── stress_test.py                   ← Concurrent load testing
├── analysis.py                      ← EDA + metrics plots
├── compare_models.py                ← VeritasXR vs ResNet50 comparison
└── requirements.txt
```

---

## Prerequisites

- Windows 10/11 with WSL2 or Linux
- NVIDIA GPU with CUDA support
- CUDA 12.x drivers
- Anaconda or Python 3.10+
- Docker Desktop with GPU support enabled
- Kaggle account (for dataset download)

---

## Setup

### Step 1 — Clone the repo

```bash
git clone https://github.com/VanshRajput-dev/Veritas-XR-.git
cd Veritas-XR-
```

### Step 2 — Install dependencies

```bash
pip install torch torchvision onnx onnxruntime
pip install fastapi uvicorn python-multipart pillow
pip install tritonclient[http] numpy matplotlib tqdm
pip install aiohttp
```

### Step 3 — Download the dataset

Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Unzip into:
```
data/chest_xray/chest_xray/
├── train/  (NORMAL / PNEUMONIA)
├── val/    (NORMAL / PNEUMONIA)
└── test/   (NORMAL / PNEUMONIA)
```

---

## Running the Project

### Step 4 — Train VeritasXR

```bash
python train_veritaxr.py
```

Trains dual-path architecture from scratch with class-weighted loss to handle dataset imbalance (2.9× more pneumonia than normal). Saves best weights to `models/veritasxr.pth`.

### Step 5 — Train ResNet50 Ablation

```bash
python train_resnet_ablation.py
```

Trains 6 ResNet50 variants with progressive layer unfreezing. Saves each model to `models/resnet_ablation/` and results to `results/resnet_ablation.json`. Includes tqdm progress bars.

### Step 6 — Export VeritasXR to ONNX

```bash
python export_onnx.py
```

Verify output names:
```python
import onnx
m = onnx.load("models/veritasxr.onnx")
for o in m.graph.output:
    print(o.name)
# verdict
# uncertainty
```

### Step 7 — Set Up Triton Model Repository

```powershell
New-Item -ItemType Directory -Path "triton_models\veritasxr\1" -Force
copy "models\veritasxr.onnx" "triton_models\veritasxr\1\model.onnx"
```

`triton_models/veritasxr/config.pbtxt`:
```protobuf
name: "veritasxr"
backend: "onnxruntime"
max_batch_size: 8

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 1, 224, 224 ]
  }
]

output [
  {
    name: "verdict"
    data_type: TYPE_FP32
    dims: [ 2 ]
  },
  {
    name: "uncertainty"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 1, 4, 8 ]
}
```

### Step 8 — Start Triton Inference Server

```powershell
docker run --gpus all --rm `
  -p 8010:8000 -p 8011:8001 -p 8012:8002 `
  -v C:\Users\<you>\Documents\trt_speedtest\triton_models:/models `
  nvcr.io/nvidia/tritonserver:25.01-py3 `
  tritonserver --model-repository=/models
```

Wait for:
```
| veritasxr | 1 | READY |
```

### Step 9 — Start the API

```bash
uvicorn api.app:app --reload --port 8080
```

Open **http://localhost:8080**

### Step 10 — Run Benchmarks

```bash
python benchmark.py
```

### Step 11 — Stress Test

```bash
python stress_test.py
```

Sends 100 concurrent requests and reports P50 / P90 / P99 latency and throughput.

---

## Demo UI

The frontend includes a model selector to switch between all 7 models — VeritasXR and all 6 ResNet50 fine-tuning levels — and run live inference on any uploaded chest X-ray. Each inference returns prediction, confidence, latency, and for VeritasXR, an uncertainty score with clinical triage recommendation.

---

## Uncertainty Triage System

| Uncertainty Score | Label | Action |
|---|---|---|
| < 0.4 | High Confidence | Safe to act on |
| 0.4 – 0.7 | Moderate Confidence | Consider review |
| > 0.7 | Low Confidence | Flag for radiologist review |

---

## Troubleshooting

**Triton: `unexpected inference output` error**
Re-run `export_onnx.py` and verify ONNX output names are `verdict` and `uncertainty`.

**Port already allocated**
```powershell
netstat -ano | findstr :8010
taskkill /PID <PID> /F
```

**Docker not running**
Start Docker Desktop, wait for the whale icon to stop animating, then retry.

**Dataset being tracked by Git**
```bash
git rm -r --cached data/
git add .
git commit -m "remove dataset from tracking"
```

**Model predicting pneumonia for everything**
Dataset has 2.9× more pneumonia samples. Retrain with class weights or apply a threshold:
```python
# Quick fix in app.py
label = "PNEUMONIA" if probs[1] > 0.62 else "NORMAL"
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Custom Model | VeritasXR (dual-path CNN, built from scratch) |
| Baseline | ResNet50 (6 fine-tuning levels) |
| Training | PyTorch 2.x |
| Export | ONNX opset 17 |
| Serving | NVIDIA Triton Inference Server 2.54 |
| Backend | FastAPI + uvicorn |
| Frontend | HTML / CSS / Vanilla JS |
| Dataset | Kaggle Chest X-Ray (Pneumonia) — 5,216 train / 624 test |

---