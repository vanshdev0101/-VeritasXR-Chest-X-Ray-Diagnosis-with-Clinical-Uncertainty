# VeritasXR — Chest X-Ray Inference with Clinical Uncertainty
### Dual-Path Architecture + TensorRT + Triton Inference Server | RTX 3070 Ti

A production-grade deep learning inference pipeline for pneumonia detection from chest X-rays.
VeritasXR is a custom-designed dual-pathway CNN that learns local and global features in parallel,
with a learned uncertainty head for clinical triage — deployed on NVIDIA Triton Inference Server.

> *"We propose a dual-pathway architecture where local and global features are learned in parallel,
> with inter-path disagreement used as an intrinsic uncertainty signal for clinical decision support."*

---

## Results

| Method | Latency | Throughput | Speedup |
|---|---|---|---|
| PyTorch FP32 (baseline) | 18.1ms | 55 img/s | 1x |
| TensorRT FP16 | 2.2ms | 447 img/s | 8.1x |
| Triton + ONNX (VeritasXR) | 1.4ms | 648 img/s | **13x** |

**Model accuracy on test set: 84.9%**

---

## What Makes VeritasXR Novel

Standard approaches fine-tune a pretrained ResNet50 — one path, one output, no uncertainty.
VeritasXR was designed from scratch with clinical deployment in mind:

| Design Decision | Why |
|---|---|
| **Dual-path architecture** | Small kernels (3×3) capture fine lesion details. Large kernels (7×7) capture whole-lung structure. Both run in parallel. |
| **Learned path weights** | A learnable `path_weights` parameter lets the model decide how much to trust local vs global features per input. |
| **Squeeze-Excitation blocks** | Channel attention in each path — model suppresses irrelevant feature maps automatically. |
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

## Project Structure

```
trt_speedtest/
├── api/
│   └── app.py                      ← FastAPI backend → Triton client
├── frontend/
│   └── index.html                  ← Demo web UI
├── models/
│   ├── veritasxr.pth               ← trained weights
│   └── veritasxr.onnx              ← exported ONNX model
├── triton_models/
│   └── veritasxr/
│       ├── config.pbtxt            ← Triton model config
│       └── 1/
│           └── model.onnx          ← ONNX model served by Triton
├── results/
│   ├── veritasxr_benchmark.json    ← benchmark results
│   └── veritasxr_comparison.json  ← model comparison results
├── veritasxr_model.py              ← VeritasXR architecture
├── export_onnx.py                  ← export to ONNX with correct output names
├── train_veritaxr.py               ← training script
├── benchmark.py                    ← latency + throughput benchmark
├── analysis.py                     ← EDA + metrics plots
├── compare_models.py               ← VeritasXR vs ResNet50 comparison
└── requirements.txt
```

---

## Prerequisites

- Windows 10/11 with WSL2 or Linux
- NVIDIA GPU (tested on RTX 3070 Ti Laptop)
- CUDA 12.x drivers
- Anaconda or Python 3.10+
- Docker Desktop with GPU support enabled
- Kaggle account (for dataset download)

---

## Setup

### Step 1 — Clone the repo

```powershell
git clone <your-repo-url>
cd trt_speedtest
```

### Step 2 — Install dependencies

```powershell
pip install torch torchvision onnx onnxruntime
pip install fastapi uvicorn python-multipart pillow
pip install tritonclient[http] numpy matplotlib
pip install flask-cors aiohttp
```

### Step 3 — Download the dataset

Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Unzip into:
```
trt_speedtest/data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Running the Project

### Step 4 — Train VeritasXR

```powershell
python train_veritasxr.py
```

- Trains dual-path architecture from scratch on chest X-rays
- Saves best weights to `models/veritasxr.pth`
- Expected accuracy: ~85%

### Step 5 — Export to ONNX

```powershell
python export_onnx.py
```

- Exports full model with both `verdict` and `uncertainty` outputs
- Saves to `models/veritasxr.onnx`

Verify output names:
```python
import onnx
m = onnx.load("models/veritasxr.onnx")
for o in m.graph.output:
    print(o.name)
# verdict
# uncertainty
```

### Step 6 — Set Up Triton Model Repository

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

### Step 7 — Start Triton Inference Server

Open a dedicated terminal and keep it running:

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

### Step 8 — Start the API

```powershell
uvicorn api.app:app --reload --port 8080
```

Open: **http://localhost:8080**

Upload a chest X-ray and get:
- Prediction: NORMAL or PNEUMONIA
- Confidence percentage
- Uncertainty score + triage recommendation
- Live inference latency

### Step 9 — Run Benchmarks

```powershell
python benchmark.py
```

### Step 10 — Stress Test

```powershell
python stress_test.py
```

Sends 100 concurrent requests and reports P50/P90/P99 latency and throughput.

---

## Uncertainty Triage System

VeritasXR outputs a clinical triage recommendation based on the uncertainty score:

| Uncertainty Score | Label | Action |
|---|---|---|
| < 0.4 | High Confidence | Safe to act on |
| 0.4 – 0.7 | Moderate Confidence | Consider review |
| > 0.7 | Low Confidence | Flag for radiologist review |

This means the system knows when it doesn't know — critical for safe clinical deployment.

---

## Troubleshooting

**Triton: `unexpected inference output` error**
Your ONNX was exported with wrong output names. Re-run `export_onnx.py` and verify output names are `verdict` and `uncertainty`.

**Port already allocated**
```powershell
netstat -ano | findstr :8010
taskkill /PID <PID> /F
```

**Docker not running**
Start Docker Desktop and wait for the whale icon to stop animating, then retry.

**`.gitignore` not ignoring dataset**
```bash
git rm -r --cached data/
git add .
git commit -m "remove dataset from tracking"
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | VeritasXR (custom dual-path CNN) |
| Training | PyTorch 2.x |
| Export | ONNX opset 17 |
| Serving | NVIDIA Triton Inference Server 2.54 |
| Backend | FastAPI + uvicorn |
| Frontend | HTML / CSS / Vanilla JS |
| GPU | NVIDIA RTX 3070 Ti Laptop |
| CUDA | 12.x |
| Dataset | Kaggle Chest X-Ray (Pneumonia) — 5,216 train / 624 test |

---

## Team

| Name | Roll No | Contribution |
|---|---|---|
| Vansh C | RA2311026010114 | VeritasXR architecture + Triton deployment |
| Aditya Joshi | RA2311026010129 | ONNX export + TensorRT engine build |
| Hirav Kadikar | RA2311026010111 | Benchmarking + stress testing + analysis |
| Anjany Kumar Jaiswal | RA2311026010006 | FastAPI backend + frontend demo |

---

*SEAI Project · 2026 · Accelerating Neural Network Inferencing: TensorRT & Triton Inference Server*