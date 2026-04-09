# Chest X-Ray Inference Accelerator
### TensorRT + Triton Inference Server | ResNet-50 | RTX 3070

A complete deep learning inference pipeline that demonstrates **13x speedup** over standard PyTorch
by combining NVIDIA TensorRT optimization with Triton Inference Server deployment.
Built on real medical data — the Kaggle Chest X-Ray (Pneumonia) dataset.

---

## Results

| Method | Latency | Throughput | Speedup |
|---|---|---|---|
| PyTorch FP32 (baseline) | 18.1ms | 55 img/s | 1x |
| TensorRT FP16 | 2.2ms | 447 img/s | 8.1x |
| Triton + TensorRT | 1.4ms | 648 img/s | **13x** |

**Model accuracy on test set: 84.9%**

---

## Project Structure

```
trt_speedtest/
├── data/
│   └── chest_xray/
│       └── chest_xray/
│           ├── train/   (NORMAL / PNEUMONIA)
│           ├── val/     (NORMAL / PNEUMONIA)
│           └── test/    (NORMAL / PNEUMONIA)
├── models/
│   ├── resnet50_xray.pth       ← fine-tuned weights
│   ├── resnet50.onnx           ← exported ONNX model
│   └── resnet50_fp16.trt       ← TensorRT engine (Windows, not used by Triton)
├── triton_models/
│   └── xray_classifier/
│       ├── config.pbtxt        ← Triton model config
│       └── 1/
│           └── model.plan      ← TensorRT engine built inside Linux container
├── results/
│   ├── benchmark_results.json  ← raw benchmark numbers
│   ├── charts.png              ← 2-way comparison chart
│   ├── final_chart.png         ← 3-way comparison chart
│   └── analysis/               ← 5 EDA + metrics plots
├── api/
│   └── app.py                  ← FastAPI backend
├── frontend/
│   └── index.html              ← demo web UI
├── 1_export_onnx.py
├── 2_build_engine.py
├── 3_benchmark.py
├── 4_plot_results.py
├── 5_finetune.py
├── 6_triton_benchmark.py
├── 7_final_chart.py
├── analysis.py
├── verify.py
└── requirements.txt
```

---

## Prerequisites

- Windows 10/11 with WSL2 or Linux
- NVIDIA GPU (tested on RTX 3070 Ti)
- CUDA 12.x drivers installed
- Anaconda or Python 3.10+
- Docker Desktop with GPU support enabled
- Kaggle account (for dataset download)

---

## Step-by-Step Setup

### Step 1 — Clone / create project folder

```powershell
mkdir C:\Users\<you>\Documents\trt_speedtest
cd C:\Users\<you>\Documents\trt_speedtest
```

### Step 2 — Install Python dependencies

```powershell
pip install torch torchvision onnx matplotlib numpy
pip install tensorrt pycuda
pip install fastapi uvicorn python-multipart pillow
pip install tritonclient[http]
```

### Step 3 — Download the dataset

Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Download and unzip into:
```
trt_speedtest\data\chest_xray\chest_xray\
```

Verify the structure looks like:
```
data\chest_xray\chest_xray\train\NORMAL\
data\chest_xray\chest_xray\train\PNEUMONIA\
data\chest_xray\chest_xray\test\NORMAL\
data\chest_xray\chest_xray\test\PNEUMONIA\
data\chest_xray\chest_xray\val\NORMAL\
data\chest_xray\chest_xray\val\PNEUMONIA\
```

---

## Running the Project

### Step 4 — Fine-tune ResNet-50 on chest X-rays

```powershell
python 5_finetune.py
```

- Takes ~20 minutes on RTX 3070 Ti
- Trains for 5 epochs, freezes backbone, only trains the final layer
- Saves best model to `models/resnet50_xray.pth`
- Expected test accuracy: ~85%

### Step 5 — Export to ONNX

```powershell
python 1_export_onnx.py
```

- Loads fine-tuned weights
- Exports to `models/resnet50.onnx` with dynamic batch axis
- Takes ~10 seconds

### Step 6 — Build TensorRT engine (Windows)

```powershell
python 2_build_engine.py
```

- Builds FP16 TensorRT engine from the ONNX file
- Saves to `models/resnet50_fp16.trt`
- Takes 5–15 minutes (first time only)

### Step 7 — Run PyTorch vs TensorRT benchmark

```powershell
python 3_benchmark.py
```

- Runs 200 warmup + 200 timed inferences for both
- Saves results to `results/benchmark_results.json`
- Takes ~2 minutes

### Step 8 — Build the Linux TensorRT engine for Triton

Triton runs on Linux inside Docker so the Windows .trt engine is incompatible.
Build a Linux-compatible engine inside the container:

```powershell
docker run --gpus all --rm `
  -v C:\Users\<you>\Documents\trt_speedtest:/workspace `
  nvcr.io/nvidia/tritonserver:25.01-py3 `
  /usr/src/tensorrt/bin/trtexec `
  --onnx=/workspace/models/resnet50.onnx `
  --saveEngine=/workspace/triton_models/xray_classifier/1/model.plan `
  --fp16
```

- Takes 5–10 minutes
- Saves engine to `triton_models/xray_classifier/1/model.plan`

### Step 9 — Start Triton Inference Server

Open a **dedicated terminal** and keep it running:

```powershell
docker run --gpus all --rm `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v C:\Users\<you>\Documents\trt_speedtest\triton_models:/models `
  nvcr.io/nvidia/tritonserver:25.01-py3 `
  tritonserver --model-repository=/models
```

Wait until you see:
```
| xray_classifier | 1 | READY |
Started HTTPService at 0.0.0.0:8000
```

### Step 10 — Benchmark Triton

Open a **new terminal**:

```powershell
python 6_triton_benchmark.py
```

- Sends 200 HTTP requests to Triton
- Appends Triton results to `results/benchmark_results.json`

### Step 11 — Generate charts

```powershell
python 7_final_chart.py
```

- Generates 3-way comparison chart
- Saves to `results/final_chart.png`

### Step 12 — Run full analysis + EDA

```powershell
python analysis.py
```

Generates 5 plots in `results/analysis/`:

| File | Contents |
|---|---|
| `1_eda.png` | Class distribution, dataset balance, image sizes, sample grid |
| `2_model_metrics.png` | Confusion matrix, precision/recall/F1, confidence distribution |
| `3_roc_pr.png` | ROC curve with AUC score, Precision-Recall curve |
| `4_pixel_analysis.png` | Pixel intensity histograms, brightness per class |
| `5_speed_comparison.png` | Full 3-way latency and throughput comparison |

### Step 13 — Verify model predictions

```powershell
python verify.py
```

Prints accuracy, precision, recall, F1, and full confusion matrix on the test set.

### Step 14 — Start the demo web app

Make sure Triton is still running (Step 9), then open a new terminal:

```powershell
uvicorn api.app:app --reload --port 8080
```

Open your browser at: **http://localhost:8080**

Upload any chest X-ray and get:
- Prediction: NORMAL or PNEUMONIA
- Confidence percentage
- Live inference latency in ms
- Speed comparison bar chart

---

## Troubleshooting

**`num_workers` error on Windows**
Add `if __name__ == '__main__':` guard and set `num_workers=0` in DataLoader.

**Triton: version mismatch error**
The `.trt` engine was built with a different TensorRT version.
Use the `trtexec` command inside the Triton container (Step 8) to rebuild.

**Triton: platform tag mismatch**
Windows-built engines don't run in Linux containers.
Always build the Triton engine inside the Docker container (Step 8).

**Triton: max-batch size mismatch**
Set `max_batch_size: 1` in `triton_models/xray_classifier/config.pbtxt`
to match the engine's batch size.

**FastAPI: "Failed to fetch"**
Make sure Triton is running on port 8000 and FastAPI is on port 8080.
Check `const API = 'http://localhost:8080'` in `frontend/index.html`.

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | ResNet-50 (torchvision pretrained) |
| Training | PyTorch 2.x |
| Optimization | NVIDIA TensorRT 10.x (FP16) |
| Serving | NVIDIA Triton Inference Server 2.54 |
| Backend API | FastAPI + uvicorn |
| Frontend | HTML / CSS / Vanilla JS |
| GPU | NVIDIA RTX 3070 Ti Laptop |
| CUDA | 13.0 |
| Dataset | Kaggle Chest X-Ray (Pneumonia) — 5,216 train / 624 test |

---

## Team

| Name | Roll No | Contribution |
|---|---|---|
| Vansh C | RA2311026010114 | PyTorch baseline + ONNX export |
| Aditya Joshi | RA2311026010129 | TensorRT engine build |
| Hirav Kadikar | RA2311026010111 | Benchmarking + analysis |
| Anjany Kumar Jaiswal | RA2311026010006 | Triton deployment + frontend |

---

*SEAI Project · 2026 · Accelerating Neural Network Inferencing: TensorRT & Triton Inference Server*


Your professor asked for novelty. This is it —

"We propose a dual-pathway architecture where local and global features are learned in parallel, with inter-path disagreement used as an intrinsic uncertainty signal for clinical decision support."