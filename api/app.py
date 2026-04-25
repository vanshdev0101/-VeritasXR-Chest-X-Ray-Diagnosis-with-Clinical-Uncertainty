from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tritonclient.http as httpclient
import numpy as np
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import io, time, os

app = FastAPI(title="VeritasXR — Chest X-Ray Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES    = ["NORMAL", "PNEUMONIA"]
TRITON_URL = "localhost:8010"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ────────────────────────────────────────
veritasxr_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Triton client ─────────────────────────────────────
triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

# ── ResNet model cache — load once, reuse ─────────────
resnet_cache = {}

RESNET_MODELS = {
    "resnet_l1": {"path": "models/resnet_ablation/resnet50_level1.pth", "label": "ResNet50 — FC only",       "unfreeze": 0},
    "resnet_l2": {"path": "models/resnet_ablation/resnet50_level2.pth", "label": "ResNet50 — Layer4 + FC",   "unfreeze": 1},
    "resnet_l3": {"path": "models/resnet_ablation/resnet50_level3.pth", "label": "ResNet50 — Layer3-4 + FC", "unfreeze": 2},
    "resnet_l4": {"path": "models/resnet_ablation/resnet50_level4.pth", "label": "ResNet50 — Layer2-4 + FC", "unfreeze": 3},
    "resnet_l5": {"path": "models/resnet_ablation/resnet50_level5.pth", "label": "ResNet50 — Layer1-4 + FC", "unfreeze": 4},
    "resnet_l6": {"path": "models/resnet_ablation/resnet50_level6.pth", "label": "ResNet50 — Full finetune", "unfreeze": 5},
}

RESNET_STATS = {
    "resnet_l1": {"accuracy": 84.29, "f1": 0.882, "params": "4K trainable"},
    "resnet_l2": {"accuracy": 72.92, "f1": 0.822, "params": "14.9M trainable"},
    "resnet_l3": {"accuracy": 81.25, "f1": 0.869, "params": "22M trainable"},
    "resnet_l4": {"accuracy": 86.38, "f1": 0.900, "params": "23.2M trainable"},
    "resnet_l5": {"accuracy": 87.98, "f1": 0.911, "params": "23.5M trainable"},
    "resnet_l6": {"accuracy": 90.87, "f1": 0.930, "params": "23.5M trainable"},
}

def load_resnet(model_key):
    if model_key in resnet_cache:
        return resnet_cache[model_key]

    cfg   = RESNET_MODELS[model_key]
    model = models.resnet50(weights=None)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze layers
    layers = [model.layer4, model.layer3, model.layer2, model.layer1, model.bn1, model.conv1]
    for i in range(min(cfg["unfreeze"], len(layers))):
        for p in layers[i].parameters():
            p.requires_grad = True

    model.fc = nn.Linear(2048, 2)
    model.load_state_dict(torch.load(cfg["path"], map_location=DEVICE))
    model.eval().to(DEVICE)
    resnet_cache[model_key] = model
    print(f"Loaded {cfg['label']} onto {DEVICE}")
    return model


app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.get("/models")
def list_models():
    """Return available models and their stats"""
    available = [
        {
            "id":       "veritasxr",
            "label":    "VeritasXR (Custom Dual-Path)",
            "type":     "custom",
            "accuracy": 75.6,
            "f1":       None,
            "params":   "17M",
            "latency":  2.53,
            "note":     "Custom architecture with uncertainty head"
        }
    ]
    for key, cfg in RESNET_MODELS.items():
        stats = RESNET_STATS[key]
        available.append({
            "id":       key,
            "label":    cfg["label"],
            "type":     "resnet",
            "accuracy": stats["accuracy"],
            "f1":       stats["f1"],
            "params":   stats["params"],
            "latency":  None,
            "note":     f"ResNet50 with {cfg['label'].split('—')[1].strip()} unfrozen"
        })
    return JSONResponse(available)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str    = Form(default="veritasxr")
):
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents))
    except Exception:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)

    # ── VeritasXR via Triton ───────────────────────────
    if model_id == "veritasxr":
        img_tensor = veritasxr_transform(pil_img.convert("RGB")).numpy().astype(np.float32)

        inp = httpclient.InferInput("input", [1, 1, 224, 224], "FP32")
        inp.set_data_from_numpy(img_tensor[np.newaxis])
        out_verdict     = httpclient.InferRequestedOutput("verdict")
        out_uncertainty = httpclient.InferRequestedOutput("uncertainty")

        t0 = time.perf_counter()
        result     = triton_client.infer("veritasxr", [inp], outputs=[out_verdict, out_uncertainty])
        latency_ms = (time.perf_counter() - t0) * 1000

        logits            = result.as_numpy("verdict")[0]
        probs             = np.exp(logits) / np.exp(logits).sum()
        label             = CLASSES[np.argmax(probs)]
        uncertainty_score = float(result.as_numpy("uncertainty")[0][0])

        if uncertainty_score < 0.4:
            uncertainty_label = "High Confidence"
            triage = "Safe to act on"
        elif uncertainty_score < 0.7:
            uncertainty_label = "Moderate Confidence"
            triage = "Consider review"
        else:
            uncertainty_label = "Low Confidence"
            triage = "Flag for radiologist review"

        return JSONResponse({
            "model_id":          "veritasxr",
            "model_label":       "VeritasXR (Custom Dual-Path)",
            "prediction":        label,
            "confidence":        f"{probs.max()*100:.1f}%",
            "latency_ms":        round(latency_ms, 2),
            "NORMAL":            f"{probs[0]*100:.1f}%",
            "PNEUMONIA":         f"{probs[1]*100:.1f}%",
            "uncertainty_score": round(uncertainty_score, 3),
            "uncertainty_label": uncertainty_label,
            "triage":            triage,
            "has_uncertainty":   True,
            "accuracy":          75.6,
            "f1":                None,
        })

    # ── ResNet50 variants — local PyTorch ──────────────
    elif model_id in RESNET_MODELS:
        cfg   = RESNET_MODELS[model_id]
        stats = RESNET_STATS[model_id]

        if not os.path.exists(cfg["path"]):
            return JSONResponse({"error": f"Model file not found: {cfg['path']}"}, status_code=404)

        model      = load_resnet(model_id)
        img_tensor = resnet_transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(img_tensor)
        latency_ms = (time.perf_counter() - t0) * 1000

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        label = CLASSES[np.argmax(probs)]

        return JSONResponse({
            "model_id":          model_id,
            "model_label":       cfg["label"],
            "prediction":        label,
            "confidence":        f"{probs.max()*100:.1f}%",
            "latency_ms":        round(latency_ms, 2),
            "NORMAL":            f"{probs[0]*100:.1f}%",
            "PNEUMONIA":         f"{probs[1]*100:.1f}%",
            "uncertainty_score": None,
            "uncertainty_label": "N/A",
            "triage":            "No uncertainty estimation available",
            "has_uncertainty":   False,
            "accuracy":          stats["accuracy"],
            "f1":                stats["f1"],
        })

    else:
        return JSONResponse({"error": f"Unknown model: {model_id}"}, status_code=400)