from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tritonclient.http as httpclient
import numpy as np
from torchvision import transforms
from PIL import Image
import io, time

app = FastAPI(title="VeritasXR — Chest X-Ray Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES    = ["NORMAL", "PNEUMONIA"]
TRITON_URL = "localhost:8010"       # ← fixed port
MODEL_NAME = "veritasxr"           # ← fixed model name

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Single Triton client — reused across requests
triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    img_tensor = transform(img).numpy().astype(np.float32)

    # Both verdict and uncertainty now come from Triton — no local PyTorch
    inp = httpclient.InferInput("input", [1, 1, 224, 224], "FP32")
    inp.set_data_from_numpy(img_tensor[np.newaxis])

    out_verdict     = httpclient.InferRequestedOutput("verdict")
    out_uncertainty = httpclient.InferRequestedOutput("uncertainty")

    t0 = time.perf_counter()
    result = triton_client.infer(MODEL_NAME, [inp], outputs=[out_verdict, out_uncertainty])
    latency_ms = (time.perf_counter() - t0) * 1000

    logits = result.as_numpy("verdict")[0]
    probs  = np.exp(logits) / np.exp(logits).sum()
    label  = CLASSES[np.argmax(probs)]

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
        "prediction":        label,
        "confidence":        f"{probs.max()*100:.1f}%",
        "latency_ms":        round(latency_ms, 2),
        "NORMAL":            f"{probs[0]*100:.1f}%",
        "PNEUMONIA":         f"{probs[1]*100:.1f}%",
        "uncertainty_score": round(uncertainty_score, 3),
        "uncertainty_label": uncertainty_label,
        "triage":            triage
    })