from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tritonclient.http as httpclient
import numpy as np
from torchvision import transforms
from PIL import Image
import io, time

app = FastAPI(title="Chest X-Ray Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES    = ["NORMAL", "PNEUMONIA"]
TRITON_URL = "localhost:8000"
MODEL_NAME = "xray_classifier"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

client = httpclient.InferenceServerClient(url=TRITON_URL)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(img).numpy().astype(np.float32)

    inp = httpclient.InferInput("input", [1, 3, 224, 224], "FP32")
    inp.set_data_from_numpy(img_tensor[np.newaxis])
    out = httpclient.InferRequestedOutput("output")

    t0 = time.perf_counter()
    result = client.infer(MODEL_NAME, [inp], outputs=[out])
    latency_ms = (time.perf_counter() - t0) * 1000

    logits = result.as_numpy("output")[0]
    probs  = np.exp(logits) / np.exp(logits).sum()
    label  = CLASSES[np.argmax(probs)]

    return JSONResponse({
        "prediction": label,
        "confidence": f"{probs.max()*100:.1f}%",
        "latency_ms": round(latency_ms, 2),
        "NORMAL":     f"{probs[0]*100:.1f}%",
        "PNEUMONIA":  f"{probs[1]*100:.1f}%"
    })