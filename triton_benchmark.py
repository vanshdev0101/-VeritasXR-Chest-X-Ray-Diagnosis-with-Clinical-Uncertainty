import tritonclient.http as httpclient
import numpy as np
import time, json, os
from torchvision import transforms
from PIL import Image
import glob

TRITON_URL  = "localhost:8000"
MODEL_NAME  = "xray_classifier"
RUNS        = 200
WARMUP      = 50
TEST_DIR    = "data/chest_xray/chest_xray/test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("Loading test images...")
image_paths = (glob.glob(f"{TEST_DIR}/NORMAL/*.jpeg") +
               glob.glob(f"{TEST_DIR}/PNEUMONIA/*.jpeg"))[:RUNS + WARMUP]

images = []
for p in image_paths:
    img = Image.open(p).convert("RGB")
    images.append(transform(img).numpy())

print(f"Loaded {len(images)} images")
print(f"Connecting to Triton at {TRITON_URL}...")

client = httpclient.InferenceServerClient(url=TRITON_URL)
print(f"Server ready: {client.is_server_ready()}")

def run_inference(image_np):
    inp = httpclient.InferInput("input", [1, 3, 224, 224], "FP32")
    inp.set_data_from_numpy(image_np[np.newaxis].astype(np.float32))
    out = httpclient.InferRequestedOutput("output")
    result = client.infer(MODEL_NAME, [inp], outputs=[out])
    return result.as_numpy("output")

print("Warming up...")
for i in range(WARMUP):
    run_inference(images[i % len(images)])

print(f"Benchmarking {RUNS} requests...")
times = []
for i in range(RUNS):
    t0 = time.perf_counter()
    run_inference(images[i % len(images)])
    times.append((time.perf_counter() - t0) * 1000)

triton_latency    = float(np.mean(times))
triton_throughput = round(1000 / triton_latency, 1)

print(f"\nTriton+TensorRT — latency: {triton_latency:.2f} ms | throughput: {triton_throughput} img/s")

with open("results/benchmark_results.json") as f:
    results = json.load(f)

results["triton"] = {
    "latency_ms": round(triton_latency, 2),
    "throughput": triton_throughput
}

with open("results/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results updated in results/benchmark_results.json")