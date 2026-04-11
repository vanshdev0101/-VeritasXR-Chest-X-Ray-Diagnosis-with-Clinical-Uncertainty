import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time, json, os
from veritasxr_model import get_model

os.makedirs("results", exist_ok=True)

WARMUP = 50
RUNS   = 200
BATCH  = 1
INPUT  = (BATCH, 1, 224, 224)  # 1 channel grayscale

dummy = torch.randn(*INPUT)

# ───────── PyTorch Benchmark ─────────
print("Benchmarking VeritasXR PyTorch (GPU)...")

model = get_model().eval().cuda()
model.load_state_dict(torch.load("models/veritasxr.pth"))
x = dummy.cuda()

with torch.no_grad():
    for _ in range(WARMUP):
        verdict, _ = model(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        verdict, _ = model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

pt_latency     = np.mean(times)
pt_throughput  = 1000 / pt_latency * BATCH
print(f"  PyTorch  — latency: {pt_latency:.2f} ms | throughput: {pt_throughput:.1f} img/s")

# ───────── TensorRT Benchmark ─────────
print("Benchmarking VeritasXR TensorRT (FP16)...")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime    = trt.Runtime(TRT_LOGGER)

with open("models/veritasxr_fp16.trt", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
input_name   = tensor_names[0]
output_name  = tensor_names[1]

context.set_input_shape(input_name, INPUT)

inp_host = np.random.randn(*INPUT).astype(np.float32)
out_host = np.empty((BATCH, 2), dtype=np.float32)  # 2 classes

inp_dev = cuda.mem_alloc(inp_host.nbytes)
out_dev = cuda.mem_alloc(out_host.nbytes)

context.set_tensor_address(input_name,  int(inp_dev))
context.set_tensor_address(output_name, int(out_dev))

stream = cuda.Stream()

# Warmup
for _ in range(WARMUP):
    cuda.memcpy_htod_async(inp_dev, inp_host, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(out_host, out_dev, stream)
    stream.synchronize()

# Benchmark
times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    cuda.memcpy_htod_async(inp_dev, inp_host, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(out_host, out_dev, stream)
    stream.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

trt_latency    = np.mean(times)
trt_throughput = 1000 / trt_latency * BATCH
print(f"  TensorRT — latency: {trt_latency:.2f} ms | throughput: {trt_throughput:.1f} img/s")

# ───────── Save Results ─────────
speedup = pt_latency / trt_latency

results = {
    "veritasxr_pytorch": {
        "latency_ms":  round(pt_latency, 2),
        "throughput":  round(pt_throughput, 1),
        "parameters":  17_020_325,
        "accuracy":    75.6
    },
    "veritasxr_tensorrt": {
        "latency_ms":  round(trt_latency, 2),
        "throughput":  round(trt_throughput, 1),
    },
    "resnet50_reference": {
        "latency_ms":  18.1,
        "throughput":  55,
        "parameters":  25_557_032,
        "accuracy":    94.0
    },
    "speedup": round(speedup, 2)
}

with open("results/veritasxr_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSpeedup: {speedup:.2f}x faster with TensorRT")
print("Results saved to results/veritasxr_benchmark.json")