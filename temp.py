import asyncio
import aiohttp
import numpy as np
import time
from PIL import Image
import io

# Config
URL = "http://localhost:8080/predict"
TOTAL_REQUESTS = 10000
CONCURRENT = 10000  # requests at the same time

async def send_request(session, i):
    # Create a dummy grayscale image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode='L')
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    form = aiohttp.FormData()
    form.add_field("file", buf, filename="xray.jpg", content_type="image/jpeg")

    t0 = time.perf_counter()
    async with session.post(URL, data=form) as resp:
        result = await resp.json()
        latency = (time.perf_counter() - t0) * 1000
        print(f"[{i:03d}] {result['prediction']} | latency: {latency:.1f}ms | triton: {result['latency_ms']}ms")
        return latency

async def main():
    print(f"Sending {TOTAL_REQUESTS} requests, {CONCURRENT} concurrent...\n")
    latencies = []

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, TOTAL_REQUESTS, CONCURRENT):
            batch = [
                send_request(session, batch_start + i)
                for i in range(CONCURRENT)
            ]
            results = await asyncio.gather(*batch)
            latencies.extend(results)

    latencies = sorted(latencies)
    print(f"\n{'='*40}")
    print(f"Total requests : {TOTAL_REQUESTS}")
    print(f"Concurrency    : {CONCURRENT}")
    print(f"Avg latency    : {np.mean(latencies):.1f}ms")
    print(f"P50 latency    : {np.percentile(latencies, 50):.1f}ms")
    print(f"P90 latency    : {np.percentile(latencies, 90):.1f}ms")
    print(f"P99 latency    : {np.percentile(latencies, 99):.1f}ms")
    print(f"Min latency    : {min(latencies):.1f}ms")
    print(f"Max latency    : {max(latencies):.1f}ms")
    print(f"Throughput     : {TOTAL_REQUESTS / (sum(latencies)/1000):.1f} req/sec")

asyncio.run(main())