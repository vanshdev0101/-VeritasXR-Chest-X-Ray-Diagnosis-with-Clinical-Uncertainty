"""Verify the exported ONNX graph reproduces the PyTorch model.

A demo that silently disagrees with the trained model is worse than no demo:
it looks authoritative while being wrong. This runs real test images through
both and fails loudly if they diverge.

    python scripts/check_onnx_parity.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.export_onnx import VeritasXRForWeb  # noqa: E402
from src.models.veritasxr import get_model  # noqa: E402

SIZE = 224
TOL_PROB = 1e-4
TOL_CAM = 1e-3


def preprocess(path):
    """The evaluation transform: greyscale, resize, /255, (x-0.5)/0.5."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((SIZE, SIZE), Image.BILINEAR),
                       dtype=np.float32)
    return (((a / 255.0) - 0.5) / 0.5)[None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("models/veritasxr_seed0.pt"))
    ap.add_argument("--onnx", type=Path, default=Path("docs/model/veritasxr.onnx"))
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("-n", type=int, default=64)
    args = ap.parse_args()

    model = get_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    wrapper = VeritasXRForWeb(model).eval()
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    print(f"  onnx outputs: {names}")

    df = pd.read_csv(args.data / "test.csv").head(args.n)
    dp = dmax = du = dcam = 0.0
    disagree = 0
    for row in df.itertuples(index=False):
        x = preprocess(args.data / row.path)
        with torch.no_grad():
            t = wrapper(torch.from_numpy(x))
        o = sess.run(None, {"input": x})
        tp, op = t[0].numpy(), o[0]
        dp += float(np.abs(tp - op).mean())
        dmax = max(dmax, float(np.abs(tp - op).max()))
        du = max(du, float(np.abs(t[1].numpy() - o[1]).max()))
        du = max(du, float(np.abs(t[2].numpy() - o[2]).max()))   # disagreement
        dcam = max(dcam, max(float(np.abs(t[3].numpy() - o[3]).max()),
                             float(np.abs(t[4].numpy() - o[4]).max())))
        if int(tp.argmax()) != int(op.argmax()):
            disagree += 1

    n = len(df)
    print(f"\n  images compared          {n}")
    print(f"  mean |Δ probability|     {dp/n:.3e}")
    print(f"  max  |Δ probability|     {dmax:.3e}   (tolerance {TOL_PROB:.0e})")
    print(f"  max  |Δ uncertainty|     {du:.3e}   (tolerance {TOL_PROB:.0e})")
    print(f"  max  |Δ CAM|             {dcam:.3e}   (tolerance {TOL_CAM:.0e})")
    print(f"  label disagreements      {disagree} / {n}")

    ok = dmax < TOL_PROB and du < TOL_PROB and dcam < TOL_CAM and disagree == 0
    print("\n  PASS: ONNX reproduces PyTorch" if ok else "\n  FAIL: export diverges")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
