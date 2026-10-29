"""Export a trained VeritasXR checkpoint to ONNX for in-browser inference.

The wrapper returns everything the demo needs from a single forward pass:
verdict probabilities, the uncertainty score, and one class activation map per
pathway.

The CAMs matter here. onnxruntime-web is inference-only, so Grad-CAM -- which
needs gradients -- is unavailable in the browser. It is not needed: each
pathway head is a single linear layer on globally average-pooled features, so
the class activation map is exactly a weighted sum of the feature maps,

    CAM_c(x, y) = sum_k W[c, k] * A_k(x, y)

which is a plain einsum on the forward pass. The maps are computed inside the
graph so the browser ships no weights and does no matrix algebra of its own.

    python scripts/export_onnx.py --checkpoint models/veritasxr_seed0.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.veritasxr import get_model  # noqa: E402


class VeritasXRForWeb(nn.Module):
    """Inference-shaped wrapper. The training model is left untouched."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        m = self.m
        x = m.stem(x)
        feats, logits, cams = {}, {}, {}
        for name in m.path_names:
            fmap = m.paths[name](x)                       # (B, C, H, W)
            pooled = fmap.mean((2, 3))                    # (B, C)
            feats[name] = pooled
            head = m.path_heads[name]
            logits[name] = head(pooled)
            # CAM for the positive class, straight from the head's weights.
            w = head.weight[1]                            # (C,)
            cams[name] = torch.einsum("bchw,c->bhw", fmap, w).unsqueeze(1)

        pooled_list = [feats[n] for n in m.path_names]
        merged, _ = m._merge(pooled_list)
        verdict = m.classifier(merged)

        if len(m.path_names) > 1:
            unc = m._disagreement([logits[n] for n in m.path_names])
        else:
            unc = m._entropy(verdict)

        probs = F.softmax(verdict, dim=1)
        return (probs, unc,
                cams[m.path_names[0]],
                cams[m.path_names[-1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("models/veritasxr_seed0.pt"))
    ap.add_argument("--out", type=Path, default=Path("docs/model/veritasxr.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = get_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    wrapper = VeritasXRForWeb(model).eval()
    dummy = torch.randn(1, 1, 224, 224)
    with torch.no_grad():
        out = wrapper(dummy)
    print("wrapper outputs:", [tuple(o.shape) for o in out])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (dummy,), str(args.out),
        export_params=True, opset_version=args.opset,
        input_names=["input"],
        output_names=["verdict", "uncertainty", "cam_local", "cam_global"],
        dynamic_axes={k: {0: "batch"} for k in
                      ("input", "verdict", "uncertainty", "cam_local", "cam_global")},
        do_constant_folding=True,
        # torch 2.14 defaults to the dynamo exporter, whose generated FX code
        # fails to parse under Python 3.14. The TorchScript path exports this
        # graph cleanly, so it is pinned rather than left to the default.
        dynamo=False,
    )
    mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({mb:.1f} MB, opset {args.opset})")


if __name__ == "__main__":
    main()
