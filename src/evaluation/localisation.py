"""Score class activation maps against the radiologist bounding boxes.

RSNA ships boxes drawn by board-certified radiologists around the opacities
they identified. That makes it possible to ask a question the accuracy table
cannot answer: when a model says "pneumonia", is it looking at the right part
of the radiograph?

For VeritasXR this is also the only direct test of the dual-pathway premise.
Each pathway gets its own map, so "do the two pathways attend to different
regions, and which one localises better" becomes a measurement rather than an
assertion.

Two metrics, both standard in weakly-supervised localisation:

  pointing game   does the map's single peak land inside a ground-truth box?
                  Robust: no threshold to choose, so no knob to tune.
  IoU @ tau       binarise the map at tau * max, intersect with the box union.
                  Sensitive to tau, reported at a fixed 0.5 for comparability.

CAM vs Grad-CAM. VeritasXR's pathway heads are a single linear layer on
globally pooled features, so the map is exactly sum_k W[c,k] * A_k -- plain CAM,
no gradients. The torchvision baselines have deeper classifiers, so they need
Grad-CAM, where the channel weights come from the gradient of the class score.
Both reduce to a channel-weighted sum of the final feature maps.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F


def boxes_to_mask(boxes, size):
    """Normalised [x, y, w, h] rows -> boolean mask at `size` x `size`."""
    m = np.zeros((size, size), dtype=bool)
    for x, y, w, h in boxes:
        x0, y0 = int(np.floor(x * size)), int(np.floor(y * size))
        x1, y1 = int(np.ceil((x + w) * size)), int(np.ceil((y + h) * size))
        m[max(y0, 0):min(y1, size), max(x0, 0):min(x1, size)] = True
    return m


def peak_in_box(cam, boxes):
    """Pointing game: the map's argmax, in normalised coordinates, inside a box."""
    h, w = cam.shape
    i = int(np.argmax(cam))
    cy, cx = divmod(i, w)
    px, py = (cx + 0.5) / w, (cy + 0.5) / h
    return any(bx <= px <= bx + bw and by <= py <= by + bh
               for bx, by, bw, bh in boxes)


def cam_iou(cam, boxes, size=224, tau=0.5):
    """IoU between the map thresholded at tau * max and the box union."""
    up = F.interpolate(torch.tensor(cam)[None, None].float(), size=(size, size),
                       mode="bilinear", align_corners=False)[0, 0].numpy()
    up = up - up.min()
    if up.max() <= 0:
        return 0.0
    pred = up >= tau * up.max()
    gt = boxes_to_mask(boxes, size)
    union = (pred | gt).sum()
    return float((pred & gt).sum() / union) if union else 0.0


def mass_in_box(cam, boxes, size=224):
    """Share of positive activation mass that falls inside the boxes.

    Less brittle than IoU: no threshold, and it rewards a map that concentrates
    on the right region even if its extent is the wrong size.
    """
    up = F.interpolate(torch.tensor(cam)[None, None].float(), size=(size, size),
                       mode="bilinear", align_corners=False)[0, 0].numpy()
    up = np.clip(up - up.min(), 0, None)
    total = up.sum()
    if total <= 0:
        return 0.0
    return float(up[boxes_to_mask(boxes, size)].sum() / total)


class GradCAM:
    """Grad-CAM over a chosen module's output. Used for the baselines."""

    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, cls=1):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits[:, cls].sum().backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((self.acts * w).sum(1))
        return cam.cpu().numpy()


def veritasxr_cams(model, x):
    """Per-pathway CAM for the positive class. Forward pass only.

    Uses the positive class row of the head, W[1]. For a two-class softmax the
    discriminative direction is arguably W[1] - W[0], since the decision is the
    difference of the logits. That alternative was measured over three seeds and
    300 positive images: the two agree to within the seed-to-seed spread and
    neither is systematically better (pointing game 0.380/0.187/0.283 for W[1]
    against 0.327/0.180/0.303 for the difference). The simpler form is kept.
    """
    with torch.no_grad():
        feat = model.stem(x)
        out = {}
        for name in model.path_names:
            fmap = model.paths[name](feat)
            w = model.path_heads[name].weight[1]
            out[name] = torch.einsum("bchw,c->bhw", fmap, w).cpu().numpy()
    return out


def summarise(records):
    """Aggregate per-image scores into the numbers that get reported."""
    out = {}
    for key, vals in records.items():
        a = np.asarray(vals, dtype=float)
        out[key] = {"mean": round(float(a.mean()), 4),
                    "std": round(float(a.std()), 4),
                    "n": int(a.size)}
    return out
