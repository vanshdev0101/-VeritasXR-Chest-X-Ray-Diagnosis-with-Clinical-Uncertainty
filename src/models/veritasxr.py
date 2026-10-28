"""VeritasXR: a dual-pathway CNN with a derived uncertainty signal.

Two branches read a shared stem and differ only in convolution kernel size, so
one sees small structure and the other sees whole-lung structure. Their features
are blended by a gate predicted from the input, and their disagreement is the
uncertainty estimate.

What changed from the first version, and why
--------------------------------------------
1. The uncertainty head received no gradient. The training loop discarded the
   model's second output, so its 528,577 parameters stayed at initialisation
   while still driving a clinical triage recommendation. It is gone. Uncertainty
   is now *computed* as the Jensen-Shannon divergence between two independently
   supervised per-path heads, so no part of the uncertainty path is untrained.
   This is also what the original README claimed ("inter-path disagreement")
   but never implemented -- the old code merged the paths before the head saw
   them, which makes disagreement unobservable by construction.

2. `path_weights` was a single global `nn.Parameter`, not the per-input
   weighting the README described, and it converged to 0.510/0.490 -- a plain
   average. Its gradient is damped by the softmax and freely absorbed by the
   4.4M-parameter layer downstream, so it had no reason to move. The default is
   now a gate predicted from the pooled features of both paths, which is
   per-input for real. `merge="scalar"` reproduces the old behaviour for the
   ablation.

3. Blocks are bottlenecked and downsample. The old version carried 256x56x56
   through six convolutions with no stride, spending 59% of its parameters on a
   7x7 branch that received 0.490 of the merge weight. Measured throughput was
   4-5x slower than baselines with more parameters.

4. The classifier pooled to 4x4 and fed `Linear(4096, 1024)` -- 4.2M parameters,
   25% of the model, in one matrix. Global average pooling replaces it.

Together: 17,020,325 -> roughly 4.2M parameters, with the architectural ideas
(dual kernel scales, squeeze-excitation, learned merge) intact.

The constructor flags are the ablation: see ABLATIONS.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Each entry is a constructor kwargs set. S4 runs these to test whether the
# design decisions earn their keep.
ABLATIONS = {
    "full": {},
    "no_se": {"use_se": False},
    "local_only": {"paths": ("local",)},
    "global_only": {"paths": ("global",)},
    "fixed_merge": {"merge": "fixed"},
    "scalar_merge": {"merge": "scalar"},
    "concat_merge": {"merge": "concat"},
}

KERNEL = {"local": 3, "global": 7}


class Bottleneck(nn.Module):
    """1x1 reduce -> kxk -> 1x1 expand, with a projection shortcut.

    The old block stacked three same-width kxk convolutions, which at k=7 and
    256 channels is where most of the parameter budget went.
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, expansion=2):
        super().__init__()
        mid = max(out_ch // expansion, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Identity() if stride == 1 and in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x), inplace=True)


class SqueezeExcitation(nn.Module):
    """Channel attention: pool each map to a scalar, learn per-channel gains."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x.mean((2, 3)))
        return x * w[:, :, None, None]


class Pathway(nn.Module):
    """One branch: three downsampling bottleneck stages at a fixed kernel size."""

    def __init__(self, in_ch, widths, kernel_size, use_se=True):
        super().__init__()
        stages, ch = [], in_ch
        for w in widths:
            stages.append(Bottleneck(ch, w, kernel_size, stride=2))
            ch = w
        self.stages = nn.Sequential(*stages)
        self.se = SqueezeExcitation(ch) if use_se else nn.Identity()
        self.out_channels = ch

    def forward(self, x):
        return self.se(self.stages(x))


class VeritasXR(nn.Module):
    def __init__(self, num_classes=2, widths=(128, 256, 384), stem_width=64,
                 paths=("local", "global"), use_se=True, merge="gate",
                 dropout=0.3):
        super().__init__()
        if not paths:
            raise ValueError("at least one pathway is required")
        for p in paths:
            if p not in KERNEL:
                raise ValueError(f"unknown pathway {p!r}; expected {list(KERNEL)}")
        if merge not in ("gate", "scalar", "fixed", "concat"):
            raise ValueError(f"unknown merge {merge!r}")

        self.path_names = tuple(paths)
        self.merge_mode = merge

        # An 11x11 stem is kept deliberately: radiographic signal is diffuse
        # texture rather than sharp edges, so the first layer looks wide.
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_width, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, stem_width, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.paths = nn.ModuleDict({
            name: Pathway(stem_width, widths, KERNEL[name], use_se=use_se)
            for name in self.path_names
        })
        ch = next(iter(self.paths.values())).out_channels
        n = len(self.path_names)

        if merge == "concat":
            merged_ch = ch * n
        else:
            merged_ch = ch

        if merge == "gate" and n > 1:
            # Predicted from the input, which is what "per input" requires.
            self.gate = nn.Sequential(
                nn.Linear(ch * n, 64), nn.ReLU(inplace=True), nn.Linear(64, n)
            )
        elif merge == "scalar" and n > 1:
            # The original formulation, kept for the ablation.
            self.path_weights = nn.Parameter(torch.ones(n) / n)

        self.classifier = nn.Sequential(
            nn.Linear(merged_ch, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        # One linear head per path. These are what make the disagreement signal
        # meaningful: each path must be independently predictive, so the paths
        # are pushed to differentiate instead of co-adapting into duplicates.
        self.path_heads = nn.ModuleDict(
            {name: nn.Linear(ch, num_classes) for name in self.path_names}
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _merge(self, pooled):
        """pooled: list of (B, C). Returns (merged, weights or None)."""
        if len(pooled) == 1:
            return pooled[0], None
        stacked = torch.stack(pooled, dim=1)             # (B, n, C)
        if self.merge_mode == "concat":
            return torch.cat(pooled, dim=1), None
        if self.merge_mode == "gate":
            w = F.softmax(self.gate(torch.cat(pooled, dim=1)), dim=1)  # (B, n)
        elif self.merge_mode == "scalar":
            w = F.softmax(self.path_weights, dim=0).expand(stacked.size(0), -1)
        else:  # fixed
            w = torch.full_like(stacked[:, :, 0], 1.0 / stacked.size(1))
        return (stacked * w[:, :, None]).sum(dim=1), w

    @staticmethod
    def _disagreement(logit_list):
        """Jensen-Shannon divergence between path predictions, scaled to [0,1].

        Epistemic only: paths that agree while both being wrong score 0. If that
        becomes the limiting factor, add the predictive entropy of `verdict`.
        """
        probs = [F.softmax(l.float(), dim=1).clamp_min(1e-8) for l in logit_list]
        m = torch.stack(probs, 0).mean(0)
        kl = sum((p * (p / m).log()).sum(1) for p in probs) / len(probs)
        return (kl / math.log(2)).clamp(0, 1).unsqueeze(1)

    @staticmethod
    def _entropy(logits):
        """Fallback for single-path variants, which have no disagreement."""
        p = F.softmax(logits.float(), dim=1).clamp_min(1e-8)
        h = -(p * p.log()).sum(1) / math.log(p.size(1))
        return h.clamp(0, 1).unsqueeze(1)

    def forward(self, x, aux=False):
        x = self.stem(x)
        feats = {n: self.paths[n](x).mean((2, 3)) for n in self.path_names}
        pooled = [feats[n] for n in self.path_names]
        path_logits = {n: self.path_heads[n](feats[n]) for n in self.path_names}

        merged, weights = self._merge(pooled)
        verdict = self.classifier(merged)

        if len(self.path_names) > 1:
            uncertainty = self._disagreement(list(path_logits.values()))
        else:
            uncertainty = self._entropy(verdict)

        if not aux:
            return verdict, uncertainty
        return verdict, uncertainty, path_logits, weights


def get_model(**kwargs):
    return VeritasXR(**kwargs)


def build_ablation(variant, **kwargs):
    if variant not in ABLATIONS:
        raise KeyError(f"unknown ablation {variant!r}; known: {list(ABLATIONS)}")
    return VeritasXR(**{**ABLATIONS[variant], **kwargs})
