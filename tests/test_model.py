"""Checks on the VeritasXR architecture.

The point of these is the third one. The previous version's uncertainty head
received no gradient -- the training loop discarded the model's second output --
so 528,577 parameters stayed at their initialisation while still driving a
clinical triage recommendation. Nothing in the repo would have caught that.
test_every_parameter_gets_gradient is what catches it now.

    python tests/test_model.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.veritasxr import ABLATIONS, build_ablation  # noqa: E402

OLD_PARAM_COUNT = 17_020_325   # the version this replaces
BATCH = 4


def loss_for(model, x, y):
    """The S3 training objective: merged verdict plus both per-path heads."""
    verdict, _, heads, _ = model(x, aux=True)
    crit = nn.CrossEntropyLoss()
    loss = crit(verdict, y)
    for logits in heads.values():
        loss = loss + 0.3 * crit(logits, y)
    return loss


def test_forward_shapes():
    x = torch.randn(BATCH, 1, 224, 224)
    for variant in ABLATIONS:
        m = build_ablation(variant).eval()
        with torch.no_grad():
            verdict, unc = m(x)
        assert verdict.shape == (BATCH, 2), f"{variant}: {verdict.shape}"
        assert unc.shape == (BATCH, 1), f"{variant}: {unc.shape}"
        assert torch.all((unc >= 0) & (unc <= 1)), f"{variant}: uncertainty outside [0,1]"
    print(f"  ok  all {len(ABLATIONS)} variants forward with uncertainty in [0,1]")


def test_every_parameter_gets_gradient():
    """No parameter may sit in the graph without receiving gradient.

    This is the regression guard for the original defect: a whole head wired
    into the forward pass, exported and served, that no loss ever touched.
    """
    x = torch.randn(BATCH, 1, 224, 224)
    y = torch.tensor([0, 1, 0, 1])
    for variant in ABLATIONS:
        m = build_ablation(variant).train()
        m.zero_grad()
        loss_for(m, x, y).backward()
        dead = [n for n, p in m.named_parameters()
                if p.requires_grad and (p.grad is None or not p.grad.abs().any())]
        assert not dead, f"{variant}: {len(dead)} parameter(s) got no gradient: {dead[:4]}"
    print(f"  ok  every parameter receives gradient in all {len(ABLATIONS)} variants")


def test_uncertainty_is_not_a_free_parameter():
    """Uncertainty must be derived from the path heads, not learned separately.

    Any module whose name suggests a dedicated uncertainty head would be a
    regression to the untrained design.
    """
    m = build_ablation("full")
    named = [n for n, _ in m.named_parameters()]
    offenders = [n for n in named if "uncertainty" in n.lower()]
    assert not offenders, f"uncertainty has its own parameters again: {offenders}"
    # It must still respond to the input.
    m.eval()
    with torch.no_grad():
        _, u1 = m(torch.randn(BATCH, 1, 224, 224))
        _, u2 = m(torch.randn(BATCH, 1, 224, 224) * 3 + 1)
    assert not torch.allclose(u1, u2), "uncertainty is constant w.r.t. the input"
    print("  ok  uncertainty is derived, parameter-free, and input-dependent")


def test_gate_is_per_input():
    """The README's 'per input' claim, which the old global nn.Parameter broke."""
    m = build_ablation("full").eval()
    with torch.no_grad():
        _, _, _, w1 = m(torch.randn(BATCH, 1, 224, 224), aux=True)
        _, _, _, w2 = m(torch.randn(BATCH, 1, 224, 224) * 2 - 1, aux=True)
    assert w1 is not None and w1.shape == (BATCH, 2)
    assert torch.allclose(w1.sum(1), torch.ones(BATCH), atol=1e-5), "gate must sum to 1"
    assert not torch.allclose(w1, w2), "gate does not vary with the input"
    # And the ablation that removes it really is input-independent.
    s = build_ablation("scalar_merge").eval()
    with torch.no_grad():
        _, _, _, sw1 = s(torch.randn(BATCH, 1, 224, 224), aux=True)
        _, _, _, sw2 = s(torch.randn(BATCH, 1, 224, 224) * 2 - 1, aux=True)
    assert torch.allclose(sw1, sw2), "scalar_merge should be input-independent"
    print("  ok  gate varies per input; scalar_merge ablation is global, as intended")


def test_parameter_budget():
    n = sum(p.numel() for p in build_ablation("full").parameters())
    assert n < OLD_PARAM_COUNT / 3, f"expected a large cut, got {n:,}"
    local = sum(p.numel() for p in build_ablation("local_only").parameters())
    glob = sum(p.numel() for p in build_ablation("global_only").parameters())
    print(f"  ok  {OLD_PARAM_COUNT:,} -> {n:,} "
          f"({1 - n / OLD_PARAM_COUNT:.1%} smaller); "
          f"local-only {local:,}, global-only {glob:,}")


if __name__ == "__main__":
    for fn in (test_forward_shapes, test_every_parameter_gets_gradient,
               test_uncertainty_is_not_a_free_parameter, test_gate_is_per_input,
               test_parameter_budget):
        fn()
    print("\nall checks passed")
