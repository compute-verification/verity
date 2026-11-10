"""`lower` against hand gate counts and against the vLLM frontend's own reference derivation of the same forward
(`lower_forward`), its coverage on a conv net, and that a failure is data."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models import TinyLM, conv_net, lm_loss, mlp
from training_slowdown.capture import capture_step
from training_slowdown.capture.lower import lower, lower_forward


def _no_macs(defs: dict) -> dict:
    return {k: {"calls": v["calls"], "gates": v["gates"]} for k, v in defs.items()}


def test_mlp_gates_by_hand_and_against_the_frontend():
    FI, H, O, N = 3, 5, 2, 4
    m = mlp(FI, H, O)
    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), lambda mm, b: F.mse_loss(mm(b[0]), b[1]),
                     (torch.empty(N, FI), torch.empty(N, O)))
    r = lower(g)
    assert r["error"] is None and r["unlowered"] == {}
    assert r["coverage"] == 1.0 and r["c_coverage"] == g.macs / g.c_macs
    # LinearRef_v1{K,N,f32} has N(K+1) gates per row (N(K+2) with a bias): a GEMM's MACs plus its outputs (twice, biased)
    outputs = {"fc1": N * H, "fc2": N * O, "dX2": N * H, "dW2": O * H, "dW1": H * FI}
    assert r["gates"] == g.macs + sum(outputs.values()) + N * H + N * O
    fwd = lower(g, phases={"forward"})
    ref = lower_forward(m, (torch.randn(N, FI),))
    assert ref["error"] is None
    assert _no_macs(fwd["definitions"]) == {k: v for k, v in ref["definitions"].items() if k.startswith("LinearRef")}


def test_tiny_lm_fully_lowered_and_its_weight_gemms_as_the_frontend_binds_them():
    B, S, H, L = 2, 4, 16, 2
    model = TinyLM()
    ids = torch.zeros(B, S, dtype=torch.long)
    g = capture_step(model, torch.optim.AdamW(model.parameters()), lm_loss, (ids, ids))
    r = lower(g)
    assert r["error"] is None and r["coverage"] == 1.0 and r["unlowered"] == {}
    fwd = lower(g, phases={"forward"})["definitions"]
    ref = lower_forward(model, (ids,))
    assert ref["error"] is None
    linear = {k: v for k, v in ref["definitions"].items() if k.startswith("LinearRef")}
    assert {k: v for k, v in _no_macs(fwd).items() if k in linear} == linear
    # the frontend lowers attention to EagerAttentionRef; here its two bmm's are LinearRef calls, T·S·H MACs each per layer
    assert sorted(v["macs"] for k, v in fwd.items() if k not in linear) == [B * S * S * H * L] * 2
    assert any(k.startswith("EagerAttentionRef") for k in ref["definitions"])


def test_conv_net_coverage_names_what_is_not_lowered():
    Bc, C, K, side, O = 2, 3, 8, 8, 4
    model = conv_net(C, K, side, O)
    g = capture_step(model, torch.optim.SGD(model.parameters(), lr=0.1), lambda m, b: F.cross_entropy(m(b[0]), b[1]),
                     (torch.empty(Bc, C, side, side), torch.empty(Bc, dtype=torch.long)), token_dims=[(0,), (0,)])
    conv = Bc * K * (side - 2) ** 2 * C * 9
    lin = Bc * K * (side - 2) ** 2 * O
    r = lower(g)
    assert r["unlowered"] == {"no reference rule for convolution": conv,
                              "no reference rule for convolution_backward.grad_weight": conv}
    assert r["coverage"] == 3 * lin / (3 * lin + 2 * conv)
    ref = lower_forward(model, (torch.randn(Bc, C, side, side),))
    assert ref["error"].startswith("unsupported") and "aten.conv2d.default" in ref["error"]


def test_failure_is_data():
    assert lower(None)["error"].startswith("AttributeError")
    assert lower_forward(None, ())["error"] is not None
