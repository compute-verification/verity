"""`StepGraph.c_macs` and `StepGraph.structure()`, a step in training_slowdown.flops's record form, against hand counts
on the 2-layer MLP."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models import TinyLM, lm_loss, mlp
from training_slowdown.capture import capture_step

FI, H, O, N = 3, 5, 2, 4
GEMM = N * (2 * FI * H + 3 * H * O)
PARAMS = FI * H + H + H * O + O


def mse(m, b):
    return F.mse_loss(m(b[0]), b[1])


def batch(dtype=torch.float32):
    return torch.empty(N, FI, dtype=dtype), torch.empty(N, O, dtype=dtype)


@pytest.mark.parametrize("foreach", [False, True])
def test_sgd_update_is_one_mac_per_parameter_element(foreach):
    m = mlp(FI, H, O)
    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1, foreach=foreach), mse, batch())
    assert g.macs == GEMM
    assert g.elementwise_macs == PARAMS  # p.add_(grad, alpha=-lr), per parameter or as one _foreach_add_
    assert g.approx_macs == 0
    assert g.c_macs == GEMM + PARAMS


def test_a_steps_updated_rows_are_of_the_weights_it_reads_as_updated():
    """The same step read as a run's first, whose weights are θ's (`updated=()`), and as a later one, which reads
    what it trains as updated (the default): the first has no updated row, and the counts are the same."""
    m = mlp(FI, H, O)
    g0 = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), mse, batch(), updated=())
    g1 = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), mse, batch())
    assert g0.digest == g1.digest and g1.updated == g1.trained == g0.trained and g0.updated == frozenset()
    assert g0.structure()["updated"] == []
    # one row per forward updated MAC node: [MACs per token, updated MACs per token upstream, MACs, tokens, the
    # narrowest width of its update-dependent operands]
    assert g1.structure()["updated"] == [[FI * H, 0, N * FI * H, N, 32], [H * O, FI * H, N * H * O, N, 32]]
    assert [(g.tokens, g.c_macs, g.macs, g.elementwise_macs) for g in (g0, g1)] == [(N, GEMM + PARAMS, GEMM, PARAMS)] * 2


def test_bits_are_the_updated_weights_width():
    m = mlp(FI, H, O).to(torch.bfloat16)
    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), mse, batch(torch.bfloat16))
    assert [r[4] for r in g.structure()["updated"]] == [16, 16]


def test_bf16_autocast_reads_through_the_casts():
    B, S, V, Hd, I, L = 2, 4, 32, 16, 32, 2
    T = B * S

    def loss(m, b):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return lm_loss(m, b)

    model = TinyLM(V, Hd, 2, I, L)
    ids = torch.zeros(B, S, dtype=torch.long)
    g = capture_step(model, torch.optim.AdamW(model.parameters()), loss, (ids, ids))
    assert g.macs == 3 * T * Hd * V + L * (12 * T * Hd * Hd + 6 * T * Hd * I + 6 * T * S * Hd)
    updated = [m for m in g.matmuls if m.role == "updated" and m.phase == "forward"]
    assert len(updated) == 8 * L + 1
    # the projections at bf16 (SDPA's math backend runs QK and PV in fp32); the norms, which autocast runs in fp32, read
    # their gains at 32 bits
    assert all(m.dtypes == (("float32",) if m.op == "aten.bmm.default" else ("bfloat16",)) * 2 for m in updated)
    assert len(g.updated_elementwise) == 4 * L + 1
    assert sorted({r[4] for r in g.structure()["updated"]}) == [16, 32]
    # the cut: the GEMMs' weights at the 16 bits the forward reads them at, the table and the gains at 32
    gemm = L * (4 * Hd * Hd + 2 * Hd * I) + Hd * V
    assert g.cut_bits == 16 * gemm + 32 * (V * Hd + (2 * L + 1) * 2 * Hd)
