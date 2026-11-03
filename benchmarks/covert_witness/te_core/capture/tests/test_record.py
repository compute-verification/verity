"""`StepGraph.c_macs`, `StepGraph.structure()` and `RunGraph.record()`, te_core.flops's record form, against hand counts
on the 2-layer MLP."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models import TinyLM, lm_loss, mlp
from te_core.capture import capture_run, capture_step

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


def test_record_rows_and_step_order():
    m = mlp(FI, H, O)
    g0 = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), mse, batch(), updated=())
    g1 = g0.with_updated(g0.trained)
    rec = capture_run([(1, g0), (2, g1), (1, g0)]).record()
    steps = rec["steps"]
    assert [s["step"] for s in steps] == [0, 1, 2, 3]
    assert steps[0]["struct"] == steps[3]["struct"] != steps[1]["struct"] == steps[2]["struct"]
    assert rec["structures"][steps[0]["struct"]]["updated"] == []
    # one row per forward updated MAC node: [MACs per token, updated MACs per token upstream, MACs, tokens, the
    # narrowest width of its update-dependent operands]
    assert rec["structures"][steps[1]["struct"]]["updated"] == [[FI * H, 0, N * FI * H, N, 32],
                                                               [H * O, FI * H, N * H * O, N, 32]]
    assert [(s["tokens"], s["macs"], s["matmul_macs"], s["elementwise_macs"]) for s in steps] == [(N, GEMM + PARAMS, GEMM,
                                                                                                    PARAMS)] * 4
    assert rec["totals"]["macs"] == 4 * (GEMM + PARAMS) == sum(s["macs"] for s in steps)
    assert rec["setup"]["macs"] == rec["tail"]["macs"] == 0


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
