"""The s8 quantizer's units (`verity.ml.circuits.ops.quant`) are the loop's own ``quantize`` of each linear's input, bit for bit in
torch on x86-64: on every row the loop quantizes after a linear while it serves (up_proj's output relu'd, RELU true; down_proj's
as it is, RELU false), and on random and edge rows (all zero, all -0, all negative, ties, NaN with saturating neighbours, ±inf,
subnormal). The row unit, and the row in four chunks (its amax, then a chunk unit a chunk); on the served rows the Boolean row
unit too."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from verity.core.protocols.sampled_proofs.one_stage import draw as D
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import quant as Q
from verity.ml.circuits.ops.boolean import quant as BQ
from verity_torch_loop import loop as L


def _same(out: np.ndarray, codes: np.ndarray, scale: np.ndarray) -> bool:
    n = codes.shape[1]
    return bool((out[:, :n].astype(np.uint8) == codes.view(np.uint8)).all() and (out[:, n].astype(np.uint32) == scale).all())


def check(y: torch.Tensor, relu: bool, codes: torch.Tensor, scale: torch.Tensor, boolean: bool = False) -> None:
    """The units on the bf16 rows ``y`` against ``quantize``'s ``codes`` and ``scale`` of them: the row unit, the row in four
    chunks, and with ``boolean`` the Boolean row unit."""
    xb, h = L.bits(y), y.shape[1]
    want_codes, want_scale = L.bits(codes), L.bits(scale)
    assert _same(E.evaluate_batch(bind(Q.QuantRowS8, N=h, RELU=relu), xb), want_codes, want_scale)
    amax = E.evaluate_batch(bind(Q.RowAbsMax, N=h, RELU=relu), xb).astype(np.uint16)
    c = h // 4
    for k in range(4):
        got = E.evaluate_batch(bind(Q.QuantChunkS8, N=c, RELU=relu), xb[:, k * c:(k + 1) * c], amax)
        assert _same(got, want_codes[:, k * c:(k + 1) * c], want_scale), k
    if boolean:
        assert _same(BS.evaluate_words(bind(BQ.QuantRowS8, N=h, RELU=relu), xb), want_codes, want_scale)


@pytest.mark.slow
def test_the_units_are_the_quantize_of_every_row_the_loop_serves(monkeypatch):
    """Each linear's output against the loop's next ``quantize`` call, which reads it (a request's last linear has none), with
    the Boolean row unit at the loop's width, which ops' tests check only at small widths: ~90 s on a check node, most of it
    tracing `QuantRowS8_v2{128}` for each RELU."""
    calls: list = []
    quantize, linear = L.quantize, L.linear

    def recorded(fn):
        def call(*a):
            calls.append(fn(*a))
            return calls[-1]
        return call

    loop = L.TorchLoop(L.Model(), [3, 7, 11], steps=2)
    monkeypatch.setattr(L, "quantize", recorded(quantize))
    monkeypatch.setattr(L, "linear", recorded(linear))
    loop.window(D.parse_law("stratified:2"), cap=16)
    per = 4 * loop.steps
    assert len(calls) == per * len(loop.requests)
    edges: dict[bool, list] = {True: [], False: []}
    for r in range(len(loop.requests)):
        c = calls[per * r:per * (r + 1)]
        for i in range(1, per - 1, 2):
            edges[i // 2 % 2 == 0].append((c[i], *c[i + 1]))
    for relu, rows in edges.items():
        y, codes, scale = (torch.cat(t) for t in zip(*rows))
        check(y, relu, codes, scale, boolean=True)


def edge_rows(h: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(12, h, generator=g) * torch.tensor([0.05, 1.0, 300.0]).repeat_interleave(4)[:, None]
    e = torch.zeros(6, h)
    e[1] = -0.0
    e[2] = -torch.randn(h, generator=g).abs()
    e[3, :12] = torch.tensor([127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 125.5, -126.5, 64.5, -63.5, 3.0])
    e[4] = torch.randn(h, generator=g)
    e[4, :8] = torch.tensor([200, float("nan"), -300, 129, -129, 128, -128, 1.5])
    e[5] = torch.randn(h, generator=g)
    e[5, 1], e[5, 2] = float("inf"), float("-inf")
    rng = np.random.default_rng(1)
    sub = (rng.integers(1, 0x80, (1, h)) | rng.integers(0, 2, (1, h)) << 15).astype(np.uint16)
    return torch.cat([x.to(torch.bfloat16), e.to(torch.bfloat16), torch.from_numpy(sub.view(np.int16)).view(torch.bfloat16)])


@pytest.mark.parametrize("relu", [True, False])
def test_the_units_are_torchs_quantize_on_random_and_edge_rows(relu):
    y = edge_rows(L.H)
    x = y.to(torch.float32)
    codes, scale = L.quantize(torch.relu(x) if relu else x)
    assert codes[16, :8].tolist() == ([127, 0, 0, 127, 0, 127, 0, 2] if relu else [127, 0, -128, 127, -128, 127, -128, 2])
    assert scale[17].item() == float("inf") and not codes[17].any()
    check(y, relu, codes, scale)
