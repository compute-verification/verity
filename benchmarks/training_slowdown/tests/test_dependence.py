"""training_slowdown 4.0's definitions, on toy steps: T2 reads update dependence,
however the circuit is written, and T1's β is the narrowest cut the step's own circuit has (structure.py's docstring)."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from models import LoRA, TinyLM, lm_loss, lora
from training_slowdown.capture import capture_step, capture_steps

B, S, V, NH, L = 2, 4, 32, 2, 2
T = B * S


def lm_batch():
    return torch.empty(B, S, dtype=torch.long), torch.empty(B, S, dtype=torch.long)


def trainable(m):
    return [p for p in m.parameters() if p.requires_grad]


def deep(g, d) -> int:
    """The MACs of the forward's updated nodes with at least `d` updated MACs per token upstream of them."""
    return sum(macs for _, upstream, macs, _, _ in g.structure()["updated"] if upstream >= d)


# --- T2: dependence ----------------------------------------------------------------------------------------------------

class Merged(nn.Module):
    """`LoRA`'s adapter merged into the weight each step: ``x (W + s·B·A)ᵀ``, the same function and parameters."""

    def __init__(self, lo: LoRA) -> None:
        super().__init__()
        self.base, self.A, self.B, self.s = lo.base, lo.A, lo.B, lo.s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.base.weight + self.s * (self.B.weight @ self.A.weight))


def merged(model: TinyLM) -> TinyLM:
    m = copy.deepcopy(model)
    for b in m.blocks:
        b.q, b.v = Merged(b.q), Merged(b.v)
    return m


def test_merged_and_unmerged_lora_agree():
    """The same LoRA run written merged (W + s·B·A) and unmerged (W x + s·B(A x)): the old rule (a MAC is updated when its
    weight operand differs from θ) scored the unmerged one's φ a few percent of the merged one's; by dependence they
    agree up to the adapters' own MACs (2r(H + 1) per adapted projection per token, over H² for the projection). The
    embedding trains too, so layer 0's input is the run's own and nothing is a cache; |C| differs by what the circuits
    compute (the merged one's backward forms the full dW), so φ = deep MACs / |C| is compared through its numerator."""
    H, I, r = 64, 128, 1
    model = lora(TinyLM(V, H, NH, I, L), r)
    model.emb.weight.requires_grad_(True)
    um = capture_step(model, torch.optim.AdamW(trainable(model)), lm_loss, lm_batch())
    mm = merged(model)
    me = capture_step(mm, torch.optim.AdamW(trainable(mm)), lm_loss, lm_batch())
    assert um.trained == me.trained

    def weight_differs(g):   # the old rule's U
        return sum(m.macs for m in g.matmuls if m.phase == "forward" and m.result == "token"
                   and any(c == "updated" for c in m.operands[1:]))

    assert weight_differs(me) > 10 * weight_differs(um)
    adapters = L * 2 * (2 * r * H + H)   # x·Aᵀ, (xA)·Bᵀ and the scaling by s, per token, q and v of each layer
    assert um.updated_macs() - me.updated_macs() == T * adapters
    d_u, d_m = (max(row[1] for row in g.structure()["updated"]) for g in (um, me))
    assert 0 <= d_u - d_m <= adapters and (d_u - d_m) / d_m < 2 * r / H
    for D in (0, H, d_m // 2, d_m):
        assert 0 <= deep(um, D) - deep(me, D) <= T * adapters
        assert deep(um, D) / deep(me, D) - 1 < 2 * r / H + 1e-9


def test_tied_lookup_only_update_is_deep():
    """Only the tied table trains: its lookup moves the run's own rows into every block (a gather, no MAC), so every
    forward GEMM is updated and the head sits under all of them."""
    H, I, Vt = 16, 32, 48
    model = TinyLM(Vt, H, NH, I, L)
    model.head.weight = model.emb.weight
    model.requires_grad_(False)
    model.emb.weight.requires_grad_(True)
    g = capture_step(model, torch.optim.AdamW([model.emb.weight]), lm_loss, lm_batch())
    assert g.trained == {"emb.weight"} and g.dependent == ["emb.weight"]
    fwd = [m for m in g.matmuls if m.phase == "forward"]
    assert all(m.role == "updated" for m in fwd) and "embedding" not in [m.node for m in g.updated_elementwise]
    h2, hi, att = H * H, H * I, S * H
    per_layer = 4 * h2 + 2 * H + 2 * att + 2 * hi   # q k v o, SDPA's scalings of q and k, QK and PV, up and down
    head = fwd[-1]
    assert head.params[1] == "emb.weight" and head.upstream_macs == L * per_layer
    assert deep(g, L * per_layer) == T * H * Vt   # the head, under every block
    assert g.cut_bits == 32 * Vt * H


class Cached(nn.Module):
    """``head(frozen(x) · cache + mid(x))``, the buffer `cache` rewritten each step from `source(x)`'s token mean."""

    def __init__(self, d: int, o: int, source: str) -> None:
        super().__init__()
        self.frozen = nn.Linear(d, d, bias=False).requires_grad_(False)
        self.mid, self.head = nn.Linear(d, d, bias=False), nn.Linear(d, o, bias=False)
        self.register_buffer("cache", torch.zeros(d))
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.cache.copy_(getattr(self, self.source)(x).mean(0))
        return self.head(self.frozen(x) * self.cache + self.mid(x))


@pytest.mark.parametrize("source", ["frozen", "mid"])
def test_theta_only_cache_is_not_dependent(source):
    """A buffer the step rewrites from θ and the tokens alone carries nothing of the run's: not dependent, and what
    reads it isn't updated. Written from a trained layer, the same buffer is the run's own."""
    d, o, n = 8, 4, 6
    m = Cached(d, o, source)
    g = capture_step(m, torch.optim.SGD(trainable(m), lr=0.1), lambda m, b: F.mse_loss(m(b[0]), b[1]),
                     (torch.empty(n, d), torch.empty(n, o)), token_dims=[(0,), (0,)])
    dep = source == "mid"
    assert g.dependent == (["cache", "head.weight", "mid.weight"] if dep else ["head.weight", "mid.weight"])
    scale = next(m for m in g.updated_elementwise + [None] if m is None or m.op == "aten.mul.Tensor")
    assert (scale is not None) == dep
    head = next(m for m in g.matmuls if m.phase == "forward" and m.params[1] == "head.weight")
    frozen = next(m for m in g.matmuls if m.phase == "forward" and m.params[1] == "frozen.weight")
    assert frozen.role == "neither"
    # mid(x), and with a dependent cache the cache's own mid(x) and the scaling by it
    assert head.upstream_macs == d * d + (d * d + d if dep else 0)


# --- T1: the cut -------------------------------------------------------------------------------------------------------

D_IN, D_OUT, N = 64, 32, 16


class EF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(D_IN, D_OUT, bias=False)
        self.register_buffer("err", torch.zeros(D_OUT, D_IN))


def sign_ef(m, batch, _opts, lr=1e-3):
    """1-bit SGD with error feedback: send sign(g + e) and one scale, keep the rest in e."""
    x, y = batch
    F.mse_loss(m.fc(x), y).backward()
    with torch.no_grad():
        acc = m.fc.weight.grad + m.err
        sign = acc > 0
        scale = acc.abs().mean()
        dec = torch.where(sign, scale, -scale)
        m.err.copy_(acc - dec)
        m.fc.weight.sub_(dec, alpha=lr)


def ef_batch():
    return torch.empty(N, D_IN), torch.empty(N, D_OUT)


def second_of_three(step):
    """The second of three such steps of one run: what its forward reads is the first's, and the third's reads its."""
    return capture_steps([step] * 3, EF(), ef_batch(), measured=1, token_dims=[(0,), (0,)])


def test_sign_update_with_error_feedback_is_one_bit_per_value():
    g = second_of_three(sign_ef)
    assert g.trained == {"fc.weight"} and len(g.sinks) == 1   # the next forward reads the new weight, never the error
    assert g.updated_macs() == N * D_IN * D_OUT
    # the sign bits and the scale: every path from the gradient to the new weight crosses them
    assert g.cut_bits == D_IN * D_OUT + 32 and sorted(b for _, b in g.cut) == [32, D_IN * D_OUT]
    assert g.beta == 1 + 32 / (D_IN * D_OUT)


def momentum_through_a_copy(m, batch, _opts, lr=1e-3):
    """Heavy-ball momentum kept in `err`, which the step writes through a copy of itself (an fp32 buffer's ``float()``
    is the buffer); the update reads the buffer alone."""
    x, y = batch
    F.mse_loss(m.fc(x), y).backward()
    with torch.no_grad():
        buf = m.err.copy_(m.err.float().mul_(0.9).add_(m.fc.weight.grad))
        m.fc.weight.sub_(buf, alpha=lr)


def test_an_update_through_a_buffer_copied_onto_itself_reads_the_gradient():
    g = second_of_three(momentum_through_a_copy)
    assert g.trained == {"fc.weight"} and len(g.sinks) == 1
    # the copy's source is the buffer, so the gradient reaches the new weight through it, at its 32 bits an element
    assert g.cut_bits == 32 * D_IN * D_OUT


@pytest.mark.parametrize("autocast", [False, True])
def test_dense_update_is_p_bits_per_value(autocast):
    m = EF()

    def loss(m, b):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            return F.mse_loss(m.fc(b[0]).float(), b[1])

    g = capture_step(m, torch.optim.AdamW(m.fc.parameters()), loss, ef_batch(), token_dims=[(0,), (0,)])
    p = 16 if autocast else 32   # the width the forward reads the weight at
    assert g.cut_bits == p * D_IN * D_OUT and g.beta == p
