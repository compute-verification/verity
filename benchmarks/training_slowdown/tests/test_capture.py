"""`capture_step` / `capture_fn` on toy models against hand counts: every GEMM's MACs, phase and role as a multiset
derived from the architecture alone, the upstream counts, φ and the cut, the token dims, and the refusals."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as F

from models import TinyLM, conv_net, lm_loss, lora, mlp
from training_slowdown.capture import CaptureRefused, capture_fn, capture_step, phase

B, S, V, H, NH, I, L, R = 2, 4, 32, 16, 2, 32, 2, 2
T = B * S
H2, HI, ATT = H * H, H * I, S * H
#: one block's updated MACs per token when its input is the run's own: ln1 and ln2 with their gains, q k v o, SDPA's
#: scalings of q and k, QK and PV, up and down
LAYER = 2 * H + 4 * H2 + 2 * H + 2 * ATT + 2 * HI


def lm_batch():
    return torch.empty(B, S, dtype=torch.long), torch.empty(B, S, dtype=torch.long)


def roles(g) -> Counter:
    return Counter((m.phase, m.role, m.macs) for m in g.matmuls)


def fwd_layer(role: str) -> Counter:
    """One block's forward: q, k, v, o (T·H² each), up and down (T·H·I), QK and PV (T·S·H), all with role `role`."""
    return Counter({("forward", role, T * H * H): 4, ("forward", role, T * H * I): 2, ("forward", role, T * S * H): 2})


def layer_upstream(b: int) -> list[int]:
    """A block's forward GEMMs' updated MACs per token upstream (q, k, v, QK, PV, o, up, down), `b` above its input."""
    qkv = b + H
    qk = qkv + 2 * H2 + 2 * H
    pv = qk + ATT + H2
    o = pv + ATT
    up = o + H2 + H
    return [qkv] * 3 + [qk, pv, o, up, up + HI]


# --- the 2-layer MLP ---------------------------------------------------------------------------------------------------

def test_mlp_full_training():
    Fi, Hh, O, N = 8, 16, 4, 6
    model = mlp(Fi, Hh, O)
    g = capture_step(model, torch.optim.AdamW(model.parameters()), lambda m, b: F.cross_entropy(m(b[0]), b[1]),
                     (torch.empty(N, Fi), torch.empty(N, dtype=torch.long)))
    assert g.tokens == N and g.axes == {"batch": N}
    # fc1 and fc2 forward, fc2's dX (fc1's input needs no gradient), dW of both; bias grads are sums, not GEMMs
    assert roles(g) == Counter({("forward", "updated", N * Fi * Hh): 1, ("forward", "updated", N * Hh * O): 1,
                                ("backward", "neither", N * O * Hh): 1, ("backward", "producing", N * O * Hh): 1,
                                ("backward", "producing", N * Hh * Fi): 1})
    assert g.macs == N * (2 * Fi * Hh + 3 * Hh * O)
    fc1, fc2, dx2, dw2, dw1 = g.matmuls
    assert fc1.params == (None, "0.weight", "0.bias") and fc1.operands == ("activation", "updated", "updated")
    assert dx2.operands == ("gradient", "updated") and dw2.operands == ("gradient", "activation")
    assert (fc1.upstream, fc2.upstream, dx2.upstream) == (0, 1, 0)   # depth is the forward's
    assert (fc1.upstream_macs, fc2.upstream_macs, dx2.upstream_macs) == (0, Fi * Hh, 0)
    assert fc1.per_token == Fi * Hh
    assert fc1.token_dims == {"m": ("batch",)} and dw1.token_dims == {"k": ("batch",)}
    assert dw1.result == "parameter" and fc1.result == "token"
    # φ at D = 1 updated GEMM upstream: fc2, over the step's |C|
    assert g.phi(1, "matmuls") == Fraction(N * Hh * O, g.c_macs) and g.phi(0) == Fraction(N * (Fi + O) * Hh, g.c_macs)
    assert g.phi(Fi * Hh) == g.phi(1, "matmuls") and g.phi(Fi * Hh + 1) == 0
    assert g.trained == g.updated == {"0.weight", "0.bias", "2.weight", "2.bias"} == set(g.dependent) == set(g.sinks)
    assert g.cut_bits == 32 * (Fi * Hh + Hh + Hh * O + O)   # a dense update: every parameter at its width
    kinds = Counter(c["kind"] for c in g.step_crossing)
    assert kinds == {"parameter": 4, "optimizer_state": 12}   # step, exp_avg, exp_avg_sq per parameter


def test_mlp_frozen_first_layer_is_registered():
    model = mlp()
    model[0].requires_grad_(False)
    g = capture_step(model, torch.optim.SGD(model[2].parameters(), lr=0.1), lambda m, b: F.mse_loss(m(b[0]), b[1]),
                     (torch.empty(6, 8), torch.empty(6, 4)), token_dims=[(0,), (0,)])
    fc1 = g.matmuls[0]
    assert fc1.operands == ("activation", "registered", "registered") and fc1.role == "neither"
    assert g.trained == {"2.weight", "2.bias"}
    assert [m.role for m in g.matmuls] == ["neither", "updated", "producing"]   # fc1, fc2, dW2: no dX is needed


# --- the tiny transformer ----------------------------------------------------------------------------------------------

def test_transformer_full_training():
    model = TinyLM(V, H, NH, I, L)
    g = capture_step(model, torch.optim.AdamW(model.parameters()), lm_loss, lm_batch())
    assert g.tokens == T and g.axes == {"batch": B, "seq": S}
    want = Counter()
    for _ in range(L):
        want += fwd_layer("updated")
        # dX and dW of q, k, v, o (T·H²), up and down (T·H·I); attention's backward: four T·S·H bmm
        want += Counter({("backward", "neither", T * H * H): 4, ("backward", "producing", T * H * H): 4,
                         ("backward", "neither", T * H * I): 2, ("backward", "producing", T * H * I): 2,
                         ("backward", "neither", T * S * H): 4})
    want += Counter({("forward", "updated", T * H * V): 1, ("backward", "neither", T * H * V): 1,
                     ("backward", "producing", T * H * V): 1})
    assert roles(g) == want
    assert g.macs == 3 * T * H * V + L * (12 * T * H * H + 6 * T * H * I + 6 * T * S * H)
    fwd = [m for m in g.matmuls if m.phase == "forward"]
    assert [m.upstream for m in fwd] == [0, 0, 0, 2, 4, 5, 6, 7, 8, 8, 8, 10, 12, 13, 14, 15, 16]
    # the updated table's lookup moves its rows with no MAC; each norm with updated gains and each of SDPA's scalings
    # of q and k is an updated node of H MACs per token
    assert len(g.updated_elementwise) == 4 * L + 1 and all(m.per_token == H for m in g.updated_elementwise)
    assert "embedding" not in [m.node for m in g.updated_elementwise]
    assert [m.upstream_macs for m in fwd] == layer_upstream(0) + layer_upstream(LAYER) + [L * LAYER + H]
    assert all(m.upstream == 0 and m.upstream_macs == 0 for m in g.matmuls if m.phase == "backward")
    q0 = fwd[0]
    assert q0.params == (None, "blocks.0.q.weight") and q0.token_dims == {"m": ("batch", "seq")} and q0.per_token == H2
    qk = next(m for m in g.matmuls if m.op == "aten.bmm.default")
    assert qk.token_dims == {"batch": ("batch",), "m": ("seq",), "n": ("seq",)} and qk.operands == ("activation", "activation")
    assert qk.role == "updated"   # q and k are the run's own: attention's products are updated
    dw = [m for m in g.matmuls if m.role == "producing"]
    assert all(m.token_dims == {"k": ("batch", "seq")} and m.result == "parameter" for m in dw)
    assert g.updated_macs() == T * (L * LAYER + H + H * V)
    assert g.phi(1, "matmuls") == Fraction(g.updated_macs() - T * (3 * H2 + H), g.c_macs)   # all but layer 0's ln1, q, k, v
    assert g.phi(0) == Fraction(g.updated_macs(), g.c_macs)


def test_transformer_lora():
    model = lora(TinyLM(V, H, NH, I, L), R)
    trainable = [p for p in model.parameters() if p.requires_grad]
    g = capture_step(model, torch.optim.AdamW(trainable), lm_loss, lm_batch())
    assert g.trained == {f"blocks.{l}.{p}.{ab}.weight" for l in range(L) for p in "qv" for ab in "AB"}
    a = T * H * R   # each LoRA GEMM: x·Aᵀ, (xA)·Bᵀ, dB, d(xA), dA, dX through A
    want = Counter({("forward", "updated", T * H * V): 1, ("backward", "neither", T * H * V): 1})
    for l in range(L):
        if l == 0:   # layer 0's k and base q and v read θ and the tokens only: a cache
            want += Counter({("forward", "neither", T * H * H): 3, ("forward", "updated", T * H * H): 1,
                             ("forward", "updated", T * H * I): 2, ("forward", "updated", T * S * H): 2})
        else:
            want += fwd_layer("updated")
        want += Counter({("forward", "updated", a): 4})
        want += Counter({("backward", "neither", T * H * I): 2, ("backward", "neither", T * H * H): 1})   # down, up, o dX
        want += Counter({("backward", "producing", a): 4})   # dB, dA of q and v
        if l == 0:   # the embedding is frozen: nothing below layer 0's projections needs a gradient
            want += Counter({("backward", "neither", T * S * H): 3, ("backward", "neither", a): 2})
        else:
            want += Counter({("backward", "neither", T * S * H): 4, ("backward", "neither", T * H * H): 3,
                             ("backward", "neither", a): 4})
    assert roles(g) == want
    fwd = [m for m in g.matmuls if m.phase == "forward" and m.role == "updated"]
    # layer 0: q's A and B, v's A and B, QK, PV, o, up, down; layer 1 (base q, q.A, q.B, k, base v, v.A, v.B, ...); head
    assert [m.upstream for m in fwd] == [0, 1, 0, 1, 2, 5, 6, 7, 8, 9, 9, 10, 9, 9, 9, 10, 13, 17, 18, 19, 20, 21]
    base_q = next(m for m in g.matmuls if m.params[1:] == ("blocks.0.q.base.weight",))
    assert base_q.operands == ("activation", "registered") and base_q.role == "neither"
    # a θ weight times the run's own activations is updated
    base_q1 = next(m for m in g.matmuls if m.params[1:] == ("blocks.1.q.base.weight",))
    assert base_q1.operands == ("activation", "registered") and base_q1.role == "updated"
    assert g.cut_bits == 32 * 4 * L * H * R   # the adapters, dense


def test_transformer_top_layer_only():
    model = TinyLM(V, H, NH, I, L)
    for p in model.parameters():
        p.requires_grad_(False)
    top = model.blocks[-1]
    top.requires_grad_(True)
    g = capture_step(model, torch.optim.AdamW(top.parameters()), lm_loss, lm_batch())
    assert g.trained == {f"blocks.{L - 1}.{n}" for n, _ in top.named_parameters()}
    # the frozen head reads the top block's output: updated
    want = Counter({("forward", "updated", T * H * V): 1, ("backward", "neither", T * H * V): 1})
    want += fwd_layer("neither") + fwd_layer("updated")
    # the top block: dX and dW of q, k, v, o, up, down (q, k, v's dX feed ln1's weight gradient), attention's four bmm
    want += Counter({("backward", "neither", T * H * H): 4, ("backward", "producing", T * H * H): 4,
                     ("backward", "neither", T * H * I): 2, ("backward", "producing", T * H * I): 2,
                     ("backward", "neither", T * S * H): 4})
    assert roles(g) == want
    fwd = [m for m in g.matmuls if m.phase == "forward" and m.role == "updated"]
    assert [m.upstream for m in fwd] == [0, 0, 0, 2, 4, 5, 6, 7, 8]
    assert [m.upstream_macs for m in fwd] == layer_upstream(0) + [LAYER]   # ln_f's gains are θ's: no MAC
    assert g.phi(1, "matmuls") == Fraction(g.updated_macs() - T * (3 * H2 + H), g.c_macs)


def test_transformer_mezo_forward_only():
    model = TinyLM(V, H, NH, I, L)
    eps, lr = 1e-3, 1e-4

    def mezo(m, batch, _opts):
        params = [p for p in m.parameters() if p.requires_grad]
        with torch.no_grad():
            zs = [torch.randn_like(p) for p in params]
            for p, z in zip(params, zs):
                p.add_(z, alpha=eps)
            l1 = lm_loss(m, batch)
            for p, z in zip(params, zs):
                p.add_(z, alpha=-2 * eps)
            l2 = lm_loss(m, batch)
            for p, z in zip(params, zs):
                p.add_(z, alpha=eps)
            g = (l1 - l2) / (2 * eps)
            with phase("update"):
                for p, z in zip(params, zs):
                    p.sub_(lr * g * z)

    g = capture_fn(mezo, model, lm_batch())
    one = Counter({("forward", "updated", T * H * V): 1})
    for _ in range(L):
        one += fwd_layer("updated")
    assert roles(g) == one + one
    assert g.role_macs("producing") == 0
    assert g.trained == g.updated == {n for n, _ in model.named_parameters()}
    fwd = [m.upstream for m in g.matmuls if m.role == "updated"]
    assert fwd == [0, 0, 0, 2, 4, 5, 6, 7, 8, 8, 8, 10, 12, 13, 14, 15, 16] * 2   # the second pass reads no product of the first
    # the update is one 32-bit loss difference times a seeded direction
    assert g.cut_bits == 32 and g.beta == 32 / (g.updated_macs() / T)


def test_with_updated_reclassifies_without_recapture():
    model = TinyLM(V, H, NH, I, L)
    g = capture_step(model, torch.optim.AdamW(model.parameters()), lm_loss, lm_batch())
    first = g.with_updated(())   # the period's first step reads θ: nothing it multiplies is new
    assert first.digest == g.digest and first.macs == g.macs
    assert first.role_macs("updated") == 0 and first.role_macs("producing") == g.role_macs("producing")
    assert first.dependent == [] and first.cut_bits == 0 and first.beta is None


def test_tied_head_reads_the_embedding():
    Vt = 48   # unlike I, so the head's GEMMs are the T·H·Vt ones
    model = TinyLM(Vt, H, NH, I, L)
    model.head.weight = model.emb.weight
    g = capture_step(model, torch.optim.AdamW(model.parameters()), lm_loss,
                     (torch.empty(B, S, dtype=torch.long), torch.empty(B, S, dtype=torch.long)))
    head = [m for m in g.matmuls if m.macs == T * H * Vt]
    assert sorted((m.phase, m.role) for m in head) == [("backward", "neither"), ("backward", "producing"), ("forward", "updated")]
    assert head[0].params[1] == "emb.weight" and head[0].operands[1] == "updated"


def test_embedding_rows_are_dependent():
    """Attention straight on an updated embedding's rows: rows gathered by token ids are token-indexed activations, and
    the run's own, so attention on them is updated, with no MAC for the gather."""
    class Bare(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(V, H)
            self.head = torch.nn.Linear(H, V, bias=False)

        def forward(self, ids):
            x = self.emb(ids)
            b, s, d = x.shape
            q = x.view(b, s, NH, d // NH).transpose(1, 2)
            return self.head(F.scaled_dot_product_attention(q, q, q, is_causal=True).transpose(1, 2).reshape(b, s, d))

    model = Bare()
    g = capture_step(model, torch.optim.AdamW(model.parameters()), lm_loss, lm_batch())
    fwd = [m for m in g.matmuls if m.phase == "forward"]
    assert sorted((m.role, m.operands, m.macs) for m in fwd) == [
        ("updated", ("activation", "activation"), T * S * H), ("updated", ("activation", "activation"), T * S * H),
        ("updated", ("activation", "updated"), T * H * V)]
    assert fwd[0].upstream_macs == 2 * H   # SDPA's scalings of q and k


def test_conv_net():
    Bc, C, K, side, O = 2, 3, 8, 8, 4
    model = conv_net(C, K, side, O)
    g = capture_step(model, torch.optim.SGD(model.parameters(), lr=0.1), lambda m, b: F.cross_entropy(m(b[0]), b[1]),
                     (torch.empty(Bc, C, side, side), torch.empty(Bc, dtype=torch.long)), token_dims=[(0,), (0,)])
    conv = Bc * K * (side - 2) ** 2 * C * 9
    lin = Bc * K * (side - 2) ** 2 * O
    # the conv's input needs no gradient: convolution_backward runs its weight GEMM only
    assert roles(g) == Counter({("forward", "updated", conv): 1, ("forward", "updated", lin): 1,
                                ("backward", "neither", lin): 1, ("backward", "producing", lin): 1,
                                ("backward", "producing", conv): 1})
    cb = next(m for m in g.matmuls if m.part == "grad_weight")
    assert cb.token_dims == {"k": ("batch",)}


@pytest.mark.parametrize("big", [False, True])
def test_upstream_sums_are_exact_over_a_wide_bitset(big):
    from training_slowdown.capture.structure import _bit_sums
    k = 20_003
    macs = [(b % 97 + 1) * (2 ** 62 if big else 1) for b in range(k)]
    tokens = [(3, 7, 64)[b % 3] for b in range(k)]
    m = sum(1 << b for b in range(k) if b % 5 != 2)
    want = sum((Fraction(macs[b], tokens[b]) for b in range(k) if b % 5 != 2), Fraction(0))
    total = _bit_sums(macs, tokens)
    assert total(m) == want and total(0) == 0 and total(1 << (k - 1)) == Fraction(macs[-1], tokens[-1])


# --- refusals ----------------------------------------------------------------------------------------------------------

class Branchy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x):
        h = self.fc(x)
        if h.sum() > 0:
            return h * 2
        return h


def test_refuses_branch_on_data():
    model = Branchy()
    with pytest.raises(CaptureRefused) as e:
        capture_step(model, torch.optim.SGD(model.parameters(), lr=0.1), lambda m, b: m(b[0]).sum(), (torch.empty(4, 8),))
    assert e.value.reason == "dynamic_control_flow" and e.value.op == "aten._local_scalar_dense.default"
    assert "if h.sum() > 0" in e.value.where


def test_refuses_item():
    model = mlp()

    def step(m, batch, _opts):
        loss = m(batch[0]).sum()
        lr = 0.1 / (1 + loss.item())
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(1 - lr)

    with pytest.raises(CaptureRefused) as e:
        capture_fn(step, model, (torch.empty(4, 8),))
    assert e.value.reason == "dynamic_control_flow" and "loss.item()" in e.value.where


def test_refuses_data_dependent_shape():
    model = mlp()
    with pytest.raises(CaptureRefused) as e:
        capture_step(model, torch.optim.SGD(model.parameters(), lr=0.1), lambda m, b: m(b[0])[m(b[0]) > 0].sum(),
                     (torch.empty(4, 8),))
    assert e.value.reason == "data_dependent_shape" and "nonzero" in e.value.op


@torch.library.custom_op("te_test::opaque", mutates_args=())
def _opaque(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@torch.library.custom_op("te_test::opaque_faked", mutates_args=())
def _opaque_faked(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_opaque_faked.register_fake
def _(x):
    return torch.empty_like(x)


def test_refuses_custom_kernel_without_fake():
    model = mlp()
    with pytest.raises(CaptureRefused) as e:
        capture_step(model, torch.optim.SGD(model.parameters(), lr=0.1), lambda m, b: torch.ops.te_test.opaque(m(b[0])).sum(),
                     (torch.empty(4, 8),))
    assert e.value.reason == "custom_kernel" and "te_test" in (e.value.op or "") and "opaque" in e.value.op


def test_refuses_custom_kernel_with_fake():
    model = mlp()
    with pytest.raises(CaptureRefused) as e:   # traced, but with no backward
        capture_fn(lambda m, b, _o: torch.ops.te_test.opaque_faked(m(b[0])).sum().backward(), model, (torch.empty(4, 8),))
    assert e.value.reason == "custom_kernel" and e.value.op == "te_test.opaque_faked.default"

    def forward_only(m, b, _o):
        with torch.no_grad():
            torch.ops.te_test.opaque_faked(m(b[0]))

    with pytest.raises(CaptureRefused) as e:   # in the graph: its products are not ATen ops
        capture_fn(forward_only, model, (torch.empty(4, 8),))
    assert e.value.reason == "custom_kernel" and e.value.op == "te_test.opaque_faked.default"


def test_refuses_cond():
    model = mlp()

    def step(m, batch, _opts):
        h = m(batch[0])
        torch.cond(h.sum() > 0, lambda x: x * 2, lambda x: x * 3, (h,)).sum().backward()

    with pytest.raises(CaptureRefused) as e:
        capture_fn(step, model, (torch.empty(4, 8),))
    assert e.value.reason == "dynamic_control_flow"


def test_refuses_batch_dependent_graph():
    model = mlp()

    def loss(m, b):
        return sum(m(b[0][i:i + 1]).sum() for i in range(b[0].shape[0]))   # one forward per example

    with pytest.raises(CaptureRefused) as e:
        capture_step(model, torch.optim.SGD(model.parameters(), lr=0.1), loss, (torch.empty(4, 8),))
    assert e.value.reason == "batch_dependent"
