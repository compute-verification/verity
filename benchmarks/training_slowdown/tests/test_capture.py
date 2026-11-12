"""`capture_step` / `capture_steps` on toy models against hand counts: every GEMM's MACs, phase and role
as a multiset derived from the architecture alone, the upstream counts, φ and the cut, the token dims, a measured
step's circuit among its neighbours, and the refusals."""
from __future__ import annotations

import copy
from collections import Counter
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as F

from models import TinyLM, conv_net, lm_loss, lora, mlp
from training_slowdown.capture import CaptureRefused, capture_step, capture_steps, products
from training_slowdown.routes._optimizers import Muon

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


def macs_of(g, role: str) -> int:
    """The MACs of the step's GEMMs with role `role`."""
    return sum(m.macs for m in g.matmuls if m.role == role)


def deep(g, d, by: str = "upstream_macs") -> int:
    """The MACs of the forward's updated nodes, GEMMs and elementwise, with at least `d` updated MACs per token upstream
    of them (by "upstream": updated GEMMs)."""
    return sum(m.macs for m in g.matmuls + g.updated_elementwise
               if m.role == "updated" and m.phase == "forward" and getattr(m, by) >= d)


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
    # the updated MACs with an updated GEMM upstream are fc2's
    assert deep(g, 1, "upstream") == N * Hh * O and deep(g, 0) == N * (Fi + O) * Hh == g.updated_macs()
    assert deep(g, Fi * Hh) == deep(g, 1, "upstream") and deep(g, Fi * Hh + 1) == 0
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
    assert deep(g, 1, "upstream") == g.updated_macs() - T * (3 * H2 + H)   # all but layer 0's ln1, q, k, v
    assert deep(g, 0) == g.updated_macs()


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
    assert deep(g, 1, "upstream") == g.updated_macs() - T * (3 * H2 + H)


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
            for p, z in zip(params, zs):
                p.sub_(lr * g * z)

    g = capture_steps([mezo] * 3, model, lm_batch(), measured=1)
    one = Counter({("forward", "updated", T * H * V): 1})
    for _ in range(L):
        one += fwd_layer("updated")
    assert roles(g) == one + one
    assert macs_of(g, "producing") == 0
    assert g.trained == {n for n, _ in model.named_parameters()}
    fwd = [m.upstream for m in g.matmuls if m.role == "updated"]
    assert fwd == [0, 0, 0, 2, 4, 5, 6, 7, 8, 8, 8, 10, 12, 13, 14, 15, 16] * 2   # the second pass reads no product of the first
    # the update is one 32-bit loss difference times a seeded direction
    assert g.cut_bits == 32 and g.beta == 32 / (g.updated_macs() / T)
    assert g.formalism["grad_cut_bits"] == g.formalism["partial_cut_bits"] == 32 and g.formalism["state_bits"] == 0


def test_a_step_that_reads_theta_alone_multiplies_nothing_updated():
    model = TinyLM(V, H, NH, I, L)
    g = capture_step(model, torch.optim.AdamW(model.parameters()), lm_loss, lm_batch())
    # the period's first step reads θ: nothing it multiplies is new
    first = capture_step(model, torch.optim.AdamW(model.parameters()), lm_loss, lm_batch(), updated=())
    assert first.digest == g.digest and first.macs == g.macs
    assert macs_of(first, "updated") == 0 and macs_of(first, "producing") == macs_of(g, "producing") > 0
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


# --- consecutive steps, one measured -----------------------------------------------------------------------------------

def standard(m, batch, opts):
    for o in opts:
        o.zero_grad(set_to_none=True)
    lm_loss(m, batch).backward()
    for o in opts:
        o.step()


def top_block():
    model = TinyLM(V, H, NH, I, L)
    model.requires_grad_(False)
    model.blocks[-1].requires_grad_(True)
    return model, torch.optim.AdamW(model.blocks[-1].parameters())


@pytest.mark.slow
def test_a_step_among_its_neighbours_is_the_step_alone_with_its_circuit():
    """Three standard steps training the top block under AdamW, the second measured. Its GEMMs and their roles, its
    updated MACs, its cut and its raw product rows are the step's captured alone. What the step alone lacks, it has:
    a step circuit, on which the gradient's cut and one token group's partial are the block's dW at the accumulator's
    32 bits an element, the carried values no token op reads are AdamW's two moments, the update's profile ends at the
    block's weights at their 32 bits, and a token reads its id and its label."""
    model, opt = top_block()
    alone = capture_step(model, opt, lm_loss, lm_batch())
    assert alone.circuit is None and alone.formalism is None
    g = capture_steps([standard] * 3, model, lm_batch(), measured=1, optimizers=opt)
    assert roles(g) == roles(alone) and g.structure()["updated"] == alone.structure()["updated"]
    assert (g.tokens, g.axes, g.cut_bits, g.beta) == (alone.tokens, alone.axes, alone.cut_bits, alone.beta)
    assert products.raw_on_dial(g) == products.raw_on_dial(alone)
    weights = 4 * H + 4 * H2 + 2 * HI   # the block's two norms, q k v o, up and down
    f = g.formalism
    assert g.circuit is not None and f["region"] == 1
    assert f["grad_cut_bits"] == f["partial_cut_bits"] == f["u1_bits"] == 32 * weights
    assert f["state_bits"] == 2 * 32 * weights and f["delta_grad_bits"] == 3 * 32 * weights
    assert (f["tokens"], f["token_factor"], f["tau_ids"], f["tau_other_bits"]) == (T, 1, 2, 0)
    assert 0 < f["affected_share"] < f["sigma"] < 1   # the frozen block's token work reads no update


def test_the_first_of_the_steps_reads_no_update():
    """Measured at the first step, nothing it reads is the run's own: no affected work and no update to read. A step
    that isn't one of them is refused."""
    model, opt = top_block()
    f = capture_steps([standard] * 3, model, lm_batch(), measured=0, optimizers=opt).formalism
    assert f["region"] == 0 and f["affected_share"] == 0 and f["u1_bits"] == 0
    with pytest.raises(ValueError, match="measured is 3"):
        capture_steps([standard] * 3, model, lm_batch(), measured=3, optimizers=opt)


def narrowed(m, batch, _opts):
    """A step whose update passes through 16 bits on its way into the weights, as an orthogonalised update does."""
    for p in m.parameters():
        p.grad = None
    lm_loss(m, batch).backward()
    with torch.no_grad():
        for p in m.blocks[-1].parameters():
            p.add_(p.grad.bfloat16().float(), alpha=-0.01)


def widths(g) -> dict:
    """The width each operand of the measured step's affected products is passed at, by product."""
    return {(r["phase"], r["op"], tuple(sorted(r["dims"].items()))): tuple(r["widths"])
            for r in products.raw_on_dial(g)["rows"] if any(r["affected"])}


def test_steps_taken_mid_run_read_the_runs_own_weights():
    """A step whose update is cast to 16 bits before it is added to a 32-bit weight. Among a run's first steps the
    measured step's weight is θ plus that update, which a block makes from the update alone, so the weight is passed
    at the update's 16 bits. Mid-run the first captured step's weight is the run's own, and a block needs it at its 32
    bits. The step's circuit is the same either way: the gradient's cut is at the cast, 16 bits an element."""
    model, _ = top_block()
    first = capture_steps([narrowed] * 3, model, lm_batch(), measured=1)
    late = capture_steps([narrowed] * 3, model, lm_batch(), measured=1, mid_run=True)
    by_weight = {k: (a, b) for (k, a), b in zip(widths(first).items(), widths(late).values()) if a != b}
    assert by_weight and set(by_weight.values()) == {((32, 16), (32, 32))}
    assert late.formalism == first.formalism and late.formalism["grad_cut_bits"] == 16 * (4 * H + 4 * H2 + 2 * HI)


def mlp_weights(m) -> list:
    return [m.blocks[-1].up.weight, m.blocks[-1].down.weight]


def momentum(through_copy: bool, nesterov: bool):
    """A momentum step written by hand: each trained matrix's fp32 buffer is scaled and takes the gradient in place,
    and the update, cast to 16 bits, reads the gradient and the buffer (`nesterov`) or the buffer alone. With
    `through_copy` the buffer is then written through a copy of itself, as a step that keeps its buffer at any width
    writes it: ``b.copy_(b.float()...)``, where an fp32 buffer's ``float()`` is the buffer."""
    def step(m, batch, _opts):
        for p in m.parameters():
            p.grad = None
        lm_loss(m, batch).backward()
        with torch.no_grad():
            for i, p in enumerate(mlp_weights(m)):
                b = m.get_buffer(f"momentum{i}")
                buf = b.float().mul_(0.9).add_(p.grad)
                if through_copy:
                    buf = b.copy_(buf).float()
                u = p.grad.add(buf, alpha=0.9) if nesterov else buf
                p.add_(u.bfloat16().float(), alpha=-0.01)
    return step


@pytest.mark.parametrize("nesterov", [True, False])
def test_a_buffer_written_through_a_copy_of_itself_is_carried(nesterov):
    """A copy's source that is the storage it writes is read, so a momentum buffer written through a copy of itself
    keeps what the step scaled and added the gradient into. The measured step reads the buffer as carried, at its 32
    bits an element; the gradient's cut and one token group's partial are the gradient at 32 bits an element, since
    the buffer takes it whole, past the update's 16-bit cast; and the step's circuit and product rows are those of
    the same step writing the buffer in place."""
    def captured(through_copy: bool):
        model = TinyLM(V, H, NH, I, 1)
        model.requires_grad_(False)
        for i, p in enumerate(mlp_weights(model)):
            p.requires_grad_(True)
            model.register_buffer(f"momentum{i}", torch.zeros_like(p))
        return capture_steps([momentum(through_copy, nesterov)] * 3, model, lm_batch(), measured=1)

    copied, in_place = captured(True), captured(False)
    weights = 2 * HI   # the one block's up and down
    f = copied.formalism
    assert f["state_bits"] == f["grad_cut_bits"] == f["partial_cut_bits"] == f["u1_bits"] == 32 * weights
    assert f["delta_grad_bits"] == 2 * 32 * weights
    assert f == in_place.formalism and products.raw_on_dial(copied) == products.raw_on_dial(in_place)


# --- a tensor an optimizer holds and the model doesn't own -------------------------------------------------------------

#: where a step that reaches its latents around its arguments finds them
BESIDE = {}


def with_latents(where: str):
    """One block's MLP matrices trained through latents: each latent takes its matrix's gradient under SGD, and the
    matrix moves by the latent's sign. `where` is who holds the latents and how the step reaches them: the model owns
    them as parameters (``owned``); or it doesn't, the optimizer holds them, and the step reads them from a plain
    attribute of the model (``attribute``), from the optimizer's param groups (``optimizer``) or from this module's
    global (``global``)."""
    model = TinyLM(V, H, NH, I, 1)
    model.requires_grad_(False)
    for p in mlp_weights(model):
        p.requires_grad_(True)
    latents = [torch.zeros_like(p).requires_grad_(True) for p in mlp_weights(model)]
    if where == "owned":
        model.latents = torch.nn.ParameterList(torch.nn.Parameter(v) for v in latents)
        latents = list(model.latents)
    elif where == "attribute":
        model.latents = latents
    BESIDE["latents"] = latents

    def step(m, batch, opts):
        vs = opts[0].param_groups[0]["params"] if where == "optimizer" else BESIDE["latents"] if where == "global" \
            else m.latents
        lm_loss(m, batch).backward()
        for v, p in zip(vs, mlp_weights(m)):
            v.grad, p.grad = p.grad, None
        opts[0].step()
        opts[0].zero_grad(set_to_none=True)
        with torch.no_grad():
            for v, p in zip(vs, mlp_weights(m)):
                p.add_(torch.sign(v), alpha=-0.01)

    return model, torch.optim.SGD(latents, lr=0.1), latents, step


@pytest.mark.parametrize("where", ["attribute", "optimizer"])
def test_a_tensor_an_optimizer_holds_beside_the_model_is_traced_as_one_the_model_owns(where):
    """A latent the optimizer steps and the model doesn't own is copied for the trace like a parameter: the step's
    circuit and its product rows are those of the same step with the latents the model's own parameters, the
    latents are among the parameters the step trains, by the name of their place in the optimizer, and carried from
    step to step at their 32 bits an element, read by no token op. The run's own latents are never touched."""
    def captured(how):
        model, opt, latents, step = with_latents(how)
        return capture_steps([step] * 3, model, lm_batch(), measured=1, optimizers=opt), latents

    (g, latents), (owned, _) = captured(where), captured("owned")
    assert g.formalism == owned.formalism and products.raw_on_dial(g) == products.raw_on_dial(owned)
    assert g.formalism["state_bits"] == 32 * 2 * HI
    trained = {p["name"] for p in g.params if p["trained"]}
    assert trained == {"blocks.0.up.weight", "blocks.0.down.weight", "optimizers.0.params.0", "optimizers.0.params.1"}
    assert {p["name"] for p in owned.params if p["trained"]} == trained - {"optimizers.0.params.0",
                                                                           "optimizers.0.params.1"} | {"latents.0", "latents.1"}
    assert all(v.grad is None and not v.any() for v in latents)


def test_a_step_that_reads_the_runs_own_tensor_around_its_arguments_is_refused_with_the_fix():
    """A step is traced on a copy of the model and of the optimizers it is handed. One that reaches a latent by a
    global reads the run's own tensor, which the copy's optimizer never steps: the graph would be a step that
    trains nothing. So does one that closes over the model. Each is refused, by the tensor's name, with the fix."""
    model, opt, _, step = with_latents("global")
    with pytest.raises(CaptureRefused) as e:
        capture_steps([step] * 3, model, lm_batch(), measured=1, optimizers=opt)
    assert e.value.reason == "trace" and "the run's own optimizers.0.params.0 and not the copy" in str(e.value)
    assert "hold it on the model as a parameter" in str(e.value) and "param_groups" in str(e.value)
    with pytest.raises(CaptureRefused, match="the run's own emb.weight and not the copy"):
        capture_steps([lambda m, batch, opts: lm_loss(model, batch).backward()], model, lm_batch(), measured=0)


def test_an_optimizer_whose_state_was_loaded_is_copied_and_its_step_traced_as_before_the_load():
    """torch's `load_state_dict` adds a key of its own to an optimizer's `defaults` (``differentiable``), which a
    constructor that names its keywords refuses: the copy passes the class only the defaults it takes, so the step of
    a job continued from a kept state is traced as the unbroken job's is."""
    model = mlp()
    opt = Muon([p for p in model.parameters() if p.dim() == 2], lr=0.02)
    loss, batch = (lambda m, b: F.mse_loss(m(b[0]), b[1])), (torch.empty(6, 8), torch.empty(6, 4))
    before = capture_step(model, opt, loss, batch, token_dims=[(0,), (0,)])
    opt.load_state_dict(opt.state_dict())
    assert "differentiable" in opt.defaults
    after = capture_step(model, opt, loss, batch, token_dims=[(0,), (0,)])
    assert roles(after) == roles(before) and after.trained == before.trained
    assert products.raw_on_dial(after) == products.raw_on_dial(before)


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
        capture_steps([step], model, (torch.empty(4, 8),), measured=0)
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
        capture_step(model, torch.optim.SGD(model.parameters(), lr=0.1),
                     lambda m, b: torch.ops.te_test.opaque_faked(m(b[0])).sum(), (torch.empty(4, 8),))
    assert e.value.reason == "custom_kernel" and e.value.op == "te_test.opaque_faked.default"

    def forward_only(m, b, _o):
        with torch.no_grad():
            torch.ops.te_test.opaque_faked(m(b[0]))

    with pytest.raises(CaptureRefused) as e:   # in the graph: its products are not ATen ops
        capture_steps([forward_only], model, (torch.empty(4, 8),), measured=0)
    assert e.value.reason == "custom_kernel" and e.value.op == "te_test.opaque_faked.default"


def test_refuses_cond():
    model = mlp()

    def loss(m, batch):
        h = m(batch[0])
        return torch.cond(h.sum() > 0, lambda x: x * 2, lambda x: x * 3, (h,)).sum()

    with pytest.raises(CaptureRefused) as e:
        capture_step(model, torch.optim.SGD(model.parameters(), lr=0.1), loss, (torch.empty(4, 8),))
    assert e.value.reason == "dynamic_control_flow", e.value


def test_refuses_batch_dependent_graph():
    model = mlp()

    def loss(m, b):
        return sum(m(b[0][i:i + 1]).sum() for i in range(b[0].shape[0]))   # one forward per example

    with pytest.raises(CaptureRefused) as e:
        capture_step(model, torch.optim.SGD(model.parameters(), lr=0.1), loss, (torch.empty(4, 8),))
    assert e.value.reason == "batch_dependent"


@pytest.mark.parametrize("what", ["model", "parameter", "optimizer"])
def test_refuses_a_deep_copy_of_what_the_trace_runs_on_and_leaves_every_other_deep_copy_alone(what):
    """A step that deep-copies the model, one of its parameters or an optimizer is refused at that line, with what
    to write instead, before torch copies a tensor that has no data (which kills the process later). A step that
    copies the state dict is traced. The refusal is in force only while a capture records: torch's own deep copy is
    back after a capture that was refused and after one that was taken, and copies the model as before."""
    torchs = torch.Tensor.__deepcopy__
    model = mlp()
    batch, opts = (torch.empty(4, 8),), [torch.optim.SGD(model.parameters(), lr=0.1)]

    def train(m, rows, optimizers):
        m(rows[0]).sum().backward()
        optimizers[0].step()
        optimizers[0].zero_grad(set_to_none=True)

    def copies(m, rows, optimizers):
        copy.deepcopy({"model": m, "parameter": m[0].weight, "optimizer": optimizers[0]}[what])
        train(m, rows, optimizers)

    def keeps(m, rows, optimizers):
        was = {k: v.detach().clone() for k, v in m.state_dict().items()}
        train(m, rows, optimizers)
        assert set(was) == set(m.state_dict())

    def copied() -> None:
        assert torch.Tensor.__deepcopy__ is torchs
        twin = copy.deepcopy(model)
        assert [k for k, v in twin.state_dict().items() if not torch.equal(v, model.state_dict()[k])] == []
        assert all(a is not b for a, b in zip(twin.parameters(), model.parameters()))

    with pytest.raises(CaptureRefused) as e:
        capture_steps([copies] * 3, model, batch, measured=1, optimizers=opts)
    assert e.value.reason == "deep_copy" and e.value.op is None and "in copies: copy.deepcopy(" in e.value.where
    assert "copy the state dict instead, {k: v.detach().clone() for k, v in model.state_dict().items()}" in str(e.value)
    copied()
    assert capture_steps([keeps] * 3, model, batch, measured=1, optimizers=opts).tokens == 4
    copied()
