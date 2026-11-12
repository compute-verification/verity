"""torch's flash attention at cc 12.0, forward (`rules.training_attention.FlashSdpa`) and backward (`FlashSdpaBackward`): at each
shape `fa2_train.TRAIN_SDPA` lists the forward is one batch over the B * H heads of the head torch's kernels compute there, its
O and LSE each head's in order, and at each shape `fa2_train_bwd.TRAIN_SDPA_BWD` lists the backward, under deterministic
algorithms, one batch of its head Definition, its dQ, dK and dV each head's in order; a shape the capture doesn't cover, a
non-causal call, dropout, another scale and a read of the forward's state outside its own backward are refused, and the backward
too in a step not under deterministic algorithms and in a variable-length call; only the training rules at the capture's cc
hold the rules; and a bf16 step traced as the GPU runs it (`export_step(..., gpu=True)`) holds the op and its backward where
the trace holds the math path otherwise."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx
from verity.circuits.evaluation.reference import evaluate
from verity.definitions.numerics.serve3_reference import materialize
from verity.definitions.registry import fa2_train as FT
from verity.definitions.registry import fa2_train_bwd as FTB

from tests.program.test_train_step import (
    CC,
    TINY_SDPA,
    _bits,
    _data,
    _deterministic_algorithms,
    _meta,
    _program,
    _sum_of_squares,
    _tiny_stand_ins,
)
from tests.program.train_step_lf import ADAMW, meta_step, square_loss
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.rules.training import training_rules
from verity_vllm.program.frontend.rules.training_attention import FlashSdpa, FlashSdpaBackward
from verity_vllm.program.frontend.train_step import AdamW, StepSurvey, export_step, step_report

aten = torch.ops.aten
BACKWARD = "aten._scaled_dot_product_flash_attention_backward.default"
#: a stand-in shape of both kernels, two heads of two key blocks
SMALL = (1, 2, 32, 32)


def _sdpa(*args, outputs=(0, 1), **kwargs):
    def f(q, k, v):
        r = aten._scaled_dot_product_flash_attention.default(q, k, v, *args, **kwargs)
        return tuple(r[i] for i in outputs)
    return f


def _sdpa_backward(dropout=0.0, causal=True, max_q=None, **kwargs):
    """The causal forward on q, k and v, then its backward on the upstream gradient g with the given arguments."""
    def f(q, k, v, g):
        T = q.shape[2]
        r = aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, True)
        return aten._scaled_dot_product_flash_attention_backward.default(g, q, k, v, r[0], r[1], None, None, max_q or T, T,
                                                                         dropout, causal, r[6], r[7], **kwargs)
    return f


def _refusal(f, xs, cc=CC, **kw) -> Unsupported:
    gm = make_fx(f, tracing_mode="fake")(*xs)
    u = derive_graph(gm, xs, mode="reference", rules=training_rules(cc=cc, **kw), name="T").unsupported
    assert isinstance(u, Unsupported), u
    return u


@pytest.mark.parametrize("shape", [pytest.param(s, marks=pytest.mark.slow) if s == (2, 8, 1024, 64) else s for s in sorted(FT.TRAIN_SDPA)])
def test_flash_forward_is_one_batch_of_the_captured_head(shape) -> None:
    prog = _program(_sdpa(0.0, True, scale=0.125), *_meta(shape, shape, shape), rules=training_rules(cc=CC))
    calls = [(nd.fn.id, nd.n) for nd in prog.root.body.nodes if nd.fn.id.startswith("Fa2Train")]
    assert calls == [(FT.train_sdpa(*shape).id, shape[0] * shape[1])]


def test_flash_forward_wires_each_heads_o_and_lse(monkeypatch) -> None:
    """At two small shapes made routes for this test, one unsplit and one split, the Program's O and LSE are each head's
    Definition's on that head's q, k and v, in [B, H] order."""
    monkeypatch.setitem(FT.TRAIN_SDPA, (1, 2, 32, 16), ("unsplit", 16))
    monkeypatch.setitem(FT.TRAIN_SDPA, (2, 1, 64, 16), ("split", 16))
    for seed, shape in enumerate(((1, 2, 32, 16), (2, 1, 64, 16))):
        q, k, v = (_data(*shape, seed=11 + 3 * seed + i).to(torch.bfloat16) for i in range(3))
        prog = _program(_sdpa(0.0, True), q, k, v, rules=training_rules(cc=CC))
        tab = materialize(prog, [_bits(x) for x in (q, k, v)])
        got = [int(tab[g]) for g in prog.circuit.root_outputs()]
        B, H, T, D = shape
        head = FT.train_sdpa(*shape)
        o, lse = [], []
        for b in range(B):
            for h in range(H):
                y = [int(w) for w in evaluate(head, *(_bits(x[b, h]) for x in (q, k, v)))]
                o += y[:T * D]
                lse += y[T * D:]
        assert got == o + lse, shape


@pytest.mark.parametrize("why, f, shape, reason", [
    ("an uncaptured shape", _sdpa(0.0, True), (2, 4, 1024, 64), "numerics_unregistered"),
    ("non-causal", _sdpa(0.0, False), (2, 12, 1024, 64), "numerics_unregistered"),
    ("dropout", _sdpa(0.1, True), (2, 12, 1024, 64), "side_input"),
    ("another scale", _sdpa(0.0, True, scale=0.5), (2, 12, 1024, 64), "numerics_unregistered"),
    ("the backward's state", _sdpa(0.0, True, outputs=(0, 6)), (2, 12, 1024, 64), "numerics_unregistered"),
])
def test_flash_forward_refusals(why, f, shape, reason) -> None:
    u = _refusal(f, _meta(shape, shape, shape))
    assert u.reason == reason, (why, u)


def test_the_ops_own_backward_may_read_its_state() -> None:
    """A graph that differentiates through the call at a captured shape, its rules not under deterministic algorithms, is
    refused at the backward, with the fix, and not at the forward whose state the backward takes; any other read of the state is
    refused at the forward."""
    shape = (2, 12, 1024, 64)
    u = _refusal(_sdpa_backward(), _meta(shape, shape, shape, shape))
    assert u.reason == "side_input" and BACKWARD in str(u) and "torch.use_deterministic_algorithms(True)" in u.hint, u
    u = _refusal(_sdpa(0.0, True, outputs=(0, 6)), _meta(shape, shape, shape))
    assert "outside the op's backward" in u.hint and BACKWARD not in str(u), u


@pytest.mark.slow
@pytest.mark.parametrize("shape", sorted(FTB.TRAIN_SDPA_BWD))
def test_flash_backward_is_one_batch_of_the_captured_head(shape) -> None:
    """At each captured shape (a derivation of about 15 s and 1.5 GB each, the head's whole loop over its key blocks), the
    backward is one batch over the B * H heads of `train_sdpa_backward`'s head, after the forward's."""
    prog = _program(_sdpa_backward(), *_meta(shape, shape, shape, shape), rules=training_rules(cc=CC, deterministic=True))
    calls = [(nd.fn.id, nd.n) for nd in prog.root.body.nodes if nd.fn.id.startswith("Fa2Train")]
    assert calls == [(FT.train_sdpa(*shape).id, shape[0] * shape[1]), (FTB.train_sdpa_backward(*shape).id, shape[0] * shape[1])]


def test_flash_backward_wires_each_heads_dq_dk_dv(monkeypatch) -> None:
    """At a small shape made a route of both kernels for this test, the Program's dQ, dK and dV are each head's backward
    Definition's on that head's upstream gradient, q, k and v and the forward Definition's O and LSE, in [B, H] order. The
    scale is the composite's double 1 / sqrt(D), which differs from D^-0.5 in its last bit at D = 32 and is the same f32."""
    monkeypatch.setitem(FT.TRAIN_SDPA, SMALL, ("unsplit", 16))
    monkeypatch.setitem(FTB.TRAIN_SDPA_BWD, SMALL, (16, 16))
    B, H, T, D = SMALL
    scale = 1 / math.sqrt(D)
    assert scale != D ** -0.5
    q, k, v, g = (_data(*SMALL, seed=31 + i).to(torch.bfloat16) for i in range(4))
    prog = _program(_sdpa_backward(scale=scale), q, k, v, g, rules=training_rules(cc=CC, deterministic=True))
    tab = materialize(prog, [_bits(x) for x in (q, k, v, g)])
    got = [int(tab[w]) for w in prog.circuit.root_outputs()]
    forward, backward = FT.train_sdpa(*SMALL), FTB.train_sdpa_backward(*SMALL)
    want: list[list[int]] = [[], [], []]
    for b in range(B):
        for h in range(H):
            y = [int(w) for w in evaluate(forward, *(_bits(x[b, h]) for x in (q, k, v)))]
            z = [int(w) for w in evaluate(backward, _bits(g[b, h]), *(_bits(x[b, h]) for x in (q, k, v)), y[:T * D], y[T * D:])]
            for j in range(3):
                want[j] += z[j * T * D:(j + 1) * T * D]
    assert got == want[0] + want[1] + want[2]


@pytest.mark.parametrize("why, kw, captured, deterministic, reason", [
    ("not under deterministic algorithms", {}, True, False, "side_input"),
    ("an uncaptured shape", {}, False, True, "numerics_unregistered"),
    ("a variable-length call", {"max_q": SMALL[2] // 2}, True, True, "numerics_unregistered"),
    ("non-causal", {"causal": False}, True, True, "numerics_unregistered"),
    ("dropout", {"dropout": 0.1}, True, True, "side_input"),
    ("another scale", {"scale": 0.5}, True, True, "numerics_unregistered"),
])
def test_flash_backward_refusals(monkeypatch, why, kw, captured, deterministic, reason) -> None:
    """Each is refused at the backward, after the forward lowered."""
    monkeypatch.setitem(FT.TRAIN_SDPA, SMALL, ("unsplit", 16))
    if captured:
        monkeypatch.setitem(FTB.TRAIN_SDPA_BWD, SMALL, (16, 16))
    u = _refusal(_sdpa_backward(**kw), _meta(SMALL, SMALL, SMALL, SMALL), deterministic=deterministic)
    assert u.reason == reason and BACKWARD in str(u), (why, u)


def _attention_ops(gm: torch.fx.GraphModule) -> dict[str, int]:
    """How many of the graph's nodes are the flash attention op, its backward, and the math path's products."""
    names = {aten._scaled_dot_product_flash_attention.default: "flash",
             aten._scaled_dot_product_flash_attention_backward.default: "backward", aten.bmm.default: "bmm"}
    out = dict.fromkeys(names.values(), 0)
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target in names:
            out[names[n.target]] += 1
    return out


def test_gpu_export_takes_the_cards_attention_op() -> None:
    """`export_step(..., gpu=True)` holds each bf16 attention call of the tiny step as the flash attention op and its backward as
    that op's, where the trace holds the math path's six products a layer: without `gpu`, and with it in f32."""
    def ops(dtype, **kw) -> dict[str, int]:
        model, batch = meta_step("tiny", 2, ctx=16, vocab=64, matmul_dtype=dtype)
        return _attention_ops(export_step(model, square_loss, AdamW(**ADAMW), batch, **kw)[0])

    assert ops(torch.bfloat16, gpu=True) == {"flash": 2, "backward": 2, "bmm": 0}
    assert ops(torch.bfloat16) == ops(None, gpu=True) == {"flash": 0, "backward": 0, "bmm": 12}


class _TwoAttentions(torch.nn.Module):
    """Two causal bf16 attention calls of [1, 2, 16, 32], the first on the input plus a parameter, `between` run between them."""

    def __init__(self, between=lambda: None) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(1, 2, 16, 32, device="meta"))
        self.between = between

    def forward(self, x):
        y = (x + self.w).bfloat16()
        y = F.scaled_dot_product_attention(y, y, y, is_causal=True)
        self.between()
        return F.scaled_dot_product_attention(y, y, y, is_causal=True)


def test_gpu_export_reads_the_steps_deterministic_algorithms() -> None:
    """At a cc the deterministic algorithms the step's flash attention runs under are the rules' `deterministic`, as
    `export_step(..., gpu=True)` reads them (warn-only is not deterministic); without `gpu` the layout's is False; a step that
    changes them between its attention calls is refused."""
    batch, opt = torch.empty(1, 2, 16, 32, device="meta"), AdamW(**ADAMW)

    def layout(gpu=True, between=lambda: None):
        return export_step(_TwoAttentions(between), _sum_of_squares, opt, batch, gpu=gpu)[2]

    with _deterministic_algorithms():
        assert layout().deterministic and not layout(gpu=False).deterministic
    with _deterministic_algorithms(False):
        assert not layout().deterministic
        torch.use_deterministic_algorithms(True, warn_only=True)
        assert not layout().deterministic
    with _deterministic_algorithms(False), pytest.raises(ValueError, match="changes torch.use_deterministic_algorithms"):
        layout(between=lambda: torch.use_deterministic_algorithms(True))


def test_tiny_bf16_step_on_sm120_not_under_deterministic_algorithms(monkeypatch) -> None:
    """The tiny bf16 step on sm_120 at its stand-ins, not under deterministic algorithms: it is refused at attention's backward,
    with the fix, and nowhere else (its survey's holes are the two backward nodes), and its attention forward is the head's
    Definition, once a head a layer, with none of the math path's f32 products or its softmax."""
    _tiny_stand_ins(monkeypatch)
    model, batch = meta_step("tiny", 2, ctx=16, vocab=64, matmul_dtype=torch.bfloat16)
    step = (model, square_loss, AdamW(**ADAMW), batch)
    with _deterministic_algorithms(False):
        exported = export_step(*step, gpu=True)
    assert not exported[2].deterministic
    u = step_report(*step, cc=CC, exported=exported).unsupported
    assert isinstance(u, Unsupported) and u.reason == "side_input" and BACKWARD in str(u), u
    assert "torch.use_deterministic_algorithms(True)" in u.hint, u
    sv = step_report(*step, cc=CC, exported=exported, survey=True)
    assert isinstance(sv, StepSurvey) and sv.census is not None
    assert [h["op"] for h in sv.holes] == [BACKWARD] * 2
    classes = sv.census["classes"]
    assert classes[FT.train_sdpa(*TINY_SDPA).id]["instances"] == 2 * TINY_SDPA[0] * TINY_SDPA[1]
    assert not [c for c in classes if c.startswith(("LinearRef_v1", "SoftmaxRef_v1", "Fa2TrainBwd"))]


def test_without_a_cc_flash_attention_is_not_lowered() -> None:
    for rule in (FlashSdpa, FlashSdpaBackward):
        assert any(isinstance(r, rule) for r in training_rules(cc=CC).rules)
        assert not any(isinstance(r, rule) for r in training_rules().rules)
    shape = (2, 12, 1024, 64)
    assert _refusal(_sdpa(0.0, True), _meta(shape, shape, shape), cc=None).reason != "identity"
    for rule in (FlashSdpa, FlashSdpaBackward):
        with pytest.raises(ValueError, match="no captured flash attention"):
            rule((9, 0))
    with pytest.raises(ValueError, match="without a cc no rule reads it"):
        training_rules(deterministic=True)
