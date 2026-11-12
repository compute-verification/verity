"""torch's flash attention forward at cc 12.0 (`rules.training_attention.FlashSdpa`): at each shape `fa2_train.TRAIN_SDPA`
lists the op is one batch over the B * H heads of the head torch's kernels compute there, its O and LSE each head's in order; a
shape the capture doesn't cover, a non-causal call, dropout, another scale and a read of the backward's state outside the op's
own backward are refused; only the training rules at the capture's cc hold the rule; and a step traced with the op
(`export_step(..., fused_attention=True)`) holds it and its backward, the forward the head's Definition and the backward the
step's one refusal."""
from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from verity.circuits.evaluation.reference import evaluate
from verity.definitions.numerics.serve3_reference import materialize
from verity.definitions.registry import cublas_bf16 as CB
from verity.definitions.registry import fa2_train as FT

from tests.program.test_train_step import CC, TINY_STEP_ROUTES, _bits, _data, _meta, _program
from tests.program.train_step_lf import ADAMW, meta_step, square_loss
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.rules.training import training_rules
from verity_vllm.program.frontend.rules.training_attention import FlashSdpa
from verity_vllm.program.frontend.train_step import AdamW, StepSurvey, export_step, step_report

aten = torch.ops.aten
BACKWARD = "aten._scaled_dot_product_flash_attention_backward.default"


def _sdpa(*args, outputs=(0, 1), **kwargs):
    def f(q, k, v):
        r = aten._scaled_dot_product_flash_attention.default(q, k, v, *args, **kwargs)
        return tuple(r[i] for i in outputs)
    return f


def _refusal(f, xs, cc=CC) -> Unsupported:
    gm = make_fx(f, tracing_mode="fake")(*xs)
    u = derive_graph(gm, xs, mode="reference", rules=training_rules(cc=cc), name="T").unsupported
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
    """A graph that differentiates through the call is refused at the backward op, which no rule lowers, and not at the forward
    whose state that op takes; any other read of the state is refused at the forward."""
    shape = (2, 12, 1024, 64)

    def f(q, k, v, g):
        r = aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, True)
        return aten._scaled_dot_product_flash_attention_backward.default(g, q, k, v, r[0], r[1], None, None, shape[2], shape[2], 0.0,
                                                                         True, r[6], r[7])

    u = _refusal(f, _meta(shape, shape, shape, shape))
    assert u.reason == "numerics_unregistered" and BACKWARD in str(u), u
    assert "outside the op's own backward" in str(_refusal(_sdpa(0.0, True, outputs=(0, 6)), _meta(shape, shape, shape)))


def _attention_ops(gm: torch.fx.GraphModule) -> dict[str, int]:
    """How many of the graph's nodes are the flash attention op, its backward, and the math path's products."""
    names = {aten._scaled_dot_product_flash_attention.default: "flash",
             aten._scaled_dot_product_flash_attention_backward.default: "backward", aten.bmm.default: "bmm"}
    out = dict.fromkeys(names.values(), 0)
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target in names:
            out[names[n.target]] += 1
    return out


def test_fused_attention_is_the_gpus_op_forward_and_backward() -> None:
    """`export_step(..., gpu=True, fused_attention=True)` holds each bf16 attention call of the tiny step as the flash attention op
    and its backward as that op's, where the trace holds the math path's six products a layer otherwise: without the keyword, and
    with it in f32; the keyword without `gpu` is refused."""
    def ops(dtype, **kw) -> dict[str, int]:
        model, batch = meta_step("tiny", 2, ctx=16, vocab=64, matmul_dtype=dtype)
        return _attention_ops(export_step(model, square_loss, AdamW(**ADAMW), batch, **kw)[0])

    math = {"flash": 0, "backward": 0, "bmm": 12}
    assert ops(torch.bfloat16, gpu=True, fused_attention=True) == {"flash": 2, "backward": 2, "bmm": 0}
    assert ops(torch.bfloat16, gpu=True) == ops(torch.bfloat16) == ops(None, gpu=True, fused_attention=True) == math
    with pytest.raises(ValueError, match="pass gpu=True"):
        ops(torch.bfloat16, fused_attention=True)


def test_tiny_bf16_step_with_fused_attention_on_sm120(monkeypatch) -> None:
    """The tiny bf16 step on sm_120 traced with the flash attention op, at stand-in routes: it is refused at attention's backward,
    which has no rule, and nowhere else (its survey's holes are the two backward nodes), and its attention forward is the head's
    Definition at the stand-in route, once a head a layer, with none of the math path's f32 products or its softmax."""
    for key, route in TINY_STEP_ROUTES.items():
        monkeypatch.setitem(CB.TRAIN_ROUTES, key, route)
    monkeypatch.setitem(FT.TRAIN_SDPA, (2, 2, 16, 32), ("unsplit", 16))
    model, batch = meta_step("tiny", 2, ctx=16, vocab=64, matmul_dtype=torch.bfloat16)
    step = (model, square_loss, AdamW(**ADAMW), batch)
    exported = export_step(*step, gpu=True, fused_attention=True)
    u = step_report(*step, cc=CC, exported=exported).unsupported
    assert isinstance(u, Unsupported) and u.reason == "numerics_unregistered" and BACKWARD in str(u), u
    sv = step_report(*step, cc=CC, exported=exported, survey=True)
    assert isinstance(sv, StepSurvey) and sv.census is not None
    assert [h["op"] for h in sv.holes] == [BACKWARD] * 2
    classes = sv.census["classes"]
    assert classes[FT.train_sdpa(2, 2, 16, 32).id]["instances"] == 2 * 2 * 2
    assert not [c for c in classes if c.startswith(("LinearRef_v1", "SoftmaxRef_v1"))]


def test_without_a_cc_flash_attention_is_not_lowered() -> None:
    assert any(isinstance(r, FlashSdpa) for r in training_rules(cc=CC).rules)
    assert not any(isinstance(r, FlashSdpa) for r in training_rules().rules)
    shape = (2, 12, 1024, 64)
    assert _refusal(_sdpa(0.0, True), _meta(shape, shape, shape), cc=None).reason != "identity"
    with pytest.raises(ValueError, match="no captured flash attention"):
        FlashSdpa((9, 0))
