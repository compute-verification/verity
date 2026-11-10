"""torch's flash attention forward at cc 12.0 (`rules.training_attention.FlashSdpa`): at each shape `fa2_train.TRAIN_SDPA`
lists the op is one batch over the B * H heads of the head torch's kernels compute there, its O and LSE each head's in order; a
shape the capture doesn't cover, a non-causal call, dropout, another scale and a read of the backward's state are refused; only
the training rules at the capture's cc hold the rule."""
from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tests.program.test_train_step import CC, _bits, _data, _meta, _program
from verity.circuits.evaluation.reference import evaluate
from verity.definitions.numerics.serve3_reference import materialize
from verity.definitions.registry import fa2_train as FT
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.rules.training import training_rules
from verity_vllm.program.frontend.rules.training_attention import FlashSdpa

aten = torch.ops.aten


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


@pytest.mark.parametrize("shape", sorted(FT.TRAIN_SDPA))
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


def test_without_a_cc_flash_attention_is_not_lowered() -> None:
    assert any(isinstance(r, FlashSdpa) for r in training_rules(cc=CC).rules)
    assert not any(isinstance(r, FlashSdpa) for r in training_rules().rules)
    shape = (2, 12, 1024, 64)
    assert _refusal(_sdpa(0.0, True), _meta(shape, shape, shape), cc=None).reason != "identity"
    with pytest.raises(ValueError, match="no captured flash attention"):
        FlashSdpa((9, 0))
