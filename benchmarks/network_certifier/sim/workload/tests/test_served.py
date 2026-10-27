"""verity_vllm's serving Program as the workload: it is `Serve_v1` with registered weights, a pipeline placement's crossings
are what a stage boundary of a decoder carries, and the blocks serve it in memory."""
import asyncio

import pytest

pytest.importorskip("torch")
pytest.importorskip("verity_vllm")

from workload import placement as P
from workload import served as S
from workload.block import Context, MemoryNet
from workload.hardware import OUTSIDE, mock

CFG = S.Config()


@pytest.fixture(scope="module")
def model():
    return S.load(CFG)


def test_the_program_is_serve_v1_with_registered_weights(model):
    assert model.report.family["family"].startswith("Serve_v1{")
    kinds = {model.kind[g] for g in model.input_gates(1)}
    assert kinds == {"registered"} and {model.kind[g] for g in model.input_gates(0)} == {"run"}
    assert len(model.input_gates(0)) == CFG.prompt and len(model.outputs) == CFG.steps + 1
    rounds = [c.request for c in model.calls]
    assert rounds == sorted(rounds) and set(rounds) == set(range(CFG.steps + 1))
    for r in range(CFG.steps + 1):
        layers = [c.layer for c in model.calls if c.request == r]
        assert layers == sorted(layers) and layers[0] == 0 and layers[-1] == CFG.layers - 1
    tokens = model.reference(0)
    assert len(tokens) == CFG.steps + 1 and all(0 <= t < CFG.vocab for t in tokens)


def test_a_stage_boundary_carries_the_hidden_state_and_residual(model):
    """Per boundary and token, the hidden state and the residual (2·H bf16 words); each round's token goes to the client
    and, but for the last, back to the first stage. The registered crossings are the tied embedding, which the LM head
    reads on the last stage, and the rotary rows of the positions served, which every later stage reads."""
    blocks = 3
    hw = mock(blocks, 1)
    p = P.place(model, hw, P.pipeline(model, hw))
    assert P.causal(model, p, hw) == []
    edges = P.crossings(model, p, hw)
    by_kind: dict[str, list[P.Edge]] = {}
    for e in edges:
        by_kind.setdefault(model.kind[e.value], []).append(e)
    tokens = CFG.prompt + CFG.steps
    activations = [e for e in by_kind["computed"] if e.width == 16]
    assert len(activations) == 2 * CFG.hidden * tokens * (blocks - 1)
    assert {(e.src, e.dst) for e in activations} == {("b0", "b1"), ("b1", "b2")}
    chosen = [e for e in by_kind["computed"] if e.width == 32]
    assert len(chosen) == 2 * CFG.steps + 1 and {e.src for e in chosen} == {"b2"}
    assert len(by_kind["run"]) == CFG.prompt
    assert len(by_kind["registered"]) == CFG.vocab * CFG.hidden + tokens * CFG.head_dim * (blocks - 1)


@pytest.mark.slow
def test_the_blocks_serve_it_and_answer_as_the_reference_does():
    config = {"workload": "served", "spec": CFG.to_json(), "blocks": 3, "devices": 1, "frame_bytes": 1024, "seed": 2}
    ctx = Context(config)
    results = asyncio.run(MemoryNet(ctx).run())
    assert results[OUTSIDE]["answers"] == ctx.model.reference(2)
    assert all(results[b]["values"] is not None for b in ctx.hw.units)
