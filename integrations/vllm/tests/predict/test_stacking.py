"""property/v1 (`verity.ml.circuits.stacking`) on real bf16 Programs, the weights as the registered store: Llama-3.2-1B's step (the
traced Build `test_predict` pins) and a small request (a prefill of 3 rows, then a decode step), and Qwen3-30B-A3B's step (a
mixture of experts). Each is refused for its bf16 tensor-core steps, every served MAC in one: a rounding of 33 operands, past
fan-in f (``fan-in``), and budgeted with the rest of its units by class. A change to a Definition these Programs call moves the
numbers with the Program's digest.
"""

from __future__ import annotations

import json

import pytest

from tests.predict.test_predict import FIX, _phi
from verity.ml.circuits import stacking as P
from verity.ml.circuits.census import census, step_ends
from verity.ml.circuits.program import Program
from verity_vllm.predict.build import build_program
from verity_vllm.query.required import SAMPLING_EVENTS

WGMMA = "HopperBF16WgmmaDot16_v1"


def _analysis(model: str, wrapper: str, **kw) -> tuple:
    r = build_program(_phi(json.loads((FIX / f"{model}.config.json").read_text())), wrapper, **kw)
    program = Program(r.program.fn, registered=("weights",))
    c = census(program, step_ends(program, SAMPLING_EVENTS), r.digest)
    return r, c, P.analyze(program, c)


@pytest.mark.parametrize("model,wrapper,kw,digest,steps,units,lifetime", [
    ("llama32-1b", "step", dict(M=1, pos0=8), "773b545605765fe9", 1,
     {"roundings": 78581580, "compare_select": 385277, "gathers": 17, "products": 0}, 1),
    ("llama32-1b", "request", dict(LP=3, T=2), "6861bab5ba6ed6d2", 3,
     {"roundings": 359600254, "compare_select": 1161975, "gathers": 85, "products": 0}, 2),
    ("qwen3-30b-a3b", "step", dict(M=1, pos0=8), "98a180a562abf1d6", 1,
     {"roundings": 198675628, "compare_select": 906429, "gathers": 2706, "products": 0}, 1),
])
def test_a_bf16_program_is_refused_for_its_tensor_core_steps(model, wrapper, kw, digest, steps, units, lifetime):
    """Roundings: each tensor-core step is one, beside the FP elementwise ops. Gathers: 17 index bits a token row's embedding
    (V = 128256) in Llama, five rows in the request; Qwen's 18 (V = 151936) and 48 layers x 8 routed experts x 7 bits (one of
    128). The request's run input is its 3 prompt tokens, first read in step 0, and its longest-held value lives 2 steps."""
    r, c, a = _analysis(model, wrapper, **kw)
    assert r.digest.startswith(digest)
    g = a.gates[WGMMA]
    assert a.refused == {} and (g["class"], g["fan_in"], g["width"]) == ("roundings", 33, 32)
    assert 16 * g["instances"] == c["totals"]["served_macs"]
    assert (a.family, a.affine, a.unproven) == (0, 0, {})
    assert len(a.steps) == steps and a.max_lifetime == lifetime
    assert [s["input_bits"] for s in a.steps] == [32 * (3 if wrapper == "request" else 1)] + [0] * (steps - 1)
    v = P.judge(a, c=dict.fromkeys(P.BUDGETED, 0), windows=[P.Window(range(steps), 1, 1)])
    assert [x["code"] for x in v.reasons] == ["fan-in", "roundings", "compare_select", "gathers"]
    assert v.reasons[0]["primitives"] == {WGMMA: 33}
    assert not v.ok and v.numbers["units"] == units
