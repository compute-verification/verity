"""property/v1 (`verity.ml.circuits.stacking`) on real bf16 Programs, the weights as the registered store: Llama-3.2-1B's step (the
traced Build `test_predict` pins) and a small request (a prefill of 3 rows, then a decode step), and Qwen3-30B-A3B's step (a
mixture of experts). Each is refused for its bf16 tensor-core steps, every served MAC in one: a rounding of 544 operand bits,
past B (``operand-bits``), and budgeted with the rest of its units by class. A change to a Definition these Programs call moves
the numbers with the Program's digest.
"""

from __future__ import annotations

import json
import random

import pytest

from tests.program.predict.test_predict import FIX, _phi
from verity.ml.circuits import stacking as P
from verity.ml.circuits.census import census, step_ends
from verity.ml.circuits.program import Program
from verity_vllm.program.predict.build import build_program
from verity.ml.definitions.registry.dsv4_qnorm_insert import f32_to_u32_rz_byte
from verity_vllm.query.required import SAMPLING_EVENTS

WGMMA = "HopperBF16WgmmaDot16_v1"


def _analysis(model: str, wrapper: str, **kw) -> tuple:
    r = build_program(_phi(json.loads((FIX / f"{model}.config.json").read_text())), wrapper, **kw)
    program = Program(r.program.fn, registered=("weights",))
    c = census(program, step_ends(program, SAMPLING_EVENTS), r.digest)
    return r, c, P.analyze(program, c)


def test_the_quantizers_range_rule_holds_on_its_evaluator():
    """`F32ToU32RzByte`'s rule (the truncation below 128.0; 0 for a NaN, -0 or negative float) on vLLM's evaluator, over its
    boundaries and sampled operand ranges; at 128.0 or above the low byte isn't monotone, and the rule gives none."""
    rule, rng = P.RANGES["F32ToU32RzByte_v1"], random.Random(0)
    edges = [P.S32[0], -0x00800000, -1, 0, 1, 0x007FFFFF, 0x3F7FFFFF, 0x3F800000, 0x42FE0000, 0x42FEFFFF, 0x42FFFFFF]
    for lo in edges:
        for hi in edges:
            if lo > hi:
                continue
            out = rule((lo, hi))
            for x in [lo, hi] + [rng.randint(lo, hi) for _ in range(200)]:
                assert out[0] <= f32_to_u32_rz_byte(x & 0xFFFFFFFF) <= out[1], (lo, hi, hex(x), out)
    assert rule((0, 0x43000000)) is None and rule(None) is None and rule((0x3F800000, 0x42FE0000)) == (1, 127)


@pytest.mark.parametrize("model,wrapper,kw,digest,steps,affine,units,lifetime", [
    ("llama32-1b", "step", dict(M=1, pos0=8), "5e34739ca3f59dda", 1, 135168,
     {"roundings": 78513484, "compare_select": 385277, "gathers": 17, "folds": 0}, 1),
    ("llama32-1b", "request", dict(LP=3, T=2), "cdf004e855f279a4", 3, 665468,
     {"roundings": 351786258, "compare_select": 1570701, "gathers": 85, "folds": 0}, 2),
    ("qwen3-30b-a3b", "step", dict(M=1, pos0=8), "d12a2b4c96ad1b35", 1, 1632256,
     {"roundings": 197326508, "compare_select": 1041597, "gathers": 3090, "folds": 0}, 1),
])
def test_a_bf16_program_is_refused_for_its_tensor_core_steps(model, wrapper, kw, digest, steps, affine, units, lifetime):
    """Roundings: each tensor-core step is one, beside the FP elementwise ops; the bf16 widenings are free affine maps.
    Gathers: 17 index bits a token row's embedding (V = 128256, or out of range) in Llama, five rows in the request; Qwen's 18
    (V = 151936) and 48 layers x 8 routed experts x 8 bits (one of 128, or out of range); each reads a registered column, so
    none pays the selection unit. The request's run input is its 3 prompt tokens, first read in step 0, and its longest-held
    value lives 2 steps."""
    r, c, a = _analysis(model, wrapper, **kw)
    assert r.digest.startswith(digest)
    v = P.judge(a, c=dict.fromkeys(P.BUDGETED, 0), windows=[P.Window(range(steps), 1, 1)])
    assert (a.affine, v.numbers["units"]) == (affine, units), (a.affine, v.numbers["units"])
    g = a.gates[WGMMA]
    assert a.refused == {} and (g["class"], g["operand_bits"]) == ("roundings", 32 + 2 * 16 * 16)
    assert 16 * g["instances"] == c["totals"]["served_macs"]
    assert (a.family, a.unproven) == (0, {})
    assert len(a.steps) == steps and a.max_lifetime == lifetime
    assert [s["input_bits"] for s in a.steps] == [32 * (3 if wrapper == "request" else 1)] + [0] * (steps - 1)
    assert [x["code"] for x in v.reasons] == ["operand-bits", "roundings", "compare_select", "gathers"]
    assert v.reasons[0]["primitives"] == {WGMMA: 544}
    assert not v.ok
