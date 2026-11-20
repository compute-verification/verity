"""property/v1 (`verity.circuits.stacking`): the quantizer's range rule on vLLM's evaluator."""

from __future__ import annotations

import random

from verity.circuits import stacking as P

from verity_vllm.definitions.registry.boolean_gate_ops import f32_to_u32_rz_byte


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
