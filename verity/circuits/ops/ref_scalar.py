"""The reference-defined f32 scalars the top-p keep word's gates read: `F32Sub_v1` and `F32GtStrict_v1`, whose evaluator is
the definition."""
from __future__ import annotations

import numpy as np

from verity.circuits.defs import primitive
from verity.circuits.ops.fp32 import F32_NAN, bits_of, f32_is_nan, f32_of
from verity.circuits.ops.scalar import BIT, F32

REF_CONF = ("reference-defined; the evaluator is the definition; conformance vs torch CPU over the tested domain only: "
            "docs/data/ref-prims/ (tests/program/test_ref_prims.py)")


def _canon_nan(u: int) -> int:
    u &= 0xFFFFFFFF
    return F32_NAN if f32_is_nan(u) else u


@primitive("F32Sub", 1, [("a", F32), ("b", F32)], F32, conformance=REF_CONF)
def F32Sub(a: int, b: int) -> int:
    """FSUB: RN(a - b), IEEE RN-even, no ftz; NaN -> 0x7FC00000 (canonical, host-independent)."""
    with np.errstate(all="ignore"):
        return _canon_nan(bits_of(np.float32(f32_of(a)) - np.float32(f32_of(b))))


@primitive("F32GtStrict", 1, [("a", F32), ("b", F32)], BIT, conformance=REF_CONF)
def F32GtStrict(a: int, b: int) -> int:
    """`a > b` on f32 values (NaN compares false) — the f32 twin of `Bf16GtStrict_v1` for greedy argmax."""
    return 1 if f32_of(a) > f32_of(b) else 0


__all__ = ["REF_CONF", "F32Sub", "F32GtStrict"]
