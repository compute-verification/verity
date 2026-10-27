"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.norm_quant`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """Both fused blocks at the frozen shape's norm statics (S = 128 chunk sums in G = 2 groups, N = 16,384, EPS = 1e-5) on
    parts of C = 8 words: each callee is one small circuit called a word at a time, so a frozen-width part is the same circuits
    called more often, and its trace (4-7 KB a gate) would be far past the cap."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import norm_quant as NQ

    return [bind(d, C=8, S=128, G=2, N=16384, EPS=1e-5) for d in (NQ.RmsNormAbsMax, NQ.RmsNormQuantChunk)]
