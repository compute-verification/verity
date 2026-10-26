"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.rmsnorm`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """CMP-92's two phases at the loop's EPS, 1e-5: Phase 1's row (width 128) as one unit, and Phase 2's (width 16,384) in the
    two-level form, chunks of 128 words and groups of 64 chunk sums, so G = 2; and CIR-109's fused output unit at Phase 2's
    chunk (C = 128, its 128 chunk sums in G = 2 groups), with the chunk sums 32 a unit (C = 4096, K = 128)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import rmsnorm as R

    return [bind(R.RmsNormRow, N=128, EPS=1e-5), bind(R.RmsSumSqPart, C=128), bind(R.F32SumPart, G=64),
            bind(R.RmsNormRecip, N=16384, G=2, EPS=1e-5), bind(R.RmsNormOut, C=128),
            bind(R.RmsNormOutSums, C=128, S=128, G=2, N=16384, EPS=1e-5), bind(R.RmsSumSqChunks, C=4096, K=128)]
