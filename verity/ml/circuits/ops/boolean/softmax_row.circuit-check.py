"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.softmax_row`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The fused unit on the H200's step at D = 64, at CMP-92's longest attention row, T = 32 (its context, in both phases), and
    at T = 1, a request's first position; and the softmax alone at N = 1 (at N = 32 it is inside the fused unit's T = 32)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.ops.boolean import softmax_row as SR

    return [bind(SR.SoftmaxRow, N=1)] + [bind(SR.QkSoftmaxRow, T=t, D=64, **BG.GEMM_HOPPER) for t in (1, 32)]
