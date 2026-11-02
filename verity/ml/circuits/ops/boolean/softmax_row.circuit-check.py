"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.softmax_row`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The fused proof block on the H200's step at D = 64, at the e2e demo's longest attention row, T = 32 (its context, in both
    phases), and at T = 1, a request's first position; the softmax alone at N = 1 (at N = 32 it is inside the fused proof block's
    T = 32); the padded proof block at T = 32, M3's one attention statement for every key count; the head's whole attention at
    the frozen shape's statics (T = K = 32, D = 64), the ``pv`` fusion's proof block; and the masked softmax at N = 4,
    small enough to pin."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.ops.boolean import softmax_row as SR

    return ([bind(SR.SoftmaxRow, N=1)] + [bind(SR.QkSoftmaxRow, T=t, D=64, **BG.GEMM_HOPPER) for t in (1, 32)]
            + [bind(SR.QkSoftmaxRowPad, T=32, D=64, **BG.GEMM_HOPPER),
               bind(SR.QkSoftmaxPvPad, T=32, K=32, D=64, **BG.GEMM_HOPPER), bind(SR.SoftmaxRowPad, N=4)])
