"""circuit-check's bindings of `verity.ml.definitions.registry.dense`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """Each vector Definition at N = 16 (the scalar ones at C = 0.5), GeGLU at I = 16, and soft-capped attention at CAP = 30: the
    frontend's `attention_softcap` instance (T = 5, two query heads on one KV head, D = 16, its one block at FA2's kBlockN of
    128), and one head over 17 keys in blocks of 16, whose second block is the first that isn't `FIRST`; with the two primitives."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import b1
    from verity.ml.definitions.registry import dense as DN

    return [
        *[bind(d, N=16) for d in (DN.SquareBf16, DN.SquareF32, DN.MeanTriton, DN.RsqrtF32, DN.AddWidenedBf16, DN.ScaleRowBf16,
                                  DN.ScaleRowF32, DN.MulVecF32, DN.NarrowF32ToBf16, DN.Bf16Tanh)],
        *[bind(d, N=16, C=0.5) for d in (DN.AddScalarF32, DN.AddScalarBf16, DN.Bf16DivScalar, DN.Bf16MulScalar)],
        bind(DN.GeluTanhMul, I=16),
        bind(DN.AttentionSoftcap, T=5, NH=2, KVH=1, D=16, BN=b1.fa2_kblock_n(16), CAP=30.0),
        bind(DN.AttentionHeadSoftcap, T=17, D=16, BN=16, CAP=30.0),
        DN.TanhF32Rn,
        DN.GeluTanhMulBf16,
    ]
