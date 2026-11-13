"""`tp_allreduce`'s MNNVL norm `RmsNormMnnvl_v1` cut where it broadcasts its scale, so that no unit returns a whole row.

    RmsNormMnnvlSumSq_v1{N,SHOT,CLUSTER,EPS,FMA}
                                              (r) -> f32   (the norm's scale rs, every normed element's one broadcast word)
    RmsNormMnnvlApply_v1{G,SHOT}              (r: Array<G, bf16>, w: Array<G, bf16>, rs: f32) -> Array<G, bf16>
                                              (the norm's G normed elements from rs: any G columns of the row)

Both are `tp_allreduce`'s own pieces of the norm (`_mnnvl_scale`, `_mnnvl_apply`), so the sum of squares and N / G apply units
return `RmsNormMnnvl_v1`'s words (`verity/definitions/tests/test_mnnvl_and_dsa_norm_splits.py`). Their Boolean versions are
`boolean_sigma`'s, `_v2`.
"""
from __future__ import annotations

from verity.circuits.defs import composite
from verity.definitions.ops.prims import ZERO32
from verity.circuits.types import Array
from verity.definitions.registry import prims as P
from verity.definitions.registry import tp_allreduce as TA
from verity.definitions.registry.prims import BF16, F32

__all__ = ["RmsNormMnnvlApply", "RmsNormMnnvlSumSq"]

_MNNVL_SPLIT = ("not-yet-established: `RmsNormMnnvl_v1` cut at its scale, the norm of AllReduceRmsNormMnnvl_v1 whose captures are "
                "exact-model-tested; the scale and N / G apply units return RmsNormMnnvl_v1's words "
                "(`verity/definitions/tests/test_mnnvl_and_dsa_norm_splits.py`)")


@composite("RmsNormMnnvlSumSq", 1, ["N", "SHOT", "CLUSTER", "EPS", "FMA"], lambda S: ((("r", Array(S.N, BF16)),), F32),
           conformance=_MNNVL_SPLIT)
def RmsNormMnnvlSumSq(B, S, r):
    """`RmsNormMnnvl_v1`'s scale on its own: ``rs = rsqrt(sum * rcp(N) + EPS)`` of r in the fused kernel's geometry, the one f32
    word each normed element reads."""
    TA._fused_geometry(int(S.N), int(S.CLUSTER))
    return TA._mnnvl_scale(B, S, B.call(ZERO32), r)


@composite("RmsNormMnnvlApply", 1, ["G", "SHOT"],
           lambda S: ((("r", Array(S.G, BF16)), ("w", Array(S.G, BF16)), ("rs", F32)), Array(S.G, BF16)),
           conformance=_MNNVL_SPLIT)
def RmsNormMnnvlApply(B, S, r, w, rs):
    """`RmsNormMnnvl_v1`'s normed elements from its scale rs, element by element: G columns of the row are this Definition at
    width G on those columns of r and w."""
    return TA._mnnvl_apply(B, TA.strategy(S.SHOT), B.call(ZERO32), B.batch(P.Bf16ToF32, r, axes=(0,)), w, rs)
