"""`b1.RMSNormFusedCuda_v2{N, EPS}` cut at its residual add, for a norm whose x is more writer rows than a unit holds:
N / C `RMSNormFusedCudaAdd_v1{N, C, AT}` parts, each the add on C elements of x and the whole residual row, and one
`RMSNormFusedCudaTail_v1{N, EPS}`, v2's body after the add on the parts' r in order of AT.

Nothing in `b1.py` is modified: the tail's body is v2's lines after the add, on `b1`'s own reduce and epilogue, so the parts
and the tail emit v2's gates on the same operands in the same order (`verity/ml/definitions/tests/test_rmsnorm_fused_add.py`
holds them to v2 word for word and gate for gate)."""
from __future__ import annotations

from verity.ml.circuits.defs import composite
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import BF16

_FUSED_SPLIT_CONFORMANCE = ("exact-model-tested: `RMSNormFusedCuda_v2{N, EPS}` cut at its residual add; the N / C add parts and "
                            "the tail return its outputs word for word (`verity/ml/definitions/tests/test_rmsnorm_fused_add.py`)")


def _fused_rms_tail(B, S, r, w, eps=None):
    """`b1.RMSNormFusedCuda_v2`'s body after the residual add, on its sum ``r = bf16(x + res)``: each thread's squares, cub's
    block reduce and the epilogue. Returns (normed, r)."""
    N = S.N
    rf = B.batch(P.Bf16ToF32, r, axes=(0,))
    nthreads = min(N, b1.FUSED_RMS_BLOCK)
    partials = []
    for t in range(nthreads):
        acc = B.call(P.F32Mul, rf[t], rf[t])  # `variance = 0.0f; variance += x*x` -> fma(x, x, 0) == x*x (exact for bf16 x)
        for j in range(t + nthreads, N, nthreads):
            acc = B.call(P.F32Fma, rf[j], rf[j], acc)
        partials.append(acc)
    var = b1._cub_block_reduce_sum(B, partials)
    return b1._fused_rms_epilogue(B, S, r, rf, var, w, eps)


def _fused_rms_add_refuse(S: dict) -> str | None:
    """Why `RMSNormFusedCudaAdd_v1`'s statics bound so far name no part of the residual row, or None."""
    N, C, AT = (S.get(k) for k in ("N", "C", "AT"))
    if None in (N, C, AT):
        return None
    if not (all(type(v) is int for v in (N, C, AT)) and C >= 1 and N % C == 0 and 0 <= AT < N // C):
        return f"part AT={AT!r} of C={C!r} elements is not one of the N / C whole parts of a residual row of N={N!r}"
    return None


@composite("RMSNormFusedCudaAdd", 1, ["N", "C", "AT"],
           lambda S: ((("x", Array(S.C, BF16)), ("res", Array(S.N, BF16))), Array(S.C, BF16)),
           conformance=_FUSED_SPLIT_CONFORMANCE, refuse=_fused_rms_add_refuse)
def RMSNormFusedCudaAdd(B, S, x, res):
    """`RMSNormFusedCuda_v2{N}`'s residual add on its elements [AT·C, (AT+1)·C): ``r_i = bf16(x_i + res[AT·C + i])``, x the norm's
    x on those C elements and res its whole residual row. `RMSNormFusedCudaTail_v1` reads the N / C parts' r in order of AT."""
    return B.batch(P.Bf16AddF2fp, x, res[S.AT * S.C:(S.AT + 1) * S.C], axes=(0, 0))


@composite("RMSNormFusedCudaTail", 1, ["N", "EPS"],
           lambda S: ((("r", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Tuple(Array(S.N, BF16), Array(S.N, BF16))),
           conformance=_FUSED_SPLIT_CONFORMANCE)
def RMSNormFusedCudaTail(B, S, r, w):
    """`RMSNormFusedCuda_v2{N, EPS}`'s body after the residual add (`_fused_rms_tail`) on the sum r that its
    `RMSNormFusedCudaAdd_v1` parts write: returns (normed, r) as v2 returns (normed, res'), r the row it reads, with no gate."""
    return _fused_rms_tail(B, S, r, w)
