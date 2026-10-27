"""`RMSNormFusedCuda_v2{N, EPS}` as parts and a combine, so that a norm reading many writer rows fits a unit's rows.

    RMSNormFusedCudaPart_v1{N, PARTS, PART}    (x: N / PARTS, res: N bf16) -> (r: N / PARTS bf16, s: NW / PARTS f32)
    RMSNormFusedCudaCombine_v1{N, EPS, PARTS}  (r: N bf16, s: NW f32, w: N bf16) -> (normed, res': N bf16)

The kernel (`b1.RMSNormFusedCuda`) runs T = min(N, 1024) threads, thread t owning elements t, t + T, ...; each of its
NW = ⌈T / 32⌉ warps reduces its lanes' partials by cub's shfl_down tree, and thread 0 adds the warp sums in order from warp
0. The residual add, the squares and a warp's tree read only that warp's elements, so they are cut by warps: part g of PARTS
owns warps [g·NW/PARTS, (g+1)·NW/PARTS), threads [g·E, (g+1)·E) with E = T / PARTS, and in each of the N / T strides c the
elements [c·T + g·E, c·T + (g+1)·E) (`part_ranges`). Its x is those elements of the norm's x, stride by stride, and its res
the norm's whole residual row; it returns r = bf16(x + res) on them, in its x's order, and its warps' sums. The combine
reads every part's r and s in part order, adds the NW sums from warp 0, and ends as the norm does: s = rsqrtf(var / N + EPS),
normed_j = bf16((f32(r_j) * s) * f32(w_j)), and res' = r in the norm's element order. Each gate is one of the norm's, on the
same operands in the same order (`b1`'s own warp tree and epilogue), so together they compute its two outputs bit for bit.

A unit reads a port of m writer rows as that port cut m ways, each row one writer row (`verity.ml.windows.interface`'s
`Layout.cut`), and holds at most 256 rows in and out. At Llama-3.2-1B's grain the norm after down_proj reads x as down_proj's
512 slices of 4 columns, 516 rows in all. At PARTS = 4 a part reads 128 of those slices and the residual row and writes 2
rows (131), and the combine reads 4 r rows, 4 s rows and w and writes 2 (11).

Refused at bind: N that is not a multiple of T (a part's elements would differ in count from stride to stride), and PARTS
above 1 that does not cut the threads into whole warps (T a multiple of 32·PARTS).
"""
from __future__ import annotations

from verity.ml.circuits.defs import composite
from verity.ml.circuits.refs import array_of, coll_concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import BF16, F32

WARP = 32
_CONF = "exact-model-tested: RMSNormFusedCuda_v2{N, EPS}'s gates on its operands, cut by warps into parts and a combine"


def geometry(N: int) -> tuple[int, int, int]:
    """The kernel's threads T = min(N, 1024), its strides N / T and its warps ⌈T / 32⌉."""
    T = min(N, b1.FUSED_RMS_BLOCK)
    return T, N // T, -(-T // WARP)


def refuse(S: dict) -> str | None:
    """Why the statics bound so far have no cut (module docstring), or None."""
    N, parts, part = S.get("N", b1.FUSED_RMS_BLOCK), S.get("PARTS", 1), S.get("PART", 0)
    if not (type(N) is int and N >= 1 and type(parts) is int and parts >= 1 and type(part) is int):
        return f"N and PARTS must be positive ints and PART an int (got N={N!r}, PARTS={parts!r}, PART={part!r})"
    T = min(N, b1.FUSED_RMS_BLOCK)
    if N % T:
        return f"N={N} is not a multiple of the kernel's {T} threads"
    if parts > 1 and T % (WARP * parts):
        return f"PARTS={parts} does not cut the kernel's {T} threads into whole warps of {WARP}"
    if not 0 <= part < parts:
        return f"PART={part} is not one of PARTS={parts}"
    return None


def part_ranges(N: int, PARTS: int, PART: int) -> list[tuple[int, int]]:
    """The norm's elements part ``PART`` of ``PARTS`` owns, ``[lo, hi)`` a stride, in the order its x and r hold them."""
    T, strides, _ = geometry(N)
    E = T // PARTS
    return [(c * T + PART * E, c * T + (PART + 1) * E) for c in range(strides)]


def _part_sig(S):
    n, w = S.N // S.PARTS, geometry(S.N)[2] // S.PARTS
    return (("x", Array(n, BF16)), ("res", Array(S.N, BF16))), Tuple(Array(n, BF16), Array(w, F32))


def _combine_sig(S):
    return ((("r", Array(S.N, BF16)), ("s", Array(geometry(S.N)[2], F32)), ("w", Array(S.N, BF16))),
            Tuple(Array(S.N, BF16), Array(S.N, BF16)))


@composite("RMSNormFusedCudaPart", 1, ["N", "PARTS", "PART"], _part_sig, conformance=_CONF, refuse=refuse)
def RMSNormFusedCudaPart(B, S, x, res):
    """Part ``PART``'s threads of `RMSNormFusedCuda_v2{N}`: ``r = bf16(x + res)`` on its elements, each thread's square of its
    first element and FFMA of each later one, and its warps' shfl_down trees. ``x`` holds its elements (`part_ranges`),
    ``res`` the whole residual row."""
    E = geometry(S.N)[0] // S.PARTS
    own = coll_concat(Array(S.N // S.PARTS, BF16), [res[lo:hi] for lo, hi in part_ranges(S.N, S.PARTS, S.PART)])
    r = B.batch(P.Bf16AddF2fp, x, own, axes=(0, 0))
    rf = B.batch(P.Bf16ToF32, r, axes=(0,))
    partials = []
    for t in range(E):
        acc = B.call(P.F32Mul, rf[t], rf[t])
        for j in range(t + E, S.N // S.PARTS, E):
            acc = B.call(P.F32Fma, rf[j], rf[j], acc)
        partials.append(acc)
    return tuple_of(r, array_of([b1._cub_block_reduce_sum(B, partials[k:k + WARP]) for k in range(0, E, WARP)]))


@composite("RMSNormFusedCudaCombine", 1, ["N", "EPS", "PARTS"], _combine_sig, conformance=_CONF, refuse=refuse)
def RMSNormFusedCudaCombine(B, S, r, s, w):
    """`RMSNormFusedCuda_v2{N, EPS}`'s end over its ``PARTS`` parts: ``r`` and ``s`` the parts' outputs in part order, the warp
    sums added from warp 0, then the norm's epilogue on r in its element order, which is also res'."""
    n = S.N // S.PARTS
    runs = []
    for g in range(S.PARTS):
        at = g * n
        for lo, hi in part_ranges(S.N, S.PARTS, g):
            runs.append((lo, at, hi - lo))
            at += hi - lo
    z = coll_concat(Array(S.N, BF16), [r[at:at + m] for _, at, m in sorted(runs)])
    var = s[0]
    for k in range(1, len(s)):
        var = B.call(P.F32Add, var, s[k])
    return b1._fused_rms_epilogue(B, S, z, B.batch(P.Bf16ToF32, z, axes=(0,)), var, w)


__all__ = ["RMSNormFusedCudaCombine", "RMSNormFusedCudaPart", "WARP", "geometry", "part_ranges", "refuse"]
