"""`registry.dense`'s RMSNorm chain on bits, the steps `boolean_dense`'s element-wise rows leave: the unfused norm a model runs
when its RMSNorm is `forward_native` (Gemma-2's `GemmaRMSNorm`), each Definition the next version of its word Definition, the
same function on bits.

    MeanTriton_v2{N}       (x: Array<N, F32>) -> Array<1, F32>            batch-invariant `mean_kernel`      word view: MeanTriton_v1{N}
    RsqrtF32_v2{N, RSQRT}  (x: Array<N, F32>) -> Array<N, F32>            rsqrt.approx.f32(x_j)              RsqrtF32_v1{N}
    NarrowF32ToBf16_v2{N}  (x: Array<N, F32>) -> Array<N, BF16>           cvt.rn.bf16.f32 (`F2fpBf16_v1`)    NarrowF32ToBf16_v1{N}

(F32 and BF16 are 32 and 16 bits.) Gemma-2-2B (H = 2304, EPS = 1e-6) calls SquareBf16, MeanTriton, AddScalarF32{1, EPS},
RsqrtF32{1} and ScaleRowBf16, then MulVecF32 by AddScalarBf16{H, 1.0} of the weight, then NarrowF32ToBf16; with a residual it
starts from AddWidenedBf16, squares and scales that f32 sum (SquareF32, ScaleRowF32), and narrows it once more as the residual.
The other steps' Boolean versions are `boolean_dense`'s (`SquareBf16_v2`, `SquareF32_v2`, `AddScalarF32_v2`, `AddScalarBf16_v2`,
`AddWidenedBf16_v2`, `ScaleRowBf16_v2`, `ScaleRowF32_v2`, `MulVecF32_v2`).

`NarrowF32ToBf16_v2` is a batch of a leaf traced from `fp`'s builder with F2FP's NaN word. `MeanTriton_v2` is the word
body's reduction in its order, cut into Calls a root can hold (`mean_triton_calls`):
`MeanThreadsTriton_v1{N}` (each thread's 8 registers added in register order, `MeanThreadTriton_v1{K}`),
`MeanWarpsTriton_v1{N}` (lane 0 of each warp's xor butterfly 16, 8, 4, 2, 1, `MeanWarpTriton_v1{L}`) and
`MeanCombineTriton_v1{N}` (each chunk's (w0 + w2) + (w1 + w3), the chunks added in order from +0, then the division). Masked
registers, idle lanes and idle warps hold +0 and the word body adds them: an add of +0 is `F32AddZero_v1` (x + 0 is not x on -0
or a NaN; 0 + x and x + 0 are one function of x), and +0 + +0, which is +0, is folded. The division is
`F32Mul_v1(DivFullScaleA_v1(acc, f32(N)), DivFullRcp_v1(f32(N)))`. `DivFullScaleA_v1` returns its numerator unchanged when the
divisor's magnitude is in [2^-126, 2^126], as f32(N) always is, and DivFullRcp_v1(f32(N)) is a constant, so the division is
`MeanDivTriton_v1{N}`, acc * rcp(N), and the mean reads no MUFU.

MUFU: ``RSQRT`` is a Definition computing `RsqrtApprox_v1`: proofs' Boolean `RsqrtApprox_v2` (`verity.ml.circuits.ops.boolean.mufu`, which
the switch binds through ``word_statics``), or the word id itself, on which each element is packed (`PackF32_v1`) and its result
unpacked (`UnpackF32_v1`) once.
"""
from __future__ import annotations

from functools import cache

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, array_of, coll_concat
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean import trace as TR
from verity_vllm.program.registry import dense as D
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.boolean_norms import (BF16, F32, PackF32, UnpackF32, _on_bits, _on_word, _rcp_n,
                                                        _traced, _word_view)

__all__ = ["MeanTriton", "NarrowF32ToBf16", "RsqrtF32", "mean_triton_calls"]

WARP = 32
#: f32 registers per thread per 1024-column chunk (`BLOCK_SIZE / THREADS`, either layout)
REGS = D.BLOCK_SIZE // D.THREADS


# --- the element leaves -------------------------------------------------------------------------------------------------------

def _one(name: str, params, build, doc: str):
    return _traced(name, 1, (), lambda S: (params, F32), lambda S: TR.trace(build, [t for _, t in params], F32), doc)


F32AddZero = _one("F32AddZero", (("x", F32),), lambda C, x: fp.f32_add(C, x, fp.f32_const(0), ir_nan=True),
                  "F32Add_v1(x, +0), which is F32Add_v1(+0, x): the mean's add of a masked register, idle lane or idle warp")
NarrowF32ToBf16El = _traced("NarrowF32ToBf16El", 1, (), lambda S: ((("x", F32),), BF16),
                            lambda S: TR.trace(lambda C, x: fp.f32_to_bf16(C, x, W.BF16_NAN_F2FP), [F32], BF16),
                            "cvt.rn.bf16.f32(x), NaN -> 0x7FFF (F2fpBf16_v1): one element of NarrowF32ToBf16_v1")
MeanDivTriton = _traced("MeanDivTriton", 1, ("N",), lambda S: ((("acc", F32),), F32),
                        lambda S: TR.trace(lambda C, a: fp.f32_mul(C, a, fp.f32_const(_rcp_n(S.N)), ir_nan=True), [F32], F32),
                        "div.full.f32(acc, f32(N)) = F32Mul_v1(acc, DivFullRcp_v1(f32(N))), the reciprocal folded "
                        "(DivFullScaleA_v1(acc, f32(N)) is acc)")


# --- the rsqrt and the narrow ---------------------------------------------------------------------------------------------------

@_on_bits("RsqrtF32", 2, ["N", "RSQRT"], lambda S: ((("x", Array(S.N, F32)),), Array(S.N, F32)),
          _word_view(D.RsqrtF32, ("N",), ("RSQRT",)), word_statics={"RSQRT": P.RsqrtApprox})
def RsqrtF32(B, S, x):
    """`RsqrtF32_v1` on bits: ``RSQRT`` (computing `RsqrtApprox_v1`) per element; on the word id, each element packed and its
    result unpacked."""
    if _on_word(S.RSQRT):
        words = B.batch(S.RSQRT, B.batch(PackF32, x, axes=(0,)), axes=(0,))
        return B.batch(UnpackF32, words, axes=(0,))
    return B.batch(S.RSQRT, x, axes=(0,))


@_on_bits("NarrowF32ToBf16", 2, ["N"], lambda S: ((("x", Array(S.N, F32)),), Array(S.N, BF16)),
          _word_view(D.NarrowF32ToBf16, ("N",)))
def NarrowF32ToBf16(B, S, x):
    """`NarrowF32ToBf16_v1` on bits: `NarrowF32ToBf16El_v1` per element (separable)."""
    return B.batch(NarrowF32ToBf16El, x, axes=(0,))


# --- the mean (batch-invariant `mean_kernel`) -----------------------------------------------------------------------------------

@cache
def _mean_layout(N: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Per 1024-column chunk, the elements of each thread holding at least one, in register order (`dense.thread_elements`).
    The live registers are a prefix of a thread's 8, and the live threads a prefix of the chunk's 128."""
    spt = D.size_per_thread(N)
    chunks = []
    for c0 in range(0, N, D.BLOCK_SIZE):
        threads = []
        for t in range(D.THREADS):
            els = D.thread_elements(c0, t, spt)
            live = [e for e in els if e < N]
            assert els[:len(live)] == live, (N, c0, t)
            assert not live or len(threads) == t, (N, c0, t)
            if live:
                threads.append(tuple(live))
        chunks.append(tuple(threads))
    return tuple(chunks)


def _warp_lanes(N: int) -> list[list[int]]:
    """Per chunk, the live lanes of each of its warps (0 for an idle warp)."""
    return [[max(0, min(WARP, len(th) - WARP * w)) for w in range(D.NUM_WARPS)] for th in _mean_layout(N)]


def _addz(B, a: Coll | None, b: Coll | None) -> Coll | None:
    """F32Add_v1(a, b), None standing for +0: +0 + +0 folded, an add of +0 `F32AddZero_v1`, any other `F32Add_v3`."""
    if a is None or b is None:
        return None if a is None and b is None else B.call(F32AddZero, b if a is None else a)
    return B.call(EW.F32Add, a, b)


@composite("MeanThreadTriton", 1, ["K"], lambda S: ((("x", Array(S.K, F32)),), F32), conformance="exact-model-tested")
def MeanThreadTriton(B, S, x):
    """One thread's 8 registers added in order, its K live elements first (`F32Add_v3`), then its 8 - K masked +0s
    (`F32AddZero_v1`)."""
    if not 1 <= S.K <= REGS:
        raise ValueError(f"a thread holds 1 to {REGS} live registers, not {S.K}")
    a = x[0]
    for k in range(1, S.K):
        a = B.call(EW.F32Add, a, x[k])
    for _ in range(REGS - S.K):
        a = B.call(F32AddZero, a)
    return a


def _thread_count(N: int) -> int:
    return sum(len(th) for th in _mean_layout(N))


@composite("MeanThreadsTriton", 1, ["N"], lambda S: ((("x", Array(S.N, F32)),), Array(_thread_count(S.N), F32)),
           conformance="exact-model-tested")
def MeanThreadsTriton(B, S, x):
    """Every live thread's sum, chunk by chunk in thread order: a batch of `MeanThreadTriton_v1{K}` per run of threads with K
    live registers (separable)."""
    out: list[Coll] = []
    for threads in _mean_layout(S.N):
        i = 0
        while i < len(threads):
            K = len(threads[i])
            j = i
            while j < len(threads) and len(threads[j]) == K:
                j += 1
            m = coll_concat(Array(j - i, Array(K, F32)), [x[e] for th in threads[i:j] for e in th])
            res = B.batch(bind(MeanThreadTriton, K=K), m, axes=(0,))
            out += [res[k] for k in range(j - i)]
            i = j
    return array_of(out)


@composite("MeanWarpTriton", 1, ["L"], lambda S: ((("p", Array(S.L, F32)),), F32), conformance="exact-model-tested")
def MeanWarpTriton(B, S, p):
    """Lane 0 after the xor butterfly 16, 8, 4, 2, 1 (lane i adds lane i ^ m, its own value first) over a warp whose lanes from
    L on are idle (+0): lane 0's cone, each add of +0 kept (`_addz`)."""
    if not 1 <= S.L <= WARP:
        raise ValueError(f"a warp has 1 to {WARP} live lanes, not {S.L}")
    masks = (16, 8, 4, 2, 1)
    memo: dict[tuple[int, int], Coll | None] = {}

    def lane(k: int, i: int) -> Coll | None:
        if k == 0:
            return p[i] if i < S.L else None
        if (k, i) not in memo:
            memo[(k, i)] = _addz(B, lane(k - 1, i), lane(k - 1, i ^ masks[k - 1]))
        return memo[(k, i)]

    return lane(len(masks), 0)


def _live_warps(N: int) -> int:
    return sum(1 for ch in _warp_lanes(N) for L in ch if L)


@composite("MeanWarpsTriton", 1, ["N"], lambda S: ((("p", Array(_thread_count(S.N), F32)),), Array(_live_warps(S.N), F32)),
           conformance="exact-model-tested")
def MeanWarpsTriton(B, S, p):
    """Each live warp's lane 0, chunk by chunk (`MeanWarpTriton_v1{L}`; the full warps of a chunk in one batch; separable)."""
    out: list[Coll] = []
    base = 0
    for lanes in _warp_lanes(S.N):
        full = sum(1 for L in lanes if L == WARP)
        if full:
            res = B.batch(bind(MeanWarpTriton, L=WARP), p[base:base + WARP * full].reshape(Array(full, Array(WARP, F32))),
                          axes=(0,))
            out += [res[w] for w in range(full)]
        rest = [L for L in lanes[full:] if L]
        assert len(rest) <= 1 and all(L == 0 for L in lanes[full + len(rest):]), lanes
        if rest:
            lo = base + WARP * full
            out.append(B.call(bind(MeanWarpTriton, L=rest[0]), p[lo:lo + rest[0]]))
        base += sum(lanes)
    return array_of(out)


@composite("MeanCombineTriton", 1, ["N"], lambda S: ((("w", Array(_live_warps(S.N), F32)),), Array(1, F32)),
           conformance="exact-model-tested")
def MeanCombineTriton(B, S, w):
    """Each chunk's `(w0 + w2) + (w1 + w3)` (idle warps +0), the chunks added in order from +0 (`_addz`), then
    `MeanDivTriton_v1{N}`."""
    acc: Coll | None = None
    i = 0
    for lanes in _warp_lanes(S.N):
        ws: list[Coll | None] = []
        for L in lanes:
            ws.append(w[i] if L else None)
            i += bool(L)
        acc = _addz(B, acc, _addz(B, _addz(B, ws[0], ws[2]), _addz(B, ws[1], ws[3])))
    return array_of([B.call(bind(MeanDivTriton, N=S.N), acc)])


def mean_triton_calls(B, N: int, x: Coll) -> Coll:
    """`MeanTriton_v2{N}`'s Calls, written into the body ``B`` (a composite's, or a Program root's)."""
    th = B.call(bind(MeanThreadsTriton, N=N), x)
    return B.call(bind(MeanCombineTriton, N=N), B.call(bind(MeanWarpsTriton, N=N), th))


@_on_bits("MeanTriton", 2, ["N"], lambda S: ((("x", Array(S.N, F32)),), Array(1, F32)), _word_view(D.MeanTriton, ("N",)))
def MeanTriton(B, S, x):
    """`MeanTriton_v1` on bits (`mean_triton_calls`): vLLM's batch-invariant `mean_dim` over one row of N f32 values."""
    return mean_triton_calls(B, S.N, x)
