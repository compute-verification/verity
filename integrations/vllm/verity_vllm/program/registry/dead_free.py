"""Versions of `RMSNormTriton_v1`, `TokenSelect_v1` and the shared-greedy top-p select without dead gates: the forms the registered
compiler emits, where no gate may be one no output depends on.  Rows of record keep the v1s their pins were made from.

* `RMSNormTriton_v5`: `RMSNormTriton_v1` with each warp reduced by the adds lane 0's sum depends on (`lane0_butterfly`).  On the GPU
  every lane adds in every round of the xor butterfly, but only lane 0's sum is read: v1 has 129 dead adds a warp.
* `TokenSelect_v3`, `TopPMask_v3` and `GumbelSelectF32GivenGreedy_v2`: the v1 scan stops one lane early, and the last lane computes
  only what is read (its index, or its store).  v1's last iteration made a final best and an i + 1 that nothing reads.
* `GumbelTopPTokenSelectSharedGreedy_v2`: `GumbelTopPTokenSelectSharedGreedy_v1` over `TopPMask_v3` and
  `GumbelSelectF32GivenGreedy_v2`.

Each computes its v1's function word for word, except at V = 1, where v1's zero-length scan returns a wrong carry and these return
index 0 (`tests/program/test_dead_free.py`).
"""
from __future__ import annotations

from verity.ml.circuits.defs import Coll, bind, composite
from verity.ml.circuits.ops.fp32 import F32AddFtz
from verity.ml.circuits.ops.prims import ZERO32, const
from verity.ml.circuits.ops.scalar import BIT, BitNot, F32, F32Eq, SelectF32, SelectI32
from verity.ml.circuits.refs import Coll as _Coll, concat, tuple_of
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import prims as P, ref_prims as RP, sampling as SA
from verity_vllm.program.registry.b1 import ArgmaxStep, f32c
from verity_vllm.program.registry.prims import BF16, I32

I64 = SA.I64


def lane0_butterfly(B, vals: list[Coll], masks: tuple[int, ...], add) -> Coll:
    """Lane 0's value after the rounds `v[i] = add(v[i], v[i ^ m])` (a lane whose peer `i ^ m` is past the last keeps its value), each
    round computed on the lanes the later rounds read for lane 0: those an xor of the later masks reaches.  At masks 16, 8, 4, 2, 1
    over 32 lanes, 31 adds of the butterfly's 160, the same adds in the same order as lane 0's."""
    cur = dict(enumerate(vals))
    for k, m in enumerate(masks):
        reads = {0}
        for r in masks[k + 1:]:
            reads |= {i ^ r for i in reads}
        cur = {i: B.call(add, cur[i], cur[i ^ m]) if (i ^ m) in cur else cur[i] for i in sorted(reads) if i in cur}
    return cur[0]


@composite("RMSNormTriton", 5, ["N", "EPS"], lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def RMSNormTritonV5(B, S, x, w):
    """`RMSNormTriton_v1{N, EPS}`'s function, each warp's butterfly computing lane 0's adds alone: `out_j = bf16_rne((inv * x_j) *
    w_j)`, `inv = div.full(1, sqrt.approx.ftz(fma(rcp(n), sum, eps)))`, `sum` = chunks of 1024 columns, 128 threads x 8 columns
    (`acc = x1*x1; fma(x0,x0,acc); fma(x2..x7)`), warp butterfly xor 16,8,4,2,1, CTA `(w0 + w2) + (w1 + w3)`, chunks added in order
    from +0."""
    N = S.N
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    chunk_sums = []
    for c0 in range(0, N, 1024):
        warps = []
        for wi in range(4):
            lanes = []
            for ln in range(32):
                base = c0 + 8 * (wi * 32 + ln)
                if base >= N:
                    lanes.append(B.call(ZERO32))
                    continue
                xs = [xf[base + j] for j in range(8)]
                acc = B.call(P.F32Mul, xs[1], xs[1])
                acc = B.call(P.F32Fma, xs[0], xs[0], acc)
                for j in range(2, 8):
                    acc = B.call(P.F32Fma, xs[j], xs[j], acc)
                lanes.append(acc)
            warps.append(lane0_butterfly(B, lanes, (16, 8, 4, 2, 1), P.F32Add))
        chunk_sums.append(B.call(P.F32Add, B.call(P.F32Add, warps[0], warps[2]), B.call(P.F32Add, warps[1], warps[3])))
    total = B.call(ZERO32)
    for cs in chunk_sums:
        total = B.call(P.F32Add, total, cs)
    rcp_n = B.call(P.DivFullRcp, B.call(f32c(float(N))))
    rms = B.call(P.MufuSqrtFtz, B.call(P.F32Fma, rcp_n, total, B.call(f32c(S.EPS))))
    inv = B.call(P.F32Mul, B.call(P.DivFullScaleA, B.call(f32c(1.0)), rms), B.call(P.DivFullRcp, rms))
    wf = B.batch(P.Bf16ToF32, w, axes=(0,))
    sx = B.batch(P.F32Mul, inv, xf, axes=(None, 0))
    y = B.batch(P.F32Mul, sx, wf, axes=(0, 0))
    return B.batch(P.F2fpBf16, y, axes=(0,))


@composite("TokenSelect", 3, ["V"], lambda S: ((("logits", Array(S.V, BF16)),), I32), conformance="model-untested")
def TokenSelectV3(B, S, logits):
    """`TokenSelect_v1{V}`'s function: argmax over the bf16 logits, lowest index on ties (torch.argmax).  `ArgmaxStep_v1`'s scan to
    logits[V-2], then the last logit's test for its index alone."""
    if S.V == 1:
        return B.call(const(32, 0))
    init = tuple_of(logits[0], B.call(const(32, 0)), B.call(const(32, 1)))
    c = B.scan(ArgmaxStep, init, xs=(logits[1:S.V - 1],))[0] if S.V > 2 else init
    return B.call(P.SelectI32, B.call(P.Bf16GtStrict, logits[S.V - 1], c[0]), c[2], c[1])


@composite("TopPMask", 3, ["V"], lambda S: ((("logits", Array(S.V, F32)), ("top_p", F32), ("splits", I32)), Array(S.V, F32)),
           conformance="exact-model-tested")
def TopPMaskV3(B, S, logits, top_p, splits):
    """`TopPMask_v1{V}`'s function: `TopPMaskStep_v1`'s scan to logits[V-2], then the last lane's store alone, `tl.where(keep, x,
    -inf)`."""
    keep = B.call(SA.topp_mask_word(S.V), logits, top_p, splits)
    i, head = B.call(const(32, 0)), []
    if S.V > 1:
        res = B.scan(bind(SA.TopPMaskStep, V=S.V), i, xs=(logits[0:S.V - 1],), shared=(keep,))
        i, head = res[0], [res[1].refs]
    last = B.call(SelectF32, B.call(SA.bit_at(S.V), keep, i), logits[S.V - 1], B.call(const(32, SA.NEG_INF_BITS)))
    return _Coll(Array(S.V, F32), concat(head + [last.refs]))


@composite("GumbelSelectF32GivenGreedy", 2, ["V"],
           lambda S: ((("logits", Array(S.V, F32)), ("seed", I64), ("pos", I64), ("greedy", BIT)), I32), conformance="exact-model-tested")
def GumbelSelectF32GivenGreedyV2(B, S, logits, seed, pos, greedy):
    """`GumbelSelectF32GivenGreedy_v1{V}`'s function: `GumbelSelectStep_v1`'s scan to logits[V-2], then the last lane's test for its
    index alone (y = x + g(key)[i] when noisy else x; its index iff y > best)."""
    if S.V == 1:
        return B.call(const(32, 0))
    key = B.call(SA.GumbelStreamKey, seed, pos)
    noisy = B.call(BitNot, greedy)
    zero = B.call(const(32, 0))
    y0 = B.call(SelectF32, noisy, B.call(F32AddFtz, logits[0], B.call(SA.GumbelNoiseLane, key, zero)), logits[0])
    init = tuple_of(y0, zero, B.call(const(32, 1)))
    c = B.scan(SA.GumbelSelectStep, init, xs=(logits[1:S.V - 1],), shared=(key, noisy))[0] if S.V > 2 else init
    x = logits[S.V - 1]
    y = B.call(SelectF32, noisy, B.call(F32AddFtz, x, B.call(SA.GumbelNoiseLane, key, c[2])), x)
    return B.call(SelectI32, B.call(RP.F32GtStrict, y, c[0]), c[2], c[1])


@composite("GumbelTopPTokenSelectSharedGreedy", 2, ["V"],
           lambda S: ((("logits", Array(S.V, BF16)), ("top_p", F32), ("seed", I64), ("pos", I64), ("temp", F32), ("splits", I32)), I32),
           conformance="exact-model-tested")
def GumbelTopPTokenSelectSharedGreedyV2(B, S, logits, top_p, seed, pos, temp, splits):
    """`GumbelTopPTokenSelectSharedGreedy_v1{V}`'s function and signature, over `TopPMask_v3` and `GumbelSelectF32GivenGreedy_v2`."""
    if S.V == 1:
        return B.call(const(32, 0))
    greedy = B.call(F32Eq, temp, B.call(const(32, 0)))
    scaled = B.call(bind(SA.TemperatureScaleGivenGreedy, V=S.V), logits, temp, greedy)
    masked = B.call(bind(TopPMaskV3, V=S.V), scaled, top_p, splits)
    return B.call(bind(GumbelSelectF32GivenGreedyV2, V=S.V), masked, seed, pos, greedy)


__all__ = ["GumbelSelectF32GivenGreedyV2", "GumbelTopPTokenSelectSharedGreedyV2", "RMSNormTritonV5", "TokenSelectV3", "TopPMaskV3",
           "lane0_butterfly"]
