"""The sampled accept of speculative decoding and its resample (vLLM d9105ea80, `rejection_sampler_utils.rejection_sample`
with a greedy draft and the standard method) as registered Definitions, read from the kernels' PTX and SASS.

    AcceptExpLane_v1                (x, m: F32) -> F32                                      tl.exp(x - m)
    AcceptTileStats_v1{N,TILE}      (x: Array<N, F32>) -> (F32, F32)                        one tile's maximum and summed exponentials
    AcceptRowLse_v1{V,TILE}         (logits: Array<V, F32>) -> F32                          the row's log-sum-exp, as lane 0 holds it
    AcceptUniform_v1                (key: U32) -> F32                                       tl_rand32(.., includes_zero=False) of its word
    LogitAt_v1{V}                   (logits: Array<V, F32>, d: I32) -> F32                  logits[d], -inf outside [0, V)
    AcceptResidual_v1{V}            (logits: Array<V, F32>, d: I32) -> Array<V, F32>        the row with lane d set to -inf
    SampledAcceptRow_v1{V,TILE}     (logits: Array<V, F32>, d: I32, seed, pos: I64) -> BIT  the ratio test of one draft token
    SampledAccept_v1{K,V,TILE}      (logits: Array<K+1, Array<V, F32>>, draft: Array<K, I32>, seed: I64, pos: Array<K+1, I64>,
                                     temp: F32) -> (Array<K, I32>, I32, Array<K+1, I32>)    verdicts, accepted, appended

A request that verifies K draft tokens hands the sampler K + 1 rows of target logits (row i follows the first i draft tokens;
row K is the bonus row), already through temperature and top-p (-inf outside the nucleus).  With a one-hot draft
(`draft_logits` None, the greedy draft) and a nonzero temperature, `rejection_sample` launches four Triton kernels:

* `_compute_local_logits_stats_kernel`, grid (rows, tiles of 8192 logits), 4 warps.  For a draft row (`expanded_local_pos` <
  the number of draft steps; the bonus row returns at once) a tile's maximum and ``sum(exp(x - max))``, 0 when the maximum is
  -inf.  Thread t of the 128 holds lanes ``512 r + 4 t + j`` (r < 16, j < 4; one `ld.global.v4` a vector, -inf past the
  vocabulary).  The maximum is order-free.  The sum is each thread's 64 exponentials added in (r, j) order, the warp's 32 threads
  by the xor butterfly 16, 8, 4, 2, 1, the four warps by the butterfly 2, 1: `add.f32` throughout, none fused (`FADD` in the
  SASS).
* `_rejection_kernel`, grid (requests), 1 warp, lane t holding tile t of 32 (-inf and 0 past the last tile).  For each draft
  position while every earlier one was accepted: the row's maximum M over the tiles' maxima; ``e_t = exp(max_t - M)``; the sum
  of ``e_t * sum_t`` with the product fused into the butterfly's first stage, ``a_t = fma(e_t, sum_t, rn(e_{t^16} *
  sum_{t^16}))`` (`fma.rn.f32`), then `add.f32` by the stages 8, 4, 2, 1; ``lse = M + log(S)``; ``logit[draft] - lse >
  log(u)`` (`setp.gt.f32`), u the uniform of the row's stream word.  The two halves of the warp fuse the other half's product,
  so lanes 0-15 and 16-31 hold sums that can differ in the last place; lane 0 writes the outputs.
* `_resample_kernel`, grid (requests, tiles of 1024), and `_insert_resampled_kernel`: at the first refused position, or the
  bonus row when none is, the first index of the maximum of ``logits[v] + g(key)[v]``, g the Gumbel noise of the row's stream
  key, over the row with the refused draft token's lane set to -inf (the bonus row and a negative draft token mask nothing).

Each instruction and the gate that states it:

    sub.f32, mul.f32, add.f32, fma.rn.f32          F32Sub_v1, F32Mul_v1, F32Add_v1, F32Fma_v1 (round to nearest, subnormals kept)
    tl.exp: mul.f32 by 0x3FB8AA3B, ex2.approx.f32  F32Mul_v1, then Ex2Approx_v1 (t < -126 ? MUFU.EX2(t / 2)^2 : MUFU.EX2(t))
    tl.log: libdevice __nv_logf inlined            NvLogf_v4 (fma.rn.ftz, the 2^23 scaling of an input below FLT_MIN a mul.f32)
    max.f32                                        SelectF32_v1 of F32GtStrict_v1: the same word but for a zero's sign
    setp.gt.f32, selp, setp.eq                     F32GtStrict_v1, SelectF32_v1 / SelectI32_v1, I32Eq_v1
    tl.rand: Philox4x32-10's word 0 of (pos, seed) GumbelStreamKey_v1: the same word the row's Gumbel noise is keyed by
       shr.s32 31, xor; cvt.rn.f32.s32             SelectI32_v1 of U32Xor_v1 by I32Le_v1; I32ToF32_v1
       mul.f32 by 0x2FFFFFFF; max.f32 0x2FFFFFFF   F32Mul_v1; SelectF32_v1 of F32GtStrict_v1 (`includes_zero=False`: the floor)
    the resample's noise, add and argmax           GumbelSelectF32_v3{V} (its add is `add.f32` on the GPU, F32AddFtz_v1 there)

Domain.  Every logit is finite or -inf and none is subnormal, a row has a finite logit, a draft token is below V, the
temperature is not zero (at zero the kernels verify greedily, which `Accept_v1` states) and the vocabulary has 17 to 32
tiles with a row stride divisible by 16 (the load's four lanes a vector and the one-warp reduction's 32 lanes: the layouts
the PTX was read at).  On it a zero maximum's sign reaches no output, `add.f32` is `F32AddFtz_v1` on the noisy lanes, and no
operand of a log is below FLT_MIN.  One case the kernel leaves undefined: when lane 0's sum accepts a draft token that is not
the last and lanes 16-31's sum refuses it, the upper half stops verifying and the next position's `shfl.sync` reads lanes that
no longer run it.  The Definition is lane 0's arithmetic there too; `sampled_accept` reports the case (`halves_split`).

`TILE` is the stats kernel's `BLOCK_SIZE`, 8192 at the pin; another multiple of 512 keeps the thread layout (128 threads,
four lanes a vector) with fewer vectors a thread, for tests at small vocabularies.

The numpy reference (`tile_stats`, `row_lse`, `accept_uniform`, `sampled_accept`) computes the same words from the kernels'
layouts written independently of the Definitions' bodies (`topp_split.tree_sum_1d`, `sampling_rows.gumbel_select_row`).
"""
from __future__ import annotations

import numpy as np

from verity.circuits.defs import bind, composite
from verity.circuits.refs import Coll as _Coll, array_of, concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.ops.fp32 import F32Add, F32Fma, F32Mul
from verity.definitions.ops.mufu import mufu_ex2_words, table
from verity.definitions.ops.prims import const
from verity.definitions.ops.scalar import BIT, BitAnd, BitNot, F32, F32Eq, I32, I32Add, I32Eq, I32Le, SelectF32, SelectI32
from verity.definitions.registry import dead_free as DF, sampling as SA
from verity.definitions.registry.dsv4_moe import Ex2Approx
from verity.definitions.registry.ref_prims import F32GtStrict, F32Sub
from verity.definitions.registry.topp_word_gates import I32ToF32, LOG2E, U32Xor
from verity.definitions.registry.torch_words import SelectI64

I64, U32 = SA.I64, SA.U32
#: the stats kernel's `BLOCK_SIZE` at the pin (`VOCAB_BLOCK_SIZE`)
STATS_TILE = 8192
#: the stats kernel's threads (4 warps) and the lanes of one load vector
THREADS, VECTOR = 128, 4
#: lanes of the one-warp reduction over a row's tiles (`PADDED_VOCAB_NUM_BLOCKS` for 17 to 32 tiles)
PADDED_TILES = 32
NEG_INF = SA.NEG_INF_BITS
_CONF = ("model-untested: read from the PTX and SASS Triton 3.7.1 compiles for sm_90a from vLLM d9105ea80's "
         "`rejection_sample` kernels at 154,880 logits and five draft tokens; a capture of those kernels on a cc 9.0 GPU "
         "would hold it")


def _refuse(b: dict) -> str | None:
    tile, v, n = b.get("TILE"), b.get("V"), b.get("N")
    if tile is not None and not (type(tile) is int and tile >= 512 and tile % 512 == 0):
        return f"TILE is a positive multiple of 512 (128 threads, four lanes a vector), not {tile!r}"
    if n is not None and not (type(n) is int and 1 <= n and (tile is None or n <= tile)):
        return f"a tile holds 1 to TILE logits, not N={n!r} at TILE={tile!r}"
    if v is not None and tile is not None:
        if not (type(v) is int and v % 16 == 0 and PADDED_TILES // 2 < -(-v // tile) <= PADDED_TILES):
            return (f"V is a multiple of 16 in 17 to 32 tiles of TILE logits, the layouts the kernels were read at "
                    f"(got V={v!r}, TILE={tile!r})")
    k = b.get("K")
    if k is not None and not (type(k) is int and k >= 1):
        return f"K is a positive number of draft tokens, not {k!r}"
    return None


# ---------------------------------------------------------------------------------------------------------
# the reference, on numpy float32 and fp32 words
# ---------------------------------------------------------------------------------------------------------

_F = np.float32


def thread_lanes(tile: int, n: int) -> list[list[int]]:
    """The lanes below ``n`` each of the stats kernel's 128 threads holds of a tile of ``tile`` logits, in the order it adds
    them: ``512 r + 4 t + j`` for r, then j."""
    return [[lane for r in range(tile // 512) for j in range(VECTOR) if (lane := 512 * r + VECTOR * t + j) < n]
            for t in range(THREADS)]


def tl_exp(x) -> np.ndarray:
    """``tl.exp`` on float32: `mul.f32` by log2(e), then `ex2.approx.f32` (`Ex2Approx_v1`)."""
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        t = (np.asarray(x, _F) * np.uint32(LOG2E).view(_F)).astype(_F)
        small = t < _F(-126.0)
        half = np.ascontiguousarray(np.where(small, (t * _F(0.5)).astype(_F), t), dtype=_F)
        e = mufu_ex2_words(half.reshape(-1).view(np.uint32), table("ex2")).view(_F).reshape(t.shape)
        return np.where(small, (e * e).astype(_F), e).astype(_F)


def tile_stats(x, tile: int = STATS_TILE) -> tuple[np.ndarray, np.ndarray]:
    """Each tile's maximum and summed exponentials of a row of float32 logits (`_compute_local_logits_stats_kernel`)."""
    from verity.definitions.registry.topp_split import tree_sum_1d

    x = np.asarray(x, _F).reshape(-1)
    tiles = -(-x.shape[0] // tile)
    padded = np.full(tiles * tile, -np.inf, _F)
    padded[:x.shape[0]] = x
    padded = padded.reshape(tiles, tile)
    m = padded.max(axis=1)
    with np.errstate(invalid="ignore"):
        s = tree_sum_1d(tl_exp(padded - m[:, None]), VECTOR, THREADS // 32)
    return m, np.where(m != -np.inf, s, _F(0.0)).astype(_F)


def row_lse(m, s) -> tuple[int, int]:
    """The row's log-sum-exp as lane 0 and as lane 16 of `_rejection_kernel` hold it (fp32 words), from the tiles' maxima and
    sums."""
    from verity.definitions.registry.topp_split import fma32

    m, s = np.asarray(m, _F), np.asarray(s, _F)
    mm, ss = np.full(PADDED_TILES, -np.inf, _F), np.zeros(PADDED_TILES, _F)
    mm[:m.shape[0]], ss[:s.shape[0]] = m, s
    top = mm.max()
    with np.errstate(invalid="ignore", over="ignore"):
        e = tl_exp(mm - top)
        lane = np.arange(PADDED_TILES)
        a = fma32(e, ss, (e * ss).astype(_F)[lane ^ 16]).astype(_F)
        for k in (8, 4, 2, 1):
            a = (a + a[lane ^ k]).astype(_F)
        logs = np.array([SA.nv_logf_gpu(int(w), ftz=False) for w in a[[0, 16]].view(np.uint32)], np.uint32).view(_F)
        lo, hi = (top + logs).astype(_F).view(np.uint32)
    return int(lo), int(hi)


def accept_uniform(key: int) -> int:
    """``tl_rand32(seed, pos, includes_zero=False)`` from its Philox word: the fp32 word of max(float(x >= 0 ? x : ~x) *
    0x2FFFFFFF, 0x2FFFFFFF)."""
    x = SA._i32(key)
    u = SA.mul_ftz(SA._u(float(-x - 1 if x < 0 else x)), SA.TL_RAND_SCALE_BITS)
    return u if SA._f(u) > SA._f(SA.TL_RAND_MIN_BITS) else SA.TL_RAND_MIN_BITS


def sampled_accept(logits, draft, seed: int, pos, temp: int, tile: int = STATS_TILE) -> dict:
    """`SampledAccept_v1{K,V,TILE}` on a [K+1, V] float32 array, K draft tokens, the request's seed, the K+1 rows' positions
    and the temperature's fp32 word: ``verdict``, ``accepted`` and ``appended`` as the Definition returns them, with each draft
    row's tile statistics, log-sum-exp (lane 0's and lane 16's), uniform and log-probability, the resampled row and token, and
    ``halves_split``: the first position lane 0 accepts and lane 16 refuses with a draft token after it, or None."""
    from verity.definitions.registry.sampling_rows import gumbel_select_row

    logits = np.ascontiguousarray(logits, dtype=_F)
    draft = [int(d) for d in draft]
    K, V = len(draft), logits.shape[1]
    out: dict = {"max": [], "sumexp": [], "lse": [], "lse_hi": [], "u": [], "log_u": [], "logprob": [], "test": [],
                 "halves_split": None}
    verdict, live = [], True
    for i, d in enumerate(draft):
        m, s = tile_stats(logits[i], tile)
        lo, hi = row_lse(m, s)
        x = logits[i, d] if 0 <= d < V else _F(-np.inf)
        u = accept_uniform(SA.stream_key(seed, int(pos[i])))
        log_u = SA.nv_logf_gpu(u, ftz=False)
        with np.errstate(invalid="ignore"):
            lp, lp_hi = (_F(x) - np.array([lo, hi], np.uint32).view(_F)).astype(_F)
            test, test_hi = bool(lp > np.uint32(log_u).view(_F)), bool(lp_hi > np.uint32(log_u).view(_F))
        if live and test and not test_hi and i < K - 1 and out["halves_split"] is None:
            out["halves_split"] = i
        live = live and test
        verdict.append(int(live))
        for name, value in (("max", m), ("sumexp", s), ("lse", lo), ("lse_hi", hi), ("u", u), ("log_u", log_u),
                            ("logprob", int(np.asarray(lp, _F).view(np.uint32))), ("test", int(test))):
            out[name].append(value)
    r = sum(verdict)
    residual = logits[r].copy()
    if r < K and 0 <= draft[r] < V:
        residual[draft[r]] = -np.inf
    token, _ = gumbel_select_row(residual.view(np.uint32), seed, int(pos[r]), temp)
    appended = [draft[j] if j < r else token if j == r else -1 for j in range(K + 1)]
    return {**out, "verdict": verdict, "accepted": r, "appended": appended, "row": r, "token": token}


# ---------------------------------------------------------------------------------------------------------
# the Definitions
# ---------------------------------------------------------------------------------------------------------


def _max(B, a, b):
    """`max.f32` of two words that are no NaN: ``b > a ? b : a`` (a zero's sign is a's on a tie of zeros)."""
    return B.call(SelectF32, B.call(F32GtStrict, b, a), b, a)


@composite("AcceptExpLane", 1, [], lambda S: ((("x", F32), ("m", F32)), F32), conformance=_CONF)
def AcceptExpLane(B, S, x, m):
    """``tl.exp(x - m)``: `sub.f32`, `mul.f32` by log2(e) (0x3FB8AA3B), `ex2.approx.f32`."""
    return B.call(Ex2Approx, B.call(F32Mul, B.call(F32Sub, x, m), B.call(const(32, LOG2E))))


@composite("AcceptTileStats", 1, ["N", "TILE"], lambda S: ((("x", Array(S.N, F32)),), Tuple(F32, F32)), conformance=_CONF,
           refuse=_refuse)
def AcceptTileStats(B, S, x):
    """One tile of `_compute_local_logits_stats_kernel` on its N logits below the vocabulary's end: the maximum, and the sum of
    ``tl.exp(x - max)`` in the kernel's order (a thread's lanes in (r, j) order, the warp's butterfly 16 .. 1, the warps'
    butterfly 2, 1), 0 when the maximum is -inf.  A lane past the vocabulary adds +0 on the GPU, which changes no sum, so it
    has no gate here."""
    top = x[0]
    for v in range(1, S.N):
        top = _max(B, top, x[v])
    e = B.batch(AcceptExpLane, x, top, axes=(0, None))
    threads = []
    for lanes in thread_lanes(S.TILE, S.N):
        if not lanes:
            break
        acc = e[lanes[0]]
        for lane in lanes[1:]:
            acc = B.call(F32Add, acc, e[lane])
        threads.append(acc)
    warps = [DF.lane0_butterfly(B, threads[w:w + 32], (16, 8, 4, 2, 1), F32Add) for w in range(0, len(threads), 32)]
    total = DF.lane0_butterfly(B, warps, (2, 1), F32Add)
    finite = B.call(BitNot, B.call(F32Eq, top, B.call(const(32, NEG_INF))))
    return tuple_of(top, B.call(SelectF32, finite, total, B.call(const(32, 0))))


@composite("AcceptRowLse", 1, ["V", "TILE"], lambda S: ((("logits", Array(S.V, F32)),), F32), conformance=_CONF, refuse=_refuse)
def AcceptRowLse(B, S, logits):
    """``max + log(sum_t sumexp_t * exp(max_t - max))`` over the row's tiles as lane 0 of `_rejection_kernel` computes it: tile
    t in lane t of 32, the product fused into the first butterfly stage (``fma(e_t, sum_t, rn(e_{t+16} * sum_{t+16}))`` for
    t < 16, a lane past the last tile holding +0), then `add.f32` by the stages 8, 4, 2, 1; `__nv_logf`; `add.f32`."""
    V, T = S.V, S.TILE
    full, rest = divmod(V, T)
    ms, ss = [], []
    if full:
        st = B.batch(bind(AcceptTileStats, N=T, TILE=T), _Coll(Array(full, Array(T, F32)), logits[0:full * T].refs), axes=(0,))
        ms += [st[t][0] for t in range(full)]
        ss += [st[t][1] for t in range(full)]
    if rest:
        st = B.call(bind(AcceptTileStats, N=rest, TILE=T), logits[full * T:V])
        ms.append(st[0])
        ss.append(st[1])
    top = ms[0]
    for m in ms[1:]:
        top = _max(B, top, m)
    e = [B.call(AcceptExpLane, m, top) for m in ms]
    half = PADDED_TILES // 2
    upper = [B.call(F32Mul, e[t], ss[t]) for t in range(half, len(ms))]
    lower = [B.call(F32Fma, e[t], ss[t], upper[t]) if t < len(upper) else B.call(F32Mul, e[t], ss[t]) for t in range(half)]
    total = DF.lane0_butterfly(B, lower, (8, 4, 2, 1), F32Add)
    return B.call(F32Add, top, B.call(SA.NvLogfV4, total))


@composite("AcceptUniform", 1, [], lambda S: ((("key", U32),), F32), conformance=_CONF)
def AcceptUniform(B, S, key):
    """``tl_rand32(seed, pos, includes_zero=False)`` from its Philox word: the word or its complement when negative as an i32
    (`shr.s32 31`, `xor`), `cvt.rn.f32.s32`, `mul.f32` by 0x2FFFFFFF, and the floor `max.f32` with 0x2FFFFFFF, so the least
    uniform is 0x2FFFFFFF and none is zero."""
    negative = B.call(BitNot, B.call(I32Le, B.call(const(32, 0)), key))
    x = B.call(SelectI32, negative, B.call(U32Xor, key, B.call(const(32, 0xFFFFFFFF))), key)
    u = B.call(F32Mul, B.call(I32ToF32, x), B.call(const(32, SA.TL_RAND_SCALE_BITS)))
    floor = B.call(const(32, SA.TL_RAND_MIN_BITS))
    return B.call(SelectF32, B.call(F32GtStrict, u, floor), u, floor)


@composite("LogitAtStep", 1, [], lambda S: ((("carry", Tuple(I32, F32)), ("x", F32), ("d", I32)), Tuple(Tuple(I32, F32), F32)),
           conformance="reference-defined: one lane of LogitAt_v1's index into its operand")
def LogitAtStep(B, S, carry, x, d):
    """One lane: the lane's logit when its index is d, else the carried one; the index plus one."""
    at = B.call(SelectF32, B.call(I32Eq, carry[0], d), x, carry[1])
    return tuple_of(tuple_of(B.call(I32Add, carry[0], B.call(const(32, 1))), at), at)


@composite("LogitAt", 1, ["V"], lambda S: ((("logits", Array(S.V, F32)), ("d", I32)), F32),
           conformance="reference-defined: an index into its operand, -inf when d is outside [0, V) (the kernel's load "
                       "`target_logits[row, token]`)")
def LogitAt(B, S, logits, d):
    """``logits[d]``, and -inf when d is no index of the row."""
    carry = tuple_of(B.call(const(32, 0)), B.call(const(32, NEG_INF)))
    if S.V > 1:
        carry = B.scan(LogitAtStep, carry, xs=(logits[0:S.V - 1],), shared=(d,))[0]
    return B.call(SelectF32, B.call(I32Eq, carry[0], d), logits[S.V - 1], carry[1])


@composite("AcceptResidualStep", 1, [], lambda S: ((("i", I32), ("x", F32), ("d", I32)), Tuple(I32, F32)),
           conformance="reference-defined: one lane of AcceptResidual_v1")
def AcceptResidualStep(B, S, i, x, d):
    """One lane: -inf when its index is d, else its logit; the index plus one."""
    y = B.call(SelectF32, B.call(I32Eq, i, d), B.call(const(32, NEG_INF)), x)
    return tuple_of(B.call(I32Add, i, B.call(const(32, 1))), y)


@composite("AcceptResidual", 1, ["V"], lambda S: ((("logits", Array(S.V, F32)), ("d", I32)), Array(S.V, F32)), conformance=_CONF)
def AcceptResidual(B, S, logits, d):
    """`_resample_kernel`'s residual of a one-hot draft: ``tl.where(block != rejected_draft_token, target_logits, -inf)``;
    a d outside [0, V) (the bonus row's, a negative draft token) leaves the row as it is."""
    i, head = B.call(const(32, 0)), []
    if S.V > 1:
        res = B.scan(AcceptResidualStep, i, xs=(logits[0:S.V - 1],), shared=(d,))
        i, head = res[0], [res[1].refs]
    last = B.call(SelectF32, B.call(I32Eq, i, d), B.call(const(32, NEG_INF)), logits[S.V - 1])
    return _Coll(Array(S.V, F32), concat(head + [last.refs]))


@composite("SampledAcceptRow", 1, ["V", "TILE"],
           lambda S: ((("logits", Array(S.V, F32)), ("d", I32), ("seed", I64), ("pos", I64)), BIT), conformance=_CONF,
           refuse=_refuse)
def SampledAcceptRow(B, S, logits, d, seed, pos):
    """The ratio test of one draft token d against its row: ``logits[d] - lse > log(u)`` (`sub.f32`, `__nv_logf`,
    `setp.gt.f32`), u the uniform of `tl.randint(seed, pos)`'s word.  A d outside [0, V) reads -inf and fails, as the kernel's
    placeholder -1 does."""
    lp = B.call(F32Sub, B.call(bind(LogitAt, V=S.V), logits, d), B.call(bind(AcceptRowLse, V=S.V, TILE=S.TILE), logits))
    u = B.call(AcceptUniform, B.call(SA.GumbelStreamKey, seed, pos))
    return B.call(F32GtStrict, lp, B.call(SA.NvLogfV4, u))


@composite("SampledAccept", 1, ["K", "V", "TILE"],
           lambda S: ((("logits", Array(S.K + 1, Array(S.V, F32))), ("draft", Array(S.K, I32)), ("seed", I64),
                       ("pos", Array(S.K + 1, I64)), ("temp", F32)),
                      Tuple(Array(S.K, I32), I32, Array(S.K + 1, I32))), conformance=_CONF, refuse=_refuse)
def SampledAccept(B, S, logits, draft, seed, pos, temp):
    """Sampled verification of K draft tokens and the resample (the module docstring): draft token p is accepted iff every
    earlier one was and its row's ratio test holds.  Returns (verdict[K] in {0, 1}, accepted = sum(verdict), appended[K+1]):
    ``appended[j]`` is draft token j for j < accepted, the resampled token at j = accepted, and -1 after it.  The resampled
    token is `GumbelSelectF32_v3{V}` on row ``accepted`` with that row's position, its refused draft token's lane at -inf
    (nothing masked on the bonus row); the row, its position and its draft token are selected by the verdicts, so the select
    runs once.  Each draft row's ratio test is one member of a batch; the bonus row has none."""
    K, V = S.K, S.V
    test = B.batch(bind(SampledAcceptRow, V=V, TILE=S.TILE), logits[0:K], draft, seed, pos[0:K], axes=(0, 0, None, 0))
    one, zero, minus_one = B.call(const(32, 1)), B.call(const(32, 0)), B.call(const(32, 0xFFFFFFFF))
    live = [test[0]]
    for p in range(1, K):
        live.append(B.call(BitAnd, live[p - 1], test[p]))
    verdict = [B.call(SelectI32, live[p], one, zero) for p in range(K)]
    accepted = verdict[0]
    for p in range(1, K):
        accepted = B.call(I32Add, accepted, verdict[p])
    row, at, d = logits[K], pos[K], minus_one
    for p in reversed(range(K)):
        row = B.batch(SelectF32, live[p], row, logits[p], axes=(None, 0, 0))
        at = B.call(SelectI64, live[p], at, pos[p])
        d = B.call(SelectI32, live[p], d, draft[p])
    token = B.call(bind(DF.GumbelSelectF32V3, V=V), B.call(bind(AcceptResidual, V=V), row, d), seed, at, temp)
    appended = [B.call(SelectI32, live[0], draft[0], token)]
    for j in range(1, K):
        appended.append(B.call(SelectI32, live[j], draft[j], B.call(SelectI32, live[j - 1], token, minus_one)))
    appended.append(B.call(SelectI32, live[K - 1], token, minus_one))
    return tuple_of(array_of(verdict), accepted, array_of(appended))


ALL_COMPOSITES = [AcceptExpLane, AcceptTileStats, AcceptRowLse, AcceptUniform, LogitAtStep, LogitAt, AcceptResidualStep,
                  AcceptResidual, SampledAcceptRow, SampledAccept]

__all__ = ["STATS_TILE", "PADDED_TILES", "AcceptExpLane", "AcceptTileStats", "AcceptRowLse", "AcceptUniform",
           "LogitAtStep", "LogitAt", "AcceptResidualStep", "AcceptResidual", "SampledAcceptRow", "SampledAccept", "ALL_COMPOSITES",
           "thread_lanes", "tl_exp", "tile_stats", "row_lse", "accept_uniform", "sampled_accept"]
