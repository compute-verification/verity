"""The sampled accept of speculative decoding and its resample (vLLM d9105ea80, `rejection_sampler_utils.rejection_sample`
with a greedy draft and the standard method) as registered Definitions, read from the kernels' PTX and SASS.

    SampledAccept_v1{K,V,TILE}      (logits: Array<K+1, Array<V, F32>>, draft: Array<K, I32>, seed: I64, pos: Array<K+1, I64>,
                                     temp: F32) -> (Array<K, I32>, I32, Array<K+1, I32>)    verdicts, accepted, appended

The Call is past a Flock block at every layout it takes, so its body is units of at most a block each and their wiring
(`plan`, which `cut` counts by Definition for `CUTS`), every unit reading its own lanes of the rows, each as rows of `ROW`
lanes (x0, x1, ..: Array<32, F32> each, the last one shorter where the row ends inside it), the rows the top-p's mask
chunks write (`topp_stages.mask_plan`; `pass_plan` wires them):

    AcceptTileHead_v1{N,AT,FIRST,LAST} (x0 .., d: I32[, i: I32, at: F32][, m: F32]) -> (F32[, I32], F32)
                                      lanes AT .. AT+N of a tile: its maximum so far; the row's logit at draft token d, a
                                      scan carried head to head
    AcceptTileSum_v1{N,AT,LAST}       (x0 .., m: F32[, acc: Array<128, F32>]) -> Array<128, F32> | F32
                                      lanes AT .. AT+N of a tile: tl.exp(x - m) into the 128 thread sums; LAST, the tile's sum
    AcceptRowEnd_v1{T,N,AT}           (x0 .., m: F32[, acc], at: F32, seed, pos: I64, m0, s0, .. : F32) -> I32
                                      the last tile's last part and the other T-1 tiles' maxima and sums: the ratio test
    AcceptResample_v1{K,N,FIRST,LAST} (x0 .. xK: Array<N, F32>, t0 .. t{K-1}: I32, draft: Array<K, I32>, seed: I64,
                                       pos: Array<K+1, I64>, temp: F32[, c: Array<3, I32>]) -> Array<3, I32> | the Call's
                                      a chunk of the K+1 rows: the resampled row's lanes, the Gumbel select's scan carried
    AcceptExpLane_v1                  (x, m: F32) -> F32        tl.exp(x - m)
    AcceptUniform_v1                  (key: U32) -> F32         tl_rand32(.., includes_zero=False) of its word
    LogitAtStep_v1                                              one lane of the head's scan

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
    setp.gt.f32, selp, setp.eq                     F32GtStrict_v1, SelectF32_v1 / SelectI32_v1, I32Eq_v1 (a lane's index: I32Sub_v1)
    tl.rand: Philox4x32-10's word 0 of (pos, seed) GumbelStreamKey_v1: the same word the row's Gumbel noise is keyed by
       shr.s32 31, xor; cvt.rn.f32.s32             SelectI32_v1 of U32Xor_v1 by I32Le_v1; I32ToF32_v1
       mul.f32 by 0x2FFFFFFF; max.f32 0x2FFFFFFF   F32Mul_v1; SelectF32_v1 of F32GtStrict_v1 (`includes_zero=False`: the floor)
    the resample's noise, add and argmax           GumbelSelectF32_v3{V}'s steps (its add is `add.f32` on the GPU, F32AddFtz_v1 there)

Domain.  Every logit is finite or -inf and none is subnormal, a row has a finite logit, a draft token is below V, the
temperature is not zero (at zero the kernels verify greedily, which `Accept_v1` states) and the vocabulary has 17 to 32
tiles with a row stride divisible by 16 (the load's four lanes a vector and the one-warp reduction's 32 lanes: the layouts
the PTX was read at).  On it a zero maximum's sign reaches no output, `add.f32` is `F32AddFtz_v1` on the noisy lanes, and no
operand of a log is below FLT_MIN.  One case the kernel leaves undefined: when lane 0's sum accepts a draft token that is not
the last and lanes 16-31's sum refuses it, the upper half stops verifying and the next position's `shfl.sync` reads lanes that
no longer run it.  The Definition is lane 0's arithmetic there too; `sampled_accept` reports the case (`halves_split`).  An
honest request meets it at a rate of 3.1e-10 +- 0.3e-10 a request on rows where the draft token holds 50% to 99.9% of the
mass and 8.9e-11 +- 1.6e-11 on the tests' `random_rows`, both over synthetic rows at K = 5, V = 154,880 and TILE = 8192
(exact over the uniform's 2^32 words given the rows); the share of a served model's rows whose two sums differ is not
measured, and were every row's two sums to differ the rate would be about 1.4e-6 a request on the rows where the draft token
holds 50% to 99.9% of the mass (9.35e-11 a reached position over the 2.31e-4 of them whose sums differ, times 3.55 reached
positions a request: a mean over those rows, not a ceiling).  When a step in the case is sampled, an
honest server's proof of it can fail, since what the card stores after the split need not be lane 0's arithmetic: there is
no retry, and the verifier cannot tell that failure from a dishonest server's.

`TILE` is the stats kernel's `BLOCK_SIZE`, 8192 at the pin; another multiple of 512 keeps the thread layout (128 threads,
four lanes a vector) with fewer vectors a thread, for tests at small vocabularies.

The units.  A draft row's tile is its heads of `head` lanes (4,096 at the pin: 128 rows and the unit's others stay within a
proof block's 256, `windows.interface.check_proof_block`), each handing the tile's maximum so far to the next, and its parts
of `part` lanes (2,048 at the pin: at 4,096 a part's exponentials and its rows' hm96 slots, one a row, are past a Flock
block); the row's last part is in `AcceptRowEnd_v1`.  The K ratio tests are K units of one Definition, each row's tiles
read once; the bonus row has no test.  The resample is `GumbelSelectF32_v3{V}`'s function on the residual row,
written over its own steps (`GumbelStreamKey_v1`, `GumbelNoiseLane_v1`, `GumbelSelectStep_v1`) in chunks of `chunk` lanes
of all K + 1 rows: every chunk selects the row, its stream key and its refused draft token by the tests, so the select runs
once, on rows that are operands, and no residual row is committed.  It keys each row's position and selects the 32-bit key,
never the 64-bit position, so no gate it computes is wider than Q_call's X.

The numpy reference (`tile_stats`, `row_lse`, `accept_uniform`, `sampled_accept`) computes the same words from the kernels'
layouts written independently of the Definitions' bodies (`topp_split.tree_sum_1d`, `sampling_rows.gumbel_select_row`).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from verity.circuits.defs import SpecializedDefinition, bind, composite
from verity.circuits.refs import array_of, coll_concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.evidence import EVIDENCE
from verity.circuits.ops.fp32 import F32Add, F32AddFtz, F32Fma, F32Mul
from verity.circuits.ops.mufu import mufu_ex2_words, table
from verity.circuits.ops.prims import const
from verity.circuits.ops.scalar import BitAnd, BitNot, F32, F32Eq, I32, I32Add, I32Eq, I32Le, SelectF32, SelectI32
from verity.definitions.registry import dead_free as DF, sampling as SA, topp_stages as TS
from verity.definitions.registry.dsa import I32Sub
from verity.definitions.registry.dsv4_moe import Ex2Approx
from verity.definitions.registry.ref_prims import F32GtStrict, F32Sub
from verity.definitions.registry.topp_word_gates import I32ToF32, LOG2E, U32Xor
I64, U32 = SA.I64, SA.U32
#: the stats kernel's `BLOCK_SIZE` at the pin (`VOCAB_BLOCK_SIZE`)
STATS_TILE = 8192
#: the stats kernel's threads (4 warps) and the lanes of one load vector
THREADS, VECTOR = 128, 4
#: lanes of the one-warp reduction over a row's tiles (`PADDED_VOCAB_NUM_BLOCKS` for 17 to 32 tiles)
PADDED_TILES = 32
#: the lanes of a row of the masked logits, `topp_stages.chunk` at every vocabulary the Call takes: a mask chunk writes one
#: row (`topp_stages.mask_plan`), and a wire reads a writer's row whole, so every unit takes its lanes as such rows, one
#: parameter each
ROW = TS.SELECT_N
#: the lanes one `AcceptTileHead_v1` unit reads, at most (`head`): half a tile at the pin, 128 rows beside its five
HEAD = 4096
#: the lanes of a tile one `AcceptTileSum_v1` unit adds, at most (`part`): a quarter of a tile at the pin, so its exponentials
#: and its rows' hm96 slots, one a row, fit a Flock block
PART = 2048
#: the lanes of an `AcceptResample_v1` chunk, at most (`chunk`): one row of each of the K + 1 rows
RESAMPLE_N = ROW
NEG_INF = SA.NEG_INF_BITS
_CONF = ("exact-model-tested: vLLM d9105ea80's `rejection_sample` kernels launched on a cc 9.0 GPU at 154,880 logits and five "
         f"draft tokens, one request of 1 + 5 rows a launch ({EVIDENCE['sampled_accept_capture']['run_record']}, "
         "adapters/vllm/tests/properties/sampled_accept_capture_gpu.py): every word they stored for a request inside the "
         "domain is the reference's, and on the requests its replay evaluates "
         f"({EVIDENCE['sampled_accept_capture']['replay_run_record']}) `SampledAccept_v1` gives the stored count and tokens; "
         "the launched PTX and SASS are the ones the Definitions were read from "
         f"({EVIDENCE['sampled_accept_ptx']['run_record']})")


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


def accept_kernel(logits, draft, seed, pos, temp, *, K: int, V: int, TILE: int) -> list[int]:
    """`sampled_accept` on one instance of `SampledAccept_v1{K,V,TILE}`'s words, its outputs in the Call's leaf order.  It
    declines outside the domain the reference is written for: a NaN, +inf or subnormal logit, a row with no finite one, a
    draft token at V or past it."""
    from verity.definitions.numerics.kernel_registry import Declined

    w = (np.asarray(logits, np.uint64) & np.uint64(0xFFFFFFFF)).astype(np.uint32).reshape(K + 1, V)
    x = w.view(_F)
    if np.isnan(x).any() or np.isposinf(x).any() or (((w & 0x7F800000) == 0) & ((w & 0x7FFFFF) != 0)).any():
        raise Declined("a NaN, +inf or subnormal logit")
    if not np.isfinite(x).any(axis=1).all():
        raise Declined("a row with no finite logit")
    d = [int(np.uint32(int(t) & 0xFFFFFFFF).view(np.int32)) for t in np.asarray(draft).reshape(-1)]
    if max(d) >= V:
        raise Declined("a draft token at V or past it")
    ref = sampled_accept(x, d, int(seed), [int(p) for p in np.asarray(pos).reshape(-1)], int(temp), TILE)
    return [*ref["verdict"], ref["accepted"], *(t & 0xFFFFFFFF for t in ref["appended"])]


def _sample_accept(rng, n: int, params, *, K: int, V: int, TILE: int) -> list[np.ndarray]:
    """Draft passes the reference's domain holds: rows of normal logits, some -inf, draft tokens at a row's maximum, at a
    random lane or the placeholder -1, a seed, positions and a temperature in [0.5, 1.5)."""
    logits = rng.normal(0, 3, (n, K + 1, V)).astype(_F)
    logits[rng.random((n, K + 1, V)) < 0.05] = -np.inf
    logits[:, :, 0] = rng.normal(0, 3, (n, K + 1)).astype(_F)
    best = logits[:, :K].argmax(axis=2)
    draft = np.where(rng.random((n, K)) < 0.6, best, np.where(rng.random((n, K)) < 0.8, rng.integers(0, V, (n, K)), -1))
    temp = rng.uniform(0.5, 1.5, n).astype(_F).view(np.uint32)
    return [logits.view(np.uint32).astype(np.uint64), (draft & 0xFFFFFFFF).astype(np.uint64),
            rng.integers(0, 1 << 63, n, dtype=np.uint64), rng.integers(0, 1 << 20, (n, K + 1)).astype(np.uint64),
            temp.astype(np.uint64)]


def register_kernels() -> None:
    """Register `accept_kernel` as `SampledAccept_v1`'s numpy kernel, which circuit-check compares with the Definition on its
    vectors and on `_sample_accept`'s."""
    from verity.definitions.numerics import kernel_registry as KR

    KR.register(SampledAccept, accept_kernel, kernel="numpy", domain="finite or -inf logits, none subnormal, each row one "
                "finite; draft tokens below V", sample=_sample_accept)


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
           conformance="reference-defined: one lane of an index into a row, `AcceptTileHead_v1`'s scan")
def LogitAtStep(B, S, carry, x, d):
    """One lane: the lane's logit when its index is d, else the carried one; the index plus one."""
    at = B.call(SelectF32, B.call(I32Eq, carry[0], d), x, carry[1])
    return tuple_of(tuple_of(B.call(I32Add, carry[0], B.call(const(32, 1))), at), at)


def _flag(b: dict, *names: str) -> str | None:
    for name in names:
        if not isinstance(b.get(name), bool):
            return f"{name} is a bool (an id's `{name}=True`), not {b.get(name)!r}"
    return None


def _lanes(b: dict) -> str | None:
    n, at = b.get("N"), b.get("AT", 0)
    if not (type(n) is int and n >= 1):
        return f"N is a unit's lanes, a positive int, not {n!r}"
    if not (type(at) is int and at >= 0 and at % (THREADS * VECTOR) == 0):
        return f"AT is the unit's first lane in its tile, a multiple of {THREADS * VECTOR} (one vector a thread), not {at!r}"
    return None


def _xrows(n: int) -> tuple:
    """A unit's ``n`` lanes of the masked row as its rows of `ROW` lanes, a parameter each (``x0``, ``x1``, ..; the last one
    shorter where the row ends inside it)."""
    return tuple((f"x{j}", Array(min(ROW, n - j * ROW), F32)) for j in range(-(-n // ROW)))


def _joined(n: int, args: tuple):
    """The lanes of a unit's leading `_xrows` parameters as one array, and its other parameters."""
    r = len(_xrows(n))
    return coll_concat(Array(n, F32), list(args[:r])), args[r:]


def _head_refuse(b: dict) -> str | None:
    if (why := _lanes(b) or _flag(b, "FIRST", "LAST")) is not None:
        return why
    if b["FIRST"] and b.get("AT"):
        return f"the row's FIRST head starts its first tile, AT=0, not AT={b['AT']}"
    return None


def _head_sig(S):
    carry = () if S.FIRST else (("i", I32), ("at", F32))
    top = (("m", F32),) if S.AT else ()
    return (*_xrows(S.N), ("d", I32), *carry, *top), Tuple(F32, F32) if S.LAST else Tuple(F32, I32, F32)


@composite("AcceptTileHead", 1, ["N", "AT", "FIRST", "LAST"], _head_sig, conformance=_CONF, refuse=_head_refuse)
def AcceptTileHead(B, S, *args):
    """Lanes AT .. AT + N of a tile (rows x0, ..) -> the tile's maximum so far (`max.f32` lane by lane, order-free), from the
    maximum ``m`` of its lanes before AT, and the row's logit at draft token d as a scan carried head to head: (the lanes
    before this head, the logit at d among them, -inf when none is d), from (0, -inf) on the row's FIRST head.  The row's
    LAST head returns (maximum, ``logits[d]``): -inf when d is no index of the row, as the kernel's load of
    ``target_logits[row, token]`` is never reached for a placeholder."""
    x, (d, *rest) = _joined(S.N, args)
    carry, m = (rest[:2], rest[2:]) if not S.FIRST else ((), rest)
    top = _max(B, m[0], x[0]) if S.AT else x[0]
    for v in range(1, S.N):
        top = _max(B, top, x[v])
    c = tuple_of(B.call(const(32, 0)), B.call(const(32, NEG_INF))) if S.FIRST else tuple_of(*carry)
    scanned = S.N - 1 if S.LAST else S.N
    if scanned:
        c = B.scan(LogitAtStep, c, xs=(x[0:scanned],), shared=(d,))[0]
    if not S.LAST:
        return tuple_of(top, c[0], c[1])
    return tuple_of(top, B.call(SelectF32, B.call(I32Eq, c[0], d), x[S.N - 1], c[1]))


def _thread_sums(B, e, at: int, acc) -> list:
    """The stats kernel's thread sums after a part's exponentials ``e`` (the tile's lanes from ``at``) are added to ``acc``
    (the 128 carried sums, None on the tile's first part): thread t adds its lanes ``512 r + 4 t + j`` in (r, j) order.  A
    thread with no lane yet has no sum; the threads that have one are a prefix."""
    stride = THREADS * VECTOR
    out = []
    for t, lanes in enumerate(thread_lanes(-(-(at + len(e.refs)) // stride) * stride, at + len(e.refs))):
        cur = acc[t] if acc is not None else None
        for lane in lanes:
            if lane >= at:
                cur = e[lane - at] if cur is None else B.call(F32Add, cur, e[lane - at])
        if cur is None:
            break
        out.append(cur)
    return out


def _tile_sum(B, threads: list, m):
    """The tile's sum from its thread sums: each warp's butterfly 16, 8, 4, 2, 1, the warps' butterfly 2, 1; 0 when the
    tile's maximum ``m`` is -inf."""
    warps = [DF.lane0_butterfly(B, threads[w:w + 32], (16, 8, 4, 2, 1), F32Add) for w in range(0, len(threads), 32)]
    total = DF.lane0_butterfly(B, warps, (2, 1), F32Add)
    finite = B.call(BitNot, B.call(F32Eq, m, B.call(const(32, NEG_INF))))
    return B.call(SelectF32, finite, total, B.call(const(32, 0)))


def _sum_refuse(b: dict) -> str | None:
    if (why := _lanes(b) or _flag(b, "LAST")) is not None:
        return why
    if not b["LAST"] and b["N"] % (THREADS * VECTOR):
        return f"a part before a tile's last is whole vectors of every thread, a multiple of {THREADS * VECTOR} lanes, not N={b['N']}"
    return None


def _sum_sig(S):
    acc = (("acc", Array(THREADS, F32)),) if S.AT else ()
    return (*_xrows(S.N), ("m", F32), *acc), F32 if S.LAST else Array(THREADS, F32)


@composite("AcceptTileSum", 1, ["N", "AT", "LAST"], _sum_sig, conformance=_CONF, refuse=_sum_refuse)
def AcceptTileSum(B, S, *args):
    """Lanes AT .. AT + N of a tile of `_compute_local_logits_stats_kernel` (rows x0, ..) and the tile's maximum m:
    ``tl.exp(x - m)`` of each lane added into the tile's 128 thread sums in the kernel's order (a thread's lanes in (r, j)
    order), ``acc`` the sums after the lanes before AT.  The tile's LAST part returns its sum: the warp's butterfly 16 .. 1,
    the warps' butterfly 2, 1, and 0 when m is -inf.  A lane past the vocabulary adds +0 on the GPU, which changes no sum, so it has no gate here."""
    x, (m, *acc) = _joined(S.N, args)
    threads = _thread_sums(B, B.batch(AcceptExpLane, x, m, axes=(0, None)), S.AT, acc[0] if acc else None)
    return _tile_sum(B, threads, m) if S.LAST else array_of(threads)


def _end_refuse(b: dict) -> str | None:
    t = b.get("T")
    if not (type(t) is int and PADDED_TILES // 2 < t <= PADDED_TILES):
        return f"T is a row's tiles, 17 to 32 (the one-warp reduction's layouts the kernel was read at), not {t!r}"
    return _lanes(b)


def _end_sig(S):
    acc = (("acc", Array(THREADS, F32)),) if S.AT else ()
    others = tuple(port for t in range(S.T - 1) for port in ((f"m{t}", F32), (f"s{t}", F32)))
    return (*_xrows(S.N), ("m", F32), *acc, ("at", F32), ("seed", I64), ("pos", I64), *others), I32


@composite("AcceptRowEnd", 1, ["T", "N", "AT"], _end_sig, conformance=_CONF, refuse=_end_refuse)
def AcceptRowEnd(B, S, *args):
    """The ratio test of one draft token against its row of T tiles, 1 or 0: ``logits[d] - lse > log(u)`` (`sub.f32`,
    `__nv_logf`, `setp.gt.f32`).  It reads the last tile's last part (lanes AT .. AT + N as rows x0, .., with its maximum m
    and the thread sums ``acc`` before AT, as `AcceptTileSum_v1`'s LAST part), ``at`` the row's logit at the draft token, and
    the other tiles' maxima and sums m0, s0, ..  The log-sum-exp is lane 0's
    of `_rejection_kernel`: tile t in lane t of 32, M the tiles' maximum, ``e_t = tl.exp(m_t - M)``, the product fused into
    the first butterfly stage (``fma(e_t, s_t, rn(e_{t+16} * s_{t+16}))`` for t < 16, a lane past the last tile holding +0),
    `add.f32` by the stages 8, 4, 2, 1, `__nv_logf`, `add.f32` to M.      u is the uniform of `tl.randint(seed, pos)`'s word."""
    x, (m, *rest) = _joined(S.N, args)
    acc, rest = (rest[0], rest[1:]) if S.AT else (None, rest)
    at, seed, pos, others = rest[0], rest[1], rest[2], rest[3:]
    last = _tile_sum(B, _thread_sums(B, B.batch(AcceptExpLane, x, m, axes=(0, None)), S.AT, acc), m)
    ms, ss = [*others[0::2], m], [*others[1::2], last]
    top = ms[0]
    for v in ms[1:]:
        top = _max(B, top, v)
    e = [B.call(AcceptExpLane, v, top) for v in ms]
    half = PADDED_TILES // 2
    upper = [B.call(F32Mul, e[t], ss[t]) for t in range(half, S.T)]
    lower = [B.call(F32Fma, e[t], ss[t], upper[t]) if t < len(upper) else B.call(F32Mul, e[t], ss[t]) for t in range(half)]
    lse = B.call(F32Add, top, B.call(SA.NvLogfV4, DF.lane0_butterfly(B, lower, (8, 4, 2, 1), F32Add)))
    u = B.call(AcceptUniform, B.call(SA.GumbelStreamKey, seed, pos))
    test = B.call(F32GtStrict, B.call(F32Sub, at, lse), B.call(SA.NvLogfV4, u))
    return B.call(SelectI32, test, B.call(const(32, 1)), B.call(const(32, 0)))


def _resample_refuse(b: dict) -> str | None:
    if (why := _refuse(b) or _lanes(b) or _flag(b, "FIRST", "LAST")) is not None:
        return why
    if b["FIRST"] and b["LAST"] and b["N"] < 2:
        return "a row of one lane has no select"
    return None


def _resample_sig(S):
    K = S.K
    rows = tuple((f"x{p}", Array(S.N, F32)) for p in range(K + 1))
    tests = tuple((f"t{p}", I32) for p in range(K))
    carry = () if S.FIRST else (("c", Array(3, I32)),)
    ret = Tuple(Array(K, I32), I32, Array(K + 1, I32)) if S.LAST else Array(3, I32)
    return (*rows, *tests, ("draft", Array(K, I32)), ("seed", I64), ("pos", Array(K + 1, I64)), ("temp", F32), *carry), ret


@composite("AcceptResample", 1, ["K", "N", "FIRST", "LAST"], _resample_sig, conformance=_CONF, refuse=_resample_refuse)
def AcceptResample(B, S, *args):
    """A chunk of N lanes of the K + 1 rows (x0 .. xK), the K draft rows' ratio tests (t0 .. each 1 or 0) and the request's
    draft tokens, seed, positions and temperature.  Draft token p is accepted iff every earlier one was and its test is 1.
    The resampled row is the first refused one, or the bonus row when none is, with its position's stream key and its refused
    draft token's lane at -inf (`_resample_kernel`'s ``tl.where(block != rejected_draft_token, target_logits, -inf)``;
    nothing masked on the bonus row or for a negative draft token): the chunk's lane j is the token's iff the token less the
    lanes so far is j.  Over the chunk's lanes of it, `GumbelSelectF32_v3`'s
    scan (`GumbelSelectStep_v1`: y = x + g(key)[i] when temp != 0 else x, taken iff y > best), carried chunk to chunk as
    c = (best, its index, the lanes so far), from lane 0 on the row's FIRST chunk.  The row's LAST chunk returns the
    Call's outputs (`SampledAccept_v1`)."""
    K, N = S.K, S.N
    rows, tests = args[:K + 1], args[K + 1:2 * K + 1]
    draft, seed, pos, temp = args[2 * K + 1:2 * K + 5]
    one, zero, minus_one = B.call(const(32, 1)), B.call(const(32, 0)), B.call(const(32, 0xFFFFFFFF))
    live = [B.call(I32Eq, tests[0], one)]
    for p in range(1, K):
        live.append(B.call(BitAnd, live[p - 1], B.call(I32Eq, tests[p], one)))
    row, key, d = rows[K], B.call(SA.GumbelStreamKey, seed, pos[K]), minus_one
    for p in reversed(range(K)):
        row = B.batch(SelectF32, live[p], row, rows[p], axes=(None, 0, 0))
        key = B.call(SelectI32, live[p], key, B.call(SA.GumbelStreamKey, seed, pos[p]))
        d = B.call(SelectI32, live[p], d, draft[p])
    noisy = B.call(BitNot, B.call(F32Eq, temp, zero))
    rem = d if S.FIRST else B.call(I32Sub, d, args[-1][2])
    masked = B.batch(I32Eq, rem, array_of([B.call(const(32, j)) for j in range(N)]), axes=(None, 0))
    r = B.batch(SelectF32, masked, B.call(const(32, NEG_INF)), row, axes=(0, None, 0))
    if S.FIRST:
        c = tuple_of(B.call(SelectF32, noisy, B.call(F32AddFtz, r[0], B.call(SA.GumbelNoiseLane, key, zero)), r[0]), zero, one)
    else:
        c = tuple_of(args[-1][0], args[-1][1], args[-1][2])
    lo, hi = int(S.FIRST), N - 1 if S.LAST else N
    if hi > lo:
        c = B.scan(SA.GumbelSelectStep, c, xs=(r[lo:hi],), shared=(key, noisy))[0]
    if not S.LAST:
        return array_of([c[0], c[1], c[2]])
    y = B.call(SelectF32, noisy, B.call(F32AddFtz, r[N - 1], B.call(SA.GumbelNoiseLane, key, c[2])), r[N - 1])
    token = B.call(SelectI32, B.call(F32GtStrict, y, c[0]), c[2], c[1])
    verdict = [B.call(SelectI32, live[p], one, zero) for p in range(K)]
    accepted = verdict[0]
    for p in range(1, K):
        accepted = B.call(I32Add, accepted, verdict[p])
    appended = [B.call(SelectI32, live[0], draft[0], token)]
    for j in range(1, K):
        appended.append(B.call(SelectI32, live[j], draft[j], B.call(SelectI32, live[j - 1], token, minus_one)))
    appended.append(B.call(SelectI32, live[K - 1], token, minus_one))
    return tuple_of(array_of(verdict), accepted, array_of(appended))


# ---------------------------------------------------------------------------------------------------------
# the plan: the Call as its units
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """One unit of the Call: ``key`` names it, ``defn`` is its Definition, ``reads`` gives each parameter that is an operand of
    the Call (parameter, operand): ``("logits", row, lo, hi)`` a row's lanes, ``("draft", p)`` and ``("pos", p)`` one element,
    ``("draft",)``, ``("pos",)``, ``("seed",)`` and ``("temp",)`` the operand whole; ``wires`` each wired parameter's
    (parameter, writer key, writer's output index)."""
    key: tuple
    defn: SpecializedDefinition
    reads: tuple[tuple[str, tuple], ...]
    wires: tuple[tuple[str, tuple, int], ...] = field(default=())


def head(tile: int) -> int:
    """The lanes of a tile one `AcceptTileHead_v1` unit reads: `HEAD`, or the whole of a smaller tile."""
    return min(HEAD, tile)


def part(tile: int) -> int:
    """The lanes of a tile one `AcceptTileSum_v1` unit adds: `PART`, and at most half of a smaller tile in whole vectors of
    every thread, so a tile of 1,024 logits or more is two parts or more."""
    stride = THREADS * VECTOR
    return min(PART, max(stride, tile // (2 * stride) * stride))


def chunk(V: int) -> int:
    """A resample chunk's lanes: a mask chunk's (`topp_stages.chunk`), `ROW` at every vocabulary the Call takes."""
    return TS.chunk(V)


def _rows(p: int, lo: int, n: int) -> tuple:
    """A unit's reads of row p's lanes lo .. lo + n, one row of `ROW` lanes a parameter (`_xrows`)."""
    return tuple((name, ("logits", p, lo + j * ROW, lo + j * ROW + ty.n)) for j, (name, ty) in enumerate(_xrows(n)))


@cache
def plan(K: int, V: int, tile: int = STATS_TILE) -> tuple[Unit, ...]:
    """The units of `SampledAccept_v1{K,V,TILE}` in an order that writes each before its readers: each draft row's tiles (its
    heads, each carrying the tile's maximum to the next, then the tile's parts; the row's last part is its `AcceptRowEnd_v1`),
    then the resample's chunks, whose last returns the Call's outputs.  Every read of a row's lanes is one row of `ROW`
    lanes, so each is a mask chunk's output row whole (`pass_plan`)."""
    if chunk(V) != ROW:
        raise ValueError(f"V={V}: a mask chunk is {chunk(V)} lanes, not the {ROW} lanes of a row every unit reads")
    T, H, P, out = -(-V // tile), head(tile), part(tile), []
    for p in range(K):
        sums, tops, prev = {}, {}, None
        for t in range(T):
            lo, n = t * tile, min(tile, V - t * tile)
            starts = list(range(0, n, H))
            for a, at in enumerate(starts):
                hn, first, last = min(H, n - at), t == 0 and a == 0, t == T - 1 and a == len(starts) - 1
                w = (() if first else (("i", prev, 1), ("at", prev, 2))) + ((("m", prev, 0),) if at else ())
                prev = ("head", p, t, a)
                out.append(Unit(prev, bind(AcceptTileHead, N=hn, AT=at, FIRST=first, LAST=last),
                                (*_rows(p, lo + at, hn), ("d", ("draft", p))), w))
            tops[t] = prev
            starts = list(range(0, n, P))
            for a, at in enumerate(starts):
                pn, end = min(P, n - at), a == len(starts) - 1
                x = _rows(p, lo + at, pn)
                w = [("m", tops[t], 0)] + ([("acc", ("sum", p, t, a - 1), 0)] if a else [])
                if not (t == T - 1 and end):
                    out.append(Unit(("sum", p, t, a), bind(AcceptTileSum, N=pn, AT=at, LAST=end), x, tuple(w)))
                    sums[t] = ("sum", p, t, a)
                    continue
                w.append(("at", tops[t], 1))
                for t2 in range(T - 1):
                    w += [(f"m{t2}", tops[t2], 0), (f"s{t2}", sums[t2], 0)]
                out.append(Unit(("row", p), bind(AcceptRowEnd, T=T, N=pn, AT=at),
                                (*x, ("seed", ("seed",)), ("pos", ("pos", p))), tuple(w)))
    N = chunk(V)
    for k, lo in enumerate(range(0, V, N)):
        hi = min(lo + N, V)
        reads = (*((f"x{p}", ("logits", p, lo, hi)) for p in range(K + 1)),
                 ("draft", ("draft",)), ("seed", ("seed",)), ("pos", ("pos",)), ("temp", ("temp",)))
        wires = (*((f"t{p}", ("row", p), 0) for p in range(K)), *((("c", ("resample", k - 1), 0),) if k else ()))
        out.append(Unit(("resample", k), bind(AcceptResample, K=K, N=hi - lo, FIRST=k == 0, LAST=hi == V), reads, wires))
    return tuple(out)


def cut(t) -> list[tuple[SpecializedDefinition, int]]:
    """The units a binding of `SampledAccept_v1{K,V,TILE}` is cut into, each Definition bound and how many of it: its
    `plan`, which is its body."""
    b = t.bindings
    return list(Counter(u.defn for u in plan(int(b["K"]), int(b["V"]), int(b["TILE"]))).items())


#: the Call's units, by family: circuit-check holds each to the Flock block in the Call's place
CUTS = {"SampledAccept_v1": cut}


@cache
def pass_plan(K: int, V: int, tile: int = STATS_TILE) -> tuple[Unit, ...]:
    """The units of a pass that verifies K draft tokens, in an order that writes each before its readers: each of the K + 1
    rows' top-p pipeline and mask chunks (`topp_stages.mask_plan`, keyed ``("topp", row, *key)``), then `plan`'s units (keyed
    ``("accept", *key)``), every read of the masked rows a wire from the mask chunk that writes that row (its output 0).
    Its operands: ``logits`` the K + 1 rows of bf16 logits, ``temp``, ``top_p`` and ``splits`` (each row's), ``draft``,
    ``seed`` and ``pos``; the last unit returns `SampledAccept_v1`'s outputs."""
    N, out = chunk(V), []
    for p in range(K + 1):
        for u in TS.mask_plan(V):
            wired = {name for name, _, _ in u.wires}
            reads = tuple((name, ("logits", p, *u.lanes) if name == "logits" else (name,))
                          for name, _ in u.defn.signature()[0] if name not in wired)
            out.append(Unit(("topp", p, *u.key), u.defn, reads, tuple((name, ("topp", p, *k), j) for name, k, j in u.wires)))
    for u in plan(K, V, tile):
        reads, wires = [], [(name, ("accept", *k), j) for name, k, j in u.wires]
        for name, ref in u.reads:
            if ref[0] != "logits":
                reads.append((name, ref))
                continue
            _, p, lo, hi = ref
            assert lo % N == 0 and hi == min(lo + N, V), (u.key, name, ref)
            wires.append((name, ("topp", p, "mask", lo // N), 0))
        out.append(Unit(("accept", *u.key), u.defn, tuple(reads), tuple(wires)))
    return tuple(out)


def _operand(ops: dict, ref: tuple):
    name, *at = ref
    if name == "logits":
        return ops[name][at[0]][at[1]:at[2]]
    return ops[name][at[0]] if at else ops[name]


@composite("SampledAccept", 1, ["K", "V", "TILE"],
           lambda S: ((("logits", Array(S.K + 1, Array(S.V, F32))), ("draft", Array(S.K, I32)), ("seed", I64),
                       ("pos", Array(S.K + 1, I64)), ("temp", F32)),
                      Tuple(Array(S.K, I32), I32, Array(S.K + 1, I32))), conformance=_CONF, refuse=_refuse)
def SampledAccept(B, S, logits, draft, seed, pos, temp):
    """Sampled verification of K draft tokens and the resample (the module docstring).  Returns (verdict[K] in {0, 1},
    accepted = sum(verdict), appended[K+1]): ``appended[j]`` is draft token j for j < accepted, the resampled token at
    j = accepted, and -1 after it.  The body is the units of `plan` and their wiring, nothing else: each draft row's ratio
    test from its tiles' heads and parts, then the resample over the K + 1 rows in chunks."""
    ops = {"logits": logits, "draft": draft, "seed": seed, "pos": pos, "temp": temp}
    done: dict = {}
    for u in plan(S.K, S.V, S.TILE):
        args = {name: _operand(ops, ref) for name, ref in u.reads}
        for name, key, i in u.wires:
            args[name] = done[key][i] if isinstance(done[key].type, Tuple) else done[key]
        done[u.key] = B.call(u.defn, *(args[name] for name, _ in u.defn.signature()[0]))
    return done[u.key]


ALL_COMPOSITES = [AcceptExpLane, AcceptUniform, LogitAtStep, AcceptTileHead, AcceptTileSum, AcceptRowEnd,
                  AcceptResample, SampledAccept]

__all__ = ["STATS_TILE", "PADDED_TILES", "PART", "RESAMPLE_N", "AcceptExpLane", "AcceptUniform", "LogitAtStep",
           "AcceptTileHead", "AcceptTileSum", "AcceptRowEnd", "AcceptResample", "SampledAccept",
           "ALL_COMPOSITES", "CUTS", "Unit", "ROW", "HEAD", "head", "part", "chunk", "plan", "pass_plan", "cut", "thread_lanes",
           "tl_exp", "tile_stats", "row_lse", "accept_uniform", "sampled_accept", "accept_kernel", "register_kernels"]
