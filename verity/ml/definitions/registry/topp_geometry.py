"""The launch geometry of vLLM's split top-p pipeline, as the sampling Definitions read it.

`splits_for` is `_apply_topp_split`'s split count (its doubling loop) and `splits_for_v1` its total form, the evaluator of
`registry.lifted`'s `SplitsForSMS{n}_v1`; `spt_for` is the sizePerThread of the kernels' row loads and `z_group` the one of the two
`Z` values a warp lane holds. The keep-word builder (`topp_keep_builder`) reads `spt_for` and `z_group` from here; the exact model of
the pipeline itself is the `topp_split` twin."""
from __future__ import annotations

import numpy as np


SPLIT_MAX_BATCH = 64      # vLLM `_SPLIT_MAX_BATCH`: the split pipeline serves sampler batches up to this size (no top-k)
SPLIT_MAX_SPLITS = 32     # `_SPLIT_MAX_SPLITS`


def splits_for(batch_size: int, num_sm: int) -> int:
    """`_apply_topp_split`: doubling S from 1 while S < 32 and S * 2 * batch_size <= num_sm, i.e. the largest power of two S <= 32 with
    S * batch_size <= num_sm (L40S 142 SMs: B1..B4 -> 32, B5..B8 -> 16, B9..B17 -> 8; verbatim loop of vLLM d9105ea80)."""
    s = 1
    while s < SPLIT_MAX_SPLITS and s * 2 * batch_size <= num_sm:
        s *= 2
    return s


SPLITS_FOR_V1_WIDTH = 6   # the result S in {1, 2, 4, 8, 16, 32} is a Value<6> (`LTopPMask_v1{V}` takes S_t as an operand)


def splits_for_v1(n_live: int, num_sms: int) -> int:
    """`SplitsFor_v1(n_live, num_SMs)` -- the split count S of vLLM d9105ea80 `_apply_topp_split` as a VALUE of the live sampler batch:
    `S_t = SplitsFor_v1(n_live_t, num_SMs)` with `n_live_t` = the number of
    rows in the sampler batch at engine step t = #{r : arrive_r <= t <= arrive_r + n_r - 1}, n_r the SERVED length (a prefill row is
    sampled; the step that samples a request's last token -- its EOS or its cap -- is live; the step after it is not: vLLM drops a
    finished request before the next step).  As activity values: live_{r,t} = 1 at t = arrive_r and
    LStepLive(out_r[t-1], E) for t > arrive_r -- under GUARD=in exactly LIsActive(tok_in_{r,t}), the step's INPUT token -- and
    n_live_t = sum_r live_{r,t}.  NOT sum_r LIsActive(out_r[t-1]): the committed EOS is an Active token, so that form counts one phantom
    row per stopped request at its a_r and moves every batch-mate's S there.  `num_SMs` is a target-profile constant.  Request r's local
    event e reads S at engine step arrive_r + e.  Value: the largest power of two S <= SPLIT_MAX_SPLITS (32) with S * n_live <= num_SMs, at least 1 -- the
    verbatim doubling loop `splits_for` (doubles while the NEXT S still satisfies it); the thresholds are a function of num_SMs and are never tabled per device class
    (142 SMs: 1..4 -> 32, 5..8 -> 16, 9..17 -> 8, 18..35 -> 4, 36..71 -> 2, 72+ -> 1; 132 SMs: 1..4 -> 32, 5..8 -> 16, 9..16 -> 8,
    17..33 -> 4, 34..66 -> 2, 67+ -> 1).  Measured against the real kernel: the B8 staggered record, 157 engine steps, 526 sampled
    rows -- the mask kernel's constexpr S == SplitsFor_v1(live, 142) at every step (tests/data/topp_split_fixture/b8_s_schedule_*.json).

    DOMAIN (total on it, refuses outside -- never a number that names no launch): 1 <= n_live <= SPLIT_MAX_BATCH (64) and num_SMs >= 1.
    n_live == 0: no sampler launch, no sampling event -- the lifted form `LSplitsFor_v1` emits bot (the sixth member of the specified
    class); n_live > 64: vLLM does not run `_apply_topp_split` (the sort path serves the batch), so TopPMask_v1 is not the
    Definition of that step and no S exists for it."""
    n, m = int(n_live), int(num_sms)
    if n < 1 or n > SPLIT_MAX_BATCH:
        raise ValueError(f"SplitsFor_v1: n_live={n} outside the domain 1..{SPLIT_MAX_BATCH} (0 = no sampling event; > {SPLIT_MAX_BATCH} = not the split pipeline)")
    if m < 1:
        raise ValueError(f"SplitsFor_v1: num_SMs={m} is not a target profile")
    s = splits_for(n, m)
    assert 1 <= s <= SPLIT_MAX_SPLITS and s & (s - 1) == 0 and s < (1 << SPLITS_FOR_V1_WIDTH)
    return s


def registered_splits(num_sms: int) -> int:
    """The registered split count: the S every served `_apply_topp_split` launch runs with, whatever its sampler batch.  The kernels'
    keep set depends on S (the slice partials, the per-slice pivot ladder, the two Z groups), so an S that follows the live batch lets
    a request's batch-mates pick its nucleus at top_p < 1.  It is SplitsFor_v1(1, num_SMs), the S vLLM picks for a lone row: the engine
    pins vLLM to it (`engine.topp_pin`), the Build declares it (`sampling.splits`, the select's `Const32[S]`) and the Match holds every
    launch's S to it (`check.match.sampler_geometry`).  Only num_SMs, a target-profile constant, enters it."""
    return splits_for_v1(1, num_sms)


def spt_for(V: int, S: int) -> int:
    """sizePerThread of the 1-D [BLOCK] logits load in all three kernels (Triton's coalescing pass, from AxisInfo):
    the row pointer `LOGITS + row * LOGITS_STRIDE_0` is 16-byte aligned only when the stride (= V for contiguous logits) carries
    Triton's `divisibility = 16` specialization, and the slice start `slice_id * ((V + S - 1) // S)` is aligned to the power of two
    dividing the slice length; the vector width is the min of the two, capped at 4 (128-bit).  Read off the TTGIR of the compiled
    kernels and confirmed bit-exactly on the stats buffers: V=128256/49152/32000/4096/1024 -> 4, V=1088/2112 -> 2,
    V=3000/1025/1040 -> 1 (L40S, vLLM d9105ea80, Triton 3.5)."""
    if V % 16 != 0:
        return 1
    slice_len = (V + S - 1) // S
    d = 1
    while d < 4 and slice_len % (d * 2) == 0:
        d *= 2
    return d


def z_group(lane: np.ndarray, S: int) -> np.ndarray:
    """Which of the TWO `Z` values the thread in warp lane `lane` (0..31) holds after `Z = tl.sum(exp_s * w)` over the [S] slice
    partials (`_topp_sb_row_stats`).  The [S] tensor is laid out lane -> element `lane % S` (S <= 32: replicated 32/S times across
    the warp), the reduction is an xor-butterfly over the S-lane group with masks S/2, ..., 1, and its FIRST stage (xor S/2)
    carries the fma-contracted lane product: lane i computes fma(a_i, b_i, prod_{i ^ S/2}) while lane i ^ S/2 computes
    fma(a_{i^S/2}, b_{i^S/2}, prod_i) -- two different roundings that the later (symmetric) stages never reconcile.  So the lanes
    whose bit S/2 is CLEAR (group 0) and SET (group 1) end with values up to 1 ulp apart, and everything downstream that a thread
    derives from Z (pr = exp(x - M) / Z, min_prob / max_prob, the pivot ladder, pivot_logit, dup, numkeep, num_preceding) is a
    per-thread value of its group.  S = 32: group = lane >= 16 (the "warp half" the S=32 model was fitted with); S = 16: group =
    (lane % 16) >= 8 -- NOT lane >= 16 (the four differing lanes of the B8 record's S=16 rows are exactly the tie lanes
    the wrong grouping assigned to the other group's Z / dup / quota); S = 1: a single value."""
    lane = np.asarray(lane, dtype=np.int64)
    if S < 2:
        return np.zeros(lane.shape, dtype=np.int64)
    return ((lane % S) >= (S // 2)).astype(np.int64)


__all__ = ["SPLIT_MAX_BATCH", "SPLIT_MAX_SPLITS", "splits_for", "SPLITS_FOR_V1_WIDTH", "splits_for_v1", "registered_splits", "spt_for",
           "z_group"]
