"""The launch geometry of vLLM's split top-p pipeline, as the sampling Definitions read it.

`splits_for` is the split count vLLM's `_apply_topp_split` picks from its batch (its doubling loop), which serving overrides, and
`registered_splits` the one every served launch runs at; `spt_for` is the sizePerThread of the kernels' row loads and `z_group` the
one of the two `Z` values a warp lane holds. The keep-word builder (`topp_keep_builder`) reads `spt_for` and `z_group` from here; the exact model of
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


def registered_splits(num_sms: int) -> int:
    """The registered split count: the S every served `_apply_topp_split` launch runs with, whatever its sampler batch.  The kernels'
    keep set depends on S (the slice partials, the per-slice pivot ladder, the two Z groups), so an S that follows the live batch lets
    a request's batch-mates pick its nucleus at top_p < 1.  It is `splits_for(1, num_SMs)`, the S vLLM picks for a lone row: the engine
    pins vLLM to it (`engine.topp_pin`), the Build declares it (`sampling.splits`, the select's `Const32[S]`) and the Match holds every
    launch's S to it (`check.match.sampler_geometry`).  Only num_SMs, a target-profile constant, enters it; no S is ever a value of
    the live batch."""
    m = int(num_sms)
    if m < 1:
        raise ValueError(f"registered_splits: num_SMs={m} is not a target profile")
    return splits_for(1, m)


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


__all__ = ["SPLIT_MAX_BATCH", "SPLIT_MAX_SPLITS", "splits_for", "registered_splits", "spt_for", "z_group"]
