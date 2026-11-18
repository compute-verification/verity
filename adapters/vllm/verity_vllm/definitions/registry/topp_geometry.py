"""The split count of vLLM's split top-p pipeline, as the sampling Definitions read it.

`splits_for` is the split count vLLM's `_apply_topp_split` picks from its batch (its doubling loop), which serving overrides, and
`registered_splits` the one every served launch runs at. The kernels' launch geometry, `spt_for` and `z_group`, is the core
ops' (`verity.circuits.ops.topp_geometry`), which the keep-word builder reads; the exact model of the pipeline itself is the
`topp_split` twin."""
from __future__ import annotations

from verity.circuits.ops.topp_geometry import spt_for, z_group

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


__all__ = ["SPLIT_MAX_BATCH", "SPLIT_MAX_SPLITS", "splits_for", "registered_splits", "spt_for", "z_group"]
