"""numpy twin of the greedy accept of vLLM d9105ea80's V2 rejection sampler on Triton 3.7.1 (temperature 0): the target argmax of
each verify row and the accept, as `registry.spec.GreedyAccept_v1` states them.

Written again from the kernels rather than from the registry (v1/worker/gpu/spec_decode/rejection_sampler_utils.py):

* A draft row's argmax, `_compute_local_logits_stats_kernel`'s greedy branch (:241-259) then `_compute_global_target_argmax`
  (:94-115): each tile of `STATS_TILE` logits loaded as f32 with -inf past V, its `tl.max(..., return_indices=True)`; then
  over the tiles' maxima, padded with -inf to a power of two, NaN mapped to -inf, `tl.argmax`, and that tile's index.
* The bonus row's argmax (every draft accepted): `_resample_kernel` takes the row's target logits (:745-748, 767-770) to
  `gumbel_block_argmax` (:826-852; sample/gumbel.py:134-189), whose `gumbel_noised_argmax` adds no noise at temperature 0 and
  is `tl.max(..., return_indices=True)` (gumbel.py:91-130), then `_insert_resampled_kernel` (:891-913): the same two stages
  over tiles of `RESAMPLE_TILE`.
* The accept, `_rejection_kernel`'s greedy branch (:554-593): draft i is accepted iff every earlier one was, it is not the -1
  placeholder, and it equals row i's argmax; the step keeps rows 0..accepted, their argmaxes (`_insert_resampled_kernel`'s
  count, :880-887).

`tl.max(..., return_indices=True)` and `tl.argmax` reduce by `_argmax_combine` with ties to the left (Triton 3.7.1
python/triton/language/standard.py:147-183): (v1, i1) is kept iff v1 > v2, or v1 == v2 and i1 < i2.  On a tile with no NaN
that is the first index of the maximum whatever order the reduction combines lanes in, -0 equal to +0.  With a NaN in the tile
it depends on that order, which reading the source does not give, so the twin declines such a row (`Declined`).
"""
from __future__ import annotations

import numpy as np

STATS_TILE, RESAMPLE_TILE = 8192, 1024


class Declined(ValueError):
    """A row whose argmax the twin does not state: a NaN in a tile, whose `tl.max` depends on the reduction's order."""


def f32_of_bf16(words) -> np.ndarray:
    """bf16 words (uint16) as the f32 values the kernels load them as (`.to(tl.float32)`, exact)."""
    return (np.asarray(words, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def tile_maxima(x: np.ndarray, tile: int) -> tuple[np.ndarray, np.ndarray]:
    """Each tile's (max, token id) of the f32 row ``x``, -inf past its end; `Declined` for a tile holding a NaN."""
    n = -(-len(x) // tile)
    padded = np.full(n * tile, -np.inf, dtype=np.float32)
    padded[:len(x)] = x
    tiles = padded.reshape(n, tile)
    if np.isnan(tiles).any():
        raise Declined(f"tiles {sorted(set(np.nonzero(np.isnan(tiles))[0].tolist()))} of {tile} logits hold a NaN")
    best = tiles.max(axis=1)
    first = np.argmax(tiles == best[:, None], axis=1)
    return best, np.arange(n, dtype=np.int64) * tile + first


def across_tiles(best: np.ndarray, token: np.ndarray) -> int:
    """The token of the tile `tl.argmax` picks over the maxima ``best``: padded with -inf to a power of two, each NaN mapped to
    -inf, the first tile of the maximum."""
    n = len(best)
    padded = np.full(1 << max(n - 1, 0).bit_length(), -np.inf, dtype=np.float32)
    padded[:n] = np.where(np.isnan(best), -np.inf, best)
    return int(token[int(np.argmax(padded == padded.max()))])


def row_argmax(words, tile: int) -> int:
    """A row's target argmax over tiles of ``tile`` from its bf16 ``words``."""
    return across_tiles(*tile_maxima(f32_of_bf16(words), tile))


def greedy_accept(draft, logits, tiles: tuple[int, int] = (STATS_TILE, RESAMPLE_TILE)) -> tuple[list[int], int, list[int]]:
    """``draft`` (K,) int32 token ids and ``logits`` (K+1, V) bf16 words -> (verdict[K], accepted, appended[K+1]) in
    `spec.Accept_v1`'s form: appended[j] the token the step keeps at slot j <= accepted, -1 past it. ``tiles``: the draft
    rows' and the bonus row's tile sizes, the kernels' by default."""
    draft, logits = [int(d) for d in draft], np.asarray(logits, dtype=np.uint16)
    stats, resample = tiles
    K = len(draft)
    if logits.shape[0] != K + 1:
        raise ValueError(f"{logits.shape[0]} logit rows for {K} drafts: one a draft and the bonus row")
    verdict, kept, verifying = [], [], True
    for i, d in enumerate(draft):
        if verifying:
            target = row_argmax(logits[i], stats)
            verifying = d >= 0 and target == d
            kept.append(d if verifying else target)
        verdict.append(int(verifying))
    accepted = sum(verdict)
    if accepted == K:
        kept.append(row_argmax(logits[K], resample))
    return verdict, accepted, kept + [-1] * (K + 1 - len(kept))
