"""numpy twin of `registry.dsv4_decode_indices` (`compute_global_topk_indices_and_lens`, vLLM d9105ea80, Triton 3.7.1): the kernel's
stores of a decode token's slot row and length, over a batch of tokens.

Written again from the kernel (common/ops/cache_utils.py:469-517) rather than from the registry: the block index by truncating
division, the block number loaded where the index is valid, `block_numbers * block_size + block_offsets` in int32, -1 where the
index is negative, and the count of valid indices stored where the token's byte is nonzero.  A token one of whose valid indices
names a block past its row (index // block_size >= NB) is declined: the kernel reads the next request's row there.
`registry.dsv4_decode_indices` registers it (`register_twin`) with `kernel_registry` as kernel ``dsv4-decode-indices-twin``.
"""
from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels.dsv4_indices_twin import s32
from verity_vllm.program.kernels.kernel_registry import Declined, register

KERNEL = "dsv4-decode-indices-twin"
DOMAIN = "every token whose valid indices name blocks in its own row"


def global_topk(topk, blocks, valid, TOPK: int, BS: int, NB: int):
    """(tokens, TOPK) top-k words, (tokens, NB) block-table rows and (tokens,) bytes -> (rows (tokens, TOPK) u32, lens (tokens,)
    u32, declined (tokens,) bool)."""
    x = s32(topk).reshape(-1, TOPK)
    blocks = s32(blocks).reshape(-1, NB)
    valid = np.asarray(valid).astype(np.int64).reshape(-1) & 0xFF
    live = x >= 0
    b = np.where(x < 0, -((-x) // BS), x // BS)
    inside = (b >= 0) & (b < NB)
    bn = np.take_along_axis(blocks, np.where(live & inside, b, 0), axis=1)
    rem = x - b * BS
    rows = np.where(live, s32(bn * BS + rem), -1)
    lens = np.where(valid != 0, live.sum(axis=1), 0)
    declined = (live & ~inside).any(axis=1)
    return (rows & 0xFFFFFFFF).astype(np.uint32), lens.astype(np.uint32), declined


def _one(topk, blocks, valid, *, TOPK, BS, NB):
    rows, lens, declined = global_topk(topk, blocks, valid, TOPK, BS, NB)
    if declined[0]:
        raise Declined("a valid index names a block past its row, which the kernel reads from the next request's")
    return np.concatenate([rows[0], lens]).astype(np.uint64)


def _batch(topk, blocks, valid, *, TOPK, BS, NB):
    rows, lens, declined = global_topk(topk, blocks, valid, TOPK, BS, NB)
    return np.concatenate([rows, lens[:, None]], axis=1).astype(np.uint64), declined


def indices(rng, shape, BS: int, NB: int) -> np.ndarray:
    """Top-k rows of words: each -1, another negative word, an index in the row (around its block edges and its last slot), and
    in a quarter of the rows also draws over the int32 words, as uint32."""
    words = rng.integers(0, 1 << 32, shape, dtype=np.uint64).astype(np.int64)
    edges = np.asarray([0, 1, BS - 1, BS, NB * BS - 1, -1, -2, -BS, -(1 << 31)], dtype=np.int64)
    kind = rng.random(shape)
    wild = (rng.random(shape[0]) < 0.25)[:, None] & (kind >= 0.9)
    out = np.where(kind < 0.2, -1, np.where(kind < 0.3, edges[rng.integers(0, len(edges), shape)], rng.integers(0, NB * BS, shape)))
    return (np.where(wild, words, out) & 0xFFFFFFFF).astype(np.uint32)


def _sample(rng, n, params, *, TOPK, BS, NB):
    topk = indices(rng, (n, TOPK), BS, NB)
    blocks = np.where(rng.random((n, NB)) < 0.7, rng.integers(0, 1 << 17, (n, NB)),
                      rng.integers(0, 1 << 32, (n, NB), dtype=np.uint64).astype(np.int64))
    valid = np.where(rng.random(n) < 0.7, rng.integers(0, 2, n), rng.integers(0, 256, n))
    return [topk, (blocks & 0xFFFFFFFF).astype(np.uint32), valid.astype(np.uint8)]


def _targets(defs: dict) -> list:
    """The served row at max_model_len 4096, and two small ones (one of odd widths)."""
    from verity.ml.circuits.defs import bind

    statics = [dict(TOPK=1024, BS=64, NB=16), dict(TOPK=8, BS=4, NB=8), dict(TOPK=5, BS=2, NB=3)]
    return [bind(defs["Dsv4GlobalTopk_v1"], **st) for st in statics]


def register_twin(defs: dict) -> None:
    """Register the twin; ``defs`` is `registry.dsv4_decode_indices`' composites by id, which binds the self-check's
    specialisations (the registry calls this, so neither module imports the other both ways)."""
    register("Dsv4GlobalTopk_v1", _one, kernel=KERNEL, domain=DOMAIN, sample=_sample, batch=_batch,
             check_targets=lambda rng: _targets(defs))
