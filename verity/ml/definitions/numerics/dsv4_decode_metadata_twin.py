"""numpy twin of `registry.dsv4_decode_metadata` (vLLM d9105ea80, Triton 3.7.1): the stores of a decode token's window row and
length (`_compute_swa_indices_and_lens_kernel`) and of its ratio-128 row and length (`_build_c128a_topk_metadata_kernel`'s
decode branch), over a batch of tokens.

Written again from the kernels rather than from the registry, lane by lane.  The window (v1/attention/backends/mla/sparse_swa.py:
950-1008): the position, start, end and length in int32, each lane's p = start + j, its block by truncating division, loaded where
p < end, `block_numbers * block_size + block_offsets` in int32 and -1 from the length on, an invalid token's row -1 and its length
0.  Ratio 128 (models/deepseek_v4/sparse_mla.py:382-416): q = (position + 1) // R in int64, truncating, min(q, W), lane j live
where j < that, its block blocks[j // BS], the slot in int32, -1 where not live, and the count of live lanes where the token's
int64 slot is >= 0.  A token one of whose live lanes is not loaded (p >= end) or names a block outside its row is declined: the
kernel reads 0, or another request's row, there.  `registry.dsv4_decode_metadata` registers it (`register_twin`) with
`kernel_registry` as kernel ``dsv4-decode-metadata-twin``.
"""
from __future__ import annotations

import numpy as np

from verity.ml.definitions.numerics.dsv4_indices_twin import s32
from verity.ml.definitions.numerics.kernel_registry import Declined, register

KERNEL = "dsv4-decode-metadata-twin"
SWA_DOMAIN = "every token whose live lanes load blocks of their own row"
C128A_DOMAIN = "every token whose live lanes name blocks of their own row"
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1


def s64(x) -> np.ndarray:
    """Words (any integer dtype) as signed 64-bit values."""
    return np.asarray(x).astype(np.uint64).view(np.int64)


def u64(x) -> np.ndarray:
    """Signed 64-bit values as uint64 words."""
    return np.asarray(x, dtype=np.int64).astype(np.uint64)


def swa_indices(t, qs, qe, seq, blocks, valid, SW: int, BS: int, NB: int):
    """(tokens,) t, qs, qe, seq words, (tokens, NB) block-table rows and (tokens,) bytes -> (rows (tokens, SW) u32, lens (tokens,)
    u32, declined (tokens,) bool)."""
    t, qs, qe, seq = (s32(v).reshape(-1) for v in (t, qs, qe, seq))
    blocks = s32(blocks).reshape(-1, NB)
    ok = (np.asarray(valid).astype(np.int64).reshape(-1) & 0xFF) != 0
    pos = s32(s32(s32(seq - s32(qe - qs)) + t) - qs)
    left_add = max(0 - (SW - 1), 0)
    start = np.maximum(s32(s32(pos - (SW - 1)) - left_add), 0)
    end = s32(s32(pos + 0) + 1)
    length = s32(end - start)
    j = np.arange(SW, dtype=np.int64)
    p = s32(start[:, None] + j)
    b = np.where(p < 0, -((-p) // BS), p // BS)
    rem = p - b * BS
    loaded = p < end[:, None]
    inside = (b >= 0) & (b < NB)
    bn = np.where(loaded, np.take_along_axis(blocks, np.where(inside, b, 0), axis=1), 0)
    live = j < length[:, None]
    rows = np.where(ok[:, None] & live, s32(bn * BS + rem), -1)
    lens = np.where(ok, length, 0)
    declined = ok & (live & ~(loaded & inside)).any(axis=1)
    return (rows & 0xFFFFFFFF).astype(np.uint32), (lens & 0xFFFFFFFF).astype(np.uint32), declined


def c128a_global_topk(position, slot, blocks, W: int, R: int, BS: int, NB: int):
    """(tokens,) int64 position and slot words and (tokens, NB) block-table rows -> (rows (tokens, W) u32, lens (tokens,) u32,
    declined (tokens,) bool)."""
    pos = s64(position).reshape(-1)
    slot = s64(slot).reshape(-1)
    blocks = s32(blocks).reshape(-1, NB)
    t = (pos.astype(np.uint64) + np.uint64(1)).view(np.int64)
    q = np.where(t < 0, (t + (R - 1)) // R, t // R)     # C's truncation: a negative t rounds up
    nc = np.minimum(q, W)
    j = np.arange(W, dtype=np.int64)
    live = j < nc[:, None]
    b = j // BS
    inside = b < NB
    bn = np.where(live & inside, blocks[:, np.minimum(b, NB - 1)], 0)
    rows = np.where(live, s32(bn * BS + j % BS), -1)
    lens = np.where(slot >= 0, live.sum(axis=1), 0)
    declined = (live & ~inside).any(axis=1)
    return (rows & 0xFFFFFFFF).astype(np.uint32), lens.astype(np.uint32), declined


def _swa_one(t, qs, qe, seq, blocks, valid, *, SW, BS, NB):
    rows, lens, declined = swa_indices(t, qs, qe, seq, blocks, valid, SW, BS, NB)
    if declined[0]:
        raise Declined("a live lane is masked, wraps or names a block past its row, where the kernel reads 0 or another row")
    return np.concatenate([rows[0], lens]).astype(np.uint64)


def _swa_batch(t, qs, qe, seq, blocks, valid, *, SW, BS, NB):
    rows, lens, declined = swa_indices(t, qs, qe, seq, blocks, valid, SW, BS, NB)
    return np.concatenate([rows, lens[:, None]], axis=1).astype(np.uint64), declined


def _c128a_one(position, slot, blocks, *, W, R, BS, NB):
    rows, lens, declined = c128a_global_topk(position, slot, blocks, W, R, BS, NB)
    if declined[0]:
        raise Declined("a live lane names a block past its row, which the kernel reads from the next request's")
    return np.concatenate([rows[0], lens]).astype(np.uint64)


def _c128a_batch(position, slot, blocks, *, W, R, BS, NB):
    rows, lens, declined = c128a_global_topk(position, slot, blocks, W, R, BS, NB)
    return np.concatenate([rows, lens[:, None]], axis=1).astype(np.uint64), declined


def block_rows(rng, n: int, NB: int) -> np.ndarray:
    """(n, NB) block-table rows as uint32: mostly small block numbers, the rest any int32 word."""
    return np.where(rng.random((n, NB)) < 0.7, rng.integers(0, 1 << 17, (n, NB)),
                    rng.integers(0, 1 << 32, (n, NB), dtype=np.uint64).astype(np.int64)).astype(np.uint32)


def swa_tokens(rng, n: int, SW: int, BS: int, NB: int) -> list[np.ndarray]:
    """n decode tokens' (t, qs, qe, seq) as uint32 words: a request of 1 to 4 query tokens (speculative decode's) whose position is
    an edge (0, the window's and a block's ends, the row's last position) or any position the row holds, and in a tenth of the
    tokens any int32 words (the int32 ends among them)."""
    span = NB * BS
    edges = np.asarray([e for e in (0, 1, SW - 2, SW - 1, SW, BS - 1, BS, BS + 1, span - SW, span - 2, span - 1) if 0 <= e < span])
    qlen = rng.integers(1, 5, n)
    qs = rng.integers(0, 1 << 16, n)
    t = qs + rng.integers(0, 1 << 16, n) % qlen
    pos = np.where(rng.random(n) < 0.4, edges[rng.integers(0, len(edges), n)], rng.integers(0, span, n))
    seq = pos + qlen - (t - qs)
    out = [t, qs, qs + qlen, seq]
    wild = rng.random(n) < 0.1
    ends = np.asarray([I32_MIN, I32_MIN + 1, I32_MIN + SW - 2, I32_MAX, I32_MAX - 1, -1], dtype=np.int64)
    for v in out:
        v[wild] = np.where(rng.random(int(wild.sum())) < 0.5, ends[rng.integers(0, len(ends), int(wild.sum()))],
                           rng.integers(0, 1 << 32, int(wild.sum()), dtype=np.uint64).astype(np.int64))
    return [(np.asarray(v, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint32) for v in out]


def _swa_sample(rng, n, params, *, SW, BS, NB):
    valid = np.where(rng.random(n) < 0.8, 1, np.where(rng.random(n) < 0.5, 0, rng.integers(0, 256, n)))
    return [*swa_tokens(rng, n, SW, BS, NB), block_rows(rng, n, NB), valid.astype(np.uint8)]


def c128a_positions(rng, n: int, W: int, R: int, BS: int, NB: int) -> np.ndarray:
    """n int64 positions as uint64 words: most give q = (position + 1) // R from 0 to min(W, BS * NB) (a lane count the row
    holds), at an offset in R; the rest the int64 ends, -1, and q at and past W."""
    q = rng.integers(0, min(W, BS * NB) + 1, n)
    pos = q * R + rng.integers(0, R, n) - 1
    ends = np.asarray([-1, -2, -R, -R - 1, I64_MIN, I64_MIN + 1, I64_MAX, I64_MAX - 1, W * R - 1, W * R + R, 1 << 40], dtype=np.int64)
    pos = np.where(rng.random(n) < 0.2, ends[rng.integers(0, len(ends), n)], pos)
    return u64(pos)


def _c128a_sample(rng, n, params, *, W, R, BS, NB):
    slot = np.where(rng.random(n) < 0.7, rng.integers(0, 1 << 40, n), np.asarray([-1, I64_MIN, 0, I64_MAX])[rng.integers(0, 4, n)])
    return [c128a_positions(rng, n, W, R, BS, NB), u64(slot), block_rows(rng, n, NB)]


def _targets(defs: dict) -> list:
    """Each row at max_model_len 4096's statics and at small ones (odd widths, and the c128a row with a block of lanes past its
    row); the window also narrower than a block."""
    from verity.ml.circuits.defs import bind

    swa = [dict(SW=128, BS=64, NB=64), dict(SW=8, BS=4, NB=8), dict(SW=5, BS=2, NB=3), dict(SW=3, BS=8, NB=4)]
    c128a = [dict(W=128, R=128, BS=2, NB=16), dict(W=16, R=4, BS=2, NB=8), dict(W=7, R=3, BS=1, NB=5),
             dict(W=12, R=3, BS=4, NB=2)]
    return ([bind(defs["Dsv4SwaIndices_v1"], **st) for st in swa]
            + [bind(defs["Dsv4C128aGlobalTopk_v1"], **st) for st in c128a])


def register_twin(defs: dict) -> None:
    """Register the twin; ``defs`` is `registry.dsv4_decode_metadata`' composites by id, which binds the self-check's
    specialisations (the registry calls this, so neither module imports the other both ways)."""
    register("Dsv4SwaIndices_v1", _swa_one, kernel=KERNEL, domain=SWA_DOMAIN, sample=_swa_sample, batch=_swa_batch,
             check_targets=lambda rng: [d for d in _targets(defs) if d.id.startswith("Dsv4SwaIndices_v1")])
    register("Dsv4C128aGlobalTopk_v1", _c128a_one, kernel=KERNEL, domain=C128A_DOMAIN, sample=_c128a_sample, batch=_c128a_batch,
             check_targets=lambda rng: [d for d in _targets(defs) if d.id.startswith("Dsv4C128aGlobalTopk_v1")])
