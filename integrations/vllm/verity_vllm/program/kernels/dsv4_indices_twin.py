"""numpy twin of `registry.dsv4_indices` (`combine_topk_swa_indices`, vLLM d9105ea80, Triton 3.7.1): the kernel's own stores
into a row first filled with -1, over a batch of tokens.

Written again from the kernel (common/ops/cache_utils.py:654-739) rather than from the registry: q by truncating division,
the top-k store over slots below topk_len, then the window's store slot by slot at topk_len + o under its mask o < swa_len,
where the registry writes the window as j < len.  A token whose window lands outside its row (pos = INT32_MAX, or pos below
INT32_MIN + WINDOW - 1) is declined: the kernel writes memory the row does not hold.  `registry.dsv4_indices` registers it
(`register_twin`) with `kernel_registry` as kernel ``dsv4-indices-twin``.
"""
from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels.kernel_registry import Declined, register

KERNEL = "dsv4-indices-twin"
DOMAIN = "every token whose window stays in its row"

I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1


def s32(x) -> np.ndarray:
    """Words (any integer dtype) as signed 32-bit values in int64."""
    x = np.asarray(x).astype(np.int64) & 0xFFFFFFFF
    return np.where(x >= 1 << 31, x - (1 << 32), x)


def combine(topk, pos, mb, n, gs, TOPK: int, RATIO: int, WINDOW: int, WIDTH: int):
    """(tokens, TOPK) top-k words and (tokens,) pos, mb = M*b, N and gather_start -> (rows (tokens, WIDTH) u32, lens (tokens,)
    u32, declined (tokens,) bool)."""
    topk = s32(topk).reshape(-1, TOPK)
    pos, mb, n, gs = (s32(v).reshape(-1) for v in (pos, mb, n, gs))
    t = s32(pos + 1)
    q = np.where(t < 0, -((-t) // RATIO), t // RATIO)
    topk_len = np.minimum(q, TOPK)
    start = np.maximum(s32(pos - (WINDOW - 1)), 0)
    swa_len = s32(pos - start + 1)
    rows = np.full((len(pos), WIDTH), -1, np.int64)
    j = np.arange(TOPK)
    hit = j[None, :] < topk_len[:, None]
    rows[:, :TOPK] = np.where(hit, s32(topk + mb[:, None]), rows[:, :TOPK])
    o = np.arange(WINDOW)
    live = o[None, :] < swa_len[:, None]
    slot = topk_len[:, None] + o[None, :]
    inside = (slot >= 0) & (slot < WIDTH)
    r, c = np.nonzero(live & inside)
    rows[r, slot[r, c]] = s32(mb + n + start - gs)[r] + c
    rows = s32(rows)
    declined = (live & ~inside).any(axis=1)
    return (rows & 0xFFFFFFFF).astype(np.uint32), (s32(topk_len + swa_len) & 0xFFFFFFFF).astype(np.uint32), declined


def _one(topk, pos, mb, n, gs, *, TOPK, RATIO, WINDOW, WIDTH):
    rows, lens, declined = combine(topk, pos, mb, n, gs, TOPK, RATIO, WINDOW, WIDTH)
    if declined[0]:
        raise Declined("its window lands outside its row (pos = INT32_MAX or below INT32_MIN + WINDOW - 1)")
    return np.concatenate([rows[0], lens]).astype(np.uint64)


def _batch(topk, pos, mb, n, gs, *, TOPK, RATIO, WINDOW, WIDTH):
    rows, lens, declined = combine(topk, pos, mb, n, gs, TOPK, RATIO, WINDOW, WIDTH)
    return np.concatenate([rows, lens[:, None]], axis=1).astype(np.uint64), declined


def positions(rng, n: int, TOPK: int, RATIO: int, WINDOW: int) -> np.ndarray:
    """n positions: each a special (around 0, the window, a ratio boundary, the top-k's end, the int32 ends, where the window
    leaves the row) or a draw below TOPK * RATIO * 2, as int32 words."""
    b = int(rng.integers(1, TOPK + 1)) * RATIO
    special = [-2, -1, 0, 1, WINDOW - 2, WINDOW - 1, WINDOW, b - 2, b - 1, b, TOPK * RATIO - 1, TOPK * RATIO, I32_MAX - 1,
               I32_MAX, I32_MIN, I32_MIN + WINDOW - 2, I32_MIN + WINDOW - 1, I32_MIN + WINDOW, I32_MIN + 2 * WINDOW]
    drawn = rng.integers(0, 2 * TOPK * RATIO, n)
    pick = rng.random(n) < 0.5
    out = np.where(pick, np.asarray(special)[rng.integers(0, len(special), n)], drawn)
    return (out & 0xFFFFFFFF).astype(np.uint32)


def _sample(rng, n, params, *, TOPK, RATIO, WINDOW, WIDTH):
    words = lambda shape: rng.integers(0, 1 << 32, shape, dtype=np.uint64).astype(np.uint32)  # noqa: E731
    topk = np.where(rng.random((n, TOPK)) < 0.5, rng.integers(-1, 1 << 16, (n, TOPK)) & 0xFFFFFFFF, words((n, TOPK)))
    scalars = [np.where(rng.random(n) < 0.5, rng.integers(0, 1 << 20, n), words(n)).astype(np.uint32) for _ in range(3)]
    return [topk.astype(np.uint32), positions(rng, n, TOPK, RATIO, WINDOW), *scalars]


def _targets(defs: dict) -> list:
    """The served ratio-4 row, ratio 128's narrowest, and two small ones (one of odd widths)."""
    from verity.ml.circuits.defs import bind

    statics = [dict(TOPK=1024, RATIO=4, WINDOW=128, WIDTH=1152), dict(TOPK=128, RATIO=128, WINDOW=128, WIDTH=256),
               dict(TOPK=8, RATIO=4, WINDOW=4, WIDTH=16), dict(TOPK=5, RATIO=2, WINDOW=3, WIDTH=9)]
    return [bind(defs["Dsv4CombineTopkSwa_v1"], **st) for st in statics]


def register_twin(defs: dict) -> None:
    """Register the twin; ``defs`` is `registry.dsv4_indices`' composites by id, which binds the self-check's
    specialisations (the registry calls this, so neither module imports the other both ways)."""
    register("Dsv4CombineTopkSwa_v1", _one, kernel=KERNEL, domain=DOMAIN, sample=_sample, batch=_batch,
             check_targets=lambda rng: _targets(defs))
