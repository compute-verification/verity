"""The replay row kernels of the Marlin MoE align (`registry.marlin_moe`), over all P pairs of one align call at once in signed i32
words: `MoeAlignRanks_v1{E,P,BM}` (each pair's rank in its expert, ordered by (key, index), and whether it opens a moe block) and
`MoeAlignBlocks_v1{P,BM}` (each pair's moe block and the launch's num_tokens_post_padded / BM).  Sums wrap at 32 bits and `/` truncates,
as the Definitions' `I32Add` and `I32Div` do.  The row kernels register from `rows` (kernel "rows").
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

EVALUATORS = {
    "MoeAlignRanks_v1": "verity.ml.definitions.numerics.moe_align_rows.moe_align_ranks (MoeAlignRank_v1: the pairs of p's expert before it by "
                        "signed (key, index); MoeAlignStart_v1: its expert in [0, E) and its rank a multiple of BM) -> rank, start",
    "MoeAlignBlocks_v1": "verity.ml.definitions.numerics.moe_align_rows.moe_align_blocks (MoeAlignBlockOf_v1: the starts of the pairs whose "
                         "expert is below p's, plus rank / BM; MoeAlignPar_v1: the starts' sum) -> blk, par",
}


def _i32(op: np.ndarray, P: int) -> np.ndarray:
    a = np.asarray(op).reshape(-1)
    if a.size != P:
        raise ValueError(f"an operand of {a.size} words, {P} expected")
    return (a.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32).view(np.int32).astype(np.int64)


def _word(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint32)


def moe_align_ranks(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MoeAlignRanks_v1{E,P,BM}(ids, key, ix)` -> ports 0 (rank) and 1 (start), P words each."""
    E, P, BM = (int(statics[k]) for k in ("E", "P", "BM"))
    e, k, i = (_i32(ops[g], P) for g in range(3))
    before = (e[None, :] == e[:, None]) & ((k[None, :] < k[:, None]) | ((k[None, :] == k[:, None]) & (i[None, :] < i[:, None])))
    rank = before.sum(axis=1)
    start = (0 <= e) & (e < E) & (rank % BM == 0)
    return {"0": _word(rank), "1": _word(start)}


def moe_align_blocks(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MoeAlignBlocks_v1{P,BM}(ids, rank, start)` -> ports 0 (blk, P words) and 1 (par, one word)."""
    P, BM = int(statics["P"]), int(statics["BM"])
    e, rank, start = (_i32(ops[g], P) for g in range(3))
    below = ((e[None, :] < e[:, None]) * start[None, :]).sum(axis=1)
    return {"0": _word(below + np.sign(rank) * (np.abs(rank) // BM)), "1": _word(start.sum(keepdims=True))}


def _sample(rng: np.random.Generator, n: int, params: Sequence[Any], **statics: Any) -> list[np.ndarray]:
    """ids mostly in [0, E) (one in eight outside it, -1 or E), keys and indices over a few values (ties), and for the blocks ranks in
    [-BM, P) and starts mostly 0 / 1 (one in eight any word)."""
    P, E = int(statics["P"]), int(statics.get("E", 4))
    ids = np.where(rng.random((n, P)) < 1 / 8, rng.choice([-1, E], (n, P)), rng.integers(0, E, (n, P)))
    if "E" in statics:
        out = [ids, rng.integers(-2, 3, (n, P)), rng.integers(0, P, (n, P))]
    else:
        BM = int(statics["BM"])
        start = np.where(rng.random((n, P)) < 1 / 8, rng.integers(-(1 << 31), 1 << 31, (n, P)), rng.integers(0, 2, (n, P)))
        out = [ids, rng.integers(-BM, P, (n, P)), start]
    return [_word(a) for a in out]


#: family -> (row kernel, the members it returns in the Definition's return order)
ROWS: dict[str, tuple[Callable[..., dict[str, np.ndarray]], tuple[str, ...]]] = {
    "MoeAlignRanks_v1": (moe_align_ranks, ("0", "1")),
    "MoeAlignBlocks_v1": (moe_align_blocks, ("0", "1")),
}


def check_targets(family: str) -> list:
    """The small specialisations `family`'s self-check binds: a moe block of 2 and of 4 over 8 and 12 pairs."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import marlin_moe as MM
    return {
        "MoeAlignRanks_v1": [bind(MM.MoeAlignRanks, E=3, P=8, BM=2), bind(MM.MoeAlignRanks, E=4, P=12, BM=4)],
        "MoeAlignBlocks_v1": [bind(MM.MoeAlignBlocks, P=8, BM=2), bind(MM.MoeAlignBlocks, P=12, BM=4)],
    }[family]


def register_rows(evaluators: dict, row: Callable) -> None:
    """Register this module's row evaluators without growing the shared rows catalog."""
    evaluators |= EVALUATORS
    for family, (fn, returns) in ROWS.items():
        row(family, fn, returns, sample=_sample, check_targets=lambda rng, f=family: check_targets(f))


__all__ = ["EVALUATORS", "ROWS", "check_targets", "moe_align_blocks", "moe_align_ranks", "register_rows"]
