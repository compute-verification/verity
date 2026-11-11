"""The attention family's vocabulary: FlashAttention's launch rules for the `MASKED_FROM` static of the per-iteration `Check_inf`
chains (`fa2_check_inf.Attention_v5` / `_v11`, `fa2_softcap.AttentionSoftcap_v2`, `fa3_check_inf.Attention_v4`).

The Definitions are generic over `MASKED_FROM`; the lowest key block a served row's kernel guards is a fact of the launch (its tile,
its causality, the sequence's query and key counts), which `targets.attention_spec` reads here to bind it.
"""
from __future__ import annotations


#: FA2's splitkv tile height (`run_mha_fwd_splitkv_align`: kBlockM = 64 at every head dim)
KBLOCK_M = 64


def fa2_masked_from(seqlen_k: int, seqlen_q: int, row: int, BN: int, causal: bool) -> int:
    """The lowest key block FA2's splitkv kernel runs with `Check_inf` for query row `row` of a sequence of `seqlen_q` query rows over
    `seqlen_k` keys: `n_block_max - n_masking_steps`, where a causal launch's `n_block_max` is its 64-row tile's,
    `min(ceil(seqlen_k / BN), ceil(((m_block + 1) * 64 + seqlen_k - seqlen_q) / BN))`, and a non-causal launch guards its first block only."""
    seqlen_k, seqlen_q, row, BN = int(seqlen_k), int(seqlen_q), int(row), int(BN)
    if not (0 <= row < seqlen_q <= seqlen_k) or BN < 1:
        raise ValueError(f"FA2 row {row} of {seqlen_q} query rows over {seqlen_k} keys (BN {BN})")
    nb = -(-seqlen_k // BN)
    if not causal:
        if seqlen_q != 1:
            raise ValueError(f"FA2: a non-causal launch has one query row per sequence, not {seqlen_q}")
        return nb - 1
    n_block_max = min(nb, -(-((row // KBLOCK_M + 1) * KBLOCK_M + seqlen_k - seqlen_q) // BN))
    return max(0, n_block_max - (-(-KBLOCK_M // BN) + 1))


def fa3_masked_from(T: int, BN: int, row: int, kBM: int, g: int) -> int:
    """`n_block_min_causal_local_mask` of the FA3 tile holding query row `row` of its sequence (causal, not local, one context-parallel
    rank): `(m_idx_min + seqlen_k - seqlen_q) / kBlockN`, with `m_idx_min` the tile's first query row (PackGQA: packed row `row * g + h`,
    `m_idx_min = (m_block * kBlockM) / g`) and `seqlen_k - seqlen_q = T - 1 - row` (the row sees T keys).  Refuses a group size that does
    not divide kBlockM (a row's heads would straddle two tiles)."""
    T, BN, row, kBM, g = int(T), int(BN), int(row), int(kBM), int(g)
    if g < 1 or kBM % g:
        raise ValueError(f"FA3 tile: qheads_per_kvhead {g} does not divide kBlockM {kBM}")
    if not 0 <= row < T:
        raise ValueError(f"FA3 row {row} outside the {T} keys it sees")
    m_idx_min = (row * g // kBM) * kBM // g
    return (m_idx_min + T - 1 - row) // BN


__all__ = ["KBLOCK_M", "fa2_masked_from", "fa3_masked_from"]
