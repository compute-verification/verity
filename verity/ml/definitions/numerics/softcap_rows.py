"""The replay row kernel of FA2's softcapped chain under per-iteration `Check_inf`, `AttentionSoftcap_v2{T,NH,KVH,D,BN,CAP,DOT,INV,
MASKED_FROM}` (`registry.fa2_softcap`), and its vectorised twin.

`softcap_head_states` is `AttentionHeadSoftcap_v2` over rows: `derived_rows.attn_block`'s online softmax (the rescale guarded only at
blocks >= MASKED_FROM, `max * scale` in every block) with stage A' in front, `MufuTanh(F32MulFtz(s, softmax_scale / CAP))`, and the exp2
constant `CAP * log2e`.  `mufu_tanh` is `MufuTanh_v1` over arrays: its three rules, and the measured table's xz blocks decoded on first
use and kept for the process (a block is 4M words; a row's capped scores touch a few).  A head whose k16 chains meet a non-finite word is
flagged and the row declined, as `rows.attention` does; the reference computes it.  The row kernel registers from `rows` (kernel "rows"), which resolves DOT / INV.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics.kernel_registry import Declined

FAMILY = "AttentionSoftcap_v2"
EVALUATOR = ("verity.ml.definitions.numerics.softcap_rows.attention_softcap (AttentionSoftcap_v2{T,NH,KVH,D,BN,CAP,DOT,INV,MASKED_FROM}: "
             "Attention_v5's evaluator with MufuTanh(F32MulFtz(s, softmax_scale / CAP)) before the row max and exp2 constant CAP * log2e "
             "-- FA2's Is_softcap under per-iteration Check_inf, registry.fa2_softcap)")
_BLOCKS: dict[int, np.ndarray] = {}


def mufu_tanh(x: np.ndarray, table: np.ndarray | None = None, lo: int | None = None) -> np.ndarray:
    """`MufuTanh_v1` on every word of `x`: NaN -> nan_out, |x| >= saturate_from -> +-1, |x| < identity_below -> x, else the measured
    table at |x| with the sign of x.  `table` (the shard's words from `lo`) replaces the lazily decoded blocks."""
    from verity.ml.circuits.ops import mufu

    rules = mufu.MUFU_TANH_RULES
    u = np.asarray(x, dtype=np.uint32)
    a, sign = u & np.uint32(0x7FFFFFFF), u & np.uint32(0x80000000)
    out = u.copy()
    tab = (a >= rules["identity_below"]) & (a < rules["saturate_from"])
    if tab.any():
        if table is not None:
            out[tab] = np.asarray(table[(a[tab] - np.uint32(lo)).astype(np.int64)], dtype=np.uint32) | sign[tab]
        else:
            (s,) = mufu.mufu_tanh_shards()
            i = (a[tab] - np.uint32(s.lo)).astype(np.int64)
            blk, off = i // s.block_words, i % s.block_words
            got = np.empty(i.shape, dtype=np.uint32)
            for k in np.unique(blk):
                b = _BLOCKS.get(int(k))
                if b is None:
                    b = _BLOCKS[int(k)] = s.block(int(k))
                sel = blk == k
                got[sel] = b[off[sel]]
            out[tab] = got | sign[tab]
    sat = (a >= rules["saturate_from"]) & (a <= 0x7F800000)
    out[sat] = np.uint32(0x3F800000) | sign[sat]
    out[a > 0x7F800000] = np.uint32(rules["nan_out"])
    return out


def softcap_block(q, kb, vb, m_prev, l_prev, o_prev, *, D: int, NVIS: int, FIRST: bool, dot16, check: bool, softcap_mul: int,
                  scale_log2: int, tanh: Callable[[np.ndarray], np.ndarray]):
    """`AttnBlockSoftcap_v2{D, NVIS, FIRST, BN, CAP, DOT, CHECK}` vectorised over rows.  Returns (m_new, l, o, nonfinite)."""
    from verity.ml.definitions.registry import b1

    BN = kb.shape[1]
    Rn = q.shape[0]
    s, f = DR._dot_bf16_chain(q[:, None, :], kb[:, :NVIS, :], dot16)
    nf = f.any(axis=-1)
    s = tanh(DR.mul_ftz(s, np.uint32(softcap_mul))).reshape(Rn, NVIS)
    bmax = s[:, 0]
    for c in range(1, NVIS):
        bmax = DR.max_fmnmx(bmax, s[:, c])
    m_new = bmax if FIRST else DR.max_fmnmx(m_prev, bmax)
    m_use = DR.guard_neg_inf_zero(m_new)
    scale = np.full(Rn, scale_log2, dtype=np.uint32)
    if not FIRST:
        scores_scale = DR.mufu_ex2_ftz(DR.mul_ftz(DR.sub_ftz(m_prev, m_use if check else m_new), scale))
    max_scaled = DR.mul_ftz(m_use, scale)
    p = DR.mufu_ex2_ftz(DR.fma_sub_ftz(s, scale[:, None], max_scaled[:, None]))
    lanes = []
    for ln in range(4):
        cols = [c for c in range(NVIS) if b1._lane(c) == ln]
        first_p = p[:, cols[0]] if cols else np.zeros(Rn, dtype=np.uint32)
        acc = first_p if FIRST else DR.fma_ftz(scores_scale, l_prev[:, ln], first_p)
        for c in cols[1:]:
            acc = DR.add_ftz(acc, p[:, c])
        lanes.append(acc)
    pb = DR.f2fp_bf16(p)
    if NVIS < BN:
        pb = np.concatenate([pb, np.zeros((Rn, BN - NVIS), dtype=np.uint16)], axis=1)
    o = np.zeros((Rn, D), dtype=np.uint32) if FIRST else DR.mul_ftz(o_prev, scores_scale[:, None])
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break
        vt = np.swapaxes(vb[:, k0:k0 + 16, :], 1, 2)
        o, f = dot16(o, np.broadcast_to(pb[:, k0:k0 + 16][:, None, :], (Rn, D, 16)), vt)
        nf |= f.any(axis=-1)
    return m_new, np.stack(lanes, axis=1), o, nf


def softcap_head_states(q, k, v, BN: int, *, cap: float, masked_from: int, dot16, inv, tanh=mufu_tanh) -> dict[str, np.ndarray]:
    """`AttentionHeadSoftcap_v2{T, D, BN, CAP, DOT, INV, MASKED_FROM}` vectorised over rows: q (R, D), k / v (R, T, D) bf16 words.
    Returns ``out`` (R, D) bf16 words and ``nonfinite`` (R,)."""
    from verity.ml.definitions.registry.dense import softcap_mul_bits, softcap_scale_log2_bits

    q, k, v = (np.ascontiguousarray(x, dtype=np.uint16) for x in (q, k, v))
    Rn, T, Dh = k.shape
    nb = -(-T // BN)
    sm, sl2 = softcap_mul_bits(Dh, cap), softcap_scale_log2_bits(cap)
    nf = np.zeros(Rn, dtype=bool)
    m = l = o = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        kb, vb = k[:, lo:hi], v[:, lo:hi]
        if hi - lo < BN:
            pad = np.zeros((Rn, BN - (hi - lo), Dh), dtype=np.uint16)
            kb, vb = np.concatenate([kb, pad], axis=1), np.concatenate([vb, pad], axis=1)
        first = b == nb - 1
        z = np.zeros(Rn, dtype=np.uint32)
        m, l, o, f = softcap_block(q, kb, vb, z if first else m, np.zeros((Rn, 4), np.uint32) if first else l,
                                   np.zeros((Rn, Dh), np.uint32) if first else o, D=Dh, NVIS=hi - lo, FIRST=first, dot16=dot16,
                                   check=b >= masked_from, softcap_mul=sm, scale_log2=sl2, tanh=tanh)
        nf |= f
    total = DR.add_ftz(DR.add_ftz(l[:, 0], l[:, 2]), DR.add_ftz(l[:, 1], l[:, 3]))
    return {"out": DR.f2fp_bf16(DR.mul_ftz(o, inv(total)[:, None])), "nonfinite": nf}


def attention_softcap(statics: dict[str, Any], ops: Sequence[np.ndarray], dot_id: str, inv_id: str) -> dict[str, np.ndarray]:
    """One `AttentionSoftcap_v2` ROW: arg 0 = the query (NH*D words), args 1 / 2 = the T committed K / V rows (KVH*D words each), as
    `rows.attention`; head h reads KV head h // (NH/KVH).  `dot_id` / `inv_id` are the DOT / INV statics resolved by `kernel_registry.prim_static`."""
    T, NH, KVH, Dh, BN, MF = (int(statics[x]) for x in ("T", "NH", "KVH", "D", "BN", "MASKED_FROM"))
    cap = float(statics["CAP"])
    if len(ops) != 3:
        raise Declined(f"{FAMILY}: the row wires {len(ops)} args, the registered Definition takes 3 (q, K history, V history)")
    if dot_id not in DR.TC_DOT16_BY_PRIM or inv_id not in DR.INV_SUM_BY_PRIM:
        raise Declined(f"no registered evaluator for {FAMILY} with DOT={dot_id}, INV={inv_id}")
    if BN <= 0 or BN % 16 or Dh % 16 or NH % KVH:
        raise Declined(f"{FAMILY}: BN = {BN}, D = {Dh}, NH = {NH}, KVH = {KVH} outside the registered relation")
    q = np.asarray(ops[0], dtype=np.uint16).reshape(NH, Dh)
    kc, vc = np.asarray(ops[1], dtype=np.uint16), np.asarray(ops[2], dtype=np.uint16)
    if kc.size != T * KVH * Dh or vc.size != T * KVH * Dh:
        raise Declined(f"{FAMILY}: K history {kc.size} / V history {vc.size} words != T*KVH*D = {T}*{KVH}*{Dh}")
    kvh = np.arange(NH) // (NH // KVH)
    k = np.ascontiguousarray(np.transpose(kc.reshape(T, KVH, Dh), (1, 0, 2))[kvh])
    v = np.ascontiguousarray(np.transpose(vc.reshape(T, KVH, Dh), (1, 0, 2))[kvh])
    st = softcap_head_states(q, k, v, BN, cap=cap, masked_from=MF, dot16=DR.TC_DOT16_BY_PRIM[dot_id], inv=DR.INV_SUM_BY_PRIM[inv_id])
    if st["nonfinite"].any():
        raise Declined(f"{FAMILY}: {int(st['nonfinite'].sum())} heads have a non-finite operand / accumulator (outside the vectorised twin's domain)")
    return {"out": np.asarray(st["out"], dtype=np.uint16).reshape(NH * Dh)}


def check_targets(rng=None) -> list:
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import fa2_softcap, hopper
    from verity.ml.definitions.registry import prims as P

    # T = 40, BN = 16: three blocks; MASKED_FROM 2 leaves blocks 1 and 0 unguarded, 0 guards every block
    return [bind(fa2_softcap.AttentionSoftcapV2, T=40, NH=2, KVH=1, D=16, BN=16, CAP=50.0, DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum,
                 MASKED_FROM=m) for m in (2, 0)]


__all__ = ["FAMILY", "EVALUATOR", "attention_softcap", "check_targets", "mufu_tanh", "softcap_block", "softcap_head_states"]
