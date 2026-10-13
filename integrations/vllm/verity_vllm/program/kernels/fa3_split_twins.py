"""Numpy twins of `registry.fa3_split`'s Definitions: `head_split_rows` (`AttentionHeadSplit_v1`: its blocks `derived_rows.attn_block`,
its split end and combine `sparse_mla_split_twins`'), checked word for word against the Definitions in
`tests/program/test_fa3_split.py`, and `split_call`, the captures' fast path: one sequence's rows split by split, each split through
the C++ row model (`fa3_model.attention_rows` over the split's key range, its debug state), then the same end and combine, checked
against `head_split_rows`.  A head whose dot reads a non-finite operand is flagged ``nonfinite``, and one with a split sum whose
MUFU.LG2 the model leaves open (`sparse_mla_split_twins.undetermined_words`) is flagged ``open``; neither is claimed exact."""
from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.kernels import fa3_model as FM
from verity_vllm.program.kernels import sparse_mla_split_twins as TW
from verity_vllm.program.registry import fa3_split as FS
from verity_vllm.program.registry import sparse_mla_split as W

U32 = np.uint32


def _inf_nan(x) -> np.ndarray:
    return (np.asarray(x, dtype=np.uint16) & 0x7F80) == 0x7F80


def _sum(l: np.ndarray) -> np.ndarray:
    return DR.add_ftz(DR.add_ftz(l[..., 0], l[..., 2]), DR.add_ftz(l[..., 1], l[..., 3]))


def end_rows(m, l, o, *, D: int, inv=DR.fa3_inv_sum) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`SparseMlaSplitEnd_v1{DV = D, INV, SCALE = D^-0.5}` on m (R,), l (R, 4), o (R, D) -> (partial o (R, D), lse (R,), open (R,))."""
    part, lse = TW.split_end_rows(m, l, o, scale=FS.scale(D), inv=inv)
    return part, lse, TW.undetermined_words(_sum(np.asarray(l, dtype=U32)))


def head_split_rows(q, k, v, *, T: int, NBM: int, NS: int, BN: int, MASKED_FROM: int, dot16=DR.tc_dot16_hopper,
                    inv=DR.fa3_inv_sum) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`AttentionHeadSplit_v1{T, NBM, NS, D, BN, DOT, INV, MASKED_FROM}`: q (R, D), k / v (R, T, D) or (T, D) bf16 words -> (out (R, D)
    bf16 words, nonfinite (R,), open (R,))."""
    FS.check_split_statics("head_split_rows", T, NBM, NS, BN, MASKED_FROM)
    q, k, v = (np.asarray(a, dtype=np.uint16) for a in (q, k, v))
    R, D = q.shape
    if k.ndim == 2:
        k, v = np.broadcast_to(k, (R,) + k.shape), np.broadcast_to(v, (R,) + v.shape)
    if k.shape[1:] != (T, D) or v.shape != k.shape:
        raise ValueError(f"head_split_rows: k {k.shape[1:]} / v {v.shape[1:]}, T={T} rows of {D}")
    nf, opn = np.zeros(R, dtype=bool), np.zeros(R, dtype=bool)
    lses, parts = [], []
    for b0, e in FS.split_ranges(T, NBM, NS, BN):
        m = l = o = None
        for b in range(e - 1, b0 - 1, -1):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            kb, vb = (np.zeros((R, BN, D), dtype=np.uint16) for _ in range(2))
            kb[:, :hi - lo], vb[:, :hi - lo] = k[:, lo:hi], v[:, lo:hi]
            first = b == e - 1
            m, l, o, f = DR.attn_block(q, kb, vb, m, l, o, D=D, NVIS=hi - lo, FIRST=first, dot16=dot16,
                                       check=first or b >= MASKED_FROM)
            nf |= f
        part, lse, op = end_rows(m, l, o, D=D, inv=inv)
        parts.append(part)
        lses.append(lse)
        opn |= op
    return TW.combine_rows(np.stack(lses, axis=1), np.stack(parts, axis=1), inv=inv), nf, opn


def split_call(q, k, v, rows, L: int, NS: int, tables, *, kblock_n: int = FM.KBLOCK_N_HD256, kblock_m: int = FM.KBLOCK_M_PREFILL,
               inv=DR.fa3_inv_sum) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One causal self-attention sequence of L tokens that FA3 runs in NS >= 2 splits: q (len(rows), H, D) words of its query rows
    ``rows`` (0-based), k / v (>= L, KVH, D) -> (out (len(rows), H, D) bf16 words, nonfinite (len(rows), H), open (len(rows), H)).
    Head h of row r is `AttentionHeadSplit_v1{T = r + 1, NBM = its tile's n_block_max (`fa3_model.kernel_geometry`, the packed
    rows' M tile), NS, ...}` reading KV head h // (H / KVH).  The C++ model guards the running max on every block, which equals
    `AttnBlock_v4`'s unguarded iterations on a finite running max; a head it reaches non-finite is flagged, and so is one whose
    query or any key or value row it sees holds a non-finite word (the C++ model's flag misses some of those)."""
    q = np.asarray(q, dtype=np.uint16)
    nrow, H, D = q.shape
    KVH, BN = k.shape[1], int(kblock_n)
    if not 2 <= int(NS) <= W.LANES:
        raise ValueError(f"split_call: NS={NS} (2 to {W.LANES}; one split is fa3_model.attention_call)")
    rows = np.asarray(rows, dtype=np.int64)
    T = rows + 1
    kvh = (np.arange(H, dtype=np.int32) // (H // KVH))
    seen = np.logical_or.accumulate((_inf_nan(k[:L]) | _inf_nan(v[:L])).any(axis=2), axis=0)
    nf = _inf_nan(q).any(axis=2) | seen[rows][:, kvh]
    _, nbm = FM.kernel_geometry(L, L, rows, BN, kblock_m=kblock_m, qh_per_kvh=H // KVH)
    ranges = [FS.split_ranges(int(t), int(nb), int(NS), BN) for t, nb in zip(T, nbm)]
    calls: dict[int, list[tuple[int, int, int]]] = {}
    for i, rr in enumerate(ranges):
        for j, (b0, e) in enumerate(rr):
            calls.setdefault(b0, []).append((i, j, e))
    m, l = np.zeros((nrow, H, NS), U32), np.zeros((nrow, H, NS, 4), U32)
    o = np.zeros((nrow, H, NS, D), U32)
    for b0, items in sorted(calls.items()):
        ii, jj, ee = (np.array(x, dtype=np.int64) for x in zip(*items))
        col = np.minimum(T[ii], ee * BN) - b0 * BN
        res = FM.attention_rows(np.ascontiguousarray(q[ii].reshape(-1, D)), k[b0 * BN:L], v[b0 * BN:L], np.tile(kvh, len(ii)),
                                np.repeat(col, H), np.repeat(ee - b0, H), FS.scale(D), tables, debug=True, kblock_n=BN)
        m[ii, :, jj] = res.dbg["m"][:, 0].reshape(len(ii), H)
        l[ii, :, jj] = res.dbg["l"].reshape(len(ii), H, 4)
        o[ii, :, jj] = res.dbg["o_pre"].reshape(len(ii), H, D)
        nf[ii] |= res.nonfinite.reshape(len(ii), H)
    part, lse, opn = end_rows(m.reshape(-1), l.reshape(-1, 4), o.reshape(-1, D), D=D, inv=inv)
    part, lse = part.reshape(nrow, H, NS, D), lse.reshape(nrow, H, NS)
    nsp = np.array([len(rr) for rr in ranges])
    used = np.arange(NS)[None, :] < nsp[:, None]
    out = np.zeros((nrow, H, D), dtype=np.uint16)
    for n in np.unique(nsp):
        ii = np.flatnonzero(nsp == n)
        out[ii] = TW.combine_rows(lse[ii, :, :n].reshape(-1, n), part[ii, :, :n].reshape(-1, n, D), inv=inv).reshape(len(ii), H, D)
    return out, nf, (opn.reshape(nrow, H, NS) & used[:, None, :]).any(axis=2)


__all__ = ["end_rows", "head_split_rows", "split_call"]
