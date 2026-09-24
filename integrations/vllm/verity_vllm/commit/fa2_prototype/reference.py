"""Deterministic NumPy attention reference in the SHAPE of the Verity semantics (`b1.AttentionHeadV3`), used as the
replay producer of the value stream defined in :mod:`encoding`.

Same dataflow as `AttnBlockV3` / `AttentionHeadV3`: 128-key blocks visited LAST -> FIRST, `S = q.k` accumulated in fp32,
running max, `alpha = exp2((m_prev - m_new) * scale_log2)`, `P = exp2(S * scale_log2 - m_new * scale_log2)`, four lane
sums (`lane(c) = (c % 8) // 2`, sequential per lane), `P -> bf16 (RNE)` before the PV product, `O = O * alpha + sum_c
P16[c] * v[c]` in fp32, epilogue `total = (l0 + l2) + (l1 + l3)`, `inv = 1 / total`, `out = bf16(O * inv)`.

NOT claimed: bit-exactness with the FA2 kernel.  The exact model (MUFU.EX2 / MUFU.RCP tables, FTZ on every op, the
`mma.sync` accumulation model, the FFMA contraction of the rescale with the first lane add) is the numerics lane's
`fa2_model.cpp` (`docs/vllm-poc/attention-model.md`) and lane B-sem's `verity_attn_ref.py`; this module is a deterministic
stand-in with the same VALUE IDENTITIES so that the commitment mechanics can be tested on a laptop.  The stream it produces
is deterministic on one host (fixed reduction order via `cumsum`), which is all replay-to-open needs here.
"""

from __future__ import annotations

import numpy as np

from .encoding import NLANES, Geometry, RowLayout, lane_of

M_LOG2E = 1.4426950408889634


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (np.asarray(u16, dtype=np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16_rne(f32: np.ndarray) -> np.ndarray:
    """RNE fp32 -> bf16 bit pattern (finite inputs; NaN -> 0x7fff like `F2FP.BF16.PACK_AB`)."""
    bits = np.ascontiguousarray(np.asarray(f32, dtype=np.float32)).view(np.uint32).astype(np.uint64)
    lsb = (bits >> np.uint64(16)) & np.uint64(1)
    rounded = (bits + np.uint64(0x7FFF) + lsb) >> np.uint64(16)
    out = rounded.astype(np.uint16)
    nan = np.isnan(np.asarray(f32, dtype=np.float32))
    if nan.any():
        out = np.where(nan, np.uint16(0x7FFF), out)
    return out


def f32_bits(x: np.ndarray | np.floating) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32)).view(np.uint32)


def f32_bit(x: np.floating | float) -> int:
    return int(np.asarray(np.float32(x)).view(np.uint32))


def scale_log2(D: int) -> np.float32:
    """FA2 `set_params_fprop`: `float(softmax_scale) * M_LOG2E` in double, narrowed to fp32 (`b1._scale_log2_bits`)."""
    s = float(np.float32(D ** -0.5))
    return np.float32(s * M_LOG2E)


def _seq_sum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Sequential fp32 accumulation (deterministic order): the last element of a cumsum."""
    if x.shape[axis] == 0:
        return np.zeros(x.shape[:axis] + x.shape[axis + 1:], dtype=np.float32) if x.ndim > 1 else np.float32(0)
    return np.take(np.cumsum(x, axis=axis, dtype=np.float32), -1, axis=axis)


def replay_row(layout: RowLayout, q16: np.ndarray, k16: np.ndarray, v16: np.ndarray, *, upto_version: int | None = None
               ) -> tuple[np.ndarray, np.ndarray | None]:
    """The committed stream of one row as `uint32` words in layout order, and the bf16 output row (`None` when the
    replay is cut at ``upto_version``: only the blocks visited at versions <= upto_version are produced, which is all a
    k-block subtree opening needs).  ``q16``: (D,) bf16 bits; ``k16``/``v16``: (T, D) bf16 bits of the visible keys."""
    T, D, BN, nb = layout.T, layout.D, layout.BN, layout.nb
    if q16.shape != (D,) or k16.shape[0] < T or v16.shape[0] < T:
        raise ValueError("fixture shape does not match the row layout")
    q = bf16_to_f32(q16)
    sl2 = scale_log2(D)
    words = np.zeros(layout.count, dtype=np.uint32)
    m_prev = np.float32(0)
    lanes = np.zeros(NLANES, dtype=np.float32)
    o = np.zeros(D, dtype=np.float32)
    last_v = nb - 1 if upto_version is None else min(upto_version, nb - 1)
    for v in range(last_v + 1):
        b = nb - 1 - v
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb = bf16_to_f32(k16[lo:hi])            # (nvis, D)
        vb = bf16_to_f32(v16[lo:hi])
        s = _seq_sum(q[None, :] * kb, axis=1)   # fp32 products of bf16 are exact; sequential k accumulation
        bmax = np.float32(s.max())
        m_new = bmax if v == 0 else np.float32(max(m_prev, bmax))
        max_scaled = np.float32(m_new * sl2)
        p = np.exp2((s * sl2 - max_scaled).astype(np.float32)).astype(np.float32)
        cols = np.arange(lo, hi)
        new_lanes = np.zeros(NLANES, dtype=np.float32)
        if v > 0:
            alpha = np.float32(np.exp2(np.float32((m_prev - m_new) * sl2)))
        for ln in range(NLANES):
            sel = p[lane_of(cols) == ln]
            if v == 0:
                acc = np.float32(sel[0]) if sel.size else np.float32(0)
                rest = sel[1:]
            else:
                first = np.float32(sel[0]) if sel.size else np.float32(0)
                acc = np.float32(alpha * lanes[ln] + first)   # the kernel contracts this into one FFMA; approximated as mul+add
                rest = sel[1:]
            if rest.size:
                acc = np.float32(_seq_sum(np.concatenate([[acc], rest]).astype(np.float32)))
            new_lanes[ln] = acc
        p16 = bf16_to_f32(f32_to_bf16_rne(p))
        if v > 0:
            o = (o * alpha).astype(np.float32)
        pv = _seq_sum((p16[:, None] * vb).astype(np.float32), axis=0)   # (D,), sequential over keys
        o = (o + pv).astype(np.float32)
        span_lo, _ = layout.block_span(v)
        cur = span_lo
        words[cur:cur + nvis] = f32_bits(s); cur += nvis
        words[cur] = f32_bit(m_new); cur += 1
        if v > 0:
            words[cur] = f32_bit(alpha); cur += 1
        words[cur:cur + nvis] = f32_bits(p); cur += nvis
        words[cur:cur + NLANES] = f32_bits(new_lanes); cur += NLANES
        words[cur:cur + D] = f32_bits(o); cur += D
        assert cur == layout.block_span(v)[1]
        m_prev, lanes = m_new, new_lanes
    if upto_version is not None and upto_version < nb - 1:
        return words, None
    total = np.float32(np.float32(lanes[0] + lanes[2]) + np.float32(lanes[1] + lanes[3]))
    inv = np.float32(1) if (total == 0 or np.isnan(total)) else np.float32(np.float32(1) / total)
    words[layout.count - 1] = f32_bit(inv)
    out16 = f32_to_bf16_rne((o * inv).astype(np.float32))
    return words, out16
