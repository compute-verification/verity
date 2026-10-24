"""`int8_attention.lut_core` on CUDA: one Triton launch for every query row of a step, the CUDA body of
``verity_int8::window_rows_batch`` (`engine.int8_ops`).

A program takes one query row (a token t, its request r = req[t] at position p = pos[t]) and one KV head g, its `rep` query
heads as the rows of a tile, and walks the keys 0..p of request r's K' and V' [B, G, s_max, d] (`kv_quant_batch`) twice:

    pass 1  S = Q' K'^T (tl.dot, s8 x s8 -> s32, exact) per key tile, m = max S over keys 0..p
    pass 2  S again, t = min(((m - S) A) >> 32, 255) in int64, e = LUT[t] (0 past p), L += sum e, e(0) at key 0, and
            Y += P V' (tl.dot, s8 x s8 -> s32) with P = e at keys 1..p (0 at key 0): P is in [0, 127]

so Y, L and e(0) are `lut_core`'s integers, every sum exact in any order (|Y| <= 127^2 s_max < 2^31 for s_max <= MAX_KEYS).
Nothing here is floating point; `int8_attention.batch_rows` applies the IEEE steps around it on both devices.
"""
from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from verity_vllm.engine import int8_attention as A

BLOCK_M = 16
BLOCK_N = 64


@triton.jit
def lut_rows(q_ptr, k_ptr, v_ptr, req_ptr, pos_ptr, a_ptr, lut_ptr, y_ptr, l_ptr, e0_ptr, H, D, S_MAX, REP,
             sq_t, sq_h, sk_b, sk_g, sk_s, sa_t, sy_t, sy_h, sl_t,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1).to(tl.int64)
    m0 = tl.program_id(2) * BLOCK_M
    r = tl.load(req_ptr + t)
    p = tl.load(pos_ptr + t)
    rows = m0 + tl.arange(0, BLOCK_M)
    heads = g * REP + rows
    row_ok = rows < REP
    dd = tl.arange(0, BLOCK_D)
    d_ok = dd < D
    q = tl.load(q_ptr + t * sq_t + heads[:, None] * sq_h + dd[None, :], mask=row_ok[:, None] & d_ok[None, :], other=0)
    a = tl.load(a_ptr + t * sa_t + heads, mask=row_ok, other=0)
    kbase = k_ptr + r * sk_b + g * sk_g
    vbase = v_ptr + r * sk_b + g * sk_g
    nn = tl.arange(0, BLOCK_N)
    m = tl.full((BLOCK_M,), -2147483647 - 1, tl.int32)
    for n0 in range(0, p + 1, BLOCK_N):
        keys = n0 + nn
        k_ok = (keys <= p) & (keys < S_MAX)
        kt = tl.load(kbase + keys[None, :] * sk_s + dd[:, None], mask=k_ok[None, :] & d_ok[:, None], other=0)
        s = tl.dot(q, kt, out_dtype=tl.int32)
        m = tl.maximum(m, tl.max(tl.where(k_ok[None, :], s, -2147483647 - 1), 1))
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.int32)
    el = tl.zeros((BLOCK_M,), tl.int32)
    e0 = tl.zeros((BLOCK_M,), tl.int32)
    for n0 in range(0, p + 1, BLOCK_N):
        keys = n0 + nn
        k_ok = (keys <= p) & (keys < S_MAX)
        kt = tl.load(kbase + keys[None, :] * sk_s + dd[:, None], mask=k_ok[None, :] & d_ok[:, None], other=0)
        s = tl.dot(q, kt, out_dtype=tl.int32)
        diff = tl.where(k_ok[None, :], m[:, None] - s, 0).to(tl.int64)
        ti = tl.minimum((diff * a[:, None]) >> 32, 255)
        e = tl.where(k_ok[None, :], tl.load(lut_ptr + ti), 0)
        el += tl.sum(e, 1)
        e0 += tl.sum(tl.where(keys[None, :] == 0, e, 0), 1)
        pt = tl.where(keys[None, :] == 0, 0, e).to(tl.int8)
        vt = tl.load(vbase + keys[:, None] * sk_s + dd[None, :], mask=k_ok[:, None] & d_ok[None, :], other=0)
        acc += tl.dot(pt, vt, out_dtype=tl.int32)
    tl.store(y_ptr + t * sy_t + heads[:, None] * sy_h + dd[None, :], acc, mask=row_ok[:, None] & d_ok[None, :])
    tl.store(l_ptr + t * sl_t + heads, el, mask=row_ok)
    tl.store(e0_ptr + t * sl_t + heads, e0, mask=row_ok)


@functools.cache
def _lut(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(A.LUT, dtype=torch.int32).to(device)


def lut_core(qi: torch.Tensor, kq: torch.Tensor, vq: torch.Tensor, req: torch.Tensor, pos: torch.Tensor,
             a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`int8_attention.lut_core` (Y int32 [T, H, d], L and e(0) int32 [T, H]) in one launch."""
    t, h, d = qi.shape
    b, g_heads, s_max, _ = kq.shape
    rep = h // g_heads
    qi, kq, vq, a = qi.contiguous(), kq.contiguous(), vq.contiguous(), a.contiguous()
    y = torch.empty((t, h, d), dtype=torch.int32, device=qi.device)
    el = torch.empty((t, h), dtype=torch.int32, device=qi.device)
    e0 = torch.empty_like(el)
    if t == 0:
        return y, el, e0
    grid = (t, g_heads, triton.cdiv(rep, BLOCK_M))
    lut_rows[grid](qi, kq, vq, req, pos, a, _lut(qi.device), y, el, e0, h, d, s_max, rep,
                   qi.stride(0), qi.stride(1), kq.stride(0), kq.stride(1), kq.stride(2), a.stride(0), y.stride(0), y.stride(1),
                   el.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=max(32, triton.next_power_of_2(d)))
    return y, el, e0
