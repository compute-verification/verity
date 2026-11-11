"""The CUDA bodies of ``verity_int8::kv_quant_batch`` / ``kv_quant_window_batch`` and ``verity_int8::window_rows_batch``
(`engine.int8_ops`): one Triton launch each a layer step, bit for bit the torch bodies (`int8_attention.kv_quant_batch` over
`paged_windows` / `packed_windows`, and `int8_attention.batch_rows`).

`kv_rows`: a program takes one request b, one KV head g and a block of channels, and walks keys 0..s-1 of b's window
(s = lens[b]) where they are: in the paged caches under b's block-table row, or rows starts[b].. of a credited read's K and V.

    K'[b,g,i,c] = clamp(rint(RN32(f32(k[i,c]) / s_k[g])), -127, 127)        div.rn.f32, cvt.rni
    a[c]        = max_i |f32(v[i,c])|                                     exact (a max)
    s_v[b,g,c]  = RN32(max(a[c], RN32(1e-30)) / 127)                      div.rn.f32
    V'[b,g,i,c] = clamp(rint(RN32(f32(v[i,c]) / s_v[c])), -127, 127),  f32(V'(0)) its key 0's, 0 for an empty window

K' and V' are 0 at keys s..s_max-1, as the torch body's zero padding quantizes.

`lut_rows`: a program takes one query row (a token t, its request r = req[t] at position p = pos[t]) and one KV head g, its
`rep` query heads as the rows of a tile:

    Q'   = clamp(rint(RN32(f32(q) / s_q)), -127, 127), s_q = RN32(max(max_c |f32(q[c])|, RN32(1e-30)) / 127) per head
    A    = min(rint(((((s_q s_k) scale) log2(e)) 16) 2^32), 2^32 - 1)   mul.rn.f64 left to right, cvt.rni.s64.f64
    pass 1  S = Q' K'^T (tl.dot, s8 x s8 -> s32, exact) per key tile, m = max S over keys 0..p
    pass 2  S again, t = min(((m - S) A) >> 32, 255) in int64, e = LUT[t] (0 past p), L += sum e, e(0) at key 0, and
            Y += P V' (tl.dot, s8 x s8 -> s32) with P = e at keys 1..p (0 at key 0): P is in [0, 127]
    out  = RN_bf16(RN32(RN32(RN32(f32(Y) RN32(1 / L)) + RN32(RN32(e(0) / L) f32(V'(0)))) s_v))

Y, L and e(0) are integers, every sum exact in any order (|Y| <= 127^2 s_max < 2^31 for s_max <= MAX_KEYS); L <= 127 s_max is
exact in float32.  Triton's `/` on float32 is div.full.f32 and its libdevice calls flush to zero by default, so the kernels use
tl.div_rn and inline cvt.rni under `int8_fused.OPTS` (no FMA contraction, no flush to zero), as `int8_fused` does.
"""
from __future__ import annotations

import functools
import math

import torch
import triton
import triton.language as tl

from verity_vllm.engine import int8_attention as A
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine.int8_fused import OPTS

BLOCK_M = 16
BLOCK_N = 64
BLOCK_S = 64
BLOCK_C = 64


@triton.jit
def _rni_s32(v):
    return tl.inline_asm_elementwise("cvt.rni.s32.f32 $0, $1;", "=r,r", [v], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _rni_s64(v):
    return tl.inline_asm_elementwise("cvt.rni.s64.f64 $0, $1;", "=l,d", [v], dtype=tl.int64, is_pure=True, pack=1)


@triton.jit
def _q8(x, s):
    """clamp(rint(RN32(x / s)), -127, 127) as int8 (tl.div_rn takes operands of one shape)."""
    return tl.minimum(tl.maximum(_rni_s32(tl.div_rn(x, tl.broadcast_to(s, x.shape))), -127), 127).to(tl.int8)


@triton.jit
def _scale(amax, floor):
    """RN32(max(amax, floor) / 127)."""
    x = tl.maximum(amax, floor)
    return tl.div_rn(x, tl.full(x.shape, 127.0, tl.float32))


@triton.jit
def kv_rows(k_ptr, v_ptr, bt_ptr, start_ptr, len_ptr, ks_ptr, kq_ptr, vq_ptr, sv_ptr, v0_ptr, floor, D, S_MAX, BS,
            sk_n, sk_o, sk_g, sv_n, sv_o, sv_g, sbt, sq_b, sq_g, sq_s, ss_b, ss_g,
            PAGED: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1).to(tl.int64)
    cc = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = cc < D
    n = tl.load(len_ptr + b)
    sk = tl.load(ks_ptr + g)
    ii = tl.arange(0, BLOCK_S)
    out = b * sq_b + g * sq_g
    amax = tl.zeros((BLOCK_C,), tl.float32)
    for i0 in range(0, n, BLOCK_S):
        keys = i0 + ii
        ok = keys < n
        if PAGED:
            blk = tl.load(bt_ptr + b * sbt + keys // BS, mask=ok, other=0).to(tl.int64)
            kr, vr = blk * sk_n + (keys % BS) * sk_o, blk * sv_n + (keys % BS) * sv_o
        else:
            row = tl.load(start_ptr + b) + keys
            kr, vr = row * sk_n, row * sv_n
        m = ok[:, None] & c_ok[None, :]
        kf = tl.load(k_ptr + kr[:, None] + g * sk_g + cc[None, :], mask=m, other=0.0).to(tl.float32)
        tl.store(kq_ptr + out + keys[:, None] * sq_s + cc[None, :], _q8(kf, sk), mask=m)
        vf = tl.load(v_ptr + vr[:, None] + g * sv_g + cc[None, :], mask=m, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(vf), 0))
    sv = _scale(amax, floor)
    v0 = tl.zeros((BLOCK_C,), tl.int32)
    for i0 in range(0, n, BLOCK_S):
        keys = i0 + ii
        ok = keys < n
        if PAGED:
            blk = tl.load(bt_ptr + b * sbt + keys // BS, mask=ok, other=0).to(tl.int64)
            vr = blk * sv_n + (keys % BS) * sv_o
        else:
            vr = (tl.load(start_ptr + b) + keys) * sv_n
        m = ok[:, None] & c_ok[None, :]
        vf = tl.load(v_ptr + vr[:, None] + g * sv_g + cc[None, :], mask=m, other=0.0).to(tl.float32)
        vq = _q8(vf, sv[None, :])
        tl.store(vq_ptr + out + keys[:, None] * sq_s + cc[None, :], vq, mask=m)
        v0 += tl.sum(tl.where((keys == 0)[:, None], vq.to(tl.int32), 0), 0)
    for i0 in range(n, S_MAX, BLOCK_S):
        keys = i0 + ii
        m = (keys < S_MAX)[:, None] & c_ok[None, :]
        zero = tl.zeros((BLOCK_S, BLOCK_C), tl.int8)
        tl.store(kq_ptr + out + keys[:, None] * sq_s + cc[None, :], zero, mask=m)
        tl.store(vq_ptr + out + keys[:, None] * sq_s + cc[None, :], zero, mask=m)
    tl.store(sv_ptr + b * ss_b + g * ss_g + cc, sv, mask=c_ok)
    tl.store(v0_ptr + b * ss_b + g * ss_g + cc, v0.to(tl.float32), mask=c_ok)


@triton.jit
def lut_rows(q_ptr, k_ptr, v_ptr, sv_ptr, v0_ptr, ks_ptr, req_ptr, pos_ptr, f64_ptr, lut_ptr, o_ptr, floor, H, D, S_MAX, REP,
             sq_t, sq_h, sk_b, sk_g, sk_s, ss_b, ss_g, so_t, so_h,
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
    qm = row_ok[:, None] & d_ok[None, :]
    qf = tl.load(q_ptr + t * sq_t + heads[:, None] * sq_h + dd[None, :], mask=qm, other=0.0).to(tl.float32)
    s_q = _scale(tl.max(tl.abs(qf), 1), floor)
    q = _q8(qf, s_q[:, None])
    sigma = (s_q.to(tl.float64) * tl.load(ks_ptr + g).to(tl.float64)) * tl.load(f64_ptr)
    a = tl.minimum(_rni_s64(((sigma * tl.load(f64_ptr + 1)) * tl.load(f64_ptr + 2)) * tl.load(f64_ptr + 3)), 4294967295)
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
    lf = tl.where(row_ok, el, 1).to(tl.float32)
    rcp = tl.div_rn(tl.full((BLOCK_M,), 1.0, tl.float32), lf)
    z = tl.div_rn(e0.to(tl.float32), lf)
    sv = tl.load(sv_ptr + r * ss_b + g * ss_g + dd, mask=d_ok, other=0.0)
    v0 = tl.load(v0_ptr + r * ss_b + g * ss_g + dd, mask=d_ok, other=0.0)
    y = ((acc.to(tl.float32) * rcp[:, None]) + (z[:, None] * v0[None, :])) * sv[None, :]
    tl.store(o_ptr + t * so_t + heads[:, None] * so_h + dd[None, :], y.to(tl.bfloat16), mask=qm)


@functools.cache
def _consts(device: torch.device) -> tuple[torch.Tensor, float]:
    """The int32 LUT, and RN32(1e-30): the floor as the torch bodies' float32 clamp rounds it."""
    floor = float(torch.tensor(E.SCALE_FLOOR, dtype=torch.float32))
    return torch.as_tensor(A.LUT, dtype=torch.int32).to(device), floor


@functools.lru_cache(maxsize=8)
def _f64(scale: float, device: torch.device) -> torch.Tensor:
    """(scale, log2(e), 16, 2^32) as float64, the factors of A in `fixed_point_a`'s order."""
    return torch.tensor([float(scale), A.LOG2E, 16.0, 4294967296.0], dtype=torch.float64).to(device)


@functools.lru_cache(maxsize=8)
def _host(xs: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """A step's host list on the device, once a step (every layer of a step reads the same)."""
    return torch.tensor(xs, dtype=torch.int64).to(device)


def _kv_launch(k: torch.Tensor, v: torch.Tensor, bt: torch.Tensor | None, starts: list[int] | None, lens: list[int],
               k_scales: torch.Tensor, paged: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g_heads, d = k.shape[-2], k.shape[-1]
    b, s_max, dev = len(lens), max(lens, default=0), k.device
    if s_max > A.MAX_KEYS or d > A.MAX_HEAD_DIM or tuple(k_scales.shape) != (g_heads,) or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError(f"int8 attention: windows of lengths {lens[:8]} over {tuple(k.shape)} keys, {tuple(k_scales.shape)} K "
                         f"scales (head dim at most {A.MAX_HEAD_DIM}, at most {A.MAX_KEYS} keys)")
    kq = torch.empty((b, g_heads, s_max, d), dtype=torch.int8, device=dev)
    vq = torch.empty_like(kq)
    sv = torch.empty((b, g_heads, d), dtype=torch.float32, device=dev)
    v0 = torch.empty_like(sv)
    if b == 0:
        return kq, vq, sv, v0
    ln = _host(tuple(lens), dev)
    st = _host(tuple(starts), dev) if starts is not None else ln
    ks = k_scales.to(device=dev, dtype=torch.float32).contiguous()
    if paged:
        bs, (skn, sko, skg), (svn, svo, svg), sbt = k.shape[1], k.stride()[:3], v.stride()[:3], bt.stride(0)
    else:
        bs, (skn, skg), (svn, svg), sko, svo, sbt, bt = 1, k.stride()[:2], v.stride()[:2], 0, 0, 0, ln
    kv_rows[(b, g_heads, triton.cdiv(d, BLOCK_C))](
        k, v, bt, st, ln, ks, kq, vq, sv, v0, _consts(dev)[1], d, s_max, bs, skn, sko, skg, svn, svo, svg, sbt,
        kq.stride(0), kq.stride(1), kq.stride(2), sv.stride(0), sv.stride(1), PAGED=paged, BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C, **OPTS)
    return kq, vq, sv, v0


def kv_quant_paged(key_cache: torch.Tensor, value_cache: torch.Tensor, block_table: torch.Tensor, lens: list[int],
                   k_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`int8_attention.kv_quant_batch` of `paged_windows`, from the paged caches [num_blocks, block_size, G, d] in one launch."""
    if block_table.shape[0] < len(lens) or max(lens, default=0) > block_table.shape[1] * key_cache.shape[1]:
        raise ValueError(f"int8 attention: windows of lengths {lens[:8]} past a block table {tuple(block_table.shape)} of "
                         f"{key_cache.shape[1]}-token blocks")
    return _kv_launch(key_cache, value_cache, block_table, None, lens, k_scales, True)


def kv_quant_packed(keys: torch.Tensor, values: torch.Tensor, starts: list[int], lens: list[int],
                    k_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`int8_attention.kv_quant_batch` of `packed_windows`, from keys and values [rows, G, d] in one launch."""
    return _kv_launch(keys, values, None, starts, lens, k_scales, False)


def batch_rows(query: torch.Tensor, kq: torch.Tensor, vq: torch.Tensor, sv: torch.Tensor, v0: torch.Tensor, k_scales: torch.Tensor,
               req: torch.Tensor, pos: torch.Tensor, scale: float) -> torch.Tensor:
    """`int8_attention.batch_rows` in one launch."""
    t, h, d = query.shape
    b, g_heads, s_max, _ = kq.shape
    if h % g_heads or tuple(sv.shape) != (b, g_heads, d) or query.stride(-1) != 1 or not math.isfinite(scale):
        raise ValueError(f"int8 attention: {h} query heads over {g_heads} KV heads, results {tuple(kq.shape)} / {tuple(sv.shape)}")
    rep = h // g_heads
    out = torch.empty((t, h, d), dtype=torch.bfloat16, device=query.device)
    if t == 0:
        return out
    kq, vq, sv, v0 = kq.contiguous(), vq.contiguous(), sv.contiguous(), v0.contiguous()
    lut, floor = _consts(query.device)
    ks = k_scales.to(device=query.device, dtype=torch.float32).contiguous()
    lut_rows[(t, g_heads, triton.cdiv(rep, BLOCK_M))](
        query, kq, vq, sv, v0, ks, req, pos, _f64(float(scale), query.device), lut, out, floor, h, d, s_max, rep,
        query.stride(0), query.stride(1), kq.stride(0), kq.stride(1), kq.stride(2), sv.stride(0), sv.stride(1), out.stride(0),
        out.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=max(32, triton.next_power_of_2(d)), **OPTS)
    return out
