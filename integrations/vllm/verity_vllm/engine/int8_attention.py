"""Exact-int8 attention: compute accounting's integer LUT softmax (``benchmarks/pouw/edited_gemm/fattn_ref.py``, and the output of
``apv_ref.py``'s served f with no dither), on a request's window, in plain torch (``engine.int8_attention_vllm`` serves it inside
vLLM; ``TargetProfile.int8_exact`` declares it).

A window is a request's tokens 0..s-1 (its KV cache: K after RoPE and V, both bf16); a call computes the window's last n query
rows (positions s-n..s-1).  A whole-prompt prefill is n = s; a decode step is n = 1, the last row of the window over every token
so far; a chunked prefill's chunk is the n rows it brings, over the s tokens cached by its end.  For every KV group g (its `rep` query heads; RN32 / RN64 = float32 / float64 round to nearest even, one rounding per
operation; every division a true IEEE division):

* Q [rep, n, d], one scale per (head, token): s_q = RN32(max(max |q|, RN32(1e-30)) / 127), Q' = clamp(round(RN32(q / s_q)),
  -127, 127) (`round`: to the nearest integer, ties to even);
* K [s, d], ONE scale per (layer, KV head), s_k (`engine.int8_k_scales`, calibrated offline): K' = clamp(round(RN32(k / s_k)),
  -127, 127);
* S = Q' K'^T exactly (one ``torch._int_mm`` over the group's rep heads stacked, int32: |S| <= d 127^2);
* A = min(rint(RN64(RN64(RN64(RN64(s_q s_k) scale) log2(e)) 16) 2^32), 2^32 - 1) per query row, fattn.cu's `fa_A` with
  sigma = s_q s_k scale (the logit of one unit of S): float64, left to right, rint ties to even;
* for the row at position pos: m = max S over keys 0..pos, t = min(((m - S) A) >> 32, 255) (int64; fattn's
  umulhi(m - S, A), A saturated to fattn's uint32_t), e = LUT[t] with
  LUT[t] = rint(127 2^(-t/16)) at keys 0..pos and 0 past pos (so the max key's e is 127);
* L = sum e (integer), r = RN32(1 / L) and z = RN32(e(0) / L) (fattn's s_q and z_q);
* P = e at keys 1..pos, 0 at key 0 (key 0 is its own block, the sink term below) and past pos: s8 in [0, 127];
* V [s, d], one scale per channel over the window: s_v = RN32(max(max_t |v|, RN32(1e-30)) / 127), V' = clamp(round(...), -127, 127);
* Y = P V' exactly (one ``torch._int_mm``, int32: s <= MAX_KEYS);
* out = bf16(RN32(RN32(RN32(RN32(Y) r) + RN32(z RN32(V'(0)))) s_v)), apv_ref.py's f = ((float(O + 64 S) s_q + z_q float(X(0)))
  s_ch) with no dither, where O + 64 S is P V' over keys 1..pos and X(0) is V' at key 0.

Every step is integer or one IEEE operation, so a CPU and a GPU compute the same bits.  V's scale reads every key of the window,
so a row's output depends on the window's later tokens: each chunk of a chunked prefill is its own window, and its rows are the
same rows of the whole prompt of its s tokens, not of the whole prompt the request ends with.  Nothing reads another request.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from verity_vllm.engine import int8_exact as E
from verity_vllm.program.frontend.kernel_readback import is_attention_impl

ARM = E.ARM
TOP = E.TOP
#: |P| <= 127 and |V'| <= 127 over s keys stay within int32
MAX_KEYS = E.MAX_K
#: cuBLASLt (sm_120) refuses an s8 product whose k is an odd multiple of 8 up to 120 at n = 128 and m not a multiple of 64
K_ALIGN = 16
#: and, for every k below 128 and every m, one whose n is an odd multiple of 8 from 40: a head dim 64
#: window of 33..40 keys
N_ALIGN = 16
LUT_SIZE = 256
#: fattn_ref.py's table: LUT[t] = rint(127 2^(-t/16)), t in 0..255
LUT = np.rint(127.0 * np.exp2(-np.arange(LUT_SIZE) / 16.0)).astype(np.int64)
LOG2E = 1.4426950408889634
#: fattn.cu's A is a uint32_t: past it, A saturates
A_CAP = (1 << 32) - 1
#: m - S <= 2 d 127^2 < 2^23 for d <= 256, so (m - S) A < 2^63
MAX_HEAD_DIM = 256


def quant(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Integers in [-127, 127] (as float32) and their scales (amax over `dim`, kept, floored at 1e-30, / 127)."""
    t = torch.full((), float(TOP), dtype=torch.float32, device=x.device)
    s = x.abs().amax(dim=dim, keepdim=True).clamp(min=E.SCALE_FLOOR) / t
    return torch.round(x / s).clamp_(-TOP, TOP), s


def quant_static(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Integers in [-127, 127] (as float32) of x under one given float32 scale `s` (a 0-d tensor)."""
    return torch.round(x / s.to(device=x.device, dtype=torch.float32)).clamp_(-TOP, TOP)


def fixed_point_a(s_q: torch.Tensor, s_k: torch.Tensor, scale: float) -> torch.Tensor:
    """A (int64) per query row: min(rint((((s_q s_k) scale) log2(e)) 16 2^32), 2^32 - 1) in float64, left to right."""
    sigma = (s_q.double() * s_k.to(device=s_q.device, dtype=torch.float64)) * float(scale)
    a = torch.round(((sigma * LOG2E) * 16.0) * 4294967296.0)
    return a.clamp_(max=float(A_CAP)).long()


def int_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a b^T (int32) of integer-valued a [m, k] and b [n, k] within int8, exactly: k padded with zeros to a multiple of
    K_ALIGN, n to a multiple of N_ALIGN and m to `int8_exact.MIN_ROWS` (``torch._int_mm``'s shape rules on CUDA)."""
    n, k = b.shape
    a8 = F.pad(a.to(torch.int8), (0, -k % K_ALIGN)).contiguous()
    b8 = F.pad(b.to(torch.int8), (0, -k % K_ALIGN, 0, -n % N_ALIGN)).contiguous()
    return E.int_mm(a8, b8)[:, :n]


def lut_e(S: torch.Tensor, a: torch.Tensor, causal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(e int64 [..., s], m [..., 1]) of S int [..., s] under A [..., 1] and the causal mask [n, s] (True at keys 0..pos)."""
    S = S.long()
    m = S.masked_fill(~causal, torch.iinfo(torch.int64).min).amax(-1, keepdim=True)
    t = (((m - S).clamp_(min=0) * a) >> 32).clamp_(max=LUT_SIZE - 1)
    return torch.as_tensor(LUT, device=S.device)[t] * causal, m


def lut_softmax(S: torch.Tensor, a: torch.Tensor, causal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(P int64 [..., s] with key 0 zeroed, r = RN32(1 / L), z = RN32(e(0) / L)) of S under A and the causal mask (`lut_e`)."""
    e, _ = lut_e(S, a, causal)
    lf = e.sum(-1).float()
    r = torch.ones_like(lf) / lf
    z = e[..., 0].float() / lf
    return F.pad(e[..., 1:], (1, 0)), r, z


def kv_quant(k: torch.Tensor, v: torch.Tensor, k_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """What every query row of a window reads, per KV head g, from its keys k and values v [s, G, d] (bf16) and the layer's K scales
    k_scales [G] (float32): (K' int8 [G, s, d], V' int8 [G, s, d], s_v float32 [G, d], f32(V'(0)) float32 [G, d])."""
    s, g_heads, d = k.shape
    if s > MAX_KEYS or d > MAX_HEAD_DIM or tuple(v.shape) != (s, g_heads, d) or tuple(k_scales.shape) != (g_heads,):
        raise ValueError(f"int8 attention: keys {tuple(k.shape)}, values {tuple(v.shape)}, {tuple(k_scales.shape)} K scales (head dim "
                         f"at most {MAX_HEAD_DIM}, at most {MAX_KEYS} keys)")
    kq = torch.stack([quant_static(k[:, g].float(), k_scales[g]) for g in range(g_heads)]).to(torch.int8)
    vi, sv = quant(v.float().transpose(0, 1), 1)
    vq = vi.to(torch.int8).contiguous()
    return kq, vq, sv[:, 0].contiguous(), vq[:, 0].float()


def window_rows(q: torch.Tensor, kq: torch.Tensor, vq: torch.Tensor, sv: torch.Tensor, v0: torch.Tensor, k_scales: torch.Tensor,
                scale: float) -> torch.Tensor:
    """out [n, H, d] bf16 of the window's last n query rows q [n, H, d] (H a multiple of G) over `kv_quant`'s results for its s keys."""
    n, h, d = q.shape
    g_heads, s, _ = kq.shape
    if h % g_heads or not 0 < n <= s:
        raise ValueError(f"int8 attention: {h} query heads over {g_heads} KV heads, {n} rows of a {s}-key window")
    rep = h // g_heads
    dev = q.device
    causal = torch.arange(s, device=dev)[None, :] <= torch.arange(s - n, s, device=dev)[:, None]
    out = torch.empty(n, h, d, dtype=torch.bfloat16, device=dev)
    for g in range(g_heads):
        qi, sq = quant(q[:, g * rep:(g + 1) * rep].transpose(0, 1).float(), -1)
        S = int_mm(qi.reshape(rep * n, d), kq[g]).view(rep, n, s)
        p, r, z = lut_softmax(S, fixed_point_a(sq, k_scales[g], scale), causal)
        y = int_mm(p.reshape(rep * n, s), vq[g].T).view(rep, n, d)
        f = ((y.float() * r[..., None]) + (z[..., None] * v0[g])) * sv[g]
        out[:, g * rep:(g + 1) * rep] = f.transpose(0, 1).to(torch.bfloat16)
    return out


def window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_scales: torch.Tensor, scale: float) -> torch.Tensor:
    """out [n, H, d] bf16 of the window's last n query rows q [n, H, d] over its keys k and values v [s, G, d] (bf16, H a
    multiple of G), with the layer's K scales k_scales [G] (float32): `window_rows` over `kv_quant`."""
    return window_rows(q, *kv_quant(k, v, k_scales), k_scales, scale)


def step_refusal(qsl: list[int], lens: list[int]) -> str | None:
    """Why a step's requests (query start offsets qsl [B + 1] and window lengths lens [>= B], host lists) are not each its window's
    last 1 .. s rows (None: they are; a request with no rows is skipped)."""
    for r, s in enumerate(lens[:len(qsl) - 1]):
        m = qsl[r + 1] - qsl[r]
        if m and not 0 < m <= s:
            return f"int8 attention: request {r} brings {m} rows of a {s}-token window; its rows are the window's last 1 .. {s}"
    return None


def token_rows(qsl: list[int], lens: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(request, position) of each of a step's query rows, int64 [T] on `device`: request r's rows qsl[r] .. qsl[r+1] - 1 are
    positions s - m .. s - 1 of its s-token window."""
    req, pos = [], []
    for r in range(len(qsl) - 1):
        m, s = qsl[r + 1] - qsl[r], lens[r]
        req += [r] * m
        pos += range(s - m, s)
    return (torch.tensor(req, dtype=torch.int64).to(device, non_blocking=True),
            torch.tensor(pos, dtype=torch.int64).to(device, non_blocking=True))


def batch_check(k: torch.Tensor, k_scales: torch.Tensor, lens: list[int]) -> None:
    b, s_max, g_heads, d = k.shape
    if s_max > MAX_KEYS or d > MAX_HEAD_DIM or tuple(k_scales.shape) != (g_heads,) or len(lens) != b or max(lens, default=0) > s_max:
        raise ValueError(f"int8 attention: windows {tuple(k.shape)} of lengths {lens[:8]}, {tuple(k_scales.shape)} K scales (head dim "
                         f"at most {MAX_HEAD_DIM}, at most {MAX_KEYS} keys)")


def paged_windows(key_cache: torch.Tensor, value_cache: torch.Tensor, block_table: torch.Tensor,
                  lens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """k, v [B, s_max, G, d] of each request's first lens[r] tokens from paged caches [num_blocks, block_size, G, d] under its
    block-table row (`window_of` of every request at once), zero past each window; s_max = max(lens)."""
    b, s_max = len(lens), max(lens, default=0)
    bs, dev = key_cache.shape[1], key_cache.device
    pos = torch.arange(s_max, device=dev)
    live = pos[None, :] < torch.tensor(lens, dtype=torch.int64).to(dev, non_blocking=True)[:, None]
    col = (pos // bs).clamp_(max=max(block_table.shape[1] - 1, 0)).expand(b, -1)
    blk = torch.where(live, block_table[:b].long().gather(1, col), 0)
    off = (pos % bs).expand(b, -1)
    zero = torch.zeros((), dtype=key_cache.dtype, device=dev)
    return tuple(torch.where(live[..., None, None], c[blk, off], zero) for c in (key_cache, value_cache))


def packed_windows(keys: torch.Tensor, values: torch.Tensor, starts: list[int], lens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """k, v [B, s_max, G, d] of request r's window at rows starts[r] .. starts[r] + lens[r] - 1 of keys and values [rows, G, d] (a
    credited read's step buffers, `engine.int8_window`), zero past each window."""
    s_max, dev = max(lens, default=0), keys.device
    pos = torch.arange(s_max, device=dev)
    live = pos[None, :] < torch.tensor(lens, dtype=torch.int64).to(dev, non_blocking=True)[:, None]
    row = torch.where(live, torch.tensor(starts, dtype=torch.int64).to(dev, non_blocking=True)[:, None] + pos[None, :], 0)
    zero = torch.zeros((), dtype=keys.dtype, device=dev)
    return tuple(torch.where(live[..., None, None], t[row], zero) for t in (keys, values))


def kv_quant_batch(k: torch.Tensor, v: torch.Tensor, k_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`kv_quant` of every request's window at once, from k, v [B, s_max, G, d] (bf16, zero past each window, as `paged_windows`
    and `packed_windows` give them): (K' int8 [B, G, s_max, d], V' int8 [B, G, s_max, d], s_v float32 [B, G, d], f32(V'(0))
    float32 [B, G, d]).  Request r's K'[r, g, :s], V'[r, g, :s], s_v[r, g] and f32(V'(0))[r, g] are `kv_quant`'s of its s-key
    window, bit for bit: the same IEEE operation per element, and the zero keys past a window quantize to 0 and leave each
    channel's max |v| as it is."""
    kf = k.float().permute(0, 2, 1, 3)
    kq = torch.round(kf / k_scales.to(device=k.device, dtype=torch.float32)[None, :, None, None]).clamp_(-TOP, TOP).to(torch.int8)
    vi, sv = quant(v.float().permute(0, 2, 1, 3), 2)
    vq = vi.to(torch.int8).contiguous()
    return kq.contiguous(), vq, sv[:, :, 0].contiguous(), vq[:, :, 0].float()


def lut_core(qi: torch.Tensor, kq: torch.Tensor, vq: torch.Tensor, req: torch.Tensor, pos: torch.Tensor,
             a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The integer heart of `window_rows` for every query row of a step at once: (Y [T, H, d], L [T, H], e(0) [T, H]), int64, of
    Q' qi [T, H, d] (integers in [-127, 127]) over the K' and V' [B, G, s_max, d] of each row's request req [T] at its position
    pos [T], under A a [T, H]: S = Q' K'^T, e = LUT[min(((m - S) A) >> 32, 255)] at keys 0..pos (0 past it), L = sum e, Y = P V'
    with P = e at keys 1..pos.  Integer products and sums (exact in any order); `int8_attention_triton.lut_rows` computes it on CUDA inside the row."""
    t, h, d = qi.shape
    g_heads, s_max = kq.shape[1], kq.shape[2]
    rep = h // g_heads
    q = qi.long().view(t, g_heads, rep, d)
    S = q @ kq[req].long().transpose(-1, -2)
    causal = (torch.arange(s_max, device=qi.device)[None, :] <= pos[:, None])[:, None, None, :]
    e, _ = lut_e(S, a.view(t, g_heads, rep, 1), causal)
    y = F.pad(e[..., 1:], (1, 0)) @ vq[req].long()
    return y.view(t, h, d), e.sum(-1).view(t, h), e[..., 0].view(t, h)


def batch_rows(query: torch.Tensor, kq: torch.Tensor, vq: torch.Tensor, sv: torch.Tensor, v0: torch.Tensor, k_scales: torch.Tensor,
               req: torch.Tensor, pos: torch.Tensor, scale: float, core=lut_core) -> torch.Tensor:
    """out [T, H, d] bf16 of a step's query rows query [T, H, d] (row i request req[i]'s row at position pos[i]) over
    `kv_quant_batch`'s results: each row `window_rows`' row, bit for bit, with every row's integer part in one `core` call."""
    t, h, d = query.shape
    b, g_heads = kq.shape[:2]
    if h % g_heads or tuple(sv.shape) != (b, g_heads, d):
        raise ValueError(f"int8 attention: {h} query heads over {g_heads} KV heads, results {tuple(kq.shape)} / {tuple(sv.shape)}")
    rep = h // g_heads
    qi, sq = quant(query.float(), -1)
    a = fixed_point_a(sq[..., 0], k_scales.to(query.device).repeat_interleave(rep)[None, :], scale)
    y, el, e0 = core(qi.to(torch.int8), kq, vq, req, pos, a)
    lf = el.float()
    r = torch.ones_like(lf) / lf
    z = e0.float() / lf
    v0r, svr = (x[req].repeat_interleave(rep, dim=1) for x in (v0, sv))
    return (((y.float() * r[..., None]) + (z[..., None] * v0r)) * svr).to(torch.bfloat16)


def reference_requests(query: torch.Tensor, qsl: list[int], lens: list[int], window, k_scales: torch.Tensor, scale: float,
                       output: torch.Tensor) -> None:
    """The per-request form the batched ops are held to: request r's rows `window_rows` over `kv_quant` of `window(r, s)`, its k
    and v [s, G, d], one request and one KV head at a time."""
    if why := step_refusal(qsl, lens):
        raise ValueError(why)
    for r, s in enumerate(lens[:len(qsl) - 1]):
        q0, q1 = qsl[r], qsl[r + 1]
        if q1 > q0:
            output[q0:q1] = window_rows(query[q0:q1], *kv_quant(*window(r, s), k_scales), k_scales, float(scale))


def reference_paged(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, query_start_loc: torch.Tensor,
                    seq_lens: torch.Tensor, block_table: torch.Tensor, k_scales: torch.Tensor, scale: float, output: torch.Tensor) -> None:
    """`int8_ops.paged_attention`'s values, request by request (`reference_requests` over `window_of`)."""
    reference_requests(query, query_start_loc.tolist(), seq_lens.tolist(),
                       lambda r, s: window_of(key_cache, value_cache, block_table[r], s), k_scales, scale, output)


def window_of(key_cache: torch.Tensor, value_cache: torch.Tensor, blocks: torch.Tensor, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k, v [s, G, d] of a request's first s tokens from paged caches [num_blocks, block_size, G, d] under its block-table row."""
    bs = key_cache.shape[1]
    pos = torch.arange(s, device=key_cache.device)
    slot = blocks[pos // bs].long()
    return key_cache[slot, pos % bs], value_cache[slot, pos % bs]


def page_layout_refusal(kv_cache: torch.Tensor, head_size: int) -> str | None:
    """Why a layer's cache, vLLM's logical view [pages, KV heads, slots, 2 head_size], is not [pages][slots][KV heads][K‖V] in memory:
    each page its slots in order, each slot its heads, each head its K then its V (None: it is)."""
    if kv_cache.dim() != 4 or kv_cache.shape[-1] != 2 * head_size:
        return f"shape {tuple(kv_cache.shape)} is not [pages, KV heads, slots, 2 x {head_size}]"
    _, g, n, c = kv_cache.shape
    want = (n * g * c, c, g * c, 1)
    if tuple(kv_cache.stride()) != want:
        return f"strides {tuple(kv_cache.stride())} are not [pages][slots][KV heads][K‖V]'s {want}"
    return None


def applied_record(model: Any) -> dict[str, dict[str, Any]]:
    """What every attention layer ran, by module name: its impl's class, int8 arm (None for an impl without one) and the calls it
    served (None for an impl that does not count them).  Structural, as `kernel_readback.int8_exact_of`."""
    return {name: {"impl": type(m.impl).__name__, "arm": getattr(m.impl, "int8_arm", None), "calls": getattr(m.impl, "calls", None)}
            for name, m in model.named_modules() if is_attention_impl(getattr(m, "impl", None))}

