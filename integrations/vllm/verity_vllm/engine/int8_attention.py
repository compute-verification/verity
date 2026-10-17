"""Exact-int8 attention: compute accounting's integer LUT softmax (``benchmarks/pouw/edited_gemm/fattn_ref.py``, and the output of
``apv_ref.py``'s served f with no dither), on a request's window, in plain torch (``engine.int8_attention_vllm`` serves it inside
vLLM; ``TargetProfile.int8_exact`` declares it).

A window is a request's tokens 0..s-1 (its KV cache: K after RoPE and V, both bf16); a call computes the window's last n query
rows (positions s-n..s-1).  A whole-prompt prefill is n = s; a decode step is n = 1, the last row of the window over every token
so far.  For every KV group g (its `rep` query heads; RN32 / RN64 = float32 / float64 round to nearest even, one rounding per
operation; every division a true IEEE division):

* Q [rep, n, d], one scale per (head, token): s_q = RN32(max(max |q|, RN32(1e-30)) / 127), Q' = clamp(round(RN32(q / s_q)),
  -127, 127) (`round`: to the nearest integer, ties to even);
* K [s, d], ONE scale per (layer, KV head), s_k (`engine.int8_k_scales`, calibrated offline): K' = clamp(round(RN32(k / s_k)),
  -127, 127);
* S = Q' K'^T exactly (one ``torch._int_mm`` over the group's rep heads stacked, int32: |S| <= d 127^2);
* A = min(rint(RN64(RN64(RN64(RN64(s_q s_k) scale) log2(e)) 16) 2^32), 2^40) per query row, fattn.cu's `fa_A` with
  sigma = s_q s_k scale (the logit of one unit of S): float64, left to right, rint ties to even;
* for the row at position pos: m = max S over keys 0..pos, t = min(((m - S) A) >> 32, 255) (int64; fattn's umulhi(m - S, A)
  for A < 2^32, and the same t for any larger A: past 255 2^32 every key but the max one is 255 already), e = LUT[t] with
  LUT[t] = rint(127 2^(-t/16)) at keys 0..pos and 0 past pos (so the max key's e is 127);
* L = sum e (integer), r = RN32(1 / L) and z = RN32(e(0) / L) (fattn's s_q and z_q);
* P = e at keys 1..pos, 0 at key 0 (key 0 is its own block, the sink term below) and past pos: s8 in [0, 127];
* V [s, d], one scale per channel over the window: s_v = RN32(max(max_t |v|, RN32(1e-30)) / 127), V' = clamp(round(...), -127, 127);
* Y = P V' exactly (one ``torch._int_mm``, int32: s <= MAX_KEYS);
* out = bf16(RN32(RN32(RN32(RN32(Y) r) + RN32(z RN32(V'(0)))) s_v)), apv_ref.py's f = ((float(O + 64 S) s_q + z_q float(X(0)))
  s_ch) with no dither, where O + 64 S is P V' over keys 1..pos and X(0) is V' at key 0.

Every step is integer or one IEEE operation, so a CPU and a GPU compute the same bits.  V's scale reads every key of the window,
so a row's output depends on the window's later tokens: a prefill split into chunks would be another window, and is refused.
Nothing reads another request.
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
LUT_SIZE = 256
#: fattn_ref.py's table: LUT[t] = rint(127 2^(-t/16)), t in 0..255
LUT = np.rint(127.0 * np.exp2(-np.arange(LUT_SIZE) / 16.0)).astype(np.int64)
LOG2E = 1.4426950408889634
#: A at or past 255 2^32 gives t = 255 at every key whose m - S >= 1, so capping A there changes no t
A_CAP = 1 << 40
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
    """A (int64) per query row: min(rint((((s_q s_k) scale) log2(e)) 16 2^32), 2^40) in float64, left to right."""
    sigma = (s_q.double() * s_k.to(device=s_q.device, dtype=torch.float64)) * float(scale)
    a = torch.round(((sigma * LOG2E) * 16.0) * 4294967296.0)
    return a.clamp_(max=float(A_CAP)).long()


def int_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a b^T (int32) of integer-valued a [m, k] and b [n, k] within int8, exactly: k padded with zeros to a multiple of
    K_ALIGN, n to a multiple of 8 and m to `int8_exact.MIN_ROWS` (``torch._int_mm``'s shape rules on CUDA)."""
    n, k = b.shape
    a8 = F.pad(a.to(torch.int8), (0, -k % K_ALIGN)).contiguous()
    b8 = F.pad(b.to(torch.int8), (0, -k % K_ALIGN, 0, -n % 8)).contiguous()
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


def window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_scales: torch.Tensor, scale: float) -> torch.Tensor:
    """out [n, H, d] bf16 of the window's last n query rows q [n, H, d] over its keys k and values v [s, G, d] (bf16, H a
    multiple of G), with the layer's K scales k_scales [G] (float32)."""
    n, h, d = q.shape
    s, g_heads, _ = k.shape
    if h % g_heads or not 0 < n <= s or s > MAX_KEYS or d > MAX_HEAD_DIM or tuple(k_scales.shape) != (g_heads,):
        raise ValueError(f"int8 attention: {h} query heads over {g_heads} KV heads ({tuple(k_scales.shape)} K scales), head dim {d} "
                         f"(at most {MAX_HEAD_DIM}), {n} rows of a {s}-key window (at most {MAX_KEYS})")
    rep = h // g_heads
    dev = q.device
    causal = torch.arange(s, device=dev)[None, :] <= torch.arange(s - n, s, device=dev)[:, None]
    out = torch.empty(n, h, d, dtype=torch.bfloat16, device=dev)
    for g in range(g_heads):
        qi, sq = quant(q[:, g * rep:(g + 1) * rep].transpose(0, 1).float(), -1)
        ki = quant_static(k[:, g].float(), k_scales[g])
        S = int_mm(qi.reshape(rep * n, d), ki).view(rep, n, s)
        p, r, z = lut_softmax(S, fixed_point_a(sq, k_scales[g], scale), causal)
        vi, sv = quant(v[:, g].float(), 0)
        y = int_mm(p.reshape(rep * n, s), vi.T).view(rep, n, d)
        f = ((y.float() * r[..., None]) + (z[..., None] * vi[0])) * sv
        out[:, g * rep:(g + 1) * rep] = f.transpose(0, 1).to(torch.bfloat16)
    return out


def window_of(key_cache: torch.Tensor, value_cache: torch.Tensor, blocks: torch.Tensor, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k, v [s, G, d] of a request's first s tokens from paged caches [num_blocks, block_size, G, d] under its block-table row."""
    bs = key_cache.shape[1]
    pos = torch.arange(s, device=key_cache.device)
    slot = blocks[pos // bs].long()
    return key_cache[slot, pos % bs], value_cache[slot, pos % bs]


def paged_attention(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, query_start_loc: torch.Tensor,
                    seq_lens: torch.Tensor, block_table: torch.Tensor, k_scales: torch.Tensor, scale: float, output: torch.Tensor) -> None:
    """Every request of a batch, one window each, into output [T, H, d]: request r's query rows query[qsl[r]:qsl[r+1]] are its
    window's last rows, its window its first seq_lens[r] tokens in the caches (vLLM's FlashAttention metadata and layout).
    A request whose rows are neither its whole window nor one row (a chunked prefill's later chunk) is refused."""
    qsl, lens = query_start_loc.tolist(), seq_lens.tolist()
    for r, s in enumerate(lens):
        q0, q1 = qsl[r], qsl[r + 1]
        if q1 == q0:
            continue
        if q1 - q0 not in (1, s):
            raise ValueError(f"int8 attention: request {r} brings {q1 - q0} rows of a {s}-token window (a chunked prefill)")
        k, v = window_of(key_cache, value_cache, block_table[r], s)
        output[q0:q1] = window_attention(query[q0:q1], k, v, k_scales, scale)


def applied_record(model: Any) -> dict[str, dict[str, Any]]:
    """What every attention layer ran, by module name: its impl's class, int8 arm (None for an impl without one) and the calls it
    served (None for an impl that does not count them).  Structural, as `kernel_readback.int8_exact_of`."""
    return {name: {"impl": type(m.impl).__name__, "arm": getattr(m.impl, "int8_arm", None), "calls": getattr(m.impl, "calls", None)}
            for name, m in model.named_modules() if is_attention_impl(getattr(m, "impl", None))}

