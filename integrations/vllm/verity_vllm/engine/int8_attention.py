"""Exact-int8 attention: ``benchmarks/ppl/steps.py``'s plain ``int8`` attention step (its defaults: Q, K, P and V rounded, every
top 127, ``p_block = 0``, ``v_scale = "channel"``), on a request's window, in plain torch (``engine.int8_attention_vllm`` serves
it inside vLLM; ``TargetProfile.int8_exact`` declares it).

A window is a request's tokens 0..s-1 (its KV cache: K after RoPE and V, both bf16); a call computes the window's last n query
rows (positions s-n..s-1).  A whole-prompt prefill is n = s, steps.py's window exactly; a decode step is n = 1, the last row of
steps.py's window over every token so far.  For every KV group g, its `rep` query heads (all in float32, RN32 = round to
nearest even; every division is a true float32 division, never a reciprocal product):

* Q [rep, n, d], one scale per (head, token): s_q = RN32(max(max |q|, RN32(1e-30)) / 127), Q' = clamp(round(RN32(q / s_q)),
  -127, 127) (`round`: to the nearest integer, ties to even);
* K [s, d], one scale per token, the same rule: s_k, K';
* S = Q' K'^T exactly (one ``torch._int_mm`` over the group's rep heads stacked, int32: |S| <= d 127^2);
* scores = RN32(RN32(S) * RN32(s_q * s_k)), then logits = RN32(scores * RN32(scale)), -inf past the causal edge (key > row);
* P = torch.softmax(logits, -1) in float32 (torch's own, so device-dependent: the one step a CPU and a GPU may round apart);
* one scale per query row: s_p = RN32(max(max P, RN32(1e-30)) / 127), P' = clamp(round(RN32(P / s_p)), 0, 127);
* V [s, d], one scale per channel over the window: s_v = RN32(max(max_t |v|, RN32(1e-30)) / 127), V' = clamp(round(...), -127, 127);
* Y = P' V' exactly (one ``torch._int_mm``, int32: s <= MAX_KEYS);
* out = bf16(RN32(RN64(RN64(s_p * Y) * s_v))).

So V's scale reads every key of the window, and a row's output depends on the window's later tokens (steps.py's as well): a
prefill split into chunks would be another window, and is refused.  Nothing reads another request.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from verity_vllm.engine import int8_exact as E
from verity_vllm.program.frontend.kernel_readback import is_attention_impl

ARM = E.ARM
TOP = E.TOP
#: |P'| <= 127 and |V'| <= 127 over s keys stay within int32
MAX_KEYS = E.MAX_K


def quant(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """steps.py's `quant(x, 127, dim)`: integers in [-127, 127] (as float32) and their scales (amax over `dim`, kept)."""
    t = torch.full((), float(TOP), dtype=torch.float32, device=x.device)
    s = x.abs().amax(dim=dim, keepdim=True).clamp(min=E.SCALE_FLOOR) / t
    return torch.round(x / s).clamp_(-TOP, TOP), s


def quant_p(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """steps.py's P quantizer (one block, round): integers in [0, 127] and one scale per row."""
    t = torch.full((), float(TOP), dtype=torch.float32, device=p.device)
    s = p.amax(-1, keepdim=True).clamp(min=E.SCALE_FLOOR) / t
    return torch.round(p / s).clamp_(0, TOP), s


def int_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a b^T (int32) of integer-valued a [m, k] and b [n, k] within int8, exactly: k and n padded with zeros to multiples of 8
    and m to `int8_exact.MIN_ROWS` (``torch._int_mm``'s shape rules on CUDA)."""
    n, k = b.shape
    a8 = F.pad(a.to(torch.int8), (0, -k % 8)).contiguous()
    b8 = F.pad(b.to(torch.int8), (0, -k % 8, 0, -n % 8)).contiguous()
    return E.int_mm(a8, b8)[:, :n]


def window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, softmax=torch.softmax) -> torch.Tensor:
    """out [n, H, d] bf16 of the window's last n query rows q [n, H, d] over its keys k and values v [s, G, d] (bf16, H a
    multiple of G).  `softmax` is torch.softmax unless a check swaps the device it runs on."""
    n, h, d = q.shape
    s, g_heads, _ = k.shape
    if h % g_heads or not 0 < n <= s or s > MAX_KEYS:
        raise ValueError(f"int8 attention: {h} query heads over {g_heads} KV heads, {n} rows of a {s}-key window (at most {MAX_KEYS})")
    rep = h // g_heads
    dev = q.device
    causal = torch.arange(s, device=dev)[None, :] <= torch.arange(s - n, s, device=dev)[:, None]
    out = torch.empty(n, h, d, dtype=torch.bfloat16, device=dev)
    for g in range(g_heads):
        qi, sq = quant(q[:, g * rep:(g + 1) * rep].transpose(0, 1).float(), -1)
        ki, sk = quant(k[:, g].float(), -1)
        scores = int_mm(qi.reshape(rep * n, d), ki).view(rep, n, s).float() * (sq * sk.T[None])
        p = softmax((scores * scale).masked_fill(~causal, float("-inf")), -1)
        pi, sp = quant_p(p)
        vi, sv = quant(v[:, g].float(), 0)
        prod = int_mm(pi.reshape(rep * n, s), vi.T).view(rep, n, d)
        out[:, g * rep:(g + 1) * rep] = ((sp.double() * prod.double()) * sv.double()).float().transpose(0, 1).to(torch.bfloat16)
    return out


def window_of(key_cache: torch.Tensor, value_cache: torch.Tensor, blocks: torch.Tensor, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k, v [s, G, d] of a request's first s tokens from paged caches [num_blocks, block_size, G, d] under its block-table row."""
    bs = key_cache.shape[1]
    pos = torch.arange(s, device=key_cache.device)
    slot = blocks[pos // bs].long()
    return key_cache[slot, pos % bs], value_cache[slot, pos % bs]


def paged_attention(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, query_start_loc: torch.Tensor,
                    seq_lens: torch.Tensor, block_table: torch.Tensor, scale: float, output: torch.Tensor, softmax=torch.softmax) -> None:
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
        output[q0:q1] = window_attention(query[q0:q1], k, v, scale, softmax)


def applied_record(model: Any) -> dict[str, dict[str, Any]]:
    """What every attention layer ran, by module name: its impl's class, int8 arm (None for an impl without one) and the calls it
    served (None for an impl that does not count them).  Structural, as `kernel_readback.int8_exact_of`."""
    return {name: {"impl": type(m.impl).__name__, "arm": getattr(m.impl, "int8_arm", None), "calls": getattr(m.impl, "calls", None)}
            for name, m in model.named_modules() if is_attention_impl(getattr(m, "impl", None))}
