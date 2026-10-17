"""The exact-int8 profile's two calls as named torch ops, so that a capture sees each int8 linear and each request's int8 attention as
one node (`observe.fold.patterns.int8`) and an export keeps it as one (`program.frontend.rules.vllm_bindings.int8`):

    verity_int8::linear(Tensor x, Tensor w, Tensor s_w) -> (Tensor xq, Tensor s_a, Tensor y)
        x bf16 [M, K], w int8 [N, K], s_w f32 [N]: `engine.int8_exact`'s plain linear, all three values it computes
        (`registry.int8_exact.Int8Linear_v1{K,N}` per row)
    verity_int8::window_attention(Tensor q, Tensor key_cache, Tensor value_cache, Tensor blocks, int s, Tensor k_scales,
                                  float scale) -> Tensor out
        one request: its last n query rows q bf16 [n, H, d] over its first s tokens in the paged caches [num_blocks, block_size, G, d]
        under its block-table row `blocks`, with the layer's K scales [G] (`registry.int8_exact.Int8LutAttentionWindow_v1{n,s,H,G,d,scale}`)

Importing this module defines them (once per process).
"""
from __future__ import annotations

import torch

from verity_vllm.engine import int8_attention as A
from verity_vllm.engine import int8_exact as E

LIBRARY = "verity_int8"


@torch.library.custom_op(f"{LIBRARY}::linear", mutates_args=())
def linear(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xq, s_a = E.quantize_rows(x)
    return xq, s_a, E.dequantize(E.int_mm(xq, w), s_a, s_w)


@linear.register_fake
def _linear_fake(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = x.shape[0]
    return x.new_empty(x.shape, dtype=torch.int8), x.new_empty(m, dtype=torch.float32), x.new_empty((m, w.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op(f"{LIBRARY}::window_attention", mutates_args=())
def window_attention(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, blocks: torch.Tensor, s: int,
                     k_scales: torch.Tensor, scale: float) -> torch.Tensor:
    k, v = A.window_of(key_cache, value_cache, blocks, s)
    return A.window_attention(q, k, v, k_scales, scale)


@window_attention.register_fake
def _window_attention_fake(q, key_cache, value_cache, blocks, s, k_scales, scale) -> torch.Tensor:
    return torch.empty_like(q)


def paged_attention(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, query_start_loc: torch.Tensor,
                    seq_lens: torch.Tensor, block_table: torch.Tensor, k_scales: torch.Tensor, scale: float, output: torch.Tensor) -> None:
    """Every request of a batch, one window each, into output [T, H, d]: request r's query rows query[qsl[r]:qsl[r+1]] are its
    window's last rows, its window its first seq_lens[r] tokens in the caches (vLLM's FlashAttention metadata and layout), one
    ``verity_int8::window_attention`` call each.  A request whose rows are neither its whole window nor one row (a chunked prefill's later chunk) is refused."""
    qsl, lens = query_start_loc.tolist(), seq_lens.tolist()
    for r, s in enumerate(lens):
        q0, q1 = qsl[r], qsl[r + 1]
        if q1 == q0:
            continue
        if q1 - q0 not in (1, s):
            raise ValueError(f"int8 attention: request {r} brings {q1 - q0} rows of a {s}-token window (a chunked prefill)")
        output[q0:q1] = window_attention(query[q0:q1], key_cache, value_cache, block_table[r], s, k_scales, float(scale))
