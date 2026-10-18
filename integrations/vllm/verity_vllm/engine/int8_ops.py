"""The exact-int8 profile's calls as named torch ops, so that a capture sees each int8 linear as two nodes, its quantizer and its
product, and each request's int8 attention as two, its KV quantization and its rows (`observe.fold.patterns.int8`):

    verity_int8::row_quant(Tensor x) -> (Tensor xq, Tensor s_a)
        x bf16 [M, K]: `engine.int8_exact.quantize_rows`, xq int8 [M, K] and s_a f32 [M] (`registry.int8_exact.Int8RowQuant_v1{K}`
        per row)
    verity_int8::scaled_mm(Tensor xq, Tensor s_a, Tensor w, Tensor s_w) -> Tensor y
        w int8 [N, K], s_w f32 [N, 1] (or [N]): the plain linear's product of row_quant's results, y bf16 [M, N]
        (`registry.int8_exact.ScaledMmS8_v1{K,N}` per row)

The quantizer's results are a separate node's, read only by the product inside the linear's body, so a Program cuts them as
values the linear's two Calls exchange (`call_boundaries`), never as a second return of the linear.  `linear` runs the two.
    verity_int8::kv_quant(Tensor key_cache, Tensor value_cache, Tensor blocks, int s, Tensor k_scales)
            -> (Tensor kq, Tensor vq, Tensor sv, Tensor v0)
        one request's first s tokens in the paged caches [num_blocks, block_size, G, d] under its block-table row `blocks`, with the
        layer's K scales [G]: K' and V' int8 [G, s, d], s_v and f32(V'(0)) float32 [G, d] (`registry.int8_exact.Int8KvQuant_v1{s,d}`
        per KV head)
    verity_int8::window_rows(Tensor q, Tensor kq, Tensor vq, Tensor sv, Tensor v0, Tensor k_scales, float scale) -> Tensor out
        that request's last n query rows q bf16 [n, H, d] over kv_quant's results (`registry.int8_exact.Int8LutAttentionRow_v1{d,
        pos+1,scale}` per (row, head))

Importing this module defines them (once per process).
"""
from __future__ import annotations

import torch

from verity_vllm.engine import int8_attention as A
from verity_vllm.engine import int8_exact as E

LIBRARY = "verity_int8"


@torch.library.custom_op(f"{LIBRARY}::row_quant", mutates_args=())
def row_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return E.quantize_rows(x)


@row_quant.register_fake
def _row_quant_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty(x.shape, dtype=torch.int8), x.new_empty(x.shape[0], dtype=torch.float32)


@torch.library.custom_op(f"{LIBRARY}::scaled_mm", mutates_args=())
def scaled_mm(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    return E.dequantize(E.int_mm(xq, w), s_a, s_w)


@scaled_mm.register_fake
def _scaled_mm_fake(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    return xq.new_empty((xq.shape[0], w.shape[0]), dtype=torch.bfloat16)


def linear(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The plain linear as its two ops: (xq, s_a, y)."""
    xq, s_a = row_quant(x)
    return xq, s_a, scaled_mm(xq, s_a, w, s_w)


@torch.library.custom_op(f"{LIBRARY}::kv_quant", mutates_args=())
def kv_quant(key_cache: torch.Tensor, value_cache: torch.Tensor, blocks: torch.Tensor, s: int,
             k_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k, v = A.window_of(key_cache, value_cache, blocks, s)
    return A.kv_quant(k, v, k_scales)


@kv_quant.register_fake
def _kv_quant_fake(key_cache, value_cache, blocks, s, k_scales):
    g, d = key_cache.shape[2], key_cache.shape[3]
    q8 = key_cache.new_empty((g, s, d), dtype=torch.int8)
    return q8, torch.empty_like(q8), key_cache.new_empty((g, d), dtype=torch.float32), key_cache.new_empty((g, d), dtype=torch.float32)


@torch.library.custom_op(f"{LIBRARY}::window_rows", mutates_args=())
def window_rows(q: torch.Tensor, kq: torch.Tensor, vq: torch.Tensor, sv: torch.Tensor, v0: torch.Tensor, k_scales: torch.Tensor,
                scale: float) -> torch.Tensor:
    return A.window_rows(q, kq, vq, sv, v0, k_scales, scale)


@window_rows.register_fake
def _window_rows_fake(q, kq, vq, sv, v0, k_scales, scale) -> torch.Tensor:
    return torch.empty_like(q)


def paged_attention(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, query_start_loc: torch.Tensor,
                    seq_lens: torch.Tensor, block_table: torch.Tensor, k_scales: torch.Tensor, scale: float, output: torch.Tensor) -> None:
    """Every request of a batch, one window each, into output [T, H, d]: request r's query rows query[qsl[r]:qsl[r+1]] are its
    window's last rows, its window its first seq_lens[r] tokens in the caches (vLLM's FlashAttention metadata and layout), one
    ``verity_int8::kv_quant`` and one ``verity_int8::window_rows`` call each.  A request whose rows are neither its whole window nor one row (a chunked prefill's later chunk) is refused."""
    qsl, lens = query_start_loc.tolist(), seq_lens.tolist()
    for r, s in enumerate(lens):
        q0, q1 = qsl[r], qsl[r + 1]
        if q1 == q0:
            continue
        if q1 - q0 not in (1, s):
            raise ValueError(f"int8 attention: request {r} brings {q1 - q0} rows of a {s}-token window (a chunked prefill)")
        kv = kv_quant(key_cache, value_cache, block_table[r], s, k_scales)
        output[q0:q1] = window_rows(query[q0:q1], *kv, k_scales, float(scale))
