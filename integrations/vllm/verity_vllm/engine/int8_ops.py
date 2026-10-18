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
    verity_int8::kv_quant_window(Tensor k, Tensor v, Tensor k_scales) -> (Tensor kq, Tensor vq, Tensor sv, Tensor v0)
        kv_quant of one request's window already read, k and v bf16 [s, G, d] (a credited read's rows, `engine.int8_window`)
    verity_int8::credited_step(Tensor seq_lens, int batch, int kv_heads, int head_dim, int read) -> (Tensor lens, Tensor wo,
            Tensor k, Tensor v)
        a credited read's step (`int8_window.CreditedRead.buffers`): request r's window at rows wo[r] .. wo[r + 1] - 1 of the
        fresh K and V [rows, G, d] bf16, rows the sum of the lengths
    verity_int8::credited_window(Tensor kv_cache, Tensor block_table, Tensor? slot_mapping, Tensor lens, Tensor wo, Tensor(a!) k,
            Tensor(b!) v, int layer, int read) -> ()
        a layer's windows into the step's K and V (`int8_window.CreditedRead.read`), kv_cache vLLM's view [pages, G, slots, 2d];
        it also sets the plaintext flags of the slots slot_mapping wrote (the store's, not an argument)

Importing this module defines them (once per process).
"""
from __future__ import annotations

import torch

from verity_vllm.engine import int8_attention as A
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_window as W

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


@torch.library.custom_op(f"{LIBRARY}::kv_quant_window", mutates_args=())
def kv_quant_window(k: torch.Tensor, v: torch.Tensor, k_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return A.kv_quant(k, v, k_scales)


@kv_quant_window.register_fake
def _kv_quant_window_fake(k, v, k_scales):
    s, g, d = k.shape
    q8 = k.new_empty((g, s, d), dtype=torch.int8)
    return q8, torch.empty_like(q8), k.new_empty((g, d), dtype=torch.float32), k.new_empty((g, d), dtype=torch.float32)


def _requests(query: torch.Tensor, query_start_loc: torch.Tensor, seq_lens: torch.Tensor, kv_of, k_scales: torch.Tensor,
              scale: float, output: torch.Tensor) -> None:
    qsl, lens = query_start_loc.tolist(), seq_lens.tolist()
    for r, s in enumerate(lens[:len(qsl) - 1]):
        q0, q1 = qsl[r], qsl[r + 1]
        if q1 == q0:
            continue
        if q1 - q0 not in (1, s):
            raise ValueError(f"int8 attention: request {r} brings {q1 - q0} rows of a {s}-token window (a chunked prefill)")
        output[q0:q1] = window_rows(query[q0:q1], *kv_of(r, s), k_scales, float(scale))


def paged_attention(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, query_start_loc: torch.Tensor,
                    seq_lens: torch.Tensor, block_table: torch.Tensor, k_scales: torch.Tensor, scale: float, output: torch.Tensor) -> None:
    """Every request of a batch, one window each, into output [T, H, d]: request r's query rows query[qsl[r]:qsl[r+1]] are its
    window's last rows, its window its first seq_lens[r] tokens in the caches (vLLM's FlashAttention metadata and layout), one
    ``verity_int8::kv_quant`` and one ``verity_int8::window_rows`` call each.  A request whose rows are neither its whole window nor one row (a chunked prefill's later chunk) is refused."""
    _requests(query, query_start_loc, seq_lens, lambda r, s: kv_quant(key_cache, value_cache, block_table[r], s, k_scales), k_scales,
              scale, output)


def window_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, offsets: list[int], query_start_loc: torch.Tensor,
                     seq_lens: torch.Tensor, k_scales: torch.Tensor, scale: float, output: torch.Tensor) -> None:
    """`paged_attention` over a step's windows already read (`engine.int8_window`): request r's window is rows
    offsets[r] .. offsets[r] + seq_lens[r] - 1 of keys and values [rows, G, d] bf16, one ``verity_int8::kv_quant_window`` and one
    ``verity_int8::window_rows`` call each."""
    _requests(query, query_start_loc, seq_lens,
              lambda r, s: kv_quant_window(keys[offsets[r]:offsets[r] + s], values[offsets[r]:offsets[r] + s], k_scales), k_scales,
              scale, output)


@torch.library.custom_op(f"{LIBRARY}::credited_step", mutates_args=())
def credited_step(seq_lens: torch.Tensor, batch: int, kv_heads: int, head_dim: int,
                  read: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return W.read_of(read).buffers(seq_lens, batch, kv_heads, head_dim)


@credited_step.register_fake
def _credited_step_fake(seq_lens, batch, kv_heads, head_dim, read):
    rows = torch.library.get_ctx().new_dynamic_size()
    k = seq_lens.new_empty((rows, kv_heads, head_dim), dtype=torch.bfloat16)
    return (seq_lens.new_empty(batch, dtype=torch.int32), seq_lens.new_empty(batch + 1, dtype=torch.int32), k, torch.empty_like(k))


@torch.library.custom_op(f"{LIBRARY}::credited_window", mutates_args=("k", "v"))
def credited_window(kv_cache: torch.Tensor, block_table: torch.Tensor, slot_mapping: torch.Tensor | None, lens: torch.Tensor,
                    wo: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: int, read: int) -> None:
    W.read_of(read).read(kv_cache, layer, block_table, slot_mapping, lens, wo, k, v)


def credited_attention(read: W.CreditedRead, query: torch.Tensor, kv_cache: torch.Tensor, layer: int, meta, kv_heads: int,
                       k_scales: torch.Tensor, scale: float, output: torch.Tensor) -> None:
    """`paged_attention` through ``read``: one ``verity_int8::credited_step`` at the step's first layer (``meta``, vLLM's
    attention metadata, is one object a step), one ``verity_int8::credited_window`` a layer, then `window_attention`."""
    st = read.current
    if st is None or st.meta is not meta:
        batch = meta.query_start_loc.numel() - 1
        read.current = None
        lens, wo, k, v = credited_step(meta.seq_lens, batch, kv_heads, query.shape[-1], read.handle)
        host = [0]
        for n in meta.seq_lens.tolist()[:batch]:
            host.append(host[-1] + max(int(n), 0))
        read.current = st = W.Step(meta, batch, lens, wo, host, k, v)
    credited_window(kv_cache, meta.block_table, meta.slot_mapping, st.lens, st.wo, st.k, st.v, layer, read.handle)
    window_attention(query, st.k, st.v, st.offsets, meta.query_start_loc, meta.seq_lens, k_scales, scale, output)
