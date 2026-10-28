"""The exact-int8 profile's calls as named torch ops, so that a capture sees each int8 linear as two nodes, its quantizer and its
product, and each layer's int8 attention as two, its batch's KV quantization and its batch's rows (`observe.fold.patterns.int8`):

    verity_int8::row_quant(Tensor x) -> (Tensor xq, Tensor s_a)
        x bf16 [M, K]: `engine.int8_exact.quantize_rows`, xq int8 [M, K] and s_a f32 [M] (`registry.int8_exact.Int8RowQuant_v1{K}`
        per row)
    verity_int8::scaled_mm(Tensor xq, Tensor s_a, Tensor w, Tensor s_w) -> Tensor y
        w int8 [N, K], s_w f32 [N, 1] (or [N]): the plain linear's product of row_quant's results, y bf16 [M, N]
        (`registry.int8_exact.ScaledMmS8_v1{K,N}` per row)
    verity_int8::scaled_mm_zy(Tensor xq, Tensor s_a, Tensor w, Tensor s_w, int rows) -> (Tensor y, Tensor zy)
        scaled_mm's y, and from the same product (on CUDA the same accumulator) the call's z/y records over `rows` slots
        (T), zy uint8 [units, `int8_exact.ZY_BYTES`]: a 32 x 32 unit's exact s32 z, then its y = dq(z) (`int8_exact.zy_records`),
        what H1 commits as one hiding leaf a unit

The quantizer's results are a separate node's, read only by the product inside the linear's body, so a Program cuts them as
values the linear's two Calls exchange (`call_boundaries`), never as a second return of the linear.  `linear` runs the two.
On CUDA each of the two is one kernel (`engine.int8_fused`: Triton, or on sm_90 at a `WGMMA` row `engine.int8_wgmma`'s) with
int8_exact's values bit for bit; the torch bodies serve every other device and stay the reference.
    verity_int8::kv_quant_batch(Tensor key_cache, Tensor value_cache, Tensor block_table, int[] lens, Tensor k_scales)
            -> (Tensor kq, Tensor vq, Tensor sv, Tensor v0)
        every request r's first lens[r] tokens in the paged caches [num_blocks, block_size, G, d] under its block-table row, with
        the layer's K scales [G]: K' and V' int8 [B, G, s_max, d] (0 past each window), s_v and f32(V'(0)) float32 [B, G, d];
        request r's K'[r, g, :s], V'[r, g, :s], s_v[r, g], f32(V'(0))[r, g] are `registry.int8_exact.Int8KvQuant_v1{s,d}`'s
        per KV head g (s = lens[r]; a request that brings no rows has lens 0 and no window)
    verity_int8::window_rows_batch(Tensor query, Tensor kq, Tensor vq, Tensor sv, Tensor v0, Tensor k_scales, int[] qsl, int[] lens,
            float scale) -> Tensor out
        every query row of the step, query bf16 [T, H, d], request r's rows qsl[r] .. qsl[r+1] - 1 the last of its lens[r]-token
        window, over kv_quant_batch's results (`registry.int8_exact.Int8LutAttentionRow_v1{d,pos+1,scale}` per (row, head))
    verity_int8::kv_quant_window_batch(Tensor keys, Tensor values, int[] starts, int[] lens, Tensor k_scales)
            -> (Tensor kq, Tensor vq, Tensor sv, Tensor v0)
        kv_quant_batch of windows already read, request r's at rows starts[r] .. starts[r] + lens[r] - 1 of keys and values bf16
        [rows, G, d] (a credited read's rows, `engine.int8_window`)

No Python loop runs over requests or KV heads: the ops take the step's offsets and lengths as host lists (read once a step,
`host_lists`) and compute every window at once, padded to the step's longest, each value bit for bit the per-request form's
(`int8_attention.reference_paged`, `kv_quant` and `window_rows`, the tests' reference).  On CUDA each of the three is one
Triton launch (`engine.int8_attention_triton`), bit for bit its torch body, which serves every other device.
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

import functools

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


@row_quant.register_kernel("cuda")
def _row_quant_cuda(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from verity_vllm.engine import int8_fused
    return int8_fused.row_quant(x)


@scaled_mm.register_kernel("cuda")
def _scaled_mm_cuda(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    from verity_vllm.engine import int8_fused
    return int8_fused.product(xq, s_a, w, s_w)


# Defined through a Library, not `custom_op`: `protocol_options.credit_gate`'s trace runs a custom op's Python body inside
# itself, where the CUDA body's Triton GEMM is no dispatcher op and would count as no GEMM; an op with no Python
# implementation it sees is one GEMM, whose outputs (y and zy) it wrote.
_LIB = torch.library.Library(LIBRARY, "FRAGMENT")
_LIB.define("scaled_mm_zy(Tensor xq, Tensor s_a, Tensor w, Tensor s_w, int rows) -> (Tensor y, Tensor zy)")


def _scaled_mm_zy_cuda(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor,
                       rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    from verity_vllm.engine import int8_fused
    return int8_fused.product_zy(xq, s_a, w, s_w, rows)


_LIB.impl("scaled_mm_zy", E.scaled_mm_zy, "CPU")
_LIB.impl("scaled_mm_zy", _scaled_mm_zy_cuda, "CUDA")


@torch.library.register_fake(f"{LIBRARY}::scaled_mm_zy", lib=_LIB)
def _scaled_mm_zy_fake(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor,
                       rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = xq.shape[0], w.shape[0]
    return xq.new_empty((m, n), dtype=torch.bfloat16), xq.new_empty((E.zy_units(m, n, rows), E.ZY_BYTES), dtype=torch.uint8)


scaled_mm_zy = torch.ops.verity_int8.scaled_mm_zy.default


def linear(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The plain linear as its two ops: (xq, s_a, y)."""
    xq, s_a = row_quant(x)
    return xq, s_a, scaled_mm(xq, s_a, w, s_w)


def linear_zy(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor,
              rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The plain linear as ``row_quant`` then ``scaled_mm_zy`` over `rows` slots: (xq, s_a, y, zy), y bit for bit `linear`'s."""
    xq, s_a = row_quant(x)
    y, zy = scaled_mm_zy(xq, s_a, w, s_w, rows)
    return xq, s_a, y, zy


def _kv_fake(like: torch.Tensor, b: int, g: int, s: int, d: int):
    q8 = like.new_empty((b, g, s, d), dtype=torch.int8)
    return q8, torch.empty_like(q8), like.new_empty((b, g, d), dtype=torch.float32), like.new_empty((b, g, d), dtype=torch.float32)


@torch.library.custom_op(f"{LIBRARY}::kv_quant_batch", mutates_args=())
def kv_quant_batch(key_cache: torch.Tensor, value_cache: torch.Tensor, block_table: torch.Tensor, lens: list[int],
                   k_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k, v = A.paged_windows(key_cache, value_cache, block_table, lens)
    A.batch_check(k, k_scales, lens)
    return A.kv_quant_batch(k, v, k_scales)


@kv_quant_batch.register_fake
def _kv_quant_batch_fake(key_cache, value_cache, block_table, lens, k_scales):
    return _kv_fake(key_cache, len(lens), key_cache.shape[2], max(lens, default=0), key_cache.shape[3])


@torch.library.custom_op(f"{LIBRARY}::kv_quant_window_batch", mutates_args=())
def kv_quant_window_batch(keys: torch.Tensor, values: torch.Tensor, starts: list[int], lens: list[int],
                          k_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(starts) != len(lens) or any(o < 0 or o + s > keys.shape[0] for o, s in zip(starts, lens) if s):
        raise ValueError(f"int8 attention: windows at rows {starts[:8]} of lengths {lens[:8]} past the {keys.shape[0]} rows read")
    k, v = A.packed_windows(keys, values, starts, lens)
    A.batch_check(k, k_scales, lens)
    return A.kv_quant_batch(k, v, k_scales)


@kv_quant_window_batch.register_fake
def _kv_quant_window_batch_fake(keys, values, starts, lens, k_scales):
    return _kv_fake(keys, len(lens), keys.shape[1], max(lens, default=0), keys.shape[2])


@kv_quant_batch.register_kernel("cuda")
def _kv_quant_batch_cuda(key_cache: torch.Tensor, value_cache: torch.Tensor, block_table: torch.Tensor, lens: list[int],
                         k_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from verity_vllm.engine import int8_attention_triton
    return int8_attention_triton.kv_quant_paged(key_cache, value_cache, block_table, lens, k_scales)


@kv_quant_window_batch.register_kernel("cuda")
def _kv_quant_window_batch_cuda(keys: torch.Tensor, values: torch.Tensor, starts: list[int], lens: list[int],
                                k_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from verity_vllm.engine import int8_attention_triton
    if len(starts) != len(lens) or any(o < 0 or o + s > keys.shape[0] for o, s in zip(starts, lens) if s):
        raise ValueError(f"int8 attention: windows at rows {starts[:8]} of lengths {lens[:8]} past the {keys.shape[0]} rows read")
    return int8_attention_triton.kv_quant_packed(keys, values, starts, lens, k_scales)


@torch.library.custom_op(f"{LIBRARY}::window_rows_batch", mutates_args=())
def window_rows_batch(query: torch.Tensor, kq: torch.Tensor, vq: torch.Tensor, sv: torch.Tensor, v0: torch.Tensor, k_scales: torch.Tensor,
                      qsl: list[int], lens: list[int], scale: float) -> torch.Tensor:
    return A.batch_rows(query, kq, vq, sv, v0, k_scales, *_rows_of_step(query, qsl, lens), scale)


@window_rows_batch.register_fake
def _window_rows_batch_fake(query, kq, vq, sv, v0, k_scales, qsl, lens, scale) -> torch.Tensor:
    return torch.empty_like(query)


@window_rows_batch.register_kernel("cuda")
def _window_rows_batch_cuda(query: torch.Tensor, kq: torch.Tensor, vq: torch.Tensor, sv: torch.Tensor, v0: torch.Tensor,
                            k_scales: torch.Tensor, qsl: list[int], lens: list[int], scale: float) -> torch.Tensor:
    from verity_vllm.engine import int8_attention_triton
    return int8_attention_triton.batch_rows(query, kq, vq, sv, v0, k_scales, *_rows_of_step(query, qsl, lens), scale)


def _rows_of_step(query: torch.Tensor, qsl: list[int], lens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    if not qsl or qsl[0] != 0 or qsl[-1] != query.shape[0] or len(lens) != len(qsl) - 1:
        raise ValueError(f"int8 attention: {query.shape[0]} query rows under start offsets {qsl[:8]} and {len(lens)} windows")
    if why := A.step_refusal(qsl, lens):
        raise ValueError(why)
    return _token_rows(tuple(qsl), tuple(lens), query.device)


@functools.lru_cache(maxsize=4)
def _token_rows(qsl: tuple[int, ...], lens: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """`int8_attention.token_rows`, once a step: every layer of a step reads the same rows."""
    return A.token_rows(list(qsl), list(lens), device)


_HOST: list = [None, None]


def host_lists(meta) -> tuple[list[int], list[int]]:
    """(query_start_loc, seq_lens) of a step's attention metadata as host lists, read once a step (every layer shares ``meta``)."""
    if _HOST[0] is not meta:
        _HOST[:] = [meta, (meta.query_start_loc.tolist(), meta.seq_lens.tolist())]
    return _HOST[1]


def _step(query_start_loc: torch.Tensor, seq_lens: torch.Tensor, host) -> tuple[list[int], list[int]] | None:
    """The step's (qsl, lens), lens 0 for a request that brings no rows (its window is read by none); None for a step of no rows."""
    qsl, lens = host if host is not None else (query_start_loc.tolist(), seq_lens.tolist())
    b = len(qsl) - 1
    if why := A.step_refusal(qsl, lens):
        raise ValueError(why)
    if b <= 0 or qsl[-1] == qsl[0]:
        return None
    return list(qsl), [s if qsl[r + 1] > qsl[r] else 0 for r, s in enumerate(lens[:b])]


def paged_attention(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, query_start_loc: torch.Tensor,
                    seq_lens: torch.Tensor, block_table: torch.Tensor, k_scales: torch.Tensor, scale: float, output: torch.Tensor,
                    host: tuple[list[int], list[int]] | None = None) -> None:
    """Every request of a batch, one window each, into output [T, H, d]: request r's query rows query[qsl[r]:qsl[r+1]] are its
    window's last rows, its window its first seq_lens[r] tokens in the caches (vLLM's FlashAttention metadata and layout), in ONE
    ``verity_int8::kv_quant_batch`` and ONE ``verity_int8::window_rows_batch`` call for the whole batch, every request and KV
    head at once (`int8_attention.reference_paged` is the same values request by request).  Its m rows are positions
    s - m .. s - 1: one for a decode step, s for a whole prompt, and a chunked prefill's chunk in between; a request with more
    rows than keys is refused.  ``host`` is (query_start_loc, seq_lens) as host lists when the caller has them (`host_lists`)."""
    step = _step(query_start_loc, seq_lens, host)
    if step is None:
        return
    qsl, lens = step
    kv = kv_quant_batch(key_cache, value_cache, block_table[:len(lens)], lens, k_scales)
    output[:qsl[-1]] = window_rows_batch(query[:qsl[-1]], *kv, k_scales, qsl, lens, float(scale))


def window_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, offsets: list[int], query_start_loc: torch.Tensor,
                     seq_lens: torch.Tensor, k_scales: torch.Tensor, scale: float, output: torch.Tensor,
                     host: tuple[list[int], list[int]] | None = None) -> None:
    """`paged_attention` over a step's windows already read (`engine.int8_window`): request r's window is rows
    offsets[r] .. offsets[r] + seq_lens[r] - 1 of keys and values [rows, G, d] bf16, in ONE ``verity_int8::kv_quant_window_batch``
    and ONE ``verity_int8::window_rows_batch`` call."""
    step = _step(query_start_loc, seq_lens, host)
    if step is None:
        return
    qsl, lens = step
    kv = kv_quant_window_batch(keys, values, [int(o) for o in offsets[:len(lens)]], lens, k_scales)
    output[:qsl[-1]] = window_rows_batch(query[:qsl[-1]], *kv, k_scales, qsl, lens, float(scale))


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
    window_attention(query, st.k, st.v, st.offsets, meta.query_start_loc, meta.seq_lens, k_scales, scale, output, host=host_lists(meta))
