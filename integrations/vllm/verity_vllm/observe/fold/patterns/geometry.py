"""View geometry of the patterns' operands.

Geometry helpers
----------------
The few view manipulations the patterns need
(`sub_rows`, `rows_operand`, `split_heads`, `as_weight_view`, `as_slot_rows`) live here; they are pure functions on
`ViewDesc`.
"""

from __future__ import annotations

from collections.abc import Sequence

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.resolver import Gather, Operand, PerRow

INT_DTYPES = frozenset({"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "bool"})


class GeometryError(ValueError):
    pass


def is_int(v: ViewDesc) -> bool:
    return v.dtype in INT_DTYPES


def contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    out: list[int] = []
    acc = 1
    for s in reversed(shape):
        out.append(acc)
        acc *= max(int(s), 1)
    return tuple(reversed(out))


def is_row_major(v: ViewDesc) -> bool:
    """Strides are the contiguous ones (dimensions of extent 1 have irrelevant strides)."""
    exp = contiguous_strides(v.shape)
    return all(s == 1 or a == b for s, a, b in zip(v.shape, v.strides, exp))


def sub_rows(v: ViewDesc, lo: int, hi: int) -> ViewDesc:
    """`v[lo:hi]` along the leading axis."""
    if v.rank < 1 or not 0 <= lo <= hi <= v.shape[0]:
        raise GeometryError(f"sub_rows({v!r}, {lo}, {hi})")
    return ViewDesc(v.alloc, v.offset + lo * v.strides[0], (hi - lo,) + v.shape[1:], v.strides, v.dtype, v.device)


def rows_operand(v: ViewDesc, rows: Sequence[int]) -> Operand:
    """The rows `rows` of `v` as an operand: `PerRow` of a leading-axis slice when they are consecutive, else a
    `Gather` with the (public) row indices."""
    rows = list(rows)
    if not rows:
        raise GeometryError("rows_operand: empty row set")
    if rows == list(range(rows[0], rows[0] + len(rows))):
        return PerRow(sub_rows(v, rows[0], rows[-1] + 1))
    return Gather(v, tuple(rows))


def split_heads(v: ViewDesc, nheads: int, d: int) -> ViewDesc:
    """`[M, nheads*d]` (unit inner stride) -> `[M, nheads, d]`; a `[M, nheads, d]` view with strides `(s, d, 1)`
    is returned unchanged."""
    if v.rank == 3:
        if v.shape[1:] == (nheads, d) and v.strides[1:] == (d, 1):
            return v
        raise GeometryError(f"split_heads: {v!r} is not [M,{nheads},{d}] with strides (*, {d}, 1)")
    if v.rank != 2 or v.shape[1] != nheads * d or v.strides[1] != 1:
        raise GeometryError(f"split_heads: {v!r} is not [M,{nheads * d}] with unit inner stride")
    return ViewDesc(v.alloc, v.offset, (v.shape[0], nheads, d), (v.strides[0], d, 1), v.dtype, v.device)


def as_weight_view(b: ViewDesc, K: int, N: int) -> ViewDesc:
    """The GEMM's `b_ptr` is `weight.t()`: a `[K, N]` view with strides `(1, K)` of the `[N, K]` row-major
    parameter.  Returns that parameter view `[N, K]` strides `(K, 1)`."""
    if b.shape == (K, N) and b.strides == (1, K):
        return ViewDesc(b.alloc, b.offset, (N, K), (K, 1), b.dtype, b.device)
    if b.shape == (N, K) and b.strides == (K, 1):
        return b
    raise GeometryError(f"as_weight_view: {b!r} is neither [K={K},N={N}] strides (1,K) nor the [N,K] row-major weight")


def as_slot_rows(cache: ViewDesc) -> ViewDesc:
    """A per-layer FlashAttention KV cache view `[num_blocks, block_size, KVH, D]` reinterpreted as
    `[num_blocks*block_size, KVH, D]`, one row per KV slot.  The slot rows must be uniformly strided
    (`strides[0] == block_size * strides[1]`, so slot `b*bs + o` is at `(b*bs + o) * strides[1]`) with unit head-dim
    stride; heads may be padded (κ's cache is `[blocks][bs][KVH][K:D | V:D]`: strides `(bs*KVH*2D, KVH*2D, 2D, 1)`,
    the value cache at `+D`).  A `[slots, KVH, D]` view with unit inner stride is returned as is."""
    if cache.rank == 4:
        nb, bs, kvh, d = cache.shape
        s_b, s_o, s_h, s_d = cache.strides
        if s_d == 1 and s_h >= d and s_o >= kvh * s_h and s_b == bs * s_o:
            return ViewDesc(cache.alloc, cache.offset, (nb * bs, kvh, d), (s_o, s_h, 1), cache.dtype, cache.device)
        raise GeometryError(f"as_slot_rows: {cache!r} is not a uniformly strided [blocks, block_size, KVH, D] cache")
    if cache.rank == 3 and cache.strides[2] == 1 and cache.strides[1] >= cache.shape[2] and cache.strides[0] >= cache.shape[1] * cache.strides[1]:
        return cache
    raise GeometryError(f"as_slot_rows: unexpected KV cache view {cache!r}")
