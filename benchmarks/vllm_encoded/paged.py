"""`paged.cu`'s launchers over torch tensors: a layer of vLLM's paged KV cache, plain or in kv.cu's codec, its decode
attention, the append of a step's new tokens and the gather of whole pages.

A layer's cache is vLLM's tensor ([blocks][KV heads][slots][K ‖ V] bf16, or the inner three in any order); `Layout`
reads its strides in words, and the words of layer l start at word index l · words_per_layer, so the pool's words have
distinct indices."""
from __future__ import annotations

import ctypes
import math
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "word_codec"))

import kv as K  # noqa: E402
import kv_bench as KB  # noqa: E402

QSCALE = math.log2(math.e) / math.sqrt(K.HEAD_DIM)


class PagedLib(KB.Lib):
    """kv.cu with paged.cu's launchers (`run.sh` builds it at the stack's field, mask, FOLD6 and query heads)."""

    def __init__(self, name: str, path: str) -> None:
        super().__init__(name, path)
        v, i, z, u = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_ulonglong
        so = self.so
        so.pg_shape.argtypes = [v]
        so.pg_attn.argtypes = [v, i, v, z, ctypes.c_float, v, z, i, v, v, v, v, i, i, i, i, i, u, i, i, i, v, v, z, z, v]
        so.pg_append_multi.argtypes = [v, i, i, i, i, i, i, i, i, v]
        so.pg_append.argtypes = [v, v, z, z, v, i, v, i, i, i, i, u, i, i, v]
        so.pg_gather.argtypes = [v, v, i, i, v, u, i, i, v]
        so.pg_reduce.argtypes = [v, i, i, i, v, i, v, v, v]
        a = (ctypes.c_int * 10)()
        so.pg_shape(ctypes.cast(a, ctypes.c_void_p))
        self.shape = dict(zip(("heads", "kv_heads", "head_dim", "hw", "wb", "mt", "part", "mw", "mma", "smem"), a))
        self.fold6 = so.kv_fold6()
        if not self.shape["mma"]:
            raise RuntimeError(f"{name}: the field's words do not tile a head")
        if self.shape["hw"] * self.shape["wb"] != 4 * K.HEAD_DIM:
            raise RuntimeError(f"{name}: a (slot, head)'s K ‖ V is not {self.shape['hw']} whole words")


@dataclass(frozen=True)
class Layout:
    """A layer cache's shape in words: blocks, slots a block, strides between blocks, heads and slots, a block's words."""
    blocks: int
    bs: int
    sb: int
    sh: int
    ss: int

    @property
    def words(self) -> int:
        return self.blocks * self.sb


def layout(cache, kv_heads: int, wb: int, dims=(0, 1, 2)) -> Layout:
    """cache's layout, its dims (block, head, slot) in `dims` and K ‖ V its last, contiguous; refuses any whose block is
    not contiguous (the gather copies whole blocks) or whose strides are not whole words."""
    es = cache.element_size()
    if cache.dim() != 4 or cache.stride(3) != 1 or cache.shape[3] * es != 4 * K.HEAD_DIM:
        raise ValueError(f"a layer cache is [blocks][heads][slots][K ‖ V]: {tuple(cache.shape)} {cache.stride()}")
    b, h, s = dims
    if cache.shape[h] != kv_heads:
        raise ValueError(f"{cache.shape[h]} KV heads, the library's {kv_heads}")
    st = [cache.stride(d) * es for d in (b, h, s)]
    if any(x % wb for x in st):
        raise ValueError(f"strides {st} bytes are not whole {wb}-byte words")
    sb, sh, ss = (x // wb for x in st)
    bs, hw = cache.shape[s], 4 * K.HEAD_DIM // wb
    if sb != kv_heads * bs * hw or sorted((sh, ss)) != sorted((hw, kv_heads * hw) if sh < ss else (bs * hw, hw)):
        raise ValueError(f"a block is not contiguous: {tuple(cache.shape)} {cache.stride()}")
    return Layout(cache.shape[b], bs, sb, sh, ss)


def stream(torch) -> int:
    return torch.cuda.current_stream().cuda_stream


def attn(lib: PagedLib, torch, cache, lay: Layout, lbase: int, encoded: int, k: int, q, out, part, lens, bt,
         batch: int, splits: int, nrows: int | None = None, qrow=None, ck=None, cv=None) -> None:
    """Decode attention of batch sequences (lens [≥ batch] int32, bt [≥ batch][bstride] int32) over one layer: q and out
    are [rows][heads · head_dim] bf16 (any row stride), sequence i's at row qrow[i] (or i); rows batch … nrows − 1
    zeroed (nrows default q's rows); part holds batch · splits · MW partials. ck, cv ([rows][KV heads · head_dim] bf16,
    any row stride): each sequence's last token's K and V at its row, in place of its pages'."""
    nrows = q.shape[0] if nrows is None else nrows
    if (ck is None) != (cv is None):
        raise ValueError("ck and cv go together")
    if part.numel() < part_floats(lib, batch, splits):
        raise ValueError(f"part holds {part.numel()} floats, batch {batch} at {splits} splits needs "
                         f"{part_floats(lib, batch, splits)}")
    lib.call("pg_attn", cache.data_ptr(), encoded, q.data_ptr(), q.stride(0), QSCALE, out.data_ptr(), out.stride(0),
             nrows, part.data_ptr(), lens.data_ptr(), bt.data_ptr(), 0 if qrow is None else qrow.data_ptr(),
             bt.stride(0), lay.bs, lay.sb, lay.sh, lay.ss, lbase, batch, splits, k,
             0 if ck is None else ck.data_ptr(), 0 if cv is None else cv.data_ptr(),
             0 if ck is None else ck.stride(0), 0 if cv is None else cv.stride(0), stream(torch))


def append(lib: PagedLib, torch, cache, lay: Layout, lbase: int, encoded: int, k: int, key, val, slots) -> None:
    """Tokens key[i], val[i] ([tokens][KV heads · head_dim] bf16, any row stride) at slot slots[i] (int64; < 0 skipped)."""
    lib.call("pg_append", key.data_ptr(), val.data_ptr(), key.stride(0), val.stride(0), slots.data_ptr(),
             slots.shape[0], cache.data_ptr(), lay.bs, lay.sb, lay.sh, lay.ss, lbase, encoded, k, stream(torch))


def table(torch, entries):
    """pg_append_multi's table on the host: a layer's (key, value, slots, rows, cache, lbase) each, tensors as pointers."""
    rows = [[key.data_ptr(), val.data_ptr(), key.stride(0), val.stride(0), slots.data_ptr(), n, cache.data_ptr(), lbase]
            for key, val, slots, n, cache, lbase in entries]
    return torch.tensor(rows, dtype=torch.int64).reshape(-1)


def append_multi(lib: PagedLib, torch, tab, layers: int, rows: int, lay: Layout, encoded: int, k: int, s=None) -> None:
    """Every layer's appends of tab (a device int64 tensor of `table`'s) in one launch on stream s (default current)."""
    lib.call("pg_append_multi", tab.data_ptr(), layers, rows, lay.bs, lay.sb, lay.sh, lay.ss, encoded, k,
             stream(torch) if s is None else s)


def gather(lib: PagedLib, torch, cache, lay: Layout, lbase: int, encoded: int, k: int, blocks, out) -> None:
    """Blocks blocks[b] (int32) of the layer into out's block b, decoded (encoded) or copied."""
    lib.call("pg_gather", cache.data_ptr(), blocks.data_ptr(), blocks.shape[0], lay.sb, out.data_ptr(), lbase, encoded,
             k, stream(torch))


def part_floats(lib: PagedLib, batch: int, splits: int) -> int:
    return batch * splits * lib.shape["mw"] * lib.shape["part"]


def splits_for(batch: int, sms: int, kv_heads: int, cap: int = 32) -> int:
    """A decode step's split count: about two CTAs an SM over batch × splits × KV heads."""
    return max(1, min(cap, -(-2 * sms // max(1, batch * kv_heads))))
