"""`paged.cu`'s launchers over torch tensors: a layer of vLLM's paged KV cache, plain or in kv.cu's codec, its decode
attention, the append of a step's new tokens, the gather of whole pages and a decode step's windows (`window`, what an
attention that reads each request's window whole takes: the exact-int8 attention's).

A layer's cache is vLLM's tensor ([blocks][KV heads][slots][K ‖ V] bf16, or the inner three in any order); `Layout`
reads its physical strides. Codec indices use the registered store instead: after `store_base`, physical slot u and
layer l are the `store-order/v0` token-layer block u · layers + l.

An encoded cache may hold plaintext slots (deferred residency, remote_bandwidth's PROTOCOL.md §4): a token-layer that
PoUB appends is written as it is until its append coin arrives, then encoded in place. Its layer's flags, a
uint8 a physical slot on the device (nonzero: plaintext), go to `attn` and `gather` as `plain`, which read such a slot as
it is and give it no receipt; `append` and `table` set them, `encode` encodes the listed plaintext slots and clears them.
A token-layer kept past the lifetime cap is re-keyed in its slot (§4): `rekey` moves listed slots from their bound offsets
to new ones in place."""
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
        so.pg_attn.argtypes = [v, i, v, z, ctypes.c_float, v, z, i, v, v, v, v, i, i, i, i, i, u, i, i, i, i, i,
                               v, v, z, z, v, v, v, v, v, v]
        so.pg_append_multi.argtypes = [v, i, i, i, i, i, i, i, i, v]
        so.pg_append.argtypes = [v, v, z, z, v, i, v, i, i, i, i, u, i, i, i, i, v, v, v]
        so.pg_encode.argtypes = [v, i, i, i, i, i, i, i, v]
        so.pg_rekey.argtypes = [v, i, i, i, i, i, i, i, i, v]
        so.pg_rekey_ctas.argtypes = []
        so.pg_gather.argtypes = [v, v, i, i, i, i, i, i, v, u, i, i, i, i, v, v, v, v, v, v]
        so.pg_window_offsets.argtypes = [v, i, v, v]
        so.pg_window.argtypes = [v, i, i, v, v, i, i, i, i, i, u, i, i, i, v, v, v, v, z, z, v, v, i, u, v, v, u, u, i, v]
        so.pg_window_ctas.argtypes = [i]
        so.pg_receipts_sparse.argtypes = [v, v, v, v, u, i, i, u, v]
        so.pg_rc_gate_pass.argtypes = [v, v, v]
        so.kv_rc_pass.argtypes = [v, v, v]
        so.pg_reduce.argtypes = [v, i, i, i, v, i, v, v, v]
        so.pg_span.argtypes = [v, i, i, i, i, v]
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


def _ptr(t) -> int:
    return 0 if t is None else t.data_ptr()


def offsets(torch, values: list[int], f: K.Field):
    """Bound offsets as `paged.cu` takes them: [len(values)][NL] int32 on the device, value u's little-endian limbs (each
    canonical, below p: `codec.binding`, or 0 for a slot under the seed's key)."""
    if any(not 0 <= b < f.p for b in values):
        raise ValueError("a bound offset is canonical, below p")
    raw = bytearray(b"".join(b.to_bytes(f.word_bytes, "little") for b in values))
    return torch.frombuffer(raw, dtype=torch.int32).reshape(len(values), f.limbs).cuda()


def attn(lib: PagedLib, torch, cache, lay: Layout, store_base: int, layers: int, layer: int, encoded: int, k: int,
         q, out, part, lens, bt, batch: int, splits: int, nrows: int | None = None, qrow=None, ck=None, cv=None,
         receipts=None, bj=None, plain=None) -> None:
    """Decode attention of batch sequences (lens [≥ batch] int32, bt [≥ batch][bstride] int32) over one layer: q and out
    are [rows][heads · head_dim] bf16 (any row stride), sequence i's at row qrow[i] (or i); rows batch … nrows − 1
    zeroed (nrows default q's rows); part holds batch · splits · MW partials. ck, cv ([rows][KV heads · head_dim] bf16,
    any row stride): each sequence's last token's K and V at its row, in place of its pages'. bj (`offsets`, one a
    physical slot): the layer's bound offsets, an encoded cache's token-layers decoded under them. plain (uint8, one a
    physical slot): the layer's flags, a flagged slot read as it is and given no receipt."""
    nrows = q.shape[0] if nrows is None else nrows
    if (ck is None) != (cv is None):
        raise ValueError("ck and cv go together")
    if part.numel() < part_floats(lib, batch, splits):
        raise ValueError(f"part holds {part.numel()} floats, batch {batch} at {splits} splits needs "
                         f"{part_floats(lib, batch, splits)}")
    rc_sk, used, epoch = receipts or (None, None, None)
    lib.call("pg_attn", cache.data_ptr(), encoded, q.data_ptr(), q.stride(0), QSCALE, out.data_ptr(), out.stride(0),
             nrows, part.data_ptr(), lens.data_ptr(), bt.data_ptr(), 0 if qrow is None else qrow.data_ptr(),
             bt.stride(0), lay.bs, lay.sb, lay.sh, lay.ss, store_base, layers, layer, batch, splits, k,
             0 if ck is None else ck.data_ptr(), 0 if cv is None else cv.data_ptr(),
             0 if ck is None else ck.stride(0), 0 if cv is None else cv.stride(0),
             0 if rc_sk is None else rc_sk.data_ptr(), 0 if used is None else used.data_ptr(),
             0 if epoch is None else epoch.data_ptr(), _ptr(bj), _ptr(plain), stream(torch))


def append(lib: PagedLib, torch, cache, lay: Layout, store_base: int, layers: int, layer: int, encoded: int, k: int,
           key, val, slots, bj=None, plain=None) -> None:
    """Tokens key[i], val[i] ([tokens][KV heads · head_dim] bf16, any row stride) at slot slots[i] (int64; < 0 skipped),
    encoded under their slots' bound offsets bj (`offsets`, one a physical slot; None: the seed's key). plain (uint8,
    one a physical slot): each appended slot's flag set to `not encoded`."""
    lib.call("pg_append", key.data_ptr(), val.data_ptr(), key.stride(0), val.stride(0), slots.data_ptr(),
             slots.shape[0], cache.data_ptr(), lay.bs, lay.sb, lay.sh, lay.ss, store_base, layers, layer, encoded, k,
             _ptr(bj), _ptr(plain), stream(torch))


def _optional(rest, n: int) -> list:
    return [_ptr(t) for t in (list(rest) + [None] * n)[:n]]


def table(torch, entries):
    """pg_append_multi's host table: (key, value, slots, rows, cache, store base, layer count, layer index[, bound
    offsets[, flags]]) each, the offsets as `append`'s bj (absent or None: the seed's key), the flags as its plain."""
    rows = [[key.data_ptr(), val.data_ptr(), key.stride(0), val.stride(0), slots.data_ptr(), n, cache.data_ptr(),
             store_base, layers, layer, *_optional(rest, 2)]
            for key, val, slots, n, cache, store_base, layers, layer, *rest in entries]
    return torch.tensor(rows, dtype=torch.int64).reshape(-1)


def encode_table(torch, entries):
    """pg_encode's host table: (slots, rows, cache, store base, layer count, layer index[, bound offsets[, flags]]) each,
    slots int64 (distinct; < 0 skipped), the offsets b_j(c_a) as `append`'s bj, the flags as its plain (absent or None:
    every listed slot encoded)."""
    rows = [[slots.data_ptr(), n, cache.data_ptr(), store_base, layers, layer, *_optional(rest, 2)]
            for slots, n, cache, store_base, layers, layer, *rest in entries]
    return torch.tensor(rows, dtype=torch.int64).reshape(-1)


def encode(lib: PagedLib, torch, tab, layers: int, rows: int, lay: Layout, k: int, s=None) -> None:
    """Deferred residency's encode on stream s (default current), once the append's coin has arrived: every layer's
    listed slots of tab (a device int64 tensor of `encode_table`'s) encoded in place under their offsets, a slot whose
    flag is clear left as it is, then every listed slot's flag cleared. Nothing may read those layers meanwhile."""
    lib.call("pg_encode", tab.data_ptr(), layers, rows, lay.bs, lay.sb, lay.sh, lay.ss, k,
             stream(torch) if s is None else s)


def rekey_table(torch, entries):
    """pg_rekey's host table: (slots, rows, cache, store base, layer count, layer index, bound offsets, new offsets[,
    flags]) each, slots int64 (distinct; < 0 skipped), the bound offsets the layer's `offsets` (one a physical slot,
    a listed slot's the old), the new ones `offsets` one a row (b_j'(c_a') of the slot's fresh index j' and coin), the
    flags as `append`'s plain (absent or None: none)."""
    rows = [[slots.data_ptr(), n, cache.data_ptr(), store_base, layers, layer, bj.data_ptr(), new.data_ptr(),
             *_optional(rest, 1)]
            for slots, n, cache, store_base, layers, layer, bj, new, *rest in entries]
    return torch.tensor(rows, dtype=torch.int64).reshape(-1)


def rekey(lib: PagedLib, torch, tab, layers: int, rows: int, lay: Layout, k: int, ctas: int = 0, s=None) -> None:
    """The lifetime cap's re-key on stream s (default current): every layer's listed slots of tab (a device int64
    tensor of `rekey_table`'s) decoded under their offsets and encoded under their new ones in place, a flagged
    (plaintext) slot left as it is, then the new offsets written into the layer's. Nothing may read the listed slots
    meanwhile. ctas: the grid (0: a word a thread; a count: that many CTAs striding over the words). Stream priority
    preempts nothing, so beside serving a caller caps the grid (`pg_rekey`)."""
    lib.call("pg_rekey", tab.data_ptr(), layers, rows, lay.bs, lay.sb, lay.sh, lay.ss, k, ctas,
             stream(torch) if s is None else s)


def append_multi(lib: PagedLib, torch, tab, layers: int, rows: int, lay: Layout, encoded: int, k: int, s=None) -> None:
    """Every layer's appends of tab (a device int64 tensor of `table`'s) in one launch on stream s (default current)."""
    lib.call("pg_append_multi", tab.data_ptr(), layers, rows, lay.bs, lay.sb, lay.sh, lay.ss, encoded, k,
             stream(torch) if s is None else s)


def gather(lib: PagedLib, torch, cache, lay: Layout, store_base: int, layers: int, layer: int, encoded: int, k: int,
           blocks, out, receipts=None, bj=None, plain=None) -> None:
    """Blocks blocks[b] (int32) of the layer into out's block b, decoded (encoded, under bj and plain as `attn` takes
    them) or copied."""
    rc_sk, used, epoch = receipts or (None, None, None)
    lib.call("pg_gather", cache.data_ptr(), blocks.data_ptr(), blocks.shape[0], lay.sb, lay.bs, lay.sb, lay.sh, lay.ss,
             out.data_ptr(), store_base, layers, layer, encoded, k, 0 if rc_sk is None else rc_sk.data_ptr(),
             0 if used is None else used.data_ptr(), 0 if epoch is None else epoch.data_ptr(), _ptr(bj), _ptr(plain),
             stream(torch))


def window_offsets(lib: PagedLib, torch, lens, batch: int, wo) -> None:
    """wo [≥ batch + 1] int32: 0, then the running sums of lens [≥ batch] (negative counted 0), on the device: request r's
    window is rows wo[r] … wo[r + 1] − 1 of `window`'s buffer. One a step: its layers share it."""
    lib.call("pg_window_offsets", lens.data_ptr(), batch, wo.data_ptr(), stream(torch))


def window(lib: PagedLib, torch, cache, lay: Layout, store_base: int, layers: int, layer: int, encoded: int, k: int,
           lens, bt, wo, batch: int, outk, outv, order: int = 0, bj=None, plain=None, ck=None, cv=None, qrow=None,
           ctas: int = 0) -> None:
    """One layer's decode windows (no receipts): request i's tokens 0 … lens[i] − 1 under its block-table row bt[i], K into
    outk and V into outv ([rows][KV heads][head_dim] bf16 views, any row and head strides, the last contiguous) at rows
    wo[i] + t (`window_offsets`), rows past outk's not written. Decoded (encoded, under bj and plain as `attn` takes them)
    or copied, the codec's word order `order` (0: K of every head then V of every head; 1: each head's K ‖ V). ck, cv:
    each sequence's last token's K and V at its row (qrow[i] or i), as `attn` takes them. ctas: the fixed grid (0: every
    SM's resident CTAs)."""
    if (ck is None) != (cv is None):
        raise ValueError("ck and cv go together")
    if outk.shape != outv.shape or outk.dim() != 3 or outk.stride(2) != 1 or outv.stride() != outk.stride():
        raise ValueError(f"outk and outv are [rows][KV heads][head_dim] views alike: {outk.shape} {outk.stride()}, "
                         f"{outv.shape} {outv.stride()}")
    lib.call("pg_window", cache.data_ptr(), encoded, order, lens.data_ptr(), bt.data_ptr(), bt.stride(0), lay.bs,
             lay.sb, lay.sh, lay.ss, store_base, layers, layer, k, _ptr(bj), _ptr(plain), _ptr(ck), _ptr(cv),
             0 if ck is None else ck.stride(0), 0 if cv is None else cv.stride(0), _ptr(qrow), wo.data_ptr(), batch,
             outk.shape[0], outk.data_ptr(), outv.data_ptr(), outk.stride(0), outk.stride(1), ctas, stream(torch))


def receipts_sparse(lib: PagedLib, torch, rc_sk, tags, used, epoch, slots: int, layers: int, layer: int,
                    jbase: int) -> None:
    """Hash this layer's selected sparse token slots into their sparse standard sketch-sha512 receipts."""
    lib.call("pg_receipts_sparse", rc_sk.data_ptr(), tags.data_ptr(), used.data_ptr(), epoch.data_ptr(), slots,
             layers, layer, jbase, stream(torch))


def part_floats(lib: PagedLib, batch: int, splits: int) -> int:
    return batch * splits * lib.shape["mw"] * lib.shape["part"]


def splits_for(batch: int, sms: int, kv_heads: int, cap: int = 32) -> int:
    """A decode step's split count: the largest power of two that keeps batch × splits × KV heads to about eleven CTAs
    an SM (paged_splits.py at p1024, k = 3, 188 SMs, contexts 1280 and 1536: the fastest count, or within 1.5% of it,
    at batches 32, 64 and 128, where two CTAs an SM left 15–17% on the table)."""
    return min(cap, 1 << max(0, (11 * sms // max(1, batch * kv_heads)).bit_length() - 1))
