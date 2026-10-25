"""The staging decode: a layer's matrices from the P2-encoded store into plaintext staging buffers on the GPU, by
memory-accounting's decode unchanged (`libp2v1.so` from `benchmarks/pous/p2_v1/`, built for sm_90a by `build.sh`, through
its own wrapper `p2_v1.gpu.Kernels`), with the refusals `schemes.p2.decode_segment` makes that the library's fast decode
leaves to its caller.

`Store` is the encoded rows of a range of W's blocks resident on the device (`rows`, [n, 2056] uint8, global block `first`
first) under a `layout.StoreLayout`. `decode_matrix(store, layer, tensor, out)` decodes the blocks holding one matrix
(`layout.MATRICES`: QKV is wq, wk and wv one after another, [16512, 16384] at the frozen shape) or one stream tensor,
`chunk` blocks at a time: `Kernels.decode` expands each block's K and t on the device (`p2v1_keys`, v4) and decodes
(`p2v1_decode`), and `scatter` copies the payload bytes 2 .. 2055 of each decoded row that fall in the tensor to their
place in `out`, the tensor's bytes (any dtype, contiguous). Nothing is re-laid out: W holds each tensor's bytes as the
GEMM reads them ([N, K] row-major s8, f32 little-endian scales).

Every chunk is screened on the device and nothing is fixed silently: a decoded row whose two lead bytes aren't zero is
wider than the payload (not a codeword: a corrupt row or the wrong keys), and a row whose first eight bytes are all 0xFF
is compared with p exactly (a row at or above p is not canonical). `Store.check` reads those counts and the library's own
flags (a tweak's first candidate at or above p) and raises `StoreError`; `decode_matrix` calls it before it returns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:
    from . import layout as L
except ImportError:
    import layout as L  # type: ignore[no-redef]

from p2_v1 import gpu  # noqa: E402  (layout put benchmarks/pous on the path)

P_BYTES = L.SPEC.p.to_bytes(L.ROW, "big")


class StoreError(RuntimeError):
    """The store held something the decode refuses."""


def scatter(m: Any, j0: int, out: Any, offset: int, size: int) -> int:
    """Copy the payload bytes of decoded rows `m` ([n, 2056] uint8, global blocks `j0 ..`) that fall in W bytes
    [offset, offset + size) to `out` (that range's bytes, flat uint8); the number of bytes copied."""
    n = m.shape[0]
    lo, hi = max(offset, j0 * L.PAYLOAD), min(offset + size, (j0 + n) * L.PAYLOAD)
    done = 0
    while lo < hi:
        i, col = divmod(lo - j0 * L.PAYLOAD, L.PAYLOAD)
        if col or hi - lo < L.PAYLOAD:
            k = min(L.PAYLOAD - col, hi - lo)
            out[lo - offset:lo - offset + k].copy_(m[i, L.LEAD + col:L.LEAD + col + k])
        else:
            rows = (hi - lo) // L.PAYLOAD
            k = rows * L.PAYLOAD
            out[lo - offset:lo - offset + k].view(rows, L.PAYLOAD).copy_(m[i:i + rows, L.LEAD:])
        lo, done = lo + k, done + k
    return done


def flat(buf: Any) -> Any:
    """`buf`'s bytes as a flat uint8 view (a staging buffer is any contiguous tensor)."""
    if not buf.is_contiguous():
        raise ValueError("a staging buffer is contiguous")
    return buf.view(-1).view(torch.uint8)


class Store:
    """Encoded rows of W's blocks `first .. first + rows.shape[0] - 1`, resident on the current CUDA device."""

    def __init__(self, layout: L.StoreLayout, rows: Any, first: int, lib_dir: str | Path, chunk: int = 1 << 16) -> None:
        if rows.dtype != torch.uint8 or rows.dim() != 2 or rows.shape[1] != L.ROW or not rows.is_contiguous():
            raise ValueError(f"the store's rows are a contiguous [n, {L.ROW}] uint8 tensor")
        if not rows.is_cuda:
            raise ValueError("the store's rows are on the GPU")
        if first < 0 or chunk < 1:
            raise ValueError("the first block is nonnegative and a chunk at least one block")
        self.layout, self.rows, self.first, self.chunk = layout, rows, first, chunk
        self.kernels = gpu.Kernels(str(lib_dir), "", L.SPEC, layout.prefix)
        self.kernels.raised()
        self._counts = torch.zeros(2, dtype=torch.int64, device=rows.device)
        self._high: list[tuple[int, Any]] = []
        self.blocks_decoded = 0

    @property
    def blocks(self) -> range:
        return range(self.first, self.first + self.rows.shape[0])

    def decode_rows(self, j0: int, j1: int) -> Any:
        """Decoded rows of blocks `j0 .. j1 - 1`, screened (`check` reports what the screen found)."""
        if not (self.first <= j0 <= j1 <= self.blocks.stop):
            raise StoreError(f"blocks [{j0}, {j1}) are not all in the store's [{self.first}, {self.blocks.stop})")
        c = self.rows[j0 - self.first:j1 - self.first]
        m = self.kernels.decode(c, j0)
        high = (c[:, :8] == 0xFF).all(dim=1)
        self._counts += torch.stack(((m[:, :L.LEAD] != 0).any(dim=1).sum(), high.sum()))
        self._high.append((j0, high))
        self.blocks_decoded += j1 - j0
        return m

    def decode_into(self, offset: int, size: int, out: Any) -> None:
        """W bytes [offset, offset + size) into `out` (`size` bytes, flat uint8)."""
        if out.numel() != size:
            raise ValueError(f"{size} bytes don't fit a {out.numel()}-byte buffer")
        r = L.blocks_of(offset, size)
        for j0 in range(r.start, r.stop, self.chunk):
            j1 = min(r.stop, j0 + self.chunk)
            scatter(self.decode_rows(j0, j1), j0, out, offset, size)

    def check(self) -> None:
        """Raise `StoreError` for anything decoded since the last check that the decode refuses."""
        wide, high = (int(v) for v in self._counts.tolist())
        flags = self.kernels.raised()
        pending, self._high = self._high, []
        self._counts.zero_()
        bad = []
        if high:
            for j0, mask in pending:
                for i in mask.nonzero().flatten().tolist():
                    if bytes(self.rows[j0 - self.first + i].cpu().numpy()) >= P_BYTES:
                        bad.append(j0 + i)
        why = []
        if wide:
            why.append(f"{wide} rows decode wider than the {L.PAYLOAD}-byte payload (not codewords: a corrupt row or "
                       "other keys)")
        if bad:
            why.append(f"{len(bad)} rows at or above p (blocks {bad[:8]})")
        if flags & 1:
            why.append("a tweak's first candidate at or above p (the spec draws the next; the library doesn't)")
        if flags & ~1:
            why.append(f"library flags {flags:#x}")
        if why:
            raise StoreError("; ".join(why))


def decode_matrix(store: Store, layer: int, tensor: str, out: Any, *, scales: bool = False, check: bool = True) -> Any:
    """Decode `tensor` of `layer` (a matrix of `layout.MATRICES`, or one stream tensor such as ``wq``) from `store` into
    `out`, a contiguous tensor of exactly its bytes ([N, K] int8 for a matrix, [N] float32 for its ``scales``); `out`."""
    parts = (store.layout.parts(layer, tensor, scales) if tensor in L.MATRICES
             else (store.layout.tensor(layer, f"{tensor}.scale" if scales else tensor),))
    dst = flat(out)
    need = sum(p.size for p in parts)
    if dst.numel() != need:
        raise ValueError(f"layer {layer}'s {tensor}{' scales' if scales else ''} is {need} bytes, the buffer "
                         f"{dst.numel()}")
    at = 0
    for p in parts:
        store.decode_into(p.offset, p.size, dst[at:at + p.size])
        at += p.size
    if check:
        store.check()
    return out


def decode_scales(store: Store, layer: int, tensor: str, out: Any, *, check: bool = True) -> Any:
    return decode_matrix(store, layer, tensor, out, scales=True, check=check)


def encode_rows(kernels: Any, payload: Any, first: int) -> Any:
    """The store's rows for blocks `first ..` carrying `payload` ([n, 2054] uint8 on the device) by the library's encode
    (`p2v1_encode`, two roots a block): how a development store is made when no encoded store is given."""
    n = payload.shape[0]
    m = torch.zeros(n, L.ROW, dtype=torch.uint8, device=payload.device)
    m[:, L.LEAD:] = payload
    c = kernels.encode(m, first)
    f = kernels.raised()
    if f:
        raise StoreError(f"p2v1_encode raised flags {f:#x} on blocks [{first}, {first + n})")
    return c
