"""The staging decode: a layer's matrices from the P2-encoded store into plaintext staging buffers on the GPU, by
memory-accounting's decode unchanged (`libp2v1.so` from `benchmarks/pous/p2_v1/`, built for sm_90a by `build.sh`, through
its own wrapper `p2_v1.gpu.Kernels`), with the refusals the store's reference decode (`p2_v1/store.py`'s `decode_range`)
makes that the library's fast decode leaves to its caller.

`Store` is the encoded rows of a range of the store's blocks resident on the device (`rows`, [n, 2056] uint8, block
`first` first) under a `layout.StoreLayout`. `decode_matrix(store, layer, tensor, out)` decodes one tensor's blocks (a
matrix of `layout.MATRICES`: QKV is [16512, 16384] s8 at the frozen shape) `chunk` blocks at a time: `Kernels.decode`
expands each block's K and t on the device (`p2v1_keys`, v4) and decodes (`p2v1_decode`), and `scatter` copies the
payload bytes 2 .. 2055 of each decoded row that fall in the tensor to their place in `out`. `decode_layer(store, layer,
staging)` is the fixture's `decode_layer`: every tensor of the layer at its `staging_offset`, zero between them. Nothing
is re-laid out: W holds each tensor's bytes as the GEMM reads them.

Every chunk is screened on the device and nothing is fixed silently. `Store.check` raises `StoreError`, with the fixture's
flag bits, for anything decoded since the last check that the reference refuses: bit 2 for a decoded row whose two lead
bytes aren't zero (wider than the payload: a corrupt row or other keys) or a row at or above p (a row whose first eight
bytes are all 0xFF is compared with p exactly), bit 3 for a tensor's last block nonzero past the tensor's end, and the
library's own flags (bit 0: a tweak's first candidate at or above p, where the spec draws the next). `decode_matrix` and
`decode_layer` call it before they return.

With a pass of PoUB attached (`store.sketch = sketch.Pass(...)`), `decode_rows` sketches the units
each chunk's blocks lie in from the same rows on the device before the decode reads them (`sketch.py`); the rows then
hold whole units.

`timed` is the drivers' one timer (`layer.py`, `full.py`, `bw_check.py`).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

try:
    from . import layout as L
except ImportError:
    import layout as L  # type: ignore[no-redef]

from p2_v1 import gpu

P_BYTES = L.SPEC.p.to_bytes(L.ROW, "big")
WIDE, PADDING = 1 << 2, 1 << 3


class StoreError(RuntimeError):
    """The store held something the decode refuses; `flags` holds the fixture's flag bits."""

    def __init__(self, message: str, flags: int) -> None:
        super().__init__(message)
        self.flags = flags


def scatter(m: Any, j0: int, out: Any, offset: int, size: int) -> int:
    """Copy the payload bytes of decoded rows `m` ([n, 2056] uint8, blocks `j0 ..`) that fall in W bytes
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
    """Encoded rows of the store's blocks `first .. first + rows.shape[0] - 1`, on the decode's device."""

    def __init__(self, layout: L.StoreLayout, rows: Any, first: int, lib_dir: str | Path, chunk: int = 1 << 16) -> None:
        if rows.dtype != torch.uint8 or rows.dim() != 2 or rows.shape[1] != L.ROW or not rows.is_contiguous():
            raise ValueError(f"the store's rows are a contiguous [n, {L.ROW}] uint8 tensor")
        if first < 0 or chunk < 1:
            raise ValueError("the first block is nonnegative and a chunk at least one block")
        self.layout, self.rows, self.first, self.chunk = layout, rows, first, chunk
        self.kernels = gpu.Kernels(str(lib_dir), "", layout.spec, layout.prefix)
        if rows.device.type != self.kernels.device.type:
            raise ValueError(f"the store's rows are on {rows.device}, the decode on {self.kernels.device}")
        self.kernels.raised()
        self._counts = torch.zeros(3, dtype=torch.int64, device=rows.device)
        self._high: list[tuple[int, Any]] = []
        self.blocks_decoded = 0
        self.sketch: Any = None

    @property
    def blocks(self) -> range:
        return range(self.first, self.first + self.rows.shape[0])

    def decode_rows(self, j0: int, j1: int) -> Any:
        """Decoded rows of blocks `j0 .. j1 - 1`, screened (`check` reports what the screen found)."""
        if not (self.first <= j0 <= j1 <= self.blocks.stop):
            raise ValueError(f"blocks [{j0}, {j1}) are not all in the store's [{self.first}, {self.blocks.stop})")
        c = self.rows[j0 - self.first:j1 - self.first]
        if self.sketch is not None:
            self.sketch.read(self.rows, self.first, j0, j1)
        m = self.kernels.decode(c, j0)
        high = (c[:, :8] == 0xFF).all(dim=1)
        self._counts[:2] += torch.stack(((m[:, :L.LEAD] != 0).any(dim=1).sum(), high.sum()))
        self._high.append((j0, high))
        self.blocks_decoded += j1 - j0
        return m

    def decode_into(self, offset: int, size: int, out: Any, *, end: bool = False) -> None:
        """W bytes [offset, offset + size) into `out` (`size` bytes, flat uint8). With `end`, the range ends a tensor,
        and the rest of its last block's payload must be zero."""
        if out.numel() != size:
            raise ValueError(f"{size} bytes don't fit a {out.numel()}-byte buffer")
        r = L.blocks_of(offset, size)
        used = offset + size - (r.stop - 1) * L.PAYLOAD
        for j0 in range(r.start, r.stop, self.chunk):
            j1 = min(r.stop, j0 + self.chunk)
            m = self.decode_rows(j0, j1)
            scatter(m, j0, out, offset, size)
            if end and j1 == r.stop and used < L.PAYLOAD:
                self._counts[2] += (m[-1, L.LEAD + used:] != 0).any().to(torch.int64)

    def check(self) -> None:
        """Raise `StoreError` for anything decoded since the last check that the decode refuses."""
        wide, high, padded = (int(v) for v in self._counts.tolist())
        lib = self.kernels.raised()
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
        if padded:
            why.append(f"{padded} tensors' last blocks are nonzero past the tensor's end")
        if lib & 1:
            why.append("a tweak's first candidate at or above p (the spec draws the next; the library doesn't)")
        if lib & ~1:
            why.append(f"library flags {lib:#x}")
        if why:
            raise StoreError("; ".join(why), lib | (WIDE if wide or bad else 0) | (PADDING if padded else 0))


def decode_matrix(store: Store, layer: int, tensor: str, out: Any, *, check: bool = True) -> Any:
    """Decode layer `layer`'s `tensor` (a key of `layout.MATRICES`, or the fixture's name) from `store` into `out`, a
    contiguous tensor of exactly its bytes ([N, K] int8); `out`."""
    t = store.layout.tensor(layer, tensor)
    dst = flat(out)
    if dst.numel() != t.size:
        raise ValueError(f"{t.name} is {t.size} bytes, the buffer {dst.numel()}")
    store.decode_into(t.offset, t.size, dst, end=True)
    if check:
        store.check()
    return out


def decode_layer(store: Store, layer: int, staging: Any, *, check: bool = True) -> Any:
    """Layer `layer`'s staging buffer into `staging` (contiguous, exactly `staging_bytes(layer)` bytes): each tensor at
    its `staging_offset`, zero between them, as the fixture's `decode_layer(c_star, layout, layer, prefix, staging)`
    writes; `staging`."""
    dst, total = flat(staging), store.layout.staging_bytes(layer)
    if dst.numel() != total:
        raise ValueError(f"layer {layer}'s staging buffer is {total} bytes, not {dst.numel()}")
    at = 0
    for t in sorted(store.layout.layer(layer), key=lambda t: t.staging_offset):
        dst[at:t.staging_offset].zero_()
        store.decode_into(t.offset, t.size, dst[t.staging_offset:t.staging_offset + t.size], end=True)
        at = t.staging_offset + t.size
    if check:
        store.check()
    return staging


def encode_rows(kernels: Any, payload: Any, first: int) -> Any:
    """The store's rows for blocks `first ..` carrying `payload` ([n, 2054] uint8 on the device) by the library's encode
    (`p2v1_encode`, two roots a block): how a development store is made when no encoded store is given."""
    n = payload.shape[0]
    m = torch.zeros(n, L.ROW, dtype=torch.uint8, device=payload.device)
    m[:, L.LEAD:] = payload
    c = kernels.encode(m, first)
    f = kernels.raised()
    if f:
        raise StoreError(f"p2v1_encode raised flags {f:#x} on blocks [{first}, {first + n})", f)
    return c


def timed(fn: Callable[[], Any], reps: int, device: torch.device, before: Callable[[], Any] | None = None,
          warmup: int = 0) -> list[float]:
    """The seconds of `reps` runs of `fn` on `device`, each timed alone (by CUDA events on a GPU, else the host's clock),
    after `warmup` untimed runs and a sync, with `before` run untimed ahead of each timed one."""
    for _ in range(warmup):
        fn()
    if warmup and device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        if before is not None:
            before()
        if device.type == "cuda":
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) / 1e3)
        else:
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    return times
