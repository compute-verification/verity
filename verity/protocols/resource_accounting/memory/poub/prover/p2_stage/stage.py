"""The store's load of a layer: its matrices from the P2-encoded store into plaintext load buffers on the GPU, by
memory-accounting's decode unchanged (`libp2v1.so` from `verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/`, built for sm_90a by
`build.sh`, through its own wrapper `p2_store.gpu.Kernels`) under the store's own load, `p2_store.load`: `EncodedRows`,
the encoded rows of a range of the store's blocks resident on the device, its screen (`EncodedRows.check`, raising
`StoreError` with the flag bits `WIDE` and `PADDING`) and `scatter`. They are defined there and read here by this
directory's modules.

`decode_matrix(store, layer, tensor, out)` decodes one tensor's blocks (a matrix of `layout.MATRICES`: QKV is
[16512, 16384] s8 at the frozen shape) into `out`. `decode_layer(store, layer, staging)` is the fixture's `decode_layer`:
every tensor of the layer at its `staging_offset`, zero between them. Nothing is re-laid out: W holds each tensor's bytes
as the GEMM reads them. Both call `EncodedRows.check` before they return.

`Segments` holds an encoded store's segment files on the host, each read once in block order and held to its
`seg/<s>.json`'s SHA-512, and hands a range of blocks' rows to `DEVICE`; `sync` waits for it.

With a pass of PoUB attached (`store.sketch = sketch.Pass(...)`), `decode_rows` sketches the units
each chunk's blocks lie in from the same rows on the device before the decode reads them (`sketch.py`); the rows then
hold whole units.

`timed` is the drivers' one timer (`layer.py`, `full.py`, `bw_check.py`).
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import layout as L
except ImportError:
    import layout as L  # type: ignore[no-redef]

from p2_store import gpu  # noqa: F401
from p2_store.load import P_BYTES, PADDING, WIDE, EncodedRows, StoreError, scatter  # noqa: F401

DEVICE = torch.device("cuda")


def flat(buf: Any) -> Any:
    """`buf`'s bytes as a flat uint8 view (a load buffer is any contiguous tensor)."""
    if not buf.is_contiguous():
        raise ValueError("a load buffer is contiguous")
    return buf.view(-1).view(torch.uint8)


def decode_matrix(store: EncodedRows, layer: int, tensor: str, out: Any, *, check: bool = True) -> Any:
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


def decode_layer(store: EncodedRows, layer: int, staging: Any, *, check: bool = True) -> Any:
    """Layer `layer`'s load buffer into `staging` (contiguous, exactly `staging_bytes(layer)` bytes): each tensor at
    its `staging_offset`, zero between them, as the fixture's `decode_layer(c_star, layout, layer, prefix, staging)`
    writes; `staging`."""
    dst, total = flat(staging), store.layout.staging_bytes(layer)
    if dst.numel() != total:
        raise ValueError(f"layer {layer}'s load buffer is {total} bytes, not {dst.numel()}")
    at = 0
    for t in sorted(store.layout.layer(layer), key=lambda t: t.staging_offset):
        dst[at:t.staging_offset].zero_()
        store.decode_into(t.offset, t.size, dst[t.staging_offset:t.staging_offset + t.size], end=True)
        at = t.staging_offset + t.size
    if check:
        store.check()
    return staging


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class Segments:
    """An encoded store's segment files `needed`, in that order, each read once and held to its `seg/<s>.json`'s SHA-512,
    `ahead` of the one in use in a thread pool; `rows` asks for blocks in increasing order."""

    def __init__(self, root: Path, bps: int, needed: list[int], pool: ThreadPoolExecutor, ahead: int) -> None:
        self.root, self.bps, self.pool, self.ahead = root, bps, pool, ahead
        self.queue = deque(needed)
        self.pending: dict[int, Future] = {}
        self.held: dict[int, np.ndarray] = {}
        self.read = 0

    def _load(self, s: int) -> np.ndarray:
        meta = L.segment_path(self.root, s, "json")
        if not meta.exists():
            raise SystemExit(f"segment {s} of {self.root} is not done: no {meta.name}")
        data = np.fromfile(L.segment_path(self.root, s, "c"), dtype=np.uint8)
        if hashlib.sha512(data).hexdigest() != json.loads(meta.read_text())["sha512"]:
            raise SystemExit(f"segment {s}'s rows are not the SHA-512 its {meta.name} records")
        return data

    def segment(self, s: int) -> np.ndarray:
        while self.queue and len(self.pending) < self.ahead + 1:
            k = self.queue.popleft()
            self.pending[k] = self.pool.submit(self._load, k)
        if s not in self.held:
            if s not in self.pending:
                raise ValueError(f"segment {s} isn't next: the rows are asked for in block order")
            self.held[s] = self.pending.pop(s).result()
            self.read += 1
        return self.held[s]

    def rows(self, blocks: range) -> torch.Tensor:
        """The rows of `blocks` ([n, 2056] uint8) on the device; the segments before the one before the last they touch
        let go (the next tensor's rows in whole units can start a few blocks before these end)."""
        out = torch.empty(len(blocks), L.ROW, dtype=torch.uint8, device=DEVICE)
        last = (blocks.stop - 1) // self.bps
        for s in range(blocks.start // self.bps, last + 1):
            data = self.segment(s)
            lo, hi = max(blocks.start, s * self.bps), min(blocks.stop, (s + 1) * self.bps)
            part = data[(lo - s * self.bps) * L.ROW:(hi - s * self.bps) * L.ROW]
            if len(part) != (hi - lo) * L.ROW:
                raise ValueError(f"segment {s} ends before block {hi}")
            out[lo - blocks.start:hi - blocks.start] = torch.from_numpy(part.reshape(hi - lo, L.ROW)).to(DEVICE)
        for s in [s for s in self.held if s < last - 1]:
            del self.held[s]
        return out


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
