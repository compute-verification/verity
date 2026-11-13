"""The store's load of a layer: its matrices from the P2-encoded store into plaintext load buffers on the GPU, by
memory-accounting's decode unchanged (`libp2v1.so` from `benchmarks/pous/p2_v1/`, built for sm_90a by `build.sh`, through
its own wrapper `p2_v1.gpu.Kernels`) under the store's own load, `p2_v1.load`: `EncodedRows`, the encoded rows of a range
of the store's blocks resident on the device, its screen (`EncodedRows.check`, raising `StoreError` with the flag bits
`WIDE` and `PADDING`) and `scatter`. They are defined there and read here by this directory's modules.

`decode_matrix(store, layer, tensor, out)` decodes one tensor's blocks (a matrix of `layout.MATRICES`: QKV is
[16512, 16384] s8 at the frozen shape) into `out`. `decode_layer(store, layer, staging)` is the fixture's `decode_layer`:
every tensor of the layer at its `staging_offset`, zero between them. Nothing is re-laid out: W holds each tensor's bytes
as the GEMM reads them. Both call `EncodedRows.check` before they return.

With a pass of PoUB attached (`store.sketch = sketch.Pass(...)`), `decode_rows` sketches the units
each chunk's blocks lie in from the same rows on the device before the decode reads them (`sketch.py`); the rows then
hold whole units.

`timed` is the drivers' one timer (`layer.py`, `full.py`, `bw_check.py`).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

try:
    from . import layout as L
except ImportError:
    import layout as L  # type: ignore[no-redef]

from p2_v1 import gpu  # noqa: F401
from p2_v1.load import P_BYTES, PADDING, WIDE, EncodedRows, StoreError, scatter  # noqa: F401


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
