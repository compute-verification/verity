"""sha256 of every live parameter and buffer of a served model, as raw contiguous bytes, by `named_parameters()` name.

A tensor's bytes go to the host a chunk at a time and into its hash (the digest of the whole byte string, never a whole host copy),
and tensors hash on a thread pool: `hashlib` and the device copy release the GIL.  A rank's ~94 GB of GLM-5.3 parameters then holds
`workers x chunk` bytes of host memory and hashes on `workers` cores.
"""
from __future__ import annotations

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

#: bytes of one tensor copied to the host at a time
CHUNK = 1 << 28

#: tensors hashed at once
WORKERS = 8


def _bytes_of(p):
    import torch
    flat = p.detach().contiguous().reshape(-1)
    return flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat


def _digest(p, chunk: int) -> tuple[str, int]:
    raw = _bytes_of(p)
    h = hashlib.sha256()
    for lo in range(0, raw.numel(), chunk):
        h.update(raw[lo:lo + chunk].cpu().numpy())
    return h.hexdigest(), int(raw.numel())


def param_digests(model, *, include_buffers: bool = True, raw_for: "re.Pattern[str] | None" = None, chunk: int = CHUNK,
                  workers: int = WORKERS) -> dict[str, Any]:
    """{params: {name: {digest, dtype, shape, bytes} | {..., alias_of}}, raw: {name: uint16 words}, n, wall_s}: a tensor reached under
    two names is hashed once and its later name carries `alias_of`; a bf16 tensor whose name matches `raw_for` also keeps its words."""
    import torch
    t0 = time.perf_counter()
    items = list(model.named_parameters())
    if include_buffers:
        items += list(model.named_buffers())
    first: dict[int, str] = {}
    order: list[tuple[str, Any]] = []
    for name, p in items:
        if not isinstance(p, torch.Tensor) or p.numel() == 0:
            continue
        order.append((name, p))
        first.setdefault(id(p), name)
    owners = [(name, p) for name, p in order if first[id(p)] == name]
    with ThreadPoolExecutor(max(1, min(workers, len(owners) or 1))) as ex:
        digests = dict(zip((name for name, _p in owners), ex.map(lambda np_: _digest(np_[1], chunk), owners)))
    out: dict[str, dict[str, Any]] = {}
    raws: dict[str, np.ndarray] = {}
    for name, p in order:
        own = first[id(p)]
        if own != name:
            out[name] = dict(out[own], alias_of=own)
            if own in raws:
                raws[name] = raws[own]
            continue
        sha, n = digests[name]
        out[name] = {"digest": sha, "dtype": str(p.dtype).replace("torch.", ""), "shape": list(p.shape), "bytes": n}
        if raw_for is not None and raw_for.search(name) and p.dtype == torch.bfloat16:
            raws[name] = np.ascontiguousarray(_bytes_of(p).cpu().numpy()).view(np.uint16).reshape(list(p.shape))
    return {"params": out, "raw": raws, "n": len(out), "wall_s": round(time.perf_counter() - t0, 3)}
