"""What the rows of a grouped Commit (`commit_group`: the Commits of several rows of one engine in one process) reuse from each other:
the pinned host pool the first row's warm-up registered, and the weights root, its tree and the live weight digests, which each row
would compute from the same engine.  Nothing a row's commitment is over is shared.  `commit` calls these where it would build or
compute them; outside a grouped Commit (`commit._HOT` without ``group``) each is the plain call.

A weights root or a set of live digests is reused only while the engine's weights are the tensors and words it was computed from
(`fingerprint`: each tensor's storage, layout and version counter, and the sum of its words, taken on its device).  The fingerprint
detects the engine changing its own weights between rows, not an adversary: the root a row commits is checked by its replay, which
opens the checkpoint of record against it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable
from typing import Any

from verity_vllm.engine import hooks

SUM_CHUNK = 1 << 24          # words per device-side sum: bounds its int64 temporaries


def _group(hot: dict[str, Any] | None) -> dict[str, Any] | None:
    return (hot or {}).get("group")


def _event(hot: dict[str, Any], what: str, **fields: Any) -> None:
    hot["group"]["events"].append({"job": hot.get("jobs"), "what": what, **fields})


def adopt(hot: dict[str, Any] | None, com: Any) -> Any:
    """``com``, a row's committer, on the group's pinned pool: the first row's committer gives the group its pool (and the patches live
    beside the engine before any committer installs, the baseline `commit_group` checks each row's cleanup against); a later row's
    takes that pool with every block free (``reset``), so its warm-up registers only what the earlier rows' did not."""
    g, pool = _group(hot), getattr(com, "pool", None)
    if g is None or pool is None:
        return com
    if g["pool"] is None:
        g["pool"], g["patches"] = pool, hooks.live()
        _event(hot, "pool.kept", mode=pool.mode, bytes_total=pool.bytes_total)
        return com
    shared = g["pool"]
    shared.reset()
    shared.allocs, shared.alloc_s = 0, 0.0
    com.pool = shared
    _event(hot, "pool.adopted", mode=shared.mode, bytes_total=shared.bytes_total, blocks=len(shared.all))
    print(f"[delta] GROUP: pinned pool of the group's first row adopted ({shared.bytes_total >> 20} MiB in {len(shared.all)} "
          f"blocks, {shared.mode}): this row's warm-up registers only what it lacks", flush=True)
    return com


def fingerprint(model: Any) -> str:
    """sha256 over the tensors `register_weights` and the live digests read (parameters, then buffers): each one's name, storage
    address, shape, strides, dtype, device and version counter (None for an inference tensor, which keeps none), and the sum of its
    32-bit words (bytes, when its size is not a multiple of four), summed on its device and read back once."""
    import torch

    meta: list[list[Any]] = []
    sums: list[Any] = []
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if not isinstance(p, torch.Tensor):
            continue
        try:
            version: int | None = p._version
        except RuntimeError:
            version = None
        meta.append([name, p.data_ptr(), list(p.shape), list(p.stride()), str(p.dtype), str(p.device), version])
        flat = p.detach().reshape(-1)
        words = flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat
        if words.numel() % 4 == 0:
            words = words.view(torch.int32)
        parts = [words[i:i + SUM_CHUNK].sum(dtype=torch.int64) for i in range(0, words.numel(), SUM_CHUNK)]
        sums.append(torch.stack(parts).sum() if parts else torch.zeros((), dtype=torch.int64, device=p.device))
    totals = torch.stack([s.to(sums[0].device) for s in sums]).tolist() if sums else []
    return hashlib.sha256(json.dumps([meta, totals], separators=(",", ":")).encode()).hexdigest()


def _fingerprint(hot: dict[str, Any], model: Any) -> str:
    """The row's fingerprint of ``model``, taken once per row (its weights registration and live digests come before any run)."""
    g = hot["group"]
    key = (hot.get("jobs"), id(model))
    if g.get("fp", (None, None))[0] != key:
        t0 = time.perf_counter()
        g["fp"] = (key, fingerprint(model))
        _event(hot, "weights.fingerprint", s=round(time.perf_counter() - t0, 3), digest=g["fp"][1][:16])
    return g["fp"][1]


def _reused(hot: dict[str, Any], what: str, key: str, cached: tuple[str, int, Any] | None) -> Any:
    if cached is not None and cached[0] == key:
        _event(hot, f"{what}.reused", from_job=cached[1])
        return cached
    _event(hot, f"{what}.computed", why="the group's first row" if cached is None else "the weights' fingerprint changed")
    return None


def register_weights(hot: dict[str, Any] | None, com: Any, mode: str) -> dict[str, Any]:
    """``com.register_weights()``, or in a grouped Commit the root and tree an earlier row registered, when the weights' fingerprint,
    the committer's class, chunk and geometry digest and the hashing ``mode`` (``--weights-hash``) are the earlier row's."""
    g = _group(hot)
    if g is None:
        return com.register_weights()
    geo, fp = getattr(com, "_geo_digest", b""), _fingerprint(hot, com.model)
    key = json.dumps([fp, type(com).__name__, getattr(com, "chunk", None), geo.hex() if geo else None, mode])
    hit = _reused(hot, "weights", key, g["weights"])
    if hit is None:
        w = com.register_weights()
        g["weights"] = (key, hot.get("jobs"), (copy.deepcopy(com.weights_tree), dict(w)))
        return w
    tree, w = hit[2]
    com.weights_tree = copy.deepcopy(tree)
    com.weights = {**w, "wall_s": 0.0, "reused": {"from_job": hit[1], "wall_s": w.get("wall_s"), "fingerprint": fp[:16]}}
    print(f"[delta] GROUP: weights root {w['root'][:16]} reused from row {hit[1]} of the group (same tensors and words: fingerprint)",
          flush=True)
    return com.weights


def live_digests(hot: dict[str, Any] | None, model: Any, digests: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
    """``digests(model)`` (the live weight digests the weights of record are checked against), or in a grouped Commit an earlier
    row's, when the weights' fingerprint is that row's."""
    g = _group(hot)
    if g is None:
        return digests(model)
    key = _fingerprint(hot, model)
    hit = _reused(hot, "live_digests", key, g["live"])
    if hit is None:
        d = digests(model)
        g["live"] = (key, hot.get("jobs"), copy.deepcopy(d))
        return d
    print(f"[delta] GROUP: live weight digests reused from row {hit[1]} of the group (same tensors and words: fingerprint)", flush=True)
    return {**copy.deepcopy(hit[2]), "reused": {"from_job": hit[1]}}
