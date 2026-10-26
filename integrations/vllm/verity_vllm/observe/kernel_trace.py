"""The device kernels of a Kineto Chrome trace (`torch.profiler.profile(...).export_chrome_trace`): its kernels, memcpys and
memsets in launch order (`kernels_of_trace_events`), and their summary by name (`summarize_kernels`).  The kernel census reads
one trace per engine step (`properties.census`), and each TP rank one over its whole run
(`pipeline.tp.rank_worker.TP2CaptureWorkerExtension.tp2_profiler_stop`).  Torch-free."""

from __future__ import annotations

import gzip
import json
from collections import OrderedDict
from pathlib import Path

_KERNEL_CATS = {"kernel", "Kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime_kernel", "gpu_op"}


def kernels_of_trace_events(evs: list[dict]) -> list[dict]:
    """Device-side events (`ph == "X"`) of a Kineto Chrome trace: kernels, memcpys, memsets; in launch order."""
    ks = [e for e in evs if e.get("ph") == "X" and e.get("cat") in _KERNEL_CATS]
    ks.sort(key=lambda e: (e.get("ts", 0), e.get("name", "")))
    return ks


def summarize_kernels(ks: list[dict]) -> "OrderedDict[str, dict]":
    """name -> {count, us, grid, block, first_index} (first launch's grid/block), by first appearance."""
    out: "OrderedDict[str, dict]" = OrderedDict()
    for i, e in enumerate(ks):
        n = e.get("name", "")
        d = out.get(n)
        if d is None:
            a = e.get("args", {}) or {}
            d = out[n] = {"count": 0, "us": 0.0, "grid": a.get("grid"), "block": a.get("block"), "first_index": i,
                          "cat": e.get("cat")}
        d["count"] += 1
        d["us"] += float(e.get("dur", 0.0) or 0.0)
    for d in out.values():
        d["us"] = round(d["us"], 1)
    return out


def load_trace(path: Path) -> list[dict]:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        doc = json.load(f)
    return doc["traceEvents"] if isinstance(doc, dict) else doc
