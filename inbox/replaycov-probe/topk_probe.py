"""A vLLM general plugin for a localization replay of the DSA indexer's prefill top-k.

It wraps ``vllm._custom_ops.top_k_per_row_prefill`` and, per call, records three digests of the rows it wrote: the indexer
logits summed over each row's key range (row-weighted), the selected set (order-free) and its raw order. With
TOPK_PROBE=sort it then rewrites each row's indices in ascending order, padding (-1) last, before attention reads them; any
other value only records. TOPK_PROBE_REAL=module:function makes the probe call that function (same signature, writing the
rows in place) instead of vLLM's kernel, so a replacement top-k is recorded whatever order the plugins load in.
Digests go to TOPK_PROBE_OUT/dev<cuda device>.jsonl, flushed every 256 calls and at exit.
"""
from __future__ import annotations

import atexit
import importlib
import json
import os

FLUSH = 256
BIG = 2**31 - 1


def register() -> None:
    mode, out = os.environ.get("TOPK_PROBE"), os.environ.get("TOPK_PROBE_OUT")
    if not mode or not out:
        return
    import torch
    from vllm import _custom_ops as ops

    if getattr(ops.top_k_per_row_prefill, "_probe", False):
        return
    if spec := os.environ.get("TOPK_PROBE_REAL"):
        mod, fn = spec.split(":")
        real = getattr(importlib.import_module(mod), fn)
    else:
        real = ops.top_k_per_row_prefill
    os.makedirs(out, exist_ok=True)
    pending: list = []
    calls = [0]

    def flush() -> None:
        if not pending:
            return
        ints = torch.stack([p[1] for p in pending]).cpu().tolist()
        flts = torch.stack([p[2] for p in pending]).cpu().tolist()
        with open(os.path.join(out, f"dev{torch.cuda.current_device()}.jsonl"), "a") as f:
            for (meta, _, _), i, x in zip(pending, ints, flts):
                f.write(json.dumps({**meta, "set": i[0], "order": i[1], "logits": x}) + "\n")
        pending.clear()

    def wrapped(logits, ks, ke, topk, num_rows, s0, s1, k):
        real(logits, ks, ke, topk, num_rows, s0, s1, k)
        rows = topk[:num_rows, :k]
        dev = rows.device
        rw = torch.arange(1, rows.shape[0] + 1, device=dev, dtype=torch.int64)[:, None]
        pos = torch.arange(1, rows.shape[1] + 1, device=dev, dtype=torch.int64)[None, :]
        r64 = rows.to(torch.int64) + 1
        order = (r64 * pos * rw).sum()
        srt = torch.where(rows >= 0, rows, BIG).sort(dim=-1).values
        s64 = torch.where(srt == BIG, 0, srt.to(torch.int64) + 1)
        setd = (s64 * s64 * rw).sum() + s64.sum() * 1000003
        cols = torch.arange(logits.shape[1], device=dev)[None, :]
        lg = torch.zeros((), dtype=torch.float64, device=dev)
        for r0 in range(0, num_rows, 1024):
            r1 = min(num_rows, r0 + 1024)
            mask = (cols >= ks[r0:r1, None]) & (cols < ke[r0:r1, None])
            lg += (torch.where(mask, logits[r0:r1], 0.0).sum(dim=1).double() * rw[r0:r1, 0].double()).sum()
        pending.append(({"call": calls[0], "rows": int(num_rows), "cols": int(logits.shape[1]), "k": int(k)},
                        torch.stack([setd, order]), lg))
        calls[0] += 1
        if mode == "sort":
            rows.copy_(torch.where(srt == BIG, -1, srt).to(rows.dtype))
        if len(pending) >= FLUSH:
            flush()

    wrapped._probe = True
    ops.top_k_per_row_prefill = wrapped
    atexit.register(flush)
