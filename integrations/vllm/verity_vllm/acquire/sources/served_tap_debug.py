"""DEBUG ONLY, not for merge (Route 1's debug row): do the served words of a call-boundary body equal the words the host evaluates?

With VERITY_SERVED_TAP=<dir>, every body of the call-boundary source whose module matches VERITY_SERVED_TAP_MODULES (a regex, default
`logits_processor$`) captures every tensor its forward computes (a TorchFunctionMode from a pre-hook after the source's to a forward hook),
and each target's host words are compared with the captured tensor of its size and word width that agrees with them most, word for word.
One JSON line per (engine step, target) goes to <dir>/served_tap.jsonl.  The warm-up's placeholder run is skipped."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FIRST = 8
#: a target's fx node (its `_<n>` dropped) -> the torch functions that compute it in the served forward
OPS = {"triton_launch": ("linear", "matmul", "mm", "addmm"), "div": ("__truediv__", "div", "true_divide"), "tanh": ("tanh",),
       "mul": ("__mul__", "mul")}


def ordinal(w: np.ndarray) -> np.ndarray:
    """Sign-magnitude words as integers ordered like the values they encode, so a difference counts ulps."""
    bits = w.dtype.itemsize * 8
    x = w.astype(np.int64)
    mag = x & ((1 << (bits - 1)) - 1)
    return np.where(((x >> (bits - 1)) & 1) == 1, -mag, mag)


def compare(host: np.ndarray, served: Sequence[tuple[int, str, np.ndarray]]) -> dict[str, Any]:
    h = np.ascontiguousarray(host).reshape(-1)
    best = None
    for i, fn, w in served:
        if w.size != h.size or w.dtype.itemsize != h.dtype.itemsize:
            continue
        eq = int(np.count_nonzero(w == h.view(w.dtype)))
        if best is None or eq > best[0]:
            best = (eq, i, fn, w)
    if best is None:
        return {"numel": int(h.size), "served": None, "captured": [[i, fn, int(w.size), w.dtype.itemsize] for i, fn, w in served]}
    eq, i, fn, w = best
    hv = h.view(w.dtype)
    bad = np.flatnonzero(w != hv)
    ulp = np.abs(ordinal(w[bad]) - ordinal(hv[bad]))
    k, n = np.unique(np.minimum(ulp, 4), return_counts=True)
    return {"numel": int(h.size), "equal": eq, "served": [i, fn], "max_ulp": int(ulp.max()) if ulp.size else 0,
            "ulp_hist": {str(int(a)): int(b) for a, b in zip(k, n)},
            "first": [[int(j), f"{int(hv[j]):#x}", f"{int(w[j]):#x}"] for j in bad[:FIRST]]}


def install(src: Any, mods: dict[str, Any]) -> None:
    import torch
    from torch.overrides import TorchFunctionMode

    from verity_vllm.acquire.sources.call_boundary_source import words

    out = Path(os.environ["VERITY_SERVED_TAP"])
    out.mkdir(parents=True, exist_ok=True)
    log = out / "served_tap.jsonl"
    want = re.compile(os.environ.get("VERITY_SERVED_TAP_MODULES") or r"logits_processor$")

    class Capture(TorchFunctionMode):
        def __init__(self) -> None:
            super().__init__()
            self.got: list[tuple[str, Any]] = []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            r = func(*args, **(kwargs or {}))
            if isinstance(r, torch.Tensor):     # its words now: a later in-place op of the forward does not reach them
                self.got.append((getattr(func, "__name__", repr(func)), r.detach().clone()))
            return r

    for op in src.modules:
        if not want.search(op):
            continue
        live: dict[str, Any] = {}

        def pre(_m: Any, _args: tuple, _kw: dict, op: str = op, live: dict = live) -> None:
            job = src.c._cur
            body = src.plan.bodies.get((int(job.step), op)) if job is not None else None
            if body is None or src.placeholder:
                return None
            mode = Capture()
            mode.__enter__()
            live["mode"], live["body"] = mode, body
            return None

        def post(_m: Any, _args: tuple, _kw: dict, output: Any, op: str = op, live: dict = live) -> None:
            mode, body = live.pop("mode", None), live.pop("body", None)
            if mode is None:
                return None
            mode.__exit__(None, None, None)
            got = []
            for i, (fn, t) in enumerate([*mode.got, *([("<output>", output)] if isinstance(output, torch.Tensor) else [])]):
                try:
                    got.append((i, fn, words(torch, t)))
                except Exception as e:  # noqa: BLE001 -- a capture that has no words (a 0-d bool) is listed, not compared
                    got.append((i, f"{fn}!{type(e).__name__}", np.zeros(0, np.uint8)))
            host = src._values.get((op, body.engine_step)) or {}
            with log.open("a") as f:
                for t in body.targets:
                    rec: dict[str, Any] = {"step": body.engine_step, "module": op, "target": t.name, "dtype": t.dtype}
                    if t.name not in host:
                        rec["host"] = None
                    else:
                        rec.update(compare(host[t.name], got))
                        fns = OPS.get(re.sub(r"_\d+$", "", t.name.split("/")[-2]), ())
                        mine = [g for g in got if g[1] in fns and g[2].size == host[t.name].size][:1]
                        rec["by_op"] = compare(host[t.name], mine) if mine else None
                    f.write(json.dumps(rec) + "\n")
            return None

        src._handles.append(mods[op].register_forward_pre_hook(pre, with_kwargs=True))
        src._handles.append(mods[op].register_forward_hook(post, with_kwargs=True))
    print(f"[delta] DEBUG served tap on {[m for m in src.modules if want.search(m)]} -> {log}", flush=True)
