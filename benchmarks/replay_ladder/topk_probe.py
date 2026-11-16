"""Where two replays part inside a DSA model's indexer: a probe of its prefill top-k in each engine process (`apply`, which
ladder.py calls from the ladder's settings) and the call-by-call comparison of two replays' records (`compare`).

    topk_probe.py compare A B     # A and B: two replays' probe directories (replay-I/topk)

The probe wraps ``vllm._custom_ops.top_k_per_row_prefill``, which writes, for each of a prefill's rows, the indices of its
``k`` largest indexer logits over the row's key range. Per call it records three checksums of what the call read and wrote:
the logits (each row's sum over its key range, weighted by its row number, in float64), the selected set (order-free) and
the order the indices are listed in. In ``sort`` mode it then rewrites each row's indices in ascending order, padding (-1)
last, before attention reads them; ``record`` only records. With ``real`` (``module:function``, the kernel's signature,
writing the rows in place) the probe calls that function instead of vLLM's kernel, so a replacement top-k is the one
recorded. Records go to ``OUT/dev<device>.jsonl``, flushed every `FLUSH` calls and at exit. Decode's top-k isn't probed.
"""
from __future__ import annotations

import argparse
import atexit
import importlib
import json
import sys
from pathlib import Path
from typing import Any

FLUSH = 256
BIG = 2**31 - 1
MODES = ("record", "sort")
CHECKS = ("logits", "set", "order")


def apply(*, mode: str | None, out: str | None, real: str | None = None) -> None:
    """Probe this process's prefill top-k (``mode`` one of `MODES`), recording to the directory ``out``."""
    if not mode or not out:
        return
    if mode not in MODES:
        raise ValueError(f"the top-k probe's mode is one of {', '.join(MODES)}, not {mode}")
    from vllm import _custom_ops as ops

    if getattr(ops.top_k_per_row_prefill, "topk_probe", False):
        return
    if real:
        mod, fn = real.split(":")
        kernel = getattr(importlib.import_module(mod), fn)
    else:
        kernel = ops.top_k_per_row_prefill
    ops.top_k_per_row_prefill = probe(kernel, mode, Path(out))


def probe(kernel: Any, mode: str, out: Path) -> Any:
    """``kernel`` wrapped to record each call's checksums under ``out`` (and, in ``sort`` mode, sort its rows)."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    pending: list = []
    calls = [0]

    def flush() -> None:
        if not pending:
            return
        ints = torch.stack([p[1] for p in pending]).cpu().tolist()
        flts = torch.stack([p[2] for p in pending]).cpu().tolist()
        by: dict[int, list[str]] = {}
        for (meta, _, _), i, x in zip(pending, ints, flts):
            dev = meta.pop("device")
            by.setdefault(dev, []).append(json.dumps({**meta, "set": i[0], "order": i[1], "logits": x}))
        for dev, lines in by.items():
            with open(out / f"dev{dev}.jsonl", "a") as f:
                f.write("".join(x + "\n" for x in lines))
        pending.clear()

    def wrapped(logits, ks, ke, topk, num_rows, s0, s1, k):
        kernel(logits, ks, ke, topk, num_rows, s0, s1, k)
        rows = topk[:num_rows, :k]
        dev = rows.device
        rw = torch.arange(1, rows.shape[0] + 1, device=dev, dtype=torch.int64)[:, None]
        pos = torch.arange(1, rows.shape[1] + 1, device=dev, dtype=torch.int64)[None, :]
        order = ((rows.to(torch.int64) + 1) * pos * rw).sum()
        srt = torch.where(rows >= 0, rows, BIG).sort(dim=-1).values
        s64 = torch.where(srt == BIG, 0, srt.to(torch.int64) + 1)
        setd = (s64 * s64 * rw).sum() + s64.sum() * 1000003
        cols = torch.arange(logits.shape[1], device=dev)[None, :]
        lg = torch.zeros((), dtype=torch.float64, device=dev)
        for r0 in range(0, num_rows, 1024):
            r1 = min(num_rows, r0 + 1024)
            mask = (cols >= ks[r0:r1, None]) & (cols < ke[r0:r1, None])
            lg += (torch.where(mask, logits[r0:r1], 0.0).sum(dim=1).double() * rw[r0:r1, 0].double()).sum()
        meta = {"call": calls[0], "rows": int(num_rows), "cols": int(logits.shape[1]), "k": int(k),
                "device": dev.index if dev.index is not None else 0}
        pending.append((meta, torch.stack([setd, order]), lg))
        calls[0] += 1
        if mode == "sort":
            rows.copy_(torch.where(srt == BIG, -1, srt).to(rows.dtype))
        if len(pending) >= FLUSH:
            flush()

    wrapped.topk_probe = True  # type: ignore[attr-defined]
    wrapped.flush = flush  # type: ignore[attr-defined]
    atexit.register(flush)
    return wrapped


def records(d: Path) -> dict[str, list[dict]]:
    return {f.name: [json.loads(x) for x in f.read_text().splitlines() if x.strip()] for f in sorted(d.glob("dev*.jsonl"))}


def compare(a: Path, b: Path) -> dict:
    """Per device, the calls each replay made, the first call whose shape differs, and for each checksum the first call
    (before any shape differs) where the two replays' differ, with how many differ and replay A's record of that call."""
    xs_all, ys_all = records(a), records(b)
    out = {}
    for dev in sorted(set(xs_all) | set(ys_all)):
        xs, ys = xs_all.get(dev, []), ys_all.get(dev, [])
        n = min(len(xs), len(ys))
        shape = next((i for i in range(n)
                      if (xs[i]["rows"], xs[i]["cols"], xs[i]["k"]) != (ys[i]["rows"], ys[i]["cols"], ys[i]["k"])), None)
        m = shape if shape is not None else n
        row: dict[str, Any] = {"calls": [len(xs), len(ys)], "first_shape_mismatch": shape}
        for key in CHECKS:
            diff = [i for i in range(m) if xs[i][key] != ys[i][key]]
            row[key] = {"differing": len(diff), "first": diff[0] if diff else None, "first_call": xs[diff[0]] if diff else None}
        row["selecting_calls"] = sum(1 for i in range(m) if xs[i]["cols"] > xs[i]["k"])
        out[dev] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    a = ap.parse_args(argv)
    print(json.dumps(compare(a.a, a.b), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
