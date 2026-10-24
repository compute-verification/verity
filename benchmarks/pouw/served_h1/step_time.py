"""The receipt producer's time a decode step (`producer.Receipts` on h1u.cu's units kernel, `producer.Units`) with dead
tiles computed (v1's unit) and skipped (``skip_dead``): Llama-3.1-8B's linears (qkv, o, gate_up, down) over ``--layers``
layers at a ``--slot``-row slot holding ``--rows`` served rows. ``gpu_ms`` is the step's seeds and receipts on the device
(CUDA events around each call's `Units.seeds` and `Units.receipts`), ``wall_ms`` the step's `Receipts` calls as `Served`
makes them (operands to the device, receipts back to the host), each the median over ``--reps`` steps after one warm-up
step. Before anything is timed, each linear's skipped receipts are held to the computed ones: a row block holding a served
row word for word, a dead one zero words. Writes ``--out`` (JSON) and prints one line.

    producer_gpu.sh runs it on each build under its lease (``STEP_ROWS``), into ``$OUT/step-<build>.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import producer as P  # noqa: E402

#: Llama-3.1-8B's linears (name, n, K)
LLAMA = [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 28672, 4096), ("down", 4096, 14336)]
SALT, RA, RB = (hashlib.sha512(x).digest() for x in (b"salt", b"A", b"B"))


class _At(P.Receipts):
    """The producer with its calls' first units laid out in step order, without a window Program."""

    def __init__(self, starts: dict[int, tuple[int, int]], *args, **kw):
        super().__init__(*args, **kw)
        self.starts = starts

    def start(self, call):
        return self.starts[call.index]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--so", required=True, help="h1u.cu's -hbc build")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--slot", type=int, default=64)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.slot % 32 or not 0 <= a.rows <= a.slot:
        ap.error(f"--slot is a multiple of 32 and --rows at most it, not {a.slot} and {a.rows}")
    units, dev = P.Units(a.so), torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(20261024)
    ws = [torch.randint(-127, 128, (n, k), dtype=torch.int8, device=dev, generator=gen) for _name, n, k in LLAMA]
    codes = {k: torch.randint(-127, 128, (a.rows, k), dtype=torch.int8, device=dev, generator=gen).cpu().numpy()
             for k in {k for _name, _n, k in LLAMA}}
    calls, starts, u0 = [], {}, 0
    for layer in range(a.layers):
        for i, (_name, n, k) in enumerate(LLAMA):
            c = SimpleNamespace(index=len(calls), step=0, layer=i, module=LLAMA[i][0], m=a.rows, k=k, n=n)
            starts[c.index] = (u0, (a.slot // 32) * (n // 32))
            u0 += starts[c.index][1]
            calls.append(c)

    def made(skip: bool) -> _At:
        return _At(starts, SimpleNamespace(m=a.slot), SimpleNamespace(salt=SALT), units=units,
                   weight=lambda c: ws[c.layer], roots=lambda c: (RA, RB), skip_dead=skip)

    held = {}
    for c in calls[:len(LLAMA)]:
        full, skipped = made(False)(c, codes[c.k], None, None, None), made(True)(c, codes[c.k], None, None, None)
        live = np.repeat(np.arange(a.slot // 32) * 32 < a.rows, c.n // 32)
        held[c.module] = {"units": len(live), "live": int(live.sum()),
                          "ok": bool(np.array_equal(skipped[live], full[live]) and not skipped[~live].any())}
    if not all(v["ok"] for v in held.values()):
        print(f"step_time: SKIP_MISMATCH {held}")
        return 1

    record = torch.from_numpy(P.seed_record(SALT, RA, RB)).to(dev)
    a_dev = {k: torch.zeros(a.slot, k, dtype=torch.int8, device=dev) for k in codes}
    for k, x in codes.items():
        a_dev[k][:a.rows] = torch.from_numpy(x).to(dev)
    live_dev = (torch.arange(a.slot // 32, device=dev) * 32 < a.rows).to(torch.uint8)

    def gpu_step(skip: bool) -> tuple[float, dict[str, float]]:
        ev = []
        for c in calls:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            seeds = units.seeds(record, starts[c.index][0], starts[c.index][1])
            units.receipts(a_dev[c.k], ws[c.layer], seeds, live_dev if skip else None)
            e.record()
            ev.append((c.module, s, e))
        torch.cuda.synchronize()
        per: dict[str, float] = {}
        for module, s, e in ev:
            per[module] = per.get(module, 0.0) + s.elapsed_time(e)
        return sum(per.values()), per

    def wall_step(skip: bool) -> float:
        p = made(skip)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for c in calls:
            p(c, codes[c.k], None, None, None)
        return 1e3 * (time.perf_counter() - t)

    out: dict = {"rows": a.rows, "slot": a.slot, "layers": a.layers, "reps": a.reps, "calls": len(calls),
                 "units": u0, "live_units": sum(v["live"] for v in held.values()) * a.layers,
                 "device": torch.cuda.get_device_name(dev), "so": a.so, "held": held, "load": os.getloadavg()}
    for skip in (False, True):
        gpu_step(skip)
        wall_step(skip)
        g = [gpu_step(skip) for _ in range(a.reps)]
        w = [wall_step(skip) for _ in range(a.reps)]
        out["skip" if skip else "full"] = {
            "gpu_ms": statistics.median(x for x, _ in g), "gpu_ms_all": [x for x, _ in g],
            "gpu_ms_by_linear": {m: statistics.median(p[m] for _, p in g) for m in g[0][1]},
            "wall_ms": statistics.median(w), "wall_ms_all": w}
    f, s = out["full"], out["skip"]
    out["gpu_speedup"] = f["gpu_ms"] / s["gpu_ms"] if s["gpu_ms"] else None
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"step_time: {a.rows}/{a.slot} rows, {len(calls)} calls, {u0} units: gpu {f['gpu_ms']:.1f} -> "
          f"{s['gpu_ms']:.1f} ms a step (x{out['gpu_speedup']:.2f}), wall {f['wall_ms']:.1f} -> {s['wall_ms']:.1f} ms; "
          f"held {held}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
