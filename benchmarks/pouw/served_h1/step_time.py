"""The receipt producer's time a decode step (`producer.Receipts` on h1u.cu's units kernel, `producer.Units`) with dead
tiles computed (v1's unit) and skipped (``skip_dead``), each call on its own and a step's calls of one K in one launch
(`Receipts.prepare`): Llama-3.1-8B's linears (qkv, o, gate_up, down) over ``--layers`` layers at a ``--slot``-row slot
holding each of ``--rows`` served rows. Per call, ``gpu_ms`` is the step's seeds and receipts on the device (CUDA events
around each call's `Units.seeds` and `Units.receipts`) and ``wall_ms`` the step's `Receipts` calls as `Served` made them
before `prepare` (operands to the device, receipts back to the host, a call at a time). Batched, ``gpu_ms`` is each K's
seeds and `Units.multi` launch and ``wall_ms`` the step as `Served` makes it now: at its last call of K = 4096 `prepare`
of its qkv, o and gate_up calls then each one's ask, then the same for its down calls at K = 14336 (``prepare_ms`` the
two `prepare` calls of it), at each of ``--launches`` (`Receipts.launches`). Each is the median over ``--reps`` steps
after one warm-up step. Before anything is timed, each linear's skipped receipts are held to the computed ones (a row block
holding a served row word for word, a dead one zero words), and every call's batched receipts of a whole step, at each
``--launches``, to its own, word for word. Writes ``--out`` (JSON) and prints a line a row count.

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


def _rows(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--so", required=True, help="h1u.cu's -hbc build")
    ap.add_argument("--rows", type=_rows, default=[20, 64], help="served rows, comma-separated")
    ap.add_argument("--slot", type=int, default=64)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--launches", type=_rows, default=[1, 3, 4, 6], help="Receipts.launches, comma-separated")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.slot % 32 or not a.rows or not all(0 <= r <= a.slot for r in a.rows):
        ap.error(f"--slot is a multiple of 32 and each of --rows at most it, not {a.slot} and {a.rows}")
    units, dev = P.Units(a.so), torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(20261024)
    ws = [torch.randint(-127, 128, (n, k), dtype=torch.int8, device=dev, generator=gen) for _name, n, k in LLAMA]
    calls, starts, u0 = [], {}, 0
    for _layer in range(a.layers):
        for i, (name, n, k) in enumerate(LLAMA):
            c = SimpleNamespace(index=len(calls), step=0, layer=i, module=name, m=0, k=k, n=n)
            starts[c.index] = (u0, (a.slot // 32) * (n // 32))
            u0 += starts[c.index][1]
            calls.append(c)
    groups = {k: [c for c in calls if c.k == k] for k in sorted({c.k for c in calls})}
    record = torch.from_numpy(P.seed_record(SALT, RA, RB)).to(dev)
    out: dict = {"slot": a.slot, "layers": a.layers, "reps": a.reps, "calls": len(calls), "units": u0,
                 "groups": {str(k): len(g) for k, g in groups.items()}, "device": torch.cuda.get_device_name(dev),
                 "so": a.so, "rows": {}}

    def made(skip: bool, launches: int = 1) -> _At:
        return _At(starts, SimpleNamespace(m=a.slot), SimpleNamespace(salt=SALT), units=units,
                   weight=lambda c: ws[c.layer], roots=lambda c: (RA, RB), skip_dead=skip, launches=launches)

    def at_rows(rows: int) -> dict | None:
        codes = {k: torch.randint(-127, 128, (rows, k), dtype=torch.int8, device=dev, generator=gen).cpu().numpy()
                 for k in groups}
        held = {}
        for c in calls[:len(LLAMA)]:
            full, skipped = made(False)(c, codes[c.k], None, None, None), made(True)(c, codes[c.k], None, None, None)
            live = np.repeat(np.arange(a.slot // 32) * 32 < rows, c.n // 32)
            held[c.module] = {"units": len(live), "live": int(live.sum()),
                              "ok": bool(np.array_equal(skipped[live], full[live]) and not skipped[~live].any())}

        def batched(p: _At) -> list[np.ndarray]:
            got = []
            for g in groups.values():
                p.prepare([(c, codes[c.k], None, None, None) for c in g])
                got += [p(c, codes[c.k], None, None, None) for c in g]
            return got

        same = {}
        for skip in (False, True):
            alone = made(skip)
            want = {c.index: alone(c, codes[c.k], None, None, None) for c in calls}
            for launch in a.launches:
                p = made(skip, launch)
                got = dict(zip([c.index for g in groups.values() for c in g], batched(p)))
                same[f"{'skip' if skip else 'full'}@{launch}"] = bool(
                    sorted(got) == sorted(want) and not p._ready and all(np.array_equal(got[i], want[i]) for i in want))
                del got
            del want
        if not all(v["ok"] for v in held.values()) or not all(same.values()):
            print(f"step_time: MISMATCH at {rows} rows: skip {held}, batched {same}")
            out["rows"][str(rows)] = {"rows": rows, "held": held, "batched_same": same}
            return None

        a_dev = {k: torch.zeros(a.slot, k, dtype=torch.int8, device=dev) for k in codes}
        for k, x in codes.items():
            a_dev[k][:rows] = torch.from_numpy(x).to(dev)
        live_dev = (torch.arange(a.slot // 32, device=dev) * 32 < rows).to(torch.uint8)

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

        def gpu_batch(skip: bool) -> tuple[float, dict[str, float]]:
            ev = []
            for k, g in groups.items():
                n = [starts[c.index][1] for c in g]
                offs = np.cumsum([0] + n)
                seeds = torch.empty(int(offs[-1]), 16, dtype=torch.int64, device=dev)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for c, lo, hi in zip(g, offs[:-1], offs[1:]):
                    units.seeds(record, starts[c.index][0], int(hi - lo), out=seeds[lo:hi])
                units.multi([(a_dev[k], ws[c.layer], seeds[lo:hi]) for c, lo, hi in zip(g, offs[:-1], offs[1:])],
                            live_dev if skip else None)
                e.record()
                ev.append((str(k), s, e))
            torch.cuda.synchronize()
            per = {k: s.elapsed_time(e) for k, s, e in ev}
            return sum(per.values()), per

        def wall_step(skip: bool) -> float:
            p = made(skip)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for c in calls:
                p(c, codes[c.k], None, None, None)
            return 1e3 * (time.perf_counter() - t)

        def wall_batch(skip: bool, launch: int) -> tuple[float, float]:
            p, prep = made(skip, launch), 0.0
            torch.cuda.synchronize()
            t = time.perf_counter()
            for g in groups.values():
                t0 = time.perf_counter()
                p.prepare([(c, codes[c.k], None, None, None) for c in g])
                prep += time.perf_counter() - t0
                for c in g:
                    p(c, codes[c.k], None, None, None)
            return 1e3 * (time.perf_counter() - t), 1e3 * prep

        r: dict = {"rows": rows, "live_units": sum(v["live"] for v in held.values()) * a.layers, "held": held,
                   "batched_same": same, "load": os.getloadavg()}
        for skip in (False, True):
            for warm in (gpu_step, gpu_batch, wall_step):
                warm(skip)
            g = [gpu_step(skip) for _ in range(a.reps)]
            w = [wall_step(skip) for _ in range(a.reps)]
            gb = [gpu_batch(skip) for _ in range(a.reps)]
            walls = {}
            for launch in a.launches:
                wall_batch(skip, launch)
                wb = [wall_batch(skip, launch) for _ in range(a.reps)]
                walls[str(launch)] = {"wall_ms": statistics.median(x for x, _ in wb),
                                                     "wall_ms_all": [x for x, _ in wb],
                                                     "prepare_ms": statistics.median(x for _, x in wb)}
            r["skip" if skip else "full"] = {
                "gpu_ms": statistics.median(x for x, _ in g), "gpu_ms_all": [x for x, _ in g],
                "gpu_ms_by_linear": {m: statistics.median(p[m] for _, p in g) for m in g[0][1]},
                "wall_ms": statistics.median(w), "wall_ms_all": w,
                "batch": {"gpu_ms": statistics.median(x for x, _ in gb), "gpu_ms_all": [x for x, _ in gb],
                          "gpu_ms_by_k": {k: statistics.median(p[k] for _, p in gb) for k in gb[0][1]},
                          "wall_by_launches": walls}}
        r["load_after"] = os.getloadavg()
        out["rows"][str(rows)] = r
        f, s = r["full"], r["skip"]
        fw, sw = f["batch"]["wall_by_launches"], s["batch"]["wall_by_launches"]
        bw = ", ".join(f"{c}: {fw[c]['wall_ms']:.1f} / {sw[c]['wall_ms']:.1f}" for c in fw)
        print(f"step_time: {rows}/{a.slot} rows, {len(calls)} calls, {u0} units, load {r['load'][0]:.0f}: per call gpu "
              f"{f['gpu_ms']:.1f} / {s['gpu_ms']:.1f} ms (full / skip), wall {f['wall_ms']:.1f} / {s['wall_ms']:.1f}; "
              f"batched gpu {f['batch']['gpu_ms']:.1f} / {s['batch']['gpu_ms']:.1f}, wall by launches {{{bw}}}; "
              f"batched words same {same}")
        return r

    rc = 0
    for rows in a.rows:
        if at_rows(rows) is None:
            rc = 1
            break
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
