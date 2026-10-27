"""J10-R's noise and receipts against v1's on one GPU, interleaved (each repetition times both forms, their order
alternating), at Llama-3.1-8B's linears (qkv, o, gate_up, down) at a ``--slot``-row slot over ``--layers`` layers:

* noise: noise.cu's blocks a call's units read. v1 derives each unit's own (`h1_noise`, U × (32 RB + 32 CB) blocks of
  its seed_j), J10-R its call's one stream (`h1_noise_calls`, M RB + N CB blocks of seed_c), each hashed alone
  (``sink``, no stream written) and written; and J10-R's gather of each unit's rows and columns from its stream
  (receipts.py's `j10r_counters`), which the torch chain pays and the units kernel doesn't;
* receipts: h1u.cu's units kernel, each call's seeds and receipts (`Units.seeds` and `Units.receipts`, v1's `kh_unit`,
  J10-R's `kh_units_multi_j10r` of the one call), and a step's calls of one K in one launch (`Units.multi`);
* with ``--torch``, receipts.py's torch chain over noise.cu's noise (`receipts.call_receipts`), a call of each linear.

Each time is CUDA events around the work on torch's stream, the median over ``--reps`` after a warm-up, given per call
(``ms``), per unit (``ns_unit``) and per decode step of ``--layers`` layers (``step_ms``). Before timing, each linear's
J10-R units-kernel receipts are held to the torch chain's over hashlib's stream, word for word, and noise.cu's J10-R
stream to hashlib's. Writes ``--out`` (JSON) and prints a line a linear.

    research run --on vy-nebius-2 --pool research --mem-gb 48 --cwd source --source . --as LANE --env JOB=tests -- \
      bash benchmarks/pouw/served_h1/j10r_gpu.sh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import producer as P  # noqa: E402
import receipts as RC  # noqa: E402

from verity.core.protocols.pouw.circuit import h1tile as HT  # noqa: E402
from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402

#: Llama-3.1-8B's linears (name, n, K)
LLAMA = [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 28672, 4096), ("down", 4096, 14336)]
SALT, RA, RB = (hashlib.sha512(x).digest() for x in (b"salt", b"A", b"B"))
U0 = 1000


def _clock(fn: Callable[[], object]) -> float:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--so", required=True, help="h1u.cu's -hbc build (with kh_units_multi_j10r)")
    ap.add_argument("--cubin", required=True, help="noise.cu's cubin")
    ap.add_argument("--slot", type=int, default=64)
    ap.add_argument("--rows", type=int, default=20, help="served rows of the slot (the rest zero)")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--torch", action="store_true", help="also time receipts.py's torch chain, a call of each linear")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.slot % 32 or not 0 <= a.rows <= a.slot:
        ap.error(f"--slot is a multiple of 32 and --rows at most it, not {a.slot} and {a.rows}")
    dev = torch.device("cuda", torch.cuda.current_device())
    units, cubin = P.Units(a.so), RC.Cubin(a.cubin)
    if not units.j10r:
        ap.error(f"{a.so} has no J10-R units (kh_units_multi_j10r)")
    gen = torch.Generator(device=dev).manual_seed(20261024)
    rec_v1 = torch.from_numpy(P.seed_record(SALT, RA, RB)).to(dev)
    rec_j = torch.from_numpy(P.seed_record(SALT, RA, RB, j10r=True)).to(dev)
    out: dict = {"slot": a.slot, "rows": a.rows, "layers": a.layers, "reps": a.reps, "so": a.so, "cubin": a.cubin,
                 "device": torch.cuda.get_device_name(dev), "load": os.getloadavg(), "linears": {}}
    ok = True
    calls = {}
    for name, n, k in LLAMA:
        x = torch.zeros(a.slot, k, dtype=torch.int8, device=dev)
        x[:a.rows] = torch.randint(-127, 128, (a.rows, k), dtype=torch.int8, device=dev, generator=gen)
        calls[name] = (x, torch.randint(-127, 128, (n, k), dtype=torch.int8, device=dev, generator=gen))

    for name, n, k in LLAMA:
        x, b = calls[name]
        u = (a.slot // 32) * (n // 32)
        lay = HT.Layout(32, 32, k)
        nb_v1, nb_j = lay.blocks, RC.stream_blocks(a.slot, n, k)
        call = RC.Call(x, b)
        seed_c = SD.call_seed(SALT, U0, RA, RB)
        stream = RC.call_streams([seed_c], [nb_j], dev, cubin)[0]
        same_stream = bool(torch.equal(stream.cpu(), RC.call_streams([seed_c], [nb_j], torch.device("cpu"), None)[0]))
        sj = units.seeds(rec_j, U0, u, call=U0)
        got = units.receipts(x, b, sj, j10r=True).cpu().numpy().view(np.uint32)
        same_receipts = bool(np.array_equal(got, RC.call_receipts(call, SALT, U0, RA, RB, None, j10r=True)))
        ok = ok and same_stream and same_receipts

        sv1 = RC.seed_words(RC.unit_seeds(SALT, [U0 + t for t in range(u)], RA, RB)[0], dev)
        sc = RC.seed_words([seed_c], dev)
        starts = torch.tensor([0, nb_j], dtype=torch.int64, device=dev)
        full_v1 = torch.empty(u, nb_v1, 16, dtype=torch.int32, device=dev)
        sink_v1 = torch.empty(u, nb_v1, dtype=torch.int32, device=dev)
        full_j = torch.empty(nb_j, 16, dtype=torch.int32, device=dev)
        sink_j = torch.empty(nb_j, dtype=torch.int32, device=dev)
        us = call.units
        ri = torch.tensor([i for i, _j in us], device=dev)
        cj = torch.tensor([j for _i, j in us], device=dev)
        counters = RC.j10r_counters(ri, cj, a.slot, k)
        sv1_dev = units.seeds(rec_v1, U0, u)

        work: dict[str, Callable[[], object]] = {
            "noise_hash_v1": lambda: cubin.launch(sv1, nb_v1, sink_v1, sink=True),
            "noise_hash_j10r": lambda: cubin.launch_calls(sc, starts, sink_j, sink=True),
            "noise_write_v1": lambda: cubin.launch(sv1, nb_v1, full_v1),
            "noise_write_j10r": lambda: cubin.launch_calls(sc, starts, full_j),
            "noise_gather_j10r": lambda: stream[counters],
            "receipts_v1": lambda: units.receipts(x, b, units.seeds(rec_v1, U0, u)),
            "receipts_j10r": lambda: units.receipts(x, b, units.seeds(rec_j, U0, u, call=U0), j10r=True),
            "units_only_v1": lambda: units.receipts(x, b, sv1_dev),
            "units_only_j10r": lambda: units.receipts(x, b, sj, j10r=True),
        }
        if a.torch:
            work["torch_v1"] = lambda: RC.call_receipts(call, SALT, U0, RA, RB, cubin)
            work["torch_j10r"] = lambda: RC.call_receipts(call, SALT, U0, RA, RB, cubin, j10r=True)
        pairs = sorted({key.rsplit("_", 1)[0] for key in work})
        times: dict[str, list[float]] = {key: [] for key in work}
        for fn in work.values():
            fn()
        torch.cuda.synchronize()
        for rep in range(a.reps):
            for pair in pairs:
                forms = [f for f in ("v1", "j10r") if f"{pair}_{f}" in work]
                for f in (forms if rep % 2 == 0 else forms[::-1]):
                    times[f"{pair}_{f}"].append(_clock(work[f"{pair}_{f}"]))
        full_v1 = sink_v1 = None
        torch.cuda.empty_cache()
        row = {"n": n, "k": k, "units": u, "blocks_v1": u * nb_v1, "blocks_j10r": nb_j,
               "same_stream": same_stream, "same_receipts": same_receipts, "times": {}}
        for key, ts in times.items():
            ms = statistics.median(ts)
            row["times"][key] = {"ms": ms, "ns_unit": 1e6 * ms / u, "step_ms": ms * a.layers, "all": ts}
        out["linears"][name] = row
        t = row["times"]
        print(f"j10r_time: {name} ({n} x {k}, {u} units; noise blocks v1 {u * nb_v1} / J10-R {nb_j}, same "
              f"{same_stream and same_receipts}): noise hashed {t['noise_hash_v1']['ms']:.3f} / "
              f"{t['noise_hash_j10r']['ms']:.3f} ms, written {t['noise_write_v1']['ms']:.3f} / "
              f"{t['noise_write_j10r']['ms']:.3f} (+ gather {t['noise_gather_j10r']['ms']:.3f}); receipts "
              f"{t['receipts_v1']['ms']:.3f} / {t['receipts_j10r']['ms']:.3f} ms ("
              f"{t['receipts_v1']['ns_unit']:.0f} / {t['receipts_j10r']['ns_unit']:.0f} ns a unit)"
              + (f"; torch {t['torch_v1']['ms']:.1f} / {t['torch_j10r']['ms']:.1f} ms" if a.torch else ""), flush=True)

    groups = {k: [name for name, _n, kk in LLAMA if kk == k] for k in sorted({k for _name, _n, k in LLAMA})}
    batch: dict = {}
    for k, names in groups.items():
        spans, at = [], U0
        for name in names:
            u = (a.slot // 32) * (calls[name][1].shape[0] // 32)
            spans.append((at, u))
            at += u

        def step(j10r: bool, names=names, spans=spans) -> None:
            rec = rec_j if j10r else rec_v1
            seeds = [units.seeds(rec, u0, u, call=u0 if j10r else None) for u0, u in spans]
            units.multi([(calls[nm][0], calls[nm][1], s) for nm, s in zip(names, seeds)], None, j10r)

        step(False), step(True)
        torch.cuda.synchronize()
        ts: dict[str, list[float]] = {"v1": [], "j10r": []}
        for rep in range(a.reps):
            for f in (("v1", "j10r") if rep % 2 == 0 else ("j10r", "v1")):
                ts[f].append(_clock(lambda f=f: step(f == "j10r")))
        total = sum(u for _u0, u in spans)
        batch[str(k)] = {"calls": names, "units": total,
                         **{f: {"ms": statistics.median(v), "ns_unit": 1e6 * statistics.median(v) / total,
                                "step_ms": statistics.median(v) * a.layers, "all": v} for f, v in ts.items()}}
        print(f"j10r_time: K = {k} ({'+'.join(names)}, {total} units) in one launch: v1 {batch[str(k)]['v1']['ms']:.3f} "
              f"/ J10-R {batch[str(k)]['j10r']['ms']:.3f} ms", flush=True)
    out["batch"] = batch
    lin = out["linears"]
    out["step"] = {key: sum(lin[nm]["times"][key]["step_ms"] for nm in lin) for key in lin["qkv"]["times"]}
    out["step"]["batch_v1"] = sum(v["v1"]["step_ms"] for v in batch.values())
    out["step"]["batch_j10r"] = sum(v["j10r"]["step_ms"] for v in batch.values())
    out["ok"] = ok
    out["load_after"] = os.getloadavg()
    st = out["step"]
    print(f"j10r_time: a decode step of {a.layers} layers at {a.rows}/{a.slot} rows (v1 / J10-R): noise hashed "
          f"{st['noise_hash_v1']:.1f} / {st['noise_hash_j10r']:.1f} ms, written {st['noise_write_v1']:.1f} / "
          f"{st['noise_write_j10r']:.1f} (+ gather {st['noise_gather_j10r']:.1f}); receipts a call at a time "
          f"{st['receipts_v1']:.1f} / {st['receipts_j10r']:.1f}, batched {st['batch_v1']:.1f} / {st['batch_j10r']:.1f}; "
          f"words {'same' if ok else 'MISMATCH'}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
