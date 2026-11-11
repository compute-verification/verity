"""H1's call z tables and their registered ``zy@<c>`` values hashed apart and from one read of each record, on one GPU, held
equal and timed (not collected; `h1_zy_gpu.sh` runs it under a lease).

Each linear of a decoder layer is one call of ``--units`` z/y records of ``ZY_BYTES`` (`pouw_h1_records`), random, in the
order qkv, o, gate_up, down; the defaults are a 16-layer model's at T 8,192. Apart: the call table's z root
(`tables_device.device_hash` over `tables_device.record_rows`) and the value's ``b || c`` strings and tree levels
(`registered_device.device_hash`), as a served call is hashed with ``tables_device.FUSE`` off. Together: the value's
`registered_device.device_hash` with ``also`` the z table. For each call:

  equal    the z root, strings and levels of the two, byte for byte;
  wall     each's seconds from launch to every result on the host, ``--reps`` times, interleaved;
  kernels  each's device milliseconds by kernel name, from one more rep under torch.profiler (CUPTI).

``DIR/h1_zy_gpu.json`` holds them and a step's totals (each call times ``--layers``). The verdict line is
``h1_zy_gpu: PASS`` only when every call's are equal.

    python tests/protocols/h1_zy_gpu.py --out DIR [--units 24576,16384,131072,16384] [--layers 16] [--reps 5]
"""
from __future__ import annotations

import argparse
import functools
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from verity_vllm.protocols.pouw.h1 import records as ZR
from verity_vllm.protocols.pouw.h1 import registered_device as D
from verity_vllm.protocols.pouw.h1 import tables_device as TD

CALLS = ("qkv", "o", "gate_up", "down")
KEY = bytes(range(32))


def apart(records: Any, z: tuple[bytes, int], value: tuple[bytes, int]) -> tuple[bytes, Any]:
    pending = TD.device_hash([(*z, *TD.record_rows(records))], KEY, ("z",))
    hashing = D.device_hash(records.view(torch.uint16), KEY, value[1], None, value[0])
    return pending.roots()[0], hashing.result()


def together(records: Any, z: tuple[bytes, int], value: tuple[bytes, int]) -> tuple[bytes, Any]:
    hashing = D.device_hash(records.view(torch.uint16), KEY, value[1], None, value[0], also=z)
    return hashing.also(), hashing.result()


def wall(fn: Callable[[], Any]) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def kernels(fn: Callable[[], Any]) -> dict[str, float]:
    """Device milliseconds by kernel (and copy) name over one call of ``fn``."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    out: dict[str, float] = {}
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", None)
        us = e.self_cuda_time_total if us is None else us
        if us:
            out[e.key] = out.get(e.key, 0.0) + us / 1e3
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def one(i: int, name: str, units: int, reps: int, gen: Any) -> dict:
    records = torch.randint(0, 256, (units, ZR.ZY_BYTES), dtype=torch.uint8, device="cuda", generator=gen)
    z, value = (bytes([1 + i]) * 64, (1 << 40) + 3 * i + 1), (bytes([101 + i]) * 64, (1 << 41) + 2 * i)
    arms = {"apart": functools.partial(apart, records, z, value), "together": functools.partial(together, records, z, value)}
    got = {arm: fn() for arm, fn in arms.items()}
    equal = got["apart"] == got["together"]
    del got
    walls: dict[str, list[float]] = {arm: [] for arm in arms}
    for _ in range(reps):
        for arm, fn in arms.items():
            walls[arm].append(wall(fn))
    by = {arm: kernels(fn) for arm, fn in arms.items()}
    del arms, records
    torch.cuda.empty_cache()
    return {"call": name, "units": units, "record_bytes": units * ZR.ZY_BYTES, "equal": equal, "wall_s": walls, "kernels_ms": by,
            "kernel_ms": {arm: sum(v.values()) for arm, v in by.items()},
            "wall_median_s": {arm: statistics.median(v) for arm, v in walls.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--units", default="24576,16384,131072,16384", help="each call's records, in the order qkv, o, gate_up, down")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if not (TD.available() and D.zy_available()):
        raise SystemExit("h1_zy_gpu: the row-table and zy_commit kernels don't build for this device")
    gen = torch.Generator(device="cuda").manual_seed(0)
    calls = []
    for i, (name, units) in enumerate(zip(CALLS, (int(u) for u in a.units.split(",")), strict=True)):
        row = one(i, name, units, a.reps, gen)
        calls.append(row)
        k, w = row["kernel_ms"], row["wall_median_s"]
        print(f"{name:8s} {units:7d} records  equal {row['equal']}  kernels {k['apart']:8.2f} -> {k['together']:8.2f} ms  "
              f"wall {1e3 * w['apart']:8.2f} -> {1e3 * w['together']:8.2f} ms", flush=True)
    equal = all(c["equal"] for c in calls)
    step = {arm: {"kernel_ms": a.layers * sum(c["kernel_ms"][arm] for c in calls),
                  "wall_s": a.layers * sum(c["wall_median_s"][arm] for c in calls)} for arm in ("apart", "together")}
    step["saved"] = {k: step["apart"][k] - step["together"][k] for k in ("kernel_ms", "wall_s")}
    step["records"] = a.layers * sum(c["units"] for c in calls)
    step["record_bytes"] = step["records"] * ZR.ZY_BYTES
    out = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "layers": a.layers, "reps": a.reps,
           "calls": calls, "step": step, "equal": equal}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "h1_zy_gpu.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"step ({a.layers} layers, {step['records']} records, {step['record_bytes'] / 1e9:.1f} GB): kernels "
          f"{step['apart']['kernel_ms']:.1f} -> {step['together']['kernel_ms']:.1f} ms, saved {step['saved']['kernel_ms']:.1f} "
          f"ms; wall {step['apart']['wall_s']:.3f} -> {step['together']['wall_s']:.3f} s, saved {step['saved']['wall_s']:.3f} s")
    print(f"h1_zy_gpu: {'PASS' if equal else 'FAIL'}")
    return 0 if equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
