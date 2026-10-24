"""The prover's pass over one GPU's arena beside an engine stand-in (`fused_bench.py --mode serve`, another process on
the same GPU, as gate run 7's engines and prover were: no MPS, so their kernels share the GPU by time slices), for
each arm of --phases in turn: the stand-in steps at M 64 at the host clock's multiples of 1/hz with that arm, and the
store (`device.DeviceStore` with the arena as its HBM tier, `node.open_arena`, gate run 7's pass shape) runs chained
passes as a session does for the phase's seconds.

For each phase: the passes' intervals as `Device.calibrate` takes them (each root's arrival from the one before, the
first dropped), their GPU times, the slowest adjacent pair (which sets a stretch's L, `gate_whatif`: L_max is about two
passes plus 2.3 ms), and the stand-in's step GPU times. Writes OUT/beside.json.

    python fused_beside.py --dir D --lib session.so --kv kv.so --out OUT \
        --phases idle:60,plain:60,scratch:kv:60,fused:fused6:32:60
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import device  # noqa: E402
from engine import write_json  # noqa: E402
from node import open_arena  # noqa: E402

LOOP_MS = 2.3  # gate_whatif's loop term: L_max is the slowest adjacent pair of passes plus this (pass-bound report)


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def chain(torch, s, seconds: float) -> tuple[list[int], list[float], list[int]]:
    """Chained passes for `seconds` as a session queues them: each root's host arrival (perf ns and wall ns), and
    each pass's GPU ms."""
    gc.collect()
    gc.freeze()
    gc.disable()
    g0 = len(s.pass_gpu_ms)
    s.start_chain(os.urandom(32), bytes(64))
    at, wall = [], []
    for _i, _root, handle, t in s.chain(until_ns=time.perf_counter_ns() + int(seconds * 1e9)):
        at.append(t)
        wall.append(time.time_ns())
        del handle
    gc.enable()
    gc.unfreeze()
    s.sync()
    return at, s.pass_gpu_ms[g0:], wall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="fused_bench.py --serve-dir")
    ap.add_argument("--lib", required=True, help="session.so")
    ap.add_argument("--kv", required=True, help="kv.so")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", default="idle:60,plain:60,scratch:kv:60,scratch:kv6:60,fused:fused:32:60,"
                                       "fused:fused6:32:60,idle:20")
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--warm-s", type=float, default=10.0)
    ap.add_argument("--wait-s", type=float, default=900.0)
    a = ap.parse_args()
    import torch

    d = a.dir
    seq = 0

    def ctl(**kw) -> None:
        nonlocal seq
        seq += 1
        write_json(d / "ctl.json", {"seq": seq, **kw})
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            try:
                if json.loads((d / "status.json").read_text()).get("seq") == seq:
                    return
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.02)
        raise TimeoutError(f"the engine stand-in never took ctl {seq}")

    t0 = time.monotonic()
    while not (d / "engine.json").exists():
        if time.monotonic() - t0 > a.wait_s:
            raise TimeoutError("no engine.json from the engine stand-in")
        time.sleep(0.5)
    export = json.loads((d / "engine.json").read_text())
    arena = open_arena(torch, export, 0)
    dev = device.Device(torch, a.lib, a.kv, arena.numel() // device.BLOCK, seed=1, gpus=1, hbm=[arena])
    s = dev.store
    ctl(arm=None)
    warm = s.warmup(int(a.warm_s * 1e9))
    out = {"schema": "verity/remote-session/fused-beside/v0", "bytes": arena.numel(), "warmup": warm,
           "read_ms": dev.read_ms([0]), "shape": dev.shape, "phases": []}
    print(f"beside: {arena.numel()} bytes, bare read {out['read_ms']:.2f} ms, warmup {warm}", flush=True)
    for spec in a.phases.split(","):
        *arm, secs = spec.split(":")
        arm = ":".join(arm)
        ctl(arm=None if arm == "idle" else arm)
        time.sleep(a.settle_s)
        w0 = time.time()
        at, gpu, wall = chain(torch, s, float(secs))
        iv = [(b - x) / 1e6 for x, b in zip(at[1:], at[2:])]
        pairs = [x + y for x, y in zip(iv, iv[1:])]
        ph = {"arm": arm, "seconds": float(secs), "passes": len(iv), "start": w0, "end": time.time(),
              "pass_ms": {"median": statistics.median(iv), "p90": pct(iv, 0.9), "p99": pct(iv, 0.99), "max": max(iv),
                          "min": min(iv)},
              "pass_gpu_ms": {"median": statistics.median(gpu[1:]), "max": max(gpu[1:])},
              "max_adjacent_pair_ms": max(pairs), "l_max_model_ms": max(pairs) + LOOP_MS,
              "intervals_ms": iv, "root_wall_ns": wall}
        out["phases"].append(ph)
        write_json(a.out / "beside.json", out)
        print(f"beside {arm}: {len(iv)} passes, median {ph['pass_ms']['median']:.2f} ms, p99 {ph['pass_ms']['p99']:.2f}, "
              f"max {ph['pass_ms']['max']:.2f}, slowest pair {max(pairs):.2f} ms", flush=True)
    write_json(d / "ctl.json", {"seq": seq + 1, "stop": True})
    steps = []
    p = a.out / "serve.steps.jsonl"
    t1 = time.monotonic()
    while time.monotonic() - t1 < 30 and not (d / "stopped").exists():
        time.sleep(0.2)
    if p.exists():
        steps = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    for ph in out["phases"]:
        xs = [x["gpu_ms"] for x in steps if x["arm"] == ph["arm"] and ph["start"] <= x["t"] <= ph["end"]]
        ph["engine_steps"] = len(xs)
        ph["engine_step_gpu_ms"] = {"median": statistics.median(xs), "max": max(xs)} if xs else None
    write_json(a.out / "beside.json", out)
    print("beside: " + "; ".join(
        f"{ph['arm']} pass {ph['pass_ms']['median']:.1f}/{ph['pass_ms']['max']:.1f} ms, step "
        f"{(ph['engine_step_gpu_ms'] or {}).get('median', 0):.1f} ms" for ph in out["phases"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
