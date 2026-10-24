"""The prover's pass over one GPU's arena beside an engine stand-in (`fused_bench.py --mode serve`, another process on
the same GPU): with no MPS daemon their kernels share the GPU by time slices, as gate run 7's engines and prover did;
under one (`fused.sh`'s BESIDE) they run side by side, each on the SMs its CUDA_MPS_ACTIVE_THREAD_PERCENTAGE allows.
For each phase of --phases in turn the stand-in steps at M 64 at the host clock's multiples of 1/hz with the phase's
arm, and the store (`device.DeviceStore` with the arena as its HBM tier, `node.open_arena`, gate run 7's pass shape)
runs chained passes as a session does for the phase's seconds; an `alone:ARM:SECS` phase runs no passes, so its steps
are the arm's on a GPU the prover leaves alone.

For each phase: the passes' intervals as `Device.calibrate` takes them (each root's arrival from the one before, the
first dropped), their GPU times, the slowest adjacent pair (which sets a stretch's L, `gate_whatif`: L_max is about two
passes plus 2.3 ms), the stand-in's step GPU times, the host's load and every other process on the GPU. Every phase
chains from one fixed challenge and previous root over an arena nothing writes, so pass i's root is the same in every
phase: each root is checked against the same pass's in the phases before (and in --reference's, another run's
beside.json), and pass 0, the pass whose root came slowest and the last pass are checked as a verifier would: the
device's seed is the chain's, the root and every kept level equal the plain pass's (`DeviceStore.plain_root`), the
signs the reference's, and --openings random positions' receipts are the receipter's on the opened blocks, their leaves
opening to the root. Writes OUT/beside.json.

    python fused_beside.py --dir D --lib session.so --kv kv.so --out OUT \
        --phases idle:60,plain:60,scratch:kv:60,fused:fused6:32:60,alone:scratch:kv:20
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import device  # noqa: E402
from engine import write_json  # noqa: E402
from node import open_arena  # noqa: E402
from verity.core.protocols.remote_bandwidth import receipts as rb  # noqa: E402

LOOP_MS = 2.3  # gate_whatif's loop term: L_max is the slowest adjacent pair of passes plus this (pass-bound report)
CHALLENGE = hashlib.sha512(b"verity/remote-session/fused-beside/challenge").digest()[:rb.CHALLENGE_BYTES]
PREVIOUS = bytes(rb.DIGEST_BYTES)


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def chain(s, seconds: float) -> dict:
    """Chained passes from CHALLENGE and PREVIOUS for `seconds` as a session queues them: each root, its host arrival
    (perf ns and wall ns) and its pass's GPU ms, and the handles of pass 0, of the pass whose root came slowest after
    the one before (the first interval dropped, as everywhere here) and of the last pass."""
    gc.collect()
    gc.freeze()
    gc.disable()
    g0 = len(s.pass_gpu_ms)
    s.start_chain(CHALLENGE, PREVIOUS)
    roots, at, wall, kept, slow, last = [], [], [], {}, (0, None, None), None
    for i, root, handle, t in s.chain(until_ns=time.perf_counter_ns() + int(seconds * 1e9)):
        roots.append(root)
        at.append(t)
        wall.append(time.time_ns())
        if i == 0:
            kept[0] = handle
        elif i >= 2 and t - at[-2] > slow[0]:
            slow = (t - at[-2], i, handle)
        last = (i, handle)
        del handle
    gc.enable()
    gc.unfreeze()
    s.sync()
    if slow[1] is not None:
        kept[slow[1]] = slow[2]
    kept[last[0]] = last[1]
    return {"roots": roots, "at": at, "wall": wall, "gpu": s.pass_gpu_ms[g0:], "kept": kept, "slowest": slow[1]}


def verify(s, roots: list[bytes], kept: dict, rng, openings: int) -> dict:
    """Each kept pass i checked as a verifier would: the device's seed is rb.seed(CHALLENGE, i, root i − 1); its root
    and every kept level are the plain pass's at that seed and its signs the reference's; and `openings` random
    positions' receipts are the receipter's on the opened blocks, their leaves opening to root i (`rb.verify_leaf`)."""
    out = {}
    for i, handle in sorted(kept.items()):
        h = s.ready(handle)
        seed = rb.seed(CHALLENGE, i, roots[i - 1] if i else PREVIOUS)
        r = {"device_seed": h[0] == seed, "plain_root": s.plain_root(seed, h) == roots[i]}
        r["plain_levels"] = all(s.plain_level_matches.values())
        r["signs"] = s.signs_match(seed)
        ps = sorted(int(p) for p in rng.choice(s.blocks, openings, replace=False))
        receipter = rb.receipter(rb.DEFAULT, seed)
        got = s.open_window([(handle, ps)])[0]
        r["receipts"] = sum(rb.leaf_receipt(leaf, p, s.plan.length) == receipter(s.plan.block(p), block)
                            for p, (block, leaf, *_) in zip(ps, got))
        r["paths"] = sum(rb.verify_leaf(seed, s.plan, roots[i], p, leaf, salt, path)
                         for p, (_block, leaf, salt, path, *_) in zip(ps, got))
        r["ok"] = (r["device_seed"] and r["plain_root"] and r["plain_levels"] and r["signs"]
                   and r["receipts"] == r["paths"] == openings)
        out[str(i)] = r
    return out


def host_state(ours: set[int]) -> dict:
    """The host's load and the compute processes on this lease's GPU other than `ours` and an MPS server, which would
    share it with the phase."""
    out = {"load": os.getloadavg()}
    uuid = os.environ.get("GPU_LEASE_UUID")
    if uuid:
        r = subprocess.run(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        apps = [[x.strip() for x in ln.split(",")] for ln in r.stdout.splitlines() if ln.strip()]
        out["others_on_gpu"] = [a for a in apps if len(a) == 3 and a[0].isdigit() and int(a[0]) not in ours
                                and not a[1].endswith("nvidia-cuda-mps-server")]
        if r.returncode:
            out["nvidia_smi"] = r.stderr.strip()[-200:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="fused_bench.py --serve-dir")
    ap.add_argument("--lib", required=True, help="session.so")
    ap.add_argument("--kv", required=True, help="kv.so")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", default="idle:60,plain:60,scratch:kv:60,scratch:kv6:60,fused:fused:32:60,"
                                       "fused:fused6:32:60,idle:20")
    ap.add_argument("--reference", type=Path, help="another run's beside.json over the same arena: its roots too")
    ap.add_argument("--openings", type=int, default=32)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--warm-s", type=float, default=10.0)
    ap.add_argument("--wait-s", type=float, default=900.0)
    a = ap.parse_args()
    import torch

    d = a.dir
    seq = 0

    def ctl(**kw) -> str | None:
        """The arm the stand-in took (None: idle, as for an arm whose library failed its gate)."""
        nonlocal seq
        seq += 1
        write_json(d / "ctl.json", {"seq": seq, **kw})
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            try:
                st = json.loads((d / "status.json").read_text())
                if st.get("seq") == seq:
                    return st.get("arm")
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
    ours = {os.getpid(), export.get("pid", -1)}
    arena = open_arena(torch, export, 0)
    dev = device.Device(torch, a.lib, a.kv, arena.numel() // device.BLOCK, seed=1, gpus=1, hbm=[arena])
    s = dev.store
    rng = np.random.default_rng(3)
    ctl(arm=None)
    dev.payload_root(b"fused-beside")
    warm = s.warmup(int(a.warm_s * 1e9))
    reference: list[bytes] = []
    if a.reference is not None:
        ref = json.loads(a.reference.read_text())
        reference = [bytes.fromhex(x) for x in max((ph.get("roots", []) for ph in ref["phases"]), key=len, default=[])]
    out = {"schema": "verity/remote-session/fused-beside/v1", "bytes": arena.numel(), "warmup": warm,
           "read_ms": dev.read_ms([0]), "shape": dev.shape, "challenge": CHALLENGE.hex(), "previous": PREVIOUS.hex(),
           "reference": str(a.reference) if a.reference else None, "reference_roots": len(reference),
           "mps_active_thread_percentage": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"), "phases": []}
    print(f"beside: {arena.numel()} bytes, bare read {out['read_ms']:.2f} ms, warmup {warm}, "
          f"{len(reference)} reference roots", flush=True)
    for spec in a.phases.split(","):
        *arm, secs = spec.split(":")
        alone = arm[0] == "alone"
        arm = ":".join(arm[alone:])
        took = ctl(arm=None if arm == "idle" else arm)
        if arm != "idle" and took != arm:
            print(f"beside {arm}: the stand-in idles instead (its library failed the gate), skipped", flush=True)
            continue
        time.sleep(a.settle_s)
        h0, w0 = host_state(ours), time.time()
        if alone:
            time.sleep(float(secs))
            ph = {"arm": arm, "alone": True, "seconds": float(secs), "start": w0, "end": time.time(),
                  "host": [h0, host_state(ours)]}
            out["phases"].append(ph)
            print(f"beside {arm} alone: {secs} s", flush=True)
            continue
        c = chain(s, float(secs))
        w1 = time.time()
        at, gpu, roots = c["at"], c["gpu"], c["roots"]
        iv = [(b - x) / 1e6 for x, b in zip(at[1:], at[2:])]
        pairs = [x + y for x, y in zip(iv, iv[1:])]
        n = min(len(roots), len(reference))
        same = sum(x == y for x, y in zip(roots, reference))
        reference += roots[n:]
        checks = verify(s, roots, c["kept"], rng, a.openings)
        ph = {"arm": arm, "alone": False, "seconds": float(secs), "passes": len(iv), "start": w0, "end": w1,
              "host": [h0, host_state(ours)],
              "pass_ms": {"median": statistics.median(iv), "p90": pct(iv, 0.9), "p99": pct(iv, 0.99), "max": max(iv),
                          "min": min(iv)},
              "pass_gpu_ms": {"median": statistics.median(gpu[1:]), "max": max(gpu[1:])},
              "max_adjacent_pair_ms": max(pairs), "l_max_model_ms": max(pairs) + LOOP_MS,
              "roots_vs_reference": {"compared": n, "equal": same}, "slowest_pass": c["slowest"], "checks": checks,
              "verified": same == n and all(r["ok"] for r in checks.values()),
              "intervals_ms": iv, "root_wall_ns": c["wall"], "roots": [r.hex() for r in roots]}
        out["phases"].append(ph)
        write_json(a.out / "beside.json", out)
        print(f"beside {arm}: {len(iv)} passes, median {ph['pass_ms']['median']:.2f} ms, p99 {ph['pass_ms']['p99']:.2f}, "
              f"max {ph['pass_ms']['max']:.2f}, slowest pair {max(pairs):.2f} ms; roots {same}/{n} equal the "
              f"reference's, checks {'ok' if ph['verified'] else checks}; load {h0['load'][0]:.0f}, others on the GPU "
              f"{h0.get('others_on_gpu')}", flush=True)
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
    out["verified"] = all(ph["verified"] for ph in out["phases"] if not ph["alone"])
    write_json(a.out / "beside.json", out)
    print("beside: " + "; ".join(
        f"{ph['arm']}{' alone' if ph['alone'] else ''} pass "
        + ("-" if ph["alone"] else f"{ph['pass_ms']['median']:.1f}/{ph['max_adjacent_pair_ms']:.1f} ms")
        + f", step {(ph['engine_step_gpu_ms'] or {}).get('median', 0):.1f} ms" for ph in out["phases"])
        + f"; verified {out['verified']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
