"""Option A's prover side priced (served_h1/partials.py, CA 21 Oct 16:04Z): a drawn leaf's 32 lane partials after the
draw, on the cpu (receipts.py's chain split by lane) and the gpu (h1u.cu's kh_lanes on the drawn leaves' gathered
operands), over Llama-3.1-8B's K mix at T 64: a step's 86,016 units, 77,824 at K = 4096 (qkv, o, gate_up) and 8,192 at
K = 14,336 (down), so a window of D draws is D 77,824 / 86,016 leaves at 4096 and the rest at 14,336 (SERVE-REG's K'
of about 30-37k draws a window at eps 0.1-0.081 %, delta 2^-40: --window 37639).

Gated first: GATE_LEAVES leaves of each K on the cpu and the gpu byte for byte, each XOR over lanes kh_lanes' receipt
(exit 4 otherwise).  Then timed:

* cpu, one core (torch on one thread): leaves a second at each K, `lane_partials_many` whole and split into hashlib's
  noise and fold keys (`noise_blocks_cpu`) and the torch chain (`tile_receipts`);
* cpu, every core: a spawn pool of one-thread workers (os.sched_getaffinity's count, at most a quarter of MemAvailable
  at WORKER_BYTES a worker), each with its own window (POOL_CALLS), the leaves split into tasks of one chunk; leaves a
  second over the pool's wall time (the node is shared: the load averages are recorded before and after);
* gpu kernel: kh_lanes alone on pre-gathered operands and staged seeds, CUDA events, at each batch size;
* gpu end to end: a window's leaves through `gpu_batches` (their seeds by kh_seeds_at, the gather, kh_lanes) to the
  device, and through `lane_partials_many` to host numpy (the records' copy out too); beside them the seeds alone, on
  the device (`_seed_inputs` and kh_seeds_at) and by hashlib on the host (`_seeds`, the cpu backend's).

The window: per K one B of the K's widest call (gate_up's [28672, 4096], down's [4096, 14336]) and STEPS steps of
A [64, K], every row live; leaves drawn uniformly over steps and units (the cost is the operands' size, not their values).
Run it with sbench.sh's libraries (SBENCH_SCRIPT=partialsbench.py; its --arms dither-a-hbc builds the served h1u_kc.so,
$HB_H1U_KC, with HB_KC_FLAGS), or --lib any h1u.cu build.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/pouw/served_h1"))
import partials as PT  # noqa: E402
import receipts as RC  # noqa: E402

T, STEPS = 64, 16
CALLS = {4096: 28672, 14336: 4096}  # K: the widest call's N
STEP_UNITS = {4096: 77824, 14336: 8192}
GATE_LEAVES = 16


def window(seed: int = 0, device="cpu", calls: dict[int, int] = CALLS) -> PT.Window:
    g = torch.Generator().manual_seed(seed)
    ks = list(calls)
    b = [torch.randint(-128, 128, (calls[k], k), generator=g, dtype=torch.int8).to(device) for k in ks]
    units = [(T // 32) * (calls[k] // 32) for k in ks]
    win = PT.Window(b, [bytes([1 + w]) * 64 for w in range(len(ks))], [0, units[0]], sum(units))
    for s in range(STEPS):
        a = [torch.randint(-128, 128, (T, k), generator=g, dtype=torch.int8).to(device) for k in ks]
        win.steps[s] = PT.Step(a, [bytes([s, w]) * 32 for w in range(len(ks))], T, bytes([s]) * 64)
    return win


def leaves(win: PT.Window, k: int, n: int, seed: int) -> list[PT.Leaf]:
    w = [x.shape[1] for x in win.b].index(k)
    rng = np.random.default_rng(seed)
    nu = (T // 32) * (win.b[w].shape[0] // 32)
    return [PT.Leaf(int(s), w, int(u)) for s, u in zip(rng.integers(STEPS, size=n), rng.integers(nu, size=n))]


def mix(n: int) -> dict[int, int]:
    tot = sum(STEP_UNITS.values())
    n4 = round(n * STEP_UNITS[4096] / tot)
    return {4096: n4, 14336: n - n4}


# -- cpu ---------------------------------------------------------------------------------------------------------------

def cpu_one(win: PT.Window, k: int, n: int, chunk: int) -> dict:
    torch.set_num_threads(1)
    lv = leaves(win, k, n, 1)
    PT.lane_partials_many(win, lv[:chunk], chunk=chunk)
    t0 = time.perf_counter()
    PT.lane_partials_many(win, lv, chunk=chunk)
    whole = time.perf_counter() - t0
    # the split, on one chunk: hashlib's noise and keys, then the torch chain
    part = lv[:chunk]
    ats = [PT._at(win, x) for x in part]
    lay = RC.layout(k)
    t0 = time.perf_counter()
    sj, sr = PT._seeds(win, ats)
    t1 = time.perf_counter()
    noise, keys = RC.noise_blocks_cpu(sj, lay.blocks), RC.noise_blocks_cpu(sr, PT.LANES)
    t2 = time.perf_counter()
    a, b = PT._gather(win, ats, torch.device("cpu"))
    RC.tile_receipts(a.to(torch.int64), b.to(torch.int64), noise, keys, lanes=True)
    t3 = time.perf_counter()
    c = len(part)
    return {"leaves": n, "chunk": chunk, "us_per_leaf": 1e6 * whole / n,
            "split_us_per_leaf": {"seeds": 1e6 * (t1 - t0) / c, "noise_and_keys_hashlib": 1e6 * (t2 - t1) / c,
                                  "gather_and_chain_torch": 1e6 * (t3 - t2) / c},
            "sha512_blocks_per_leaf": lay.blocks + PT.LANES + 4}


_W: PT.Window | None = None
# a pool worker's window: B of 4096 rows at each K (a leaf's cost is its operands' size, not which column block), and
# the bytes a worker may hold at its chunk's peak (tile_receipts' int64 temporaries)
POOL_CALLS = {4096: 4096, 14336: 4096}
WORKER_BYTES = 1 << 30


def _init() -> None:
    global _W
    torch.set_num_threads(1)
    _W = window(calls=POOL_CALLS)


def _task(arg: tuple[int, int, int, int]) -> int:
    k, n, seed, chunk = arg
    PT.lane_partials_many(_W, leaves(_W, k, n, seed), chunk=chunk)
    return n


def cpu_pool(k: int, workers: int, per_task: int, tasks_per_worker: int, chunk: int) -> dict:
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init) as pool:
        pool.map(_task, [(k, chunk, 10_000 + i, chunk) for i in range(workers)], chunksize=1)
        jobs = [(k, per_task, 20_000 + i, chunk) for i in range(workers * tasks_per_worker)]
        t0 = time.perf_counter()
        n = sum(pool.map(_task, jobs, chunksize=1))
        dt = time.perf_counter() - t0
    return {"workers": workers, "leaves": n, "chunk": chunk, "task_leaves": per_task, "seconds": dt,
            "leaves_per_s": n / dt, "us_per_leaf_wall": 1e6 * dt / n, "us_per_leaf_per_worker": 1e6 * dt * workers / n}


# -- gpu ---------------------------------------------------------------------------------------------------------------

def staged(win: PT.Window, lv: list[PT.Leaf], dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ats = [PT._at(win, x) for x in lv]
    sj, sr = PT._seeds(win, ats)
    a, b = PT._gather(win, ats, dev)
    return a, b, torch.cat([RC.seed_words(sj, dev), RC.seed_words(sr, dev)], 1)


def gpu_kernel(kern: PT.Kernel, win: PT.Window, k: int, sizes: list[int], reps: int, dev) -> dict:
    out = {}
    big = max(sizes)
    a, b, sd = staged(win, leaves(win, k, big, 3), dev)
    for n in sizes:
        ms = []
        for r in range(reps + 1):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            kern.run(a[:n], b[:n], sd[:n])
            e1.record()
            torch.cuda.synchronize()
            if r:
                ms.append(e0.elapsed_time(e1))
        med = statistics.median(ms)
        out[str(n)] = {"ms": med, "ns_per_leaf": 1e6 * med / n, "ms_all": ms}
    return out


def gpu_window(kern: PT.Kernel, win: PT.Window, n: int, reps: int) -> dict:
    lv = [x for k, c in mix(n).items() for x in leaves(win, k, c, 5 + k)]
    res: dict = {"leaves": len(lv), "by_k": {str(k): c for k, c in mix(n).items()}}
    dev_ms, host_ms, seed_ms, dseed_ms = [], [], [], []
    for r in range(reps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in PT.gpu_batches(win, lv, kern):
            pass
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        PT.lane_partials_many(win, lv, "gpu", kern)
        t2 = time.perf_counter()
        ats = [PT._at(win, x) for x in lv]
        PT._seeds(win, ats)
        t3 = time.perf_counter()
        kern.seeds(*PT._seed_inputs(win, ats, win.b[0].device))
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        if r:
            dev_ms.append(1e3 * (t1 - t0))
            host_ms.append(1e3 * (t2 - t1))
            seed_ms.append(1e3 * (t3 - t2))
            dseed_ms.append(1e3 * (t4 - t3))
    res["to_device_ms"] = statistics.median(dev_ms)
    res["to_host_ms"] = statistics.median(host_ms)
    res["host_seeds_ms"] = statistics.median(seed_ms)
    res["device_seeds_ms"] = statistics.median(dseed_ms)
    res["to_device_us_per_leaf"] = 1e3 * res["to_device_ms"] / len(lv)
    return res


def gate(kern: PT.Kernel, win: PT.Window) -> dict:
    bad = {}
    for k in CALLS:
        lv = leaves(win, k, GATE_LEAVES, 7)
        cpu = PT.lane_partials_many(win, lv)
        gpu = PT.lane_partials_many(win, lv, "gpu", kern)
        rc = {}
        for sel, out, _ in PT.gpu_batches(win, lv, kern):
            for j, i in enumerate(sel):
                rc[i] = out[j].cpu().numpy().view(np.uint32)
        bad[str(k)] = sum(c.tobytes() != g.tobytes() or not np.array_equal(PT.combine(c), rc[i])
                          for i, (c, g) in enumerate(zip(cpu, gpu)))
    return {"leaves_per_k": GATE_LEAVES, "bad": bad, "ok": not any(bad.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.environ.get("HB_H1U_KC"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch")
    ap.add_argument("--arms", help="sbench.sh's: dither-a-hbc builds the served h1u_kc.so ($HB_H1U_KC)")
    ap.add_argument("--window", type=int, default=37639)
    ap.add_argument("--cpu-leaves", default="4096=64,14336=32", help="K=N: one core's leaves at each K and chunk")
    ap.add_argument("--cpu-chunks", default="8,32", help="one core's chunks (the best is its price)")
    ap.add_argument("--pool-chunk", type=int, default=8)
    ap.add_argument("--pool-task-chunks", type=int, default=4, help="chunks a pool task")
    ap.add_argument("--pool-tasks", type=int, default=3, help="tasks a worker")
    ap.add_argument("--sizes", default="1,32,188,1024,4096,16384")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-gpu", action="store_true", help="the cpu's prices alone")
    ap.add_argument("--no-cpu", action="store_true", help="the gpu's prices alone")
    a, _ = ap.parse_known_args()
    if not a.lib and not a.no_gpu:
        print("partialsbench: no h1u build (--lib or $HB_H1U_KC)", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc: dict = {"lib": a.lib, "window": a.window, "mix": {str(k): c for k, c in mix(a.window).items()},
                 "steps": STEPS, "t": T, "calls": {str(k): n for k, n in CALLS.items()}}
    cores = len(os.sched_getaffinity(0))
    avail = int(next(x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable")).split()[1])
    # the node is shared: the pool holds at most a quarter of what is free
    workers = max(1, min(cores, (avail * 1024 // 4) // WORKER_BYTES))
    doc["host"] = {"cpus": os.cpu_count(), "affinity": cores, "pool_workers": workers, "mem_available_gib": avail / 2**20,
                   "loadavg_before": os.getloadavg(), "torch": torch.__version__}
    m = mix(a.window)
    price = doc["window_price"] = {}
    # cpu first: the pool spawns before this process starts CUDA
    if not a.no_cpu:
        win = window()
        cl = {int(k): int(n) for k, n in (x.split("=") for x in a.cpu_leaves.split(","))}
        chunks = [int(x) for x in a.cpu_chunks.split(",")]
        doc["cpu_one_core"] = {}
        for k, n in cl.items():
            by = {str(c): cpu_one(win, k, n, c) for c in chunks}
            best = min(by.values(), key=lambda r: r["us_per_leaf"])
            doc["cpu_one_core"][str(k)] = {**best, "by_chunk": {c: r["us_per_leaf"] for c, r in by.items()}}
        print(json.dumps({"cpu_one_core": doc["cpu_one_core"]}, indent=1), flush=True)
        doc["cpu_all_cores"] = {}
        for k in CALLS:
            doc["cpu_all_cores"][str(k)] = cpu_pool(k, workers, a.pool_chunk * a.pool_task_chunks, a.pool_tasks,
                                                    a.pool_chunk)
            print(json.dumps({"cpu_all_cores": {str(k): doc["cpu_all_cores"][str(k)]}}), flush=True)
        doc["host"]["loadavg_after_cpu"] = os.getloadavg()
        # the window's price: its leaves at each K, per backend
        c1, ca = doc["cpu_one_core"], doc["cpu_all_cores"]
        price.update(cpu_one_core_s=sum(m[k] * c1[str(k)]["us_per_leaf"] for k in CALLS) / 1e6,
                     cpu_all_cores_s=sum(m[k] / ca[str(k)]["leaves_per_s"] for k in CALLS))
    if not a.no_gpu:
        dev = torch.device("cuda")
        doc["gpu"] = {"name": torch.cuda.get_device_name(dev)}
        kern = PT.Kernel(a.lib)
        gwin = window(device=dev)
        doc["gate"] = gate(kern, gwin)
        print(json.dumps({"gate": doc["gate"]}), flush=True)
        if not doc["gate"]["ok"]:
            (out / "partials-price.json").write_text(json.dumps(doc, indent=1))
            return 4
        sizes = [int(x) for x in a.sizes.split(",")]
        gk = doc["gpu"]["kernel"] = {str(k): gpu_kernel(kern, gwin, k, sorted(set(sizes + [m[k]])), a.reps, dev)
                                     for k in CALLS}
        gw = doc["gpu"]["window"] = gpu_window(kern, gwin, a.window, a.reps)
        price.update(gpu_kernel_ms=sum(gk[str(k)][str(m[k])]["ms"] for k in CALLS), gpu_to_device_ms=gw["to_device_ms"],
                     gpu_to_host_ms=gw["to_host_ms"], gpu_device_seeds_ms=gw["device_seeds_ms"],
                     host_seeds_ms=gw["host_seeds_ms"])
        print(json.dumps({"gpu_window": gw, "gpu_kernel": {k: {n: round(v["ns_per_leaf"], 1) for n, v in d.items()}
                                                           for k, d in gk.items()}}, indent=1), flush=True)
    (out / "partials-price.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"window_price": price}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
