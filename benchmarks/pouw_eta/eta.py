#!/usr/bin/env python3
"""PoUW η on one RTX PRO 6000 (sm_120): the served Pearl-C call's useful FP8 MACs a second over the tensor pipe's peak at the
same clock, by schedule and batch, from build.sh's ship (its run.py's `Dev` and `Pipeline`, unchanged).

Each shape (m real rows x n x k) gets `W` pipelines of its own operands (W = 1: one weight, L2-warm; W = 3: three weights of
n k bytes each in rotation, L2-cold past the 128 MiB L2 at n = k = 8192).  A schedule enqueues 2 W calls on two streams (main
at the greatest priority, side at the least), captured into one CUDA graph as the served path runs its calls, and the graph is
timed back to back.  The schedules, over the served call's steps (`SERVED`, pouw_pearl_c_device.CALL_STEPS):

  serial      every step on the main lane, in order
  serial+tt   serial plus a per-call tile tree (run.py's tree_leaves_t, tree_t): the pass tree's cost, amortized, bounded above
  deferred    the served `deferred`: hash_leaf on the side lane behind the GEMM, c/u rotated over the W pipelines
  deferred-a  what-if (not servable under -h3, card 448a9d2d): A's hm96 tree and seed_A on the side lane too; the padding
              rows' line key is stale, so its bytes are wrong, and only its time means anything
  main-a      deferred-a's main lane alone (the floor rowdig_a + seeds_a, forming, clean-up): hashing hidden for free
  main-a-nf   main-a without forming (fused forming at no cost, the bound)
  nohash      the served call without its hashing (pouw_pearl_c_device's ``hashing`` off)
  gemm        gemm_pearlc alone (C~, U and y), back to back
  gemm_ct     the chain alone (C~), back to back
  plain       this build's own plain FP8 GEMM (gemm_fast, the faster of run.py's two), at m a multiple of 128
  par<c>      c independent served calls at once, each serial on a stream of its own (W a multiple of c): batching calls
              by concurrency rather than by rows

Modes (ETA_MODE): `time` (default) prints one JSON line per (shape, W, schedule); `profile` runs each shape's steps once
under cuProfilerStart/Stop for ncu (`--profile-from-start off`): one serial call kernel by kernel, then (ETA_GRAPHS=1, for
`--graph-profiling graph`) each schedule's graph once, in the order its `profile` line lists; `clock` holds gemm_pearlc at
the largest shape for ETA_HOLD_S seconds with NVML sampled every 20 ms (read-only: nothing here sets a clock); `power`
holds each schedule's graph the same way (its clock, power and throttle reasons, so its energy a call); `sha` times
s512_rows at 128 (served), 64 and 32 threads a CTA, checking the digests are the served ones, and s512_seeds over 2^20
independent rows (one thread each: this SHA-512 code's compression rate when nothing serializes it).

  SHIP=<build.sh output> PEARLC_EXPECT_UUID=... ETA_SHAPES="32x8192x8192 8192x8192x8192" python3 eta.py
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
import statistics
import sys
import threading
import time

SHIP = os.environ.get("SHIP") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SHIP)
import run as R  # noqa: E402

SERVED = ("pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "lines_ea", "stats_a", "eprime_a", "form_a",
          "af", "pa", "gemm_pearlc", "hash_leaf")
HASH = frozenset({"pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "lines_ea", "hash_leaf"})
FLOOR = ("pad_a", "rowdig_a", "seeds_a")
A_TREE = ("hash_rows_a", "tree_a", "seed_a")
FORMING = frozenset({"lines_ea", "stats_a", "eprime_a", "form_a"})
TILE_TREE = ("tree_leaves_t", "tree_t")
SCHEDULES = ("serial", "serial+tt", "deferred", "deferred-a", "main-a", "main-a-nf", "nohash", "gemm", "gemm_ct", "plain")
MACS_PER_SM_CLOCK, CAP_MHZ = 1024, 2430
CAPTURE_RELAXED, STREAM_NON_BLOCKING, EVENT_NO_TIMING = 2, 1, 2

u, i32, vp, P = C.c_uint, C.c_int, C.c_void_p, C.POINTER
EXTRA = {
    "cuStreamCreateWithPriority": [P(vp), u, i32], "cuCtxGetStreamPriorityRange": [P(i32), P(i32)],
    "cuStreamWaitEvent": [vp, vp, u], "cuStreamSynchronize": [vp], "cuStreamBeginCapture_v2": [vp, i32],
    "cuStreamEndCapture": [vp, P(vp)], "cuGraphInstantiateWithFlags": [P(vp), vp, C.c_ulonglong], "cuGraphLaunch": [vp, vp],
    "cuGraphExecDestroy": [vp], "cuGraphDestroy": [vp], "cuProfilerStart": [], "cuProfilerStop": [],
}
NVML_EXTRA = {"nvmlDeviceGetMaxClockInfo": [vp, i32, P(u)], "nvmlDeviceGetEnforcedPowerLimit": [vp, P(u)],
              "nvmlDeviceGetApplicationsClock": [vp, i32, P(u)]}


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


class Lanes:
    def __init__(self, dev: R.Dev):
        self.dev = dev
        lo, hi = i32(), i32()
        dev.ck("cuCtxGetStreamPriorityRange", C.byref(lo), C.byref(hi))
        self.prio = (lo.value, hi.value)
        self.main, self.side = vp(), vp()
        dev.ck("cuStreamCreateWithPriority", C.byref(self.main), STREAM_NON_BLOCKING, hi.value)
        dev.ck("cuStreamCreateWithPriority", C.byref(self.side), STREAM_NON_BLOCKING, lo.value)
        self.par: list = []

    def streams(self, c: int) -> list:
        """c streams at the main lane's priority, made once."""
        while len(self.par) < c:
            s = vp()
            self.dev.ck("cuStreamCreateWithPriority", C.byref(s), STREAM_NON_BLOCKING, self.prio[1])
            self.par.append(s)
        return self.par[:c]

    def event(self, timing: bool = False) -> vp:
        e = vp()
        self.dev.ck("cuEventCreate", C.byref(e), 0 if timing else EVENT_NO_TIMING)
        return e

    def on(self, stream, fn) -> None:
        old, self.dev.stream = self.dev.stream, stream
        try:
            fn()
        finally:
            self.dev.stream = old

    def record(self, stream) -> vp:
        e = self.event()
        self.dev.ck("cuEventRecord", e, stream)
        return e

    def wait(self, stream, e) -> None:
        self.dev.ck("cuStreamWaitEvent", stream, e, 0)


PLAIN_PHASES = frozenset({"plain", "plain_job", "adopted", "adopted_job"})


def runnable(pipe: R.Pipeline) -> list:
    """The pipeline's steps, less the plain baselines when the tile is forced to 64 x 64 at m a multiple of 128 (they run on
    the 128 x 128 mainloop, whose tensor maps such a pipeline doesn't encode)."""
    return [(nm, ph, fn) for nm, ph, fn in pipe.steps() if pipe.bm == R.BM or ph not in PLAIN_PHASES]


def steps_of(pipe: R.Pipeline) -> dict:
    return {**{nm: fn for nm, _, fn in runnable(pipe)}, **pipe.extra()}


def enqueue(lanes: Lanes, pipes: list, fns: list[dict], sched: str, calls: int) -> None:
    """`calls` calls of `sched` over the pipelines in rotation, on lanes.main and lanes.side, the side joined at the end."""
    main, side = lanes.main, lanes.side
    if sched.startswith("par"):
        c = int(sched[3:])
        if len(pipes) % c:
            raise ValueError(f"{sched} takes W a multiple of {c}, not {len(pipes)}")
        fork = lanes.record(main)
        for s_i, s in enumerate(lanes.streams(c)):
            lanes.wait(s, fork)
            for i in range(s_i, calls, c):
                f = fns[i % len(pipes)]
                for nm in SERVED:
                    if nm in f:
                        lanes.on(s, f[nm])
            lanes.wait(main, lanes.record(s))
        return
    pending: dict[int, vp] = {}
    for i in range(calls):
        j = i % len(pipes)
        f = fns[j]
        run = lambda names, s=main: [lanes.on(s, f[nm]) for nm in names if nm in f]  # noqa: E731
        if sched in ("serial", "serial+tt"):
            run(SERVED + (TILE_TREE if sched == "serial+tt" else ()))
        elif sched == "nohash":
            run([nm for nm in SERVED if nm not in HASH])
        elif sched in ("gemm", "gemm_ct"):
            run(["gemm_pearlc" if sched == "gemm" else "gemm_ct"])
        elif sched == "plain":
            run(["gemm_fast"])
        elif sched in ("main-a", "main-a-nf"):
            keep = [nm for nm in SERVED if nm not in HASH or nm in FLOOR or nm == "lines_ea"]
            run([nm for nm in keep if not (sched == "main-a-nf" and nm in FORMING)])
        elif sched == "deferred":
            run([nm for nm in SERVED if nm not in ("af", "pa", "gemm_pearlc", "hash_leaf")])
            run(["af", "pa"])
            if j in pending:
                lanes.wait(main, pending.pop(j))
            run(["gemm_pearlc"])
            lanes.wait(side, lanes.record(main))
            run(["hash_leaf"], side)
            pending[j] = lanes.record(side)
        elif sched == "deferred-a":
            if j in pending:
                lanes.wait(main, pending.pop(j))
            run(FLOOR)
            lanes.wait(side, lanes.record(main))
            run(A_TREE, side)
            run([nm for nm in SERVED if nm not in HASH or nm == "lines_ea"])
            lanes.wait(side, lanes.record(main))
            run(["hash_leaf"], side)
            pending[j] = lanes.record(side)
        else:
            raise ValueError(sched)
    for e in pending.values():
        lanes.wait(main, e)


class Graph:
    def __init__(self, lanes: Lanes, body):
        d = lanes.dev
        self.d, self.lanes, self.g, self.x = d, lanes, vp(), vp()
        d.ck("cuStreamBeginCapture_v2", lanes.main, CAPTURE_RELAXED)
        try:
            body()
        finally:
            d.ck("cuStreamEndCapture", lanes.main, C.byref(self.g))
        d.ck("cuGraphInstantiateWithFlags", C.byref(self.x), self.g, 0)

    def launch(self) -> None:
        self.d.ck("cuGraphLaunch", self.x, self.lanes.main)

    def close(self) -> None:
        self.d.cu.cuGraphExecDestroy(self.x)
        self.d.cu.cuGraphDestroy(self.g)


def timed(lanes: Lanes, body, reps: int) -> list[float]:
    d, s = lanes.dev, lanes.main
    body()
    d.ck("cuStreamSynchronize", s)
    out = []
    for _ in range(reps):
        a, b, ms = lanes.event(True), lanes.event(True), C.c_float()
        d.ck("cuEventRecord", a, s)
        body()
        d.ck("cuEventRecord", b, s)
        d.ck("cuEventSynchronize", b)
        d.ck("cuEventElapsedTime", C.byref(ms), a, b)
        d.cu.cuEventDestroy_v2(a)
        d.cu.cuEventDestroy_v2(b)
        out.append(ms.value)
    return out


def peak_macs(mhz: float, sms: int) -> float:
    return sms * MACS_PER_SM_CLOCK * mhz * 1e6


def make(dev: R.Dev, shape, w: int, bm: int | None, seed0: int):
    """W pipelines at `shape` (m real rows, padded to the kernels' 64), each its own A and B, run once (weight side, call)."""
    m_real, n, k = shape
    m = R.cdiv(m_real, R.BM_D) * R.BM_D
    keys = {nm: dev.put(hashlib.shake_256(f"pearl-c-timing/{nm}".encode()).digest(sz))
            for nm, sz in R.TIMING_KEYS.items()}
    pipes, own = [], []
    for j in range(w):
        xa, xb = dev.alloc(m * 4 * k), dev.alloc(n * 4 * k)
        for x, rows, seed in ((xa, m_real, seed0 + 2 * j), (xb, n, seed0 + 2 * j + 1)):
            dev.launch("gen_f32", R.cdiv(rows * k, 256), 256, p=(x,), i=(rows, k, seed), stream=None)
        kw = {"bm": bm} if bm else {}
        pipe = R.Pipeline(dev, m, n, k, "sm120", keys, xa, xb, m_real=m_real, **kw)
        dev.sync()
        for _, _, fn in runnable(pipe):
            fn()
        dev.sync()
        pipes.append(pipe)
        own += [xa, xb]
    return pipes, own + list(keys.values())


def release(dev, pipes, own) -> None:
    for p in pipes:
        p.free()
    for x in own:
        dev.free(x)


def shape_row(shape, w, bm, pipes) -> dict:
    p = pipes[0]
    return {"shape": list(shape), "m_padded": p.m, "tile": [p.bm, p.bm], "w": w, "macs_useful": shape[0] * shape[1] * shape[2],
            "macs_computed": p.m * shape[1] * shape[2], "bm_forced": bm}


def time_mode(dev: R.Dev, nv: R.Nvml, lanes: Lanes) -> None:
    reps = int(os.environ.get("ETA_REPS", "15"))
    ws = [int(x) for x in os.environ.get("ETA_WS", "1 3").split()]
    scheds = os.environ.get("ETA_SCHEDULES", " ".join(SCHEDULES)).split()
    for spec in os.environ.get("ETA_SHAPES", "32x8192x8192 128x8192x8192 8192x8192x8192").split():
        bm = None
        if "@" in spec:
            spec, bm_s = spec.split("@")
            bm = int(bm_s)
        shape = R.parse_shape(spec)
        for w in ws:
            pipes, own = make(dev, shape, w, bm, 101)
            fns = [steps_of(p) for p in pipes]
            base = shape_row(shape, w, bm, pipes)
            calls = 2 * w if w > 1 else 2
            for sched in scheds:
                if sched == "plain" and "gemm_fast" not in fns[0] or sched.startswith("par") and w % int(sched[3:]):
                    continue
                row = {"kind": "eta", **base, "schedule": sched, "calls_per_graph": calls}
                try:
                    g = Graph(lanes, lambda: enqueue(lanes, pipes, fns, sched, calls))
                except RuntimeError as e:
                    row["graph_error"] = str(e)
                    emit(row)
                    continue
                before = nv.read()
                t = timed(lanes, g.launch, reps)
                after = nv.read()
                g.close()
                mhz = [c["sm_mhz"] for c in (before, after) if "sm_mhz" in c]
                per_call = statistics.median(t) / calls
                row.update({"graph_ms": [round(x, 4) for x in t], "per_call_ms": round(per_call, 5), "clk_before": before,
                            "clk_after": after})
                if mhz:
                    f = statistics.mean(mhz)
                    useful = base["macs_useful"] / (per_call * 1e-3)
                    row.update({"sm_mhz": f, "useful_macs_per_s": useful,
                                "eta": round(useful / peak_macs(f, dev.sms), 5),
                                "eta_cap": round(useful / peak_macs(CAP_MHZ, dev.sms), 5),
                                "eta_computed": round(base["macs_computed"] / (per_call * 1e-3) / peak_macs(f, dev.sms), 5)})
                emit(row)
            release(dev, pipes, own)


def profile_mode(dev: R.Dev, lanes: Lanes) -> None:
    graphs = os.environ.get("ETA_GRAPHS") == "1"
    scheds = os.environ.get("ETA_SCHEDULES", "serial deferred deferred-a main-a nohash gemm gemm_ct plain").split()
    for spec in os.environ.get("ETA_SHAPES", "32x8192x8192 128x8192x8192 8192x8192x8192").split():
        shape = R.parse_shape(spec)
        w = int(os.environ.get("ETA_W", "3"))
        pipes, own = make(dev, shape, w, None, 101)
        fns = [steps_of(p) for p in pipes]
        calls = 2 * w
        order = []
        if graphs:
            gs = []
            for sched in scheds:
                if sched == "plain" and "gemm_fast" not in fns[0]:
                    continue
                g = Graph(lanes, lambda: enqueue(lanes, pipes, fns, sched, calls))
                g.launch()
                dev.ck("cuStreamSynchronize", lanes.main)
                gs.append((sched, g))
            dev.ck("cuProfilerStart")
            for sched, g in gs:
                g.launch()
                dev.ck("cuStreamSynchronize", lanes.main)
                order.append(sched)
            dev.ck("cuProfilerStop")
            for _, g in gs:
                g.close()
        else:
            names = list(SERVED) + list(TILE_TREE) + ["gemm_ct", "gemm_fast", "gemm_plain"]
            f = fns[0]
            dev.ck("cuProfilerStart")
            for nm in names:
                if nm in f:
                    lanes.on(lanes.main, f[nm])
                    dev.ck("cuStreamSynchronize", lanes.main)
                    order.append(nm)
            dev.ck("cuProfilerStop")
        emit({"kind": "profile", **shape_row(shape, w, None, pipes), "graphs": graphs, "calls_per_graph": calls if graphs else 1,
              "order": order})
        release(dev, pipes, own)


def sampled(nv: R.Nvml, body) -> tuple[float, dict]:
    """body() (returning its ms) with NVML sampled every 20 ms from a thread (ctypes drops the GIL in foreign calls): its ms
    and the middle half's clock, power and throttle reasons."""
    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            samples.append({"t": time.monotonic(), **nv.read()})
            stop.wait(0.02)
    th = threading.Thread(target=sample)
    th.start()
    try:
        ms = body()
    finally:
        stop.set()
        th.join()
    mid = samples[len(samples) // 4: 3 * len(samples) // 4] or samples
    has = bool(mid) and "sm_mhz" in mid[0]
    return ms, {"samples": len(samples),
                "sm_mhz_median": statistics.median(s["sm_mhz"] for s in mid) if has else None,
                "power_w_median": statistics.median(s["power_w"] for s in mid) if has else None,
                "power_w_max": max(s["power_w"] for s in mid) if has else None,
                "throttle_seen": sorted({s.get("throttle") for s in mid}), "temp_c_max": max(s.get("temp_c", 0) for s in mid),
                "trace": [{k: s.get(k) for k in ("sm_mhz", "power_w", "throttle", "temp_c")} for s in samples[::5]]}


def limits(nv: R.Nvml) -> dict:
    lib, info = nv.lib, {}
    if lib:
        R.bind(lib, NVML_EXTRA)
        for nm, t in (("max_sm_mhz", 1), ("max_mem_mhz", 2)):
            v = u()
            info[nm] = v.value if lib.nvmlDeviceGetMaxClockInfo(nv.h, t, C.byref(v)) == 0 else None
        v = u()
        info["enforced_power_limit_w"] = v.value / 1000 if lib.nvmlDeviceGetEnforcedPowerLimit(nv.h, C.byref(v)) == 0 else None
        v = u()
        info["applications_sm_mhz"] = v.value if lib.nvmlDeviceGetApplicationsClock(nv.h, 1, C.byref(v)) == 0 else None
    return info


def clock_mode(dev: R.Dev, nv: R.Nvml, lanes: Lanes) -> None:
    """gemm_pearlc (and gemm_fast) held for ETA_HOLD_S each: the clock, power and throttle reasons a sustained GEMM sees
    under the node's lock."""
    hold = float(os.environ.get("ETA_HOLD_S", "4"))
    shape = R.parse_shape(os.environ.get("ETA_CLOCK_SHAPE", "8192x8192x8192"))
    pipes, own = make(dev, shape, 1, None, 101)
    f = steps_of(pipes[0])
    info = limits(nv)
    for nm in ("gemm_pearlc", "gemm_fast", "gemm_ct"):
        if nm not in f:
            continue
        t = timed(lanes, lambda: lanes.on(lanes.main, f[nm]), 3)
        n_launch = max(1, int(hold * 1e3 / statistics.median(t)))
        ms, st = sampled(nv, lambda: timed(lanes, lambda: [lanes.on(lanes.main, f[nm]) for _ in range(n_launch)], 1)[0])
        row = {"kind": "clock", "shape": list(shape), "kernel": nm, "launches": n_launch, "ms": ms,
               "per_launch_ms": ms / n_launch, **info, **st}
        row["tflops"] = 2 * shape[0] * shape[1] * shape[2] / (row["per_launch_ms"] * 1e-3) / 1e12
        emit(row)
    release(dev, pipes, own)


def power_mode(dev: R.Dev, nv: R.Nvml, lanes: Lanes) -> None:
    """Each schedule's graph (W = ETA_W) held for ETA_HOLD_S: per-call time under the hold, the clock it settles at, its power
    and so its energy a call, and η at that clock."""
    hold = float(os.environ.get("ETA_HOLD_S", "2"))
    w = int(os.environ.get("ETA_W", "3"))
    scheds = os.environ.get("ETA_SCHEDULES", "serial nohash gemm gemm_ct plain").split()
    info = limits(nv)
    for spec in os.environ.get("ETA_SHAPES", "32x8192x8192 128x8192x8192 8192x8192x8192").split():
        shape = R.parse_shape(spec)
        pipes, own = make(dev, shape, w, None, 101)
        fns = [steps_of(p) for p in pipes]
        base, calls = shape_row(shape, w, None, pipes), 2 * w
        for sched in scheds:
            if sched == "plain" and "gemm_fast" not in fns[0] or sched.startswith("par") and w % int(sched[3:]):
                continue
            g = Graph(lanes, lambda: enqueue(lanes, pipes, fns, sched, calls))
            n = max(1, int(hold * 1e3 / statistics.median(timed(lanes, g.launch, 3))))
            ms, st = sampled(nv, lambda: timed(lanes, lambda: [g.launch() for _ in range(n)], 1)[0])
            g.close()
            per_call = ms / (n * calls)
            row = {"kind": "power", **base, "schedule": sched, "calls_per_graph": calls, "graphs": n,
                   "per_call_ms": round(per_call, 5), **info, **st}
            if st["sm_mhz_median"]:
                useful = base["macs_useful"] / (per_call * 1e-3)
                row.update({"eta": round(useful / peak_macs(st["sm_mhz_median"], dev.sms), 5),
                            "eta_cap": round(useful / peak_macs(CAP_MHZ, dev.sms), 5),
                            "energy_mj_per_call": round(st["power_w_median"] * per_call, 4)})
            emit(row)
        release(dev, pipes, own)


def sha_mode(dev: R.Dev, lanes: Lanes) -> None:
    """s512_rows at 128 (the served launch), 64 and 32 threads a CTA over each shape's real rows of A, its digests compared
    with the served ones; then s512_seeds over 2^20 rows, one thread a row (5 compressions each)."""
    reps = int(os.environ.get("ETA_REPS", "15"))
    for n_spec, spec in enumerate(os.environ.get("ETA_SHAPES", "32x8192x8192 128x8192x8192 8192x8192x8192").split()):
        shape = R.parse_shape(spec)
        m_real, _, k = shape
        pipes, own = make(dev, shape, 1, None, 101)
        p = pipes[0]
        segs = k // 256
        per_row = 10 * segs + 2 + segs // 2   # a segment: header, 8 blocks, pad; the fold: Q, digest pairs, pad
        want, out = dev.read(p.b["rowdig_a"], 64 * m_real), dev.alloc(64 * m_real)
        for block in (128, 64, 32):
            launch = lambda: dev.launch("s512_rows", m_real, block, p=(p.b["xa"], out), i=(m_real, k), stream=lanes.main)  # noqa: E731
            dev.ck("cuMemsetD8_v2", out, 0, 64 * m_real)
            launch()
            dev.ck("cuStreamSynchronize", lanes.main)
            same = dev.read(out, 64 * m_real) == want
            g = Graph(lanes, lambda: [launch() for _ in range(10)])
            ms = statistics.median(timed(lanes, g.launch, reps)) / 10
            g.close()
            emit({"kind": "sha", "kernel": "s512_rows", "shape": list(shape), "block": block, "ms": round(ms, 5),
                  "same_digests": same, "compressions": m_real * per_row,
                  "compressions_per_s": m_real * per_row / (ms * 1e-3)})
        dev.free(out)
        if n_spec == 0:
            rows = 1 << 20
            x, seeds, lk = dev.alloc(64 * rows), dev.alloc(32 * rows), dev.alloc(32 * rows)
            dev.ck("cuMemsetD8_v2", x, 7, 64 * rows)
            launch = lambda: dev.launch("s512_seeds", R.cdiv(rows, 128), 128, p=(x, seeds, lk, p.keys["unit"]),  # noqa: E731
                                        i=(rows, 0, 1), stream=lanes.main)
            g = Graph(lanes, lambda: [launch() for _ in range(5)])
            ms = statistics.median(timed(lanes, g.launch, reps)) / 5
            g.close()
            emit({"kind": "sha", "kernel": "s512_seeds", "rows": rows, "block": 128, "ms": round(ms, 5),
                  "compressions": 5 * rows, "compressions_per_s": 5 * rows / (ms * 1e-3)})
            for b in (x, seeds, lk):
                dev.free(b)
        release(dev, pipes, own)


def main() -> int:
    image = open(os.path.join(SHIP, "pearl_c_sm120.cubin"), "rb").read()
    dev = R.Dev(image)
    refused = dev.identity(os.environ.get("PEARLC_EXPECT_UUID"))
    nv = R.Nvml(dev.uuid)
    emit({"kind": "env", "device": dev.name, "uuid": dev.uuid, "sms": dev.sms, "cc": list(dev.cc), "driver": dev.driver,
          "nvml_driver": nv.driver, "clk": nv.read(), "refused": refused, "mode": os.environ.get("ETA_MODE", "time"),
          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if refused:
        print("REFUSED: " + "; ".join(refused), file=sys.stderr)
        return 3
    dev.open()
    R.bind(dev.cu, EXTRA)
    lanes = Lanes(dev)
    emit({"kind": "lanes", "priority_least_greatest": list(lanes.prio)})
    mode = os.environ.get("ETA_MODE", "time")
    {"time": lambda: time_mode(dev, nv, lanes), "profile": lambda: profile_mode(dev, lanes),
     "clock": lambda: clock_mode(dev, nv, lanes), "power": lambda: power_mode(dev, nv, lanes),
     "sha": lambda: sha_mode(dev, lanes)}[mode]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
