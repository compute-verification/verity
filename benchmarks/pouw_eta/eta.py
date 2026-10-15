#!/usr/bin/env python3
"""PoUW η on one RTX PRO 6000 (sm_120): the served Pearl-C call's useful FP8 MACs a second over the tensor pipe's peak at the
same clock, by schedule and batch, from build.sh's ship (its run.py's `Dev` and `Pipeline`, unchanged).

Each shape (m real rows x n x k) gets `W` pipelines of its own operands (W = 1: one weight, L2-warm; W = 3: three weights of
n k bytes each in rotation, L2-cold past the 128 MiB L2 at n = k = 8192).  A schedule enqueues 2 W calls on two streams (main
at the greatest priority, side at the least), captured into one CUDA graph as the served path runs its calls, and the graph is
timed back to back.  The schedules, over the served call's steps (`SERVED`, pouw_pearl_c_device.CALL_STEPS):

  serial      every step on the main lane, in order
  serial+tt   serial plus a per-call tile tree (run.py's tree_leaves_t, tree_t): the pass tree's cost, amortized, bounded above
  serial-r32  serial with s512_rows launched at 32 threads a CTA instead of 128 (the same digests: `sha` mode checks them)
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

A shape `MxNxK@64` (time and power modes) forces the 64 x 64 tile: its GEMMs (96 registers x 256 threads, 48 KB) leave room
on an SM for a 128-thread hash CTA, where the 128 x 128 mainloop's (168 x 384, the whole register file) leave none.

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
#: also: serial-r32 and par<c>, run when ETA_SCHEDULES names them
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


#: gemm_pearlc pinned to N persistent CTAs (one an SM), leaving the other SMs to the side lane: `deferred-g<N>`
PINNED_GRIDS = (140, 160)


def steps_of(pipe: R.Pipeline) -> dict:
    d, b = pipe.dev, pipe.b
    r32 = {"rowdig_a_32": lambda: d.launch("s512_rows", pipe.m_real, 32, p=(b["xa"], b["rowdig_a"]), i=(pipe.m_real, pipe.k))}
    tile = "_d" if pipe.bm == R.BM_D else ""
    peel = {"pa": b["pa_t"], "pb": b["pb_t"], "rs": b["ia_a"], "cs": b["ia_b"]}
    pinned = {f"gemm_pearlc_g{g}": (lambda g=g: pipe.gemm(f"gemm_pearlc{tile}{pipe.suffix}", "at", "bt", pipe.m, pipe.n, grid=g,
                                                          out=("c", "u", "y"), **peel)) for g in PINNED_GRIDS}
    return {**{nm: fn for nm, _, fn in runnable(pipe)}, **pipe.extra(), **r32, **pinned}


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
        if sched in ("serial", "serial+tt", "serial-r32"):
            names = SERVED + (TILE_TREE if sched == "serial+tt" else ())
            run(["rowdig_a_32" if sched == "serial-r32" and nm == "rowdig_a" else nm for nm in names])
        elif sched == "nohash":
            run([nm for nm in SERVED if nm not in HASH])
        elif sched in ("gemm", "gemm_ct"):
            run(["gemm_pearlc" if sched == "gemm" else "gemm_ct"])
        elif sched == "plain":
            run(["gemm_fast"])
        elif sched in ("hashonly", "hashall"):
            run(["rowdig_a", "hash_leaf"] if sched == "hashonly" else [nm for nm in SERVED if nm in HASH])
        elif sched in ("main-a", "main-a-nf", "main-a-nf-ct"):
            keep = [nm for nm in SERVED if nm not in HASH or nm in FLOOR or nm == "lines_ea"]
            keep = [nm for nm in keep if not (sched != "main-a" and nm in FORMING)]
            run(["gemm_ct" if sched == "main-a-nf-ct" and nm == "gemm_pearlc" else nm for nm in keep])
        elif sched == "deferred" or sched.startswith("deferred-g"):
            gemm = "gemm_pearlc" if sched == "deferred" else f"gemm_pearlc_g{sched[10:]}"
            run([nm for nm in SERVED if nm not in ("af", "pa", "gemm_pearlc", "hash_leaf")])
            run(["af", "pa"])
            if j in pending:
                lanes.wait(main, pending.pop(j))
            run([gemm])
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


def shape_spec(spec: str) -> tuple:
    """`MxNxK`, or `MxNxK@64` to force the 64 x 64 tile: (shape, bm or None)."""
    spec, _, bm_s = spec.partition("@")
    return R.parse_shape(spec), int(bm_s) if bm_s else None


def time_mode(dev: R.Dev, nv: R.Nvml, lanes: Lanes) -> None:
    reps = int(os.environ.get("ETA_REPS", "15"))
    ws = [int(x) for x in os.environ.get("ETA_WS", "1 3").split()]
    scheds = os.environ.get("ETA_SCHEDULES", " ".join(SCHEDULES)).split()
    for spec in os.environ.get("ETA_SHAPES", "32x8192x8192 128x8192x8192 8192x8192x8192").split():
        shape, bm = shape_spec(spec)
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
        shape, bm = shape_spec(spec)
        pipes, own = make(dev, shape, w, bm, 101)
        fns = [steps_of(p) for p in pipes]
        base, calls = shape_row(shape, w, bm, pipes), 2 * w
        for sched in scheds:
            if sched == "plain" and "gemm_fast" not in fns[0] or sched.startswith("par") and w % int(sched[3:]):
                continue
            g = Graph(lanes, lambda: enqueue(lanes, pipes, fns, sched, calls))
            n = max(1, int(hold * 1e3 / max(statistics.median(timed(lanes, g.launch, 3)), 1e-3)))
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


#: what the row digests and tile leaves feed: every committed buffer downstream of them (the call's, then the epoch's B side)
COMMITTED = ("rowdig_a", "tree_a", "seed_a", "line_a", "seeds_a", "lkeys_a", "leaves", "tree_t", "rowdig_b", "tree_b")


def digest(dev: R.Dev, pipe: R.Pipeline, name: str) -> str:
    return hashlib.sha256(dev.read(pipe.b[name], pipe.size[name])).hexdigest()


def same_mode(dev: R.Dev, lanes: Lanes) -> None:
    """The gate for hash_fast.cuh: per shape, every step with sha = "ref" (s512_rows, s512_tile_leaves), the committed buffers'
    bytes recorded; then those buffers poisoned (0xA5) and every step again with sha = "fast", and poisoned once more and the
    served call (`serial+tt`, its tile tree too) captured and launched as a graph with sha = "fast": each buffer's bytes must
    equal the reference's both times."""
    for spec in os.environ.get("ETA_SHAPES", "32x8192x8192 80x8192x8192 176x8192x8192 128x8192x8192 112x384x1024 "
                               "8192x8192x8192").split():
        shape = R.parse_shape(spec)
        pipes, own = make(dev, shape, 1, None, 101)
        p = pipes[0]
        p.sha = "ref"
        for _, _, fn in runnable(p):
            fn()
        dev.sync()
        want = {nm: digest(dev, p, nm) for nm in COMMITTED}
        row = {"kind": "same", "shape": list(shape), "m_padded": p.m, "tiles": p.tiles}
        p.sha = "fast"
        for stage in ("steps", "graph"):
            names = COMMITTED if stage == "steps" else COMMITTED[:8]
            for nm in names:
                dev.ck("cuMemsetD8_v2", R.u64(p.b[nm]), 0xA5, p.size[nm])
            dev.sync()
            if stage == "steps":
                for _, _, fn in runnable(p):
                    fn()
            else:
                fns = steps_of(p)
                if "pad_t" in fns:   # the tile tree's pads: the unit's, not the call's
                    fns["pad_t"]()
                g = Graph(lanes, lambda: enqueue(lanes, [p], [fns], "serial+tt", 1))
                g.launch()
                dev.ck("cuStreamSynchronize", lanes.main)
                g.close()
            dev.sync()
            row[stage] = {nm: digest(dev, p, nm) == want[nm] for nm in names}
        row["identical"] = all(all(v.values()) for v in (row["steps"], row["graph"]))
        row["sha256"] = {nm: want[nm][:16] for nm in ("rowdig_a", "leaves", "tree_a", "tree_t")}
        emit(row)
        release(dev, pipes, own)


def sha2_mode(dev: R.Dev, lanes: Lanes) -> None:
    """hash_fast.cuh's kernels per launch-bound variant (ETA_PROBES: probe cubins built from hash_probe.cu, NAME=PATH each),
    then the ship's own, and the reference kernels, over each shape's A rows and its C~ and U: their time, digests against
    the reference's, and the compressions a second (the format's, 10 s + 2 + s / 2 a row or tile, and those executed with the
    headers precomputed, 9 s + 1 + s / 2)."""
    reps = int(os.environ.get("ETA_REPS", "15"))
    probes = [x.split("=", 1) for x in os.environ.get("ETA_PROBES", "").split()]
    mods = []
    for name, path in probes:
        pd = R.Dev(open(path, "rb").read())
        pd.open()
        pd.launch("s512_heads", R.cdiv(R.RS_MAX_SEGS * (R.RS_MAX_SEGS + 1) // 2 + R.RS_MAX_SEGS + 1, 128), 128)
        pd.sync()
        mods.append((name, pd, int(name.split("x")[0])))
    for spec in os.environ.get("ETA_SHAPES", "32x8192x8192 128x8192x8192 8192x8192x8192").split():
        shape = R.parse_shape(spec)
        pipes, own = make(dev, shape, 1, None, 101)
        p, b = pipes[0], pipes[0].b
        mr, n, k = p.m_real, p.n, p.k
        s = k // 256
        full, last = mr // R.TILE * (n // R.TILE), mr % R.TILE // 2
        units = {"rows": (mr, mr * s), "tiles": (p.tiles, 32 * full + n // R.TILE * last)}
        fmt = {"rows": mr * (10 * s + 2 + s // 2), "tiles": full * 338 + n // R.TILE * (10 * last + 2 + last // 2)}
        execd = {"rows": mr * (9 * s + 1 + s // 2), "tiles": full * 305 + n // R.TILE * (9 * last + 1 + last // 2)}
        p.sha = "ref"
        f = steps_of(p)
        f["rowdig_a"]()
        f["hash_leaf"]()
        dev.sync()
        want = {"rows": dev.read(b["rowdig_a"], 64 * mr), "tiles": dev.read(b["leaves"], 64 * p.tiles)}
        out = {"rows": b["rowdig_a"], "tiles": b["leaves"]}

        def variants():
            yield "ref", None, None
            yield "ship", None, None
            for nm, pd, nt in mods:
                yield nm, pd, nt

        for name, pd, nt in variants():
            for kind in ("rows", "tiles"):
                count, segs = units[kind]
                if pd is None:
                    p.sha = "fast" if name == "ship" else "ref"
                    both = f["rowdig_a"] if kind == "rows" else f["hash_leaf"]
                    parts = {"both": both}
                else:
                    seg = (lambda pd=pd, nt=nt: pd.launch("s512_row_segs", R.cdiv(segs, nt), nt, p=(b["xa"], b["segdig_a"]),
                                                          i=(mr, k), stream=lanes.main)) if kind == "rows" else \
                          (lambda pd=pd, nt=nt: pd.launch("s512_tile_segs", R.cdiv(segs, nt), nt, p=(b["c"], b["u"], b["segdig_t"]),
                                                          i=(mr, n), stream=lanes.main))
                    fold = (lambda pd=pd: pd.launch("s512_folds", R.cdiv(mr, 32), 32, p=(b["segdig_a"], b["rowdig_a"]),
                                                    i=(mr, s, mr, s), stream=lanes.main)) if kind == "rows" else \
                           (lambda pd=pd: pd.launch("s512_folds", R.cdiv(p.tiles, 32), 32, p=(b["segdig_t"], b["leaves"]),
                                                    i=(p.tiles, 32, full, last), stream=lanes.main))
                    parts = {"both": lambda seg=seg, fold=fold: (seg(), fold()), "segs": seg, "folds": fold}
                dev.ck("cuMemsetD8_v2", R.u64(out[kind]), 0xA5, len(want[kind]))
                lanes.on(lanes.main, parts["both"])
                dev.ck("cuStreamSynchronize", lanes.main)
                row = {"kind": "sha2", "variant": name, "unit": kind, "shape": list(shape), "units": count, "segments": segs,
                       "same": dev.read(out[kind], len(want[kind])) == want[kind], "compressions_format": fmt[kind],
                       "compressions_executed": execd[kind] if name != "ref" else fmt[kind]}
                for part, fn in parts.items():
                    g = Graph(lanes, lambda fn=fn: [lanes.on(lanes.main, fn) for _ in range(10)])
                    ms = statistics.median(timed(lanes, g.launch, reps)) / 10
                    g.close()
                    row[f"{part}_ms"] = round(ms, 5)
                sec = max(row["both_ms"], 1e-9) * 1e-3
                row["format_per_s"], row["executed_per_s"] = fmt[kind] / sec, row["compressions_executed"] / sec
                emit(row)
        release(dev, pipes, own)


#: FP8 E4M3 codes with the NaNs (0x7F, 0xFF) moved to the largest finite magnitudes
FINITE = bytes(v - 1 if v & 0x7F == 0x7F else v for v in range(256))


def pattern(name: str, n: int, honest: bytes) -> bytes:
    """n FP8 E4M3 operand bytes of a data pattern: `honest` is plainq's codes of gen_f32's rows (the served data)."""
    if name == "honest":
        return honest
    if name == "zeros":
        return bytes(n)
    if name == "ones":
        return b"\x38" * n
    if name == "alt_sign":
        return b"\x38\xb8" * (n // 2)
    rnd = os.urandom(n)
    if name == "uniform":
        return rnd.translate(FINITE)
    if name == "ternary":
        return rnd.translate(bytes((0x00, 0x38, 0xB8)[v % 3] for v in range(256)))
    if name == "onebit":
        return rnd.translate(bytes(1 << (v & 7) for v in range(256)))
    if name == "sparse8":
        return rnd.translate(bytes(FINITE[(37 * v + 11) & 0xFF] if v < 32 else 0 for v in range(256)))
    if name == "pow2":
        return rnd.translate(bytes(v & 0xF8 for v in range(256)))
    raise ValueError(name)


def data_mode(dev: R.Dev, nv: R.Nvml, lanes: Lanes) -> None:
    """gemm_fast (the plain dense FP8 GEMM) at ETA_DATA_SHAPE held ETA_HOLD_S per operand pattern (ETA_PATTERNS; `a_zero`
    zeros A alone), with NVML sampled every 20 ms: the SM clock the GPU picks for itself on that data, its power and throttle
    reasons, and the MAC rate.  An idle hold first (the board at the lock with nothing running).  Read-only: nothing here sets
    a clock or a power limit."""
    hold = float(os.environ.get("ETA_HOLD_S", "3"))
    shape = R.parse_shape(os.environ.get("ETA_DATA_SHAPE", "8192x8192x8192"))
    names = os.environ.get("ETA_PATTERNS", "honest zeros ones alt_sign ternary onebit pow2 sparse8 a_zero uniform honest").split()
    pipes, own = make(dev, shape, 1, None, 101)
    p, f = pipes[0], steps_of(pipes[0])
    m, n, k = shape
    info = limits(nv)
    idle = threading.Event()
    _, st = sampled(nv, lambda: (idle.wait(hold), 0.0)[1])
    emit({"kind": "data", "pattern": "idle", "shape": list(shape), **info, **st})
    qa, qb = dev.read(p.b["qa"], m * k), dev.read(p.b["qb"], n * k)
    for name in names:
        a = bytes(m * k) if name == "a_zero" else pattern(name, m * k, qa)
        b = qb if name == "a_zero" else pattern(name, n * k, qb)
        for buf, data in ((p.b["qa"], a), (p.b["qb"], b)):
            dev.ck("cuMemcpyHtoD_v2", R.u64(buf), C.c_char_p(data), len(data))
        dev.sync()
        t = timed(lanes, lambda: lanes.on(lanes.main, f["gemm_fast"]), 3)
        n_launch = max(1, int(hold * 1e3 / statistics.median(t)))
        ms, st = sampled(nv, lambda: timed(lanes, lambda: [lanes.on(lanes.main, f["gemm_fast"]) for _ in range(n_launch)], 1)[0])
        per = ms / n_launch
        macs = m * n * k / (per * 1e-3)
        row = {"kind": "data", "pattern": name, "shape": list(shape), "kernel": "gemm_fast", "launches": n_launch,
               "per_launch_ms": round(per, 5), "macs_per_s": macs, "tflops": 2 * macs / 1e12,
               "byte_hist_a": len(set(a[:1 << 16])), "nonzero_a": round(1 - a[:1 << 16].count(0) / (1 << 16), 4), **info, **st}
        if st["sm_mhz_median"]:
            row["eta_at_clock"] = round(macs / peak_macs(st["sm_mhz_median"], dev.sms), 5)
            row["macs_over_cap_peak"] = round(macs / peak_macs(CAP_MHZ, dev.sms), 5)
            row["energy_pj_per_mac"] = round(st["power_w_median"] / macs * 1e12, 4)
        emit(row)
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
     "sha": lambda: sha_mode(dev, lanes), "data": lambda: data_mode(dev, nv, lanes), "same": lambda: same_mode(dev, lanes),
     "sha2": lambda: sha2_mode(dev, lanes)}[mode]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
