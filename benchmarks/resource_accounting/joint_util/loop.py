"""The honest joint run's driver on one leased RTX PRO 6000 (sm_120): Pearl-C's call loop, the remote bandwidth audit's
chained prover passes, or both at once in one process on two streams.

- `pouw`: Pearl-C from a built ship (`benchmarks/pouw/pearl_c_sm120/build.sh`, its `run.py` imported from the ship): the
  check at the fixture's shape first (nothing is timed unless every buffer matches), then `--weights` pipelines at
  `--shape` (one weight each, as serving's linears; their B side set up once), and the call (commit, forming, clean-up,
  hashing: `run.measure`'s `call`) issued back to back on a non-blocking stream of its own, cycling over the weights,
  replayed as one CUDA graph of `--graph-calls` calls (`--no-graph` launches each kernel, as ncu wants).
- `audit`: `benchmarks/remote_session/device.py`'s DeviceStore over a store of random words (`pass_bench.py`'s: no
  codec, a pass's cost does not depend on the bytes), the honest prover's shape (chunk 2^18 blocks, h0 = 6, 128 sketch
  CTAs), a pass's root checked against the plain pass's first, then one window's chained passes (`run_chain`, each
  root copied to the host as the prover sends it) back to back on a stream of its own.
- `joint`: both in one process (the audit on a thread), started together, for the same `--seconds`.

The timed window is bracketed by cuProfilerStart/Stop, so `nsys --capture-range=cudaProfilerApi` and `ncu
--profile-from-start off` see that window only; `--count` stops a loop after that many calls or passes instead of
`--seconds`. `--ready`/`--go` hold the window until a file exists (two processes started together). NVML's SM clock,
power and throttle reasons are sampled through the window. One JSON object to `--out`.
"""
from __future__ import annotations

import argparse
import ctypes as C
import faulthandler
import gc
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RS = ROOT / "benchmarks" / "remote_session"


def wait_go(a) -> None:
    if a.ready:
        Path(a.ready).write_text("ready\n")
    if a.go:
        while not os.path.exists(a.go):
            time.sleep(0.002)


class Sampler:
    """NVML through the window: the SM clock, power and throttle reasons every `ms`."""

    def __init__(self, pc, uuid: str, ms: float = 50.0):
        self.nv, self.ms, self.rows, self.stop = pc.Nvml(uuid), ms, [], threading.Event()
        self.th = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        while not self.stop.is_set():
            self.rows.append(self.nv.read())
            self.stop.wait(self.ms / 1e3)

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.th.join()

    def summary(self) -> dict:
        rows = [r for r in self.rows if "sm_mhz" in r]
        if not rows:
            return {"nvml": "unavailable"}
        return {"samples": len(rows), "sm_mhz_median": statistics.median(r["sm_mhz"] for r in rows),
                "sm_mhz_min": min(r["sm_mhz"] for r in rows), "power_w_median": statistics.median(r["power_w"] for r in rows),
                "power_w_max": max(r["power_w"] for r in rows), "temp_c_max": max(r["temp_c"] for r in rows),
                "throttle": sorted({r["throttle"] for r in rows})}


def load_pc(ship: str):
    sys.path.insert(0, ship)
    import run as pc  # noqa: E402  (the ship's run.py)
    return pc


class Pouw:
    """Pearl-C's call loop on one stream of its own."""

    EXTRA = None

    def __init__(self, pc, ship: str, shape: tuple[int, int, int], weights: int, graph_calls: int, graph: bool, log):
        P, vp, u, i32, u64 = C.POINTER, C.c_void_p, C.c_uint, C.c_int, C.c_uint64
        self.pc, self.log = pc, log
        dev = self.dev = pc.Dev(open(os.path.join(ship, "pearl_c_sm120.cubin"), "rb").read())
        refused = dev.identity(os.environ.get("PEARLC_EXPECT_UUID"))
        if refused:
            raise SystemExit(f"REFUSED: {refused}")
        pc.bind(dev.cu, {"cuStreamCreate": [P(vp), u], "cuStreamBeginCapture_v2": [vp, i32],
                         "cuStreamEndCapture": [vp, P(vp)], "cuGraphInstantiateWithFlags": [P(vp), vp, u64],
                         "cuGraphLaunch": [vp, vp], "cuStreamSynchronize": [vp], "cuProfilerStart": [],
                         "cuProfilerStop": [], "cuCtxGetCurrent": [P(vp)]})
        dev.open()
        self.ctx = vp()
        dev.ck("cuCtxGetCurrent", C.byref(self.ctx))
        man_path = os.path.join(ship, "check", "check.json")
        t0 = time.time()
        res = pc.check(dev, json.load(open(man_path)), os.path.dirname(man_path), ["sm120"])["sm120"]
        self.check = {"shape": json.load(open(man_path))["shape"], "buffers": res["buffers"], "failed": res["failed"],
                      "pearl_c_ok": res["pearl_c_ok"], "seconds": round(time.time() - t0, 2)}
        log(f"pouw: check {self.check}")
        if not res["pearl_c_ok"]:
            raise SystemExit(f"Pearl-C's check failed: {res['failed']}")
        m_real, n, k = shape
        m = pc.cdiv(m_real, pc.BM_D) * pc.BM_D
        self.shape, self.m, self.weights = shape, m, weights
        self.keys = {nm: dev.put(hashlib.shake_256(f"pearl-c-timing/{nm}".encode()).digest(size))
                     for nm, size in pc.TIMING_KEYS.items()}
        self.pipes, self.calls = [], []
        for w in range(weights):
            xa, xb = dev.alloc(m * 4 * k), dev.alloc(n * 4 * k)
            for x, rows, seed in ((xa, m_real, 11 + 2 * w), (xb, n, 12 + 2 * w)):
                dev.launch("gen_f32", pc.cdiv(rows * k, 256), 256, p=(x,), i=(rows, k, seed))
            p = pc.Pipeline(dev, m, n, k, "sm120", self.keys, xa, xb, m_real=m_real, weight=w)
            steps = p.steps()
            for _, phase, f in steps:
                if phase in ("epoch", "unit", "job"):
                    f()
            call = [f for _, phase, f in steps if phase in ("commit", "forming", "cleanup", "hashing")]
            for f in call:
                f()
            dev.sync()
            self.pipes.append(p)
            self.calls.append(call)
        stream = vp()
        dev.ck("cuStreamCreate", C.byref(stream), 1)
        self.stream = dev.stream = stream
        self.launches_per_call = None
        self.graph_calls = max(weights, graph_calls // weights * weights) if graph else weights
        self.exec = None
        if graph:
            dev.ck("cuStreamBeginCapture_v2", stream, 2)
            self.issue(self.graph_calls)
            g = vp()
            dev.ck("cuStreamEndCapture", stream, C.byref(g))
            ex = vp()
            dev.ck("cuGraphInstantiateWithFlags", C.byref(ex), g, 0)
            self.exec = ex
        self.events = []
        for _ in range(4):
            e = vp()
            dev.ck("cuEventCreate", C.byref(e), 0)
            self.events.append(e)
        for _ in range(3):
            self.step()
        dev.ck("cuStreamSynchronize", stream)
        self.done = 0

    def issue(self, calls: int) -> None:
        for j in range(calls):
            for f in self.calls[j % self.weights]:
                f()

    def step(self) -> int:
        """One graph (or `weights` calls without a graph); the calls it issues."""
        if self.exec is not None:
            self.dev.ck("cuGraphLaunch", self.exec, self.stream)
            return self.graph_calls
        self.issue(self.weights)
        return self.weights

    def loop(self, seconds: float, count: int, started: threading.Barrier | None) -> dict:
        dev, ev = self.dev, self.events
        dev.ck("cuCtxSetCurrent", self.ctx)
        if started is not None:
            started.wait()
        dev.ck("cuEventRecord", ev[0], self.stream)
        t0, calls, i, ring = time.perf_counter(), 0, 0, ev[2:]
        while (calls < count) if count else (time.perf_counter() - t0 < seconds):
            calls += self.step()
            dev.ck("cuEventRecord", ring[i % 2], self.stream)
            if i:
                dev.ck("cuEventSynchronize", ring[(i - 1) % 2])
                self.done = calls - self.graph_calls
            i += 1
        dev.ck("cuEventRecord", ev[1], self.stream)
        dev.ck("cuEventSynchronize", ev[1])
        wall = time.perf_counter() - t0
        ms = C.c_float()
        dev.ck("cuEventElapsedTime", C.byref(ms), ev[0], ev[1])
        return {"calls": calls, "gpu_ms": ms.value, "wall_s": wall, "calls_per_s": calls / (ms.value / 1e3),
                "call_ms": ms.value / calls, "graph_calls": self.graph_calls if self.exec is not None else None,
                "weights": self.weights, "shape": list(self.shape), "m_padded": self.m}

    def profiler(self, on: bool) -> None:
        self.dev.ck("cuProfilerStart" if on else "cuProfilerStop")


class Audit:
    """The bandwidth audit's chained prover passes on one stream of its own (and its side stream)."""

    def __init__(self, lib_path: str, gib: float, ctas: int, chunk_log2: int, kept_from: int, side_priority: int | None,
                 log):
        import torch

        sys.path.insert(0, str(RS))
        import device  # noqa: E402
        from verity.core.protocols.remote_bandwidth import receipts as rb  # noqa: E402

        self.torch, self.device, self.rb, self.log = torch, device, rb, log
        lib = device.Lib(lib_path, torch)
        self._copy = tempfile.NamedTemporaryFile(suffix=".so")
        shutil.copyfile(lib_path, self._copy.name)
        lib_open = device.Lib(self._copy.name, torch)
        n = int(gib * 2 ** 30) // device.BLOCK
        blocks = n // (1 << chunk_log2) * (1 << chunk_log2)
        self.stream = torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            g = torch.Generator(device="cuda")
            g.manual_seed(1)
            self.store = s = device.DeviceStore(torch, lib, lib_open, None, blocks, b"k" * 32, g, chunk=1 << chunk_log2,
                                                kept_from=kept_from, sketch_ctas=ctas)
            if side_priority is not None:
                s.side = torch.cuda.Stream(priority=side_priority)
            seed = rb.seed(b"\1" * 32, 0, bytes(64))
            root, _ = s.run_pass(seed)
            plain = s.plain_root(seed)
        torch.cuda.synchronize()
        self.gate = {"pass_root_vs_plain": root == plain, "blocks": blocks, "bytes": blocks * device.BLOCK,
                     "chunk_log2": chunk_log2, "h0": s.h0, "sketch_ctas": ctas, "fused_consumer": s.h0 <= 8,
                     "side_priority": side_priority if side_priority is not None else -1}
        self.done = 0
        log(f"audit: gate {self.gate}")
        if not self.gate["pass_root_vs_plain"]:
            raise SystemExit("the pipelined pass's root is not the plain pass's")
        torch.cuda.empty_cache()
        with torch.cuda.stream(self.stream):
            s.start_chain(b"\5" * 32, bytes(64))
            for i in range(5):
                s.run_chain(i)
        s.pass_gpu_ms.clear()
        s.chain_host_phases.clear()

    def loop(self, seconds: float, count: int, started: threading.Barrier | None) -> dict:
        torch, s = self.torch, self.store
        with torch.cuda.stream(self.stream):
            s.start_chain(b"\6" * 32, bytes(64))
            if started is not None:
                started.wait()
            t0, i = time.perf_counter(), 0
            while (i < count) if count else (time.perf_counter() - t0 < seconds):
                s.run_chain(i)
                i += 1
                self.done = i
            wall = time.perf_counter() - t0
        gpu = sorted(s.pass_gpu_ms[-i:])
        host = sorted(x[2] for x in s.chain_host_phases[-i:])
        s.pass_gpu_ms.clear()
        s.chain_host_phases.clear()
        nbytes = self.gate["bytes"]
        return {"passes": i, "wall_s": wall, "passes_per_s": i / wall, "pass_ms_wall": wall * 1e3 / i,
                "pass_gpu_ms_p50": gpu[len(gpu) // 2], "pass_gpu_ms_max": gpu[-1], "pass_gpu_ms_min": gpu[0],
                "wait_ns_p50": host[len(host) // 2], "store_bytes": nbytes, "read_bytes_per_s": nbytes * i / wall}

    def profiler(self, on: bool) -> None:
        (self.torch.cuda.profiler.start if on else self.torch.cuda.profiler.stop)()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("pouw", "audit", "joint"), required=True)
    ap.add_argument("--ship", required=True, help="Pearl-C's built ship (its run.py, cubin and check/)")
    ap.add_argument("--lib", help="session.so (audit, joint)")
    ap.add_argument("--shape", default="128x8192x8192")
    ap.add_argument("--weights", type=int, default=4)
    ap.add_argument("--graph-calls", type=int, default=16)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--gib", type=float, default=16.0)
    ap.add_argument("--sketch-ctas", type=int, default=128)
    ap.add_argument("--chunk-log2", type=int, default=18)
    ap.add_argument("--kept-from", type=int, default=6, help="h0; above 8 the pass's tree takes the unfused path")
    ap.add_argument("--side-priority", type=int, help="the pass's side stream's priority (device.py's: -1)")
    ap.add_argument("--watchdog", type=float, default=0.0,
                    help="seconds past the window after which every thread's stack is dumped and the process exits")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--count", type=int, default=0, help="calls or passes instead of --seconds")
    ap.add_argument("--ready")
    ap.add_argument("--go")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    pc = load_pc(a.ship)
    out: dict = {"mode": a.mode, "args": vars(a), "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "uuid": os.environ.get("GPU_LEASE_UUID")}
    audit = pouw = None
    if a.mode in ("audit", "joint"):
        import torch

        torch.cuda.init()
        audit = Audit(a.lib, a.gib, a.sketch_ctas, a.chunk_log2, a.kept_from, a.side_priority, log)
        out["audit_gate"] = audit.gate
    if a.mode in ("pouw", "joint"):
        pouw = Pouw(pc, a.ship, pc.parse_shape(a.shape), a.weights, a.graph_calls, not a.no_graph, log)
        out["pouw_check"] = pouw.check
    gc.collect()
    gc.freeze()
    gc.disable()
    wait_go(a)
    prof = pouw or audit
    if a.watchdog:
        faulthandler.dump_traceback_later(a.seconds + a.watchdog, exit=True)
    stop = threading.Event()

    def progress() -> None:
        t0 = time.perf_counter()
        while not stop.wait(1.0):
            log(json.dumps({"t": round(time.perf_counter() - t0, 1), "calls": getattr(pouw, "done", None),
                            "passes": getattr(audit, "done", None)}))
    threading.Thread(target=progress, daemon=True).start()
    with Sampler(pc, os.environ.get("GPU_LEASE_UUID") or os.environ.get("PEARLC_EXPECT_UUID") or "") as smp:
        prof.profiler(True)
        if a.mode == "joint":
            started, got = threading.Barrier(2), {}
            th = threading.Thread(target=lambda: got.update(audit=audit.loop(a.seconds, a.count, started)))
            th.start()
            out["pouw"] = pouw.loop(a.seconds, a.count, started)
            th.join()
            out["audit"] = got["audit"]
        elif a.mode == "pouw":
            out["pouw"] = pouw.loop(a.seconds, a.count, None)
        else:
            out["audit"] = audit.loop(a.seconds, a.count, None)
        prof.profiler(False)
    stop.set()
    faulthandler.cancel_dump_traceback_later()
    gc.enable()
    out["nvml"] = smp.summary()
    out["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.out).write_text(json.dumps(out, indent=1))
    log(json.dumps({k: out.get(k) for k in ("mode", "pouw", "audit", "nvml")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
