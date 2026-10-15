#!/usr/bin/env python3
"""Measurement A of the PoUW capacity study: the FP8 tensor throughput an RTX PRO 6000 (sm_120) sustains beside a Pearl-C
call, and what that does to the call's wall time.

Run from a ship directory of `benchmarks/pouw/pearl_c_sm120/build.sh` (its `run.py`, cubin and `check/`), with torch:

  PEARLC_EXPECT_UUID=... python corun.py check/check.json > corun.jsonl

1. The ship's fixture check (`run.check`, variant `sm120`) on this card; nothing is timed unless it passes.
2. For each Pearl-C shape (m_real x n x k, CORUN_SHAPES), the served call (commit, forming, clean-up with the chain, hashing:
   `run.measure`'s `call`) captured as one CUDA graph on a non-blocking stream P at the highest priority. The graph's outputs
   (C~, U, y, the tile tree) are held byte for byte to the eager call's, before and after co-running.
3. Co-runners, each one CUDA graph of R FP8 GEMMs replayed on a non-blocking stream Q at the lowest priority (or P's, for
   `eq`): cuBLASLt through torch._scaled_mm (E4M3 x E4M3 -> BF16) at MxNxK, and the ship's own `gemm_fast` (persistent, its
   grid an SM count) at 8192^3.
4. Windows, alternated: P alone, Q alone, P with Q, P alone, P with Q, P alone. P with Q enqueues Q's replays first (each
   followed by an event), makes P wait on Q's LEAD-th event so Q is in steady state, then enqueues the calls with an event
   after each. Q's work inside P's window [p_0, p_N] is its cumulative GEMM count interpolated at both ends from Q's
   events (cuEventElapsedTime against p_0). Valid only if Q's last event is after p_N.

One JSON object per line on stdout. Peak and fractions are at the SM clock NVML reads around each window (188 SMs x 1,024
E4M3 MACs an SM a clock).
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

sys.path.insert(0, os.getcwd())
import run as R  # noqa: E402  (the ship's run.py)

import torch  # noqa: E402

u, i32, vp, P_ = C.c_uint, C.c_int, C.c_void_p, C.POINTER
EXTRA = {
    "cuStreamCreateWithPriority": [P_(vp), u, i32], "cuCtxGetStreamPriorityRange": [P_(i32), P_(i32)],
    "cuStreamBeginCapture_v2": [vp, i32], "cuStreamEndCapture": [vp, P_(vp)],
    "cuGraphInstantiateWithFlags": [P_(vp), vp, C.c_ulonglong], "cuGraphLaunch": [vp, vp],
    "cuStreamSynchronize": [vp], "cuStreamWaitEvent": [vp, vp, u],
}
NON_BLOCKING, CAPTURE_RELAXED = 1, 2
MACS_PER_SM_CLOCK = 1024
LEAD = 2


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


class Sampler:
    """NVML's SM clock and power every 5 ms while a window runs (NVML reads, no device work)."""

    def __init__(self, nv: R.Nvml):
        self.nv, self.rows, self.stop = nv, [], threading.Event()

    def __enter__(self):
        self.rows, self.stop = [], threading.Event()
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()
        return self

    def loop(self):
        while not self.stop.is_set():
            r = self.nv.read()
            if "sm_mhz" in r:
                self.rows.append((r["sm_mhz"], r["power_w"], r["throttle"]))
            self.stop.wait(0.005)

    def __exit__(self, *a):
        self.stop.set()
        self.t.join()

    def summary(self) -> dict:
        if not self.rows:
            return {}
        mhz = [r[0] for r in self.rows]
        pw = [r[1] for r in self.rows]
        return {"samples": len(mhz), "sm_mhz_min": min(mhz), "sm_mhz_med": statistics.median(mhz), "sm_mhz_max": max(mhz),
                "power_w_med": statistics.median(pw), "power_w_max": max(pw),
                "throttle": sorted({r[2] for r in self.rows})}


class Bench:
    def __init__(self, dev: R.Dev, nv: R.Nvml):
        self.dev, self.nv, self.cu = dev, nv, dev.cu
        R.bind(self.cu, EXTRA)
        lo, hi = i32(), i32()
        dev.ck("cuCtxGetStreamPriorityRange", C.byref(lo), C.byref(hi))
        self.prio_low, self.prio_high = lo.value, hi.value
        self.P = self.stream(self.prio_high)
        self.Q = self.stream(self.prio_low)
        self.Qeq = self.stream(self.prio_high)
        self.pool: list[vp] = []

    def stream(self, prio: int) -> vp:
        s = vp()
        self.dev.ck("cuStreamCreateWithPriority", C.byref(s), NON_BLOCKING, prio)
        return s

    def events(self, n: int) -> list[vp]:
        while len(self.pool) < n:
            e = vp()
            self.dev.ck("cuEventCreate", C.byref(e), 0)
            self.pool.append(e)
        return self.pool[:n]

    def ms(self, a: vp, b: vp) -> float:
        x = C.c_float()
        self.dev.ck("cuEventElapsedTime", C.byref(x), a, b)
        return x.value

    def capture(self, body) -> vp:
        """body's launches on P as one instantiated graph."""
        d = self.dev
        d.stream = self.P
        d.ck("cuStreamBeginCapture_v2", self.P, CAPTURE_RELAXED)
        body()
        g, ex = vp(), vp()
        d.ck("cuStreamEndCapture", self.P, C.byref(g))
        d.ck("cuGraphInstantiateWithFlags", C.byref(ex), g, 0)
        d.stream = None
        return ex

    def sync(self):
        self.dev.ck("cuCtxSynchronize")


class TorchGemm:
    """R cuBLASLt FP8 GEMMs (torch._scaled_mm, E4M3 x E4M3 -> BF16) at M x N x K, one torch CUDA graph."""

    def __init__(self, M: int, N: int, K: int, reps: int, fast_accum: bool):
        self.shape, self.reps, self.flop = (M, N, K), reps, 2 * M * N * K * reps
        g = torch.Generator(device="cuda").manual_seed(M * 7 + N * 3 + K)
        self.a = (torch.randn(M, K, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
        self.b = (torch.randn(N, K, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
        self.sa = torch.tensor(1.0, device="cuda")
        self.sb = torch.tensor(1.0, device="cuda")
        self.fast = fast_accum
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                self.one()
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            for _ in range(reps):
                self.out = self.one()
        torch.cuda.synchronize()
        self.label = f"cublaslt {M}x{N}x{K}" + (" fast" if fast_accum else "")

    def one(self):
        return torch._scaled_mm(self.a, self.b.t(), scale_a=self.sa, scale_b=self.sb, out_dtype=torch.bfloat16,
                                use_fast_accum=self.fast)

    def launch(self, stream: vp) -> None:
        with torch.cuda.stream(torch.cuda.ExternalStream(stream.value)):
            self.g.replay()


class ShipGemm:
    """R launches of the ship's gemm_fast (the plain E4M3 GEMM, unpromoted, persistent over `grid` CTAs) at 8192^3, one
    driver graph."""

    def __init__(self, bench: Bench, pipe: R.Pipeline, grid: int, reps: int):
        self.bench, self.reps, self.grid = bench, reps, grid
        m, n, k = pipe.m, pipe.n, pipe.k
        self.flop = 2 * m * n * k * reps
        b = pipe.b
        body = lambda: [pipe.gemm("gemm_fast", "qa", "qb", m, n, grid=grid, out=(None, None, "y_fast"), rs=b["sa"],  # noqa: E731
                                  cs=b["sb"]) for _ in range(reps)]
        self.ex = bench.capture(body)
        self.label = f"ship gemm_fast {m}x{n}x{k} grid {grid}"

    def launch(self, stream: vp) -> None:
        self.bench.dev.ck("cuGraphLaunch", self.ex, stream)


def window(bench: Bench, call_ex: vp | None, n_calls: int, co=None, q: vp | None = None, n_co: int = 0) -> dict:
    """One timed window; returns per-call ms, P's span and Q's work inside it."""
    d = bench.dev
    eq = bench.events(n_co + 1 + n_calls + 1)
    ev_q, ev_p = eq[:n_co + 1], eq[n_co + 1:]
    sampler = Sampler(bench.nv)
    with sampler:
        if co is not None:
            d.ck("cuEventRecord", ev_q[0], q)
            for i in range(n_co):
                co.launch(q)
                d.ck("cuEventRecord", ev_q[i + 1], q)
        if call_ex is not None:
            if co is not None:
                d.ck("cuStreamWaitEvent", bench.P, ev_q[min(LEAD, n_co)], 0)
            d.ck("cuEventRecord", ev_p[0], bench.P)
            for j in range(n_calls):
                d.ck("cuGraphLaunch", call_ex, bench.P)
                d.ck("cuEventRecord", ev_p[j + 1], bench.P)
        bench.sync()
    out: dict = {"clk": sampler.summary()}
    if call_ex is not None:
        calls = [bench.ms(ev_p[j], ev_p[j + 1]) for j in range(n_calls)]
        span = bench.ms(ev_p[0], ev_p[n_calls])
        out.update({"call_ms_med": statistics.median(calls), "call_ms_mean": span / n_calls, "span_ms": span,
                    "call_ms_p10": sorted(calls)[n_calls // 10], "call_ms_p90": sorted(calls)[(9 * n_calls) // 10]})
    if co is not None:
        if call_ex is None:
            span = bench.ms(ev_q[1], ev_q[n_co])
            out.update({"co_span_ms": span, "co_tflops": co.flop * (n_co - 1) / span / 1e9})
        else:
            t = [bench.ms(ev_p[0], e) for e in ev_q]           # Q's events relative to p_0 (negative before)
            W = out["span_ms"]

            def done(x):                                       # Q's replays completed at time x, interpolated
                for i in range(1, len(t)):
                    if t[i] >= x:
                        return (i - 1) + (x - t[i - 1]) / (t[i] - t[i - 1])
                return float("nan")
            work = done(W) - done(0.0)
            out.update({"co_valid": t[-1] > W and t[min(LEAD, n_co)] <= 1e-3, "co_replays_in_window": work,
                        "co_tflops": co.flop * work / W / 1e9 if work == work else None, "co_tail_ms": t[-1] - W})
    return out


def peak_tflops(sms: int, mhz: float) -> float:
    return 2 * sms * MACS_PER_SM_CLOCK * mhz * 1e6 / 1e12


def call_steps(pipe: R.Pipeline):
    return [f for _, p, f in pipe.steps() if p in ("commit", "forming", "cleanup", "hashing")]


def snapshot(pipe: R.Pipeline) -> str:
    h = hashlib.sha256()
    for name in ("c", "u", "y", "leaves"):
        h.update(pipe.dev.read(pipe.b[name], pipe.size[name]))
    h.update(pipe.dev.read(pipe.root("t"), R.NB))
    return h.hexdigest()


def clobber(pipe: R.Pipeline) -> None:
    for name in ("c", "u", "y", "leaves", "tree_t"):
        pipe.dev.ck("cuMemsetD8_v2", R.u64(pipe.b[name]), 0xAB, pipe.size[name])


def main(argv: list[str]) -> int:
    man_path = argv[1] if len(argv) > 1 else "check/check.json"
    man = json.load(open(man_path))
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    image = open("pearl_c_sm120.cubin", "rb").read()
    dev = R.Dev(image)
    refused = dev.identity(os.environ.get("PEARLC_EXPECT_UUID"))
    nv = R.Nvml(dev.uuid)
    emit({"kind": "env", "device": dev.name, "uuid": dev.uuid, "sms": dev.sms, "cc": list(dev.cc), "torch": torch.__version__,
          "torch_cuda": torch.version.cuda, "cubin_sha256": hashlib.sha256(image).hexdigest(), "clk": nv.read(),
          "refused": refused, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if refused:
        print("REFUSED: " + "; ".join(refused), file=sys.stderr)
        return 3
    dev.open()
    res = R.check(dev, man, os.path.dirname(man_path), ["sm120"])
    ok = res["sm120"]["pearl_c_ok"] and res["sm120"]["plain_ok"]
    emit({"kind": "check", "ok": ok, "failed": res["sm120"]["failed"]})
    if not ok:
        return 4
    bench = Bench(dev, nv)
    emit({"kind": "streams", "prio_low": bench.prio_low, "prio_high": bench.prio_high})

    # the co-runners
    corunners = []
    for spec in os.environ.get("CORUN_TORCH", "8192x8192x8192 8192x8192x2048 8192x8192x1024").split():
        M, N, K = (int(x) for x in spec.split("x"))
        reps = max(1, int(round(4e-3 / (2 * M * N * K / 7.5e14))))     # about 4 ms a replay
        try:
            corunners.append(TorchGemm(M, N, K, reps, fast_accum=bool(int(os.environ.get("CORUN_FAST_ACCUM", "0")))))
        except RuntimeError as e:
            emit({"kind": "co_error", "co": spec, "error": str(e)[:400]})
    tk = {nm: dev.put(hashlib.shake_256(f"pearl-c-timing/{nm}".encode()).digest(sz)) for nm, sz in R.TIMING_KEYS.items()}
    big = 8192
    xa, xb = dev.alloc(big * 4 * big), dev.alloc(big * 4 * big)
    for x, seed in ((xa, 11), (xb, 12)):
        dev.launch("gen_f32", R.cdiv(big * big, 256), 256, p=(x,), i=(big, big, seed))
    host = R.Pipeline(dev, big, big, big, "sm120", tk, xa, xb)
    host.run_all()
    for grid in [int(g) for g in os.environ.get("CORUN_SHIP_GRIDS", "188 140 94").split()]:
        corunners.append(ShipGemm(bench, host, min(grid, dev.sms), 1))
    co_alone = {}
    for co in corunners:
        a = window(bench, None, 0, co, bench.Q, 30)
        co_alone[co.label] = a["co_tflops"]
        emit({"kind": "co_alone", "co": co.label, "reps_per_replay": co.reps, **a,
              "peak_tflops_at_clk": peak_tflops(dev.sms, a["clk"].get("sm_mhz_med", 0))})

    n, k = 8192, 8192
    for m_real in [int(s) for s in os.environ.get("CORUN_SHAPES", "1 32 128 512 2048 8192").split()]:
        m = R.cdiv(m_real, R.BM_D) * R.BM_D
        pipe = R.Pipeline(dev, m, n, k, "sm120", tk, xa, xb, m_real=m_real)
        pipe.run_all()
        eager = snapshot(pipe)
        clobber(pipe)
        call_ex = bench.capture(lambda: [f() for f in call_steps(pipe)])
        dev.ck("cuGraphLaunch", call_ex, bench.P)
        bench.sync()
        graphed = snapshot(pipe)
        alone = window(bench, call_ex, 20)
        n_calls = max(30, min(400, int(150 / alone["call_ms_med"])))
        emit({"kind": "shape", "m_real": m_real, "m": m, "n": n, "k": k, "graph_equals_eager": graphed == eager,
              "n_calls": n_calls, "useful_macs_per_call": m_real * n * k})
        for co in corunners:
            for q, qname in ((bench.Q, "low"), (bench.Qeq, "eq")):
                if qname == "eq" and not isinstance(co, TorchGemm):
                    continue
                rows, unchanged = [], True
                for phase in ("alone", "co", "alone", "co", "alone"):
                    if phase == "alone":
                        w = window(bench, call_ex, n_calls)
                    else:
                        co_ms = 4.0 if isinstance(co, TorchGemm) else 2 * 2 * 8192 ** 3 / 7.5e14 * 1e3
                        n_co = int(3 * n_calls * alone["call_ms_med"] / co_ms) + 2 * LEAD + 4
                        clobber(pipe)
                        w = window(bench, call_ex, n_calls, co, q, min(n_co, 600))
                        unchanged = unchanged and snapshot(pipe) == eager
                    rows.append({"phase": phase, **w})
                al = [r for r in rows if r["phase"] == "alone"]
                cr = [r for r in rows if r["phase"] == "co"]
                base = statistics.mean(r["call_ms_mean"] for r in al)
                noise = (max(r["call_ms_mean"] for r in al) - min(r["call_ms_mean"] for r in al)) / base
                mhz = statistics.median([r["clk"].get("sm_mhz_med", 0) for r in rows])
                pk = peak_tflops(dev.sms, mhz)
                summary = {"kind": "corun", "m_real": m_real, "co": co.label, "q_prio": qname,
                           "call_ms_alone": base, "alone_spread": noise,
                           "call_ms_co": statistics.mean(r["call_ms_mean"] for r in cr),
                           "slowdown": statistics.mean(r["call_ms_mean"] for r in cr) / base - 1,
                           "co_tflops": statistics.mean(r["co_tflops"] or 0 for r in cr),
                           "co_valid": all(r.get("co_valid") for r in cr), "sm_mhz": mhz, "peak_tflops": pk,
                           "pearlc_tflops_alone": 2 * m_real * n * k / base / 1e9,
                           "eta_alone": 2 * m_real * n * k / base / 1e9 / pk,
                           "outputs_unchanged_under_corun": unchanged, "windows": rows}
                summary["co_share_of_peak"] = summary["co_tflops"] / pk
                summary["co_share_of_alone"] = summary["co_tflops"] / co_alone[co.label]
                emit(summary)
                print(f"m={m_real} {co.label} q={qname}: call {base:.4f} ms alone (spread {noise:.3%}), "
                      f"{summary['call_ms_co']:.4f} with ({summary['slowdown']:+.2%}); co-runner {summary['co_tflops']:.0f} "
                      f"TFLOPS = {summary['co_share_of_peak']:.3f} of peak {pk:.0f}; valid {summary['co_valid']}",
                      file=sys.stderr, flush=True)
        pipe.free()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
