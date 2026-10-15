#!/usr/bin/env python3
"""Measurement A2 of the PoUW capacity study: the Pearl-C call on X SMs of an RTX PRO 6000 (sm_120) and an FP8 GEMM on the
other 188 - X, as two CUDA green contexts (disjoint SM partitions of one device, cuDevSmResourceSplitByCount).

Run from a ship directory of `benchmarks/pouw/pearl_c_sm120/build.sh`:

  PEARLC_EXPECT_UUID=... python3 green.py check/check.json > green.jsonl

1. The ship's fixture check (`run.check`, variant `sm120`) in the primary context; nothing is timed unless it passes. Then,
   per shape, the eager call's outputs (C~, U, y, the tile leaves and the tile tree's root) in the primary context, and the
   graphed call's span there (the full-GPU reference).
2. For each X in GREEN_SMS: green context A (X SMs) holds the Pearl-C pipeline, its call captured as one graph on a stream at
   the highest priority; green context B (the remaining SMs) holds the ship's gemm_fast at 8192^3 (plain E4M3, unpromoted,
   persistent over B's SMs), one graph a replay. Windows: A alone, B alone, A with B (twice), A alone. A with B enqueues
   B's replays first, A waits on B's LEAD-th replay (a cross-context event wait), then the calls. Every window boundary
   and replay is a one-thread `stamp` kernel writing %globaltimer (ns, one clock for both contexts), so B's work inside A's
   window is its cumulative replay count interpolated at A's first and last stamps.
3. The graphed call's outputs in A, after co-running, are held byte for byte to the primary context's eager ones.

One JSON object per line on stdout; a summary line per (shape, X) on stderr.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import run as R  # noqa: E402  (the ship's run.py)

u, i32, vp, P_ = C.c_uint, C.c_int, C.c_void_p, C.POINTER
EXTRA = {
    "cuStreamBeginCapture_v2": [vp, i32], "cuStreamEndCapture": [vp, P_(vp)],
    "cuGraphInstantiateWithFlags": [P_(vp), vp, C.c_ulonglong], "cuGraphLaunch": [vp, vp], "cuGraphExecDestroy": [vp],
    "cuStreamSynchronize": [vp], "cuStreamWaitEvent": [vp, vp, u], "cuCtxGetStreamPriorityRange": [P_(i32), P_(i32)],
    "cuDeviceGetDevResource": [i32, vp, i32], "cuDevSmResourceSplitByCount": [vp, P_(u), vp, vp, u, u],
    "cuDevResourceGenerateDesc": [P_(vp), vp, u], "cuGreenCtxCreate": [P_(vp), vp, i32, u], "cuGreenCtxDestroy": [vp],
    "cuCtxFromGreenCtx": [P_(vp), vp], "cuGreenCtxStreamCreate": [P_(vp), vp, u, i32],
    "cuStreamCreateWithPriority": [P_(vp), u, i32],
}
NON_BLOCKING, CAPTURE_RELAXED, RES_SM, GREEN_DEFAULT_STREAM = 1, 2, 1, 1
RES_BYTES, RES_SM_OFFSET = 144, 96       # CUdevResource (RESOURCE_ABI_VERSION 1): type, 92 bytes, a 48-byte union
LEAD = 2
STAMP_PTX = b"""
.version 8.0
.target sm_90
.address_size 64
.visible .entry stamp(.param .align 8 .b8 a[192])
{
  .reg .b64 %rd<6>;
  .reg .b32 %r<2>;
  ld.param.u64 %rd1, [a];
  ld.param.u32 %r1, [a+128];
  cvta.to.global.u64 %rd2, %rd1;
  mul.wide.u32 %rd3, %r1, 8;
  add.s64 %rd4, %rd2, %rd3;
  mov.u64 %rd5, %globaltimer;
  st.global.u64 [%rd4], %rd5;
  ret;
}
\x00"""
MAX_STAMPS = 4096


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def sm_info(raw: C.Array) -> dict:
    sm_count, min_part, align = (int.from_bytes(bytes(raw[RES_SM_OFFSET + 4 * j:RES_SM_OFFSET + 4 * j + 4]), "little") for j in range(3))
    return {"sm_count": sm_count, "min_partition": min_part, "coscheduled_alignment": align}


class Side:
    """One context (primary or green) with its own copy of the ship's module, a stamp kernel, a stream and a stamp buffer.
    `sms` is the partition's SM count, so the ship's persistent GEMMs launch one CTA per SM of it."""

    def __init__(self, dev: R.Dev, ctx: vp, sms: int, green: vp | None, prio: int):
        self.dev, self.ctx, self.green = dev, ctx, green
        R.bind(dev.cu, EXTRA)
        self.gemm_smem: dict[str, int] = {}     # the GEMMs' dynamic shared memory, set on this module's functions
        self.cur()
        dev.mod = vp()
        dev.ck("cuModuleLoadData", C.byref(dev.mod), dev.cubin)
        dev.fns = {}
        dev.sms = sms
        self.smod, self.sfn = vp(), vp()
        dev.ck("cuModuleLoadData", C.byref(self.smod), STAMP_PTX)
        dev.ck("cuModuleGetFunction", C.byref(self.sfn), self.smod, b"stamp")
        self.stream = vp()
        if green is not None:
            dev.ck("cuGreenCtxStreamCreate", C.byref(self.stream), green, NON_BLOCKING, prio)
        else:
            dev.ck("cuStreamCreateWithPriority", C.byref(self.stream), NON_BLOCKING, prio)
        self.stamps = dev.alloc(8 * MAX_STAMPS)
        self.ev: list[vp] = []

    def cur(self) -> None:
        self.dev.ck("cuCtxSetCurrent", self.ctx)

    def stamp(self, idx: int) -> None:
        a = R.Args()
        a.p[0], a.i[0] = self.stamps, idx
        params = (vp * 1)(C.addressof(a))
        self.dev.ck("cuLaunchKernel", self.sfn, 1, 1, 1, 1, 1, 1, 0, self.stream, params, None)

    def read_stamps(self, n: int) -> list[int]:
        raw = self.dev.read(self.stamps, 8 * n)
        return [int.from_bytes(raw[8 * j:8 * j + 8], "little") for j in range(n)]

    def events(self, n: int) -> list[vp]:
        while len(self.ev) < n:
            e = vp()
            self.dev.ck("cuEventCreate", C.byref(e), 0)
            self.ev.append(e)
        return self.ev[:n]

    def capture(self, body) -> vp:
        d = self.dev
        d.stream = self.stream
        d.ck("cuStreamBeginCapture_v2", self.stream, CAPTURE_RELAXED)
        body()
        g, ex = vp(), vp()
        d.ck("cuStreamEndCapture", self.stream, C.byref(g))
        d.ck("cuGraphInstantiateWithFlags", C.byref(ex), g, 0)
        d.stream = None
        return ex

    def launch_graph(self, ex: vp) -> None:
        self.dev.ck("cuGraphLaunch", ex, self.stream)

    def sync(self) -> None:
        self.dev.ck("cuStreamSynchronize", self.stream)


def keys_on(dev: R.Dev) -> dict[str, int]:
    return {nm: dev.put(hashlib.shake_256(f"pearl-c-timing/{nm}".encode()).digest(sz)) for nm, sz in R.TIMING_KEYS.items()}


def inputs_on(dev: R.Dev, m: int, n: int, k: int) -> tuple[int, int]:
    xa, xb = dev.alloc(m * 4 * k), dev.alloc(n * 4 * k)
    dev.launch("gen_f32", R.cdiv(m * k, 256), 256, p=(xa,), i=(m, k, 11))
    dev.launch("gen_f32", R.cdiv(n * k, 256), 256, p=(xb,), i=(n, k, 12))
    return xa, xb


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


class PearlSide:
    """The Pearl-C call at one shape on a side: its pipeline, the call graph and the eager reference check."""

    def __init__(self, side: Side, m_real: int, n: int, k: int):
        self.side, d = side, side.dev
        side.cur()
        self.m_real, self.m = m_real, R.cdiv(m_real, R.BM_D) * R.BM_D
        self.keys = keys_on(d)
        self.xa, self.xb = inputs_on(d, self.m, n, k)
        self.pipe = R.Pipeline(d, self.m, n, k, "sm120", self.keys, self.xa, self.xb, m_real=m_real)
        self.pipe.GEMM_SMEM = side.gemm_smem
        self.pipe.run_all()
        self.eager = snapshot(self.pipe)
        clobber(self.pipe)
        self.ex = side.capture(lambda: [f() for f in call_steps(self.pipe)])
        side.launch_graph(self.ex)
        side.sync()
        self.graphed = snapshot(self.pipe)

    def free(self) -> None:
        self.side.cur()
        self.side.dev.ck("cuGraphExecDestroy", self.ex)
        self.pipe.free()
        for p_ in (self.xa, self.xb, *self.keys.values()):
            self.side.dev.free(p_)


class GemmSide:
    """The ship's gemm_fast (8192^3, E4M3, unpromoted) persistent over the side's SMs, `reps` launches a graph."""

    def __init__(self, side: Side, reps: int = 1):
        self.side, d, big = side, side.dev, 8192
        side.cur()
        self.keys = keys_on(d)
        self.xa, self.xb = inputs_on(d, big, big, big)
        self.pipe = R.Pipeline(d, big, big, big, "sm120", self.keys, self.xa, self.xb)
        self.pipe.GEMM_SMEM = side.gemm_smem
        self.pipe.run_all()
        b = self.pipe.b
        self.flop = 2 * big ** 3 * reps
        self.ex = side.capture(lambda: [self.pipe.gemm("gemm_fast", "qa", "qb", big, big, out=(None, None, "y_fast"),
                                                       rs=b["sa"], cs=b["sb"]) for _ in range(reps)])

    def free(self) -> None:
        self.side.cur()
        self.side.dev.ck("cuGraphExecDestroy", self.ex)
        self.pipe.free()
        for p_ in (self.xa, self.xb, *self.keys.values()):
            self.side.dev.free(p_)


def window(P: PearlSide | None, n_calls: int, Q: GemmSide | None = None, n_co: int = 0) -> dict:
    out: dict = {}
    if Q is not None:
        q = Q.side
        q.cur()
        ev = q.events(1)[0]
        q.stamp(0)
        for i in range(n_co):
            q.launch_graph(Q.ex)
            q.stamp(i + 1)
            if i + 1 == LEAD:
                q.dev.ck("cuEventRecord", ev, q.stream)
    if P is not None:
        p = P.side
        p.cur()
        if Q is not None:
            p.dev.ck("cuStreamWaitEvent", p.stream, ev, 0)
        evs = p.events(n_calls + 1)
        p.stamp(0)
        p.dev.ck("cuEventRecord", evs[0], p.stream)
        for j in range(n_calls):
            p.launch_graph(P.ex)
            p.dev.ck("cuEventRecord", evs[j + 1], p.stream)
        p.stamp(1)
        p.sync()
        calls = []
        for j in range(n_calls):
            x = C.c_float()
            p.dev.ck("cuEventElapsedTime", C.byref(x), evs[j], evs[j + 1])
            calls.append(x.value)
        ps = p.read_stamps(2)
        span = (ps[1] - ps[0]) / 1e6
        out.update({"call_ms_mean": span / n_calls, "call_ms_med": statistics.median(calls), "span_ms": span,
                    "call_ms_p10": sorted(calls)[n_calls // 10], "call_ms_p90": sorted(calls)[(9 * n_calls) // 10]})
    if Q is not None:
        q.cur()
        q.sync()
        qs = q.read_stamps(n_co + 1)
        if P is None:
            span = (qs[n_co] - qs[1]) / 1e6
            out.update({"co_span_ms": span, "co_tflops": Q.flop * (n_co - 1) / span / 1e9})
        else:
            t = [(x - ps[0]) / 1e6 for x in qs]
            W = out["span_ms"]

            def done(x):
                for i in range(1, len(t)):
                    if t[i] >= x:
                        return (i - 1) + (x - t[i - 1]) / max(t[i] - t[i - 1], 1e-9)
                return float("nan")
            work = done(W) - done(0.0)
            out.update({"co_valid": t[-1] > W and t[LEAD] <= 0.0, "co_tail_ms": t[-1] - W,
                        "co_tflops": Q.flop * work / W / 1e9 if work == work else None})
    return out


def green_pair(dev0: R.Dev, full: C.Array, x: int) -> tuple:
    """Green contexts of x SMs (A) and the rest (B), each as (green ctx, CUcontext, SM info)."""
    result, remaining, n = (C.c_ubyte * RES_BYTES)(), (C.c_ubyte * RES_BYTES)(), u(1)
    dev0.ck("cuDevSmResourceSplitByCount", C.addressof(result), C.byref(n), C.addressof(full), C.addressof(remaining), 0, x)
    out = []
    for res in (result, remaining):
        desc, g, ctx = vp(), vp(), vp()
        dev0.ck("cuDevResourceGenerateDesc", C.byref(desc), C.addressof(res), 1)
        dev0.ck("cuGreenCtxCreate", C.byref(g), desc, dev0.dev, GREEN_DEFAULT_STREAM)
        dev0.ck("cuCtxFromGreenCtx", C.byref(ctx), g)
        out.append((g, ctx, sm_info(res)))
    return out[0], out[1]


def peak_tflops(sms: int, mhz: float) -> float:
    return 2 * sms * R.MACS_PER_SM_CLOCK * mhz * 1e6 / 1e12


def main(argv: list[str]) -> int:
    man_path = argv[1] if len(argv) > 1 else "check/check.json"
    man = json.load(open(man_path))
    image = open("pearl_c_sm120.cubin", "rb").read()
    dev0 = R.Dev(image)
    R.bind(dev0.cu, EXTRA)
    refused = dev0.identity(os.environ.get("PEARLC_EXPECT_UUID"))
    nv = R.Nvml(dev0.uuid)
    emit({"kind": "env", "device": dev0.name, "uuid": dev0.uuid, "sms": dev0.sms, "cc": list(dev0.cc),
          "cubin_sha256": hashlib.sha256(image).hexdigest(), "clk": nv.read(), "refused": refused,
          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if refused:
        print("REFUSED: " + "; ".join(refused), file=sys.stderr)
        return 3
    dev0.open()
    res = R.check(dev0, man, os.path.dirname(man_path), ["sm120"])
    ok = res["sm120"]["pearl_c_ok"] and res["sm120"]["plain_ok"]
    emit({"kind": "check", "ok": ok, "failed": res["sm120"]["failed"]})
    if not ok:
        return 4
    lo, hi = i32(), i32()
    dev0.ck("cuCtxGetStreamPriorityRange", C.byref(lo), C.byref(hi))
    full = (C.c_ubyte * RES_BYTES)()
    dev0.ck("cuDeviceGetDevResource", dev0.dev, C.addressof(full), RES_SM)
    emit({"kind": "sm_resource", **sm_info(full)})
    primary = vp()
    dev0.ck("cuDevicePrimaryCtxRetain", C.byref(primary), dev0.dev)

    n = k = 8192
    shapes = [int(s) for s in os.environ.get("GREEN_SHAPES", "1 32 128 512 2048 8192").split()]
    xs = [int(s) for s in os.environ.get("GREEN_SMS", "176 160 144 128 112 96 80 64 48 32 16").split()]
    base_side = Side(R.Dev(image), primary, dev0.sms, None, hi.value)
    ref: dict[int, tuple[str, float, int]] = {}
    for m_real in shapes:
        P = PearlSide(base_side, m_real, n, k)
        w0 = window(P, 20)
        n_calls = max(30, min(400, int(150 / w0["call_ms_med"])))
        al = [window(P, n_calls) for _ in range(3)]
        ref[m_real] = (P.eager, statistics.mean(w["call_ms_mean"] for w in al), n_calls)
        emit({"kind": "full_gpu", "m_real": m_real, "graph_equals_eager": P.graphed == P.eager, "n_calls": n_calls,
              "call_ms": ref[m_real][1], "windows": al})
        P.free()

    for x in xs:
        (ga, ca, ia), (gb, cb, ib) = green_pair(dev0, full, x)
        A = Side(R.Dev(image), ca, ia["sm_count"], ga, hi.value)
        B = Side(R.Dev(image), cb, ib["sm_count"], gb, lo.value)
        Q = GemmSide(B)
        qa = window(None, 0, Q, 12)
        emit({"kind": "green", "x_request": x, "a": ia, "b": ib, "co_alone": qa})
        for m_real in shapes:
            eager, full_ms, n_calls = ref[m_real]
            P = PearlSide(A, m_real, n, k)
            rows, unchanged = [], P.graphed == eager and P.eager == eager
            co_ms = qa["co_span_ms"] / 11
            for phase in ("alone", "co", "alone", "co", "alone"):
                if phase == "alone":
                    w = window(P, n_calls)
                else:
                    span_est = 3 * rows[0]["span_ms"]
                    n_co = min(MAX_STAMPS - 2, int(span_est / co_ms) + 2 * LEAD + 4)
                    A.cur()
                    clobber(P.pipe)
                    w = window(P, n_calls, Q, n_co)
                    A.cur()
                    unchanged = unchanged and snapshot(P.pipe) == eager
                rows.append({"phase": phase, **w})
            al = [r for r in rows if r["phase"] == "alone"]
            cr = [r for r in rows if r["phase"] == "co"]
            a_ms = statistics.mean(r["call_ms_mean"] for r in al)
            c_ms = statistics.mean(r["call_ms_mean"] for r in cr)
            pk = peak_tflops(dev0.sms, 2092)
            s = {"kind": "green_corun", "m_real": m_real, "a_sms": ia["sm_count"], "b_sms": ib["sm_count"],
                 "call_ms_full_gpu": full_ms, "call_ms_alone_on_a": a_ms, "call_ms_with_b": c_ms,
                 "slowdown_vs_full": c_ms / full_ms - 1, "slowdown_vs_a_alone": c_ms / a_ms - 1,
                 "alone_spread": (max(r["call_ms_mean"] for r in al) - min(r["call_ms_mean"] for r in al)) / a_ms,
                 "co_tflops": statistics.mean(r["co_tflops"] or 0 for r in cr), "co_alone_tflops": qa["co_tflops"],
                 "co_valid": all(r.get("co_valid") for r in cr), "peak_tflops_2092": pk,
                 "outputs_equal_primary_eager": unchanged, "windows": rows}
            s["co_share_of_peak"] = s["co_tflops"] / pk
            emit(s)
            print(f"m={m_real} A={ia['sm_count']} B={ib['sm_count']}: call {full_ms:.4f} full GPU, {a_ms:.4f} on A alone, "
                  f"{c_ms:.4f} with B ({s['slowdown_vs_full']:+.2%} vs full); B {s['co_tflops']:.0f} TFLOPS "
                  f"({s['co_share_of_peak']:.3f} of peak); valid {s['co_valid']}, outputs equal {unchanged}",
                  file=sys.stderr, flush=True)
            P.free()
        Q.free()
        for side in (A, B):
            side.cur()
            side.dev.free(side.stamps)
        dev0.ck("cuCtxSetCurrent", primary)
        dev0.ck("cuGreenCtxDestroy", ga)
        dev0.ck("cuGreenCtxDestroy", gb)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
