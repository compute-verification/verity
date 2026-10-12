"""Filler Pearl-C tiles in a decode step's idle tensor cores, on one RTX PRO 6000 (sm_120): lane ra-fill-compute's first climb.

The step is a stand-in for vLLM's decode step, captured as one CUDA graph: every linear of Llama-3.1-8B (or one GPU's
tensor-parallel slice of Llama-3.1-70B, `70b-tp8`) in FP8 through `torch._scaled_mm` (cuBLASLt), with distinct weights per
layer, the activation casts, residual adds and SiLU between them, and, at a context > 0, attention over a BF16 KV cache as two
batched GEMMs (QK^T, softmax, PV), so the step reads its weights and its KV from HBM as vLLM's does. It is memory-bound, and
its tensor cores are mostly idle.

The filler is one large Pearl-C call (`pearl-c-sm120-v1`, 64 x 64 tiles; fill.cu) whose operands are already formed, run a
tile at a time in the step's idle SM time. Each arm is one lever setting:

  chunk    a host thread keeps launching chunks of `per` tiles a CTA over `grid` CTAs on the filler's stream, which has the
           lowest priority (or the step's, `prio=same`); every CTA leaves at the stop flag, checked per tile
  persist  `grid` persistent CTAs on the low-priority stream, each taking the queue's next tile until the stop flag

and `store` says what a tile writes: `all` (C~, U and y, the buffers PoUW's tile leaves read), `y`, or `fold` (two XOR words a
tile, a stand-in for an in-kernel leaf). The host sets the stop flag (pinned memory) when the step's window ends, or after
`--starve-s` if the step never finishes (an arm whose filler starves the step is recorded as `starved`).

Gates, before any timing, in this process (exit 4 if one fails):
  1. fill.cu's kernels against PoUW's Python reference: fixture.py's `pearl-c-sm120-v1` check (C~, U and y of every tile,
     bit for bit, from `pearl_c` on `SM120`), at every store mode, chunked and persistent, at 1, 3 and all CTAs, alone and
     beside a step; and the fold against the fixture's words. PoUW's own `gemm_pearlc_d` must match it too.
  2. At the timing shape, the filler alone against `gemm_pearlc_d` on the same operands.
  3. Every arm: the step's output (the last layer's hidden state) bit-identical to the unfilled step's, and every tile the
     filler wrote in the window equal to `gemm_pearlc_d`'s (all, y) or folded to its words (fold).

Per arm (one JSON line in OUT/arms.jsonl): the step's time per replay (median, p90) against the unfilled step's, the filler
tiles that ended inside the step's window, the tensor-core utilization over the window ((filler + step) FP8 FLOPs / the E4M3
mma.sync peak, plus BF16 FLOPs / the BF16 peak, both at the SM clock NVML read during the window; the peaks are 807.2 and
403.7 TFLOPS at 2,100 MHz, kb/sm120/gotchas.md), the credited PoUW work a second ((1 - 1/1000) x `credit_of` of the filler's
shape, its tiles' share, W1 units), and the step's HBM read rate. OUT/leaderboard.json holds each config's best arm under
each step-slowdown budget.

What it is not: a timed panel row (no whole-node lease, no locked clocks, no autotuned baseline beside it); the filler's
forming (about 1% of its credit at n = 4096) and its SHA-512 tile leaves are not run, so `store=all` writes the words a leaf
kernel would read, and `fold` is not the verifier's bytes.

  python bench.py --lib libfill.so --fixture CHECK_DIR --out OUT [--configs 8b:32:4096,...] [--quick]
"""

from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PEAK_E4M3, PEAK_BF16, PEAK_MHZ = 807.2e12, 403.7e12, 2100.0
RASTER, TILE, PEEL_ROW = 8, 64, 128
STORES = {"all": 0, "y": 1, "fold": 2}
CAP = 1.0 / 1000

MODELS = {
    # hidden, intermediate, q heads, kv heads, head dim, layers, tensor-parallel degree
    "8b": (4096, 14336, 32, 8, 128, 32, 1),
    "70b-tp8": (8192, 28672, 64, 8, 128, 80, 8),
}


def emit(f, obj: dict) -> None:
    line = json.dumps(obj)
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


class Args(C.Structure):
    _fields_ = [("p", C.c_uint64 * 16), ("i", C.c_int32 * 16)]


class Lib:
    def __init__(self, path: str):
        self.l = C.CDLL(path)
        vp, i = C.c_void_p, C.c_int
        self.l.fill_launch.argtypes = [i, i, i, i, C.POINTER(Args), vp]
        self.l.pearlc_d.argtypes = [i, C.POINTER(Args), vp]
        self.l.stamp_at.argtypes = [vp, vp]
        self.l.flag_set.argtypes = [vp, i, vp]
        self.l.fill_occupancy.argtypes = [i, i, i, C.POINTER(i)]
        for f in ("fill_launch", "pearlc_d", "stamp_at", "flag_set", "fill_occupancy", "fill_smem"):
            getattr(self.l, f).restype = i

    @staticmethod
    def ck(rc: int, what: str) -> None:
        if rc:
            raise RuntimeError(f"{what}: error {rc}")

    def fill(self, persist: int, store: int, grid: int, a: Args, stream: torch.cuda.Stream) -> None:
        self.ck(self.l.fill_launch(persist, 1, store, grid, C.byref(a), C.c_void_p(stream.cuda_stream)), "fill_launch")

    def pearlc_d(self, grid: int, a: Args, stream: torch.cuda.Stream) -> None:
        self.ck(self.l.pearlc_d(grid, C.byref(a), C.c_void_p(stream.cuda_stream)), "pearlc_d")

    def stamp(self, slot: torch.Tensor, stream: torch.cuda.Stream) -> None:
        self.ck(self.l.stamp_at(C.c_void_p(slot.data_ptr()), C.c_void_p(stream.cuda_stream)), "stamp")

    def occupancy(self, persist: int, store: int) -> tuple[int, int]:
        regs = C.c_int(0)
        n = self.l.fill_occupancy(persist, 1, store, C.byref(regs))
        return n, regs.value


def raster(t: int, tiles_m: int, tiles_n: int, group: int = RASTER) -> tuple[int, int]:
    """pearl_c_sm120.cu's `raster`: queue position t's tile (tm, tn)."""
    per = group * tiles_n
    first = (t // per) * group
    size = min(tiles_m - first, group)
    r = t % per
    return first + r % size, r // size


class Filler:
    """One Pearl-C call's formed operands and the filler's outputs, queue and log, on the device."""

    def __init__(self, M: int, N: int, K: int, at, bt, pa, pb, ia, ib, qcap: int):
        assert M % TILE == 0 and N % TILE == 0 and K % 64 == 0, (M, N, K)
        dev = "cuda"
        self.M, self.N, self.K = M, N, K
        self.tiles_m, self.tiles_n = M // TILE, N // TILE
        self.tiles = self.tiles_m * self.tiles_n
        self.at, self.bt, self.pa, self.pb, self.ia, self.ib = at, bt, pa, pb, ia, ib
        self.c = torch.empty((M, N), dtype=torch.int32, device=dev)
        self.u = torch.empty((M, N), dtype=torch.int32, device=dev)
        self.y = torch.empty((M, N), dtype=torch.int16, device=dev)
        self.fold = torch.zeros(2 * self.tiles, dtype=torch.int32, device=dev)
        self.qcap = qcap
        self.log = torch.zeros(qcap, dtype=torch.int64, device=dev)
        self.counter = torch.zeros(1, dtype=torch.int32, device=dev)
        self.stop = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.tm_of = np.empty(self.tiles, np.int64)
        self.tn_of = np.empty(self.tiles, np.int64)
        for t in range(self.tiles):
            self.tm_of[t], self.tn_of[t] = raster(t, self.tiles_m, self.tiles_n)

    @property
    def fp8_flops_tile(self) -> float:
        return 2.0 * TILE * TILE * self.K

    @property
    def bf16_flops_tile(self) -> float:
        return 2.0 * TILE * TILE * PEEL_ROW

    def args(self, first: int, count: int, *, log: bool = True, stop: bool = True, c=None, u=None, y=None) -> Args:
        a = Args()
        p = [self.at.data_ptr(), self.bt.data_ptr(), (c if c is not None else self.c).data_ptr(),
             (u if u is not None else self.u).data_ptr(), (y if y is not None else self.y).data_ptr(),
             self.pa.data_ptr(), self.pb.data_ptr(), self.ia.data_ptr(), self.ib.data_ptr(), 0, 0,
             self.log.data_ptr() if log else 0, self.counter.data_ptr(), self.stop.data_ptr() if stop else 0,
             self.fold.data_ptr()]
        for j, x in enumerate(p):
            a.p[j] = x
        for j, x in enumerate((self.M, self.N, self.K, RASTER, first, count)):
            a.i[j] = x
        return a

    def poison(self) -> None:
        for x in (self.c, self.u, self.y):
            x.view(torch.uint8).fill_(0xA5)
        self.fold.zero_()
        self.log.zero_()
        self.counter.zero_()
        self.stop[0] = 0

    def tile_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.tiles_m, TILE, self.tiles_n, TILE)

    def written(self) -> np.ndarray:
        """How many times each tile (by raster position) was written, from the log."""
        pos = torch.nonzero(self.log).flatten().cpu().numpy()
        return np.bincount(pos % self.tiles, minlength=self.tiles)


def fold_of(c: np.ndarray, u: np.ndarray, f: Filler) -> np.ndarray:
    out = np.zeros(2 * f.tiles, np.uint32)
    for t in range(f.tiles):
        r, q = f.tm_of[t] * TILE, f.tn_of[t] * TILE
        out[2 * t] = np.bitwise_xor.reduce(c[r:r + TILE, q:q + TILE].ravel())
        out[2 * t + 1] = np.bitwise_xor.reduce(u[r:r + TILE, q:q + TILE].ravel())
    return out


def compare(f: Filler, store: int, ref_c, ref_u, ref_y, ref_fold, counts: np.ndarray) -> dict:
    """Every tile written at least once equals the reference (all, y), or the fold equals the reference fold of each tile
    written an odd number of times and 0 otherwise (fold)."""
    mask = torch.zeros(f.tiles_m, f.tiles_n, dtype=torch.bool)
    hit = np.nonzero(counts)[0]
    mask[torch.from_numpy(f.tm_of[hit]), torch.from_numpy(f.tn_of[hit])] = True
    mask = mask.to("cuda")
    bad = 0
    if store == 0:
        for x, ref in ((f.c, ref_c), (f.u, ref_u), (f.y, ref_y)):
            eq = (f.tile_view(x) == f.tile_view(ref)).all(dim=3).all(dim=1)
            bad += int((mask & ~eq).sum())
    elif store == 1:
        eq = (f.tile_view(f.y) == f.tile_view(ref_y)).all(dim=3).all(dim=1)
        bad += int((mask & ~eq).sum())
    else:
        odd = torch.from_numpy(np.repeat(counts % 2 == 1, 2)).to("cuda")
        want = torch.where(odd, ref_fold, torch.zeros_like(ref_fold))
        bad += int((f.fold != want).sum())
    return {"tiles_checked": len(hit), "mismatched": bad}


# ---------------------------------------------------------------- the step

class Step:
    """One decode step as a CUDA graph on `stream`: FP8 linears through cuBLASLt, attention over a BF16 KV cache."""

    def __init__(self, model: str, batch: int, ctx: int, stream: torch.cuda.Stream, seed: int = 1):
        hidden, inter, heads, kvh, hd, layers, tp = MODELS[model]
        self.model, self.batch, self.ctx = model, batch, ctx
        self.q_heads, self.kv_heads, self.hd = heads // tp, max(1, kvh // tp), hd
        self.inter = inter // tp
        m = -(-batch // 16) * 16                    # cuBLASLt FP8 wants 16 | m; vLLM pads decode too
        self.m = m
        g = torch.Generator(device="cuda").manual_seed(seed)
        fp8 = torch.float8_e4m3fn
        qkv_n = (self.q_heads + 2 * self.kv_heads) * hd
        # (n, k) per linear: QKV, O, gate-up, down; O's and down's k are the slice's, their n the full hidden
        self.shapes = [(qkv_n, hidden), (hidden, self.q_heads * hd), (2 * self.inter, hidden), (hidden, self.inter)]

        def weight(n, k):
            return (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(fp8)

        self.w = [[weight(n, k) for n, k in self.shapes] for _ in range(layers)]
        self.kv = []
        if ctx:
            for _ in range(layers):
                shape = (batch * self.kv_heads, ctx, hd)
                self.kv.append((torch.randn(shape, device="cuda", generator=g, dtype=torch.bfloat16),
                                torch.randn(shape, device="cuda", generator=g, dtype=torch.bfloat16)))
        self.one = torch.ones((), device="cuda", dtype=torch.float32)
        self.x0 = (torch.randn(m, hidden, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        self.weight_bytes = sum(w.numel() for ws in self.w for w in ws)
        self.kv_bytes = sum(k.numel() * 2 * 2 for k, _ in self.kv)
        self.fp8_flops = 2.0 * m * sum(n * k for n, k in self.shapes) * layers
        self.bf16_flops = 2.0 * 2 * batch * self.q_heads * ctx * hd * layers if ctx else 0.0
        self.stream = stream
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.forward()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.out = self.forward()
        torch.cuda.synchronize()
        self.replay(1)
        torch.cuda.synchronize()
        self.ref = self.out.clone()

    def mm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch._scaled_mm(x.to(torch.float8_e4m3fn), w.t(), scale_a=self.one, scale_b=self.one, out_dtype=torch.bfloat16)

    def forward(self) -> torch.Tensor:
        x = self.x0
        b, hq, hk, hd = self.batch, self.q_heads, self.kv_heads, self.hd
        for li, (wqkv, wo, wgu, wd) in enumerate(self.w):
            qkv = self.mm(x, wqkv)
            q = qkv[:, :hq * hd]
            if self.ctx:
                k_cache, v_cache = self.kv[li]
                qg = q[:b].reshape(b * hk, hq // hk, hd)
                s = torch.bmm(qg, k_cache.transpose(1, 2))
                p = torch.softmax(s.float() * (hd ** -0.5), dim=-1).to(torch.bfloat16)
                o = torch.bmm(p, v_cache).reshape(b, hq * hd)
                if self.m > b:
                    o = torch.cat([o, q[b:]], dim=0)
            else:
                o = q
            x = x + self.mm(o, wo)
            gu = self.mm(x, wgu)
            a = torch.nn.functional.silu(gu[:, :self.inter]) * gu[:, self.inter:]
            x = x + self.mm(a, wd)
        return x

    def replay(self, n: int) -> None:
        with torch.cuda.stream(self.stream):
            for _ in range(n):
                self.graph.replay()

    def identical(self) -> bool:
        return bool(torch.equal(self.out.view(torch.int16), self.ref.view(torch.int16)))


# ---------------------------------------------------------------- clocks

class Clock:
    """The SM clock of the leased GPU (NVML by UUID), sampled every 10 ms on a thread; `mean(t0, t1)` over host time."""

    def __init__(self):
        self.samples: list[tuple[float, float]] = []
        self.nvml = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            uuid = os.environ.get("GPU_LEASE_UUID")
            h = pynvml.nvmlDeviceGetHandleByUUID(uuid) if uuid else pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml, self.h = pynvml, h
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
        self.done = threading.Event()
        if self.nvml:
            threading.Thread(target=self.run, daemon=True).start()

    def run(self) -> None:
        while not self.done.is_set():
            try:
                self.samples.append((time.monotonic(), float(self.nvml.nvmlDeviceGetClockInfo(self.h, 1))))
            except Exception as e:  # noqa: BLE001
                self.error = repr(e)
            self.done.wait(0.01)

    def mean(self, t0: float, t1: float) -> float | None:
        xs = [c for t, c in self.samples if t0 <= t <= t1]
        if not xs and self.samples:
            xs = [min(self.samples, key=lambda s: abs(s[0] - (t0 + t1) / 2))[1]]
        return statistics.mean(xs) if xs else None

    def smi(self) -> str:
        uuid = os.environ.get("GPU_LEASE_UUID", "0")
        try:
            query = "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"
            return subprocess.run(["nvidia-smi", "-i", uuid, query, "--format=csv,noheader"], capture_output=True, text=True,
                                  timeout=20, check=False).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return repr(e)


# ---------------------------------------------------------------- one window

def window(lib: Lib, f: Filler | None, step: Step, reps: int, arm: dict, lo: torch.cuda.Stream, clock: Clock,
           starve_s: float) -> dict:
    """The step replayed `reps` times on its stream with the arm's filler beside it: per-replay times, the window on the
    device's clock, the filler tiles that ended inside it, and whether the step finished before the starvation limit."""
    hi = step.stream
    stamps = torch.zeros(2, dtype=torch.int64, device="cuda")
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(reps + 1)]
    pump_done, pumped = threading.Event(), [0]
    if f is not None:
        f.poison()
    torch.cuda.synchronize()
    store = STORES[arm["store"]] if f is not None else 0
    thread = None
    if f is not None and arm["mode"] == "chunk":
        per_launch = arm["grid"] * arm["per"]
        stream = lo if arm["prio"] == "low" else arm["peer"]

        def pump():
            pos = 0
            while not pump_done.is_set() and pos + per_launch <= f.qcap:
                lib.fill(0, store, arm["grid"], f.args(pos, per_launch), stream)
                pos += per_launch
            pumped[0] = pos

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
    elif f is not None and arm["mode"] == "persist":
        lib.fill(1, store, arm["grid"], f.args(0, f.qcap), lo)
    if f is not None:
        time.sleep(arm.get("lead_s", 0.02))         # the filler is resident before the window opens
    h0 = time.monotonic()
    with torch.cuda.stream(hi):
        lib.stamp(stamps[0:1], hi)
        for r in range(reps):
            ev[r].record(hi)
            step.graph.replay()
        ev[reps].record(hi)
        lib.stamp(stamps[1:2], hi)
    starved = False
    while not ev[reps].query():
        if time.monotonic() - h0 > starve_s:
            starved = True
            break
        time.sleep(0.0005)
    h1 = time.monotonic()
    if f is not None:
        f.stop[0] = 1
        pump_done.set()
    if thread is not None:
        thread.join()
    torch.cuda.synchronize()
    times = [ev[r].elapsed_time(ev[r + 1]) for r in range(reps)]
    t0, t1 = stamps.cpu().tolist()
    out = {"step_ms": statistics.median(times), "step_p90_ms": sorted(times)[int(0.9 * (reps - 1))],
           "window_ns": t1 - t0, "starved": starved, "clock_mhz": clock.mean(h0, h1), "identical": step.identical()}
    if f is not None:
        log = f.log
        out["tiles_in_window"] = int(((log >= t0) & (log <= t1)).sum())
        out["tiles_total"] = int((log > 0).sum())
        out["pumped"] = pumped[0]
    return out


def score(out: dict, base: dict, f: Filler | None, step: Step, credit_tile: float | None) -> dict:
    w = out["window_ns"] * 1e-9
    reps_flops8 = step.fp8_flops * out["reps"]
    reps_flops16 = step.bf16_flops * out["reps"]
    tiles = out.get("tiles_in_window", 0)
    clk = out["clock_mhz"] or PEAK_MHZ
    p8, p16 = PEAK_E4M3 * clk / PEAK_MHZ, PEAK_BF16 * clk / PEAK_MHZ
    fill8 = tiles * f.fp8_flops_tile if f else 0.0
    fill16 = tiles * f.bf16_flops_tile if f else 0.0
    res = {
        "slowdown": out["step_ms"] / base["step_ms"],
        "p90_slowdown": out["step_p90_ms"] / base["step_p90_ms"],
        "util": (fill8 + reps_flops8) / w / p8 + (fill16 + reps_flops16) / w / p16,
        "util_filler": fill8 / w / p8 + fill16 / w / p16,
        "util_step": reps_flops8 / w / p8 + reps_flops16 / w / p16,
        "filler_tflops": (fill8 + fill16) / w / 1e12,
        "step_hbm_gbs": (step.weight_bytes + step.kv_bytes) / (out["step_ms"] * 1e-3) / 1e9,
    }
    if credit_tile is not None and f is not None:
        res["credit_w1_per_s"] = (1 - CAP) * credit_tile * tiles / w
    return res


# ---------------------------------------------------------------- gates

def load_fixture(d: Path, qcap: int) -> tuple[Filler, dict]:
    man = json.loads((d / "check.json").read_text())
    m, n, k = man["shape"]
    v = man["variants"]["sm120"]

    def raw(part: dict, name: str, sub: str = "") -> np.ndarray:
        return np.fromfile(d / sub / part[name]["file"], np.uint8)

    common = man["common"]

    def to(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.copy()).to("cuda")

    f = Filler(m, n, k, to(raw(common, "at")), to(raw(common, "bt")), to(raw(v["files"], "pa_t", "sm120")),
               to(raw(v["files"], "pb_t", "sm120")), to(raw(common, "ia_a")), to(raw(common, "ia_b")), qcap)
    exp = {name: raw(v["files"], name, "sm120") for name in ("c", "u", "y")}
    exp["c32"] = exp["c"].view(np.uint32).reshape(m, n)
    exp["u32"] = exp["u"].view(np.uint32).reshape(m, n)
    return f, exp


def gate_fixture(lib: Lib, d: Path, lo: torch.cuda.Stream, step: Step | None, log) -> bool:
    f, exp = load_fixture(d, qcap=4096)
    want = {"c": exp["c"], "u": exp["u"], "y": exp["y"]}
    want_fold = fold_of(exp["c32"], exp["u32"], f)
    ok = True

    def check(name: str, store: int) -> None:
        nonlocal ok
        torch.cuda.synchronize()
        got = {"c": f.c, "u": f.u, "y": f.y}
        if store == 2:
            res = bool(np.array_equal(f.fold.cpu().numpy().view(np.uint32), want_fold))
            detail = {"fold": res}
        else:
            keys = ("c", "u", "y") if store == 0 else ("y",)
            detail = {kk: bool(np.array_equal(got[kk].cpu().numpy().view(np.uint8).ravel(), want[kk])) for kk in keys}
            res = all(detail.values())
        ok &= res
        emit(log, {"gate": name, "store": store, "pass": res, **detail})

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    f.poison()
    lib.pearlc_d(sms, f.args(0, f.tiles, log=False, stop=False), lo)
    check("gemm_pearlc_d", 0)
    for store in (0, 1, 2):
        for grid, per in ((1, f.tiles), (3, 5), (sms, 1)):
            f.poison()
            for pos in range(0, f.tiles, grid * per):
                lib.fill(0, store, grid, f.args(pos, min(grid * per, f.tiles - pos)), lo)
            check(f"chunk grid={grid} per={per}", store)
        for grid in (1, 7, sms):
            f.poison()
            lib.fill(1, store, grid, f.args(0, f.tiles), lo)
            check(f"persist grid={grid}", store)
        if step is not None:
            f.poison()
            step.replay(1)
            for pos in range(0, f.tiles, 8):
                lib.fill(0, store, 4, f.args(pos, min(8, f.tiles - pos)), lo)
            step.replay(2)
            torch.cuda.synchronize()
            check("chunk beside the step", store)
            same = step.identical()
            ok &= same
            emit(log, {"gate": "step identical beside the fixture filler", "store": store, "pass": same})
    f.poison()
    f.stop[0] = 1
    lib.fill(1, 0, sms, f.args(0, f.tiles), lo)
    torch.cuda.synchronize()
    stopped = int(f.counter.item()) == 0
    ok &= stopped
    emit(log, {"gate": "persistent CTAs read the pinned stop flag", "pass": stopped})
    return ok


def random_filler(M: int, N: int, K: int, qcap: int, seed: int = 7) -> Filler:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def codes(r, c):
        x = torch.randint(0, 256, (r, c), device="cuda", generator=g, dtype=torch.int32).to(torch.uint8)
        x[(x & 0x7F) == 0x7F] ^= 1                    # no NaN code
        return x.contiguous()

    def peel(r):
        return (torch.randn(r, PEEL_ROW // 2, device="cuda", generator=g) * 0.1).to(torch.bfloat16).view(torch.uint8).contiguous()


    def scale(r):
        return torch.full((r,), 1.0 / 1024, device="cuda", dtype=torch.float32)

    return Filler(M, N, K, codes(M, K), codes(N, K), peel(M), peel(N), scale(M), scale(N), qcap)


# ---------------------------------------------------------------- main

def credit_tile(M: int, N: int, K: int) -> float | None:
    try:
        sys.path.insert(0, str(ROOT))
        from verity.core.protocols.pouw.protocol import Shape
        from verity.core.protocols.pouw.schemes import pearl_c as PC
        from verity.core.protocols.pouw.schemes.pearl_c_device import SM120

        s = PC.PearlC(forming="v1", device=SM120, hashing="h3")
        return s.credit_of(Shape(M, K, N)) * TILE * TILE / (M * N)
    except Exception as e:  # noqa: BLE001
        print(f"credit_of unavailable: {e!r}", file=sys.stderr)
        return None


def arms(quick: bool, sms: int) -> list[dict]:
    out = []
    grids = (sms // 4, sms // 2, sms, 2 * sms)
    pers = (1, 4) if quick else (1, 4, 16)
    stores = ("all", "fold") if quick else ("all", "y", "fold")
    for store in stores:
        for grid in grids:
            for per in pers:
                out.append({"mode": "chunk", "grid": grid, "per": per, "store": store, "prio": "low"})
            out.append({"mode": "persist", "grid": grid, "store": store, "prio": "low"})
        out.append({"mode": "chunk", "grid": sms, "per": 4, "store": store, "prio": "same"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--configs", default="8b:16:0,8b:32:4096,70b-tp8:16:0,70b-tp8:32:4096", help="model:batch:ctx,...")
    ap.add_argument("--filler", default="4096x4096x8192", help="the filler call's MxNxK")
    ap.add_argument("--window-ms", type=float, default=150.0, help="each arm's window, at least")
    ap.add_argument("--starve-s", type=float, default=5.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--budgets", default="1.02,1.05,1.10,1.25,1.5,2,4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "arms.jsonl", "w") as log:
        return run(args, log)


def run(args: argparse.Namespace, log) -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    lib = Lib(args.lib)
    clock = Clock()
    lo = torch.cuda.Stream(priority=0)
    hi = torch.cuda.Stream(priority=-5)
    peer = torch.cuda.Stream(priority=-5)            # the filler at the step's priority, on a stream of its own
    emit(log, {"device": props.name, "sms": sms, "torch": torch.__version__, "cuda": torch.version.cuda,
               "uuid": os.environ.get("GPU_LEASE_UUID"), "smi": clock.smi(), "nvml": bool(clock.nvml),
               "stream_priority": {"lo": lo.priority, "hi": hi.priority, "peer": peer.priority},
               "occupancy": {f"{m}/{s}": lib.occupancy(p, STORES[s]) for p, m in ((0, "chunk"), (1, "persist"))
                             for s in STORES}, "fill_smem": lib.l.fill_smem()})

    # gate 1, with a small step beside it
    gstep = Step("8b", 16, 0, hi)
    if not gate_fixture(lib, args.fixture, lo, gstep, log):
        emit(log, {"verdict": "REFUSED", "why": "a fixture gate failed"})
        return 4
    del gstep
    torch.cuda.empty_cache()

    M, N, K = (int(x) for x in args.filler.split("x"))
    credit = credit_tile(M, N, K)
    f = random_filler(M, N, K, qcap=1 << 23)
    ref_c, ref_u, ref_y = torch.empty_like(f.c), torch.empty_like(f.u), torch.empty_like(f.y)
    lib.pearlc_d(sms, f.args(0, f.tiles, log=False, stop=False, c=ref_c, u=ref_u, y=ref_y), lo)
    torch.cuda.synchronize()

    def tv(x: torch.Tensor) -> torch.Tensor:
        return x.view(f.tiles_m, TILE, f.tiles_n, TILE).permute(0, 2, 1, 3).reshape(f.tiles_m, f.tiles_n, -1)

    ref_fold = torch.zeros(2 * f.tiles, dtype=torch.int32, device="cuda")
    fc, fu = tv(ref_c).cpu().numpy().view(np.uint32), tv(ref_u).cpu().numpy().view(np.uint32)
    fold_np = np.zeros(2 * f.tiles, np.uint32)
    fold_np[0::2] = np.bitwise_xor.reduce(fc[f.tm_of, f.tn_of], axis=1)
    fold_np[1::2] = np.bitwise_xor.reduce(fu[f.tm_of, f.tn_of], axis=1)
    ref_fold.copy_(torch.from_numpy(fold_np.view(np.int32)))

    # gate 2 and the filler alone: every store mode at the full grid, two passes over the problem
    alone = {}
    for store, code in STORES.items():
        f.poison()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        per = 2 * f.tiles
        e0.record(lo)
        for pos in range(0, per, sms * 4):
            lib.fill(0, code, sms, f.args(pos, min(sms * 4, per - pos), stop=False), lo)
        e1.record(lo)
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        chk = compare(f, code, ref_c, ref_u, ref_y, ref_fold, f.written())
        alone[store] = (per / (ms * 1e-3))
        clk = clock.mean(time.monotonic() - 1, time.monotonic()) or PEAK_MHZ
        tfl = per * (f.fp8_flops_tile + f.bf16_flops_tile) / (ms * 1e-3) / 1e12
        emit(log, {"filler_alone": store, "tiles_per_s": alone[store], "tflops": tfl,
                   "of_peak": per * (f.fp8_flops_tile / PEAK_E4M3 + f.bf16_flops_tile / PEAK_BF16) / (ms * 1e-3) / (clk / PEAK_MHZ),
                   "clock_mhz": clk, **chk})
        if chk["mismatched"]:
            emit(log, {"verdict": "REFUSED", "why": f"filler alone differs from gemm_pearlc_d ({store})"})
            return 4
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(lo)
    lib.pearlc_d(sms, f.args(0, f.tiles, log=False, stop=False, c=ref_c, u=ref_u, y=ref_y), lo)
    e1.record(lo)
    torch.cuda.synchronize()
    emit(log, {"pearlc_d_alone": True, "tiles_per_s": f.tiles / (e0.elapsed_time(e1) * 1e-3),
               "tflops": f.tiles * (f.fp8_flops_tile + f.bf16_flops_tile) / (e0.elapsed_time(e1) * 1e-3) / 1e12})

    budgets = [float(b) for b in args.budgets.split(",")]
    board = []
    failed = False
    for cfg in args.configs.split(","):
        model, batch, ctx = cfg.split(":")
        step = Step(model, int(batch), int(ctx), hi)
        step.replay(3)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(hi)
        step.replay(5)
        e1.record(hi)
        torch.cuda.synchronize()
        reps = max(10, math.ceil(args.window_ms / (e0.elapsed_time(e1) / 5)))
        base = window(lib, None, step, reps, {}, lo, clock, args.starve_s)
        base["reps"] = reps
        sbase = score(base, base, None, step, None)
        head = {"config": cfg, "m_padded": step.m, "weight_gb": step.weight_bytes / 1e9, "kv_gb": step.kv_bytes / 1e9,
                "reps": reps}
        emit(log, {**head, "arm": "unfilled", **base, **sbase})
        rows = []
        for arm in arms(args.quick, sms):
            out = window(lib, f, step, reps, dict(arm, peer=peer), lo, clock, args.starve_s)
            out["reps"] = reps
            chk = compare(f, STORES[arm["store"]], ref_c, ref_u, ref_y, ref_fold, f.written())
            s = score(out, base, f, step, credit)
            row = {**head, "arm": arm, **out, **s, **chk}
            emit(log, row)
            if chk["mismatched"] or not out["identical"]:
                failed = True
            rows.append(row)
        best = {}
        for b in budgets:
            ok = [r for r in rows if not r["starved"] and r["identical"] and not r["mismatched"] and r["slowdown"] <= b]
            if ok:
                r = max(ok, key=lambda r: r["util"])
                best[str(b)] = {k: r[k] for k in ("arm", "slowdown", "p90_slowdown", "util", "util_filler", "util_step",
                                                  "filler_tflops", "credit_w1_per_s", "step_hbm_gbs", "clock_mhz")
                                if k in r}
        board.append({**head, "unfilled": {**base, **sbase}, "best": best})
        del step
        torch.cuda.empty_cache()
    (args.out / "leaderboard.json").write_text(json.dumps({"filler": [M, N, K], "credit_w1_per_tile": credit,
                                                          "configs": board}, indent=1) + "\n")
    clock.done.set()
    emit(log, {"verdict": "FAILED" if failed else "ok", "smi_end": clock.smi()})
    return 5 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
