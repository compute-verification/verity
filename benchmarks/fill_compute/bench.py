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

and `store` says what a tile writes: `all` (C~, U and y, the buffers PoUW's tile leaves read), `y`, `fold` (two XOR words a
tile, a stand-in for an in-kernel leaf), or `rcpt` (the tile's sketch-sha512 receipts of its C~ and U blocks, receipt.py:
the one receipt form, computed in the kernel from the tile as it leaves the accumulators). The host sets the stop flag (pinned memory) when the step's window ends, or after
`--starve-s` if the step never finishes (an arm whose filler starves the step is recorded as `starved`). These are family
`stream`; family `short` runs chunks of one tile a CTA from shallower filler calls (`--short-k`), so a filler CTA holds its
SM for less time; family `gc` captures the step on a green context of n SMs (`alone`: the step's rate against its SMs) and
runs the persistent filler on a green context of the rest (`gc_arms`: the 64 x 64 tile at two CTAs an SM, the 128 x 128 tile
at one, which reads half the operand bytes a FLOP, and the throttle, each CTA idle a set time after each tile). `sm_spread`
checks that a green context's launches, direct and from a graph, stay on its SMs.

The protocol-kernel gate is reported beside every timing: run.sh refuses the build unless `cuobjdump -res-usage` reports
STACK:0 and LOCAL:0, and `--l2` is l2_gate.py's verdict on the filler's L2 write and atomic sectors (ncu) against what its
outputs declare. The step's HBM rate is its bytes over its time, so `hbm_ratio` = 1 / slowdown, and the budget 1.01 is the
rule that a filler keep the step's bytes/s within 1% of the unfilled rate.

Gates, before any timing, in this process (exit 4 if one fails):
  1. fill.cu's kernels against PoUW's Python reference: fixture.py's `pearl-c-sm120-v1` check (C~, U and y of every tile,
     bit for bit, from `pearl_c` on `SM120`), at every store mode, chunked and persistent, at 1, 3 and all CTAs, alone and
     beside a step; the fold against the fixture's words, and the receipts against receipt.py's of the fixture's C~ and U. PoUW's own `gemm_pearlc_d` must match it too.
  2. At the timing shape, the filler alone against `gemm_pearlc_d` on the same operands.
  3. Every arm: the step's output (the last layer's hidden state) bit-identical to the unfilled step's, and every tile the
     filler wrote in the window equal to `gemm_pearlc_d`'s (all, y), folded to its words (fold), or with receipt.py's
     receipts of its words (rcpt).

Per arm (one JSON line in OUT/arms.jsonl): the step's time per replay (median, p90) against the unfilled step's, the filler
tiles that ended inside the step's window, the tensor-core utilization over the window ((filler + step) FP8 FLOPs / the E4M3
mma.sync peak, plus BF16 FLOPs / the BF16 peak, both at the SM clock NVML read during the window; the peaks are 807.2 and
403.7 TFLOPS at 2,100 MHz, kb/sm120/gotchas.md), the credited PoUW work a second ((1 - 1/1000) x `credit_of` of the filler's
shape, its tiles' share, W1 units), and the step's HBM read rate. OUT/leaderboard.json holds each config's best arm under
each step-slowdown budget.

What it is not: a timed panel row (no whole-node lease, no locked clocks, no autotuned baseline beside it); the filler's
forming (about 1% of its credit at n = 4096) and its SHA-512 tile leaves are not run, so `store=all` writes the words a leaf
kernel would read, `fold` is not the verifier's bytes, and `rcpt`'s seed is a fixed one (RSEED), not a verifier's draw.

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

import receipt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PEAK_E4M3, PEAK_BF16, PEAK_MHZ = 807.2e12, 403.7e12, 2100.0
RASTER, TILE, PEEL_ROW = 8, 64, 128
L2_REPEATS = 5
STORES = {"all": 0, "y": 1, "fold": 2, "rcpt": 3}
RSEED = bytes(range(64))         # the receipts' pass seed (a fixture's, not a verifier's draw)
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
        self.l.gc_split.argtypes = [i, i, i, C.POINTER(vp), C.POINTER(vp), C.POINTER(i)]
        self.l.smid_at.argtypes = [vp, i, i, vp]
        self.l.fill_smem.argtypes = [i]
        for f in ("fill_launch", "pearlc_d", "stamp_at", "flag_set", "fill_occupancy", "fill_smem", "gc_split", "smid_at"):
            getattr(self.l, f).restype = i

    @staticmethod
    def ck(rc: int, what: str) -> None:
        if rc:
            raise RuntimeError(f"{what}: error {rc}")

    def fill(self, persist: int, store: int, grid: int, a: Args, stream: torch.cuda.Stream, minb: int = 1) -> None:
        """minb 1 or 2: the 64 x 64 tile's kernels at that launch bound; 128: the 128 x 128 tile's (`Filler.minb`)."""
        self.ck(self.l.fill_launch(persist, minb, store, grid, C.byref(a), C.c_void_p(stream.cuda_stream)), "fill_launch")

    def pearlc_d(self, grid: int, a: Args, stream: torch.cuda.Stream) -> None:
        self.ck(self.l.pearlc_d(grid, C.byref(a), C.c_void_p(stream.cuda_stream)), "pearlc_d")

    def stamp(self, slot: torch.Tensor, stream: torch.cuda.Stream) -> None:
        self.ck(self.l.stamp_at(C.c_void_p(slot.data_ptr()), C.c_void_p(stream.cuda_stream)), "stamp")

    def occupancy(self, persist: int, store: int, minb: int = 1) -> tuple[int, int]:
        regs = C.c_int(0)
        n = self.l.fill_occupancy(persist, minb, store, C.byref(regs))
        return n, regs.value

    def sm_spread(self, stream: torch.cuda.Stream, grid: int, graph: bool) -> int:
        """How many distinct SMs `grid` CTAs launched on `stream` (directly, or as a graph captured on it) ran on."""
        out = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                self.ck(self.l.smid_at(C.c_void_p(out.data_ptr()), grid, 20000, C.c_void_p(stream.cuda_stream)), "smid")
            with torch.cuda.stream(stream):
                g.replay()
        else:
            self.ck(self.l.smid_at(C.c_void_p(out.data_ptr()), grid, 20000, C.c_void_p(stream.cuda_stream)), "smid")
        torch.cuda.synchronize()
        return int(torch.unique(out).numel())

    def gc_split(self, n_step: int, prio_step: int, prio_fill: int) -> dict:
        """Green contexts: the step's stream on (at least) `n_step` SMs and the filler's on the rest (fill.cu's gc_split)."""
        a, b, got = C.c_void_p(), C.c_void_p(), (C.c_int * 4)()
        rc = self.l.gc_split(n_step, prio_step, prio_fill, C.byref(a), C.byref(b), got)
        if rc:
            return {"error": rc}
        return {"step": torch.cuda.ExternalStream(a.value), "fill": torch.cuda.ExternalStream(b.value) if b.value else None,
                "step_sms": got[0], "fill_sms": got[1], "min_partition": got[2], "alignment": got[3]}


def raster(t: int, tiles_m: int, tiles_n: int, group: int = RASTER) -> tuple[int, int]:
    """pearl_c_sm120.cu's `raster`: queue position t's tile (tm, tn)."""
    per = group * tiles_n
    first = (t // per) * group
    size = min(tiles_m - first, group)
    r = t % per
    return first + r % size, r // size


class Filler:
    """One Pearl-C call's formed operands and the filler's outputs, queue and log, on the device, run in T x T tiles a CTA
    (64: the protocol's tile; 128: four of them, each output's arithmetic unchanged)."""

    def __init__(self, M: int, N: int, K: int, at, bt, pa, pb, ia, ib, qcap: int, T: int = TILE):
        assert T in (64, 128) and M % T == 0 and N % T == 0 and K % 64 == 0, (M, N, K, T)
        dev = "cuda"
        self.M, self.N, self.K, self.T = M, N, K, T
        self.minb = 128 if T == 128 else 1
        self.tiles_m, self.tiles_n = M // T, N // T
        self.tiles = self.tiles_m * self.tiles_n
        self.at, self.bt, self.pa, self.pb, self.ia, self.ib = at, bt, pa, pb, ia, ib
        self.c = torch.empty((M, N), dtype=torch.int32, device=dev)
        self.u = torch.empty((M, N), dtype=torch.int32, device=dev)
        self.y = torch.empty((M, N), dtype=torch.int16, device=dev)
        self.fold = torch.zeros(2 * self.tiles, dtype=torch.int32, device=dev)
        self.rc = torch.zeros((self.tiles, 2 * self.rb * 64), dtype=torch.uint8, device=dev)
        self.rkey = receipt_key()
        self.qcap = qcap
        self.log = torch.zeros(qcap, dtype=torch.int64, device=dev)
        self.counter = torch.zeros(1, dtype=torch.int32, device=dev)
        self.stop = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.tm_of = np.empty(self.tiles, np.int64)
        self.tn_of = np.empty(self.tiles, np.int64)
        for t in range(self.tiles):
            self.tm_of[t], self.tn_of[t] = raster(t, self.tiles_m, self.tiles_n)

    @property
    def rb(self) -> int:
        """A tile's blocks of C~ (and of U), 4,096 bytes each."""
        return self.T * self.T * 4 // receipt.BLOCK

    @property
    def fp8_flops_tile(self) -> float:
        return 2.0 * self.T * self.T * self.K

    @property
    def bf16_flops_tile(self) -> float:
        return 2.0 * self.T * self.T * PEEL_ROW

    def with_tile(self, T: int, qcap: int) -> "Filler":
        """The same call's operands, run in T x T tiles."""
        return Filler(self.M, self.N, self.K, self.at, self.bt, self.pa, self.pb, self.ia, self.ib, qcap, T)

    def args(self, first: int, count: int, *, log: bool = True, stop: bool = True, c=None, u=None, y=None,
             gap_ns: int = 0, rc=None) -> Args:
        a = Args()
        p = [self.at.data_ptr(), self.bt.data_ptr(), (c if c is not None else self.c).data_ptr(),
             (u if u is not None else self.u).data_ptr(), (y if y is not None else self.y).data_ptr(),
             self.pa.data_ptr(), self.pb.data_ptr(), self.ia.data_ptr(), self.ib.data_ptr(),
             (rc if rc is not None else self.rc).data_ptr(), 0,
             self.log.data_ptr() if log else 0, self.counter.data_ptr(), self.stop.data_ptr() if stop else 0,
             self.fold.data_ptr(), self.rkey.data_ptr()]
        for j, x in enumerate(p):
            a.p[j] = x
        for j, x in enumerate((self.M, self.N, self.K, RASTER, first, count, gap_ns)):
            a.i[j] = x
        return a

    def poison(self) -> None:
        for x in (self.c, self.u, self.y, self.rc):
            x.view(torch.uint8).fill_(0xA5)
        self.fold.zero_()
        self.log.zero_()
        self.counter.zero_()
        self.stop[0] = 0

    def tile_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.tiles_m, self.T, self.tiles_n, self.T)

    def written(self) -> np.ndarray:
        """How many times each tile (by raster position) was written, from the log."""
        pos = torch.nonzero(self.log).flatten().cpu().numpy()
        return np.bincount(pos % self.tiles, minlength=self.tiles)


_KEY: list[torch.Tensor] = []


def receipt_key() -> torch.Tensor:
    """fill.cu's p15 for RSEED (receipt.key), on the device, made once."""
    if not _KEY:
        _KEY.append(torch.from_numpy(receipt.key(RSEED)).to("cuda"))
    return _KEY[0]


def tile_words(x: np.ndarray, f: "Filler") -> np.ndarray:
    """An M x N matrix of 4-byte words as tiles x T*T, tile t (queue position) row-major."""
    T = f.T
    v = x.reshape(f.tiles_m, T, f.tiles_n, T).transpose(0, 2, 1, 3).reshape(f.tiles_m, f.tiles_n, T * T)
    return v[f.tm_of, f.tn_of]


def fold_of(c: np.ndarray, u: np.ndarray, f: Filler) -> np.ndarray:
    out = np.zeros(2 * f.tiles, np.uint32)
    for t in range(f.tiles):
        r, q, T = f.tm_of[t] * f.T, f.tn_of[t] * f.T, f.T
        out[2 * t] = np.bitwise_xor.reduce(c[r:r + T, q:q + T].ravel())
        out[2 * t + 1] = np.bitwise_xor.reduce(u[r:r + T, q:q + T].ravel())
    return out


def compare(f: Filler, store: int, ref_c, ref_u, ref_y, ref_fold, ref_rc, counts: np.ndarray) -> dict:
    """Every tile written at least once equals the reference (all, y) or has the reference's receipts (rcpt), or the fold
    equals the reference fold of each tile written an odd number of times and 0 otherwise (fold)."""
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
    elif store == 3:
        hit_t = torch.from_numpy(hit).to("cuda")
        bad += int((f.rc[hit_t] != ref_rc[hit_t]).any(dim=1).sum())
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
        self.ref = None
        self.rebind(stream)
        self.ref = self.out.clone()

    def rebind(self, stream: torch.cuda.Stream) -> None:
        """Capture the step's graph on `stream` (a green context's, so the step runs on that context's SMs alone)."""
        self.stream = stream
        self.graph = None
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.out = self.forward()
        self.graph = graph
        torch.cuda.synchronize()
        self.replay(1)
        torch.cuda.synchronize()

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
    """The SM clock of the leased GPU, sampled on a thread: NVML by UUID every 10 ms, or, without pynvml, `nvidia-smi -lms 20`
    streaming `clocks.sm`; `mean(t0, t1)` over host time. `source` says which (None: no clock, the peak's 2,100 MHz is
    used, which understates utilization below it)."""

    def __init__(self):
        self.samples: list[tuple[float, float]] = []
        self.nvml = None
        self.proc = None
        self.source = None
        self.error = None
        self.done = threading.Event()
        uuid = os.environ.get("GPU_LEASE_UUID")
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByUUID(uuid) if uuid else pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml, self.h, self.source = pynvml, h, "nvml"
            threading.Thread(target=self.run, daemon=True).start()
            return
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
        try:
            cmd = ["nvidia-smi", "-i", uuid or "0", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits", "-lms", "20"]
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.source = "nvidia-smi -lms 20"
            threading.Thread(target=self.read, daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.error = f"{self.error}; {e!r}"

    def run(self) -> None:
        while not self.done.is_set():
            try:
                self.samples.append((time.monotonic(), float(self.nvml.nvmlDeviceGetClockInfo(self.h, 1))))
            except Exception as e:  # noqa: BLE001
                self.error = repr(e)
            self.done.wait(0.01)

    def read(self) -> None:
        for line in self.proc.stdout:
            try:
                self.samples.append((time.monotonic(), float(line.strip())))
            except ValueError:
                continue

    def close(self) -> None:
        self.done.set()
        if self.proc is not None:
            self.proc.terminate()

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
                lib.fill(0, store, arm["grid"], f.args(pos, per_launch), stream, f.minb)
                pos += per_launch
            pumped[0] = pos

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
    elif f is not None and arm["mode"] == "persist":
        lib.fill(1, store, arm["grid"], f.args(0, min(f.qcap, arm.get("limit", f.qcap)), gap_ns=arm.get("gap_ns", 0)), lo, f.minb)
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
        "hbm_ratio": base["step_ms"] / out["step_ms"],
    }
    if credit_tile is not None and f is not None:
        res["credit_w1_per_s"] = (1 - CAP) * credit_tile * tiles / w
    return res


# ---------------------------------------------------------------- gates

def load_fixture(d: Path, qcap: int, T: int = TILE) -> tuple[Filler, dict]:
    man = json.loads((d / "check.json").read_text())
    m, n, k = man["shape"]
    v = man["variants"]["sm120"]

    def raw(part: dict, name: str, sub: str = "") -> np.ndarray:
        return np.fromfile(d / sub / part[name]["file"], np.uint8)

    common = man["common"]

    def to(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.copy()).to("cuda")

    f = Filler(m, n, k, to(raw(common, "at")), to(raw(common, "bt")), to(raw(v["files"], "pa_t", "sm120")),
               to(raw(v["files"], "pb_t", "sm120")), to(raw(common, "ia_a")), to(raw(common, "ia_b")), qcap, T)
    exp = {name: raw(v["files"], name, "sm120") for name in ("c", "u", "y")}
    exp["c32"] = exp["c"].view(np.uint32).reshape(m, n)
    exp["u32"] = exp["u"].view(np.uint32).reshape(m, n)
    return f, exp


def gate_fixture(lib: Lib, d: Path, lo: torch.cuda.Stream, step: Step | None, log, T: int = TILE) -> bool:
    f, exp = load_fixture(d, qcap=4096, T=T)
    want = {"c": exp["c"], "u": exp["u"], "y": exp["y"]}
    want_fold = fold_of(exp["c32"], exp["u32"], f)
    want_rc = receipt.all_receipts(RSEED, tile_words(exp["c32"], f), tile_words(exp["u32"], f))
    ok = True

    def check(name: str, store: int) -> None:
        nonlocal ok
        torch.cuda.synchronize()
        got = {"c": f.c, "u": f.u, "y": f.y}
        if store == 2:
            res = bool(np.array_equal(f.fold.cpu().numpy().view(np.uint32), want_fold))
            detail = {"fold": res}
        elif store == 3:
            res = bool(np.array_equal(f.rc.cpu().numpy(), want_rc))
            detail = {"receipts": res}
        else:
            keys = ("c", "u", "y") if store == 0 else ("y",)
            detail = {kk: bool(np.array_equal(got[kk].cpu().numpy().view(np.uint8).ravel(), want[kk])) for kk in keys}
            res = all(detail.values())
        ok &= res
        emit(log, {"gate": name, "tile": T, "store": store, "pass": res, **detail})

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    f.poison()
    lib.pearlc_d(sms, f.args(0, f.tiles, log=False, stop=False), lo)
    check("gemm_pearlc_d", 0)
    for store in STORES.values():
        for grid, per in ((1, f.tiles), (3, 5), (sms, 1)):
            f.poison()
            for pos in range(0, f.tiles, grid * per):
                lib.fill(0, store, grid, f.args(pos, min(grid * per, f.tiles - pos)), lo, f.minb)
            check(f"chunk grid={grid} per={per}", store)
        for grid in (1, 7, sms):
            f.poison()
            lib.fill(1, store, grid, f.args(0, f.tiles), lo, f.minb)
            check(f"persist grid={grid}", store)
        f.poison()
        lib.fill(1, store, 3, f.args(0, f.tiles, gap_ns=20000), lo, f.minb)
        check("persist grid=3 throttled 20 us a tile", store)
        f.poison()
        for pos in range(0, f.tiles, 3 * 5):
            lib.fill(0, store, 3, f.args(pos, min(3 * 5, f.tiles - pos), gap_ns=20000), lo, f.minb)
        check("chunk grid=3 per=5 throttled 20 us a tile", store)
        if step is not None:
            f.poison()
            step.replay(1)
            for pos in range(0, f.tiles, 8):
                lib.fill(0, store, 4, f.args(pos, min(8, f.tiles - pos)), lo, f.minb)
            step.replay(2)
            torch.cuda.synchronize()
            check("chunk beside the step", store)
            same = step.identical()
            ok &= same
            emit(log, {"gate": "step identical beside the fixture filler", "tile": T, "store": store, "pass": same})
    f.poison()
    f.stop[0] = 1
    lib.fill(1, 0, sms, f.args(0, f.tiles), lo, f.minb)
    torch.cuda.synchronize()
    stopped = int(f.counter.item()) == 0
    ok &= stopped
    emit(log, {"gate": "persistent CTAs read the pinned stop flag", "tile": T, "pass": stopped})
    return ok


def random_filler(M: int, N: int, K: int, qcap: int, seed: int = 7, T: int = TILE) -> Filler:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def codes(r, c):
        x = torch.randint(0, 256, (r, c), device="cuda", generator=g, dtype=torch.int32).to(torch.uint8)
        x[(x & 0x7F) == 0x7F] ^= 1                    # no NaN code
        return x.contiguous()

    def peel(r):
        return (torch.randn(r, PEEL_ROW // 2, device="cuda", generator=g) * 0.1).to(torch.bfloat16).view(torch.uint8).contiguous()


    def scale(r):
        return torch.full((r,), 1.0 / 1024, device="cuda", dtype=torch.float32)

    return Filler(M, N, K, codes(M, K), codes(N, K), peel(M), peel(N), scale(M), scale(N), qcap, T)


# ---------------------------------------------------------------- main

def credit_tile(M: int, N: int, K: int, T: int = TILE) -> float | None:
    try:
        sys.path.insert(0, str(ROOT))
        from verity.core.protocols.pouw.protocol import Shape
        from verity.core.protocols.pouw.schemes import pearl_c as PC
        from verity.core.protocols.pouw.schemes.pearl_c_device import SM120

        s = PC.PearlC(forming="v1", device=SM120, hashing="h3")
        return s.credit_of(Shape(M, K, N)) * T * T / (M * N)
    except Exception as e:  # noqa: BLE001
        print(f"credit_of unavailable: {e!r}", file=sys.stderr)
        return None


def arms(quick: bool, sms: int) -> list[dict]:
    """Family `stream`: the filler on a low-priority stream (or the step's priority) of the primary context. Persistent CTAs
    at 2 x SMs fill every SM's two slots and starve the step (the quick sweep, r20261012-214620-8387), so they are not run."""
    out = []
    pers = (1, 4) if quick else (1, 4, 16)
    stores = ("all", "fold") if quick else ("all", "y", "fold")
    for store in stores:
        for grid in (sms // 4, sms // 2, sms, 2 * sms):
            for per in pers:
                out.append({"family": "stream", "mode": "chunk", "grid": grid, "per": per, "store": store, "prio": "low"})
            if grid <= sms:
                out.append({"family": "stream", "mode": "persist", "grid": grid, "store": store, "prio": "low"})
        out.append({"family": "stream", "mode": "chunk", "grid": sms, "per": 4, "store": store, "prio": "same"})
    return out


def short_arms(sms: int, ks: list[int]) -> list[dict]:
    """Family `short`: a filler call of depth k, one tile a CTA a chunk, so a filler CTA holds its SM for a k-deep tile only."""
    out = []
    for k in ks:
        for grid in (sms // 4, sms // 2, sms, 2 * sms):
            out.append({"family": "short", "k": k, "mode": "chunk", "grid": grid, "per": 1, "store": "fold", "prio": "low"})
        out.append({"family": "short", "k": k, "mode": "chunk", "grid": sms, "per": 1, "store": "all", "prio": "low"})
    return out


def gc_targets(sms: int, quick: bool, cuts: str = "") -> list[int]:
    """Family `gc`: the step's green context's SMs (the driver rounds up to its granularity); the filler gets the rest
    (`cuts`, the SMs asked for the filler, else the defaults)."""
    cut = [int(x) for x in cuts.split(",") if x] if cuts else (0, 8, 16, 32, 64) if quick else (0, 4, 8, 12, 16, 24, 32, 48, 64, 94)
    return [sms - d for d in cut]


def gc_arms(gsms: int, fsms: int, span_s: float, sms: int, rate64: dict, rate128: dict) -> list[dict]:
    """The persistent filler on the `fsms` SMs beside the step's green context: (store, tile, CTAs an SM, duty). A duty < 1 is
    the throttle: each CTA idles `gap_ns` = its tile time x (1 / duty - 1) after each tile, its tile time taken from the
    filler alone on every SM (`rate*[store]` tiles/s at one CTA an SM), so the nominal rate is duty x the unthrottled one. The 64 x 64
    tile runs only on the smallest partitions, where it was the η line's best (r20261012-231033-73f3). `rcpt`, the tile's
    sketch-sha512 receipts, is the store the leaderboard's receipts-on rows take; `fold` and `all` run beside it as before."""
    out = []
    plan = [("rcpt", 128, 1, 1.0), ("rcpt", 128, 1, 0.5), ("rcpt", 128, 1, 0.25), ("rcpt", 128, 1, 0.1),
            ("fold", 128, 1, 1.0), ("fold", 128, 1, 0.25), ("all", 128, 1, 1.0)]
    if fsms <= 12:
        plan += [("rcpt", 64, 2, 1.0), ("rcpt", 64, 2, 0.5), ("fold", 64, 2, 1.0)]
    for store, T, per_sm, duty in plan:
        rate = (rate128 if T == 128 else rate64)[store]
        tile_ns = per_sm * sms / rate * 1e9
        out.append({"family": "gc", "mode": "persist", "tile": T, "grid": per_sm * fsms, "store": store, "prio": "low",
                    "step_sms": gsms, "fill_sms": fsms, "duty": duty, "gap_ns": int(tile_ns * (1 / duty - 1)),
                    "limit": int(rate * fsms / sms * span_s * duty) + 4 * per_sm * fsms})
    return out


def references(lib: Lib, f: Filler, sms: int, lo: torch.cuda.Stream) -> tuple:
    """`gemm_pearlc_d`'s C~, U and y of the filler's call, their per-tile fold, and every tile's receipts (receipt.py)."""
    ref_c, ref_u, ref_y = torch.empty_like(f.c), torch.empty_like(f.u), torch.empty_like(f.y)
    lib.pearlc_d(sms, f.args(0, f.tiles, log=False, stop=False, c=ref_c, u=ref_u, y=ref_y), lo)
    torch.cuda.synchronize()

    def tv(x: torch.Tensor) -> torch.Tensor:
        return x.view(f.tiles_m, f.T, f.tiles_n, f.T).permute(0, 2, 1, 3).reshape(f.tiles_m, f.tiles_n, -1)

    fc, fu = tv(ref_c).cpu().numpy().view(np.uint32), tv(ref_u).cpu().numpy().view(np.uint32)
    fold_np = np.zeros(2 * f.tiles, np.uint32)
    fold_np[0::2] = np.bitwise_xor.reduce(fc[f.tm_of, f.tn_of], axis=1)
    fold_np[1::2] = np.bitwise_xor.reduce(fu[f.tm_of, f.tn_of], axis=1)
    ref_fold = torch.from_numpy(fold_np.view(np.int32)).to("cuda")
    ref_rc = torch.from_numpy(receipt.all_receipts(RSEED, fc[f.tm_of, f.tn_of], fu[f.tm_of, f.tn_of])).to("cuda")
    return ref_c, ref_u, ref_y, ref_fold, ref_rc


def filler_alone(lib: Lib, f: Filler, refs: tuple, sms: int, lo: torch.cuda.Stream, clock: Clock, log, tag: str) -> dict | None:
    """Gate 2 and the filler alone: every store mode at the full grid, two passes over the problem. None if a tile differs."""
    alone = {}
    for store, code in STORES.items():
        f.poison()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        per = 2 * f.tiles
        e0.record(lo)
        for pos in range(0, per, sms * 4):
            lib.fill(0, code, sms, f.args(pos, min(sms * 4, per - pos), stop=False), lo, f.minb)
        e1.record(lo)
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        chk = compare(f, code, *refs, f.written())
        alone[store] = per / (ms * 1e-3)
        clk = clock.mean(time.monotonic() - 1, time.monotonic()) or PEAK_MHZ
        tfl = per * (f.fp8_flops_tile + f.bf16_flops_tile) / (ms * 1e-3) / 1e12
        emit(log, {"filler_alone": store, "filler": tag, "tiles_per_s": alone[store], "tflops": tfl,
                   "of_peak": per * (f.fp8_flops_tile / PEAK_E4M3 + f.bf16_flops_tile / PEAK_BF16) / (ms * 1e-3) / (clk / PEAK_MHZ),
                   "clock_mhz": clk, **chk})
        if chk["mismatched"]:
            emit(log, {"verdict": "REFUSED", "why": f"filler {tag} alone differs from gemm_pearlc_d ({store})"})
            return None
    return alone


def l2_probe(lib: Lib, path: Path, sms: int) -> None:
    """The launches the L2 write gate measures under ncu (`--kernel-name regex:^fill_`), in order, each with the L2 write and
    atomic sectors its outputs declare: C~, U (FP32) and y (BF16) of each tile (`all`), y (`y`), nothing (`fold`) or its 2 RB
    receipts (`rcpt`), plus the log's u64 a tile (a sector each); atomics: a warp's two fold words for each of a tile's 8
    warps (`fold`), the queue counter's add a tile plus the one each CTA takes past the end (persistent). The 128 x 128
    tile's rows are `STORE@128`. Controls (`control`, not judged): each kernel at no tiles, whose sectors are the launch's
    own, not its outputs'. Each kernel's L2_REPEATS judged launches are interleaved with L2_REPEATS + 1 controls (control,
    judged, control, ...), so l2_gate.py can ask whether the intermittent excess (271-7,352 sectors in some measurements and
    not others, r20261012-224930-ae0c, r20261013-002059-3970) comes to controls as it comes to judged launches. A zero-tile
    control lasts microseconds where a judged launch lasts milliseconds, so each fold kernel's launches also interleave
    L2_REPEATS dry launches (`dry`, not judged): the same tiles with the log off, which declare no write sectors and last
    as long as the judged launch."""
    f64 = random_filler(4096, 4096, 1024, qcap=4 * sms)
    lo = torch.cuda.Stream()
    rows = []
    for f in (f64, f64.with_tile(128, 4 * sms)):
        T = f.T
        out_sectors = {"all": T * T * (4 + 4 + 2) // 32, "y": T * T * 2 // 32, "fold": 0, "rcpt": 2 * f.rb * 64 // 32}
        suffix = "" if T == TILE else "128"
        for store, code in STORES.items():
            for persist in (0, 1):
                seq = ((0, False),) + ((4 * sms, False), (0, False)) * L2_REPEATS
                if store == "fold":
                    seq = ((0, False),) + ((4 * sms, False), (4 * sms, True), (0, False)) * L2_REPEATS
                for n, dry in seq:
                    f.poison()
                    torch.cuda.synchronize()
                    lib.fill(persist, code, sms, f.args(0, n, log=not dry), lo, f.minb)
                    torch.cuda.synchronize()
                    rows.append({"kernel": f"fill_{'persist' if persist else 'chunk'}{suffix}_1_{code}",
                                 "store": store if T == TILE else f"{store}@{T}", "tile": T, "tiles": n, "control": n == 0,
                                 "dry": dry, "logged": int((f.log > 0).sum()),
                                 "declared_write_sectors": 0 if dry else n * (out_sectors[store] + 1),
                                 "declared_atom_sectors": n * (16 if store == "fold" else 0) + (n + sms if persist else 0)})
    path.write_text(json.dumps({"filler": [4096, 4096, 1024], "launches": rows}, indent=1) + "\n")


def l2_of(l2: dict, arm: dict) -> str:
    """The L2 write gate's verdict for the arm's kernel (`none` for an arm without a filler)."""
    if "store" not in arm:
        return "none"
    kernel = f"fill_{'persist' if arm['mode'] == 'persist' else 'chunk'}{'128' if arm.get('tile', TILE) == 128 else ''}_1_" \
             f"{STORES[arm['store']]}"
    return l2.get("by_kernel", {}).get(kernel, l2.get("status", "not run"))


def run_arm(lib, f, refs, credit, step, reps, base, arm, lo, peer, clock, starve_s, head, log, l2) -> dict:
    out = window(lib, f, step, reps, dict(arm, peer=peer), lo, clock, starve_s)
    out["reps"] = reps
    chk = compare(f, STORES[arm["store"]], *refs, f.written()) if f is not None else {"tiles_checked": 0, "mismatched": 0}
    s = score(out, base, f, step, credit)
    row = {**head, "arm": arm, "gate_l2": l2_of(l2, arm), **out, **s, **chk}
    emit(log, row)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--fixture", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--configs", default="8b:16:0,8b:32:4096,70b-tp8:16:0,70b-tp8:32:4096", help="model:batch:ctx,...")
    ap.add_argument("--filler", default="4096x4096x8192", help="the filler call's MxNxK")
    ap.add_argument("--short-k", default="512,1024,2048", help="the short fillers' depths (family short); '' for none")
    ap.add_argument("--families", default="stream,short,gc")
    ap.add_argument("--window-ms", type=float, default=150.0, help="each arm's window, at least")
    ap.add_argument("--starve-s", type=float, default=5.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gc-cuts", default="", help="family gc: the filler's SMs asked for, e.g. 0,4,12,20,28,44,60,92")
    ap.add_argument("--repeats", type=int, default=1, help="family gc: each filled arm this many times (rows carry `rep`)")
    ap.add_argument("--budgets", default="1.01,1.02,1.05,1.10,1.25,1.5,2,4")
    ap.add_argument("--l2", type=Path, help="l2_gate.py's verdict, reported beside every timing")
    ap.add_argument("--l2-probe", type=Path, help="only the L2 write gate's launches (run under ncu), declared sectors to this file")
    args = ap.parse_args()
    if args.l2_probe:
        l2_probe(Lib(args.lib), args.l2_probe, torch.cuda.get_device_properties(0).multi_processor_count)
        return 0
    if args.fixture is None or args.out is None:
        ap.error("--fixture and --out are required")
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
    families = set(args.families.split(","))
    l2 = json.loads(args.l2.read_text()) if args.l2 and args.l2.exists() else {"status": "not run"}
    emit(log, {"device": props.name, "sms": sms, "torch": torch.__version__, "cuda": torch.version.cuda,
               "uuid": os.environ.get("GPU_LEASE_UUID"), "smi": clock.smi(), "clock": clock.source, "clock_error": clock.error,
               "stream_priority": {"lo": lo.priority, "hi": hi.priority, "peer": peer.priority},
               "occupancy": {f"{m}/{s}@{t}": lib.occupancy(p, STORES[s], 128 if t == 128 else 1)
                             for p, m in ((0, "chunk"), (1, "persist")) for s in STORES for t in (TILE, 128)},
               "fill_smem": {TILE: lib.l.fill_smem(1), 128: lib.l.fill_smem(128)}, "gate_l2": l2})

    # gate 1, with a small step beside it
    gstep = Step("8b", 16, 0, hi)
    for T in (TILE, 128):
        if not gate_fixture(lib, args.fixture, lo, gstep, log, T):
            emit(log, {"verdict": "REFUSED", "why": f"a fixture gate failed at the {T} x {T} tile"})
            return 4
    del gstep
    torch.cuda.empty_cache()

    M, N, K = (int(x) for x in args.filler.split("x"))
    fillers = {K: random_filler(M, N, K, qcap=1 << 23)}
    if "short" in families:
        for k in (int(x) for x in args.short_k.split(",") if x):
            fillers[k] = random_filler(M, N, k, qcap=1 << 25, seed=7 + k)
    refs, credits, alone = {}, {}, {}
    for k, f in fillers.items():
        refs[k] = references(lib, f, sms, lo)
        credits[k] = credit_tile(M, N, k)
        a = filler_alone(lib, f, refs[k], sms, lo, clock, log, f"{M}x{N}x{k}")
        if a is None:
            return 4
        alone[k] = a
    f = fillers[K]
    wide = f.with_tile(128, qcap=1 << 21)
    refs_w, credit_w = references(lib, wide, sms, lo), credit_tile(M, N, K, 128)
    alone_w = filler_alone(lib, wide, refs_w, sms, lo, clock, log, f"{M}x{N}x{K}@128")
    if alone_w is None:
        return 4
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(lo)
    lib.pearlc_d(sms, f.args(0, f.tiles, log=False, stop=False, c=refs[K][0], u=refs[K][1], y=refs[K][2]), lo)
    e1.record(lo)
    torch.cuda.synchronize()
    emit(log, {"pearlc_d_alone": True, "tiles_per_s": f.tiles / (e0.elapsed_time(e1) * 1e-3),
               "tflops": f.tiles * (f.fp8_flops_tile + f.bf16_flops_tile) / (e0.elapsed_time(e1) * 1e-3) / 1e12})

    greens: dict[int, dict] = {}
    if "gc" in families:
        for n in gc_targets(sms, args.quick, args.gc_cuts):
            g = lib.gc_split(n, hi.priority, lo.priority)
            if "error" not in g:
                g["spread_step"] = lib.sm_spread(g["step"], 4 * sms, graph=False)
                g["spread_step_graph"] = lib.sm_spread(g["step"], 4 * sms, graph=True)
                g["spread_fill"] = lib.sm_spread(g["fill"], 4 * sms, graph=False) if g["fill"] is not None else 0
            emit(log, {"green_context": n, **{k: v for k, v in g.items() if k not in ("step", "fill")}})
            if "error" not in g and g["step_sms"] not in {x["step_sms"] for x in greens.values()}:
                greens[n] = g

    budgets = [float(b) for b in args.budgets.split(",")]
    board = []
    failed = False
    keys = ("arm", "gate_l2", "slowdown", "p90_slowdown", "hbm_ratio", "util", "util_filler", "util_step", "filler_tflops",
            "credit_w1_per_s", "step_hbm_gbs", "clock_mhz")
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
        rows, curve = [], []
        todo = (arms(args.quick, sms) if "stream" in families else []) + \
            (short_arms(sms, [k for k in fillers if k != K]) if "short" in families else [])
        for arm in todo:
            k = arm.get("k", K)
            row = run_arm(lib, fillers[k], refs[k], credits[k], step, reps, base, arm, lo, peer, clock, args.starve_s,
                          head, log, l2)
            failed |= bool(row["mismatched"]) or not row["identical"]
            rows.append(row)
        for n, g in greens.items():
            try:
                step.rebind(g["step"])
            except Exception as e:  # noqa: BLE001
                emit(log, {**head, "green_context": n, "error": f"rebind: {e!r}"})
                break
            gsms, fsms = g["step_sms"], g["fill_sms"]
            row = run_arm(lib, None, None, None, step, reps, base, {"family": "gc", "mode": "alone", "step_sms": gsms},
                          lo, peer, clock, args.starve_s, head, log, l2)
            failed |= not row["identical"]
            curve.append({"step_sms": gsms, "slowdown": row["slowdown"], "hbm_ratio": row["hbm_ratio"],
                          "step_hbm_gbs": row["step_hbm_gbs"]})
            if g["fill"] is None:
                continue
            span_s = 3 * reps * row["step_ms"] * 1e-3
            for arm in (dict(a, rep=r) for r in range(args.repeats)
                        for a in gc_arms(gsms, fsms, span_s, sms, alone[K], alone_w)):
                ff, rr, cc = (wide, refs_w, credit_w) if arm["tile"] == 128 else (f, refs[K], credits[K])
                row = run_arm(lib, ff, rr, cc, step, reps, base, arm, g["fill"], peer, clock, args.starve_s, head, log, l2)
                failed |= bool(row["mismatched"]) or not row["identical"]
                rows.append(row)
        best = {}
        for b in budgets:
            ok = [r for r in rows if not r["starved"] and r["identical"] and not r["mismatched"] and r["slowdown"] <= b
                  and r["gate_l2"] != "REFUSED"]
            if ok:
                bu = max(ok, key=lambda x: x["util"])
                bc = max(ok, key=lambda x: x.get("credit_w1_per_s", 0.0))
                best[str(b)] = {"by_util": {k: bu[k] for k in keys if k in bu}, "by_credit": {k: bc[k] for k in keys if k in bc}}
            rc = [r for r in ok if r["arm"].get("store") == "rcpt"]
            if rc:
                bu, bc = max(rc, key=lambda x: x["util"]), max(rc, key=lambda x: x.get("credit_w1_per_s", 0.0))
                best[str(b)].update({"by_util_receipts": {k: bu[k] for k in keys if k in bu},
                                     "by_credit_receipts": {k: bc[k] for k in keys if k in bc}})
            loose = [r for r in rows if not r["starved"] and r["identical"] and not r["mismatched"] and r["slowdown"] <= b]
            if loose:
                bl = max(loose, key=lambda x: x.get("credit_w1_per_s", 0.0))
                best.setdefault(str(b), {})["by_credit_l2_ungated"] = {k: bl[k] for k in keys if k in bl}
        board.append({**head, "unfilled": {**base, **sbase}, "gc_curve": curve, "best": best})
        del step
        torch.cuda.empty_cache()
    (args.out / "leaderboard.json").write_text(json.dumps({"filler": [M, N, K], "credit_w1_per_tile": credits[K],
                                                          "gate_l2": l2, "clock": clock.source, "configs": board},
                                                         indent=1) + "\n")
    smi_end = clock.smi()
    clock.close()
    verdict = "FAILED" if failed else f"ok, L2 write gate {l2.get('status')} {l2.get('by_store', '')}".strip()
    emit(log, {"verdict": verdict, "smi_end": smi_end})
    return 5 if verdict == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
