"""The filler's GEMM with SHA-512 in separate CTAs on the same SMs, timed on one H200: corun.cubin (run.sh's build).

Gates first, in the same process: O of each GEMM kernel against torch._int_mm(A, B^T) word for word, O poisoned with 0xA5
before the call; every digest hash_rows writes over that O against hashlib's SHA-512 of the same 1 KiB, the digests
poisoned first. Then timed items of about `--item-s` seconds each, interleaved over `--rounds`:

- gemm (gemm_1, and gemm_2 when 66 clusters of two fit): one persistent launch of 132 CTAs, `passes` passes over the
  tiles, calibrated to the item's length;
- hash (hash_rows, the 8-warp CTA one an SM, held there by dynamic shared memory; hash_chain at 8 and at 16 warps an
  SM, the bar's own form): until the item's time;
- corun (each gemm with hash_rows, and with hash_chain): the GEMM launched first, then on a second stream gate_k, which
  waits until all 132 GEMM CTAs have started, then the hash, which stops when the GEMM's last CTA sets `stop`;
- int_mm: torch._int_mm on the same operands, calls back to back, the cuBLASLt baseline (no clock64: NVML's clock).

Each item records every CTA's stamps, the placement (an item counts only with one GEMM CTA and the stated hash CTAs on
every SM) and NVML's SM clock and clock event reasons every 10 ms. After the items: O against the reference again, and the
co-run's digests.
Rates are summary.py's. Writes items.jsonl and table.md into --out; exit 1 if a gate failed.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s8_linear"))
sys.path.insert(0, str(HERE))
import summary as S  # noqa: E402
import wgmma as W  # noqa: E402

TM, BN, BK = 128, 256, 128
GEMM_THREADS, HASH_THREADS = 256, 256
HASH_KINDS = ("hash_rows", "hash_rows_l1", "hash_chain", "idle_k")
HASHES = "hash_rows:8:1:0,hash_chain:8:1:0,hash_chain:8:2:0"
CORUNS = "hash_rows:8:0,hash_chain:8:0"
LOGMAX = 4096
MSG_BYTES = 1024
NUM_REGS, PREFERRED_CARVEOUT = 4, 9
u, i32, vp, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_uint64, C.POINTER
CUDA = dict(W.CUDA, cuFuncGetAttribute=[P(i32), i32, vp])


class Params(C.Structure):
    """corun.cu's Params."""
    _fields_ = [("a", C.c_ubyte * 128), ("b", C.c_ubyte * 128), ("o", u64), ("M", i32), ("N", i32), ("K", i32),
                ("group", i32), ("passes", i32), ("log_every", i32), ("log", u64), ("stamps", u64), ("started", u64),
                ("done", u64), ("stop", u64), ("pad", C.c_ubyte * 56)]


class HashArgs(C.Structure):
    _fields_ = [("msg", u64), ("dig", u64), ("stop", u64), ("stamps", u64), ("nmsg", u64), ("dur_ns", u64),
                ("most", u64), ("nap_ns", u), ("pad", u)]


assert C.sizeof(Params) == 384 and Params.o.offset == 256 and Params.M.offset == 264 and Params.log.offset == 288
STAMP_WORDS = 6  # t0, t1, c0, c1, (sm | n << 32), work


class Lib:
    def __init__(self, path: Path):
        self.cubin = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(self.cubin).hexdigest()
        self.cu = C.CDLL(W.settings.S8_LIBCUDA)
        for f, a in CUDA.items():
            getattr(self.cu, f).argtypes, getattr(self.cu, f).restype = a, i32
        torch.cuda.init()
        dev, ctx = i32(), vp()
        self.ck("cuDeviceGet", C.byref(dev), torch.cuda.current_device())
        self.ck("cuDevicePrimaryCtxRetain", C.byref(ctx), dev)
        self.ck("cuCtxSetCurrent", ctx)
        self.mod = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod), self.cubin)
        n = i32()
        self.ck("cuDeviceGetAttribute", C.byref(n), W.SM_COUNT, dev)
        self.sms = n.value
        self.f = {k: self.fn(k) for k in ("gemm_1", "gemm_2", *HASH_KINDS, "gate_k", "corun_layout")}
        out = torch.zeros(16, dtype=torch.int32, device="cuda")
        self.launch("corun_layout", 1, 1, 0, [u64(out.data_ptr())])
        torch.cuda.synchronize()
        lay = out.tolist()
        want = [C.sizeof(Params), Params.o.offset, Params.M.offset, Params.log.offset, 8 * STAMP_WORDS]
        if lay[:5] != want or lay[7] != GEMM_THREADS or lay[8] != LOGMAX or lay[9] != C.sizeof(HashArgs):
            raise RuntimeError(f"corun_layout {lay} is not corun.py's {want}: rebuild or update")
        self.smem, self.stages = lay[5], lay[6]
        for k in ("gemm_1", "gemm_2"):
            self.ck("cuFuncSetAttribute", self.f[k], W.MAX_DYN_SMEM, self.smem)
            self.ck("cuFuncSetAttribute", self.f[k], PREFERRED_CARVEOUT, 100)
        for k in HASH_KINDS:
            self.ck("cuFuncSetAttribute", self.f[k], W.MAX_DYN_SMEM, 120 * 1024)
            self.ck("cuFuncSetAttribute", self.f[k], PREFERRED_CARVEOUT, 100)
        self.regs = {k: self.attr(self.f[k], NUM_REGS) for k in self.f}
        cfg = W.LaunchConfig((u * 3)(2, 1, 1), (u * 3)(GEMM_THREADS, 1, 1), self.smem, None, None, 0)
        self.ck("cuOccupancyMaxActiveClusters", C.byref(n), self.f["gemm_2"], C.byref(cfg))
        self.clusters = n.value

    def ck(self, f: str, *a) -> None:
        r = getattr(self.cu, f)(*a)
        if r:
            s = C.c_char_p()
            self.cu.cuGetErrorString(r, C.byref(s))
            raise RuntimeError(f"{f} returned CUresult {r} ({s.value.decode() if s.value else 'unknown'})")

    def fn(self, name: str) -> vp:
        f = vp()
        self.ck("cuModuleGetFunction", C.byref(f), self.mod, name.encode())
        return f

    def attr(self, f: vp, a: int) -> int:
        v = i32()
        self.ck("cuFuncGetAttribute", C.byref(v), a, f)
        return v.value

    def launch(self, name: str, grid: int, threads: int, shared: int, args: list, stream=None) -> None:
        params = (vp * len(args))(*[C.cast(C.pointer(a), vp) for a in args])
        s = (stream or torch.cuda.current_stream()).cuda_stream
        self.ck("cuLaunchKernel", self.f[name], grid, 1, 1, threads, 1, 1, shared, vp(s), params, None)

    def tmap(self, t: torch.Tensor, box_rows: int) -> bytes:
        rows, depth = t.shape
        raw = C.create_string_buffer(128 + 64)
        at = (C.addressof(raw) + 63) & ~63
        dims, strides = (u64 * 2)(depth, rows), (u64 * 1)(depth)
        box, estr = (u * 2)(BK, box_rows), (u * 2)(1, 1)
        self.ck("cuTensorMapEncodeTiled", vp(at), W.TMAP_UINT8, 2, vp(t.data_ptr()), dims, strides, box, estr, 0,
                W.TMAP_SWIZZLE_128B, W.TMAP_L2_128B, 0)
        return C.string_at(at, 128)


class Clocks:
    """NVML on the leased GPU (GPU_LEASE_UUID, else device 0): the SM clock and the clock event reasons every 10 ms."""

    def __init__(self):
        import pynvml as N
        self.N = N
        N.nvmlInit()
        uuid = os.environ.get("GPU_LEASE_UUID")
        self.h = N.nvmlDeviceGetHandleByUUID(uuid) if uuid else N.nvmlDeviceGetHandleByIndex(0)
        self.samples: list[tuple[int, int]] = []
        self._stop = threading.Event()

    def _run(self) -> None:
        N = self.N
        reasons = getattr(N, "nvmlDeviceGetCurrentClocksEventReasons", None) or N.nvmlDeviceGetCurrentClocksThrottleReasons
        while not self._stop.wait(0.01):
            self.samples.append((N.nvmlDeviceGetClockInfo(self.h, N.NVML_CLOCK_SM), reasons(self.h)))

    def __enter__(self):
        self.samples, self._stop = [], threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join()

    def record(self) -> dict:
        mhz = [x[0] for x in self.samples]
        reasons = 0
        for x in self.samples:
            reasons |= x[1]
        return {"smi_mhz_p50": statistics.median(mhz) if mhz else None, "smi_mhz_min": min(mhz) if mhz else None,
                "clock_reasons": f"0x{reasons:x}", "samples": len(mhz)}


def stamps_of(t: torch.Tensor) -> list[S.Stamp]:
    out = []
    for r in t.view(-1, STAMP_WORDS).cpu().tolist():
        r = [x & 0xFFFFFFFFFFFFFFFF for x in r]
        out.append(S.Stamp(r[0], r[1], r[2], r[3], r[4] & 0xFFFFFFFF, r[4] >> 32, r[5]))
    return out


class Bench:
    def __init__(self, lib: Lib, m: int, n: int, k: int, group: int, seed: int):
        assert m % (2 * TM) == 0 and n % BN == 0 and k % BK == 0, (m, n, k)
        self.lib, self.m, self.n, self.k, self.group = lib, m, n, k, group
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.a = torch.randint(0, 4, (m, k), dtype=torch.int8, device="cuda", generator=g)
        self.b = torch.randint(0, 4, (n, k), dtype=torch.int8, device="cuda", generator=g)
        self.ref = torch._int_mm(self.a, self.b.t())
        self.o = torch.empty((m, n), dtype=torch.int32, device="cuda")
        self.nmsg = m * n * 4 // MSG_BYTES
        self.dig = torch.empty((self.nmsg, 8), dtype=torch.int64, device="cuda")
        self.maps = {cl: (lib.tmap(self.a, TM), lib.tmap(self.b, BN // cl)) for cl in (1, 2)}
        self.tiles = (m // TM) * (n // BN)
        self.macs_per_tile = TM * BN * k
        self.ctr = torch.zeros(4, dtype=torch.int32, device="cuda")  # started, done, stop, never-set
        self.gstamps = torch.zeros((lib.sms, STAMP_WORDS), dtype=torch.int64, device="cuda")
        self.glog = torch.zeros((lib.sms, LOGMAX), dtype=torch.int64, device="cuda")
        self.hstamps = torch.zeros((2 * lib.sms, STAMP_WORDS), dtype=torch.int64, device="cuda")
        self.sg, self.sh = torch.cuda.Stream(), torch.cuda.Stream()
        self.ref_digests: torch.Tensor | None = None

    def gemm_params(self, cl: int, passes: int, log_every: int) -> Params:
        p = Params()
        a, b = self.maps[cl]
        C.memmove(C.addressof(p) + Params.a.offset, a, 128)
        C.memmove(C.addressof(p) + Params.b.offset, b, 128)
        p.o, p.M, p.N, p.K, p.group = self.o.data_ptr(), self.m, self.n, self.k, self.group
        p.passes, p.log_every, p.log = passes, log_every, self.glog.data_ptr()
        p.stamps = self.gstamps.data_ptr()
        base = self.ctr.data_ptr()
        p.started, p.done, p.stop = base, base + 4, base + 8
        return p

    def hash_args(self, dur_s: float, stop: bool, most: int = 0, nap_ns: int = 0) -> HashArgs:
        base = self.ctr.data_ptr()
        return HashArgs(self.o.data_ptr(), self.dig.data_ptr(), base + 8 if stop else base + 12,
                        self.hstamps.data_ptr(), self.nmsg, int(dur_s * 1e9), most, nap_ns, 0)

    def reset(self) -> None:
        self.ctr.zero_()
        self.gstamps.zero_()
        self.glog.zero_()
        self.hstamps.zero_()
        torch.cuda.synchronize()

    def gemm(self, cl: int, passes: int, log_every: int = 0, stream=None) -> Params:
        p = self.gemm_params(cl, passes, log_every)
        self.lib.launch(f"gemm_{cl}", self.lib.sms, GEMM_THREADS, self.lib.smem, [p], stream)
        return p

    def hash(self, kind: str, threads: int, per_sm: int, args: HashArgs, alone: bool, stream=None) -> None:
        # alone, dynamic shared memory holds the CTAs to per_sm an SM (233,472 bytes an SM, 1 KB more a CTA), as the
        # GEMM CTA holds the hash beside it
        shared = (120 * 1024 if per_sm == 1 else 100 * 1024) if alone else 0
        self.lib.launch(kind, self.lib.sms * per_sm, threads, shared, [args], stream)

    # ---------------------------------------------------------------- gates
    def gate_gemm(self, cl: int) -> dict:
        self.reset()
        self.o.view(torch.uint8).fill_(0xA5)
        self.gemm(cl, 1)
        torch.cuda.synchronize()
        bad = int((self.o != self.ref).sum())
        return {"gate": f"gemm_{cl}", "ok": bad == 0, "mismatches": bad, "cells": self.m * self.n}

    def reference_digests(self) -> torch.Tensor:
        if self.ref_digests is None:
            raw = self.ref.cpu().numpy().tobytes()
            words = bytearray()
            for i in range(self.nmsg):
                words += hashlib.sha512(raw[i * MSG_BYTES:(i + 1) * MSG_BYTES]).digest()
            import numpy as np
            self.ref_digests = torch.from_numpy(np.frombuffer(bytes(words), dtype=">u8").astype(np.uint64).view(np.int64)).view(-1, 8)
        return self.ref_digests

    def check_digests(self) -> dict:
        got = self.dig.cpu()
        bad = int((got != self.reference_digests()).any(dim=1).sum())
        return {"messages": self.nmsg, "mismatches": bad, "ok": bad == 0}

    def gate_hash(self, kind: str) -> dict:
        self.o.copy_(self.ref)
        self.reset()
        self.dig.view(torch.uint8).fill_(0xA5)
        threads = self.lib.sms * HASH_THREADS
        self.hash(kind, HASH_THREADS, 1, self.hash_args(60.0, False, most=-(-self.nmsg // threads)), alone=True)
        torch.cuda.synchronize()
        return {"gate": kind, **self.check_digests()}

    # ---------------------------------------------------------------- items
    def calibrate(self, cl: int, item_s: float) -> int:
        self.reset()
        self.gemm(cl, 2)
        torch.cuda.synchronize()
        p = 40
        self.reset()
        t = time.monotonic()
        self.gemm(cl, p)
        torch.cuda.synchronize()
        return max(1, round(p * item_s / (time.monotonic() - t)))

    def log_every(self, passes: int) -> int:
        per_cta = passes * self.tiles // self.lib.sms
        return max(1, -(-per_cta // (LOGMAX - 8)))

    def item_gemm(self, pw: Clocks, cl: int, passes: int) -> dict:
        self.reset()
        le = self.log_every(passes)
        with pw:
            self.gemm(cl, passes, le)
            torch.cuda.synchronize()
        g = stamps_of(self.gstamps)
        r = S.gemm_rate(g, self.macs_per_tile)
        return {"arm": f"gemm_{cl}", "passes": passes, "placement": S.placement(g, self.lib.sms), **r,
                "nvml": pw.record()}

    def item_hash(self, pw: Clocks, kind: str, warps: int, per_sm: int, nap_ns: int, item_s: float) -> dict:
        self.reset()
        with pw:
            self.hash(kind, 32 * warps, per_sm, self.hash_args(item_s, False, nap_ns=nap_ns), alone=True)
            torch.cuda.synchronize()
        h = stamps_of(self.hstamps[: self.lib.sms * per_sm])
        return {"arm": f"{kind}:{warps}w:{per_sm}" + (f":nap{nap_ns}" if nap_ns else ""), "per_sm": per_sm,
                "placement": S.placement(h, self.lib.sms), **S.hash_rate(h, self.lib.sms), "nvml": pw.record()}

    def item_corun(self, pw: Clocks, cl: int, passes: int, kind: str, warps: int, nap_ns: int) -> dict:
        self.reset()
        le = self.log_every(passes)
        args = self.hash_args(3600.0, True, nap_ns=nap_ns)
        started = C.c_uint64(self.ctr.data_ptr())
        with pw:
            self.gemm(cl, passes, le, self.sg)
            self.lib.launch("gate_k", 1, 1, 0, [started, i32(self.lib.sms)], self.sh)
            self.hash(kind, 32 * warps, 1, args, alone=False, stream=self.sh)
            torch.cuda.synchronize()
        g, h = stamps_of(self.gstamps), stamps_of(self.hstamps[: self.lib.sms])
        r = S.gemm_rate(g, self.macs_per_tile)
        hr = S.hash_rate(h, self.lib.sms)
        logs = [[x for x in row if x] for row in self.glog.cpu().tolist()]
        w0, w1 = S.window(h)
        inside = S.gemm_rate_in(logs, le, self.macs_per_tile, w0, w1, hr["mhz"])
        arm = f"gemm_{cl}+{kind}" + (f":{warps}w" if warps != HASH_THREADS // 32 else "") + (f":nap{nap_ns}" if nap_ns else "")
        return {"arm": arm, "passes": passes, "gemm_placement": S.placement(g, self.lib.sms),
                "hash_placement": S.placement(h, self.lib.sms), "covers": S.covers(g, h), "gemm": r, "hash": hr,
                "gemm_in_hash_window": inside, "gap": S.gap(inside["peak_share"], hr["bar"]),
                "nvml": pw.record()}

    def item_int_mm(self, pw: Clocks, item_s: float) -> dict:
        torch._int_mm(self.a, self.b.t())
        torch.cuda.synchronize()
        t = time.monotonic()
        for _ in range(5):
            torch._int_mm(self.a, self.b.t())
        torch.cuda.synchronize()
        calls = max(1, round(5 * item_s / (time.monotonic() - t)))
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        with pw:
            ev[0].record()
            for _ in range(calls):
                torch._int_mm(self.a, self.b.t())
            ev[1].record()
            torch.cuda.synchronize()
        s = ev[0].elapsed_time(ev[1]) * 1e-3
        macs = calls * self.m * self.n * self.k
        pr = pw.record()
        mhz = pr["smi_mhz_p50"] or S.BAR_MHZ
        per_clock = macs / s / (self.lib.sms * mhz * 1e6)
        return {"arm": "int_mm", "calls": calls, "s": s, "macs": macs, "theta": macs / s / S.THETA_MACS_PER_S,
                "smi_mhz": mhz, "macs_per_clock_at_smi": per_clock, "nvml": pr}


def table(rows: list[dict]) -> str:
    """The median of each arm's items, one line an arm. κ: a co-run's GEMM rate over its GEMM's alone."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        if "arm" in r:
            by.setdefault(r["arm"], []).append(r)
    med = statistics.median
    alone = {arm: med(x["macs_per_clock"] for x in xs) for arm, xs in by.items() if arm.startswith("gemm_") and "+" not in arm}
    out = ["| arm | items | MACs/clk (of 4,096) | θ | κ | SM MHz | hash (of the bar) | hash G/s | gap to 0.985 | gap to 0.414 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, xs in by.items():
        if arm.startswith("gemm_") and "+" in arm:
            g = [x["gemm_in_hash_window"] for x in xs]
            h = [x["hash"] for x in xs]
            share, theta, mpc = med(x["peak_share"] for x in g), med(x["theta"] for x in g), med(x["macs_per_clock"] for x in g)
            base = alone.get(arm.split("+")[0])
            kappa = f"{mpc / base:.4f}" if base else "-"
            bar = med(x["bar"] for x in h)
            out.append(f"| {arm} | {len(xs)} | {mpc:.0f} ({share:.4f}) | {theta:.4f} | {kappa} | {med(x['mhz'] for x in h):.0f} | "
                       f"{bar:.3f} | {med(x['g_per_s'] for x in h):.3f} | {S.TARGET_THETA - share:+.4f} | {S.TARGET_BAR - bar:+.3f} |")
        elif arm.startswith("gemm_"):
            share = med(x["peak_share"] for x in xs)
            out.append(f"| {arm} | {len(xs)} | {med(x['macs_per_clock'] for x in xs):.0f} ({share:.4f}) | "
                       f"{med(x['theta'] for x in xs):.4f} | 1 | {med(x['mhz'] for x in xs):.0f} | - | - | "
                       f"{S.TARGET_THETA - share:+.4f} | - |")
        elif arm == "int_mm":
            out.append(f"| {arm} | {len(xs)} | {med(x['macs_per_clock_at_smi'] for x in xs):.0f} at NVML's clock | "
                       f"{med(x['theta'] for x in xs):.4f} | - | {med(x['smi_mhz'] for x in xs):.0f} (NVML) | - | - | - | - |")
        else:
            bar = med(x["bar"] for x in xs)
            out.append(f"| {arm} | {len(xs)} | - | - | - | {med(x['mhz'] for x in xs):.0f} | {bar:.3f} | "
                       f"{med(x['g_per_s'] for x in xs):.3f} | - | {S.TARGET_BAR - bar:+.3f} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cubin", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--m", type=int, default=8448)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--item-s", type=float, default=2.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arms", default="gemm,hash,corun,int_mm")
    ap.add_argument("--gemms", default="1,2", help="the GEMMs timed: cluster sizes")
    ap.add_argument("--hashes", default=HASHES, help="hash items alone: KIND:WARPS:CTAS_AN_SM:NAP_NS, comma-separated")
    ap.add_argument("--coruns", default=CORUNS, help="hash sides of the co-runs, beside each GEMM: KIND:WARPS:NAP_NS")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    lib = Lib(a.cubin)
    f = open(a.out / "items.jsonl", "w")

    def emit(doc: dict) -> None:
        f.write(json.dumps(doc) + "\n")
        f.flush()
        print(json.dumps(doc), flush=True)

    emit({"device": torch.cuda.get_device_name(), "sms": lib.sms, "cubin_sha256": lib.sha256, "regs": lib.regs,
          "smem": lib.smem, "stages": lib.stages, "clusters_of_two": lib.clusters, "m": a.m, "n": a.n, "k": a.k,
          "group": a.group, "operands": "u2: bytes uniform in [0, 3]", "torch": torch.__version__})
    b = Bench(lib, a.m, a.n, a.k, a.group, a.seed)
    want = [int(x) for x in a.gemms.split(",")]
    cls = [cl for cl in want if cl == 1 or lib.clusters * 2 >= lib.sms]
    ok = True
    for cl in (1, 2):
        g = b.gate_gemm(cl)
        g["timed"] = cl in cls
        emit(g)
        ok &= g["ok"] or cl not in cls
    for kind in ("hash_rows_l1", "hash_rows"):
        g = b.gate_hash(kind)
        emit(g)
        ok &= g["ok"]
    if not ok:
        emit({"verdict": "a gate failed: nothing timed"})
        return 1
    pw = Clocks()
    arms = set(a.arms.split(","))
    passes = {cl: b.calibrate(cl, a.item_s) for cl in cls}
    emit({"calibrated_passes": passes})
    rows = []
    for r in range(a.rounds):
        items = []
        if "gemm" in arms:
            items += [lambda cl=cl: b.item_gemm(pw, cl, passes[cl]) for cl in cls]
        if "hash" in arms:
            for spec in a.hashes.split(","):
                k, w, per, nap = spec.split(":")
                items.append(lambda k=k, w=int(w), per=int(per), nap=int(nap): b.item_hash(pw, k, w, per, nap, a.item_s))
        if "corun" in arms:
            for cl in cls:
                for spec in a.coruns.split(","):
                    k, w, nap = spec.split(":")
                    items.append(lambda cl=cl, k=k, w=int(w), nap=int(nap): b.item_corun(pw, cl, passes[cl], k, w, nap))
        if "int_mm" in arms:
            items += [lambda: b.item_int_mm(pw, a.item_s)]
        for it in items:
            doc = {"round": r, **it()}
            rows.append(doc)
            emit(doc)
    # the last co-run's O and digests against the reference
    after = {"after": "o", "ok": bool(torch.equal(b.o, b.ref))}
    emit(after)
    d = {"after": "digests", **b.check_digests()}
    emit(d)
    ok &= after["ok"] and d["ok"]
    for x in rows:
        if x["arm"].startswith("gemm_") and "+" in x["arm"]:
            good = x["gemm_placement"] == (1, 1) and x["hash_placement"] == (1, 1) and x["covers"]
        elif "placement" in x:
            good = x["placement"] == (x.get("per_sm", 1),) * 2
        else:
            good = True
        if not good:
            emit({"refused": x["arm"], "round": x["round"], "why": "placement or coverage"})
            ok = False
    t = table([x for x in rows])
    (a.out / "table.md").write_text(t)
    print(t)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
