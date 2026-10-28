"""The filler's exact GEMM on 2-bit operands streamed packed (four values a byte) and unpacked on chip, sustained at the
H200's power cap, alone and beside SHA-512, against the same schedule on int8-stored operands and cuBLASLt.

Each arm is gated before it is timed, in the same run: O, poisoned with 0xA5 first, must equal `torch._int_mm(A, W.t())`
bit for bit at the timed shape (res: (K / 512) times one block's product in every tile; the unpack-only arms: O left
poisoned), and that reference agrees with int64 dot products on sampled cells.  A first call that hasn't ended in
HANG_S ends the run (exit 3).  Then an item runs the arm back to back for `--secs` and reports:

- tensor: MACs a SM clock against 4,096 (the SM clocks are %clock64 differences of `stamp_k` stamps, one on every SM,
  between segments of about a quarter second, so they count every cycle between launches too), and the rate against
  THETA = 4,096 x 132 x 1.83 GHz; for the unpack-only arms the same figure counts the MACs the unpacked operands would
  feed, so 1 / of_4096 is the unpacking's share of a k-block's MMA time;
- the board: mean watts from NVML's energy counter, the largest instantaneous sample, NVML's SM clock and throttle reasons,
  and pJ a MAC (mean watts over the MAC rate);
- beside the hash (`hash_k` CTAs on the GEMM's own SMs, launched once the GEMM's first call is resident): the hash's
  compressions a second over its span against the bar, 4.355 G/s (BAR_PER_SMCLK x 132 x 1.83 GHz), and the tensor rate
  and watts over the segments that span covers.

Every item's figures are its steady part: the segments that start at least SETTLE_S after it began.  Items interleave
arms round by round (`--rounds`).  Writes items.jsonl, gates.jsonl and summary.md under --out; the cubin is $U2_PACKED_CUBIN
(run.sh builds it).
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
import statistics
import struct
import threading
import time
from pathlib import Path

import torch

THETA = 4096 * 132 * 1.83e9  # MAC/s: s8 wgmma's 4,096 MACs a SM clock on 132 SMs at the 1,830 MHz wgmma runs at
BAR_PER_SMCLK = 0.01803  # SHA-512 compressions a SM clock, sx's: 4.355 G/s on 132 SMs at 1,830 MHz
BAR = BAR_PER_SMCLK * 132 * 1.83e9
SETTLE_S = 1.0
SEG_S = 0.25
HANG_S = 20.0
POISON = -1515870811  # 0xA5A5A5A5 as an int32
TM, BN, BK = 128, 256, 128
RES_K = 4 * BK  # res: the depth of the block its slots hold

u, i32, vp, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_uint64, C.POINTER
TMAP_UINT8, TMAP_SWIZZLE_NONE, TMAP_SWIZZLE_128B, TMAP_L2_128B, MAX_DYN_SMEM, SM_COUNT = 0, 0, 3, 2, 8, 16
CUDA = {
    "cuDeviceGet": [P(i32), i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuFuncSetAttribute": [vp, i32, i32], "cuDeviceGetAttribute": [P(i32), i32, i32],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuGetErrorString": [i32, P(C.c_char_p)],
    "cuTensorMapEncodeTiled": [vp, i32, u, vp, P(u64), P(u64), P(u), P(u), i32, i32, i32, i32],
}


class Params(C.Structure):
    """packed.cu's Params: the two tensor maps, O, the stamps, M, N, K, the raster group, the start counter."""
    _fields_ = [("a", C.c_ubyte * 128), ("b", C.c_ubyte * 128), ("o", u64), ("st", u64), ("M", C.c_int32),
                ("N", C.c_int32), ("K", C.c_int32), ("group", C.c_int32), ("started", u64), ("pad", C.c_ubyte * 88)]


class Stamp(C.Structure):
    _fields_ = [(f, u64) for f in ("t0", "t1", "c0", "c1", "sm", "tiles")]


assert C.sizeof(Params) == 384 and Params.o.offset == 256 and Params.M.offset == 272 and Params.started.offset == 288
assert C.sizeof(Stamp) == 48

#: arm -> (kernel, operands it streams: "i8", "p2" or "p2m" (W's pairs reversed), its shared-memory layout in pk_layout's
#: order, whether it multiplies)
KERNELS = {
    "i8": ("pk_i8", "i8", 0, True),
    "p2": ("pk_p2", "p2", 1, True),
    "p2u": ("pk_p2u", "p2", 1, True),
    "p2m": ("pk_p2m", "p2m", 1, True),
    "res": ("pk_res", "p2", 2, True),
    "p2x": ("pk_p2x", "p2", 1, False),
    "p2ux": ("pk_p2ux", "p2", 1, False),
    "p2mx": ("pk_p2mx", "p2m", 1, False),
}
HASH_THREADS = 256


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def unpack(p: torch.Tensor) -> torch.Tensor:
    """Packed rows [R, K / 4] (uint8) to their values [R, K] as int8: value k of a row is bits 2i and 2i + 1 of byte
    4 (k // 16) + k % 4, with i = (k // 4) % 4."""
    r, q = p.shape
    shifts = torch.arange(0, 8, 2, dtype=torch.uint8, device=p.device).view(1, 1, 4, 1)
    return ((p.view(r, q // 4, 1, 4) >> shifts) & 3).reshape(r, 4 * q).to(torch.int8)


def pack(v: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """unpack's inverse: values [R, K] in {0, 1, 2, 3} to packed rows [R, K / 4] (uint8).  reverse: each byte's pairs in
    the opposite order (value k at bits 2 (3 - i) and 2 (3 - i) + 1), the masked form's W."""
    r, k = v.shape
    if k % 16:
        raise ValueError(f"K {k} must be a multiple of 16")
    w = v.to(torch.uint8).view(r, k // 16, 4, 4)  # word, i, byte j
    if reverse:
        w = w.flip(2)
    out = w[:, :, 0] | (w[:, :, 1] << 2) | (w[:, :, 2] << 4) | (w[:, :, 3] << 6)
    return out.reshape(r, k // 4)


class Lib:
    """The cubin in torch's context: its kernels, their shared memory, tensor maps and launches."""

    def __init__(self, path: Path):
        self.cubin = Path(path).read_bytes()
        self.cu = C.CDLL("libcuda.so.1")
        for f, a in CUDA.items():
            getattr(self.cu, f).argtypes, getattr(self.cu, f).restype = a, i32
        torch.cuda.init()
        dev, ctx = i32(), vp()
        self.ck("cuDeviceGet", C.byref(dev), torch.cuda.current_device())
        self.ck("cuDevicePrimaryCtxRetain", C.byref(ctx), dev)
        self.ck("cuCtxSetCurrent", ctx)
        self.mod = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod), self.cubin)
        sms = i32()
        self.ck("cuDeviceGetAttribute", C.byref(sms), SM_COUNT, dev)
        self.sms = sms.value
        out = torch.zeros(8, dtype=torch.int32, device="cuda")
        self.launch(self.fn("pk_layout"), 1, 1, 0, [u64(out.data_ptr())])
        got = out.tolist()
        if got[:4] != [C.sizeof(Params), Params.o.offset, Params.M.offset, Params.started.offset]:
            raise RuntimeError(f"pk_layout {got[:4]} is not bench.py's Params: rebuild")
        self.smem = got[4:7]
        self.fns, self.maps = {}, {}
        for arm, (name, _, lay, _) in KERNELS.items():
            f = self.fn(name)
            self.ck("cuFuncSetAttribute", f, MAX_DYN_SMEM, self.smem[lay])
            self.fns[arm] = f
        self.hash_f, self.wait_f, self.stamp_f = self.fn("hash_k"), self.fn("wait_k"), self.fn("stamp_k")
        self.stamp_smem = 120 * 1024  # two of these never fit an SM, so stamp_k's CTAs fall one an SM
        self.ck("cuFuncSetAttribute", self.stamp_f, MAX_DYN_SMEM, self.stamp_smem)
        self.ck("cuFuncSetAttribute", self.hash_f, MAX_DYN_SMEM, self.stamp_smem)

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

    def launch(self, f, grid: int, threads: int, shared: int, args, stream: int | None = None) -> None:
        keep = list(args)
        params = (vp * len(keep))(*[C.addressof(a) for a in keep])
        s = torch.cuda.current_stream().cuda_stream if stream is None else stream
        self.ck("cuLaunchKernel", f, grid, 1, 1, threads, 1, 1, shared, vp(s), params, None)

    def tmap(self, ptr: int, rows: int, width: int, box_rows: int, box_width: int, swizzle: int) -> bytes:
        """A 2D map over rows of `width` bytes, boxes of box_rows x box_width bytes."""
        key = (ptr, rows, width, box_rows, box_width, swizzle)
        if key not in self.maps:
            raw = C.create_string_buffer(128 + 64)
            at = (C.addressof(raw) + 63) & ~63
            dims, strides = (u64 * 2)(width, rows), (u64 * 1)(width)
            box, estr = (u * 2)(box_width, box_rows), (u * 2)(1, 1)
            self.ck("cuTensorMapEncodeTiled", vp(at), TMAP_UINT8, 2, vp(ptr), dims, strides, box, estr, 0, swizzle,
                    TMAP_L2_128B, 0)
            self.maps[key] = C.string_at(at, 128)
        return self.maps[key]


class Gemm:
    """One arm at one shape: O = A W^T, s32, on our kernel; `call(stamps)` launches it on the current stream.  ops: the
    packed operands (ap, wp, and wpr, W with its pairs reversed) and the int8 ones (a, w); each arm reads its own."""

    def __init__(self, lib: Lib, arm: str, ops: dict, o: torch.Tensor, group: int = 8):
        self.lib, self.arm = lib, arm
        name, kind, lay, self.mma = KERNELS[arm]
        m, k = ops["a"].shape
        n = ops["w"].shape[0]
        if m % TM or n % BN or k % (RES_K if arm == "res" else BK):
            raise ValueError(f"{arm}: M {m} must be a multiple of {TM}, N {n} of {BN} and K {k} of "
                             f"{RES_K if arm == 'res' else BK}")
        self.threads, self.smem = 256, lib.smem[lay]
        self.units = (m // TM) * (n // BN)
        per = cdiv(self.units, lib.sms)
        self.grid = cdiv(self.units, per)
        self.rounds, self.macs = per, m * n * k
        self.p = Params()
        if kind == "i8":
            ma = lib.tmap(ops["a"].data_ptr(), m, k, TM, BK, TMAP_SWIZZLE_128B)
            mb = lib.tmap(ops["w"].data_ptr(), n, k, BN, BK, TMAP_SWIZZLE_128B)
        else:
            ma = lib.tmap(ops["ap"].data_ptr(), m, k // 4, TM, BK // 4, TMAP_SWIZZLE_NONE)
            wp = ops["wpr" if kind == "p2m" else "wp"]
            mb = lib.tmap(wp.data_ptr(), n, k // 4, BN, BK // 4, TMAP_SWIZZLE_NONE)
        C.memmove(C.addressof(self.p) + Params.a.offset, ma, 128)
        C.memmove(C.addressof(self.p) + Params.b.offset, mb, 128)
        self.p.o, self.p.M, self.p.N, self.p.K, self.p.group = o.data_ptr(), m, n, k, group
        self.f = lib.fns[arm]
        self.tensors = (ops, o)

    def call(self, stamps: int = 0, started: int = 0) -> None:
        self.p.st, self.p.started = stamps, started
        self.lib.launch(self.f, self.grid, self.threads, self.smem, [self.p])

    def describe(self) -> dict:
        return {"grid": self.grid, "threads": self.threads, "smem": self.smem, "units": self.units,
                "rounds": self.rounds, "balance": self.units / (self.rounds * self.grid), "mma": self.mma}


class Lt:
    """cuBLASLt's s8 x s8 -> s32 GEMM, as torch._int_mm(A, W.t()) runs it, on the unpacked operands."""

    mma = True

    def __init__(self, ops: dict, o: torch.Tensor):
        self.a, self.wt, self.o = ops["a"], ops["w"].t(), o
        self.macs = self.a.shape[0] * self.wt.shape[1] * self.a.shape[1]
        try:
            torch._int_mm(self.a, self.wt, out=self.o)
            self.out = True
        except (TypeError, RuntimeError):
            self.out = False

    def call(self, stamps: int = 0, started: int = 0) -> None:
        if self.out:
            torch._int_mm(self.a, self.wt, out=self.o)
        else:
            self.o.copy_(torch._int_mm(self.a, self.wt))


class Nvml:
    """The board's energy counter, power, SM clock and throttle reasons, sampled every 10 ms on a thread of its own."""

    FI_POWER_INSTANT = 186

    class Field(C.Structure):
        _fields_ = [("fieldId", u), ("scopeId", u), ("timestamp", C.c_longlong), ("latencyUsec", C.c_longlong),
                    ("valueType", i32), ("nvmlReturn", i32), ("value", C.c_ubyte * 8)]

    def __init__(self, uuid: str):
        self.nv = C.CDLL("libnvidia-ml.so.1")
        if self.nv.nvmlInit_v2():
            raise RuntimeError("nvmlInit_v2 failed")
        self.h = vp()
        if self.nv.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.h)):
            raise RuntimeError(f"no NVML device {uuid}")
        self.samples: list[tuple] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def read(self) -> tuple:
        e, p, mhz, why, temp = C.c_ulonglong(), u(), u(), C.c_ulonglong(), u()
        self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h, C.byref(e))
        self.nv.nvmlDeviceGetPowerUsage(self.h, C.byref(p))
        self.nv.nvmlDeviceGetClockInfo(self.h, 1, C.byref(mhz))
        self.nv.nvmlDeviceGetCurrentClocksThrottleReasons(self.h, C.byref(why))
        self.nv.nvmlDeviceGetTemperature(self.h, 0, C.byref(temp))
        fv = self.Field(fieldId=self.FI_POWER_INSTANT)
        inst = math.nan
        if self.nv.nvmlDeviceGetFieldValues(self.h, 1, C.byref(fv)) == 0 and fv.nvmlReturn == 0:
            raw = bytes(fv.value)
            if fv.valueType == 0:  # double
                inst = struct.unpack("<d", raw)[0] / 1e3
            else:
                inst = int.from_bytes(raw[:4] if fv.valueType in (1, 5) else raw[:2] if fv.valueType == 6 else raw,
                                      "little", signed=fv.valueType in (4, 5)) / 1e3
        return (time.time_ns(), e.value / 1e3, p.value / 1e3, inst, mhz.value, why.value, temp.value)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.read())
            time.sleep(0.01)

    def window(self, h0: int, h1: int) -> dict:
        """Mean watts (the energy counter's slope), the largest instantaneous and averaged samples, NVML's SM clock and the
        throttle reasons seen, over host ns [h0, h1]."""
        s = [x for x in self.samples if h0 <= x[0] <= h1]
        if len(s) < 3:
            return {"samples": len(s)}
        # the counter steps at its own cadence: the slope between its first and last change in the window
        ch = [x for i, x in enumerate(s) if i == 0 or x[1] != s[i - 1][1]]
        a, b = ch[0], ch[-1]
        mean_w = (b[1] - a[1]) / ((b[0] - a[0]) * 1e-9) if b[0] > a[0] else math.nan
        inst = [x[3] for x in s if not math.isnan(x[3])]
        return {"samples": len(s), "mean_w": mean_w, "avg_w_mean": statistics.mean(x[2] for x in s),
                "max_w": max(inst) if inst else math.nan, "max_avg_w": max(x[2] for x in s),
                "inst_w_mean": statistics.mean(inst) if inst else math.nan,
                "nvml_sm_mhz": statistics.median(x[4] for x in s), "throttle": sorted({hex(x[5]) for x in s}),
                "temp_c_max": max(x[6] for x in s)}

    def stop(self) -> None:
        self._stop.set()
        self._t.join()


class Run:
    def __init__(self, lib: Lib, nvml: Nvml, out: Path):
        self.lib, self.nvml, self.out = lib, nvml, out
        self.nsm = lib.sms
        self.stamps = torch.zeros((4096, self.nsm, 6), dtype=torch.int64, device="cuda")
        self.cta = torch.zeros((65536, 6), dtype=torch.int64, device="cuda")  # GEMM CTAs' own stamps, the last call's
        self.hst = torch.zeros((4 * self.nsm, 6), dtype=torch.int64, device="cuda")
        self.hio = torch.zeros(4 * self.nsm * HASH_THREADS, dtype=torch.int64, device="cuda")
        self.started = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.arrive = torch.zeros(4096, dtype=torch.int32, device="cuda")  # stamp_k's arrivals, a counter a segment
        self.side = torch.cuda.Stream()

    def stamp(self, seg: int) -> None:
        self.lib.launch(self.lib.stamp_f, self.nsm, 32, self.lib.stamp_smem,
                        [u64(self.stamps[seg].data_ptr()), u64(self.arrive[seg].data_ptr())])

    def hash(self, ctas: int, reps: int, stream: int, shared: int = 0) -> None:
        self.lib.launch(self.lib.hash_f, ctas, HASH_THREADS, shared,
                        [u64(self.hio.data_ptr()), i32(reps), u64(self.hst.data_ptr())], stream)

    def segments(self, nseg: int, calls: list[int], macs: int, g0: int) -> list[dict]:
        """Each segment between stamps s and s + 1: its span (globaltimer), every SM's clocks, its MACs."""
        st = self.stamps[: nseg + 1].cpu()
        out = []
        for s in range(nseg):
            a, b = st[s], st[s + 1]
            if (sorted(a[:, 4].tolist()) != sorted(b[:, 4].tolist()) or len(set(a[:, 4].tolist())) != self.nsm
                    or not 0 <= int(a[:, 4].min()) <= int(a[:, 4].max()) < self.nsm):
                raise RuntimeError(f"segment {s}: stamp_k's CTAs were not one on every SM")
            ia, ib = a[a[:, 4].argsort()], b[b[:, 4].argsort()]
            dt = (ib[:, 0] - ia[:, 0]).double()
            dc = (ib[:, 2] - ia[:, 2]).double()
            out.append({"t0": int(ia[:, 0].min()) - g0, "t1": int(ib[:, 0].max()) - g0, "ns": float(dt.mean()),
                        "clocks": float(dc.sum()), "mhz": float(dc.sum() / dt.sum() * 1e3), "macs": calls[s] * macs,
                        "gt0": int(ia[:, 0].min()), "gt1": int(ib[:, 0].max())})
        return out

    @staticmethod
    def rates(segs: list[dict], nsm: int) -> dict:
        macs, clocks, ns = sum(s["macs"] for s in segs), sum(s["clocks"] for s in segs), sum(s["ns"] for s in segs)
        if not segs or not ns:
            return {}
        return {"s": ns * 1e-9, "macs": macs, "macs_per_smclk": macs / clocks, "of_4096": macs / clocks / 4096,
                "sm_mhz": clocks / ns / nsm * 1e3, "tmac_s": macs / (ns * 1e-9) * 1e-12, "of_theta": macs / (ns * 1e-9) / THETA}

    def offset(self) -> int:
        """globaltimer - host time.time_ns(), to place NVML's samples on the GPU's spans."""
        self.arrive[4095].zero_()
        torch.cuda.synchronize()
        h = time.time_ns()
        self.stamp(4095)
        torch.cuda.synchronize()
        g = int(self.stamps[4095][:, 0].min())
        return g - h

    def item(self, name: str, op, secs: float, t_call: float, hash_ctas: int = 0, hash_reps: int = 0) -> dict:
        """`op` back to back for about secs, stamps every SEG_S; with hash_ctas, hash_k beside it from its first call."""
        per = max(1, round(SEG_S / t_call))
        nseg = max(2, round(secs / (per * t_call)))
        if nseg + 1 > 4095:
            raise ValueError("too many segments")
        off = self.offset()
        calls = [per] * nseg
        self.arrive[: nseg + 1].zero_()
        self.started.zero_()
        torch.cuda.synchronize()
        h0 = time.time_ns()
        self.stamp(0)
        cstamp = u64(self.cta.data_ptr()).value if isinstance(op, Gemm) else 0
        if hash_ctas:
            op.call(cstamp, self.started.data_ptr())
            self.lib.launch(self.lib.wait_f, 1, 32, 0, [u64(self.started.data_ptr()), u(op.grid)], self.side.cuda_stream)
            self.hash(hash_ctas, hash_reps, self.side.cuda_stream)
            first = 1
        else:
            first = 0
        for s in range(nseg):
            for _ in range(first if s == 0 else 0, per):
                op.call(cstamp)
            self.stamp(s + 1)
        hmid = None
        if hash_ctas:
            while not self.side.query():
                time.sleep(0.002)
            hmid = time.time_ns()
        torch.cuda.synchronize()
        h1 = time.time_ns()
        g0 = int(self.stamps[0][:, 0].min())
        segs = self.segments(nseg, calls, op.macs, g0)
        g_end = segs[-1]["gt1"]
        steady = [s for s in segs if s["gt0"] - g0 >= SETTLE_S * 1e9]
        rec = {"name": name, "secs": (g_end - g0) * 1e-9, "calls": sum(calls), "t_call": t_call, "mma": op.mma,
               "all": self.rates(segs, self.nsm), "steady": self.rates(steady, self.nsm),
               "last_half": self.rates([s for s in segs if s["gt0"] - g0 >= (g_end - g0) / 2], self.nsm),
               "power_steady": self.nvml.window(g0 + int(SETTLE_S * 1e9) - off, g_end - off),
               "power_all": self.nvml.window(h0, h1), "host_s": (h1 - h0) * 1e-9}
        if isinstance(op, Gemm):
            cta = self.cta[: op.grid].cpu()
            dt = (cta[:, 1] - cta[:, 0]).double()
            rec["cta_mhz_last_call"] = float((cta[:, 3] - cta[:, 2]).double().sum() / dt.sum() * 1e3)
            rec["cta_tiles"] = sorted(set(cta[:, 5].tolist()))
            rec["cta_sms"] = len(set(cta[:, 4].tolist()))
        if hash_ctas:
            hs = self.hst[:hash_ctas].cpu()
            w0, w1 = int(hs[:, 0].min()), int(hs[:, 1].max())
            comps = hash_ctas * HASH_THREADS * hash_reps
            hdt = (hs[:, 1] - hs[:, 0]).double()
            hmhz = float((hs[:, 3] - hs[:, 2]).double().sum() / hdt.sum() * 1e3)
            inside = [s for s in segs if s["gt0"] >= w0 + SETTLE_S * 1e9 and s["gt1"] <= w1]
            rec["hash"] = {"ctas": hash_ctas, "reps": hash_reps, "s": (w1 - w0) * 1e-9, "gcomp_s": comps / (w1 - w0),
                           "of_bar": comps / ((w1 - w0) * 1e-9) / BAR, "sm_mhz": hmhz,
                           "of_bar_per_smclk": comps / ((w1 - w0) * 1e-9) / (self.nsm * hmhz * 1e6) / BAR_PER_SMCLK,
                           "sms": len(set(hs[:, 4].tolist())),
                           "max_ctas_an_sm": max(hs[:, 4].tolist().count(v) for v in set(hs[:, 4].tolist())),
                           "start_spread_us": (int(hs[:, 0].max()) - w0) * 1e-3,
                           "end_spread_us": (w1 - int(hs[:, 1].min())) * 1e-3,
                           "covered": segs[0]["gt0"] <= w0 and g_end >= w1, "host_end_s": (hmid - h0) * 1e-9}
            rec["beside"] = self.rates(inside, self.nsm)
            rec["power_beside"] = self.nvml.window(w0 + int(SETTLE_S * 1e9) - off, w1 - off) if inside else {}
        p = rec["power_beside"] if hash_ctas and rec.get("power_beside") else rec["power_steady"]
        r = rec["beside"] if hash_ctas and rec.get("beside") else rec["steady"]
        if op.mma and r and p.get("mean_w"):
            rec["pj_per_mac"] = p["mean_w"] / (r["tmac_s"] * 1e12) * 1e12
        return rec

    def hash_alone(self, ctas_per_sm: int, secs: float) -> dict:
        """hash_k alone at ctas_per_sm CTAs an SM (the shared memory request keeps at most that many on one SM)."""
        ctas = ctas_per_sm * self.nsm
        shared = 100 * 1024 if ctas_per_sm == 2 else self.lib.stamp_smem
        self.hash(ctas, 64, torch.cuda.current_stream().cuda_stream, shared)
        torch.cuda.synchronize()
        t = time.perf_counter()
        self.hash(ctas, 2048, torch.cuda.current_stream().cuda_stream, shared)
        torch.cuda.synchronize()
        rep = (time.perf_counter() - t) / 2048
        reps = int(secs / rep)
        off = self.offset()
        h0 = time.time_ns()
        self.hash(ctas, reps, torch.cuda.current_stream().cuda_stream, shared)
        torch.cuda.synchronize()
        h1 = time.time_ns()
        hs = self.hst[:ctas].cpu()
        w0, w1 = int(hs[:, 0].min()), int(hs[:, 1].max())
        comps = ctas * HASH_THREADS * reps
        mhz = float((hs[:, 3] - hs[:, 2]).double().sum() / (hs[:, 1] - hs[:, 0]).double().sum() * 1e3)
        return {"name": f"hash x{ctas_per_sm}", "ctas": ctas, "reps": reps, "s": (w1 - w0) * 1e-9,
                "sms": len(set(hs[:, 4].tolist())), "gcomp_s": comps / (w1 - w0), "of_bar": comps / ((w1 - w0) * 1e-9) / BAR,
                "sm_mhz": mhz, "of_bar_per_smclk": comps / ((w1 - w0) * 1e-9) / (self.nsm * mhz * 1e6) / BAR_PER_SMCLK,
                "rep_s": rep, "power_steady": self.nvml.window(w0 + int(SETTLE_S * 1e9) - off, w1 - off),
                "power_all": self.nvml.window(h0, h1)}


def parse_shape(text: str) -> tuple[int, int, int]:
    m, n, k = (int(v) for v in text.lower().split("x"))
    return m, n, k


def finish(event: torch.cuda.Event, what: str) -> None:
    """Waits for event; a GPU still busy after HANG_S ends the process, which ends its kernels."""
    t = time.monotonic()
    while not event.query():
        if time.monotonic() - t > HANG_S:
            print(f"HANG {what}: not done in {HANG_S} s", flush=True)
            os._exit(3)
        time.sleep(0.001)


def operands(m: int, n: int, k: int, seed: int) -> dict:
    """Uniform packed bytes (each value uniform on {0, 1, 2, 3}), W's values packed again with their pairs reversed, and
    the unpacked int8 values."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ap = torch.randint(0, 256, (m, k // 4), dtype=torch.uint8, device="cuda", generator=g)
    wp = torch.randint(0, 256, (n, k // 4), dtype=torch.uint8, device="cuda", generator=g)
    a, w = unpack(ap), unpack(wp)
    return {"ap": ap, "wp": wp, "wpr": pack(w, reverse=True), "a": a, "w": w}


def gate(arm: str, op, o: torch.Tensor, want, tag: str) -> tuple[dict, float]:
    """The first call against want (a function of O: True when O is right), a timed call of three, and want again."""
    o.fill_(POISON)
    control = bool(want(o))  # the poisoned buffer itself must fail the gate (the unpack-only arms: must pass it)
    e = torch.cuda.Event()
    op.call()
    e.record()
    finish(e, f"{tag} {arm} first call")
    ok = bool(want(o))
    op.call()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(3):
        op.call()
    e1.record()
    finish(e1, f"{tag} {arm} three calls")
    t_call = e0.elapsed_time(e1) / 3e3
    ok2 = bool(want(o))
    expect_control = not op.mma
    g = {"tag": tag, "arm": arm, "ok": ok and ok2 and control == expect_control, "first_call": ok, "after_repeats": ok2,
         "poison_control": control, "t_call": t_call, **(op.describe() if isinstance(op, Gemm) else {"lt_out": op.out})}
    return g, t_call


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cubin", type=Path, default=Path(os.environ.get("U2_PACKED_CUBIN", Path(__file__).parent / "build" / "packed.cubin")))
    ap.add_argument("--arms", default="lt,i8,p2,p2u,p2m,res,p2x,p2ux,p2mx,i8+h1,p2+h1,p2u+h1,p2m+h1,res+h1",
                    help="comma-separated: lt (torch._int_mm) and KERNELS' arms; ARM+h1 / ARM+h2: beside 1 or 2 hash_k CTAs an SM")
    ap.add_argument("--shapes", default="25344x16384x32768,25344x16384x8192", help="MxNxK, comma-separated")
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--hash-alone", default="1,2", help="hash_k alone at these CTAs an SM (empty: none)")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    uuid = os.environ.get("GPU_LEASE_UUID") or f"GPU-{props.uuid}"
    lib = Lib(args.cubin)
    nvml = Nvml(uuid)
    run = Run(lib, nvml, args.out)
    info = {"device": props.name, "sms": lib.sms, "uuid": uuid, "torch": torch.__version__, "cuda": torch.version.cuda,
            "smem": dict(zip(("i8", "p2", "res"), lib.smem)), "theta": THETA, "bar": BAR}
    (args.out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    gates = open(args.out / "gates.jsonl", "a")
    items = open(args.out / "items.jsonl", "a")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results = []
    for shape in map(parse_shape, args.shapes.split(",")):
        m, n, k = shape
        ops = operands(m, n, k, args.seed)
        ref = torch._int_mm(ops["a"], ops["w"].t())
        o = torch.empty((m, n), dtype=torch.int32, device="cuda")
        # the reference against exact int64 dot products on sampled cells, and the packing against its inverse
        idx = torch.randint(0, m * n, (512,), generator=torch.Generator().manual_seed(args.seed + 1))
        rows, cols = (idx // n).cuda(), (idx % n).cuda()
        exact = (ops["a"][rows].long() * ops["w"][cols].long()).sum(1)
        ref_ok = (bool(torch.equal(exact, ref[rows, cols].long())) and bool(torch.equal(pack(ops["a"][:64]), ops["ap"][:64]))
                  and bool(torch.equal(pack(ops["w"][:64]), ops["wp"][:64])))
        res_ref = None
        if k % RES_K == 0:
            blk = torch._int_mm(ops["a"][:TM, :RES_K].contiguous(), ops["w"][:BN, :RES_K].contiguous().t()) * (k // RES_K)
            res_ref = blk.repeat(m // TM, n // BN)
        tag = f"{m}x{n}x{k}:u2"
        print(f"SHAPE {tag} ref_vs_int64 {ref_ok} max {int(ref.max())}", flush=True)
        runnable = {}
        for spec in arms:
            arm = spec.partition("+")[0]
            if arm in runnable:
                continue
            if arm == "lt":
                op, want = Lt(ops, o), (lambda x: torch.equal(x, ref))
            else:
                op = Gemm(lib, arm, ops, o, args.group)
                if arm == "res":
                    want = lambda x: res_ref is not None and torch.equal(x, res_ref)  # noqa: E731
                elif not KERNELS[arm][3]:
                    want = lambda x: bool((x == POISON).all())  # noqa: E731
                else:
                    want = lambda x: torch.equal(x, ref)  # noqa: E731
            g, t_call = gate(arm, op, o, want, tag)
            g["ok"] = g["ok"] and ref_ok
            print("GATE", json.dumps(g), flush=True)
            gates.write(json.dumps(g) + "\n")
            gates.flush()
            if g["ok"]:
                runnable[arm] = (op, t_call)
        for rnd in range(args.rounds):
            if rnd == 0 and args.hash_alone:
                for c in (int(v) for v in args.hash_alone.split(",")):
                    rec = run.hash_alone(c, args.secs)
                    rec.update(tag=tag, round=rnd)
                    print("ITEM", json.dumps(rec), flush=True)
                    items.write(json.dumps(rec) + "\n")
                    items.flush()
                    results.append(rec)
            for spec in arms:
                arm, _, hs = spec.partition("+")
                if arm not in runnable:
                    print(f"SKIP {tag} {spec}: its gate failed", flush=True)
                    continue
                op, t_call = runnable[arm]
                if hs:
                    # the hash at about half its alone rate spans about secs; the GEMM runs 2 secs, so the span stays
                    # inside it down to a quarter of that rate
                    per_sm = int(hs[1:])
                    alone = [r for r in results if r.get("name") == f"hash x{per_sm}" and r.get("tag") == tag]
                    rep = alone[0]["rep_s"] if alone else 15e-6
                    reps = int(0.5 * args.secs / rep)
                    rec = run.item(spec, op, 2 * args.secs, t_call, per_sm * lib.sms, reps)
                else:
                    rec = run.item(spec, op, args.secs, t_call)
                rec.update(tag=tag, round=rnd, arm=arm)
                if op.mma:
                    rec["o_after_item_ok"] = bool(torch.equal(o, res_ref if arm == "res" else ref))
                print("ITEM", json.dumps(rec), flush=True)
                items.write(json.dumps(rec) + "\n")
                items.flush()
                results.append(rec)
        del ops, ref, res_ref, o, runnable
        torch.cuda.empty_cache()
    nvml.stop()
    (args.out / "summary.md").write_text(summary(results))
    print(summary(results), flush=True)
    return 0


def summary(results: list[dict]) -> str:
    lines = ["| item | shape | round | MACs/SM-clk / 4096 | θ | SM MHz | W mean | W max | pJ/MAC | hash of bar | throttle |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "hash" in r:
            t, p = r.get("beside") or {}, r.get("power_beside") or {}
            hb = f"{r['hash']['of_bar']:.3f}"
        elif r.get("name", "").startswith("hash"):
            t, p = {}, r.get("power_steady") or {}
            hb = f"{r['of_bar']:.3f}"
        else:
            t, p = r.get("steady") or {}, r.get("power_steady") or {}
            hb = "-"
        f = lambda d, k, fmt: format(d[k], fmt) if k in d and d[k] == d[k] else "-"  # noqa: E731
        name = r["name"] + ("" if r.get("mma", True) else " (unpack only)")
        lines.append(f"| {name} | {r.get('tag', '')} | {r.get('round', '')} | {f(t, 'of_4096', '.4f')} | "
                     f"{f(t, 'of_theta', '.4f')} | {f(t, 'sm_mhz', '.0f') if t else f(r, 'sm_mhz', '.0f')} | "
                     f"{f(p, 'mean_w', '.0f')} | {f(p, 'max_w', '.0f')} | {f(r, 'pj_per_mac', '.3f')} | {hb} | "
                     f"{','.join(p.get('throttle', []))} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
