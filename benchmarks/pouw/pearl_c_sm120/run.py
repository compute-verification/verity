#!/usr/bin/env python3
"""Pearl-C on one RTX PRO 6000 Blackwell (sm_120) from the prebuilt pearl_c_sm120.cubin, through the CUDA driver API
(ctypes on libcuda.so.1, stdlib only, nothing built on the card).

0. Identity, before anything runs: exactly one visible device (CUDA_VISIBLE_DEVICES a single index), compute capability
   12.0, an "NVIDIA RTX PRO 6000 Blackwell" name, and its UUID equal to PEARLC_EXPECT_UUID (server.md's table).  NVML is
   addressed by that UUID, never by index (NVML ignores CUDA_VISIBLE_DEVICES).  A mismatch exits 3 before any launch.
1. The check.  At check.json's shape (fixture.py's), every kernel runs on the fixture's inputs and every buffer it writes is
   compared byte for byte with the fixture, for each variant (`sm120`, G = 4, and `sm120-unpromoted`); the GEMMs are re-run
   persistent on 1 and 3 CTAs and compared again.  A variant is timed only if all of its buffers match (C~ and U bit-exact
   against the CPU reference), the plain baselines only if theirs do.  Exit 4 if any check fails.
2. Diagnostic timing (not the headline: GPU 6's harness times the arm against autotuned baselines), at each of SHAPES
   (default 8192, m = n = k): each kernel REPS times after a warm-up, its median, the phases (the weight side; the call's
   forming, GEMM, clean-up and hashing), the call end to end with and without hashing, and consistency checks (C~ of gemm_ct
   equals gemm_pearlc's; a second gemm_pearlc gives the same U).  NVML's clocks, power and throttle reasons before and after.

One JSON object per line on stdout as each part finishes, a summary on stderr.

  python3 run.py [check.json]   (env: PEARLC_EXPECT_UUID, SHAPES="8192", REPS=5, VARIANTS="sm120 sm120-unpromoted",
                                 PEARLC_LIBCUDA, PEARLC_LIBNVML)

Dry runs on a stand-in driver only: PEARLC_ANY_GPU=1 skips the identity check, and PEARLC_UNGATED=1 (refused without it)
times regardless of the check and marks every row ungated.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NAME_PREFIX = "NVIDIA RTX PRO 6000 Blackwell"
SM_COUNT, MAX_DYN, CC_MAJOR, CC_MINOR = 16, 8, 75, 76
NORM_256, NORM_16 = 0x4600, 0x4400      # bf16(32 x the line norm): E's lines, and F's under v1
NT, BM, RASTER = 256, 128, 8
VARIANT_G = {"sm120": 4, "sm120-unpromoted": 0}

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER
CUDA = {
    "cuInit": [u], "cuDriverGetVersion": [P(i32)], "cuDeviceGetCount": [P(i32)], "cuDeviceGet": [P(i32), i32],
    "cuDeviceGetName": [C.c_char_p, i32, i32], "cuDeviceGetUuid_v2": [C.c_char_p, i32],
    "cuDeviceGetAttribute": [P(i32), i32, i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuFuncSetAttribute": [vp, i32, i32], "cuMemAlloc_v2": [P(u64), sz], "cuMemFree_v2": [u64],
    "cuMemsetD8_v2": [u64, C.c_ubyte, sz], "cuMemcpyHtoD_v2": [u64, vp, sz], "cuMemcpyDtoH_v2": [vp, u64, sz],
    "cuMemcpyDtoD_v2": [u64, u64, sz], "cuMemGetInfo_v2": [P(sz), P(sz)],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuCtxSynchronize": [],
    "cuEventCreate": [P(vp), u], "cuEventRecord": [vp, vp], "cuEventSynchronize": [vp],
    "cuEventElapsedTime": [P(C.c_float), vp, vp], "cuEventDestroy_v2": [vp], "cuGetErrorString": [i32, P(C.c_char_p)],
}
NVML = {
    "nvmlInit_v2": [], "nvmlDeviceGetHandleByUUID": [C.c_char_p, P(vp)], "nvmlSystemGetDriverVersion": [C.c_char_p, u],
    "nvmlDeviceGetClockInfo": [vp, i32, P(u)], "nvmlDeviceGetCurrentClocksThrottleReasons": [vp, P(C.c_ulonglong)],
    "nvmlDeviceGetPowerUsage": [vp, P(u)], "nvmlDeviceGetTemperature": [vp, i32, P(u)],
}


class Args(C.Structure):
    _fields_ = [("p", C.c_uint64 * 16), ("i", C.c_int32 * 16)]


def bind(lib, sigs):
    for f, a in sigs.items():
        getattr(lib, f).argtypes, getattr(lib, f).restype = a, i32
    return lib


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def uuid_str(raw: bytes) -> str:
    h = raw[:16].hex()
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


class Nvml:
    """NVML on the device of one UUID; reads nothing (and says so) if NVML or the UUID isn't there."""

    def __init__(self, uuid: str):
        self.lib, self.h, self.driver, self.uuid = None, vp(), "", uuid
        try:
            lib = bind(C.CDLL(os.environ.get("PEARLC_LIBNVML", "libnvidia-ml.so.1")), NVML)
        except (OSError, AttributeError):
            return
        if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.h)) == 0:
            self.lib = lib
            b = C.create_string_buffer(96)
            if lib.nvmlSystemGetDriverVersion(b, 96) == 0:
                self.driver = b.value.decode()

    def read(self) -> dict:
        if not self.lib:
            return {"nvml": "unavailable"}
        sm, mem, pw, t, thr = u(), u(), u(), u(), C.c_ulonglong()
        self.lib.nvmlDeviceGetClockInfo(self.h, 1, C.byref(sm))
        self.lib.nvmlDeviceGetClockInfo(self.h, 2, C.byref(mem))
        self.lib.nvmlDeviceGetCurrentClocksThrottleReasons(self.h, C.byref(thr))
        self.lib.nvmlDeviceGetPowerUsage(self.h, C.byref(pw))
        self.lib.nvmlDeviceGetTemperature(self.h, 0, C.byref(t))
        return {"sm_mhz": sm.value, "mem_mhz": mem.value, "throttle": hex(thr.value), "power_w": round(pw.value / 1000, 1),
                "temp_c": t.value}


class Dev:
    def __init__(self, cubin: bytes):
        self.cu = bind(C.CDLL(os.environ.get("PEARLC_LIBCUDA", "libcuda.so.1")), CUDA)
        self.ck("cuInit", 0)
        self.count = self.get("cuDeviceGetCount")
        self.dev = self.get("cuDeviceGet", 0)
        name, raw = C.create_string_buffer(256), C.create_string_buffer(16)
        self.ck("cuDeviceGetName", name, 256, self.dev)
        self.ck("cuDeviceGetUuid_v2", raw, self.dev)
        self.name, self.uuid = name.value.decode(), uuid_str(raw.raw)
        self.sms = self.get("cuDeviceGetAttribute", SM_COUNT, self.dev)
        self.cc = (self.get("cuDeviceGetAttribute", CC_MAJOR, self.dev), self.get("cuDeviceGetAttribute", CC_MINOR, self.dev))
        self.driver = self.get("cuDriverGetVersion")
        self.cubin = cubin
        self.mod = None
        self.fns: dict[str, vp] = {}
        self.live: set[int] = set()

    def identity(self, expect_uuid: str | None) -> list[str]:
        """What stops the run before any context or launch: the reasons, empty if this is the expected device."""
        bad = []
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.isdigit():
            bad.append(f"CUDA_VISIBLE_DEVICES={visible!r} is not a single index")
        if self.count != 1:
            bad.append(f"{self.count} devices visible, not 1")
        if self.cc != (12, 0):
            bad.append(f"compute capability {self.cc}, not (12, 0)")
        if not self.name.startswith(NAME_PREFIX):
            bad.append(f"device {self.name!r} is not an {NAME_PREFIX}")
        if not expect_uuid:
            bad.append("PEARLC_EXPECT_UUID is not set (server.md's UUID for this index)")
        elif self.uuid != expect_uuid:
            bad.append(f"UUID {self.uuid} is not the expected {expect_uuid}")
        return bad

    def open(self) -> None:
        ctx = vp()
        self.ck("cuDevicePrimaryCtxRetain", C.byref(ctx), self.dev)
        self.ck("cuCtxSetCurrent", ctx)
        self.mod = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod), self.cubin)

    def ck(self, f: str, *a) -> None:
        r = getattr(self.cu, f)(*a)
        if r:
            s = C.c_char_p()
            self.cu.cuGetErrorString(r, C.byref(s))
            raise RuntimeError(f"{f} returned CUresult {r} ({s.value.decode() if s.value else 'unknown'})")

    def get(self, f: str, *a) -> int:
        v = i32()
        self.ck(f, C.byref(v), *a)
        return v.value

    def fn(self, name: str) -> vp:
        if name not in self.fns:
            f = vp()
            self.ck("cuModuleGetFunction", C.byref(f), self.mod, name.encode())
            self.fns[name] = f
        return self.fns[name]

    def alloc(self, n: int) -> int:
        p = u64()
        self.ck("cuMemAlloc_v2", C.byref(p), max(n, 16))
        self.ck("cuMemsetD8_v2", p, 0, max(n, 16))
        self.live.add(p.value)
        return p.value

    def free(self, p: int) -> None:
        if p in self.live:
            self.live.discard(p)
            self.ck("cuMemFree_v2", u64(p))

    def put(self, data: bytes) -> int:
        p = self.alloc(len(data))
        self.ck("cuMemcpyHtoD_v2", u64(p), C.c_char_p(data), len(data))
        return p

    def read(self, p: int, n: int) -> bytes:
        buf = C.create_string_buffer(n)
        self.ck("cuMemcpyDtoH_v2", buf, u64(p), n)
        return buf.raw

    def copy(self, dst: int, src: int, n: int) -> None:
        self.ck("cuMemcpyDtoD_v2", u64(dst), u64(src), n)

    def launch(self, name: str, grid, block: int, smem: int = 0, p=(), i=(), stream=None) -> None:
        a = Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        g = tuple(grid) + (1,) * (3 - len(grid)) if isinstance(grid, (tuple, list)) else (grid, 1, 1)
        params = (vp * 1)(C.addressof(a))
        self.ck("cuLaunchKernel", self.fn(name), g[0], g[1], g[2], block, 1, 1, smem, stream, params, None)

    def sync(self) -> None:
        self.ck("cuCtxSynchronize")

    def timed(self, body, reps: int) -> list[float]:
        body()
        self.sync()
        out = []
        for _ in range(reps):
            a, b, ms = vp(), vp(), C.c_float()
            self.ck("cuEventCreate", C.byref(a), 0)
            self.ck("cuEventCreate", C.byref(b), 0)
            self.ck("cuEventRecord", a, None)
            body()
            self.ck("cuEventRecord", b, None)
            self.ck("cuEventSynchronize", b)
            self.ck("cuEventElapsedTime", C.byref(ms), a, b)
            self.cu.cuEventDestroy_v2(a)
            self.cu.cuEventDestroy_v2(b)
            out.append(ms.value)
        return out

    def free_mem(self) -> tuple[int, int]:
        f, t = sz(), sz()
        self.ck("cuMemGetInfo_v2", C.byref(f), C.byref(t))
        return f.value, t.value


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class Pipeline:
    """One (shape, variant) instance: its buffers and its kernels in order, each a (name, phase, launch) step.  Phases:
    `job` (the weight side, once per weight and epoch), `forming`, `gemm` (gemm_ct, the chain alone, for the split),
    `cleanup` and `hashing` (the call), and the baselines' `plain_job`, `plain`, `adopted_job`, `adopted`."""

    GEMM_SMEM: dict[str, int] = {}

    def __init__(self, dev: Dev, m: int, n: int, k: int, variant: str, keys: dict[str, int], xa: int, xb: int):
        self.dev, self.m, self.n, self.k, self.variant, self.keys = dev, m, n, k, variant, keys
        self.g = VARIANT_G[variant]
        self.suffix = "" if self.g else "_u"
        if m % BM or n % BM or k % 1024:
            raise ValueError(f"the kernels take m, n multiples of 128 and k a multiple of 1024 (row-hash segments), not {(m, n, k)}")
        self.row_bytes, self.segment = 4 * k, 1024
        sizes = {
            "ea_codes": m * 32, "eb_codes": n * 32, "fa_codes": k * 32, "faT32": 32 * k, "fb_codes": k * 32, "fbT32": 32 * k,
            "fbT128": 128 * k, "g": 128 * 32 * 4,
            "alpha_a": 4 * m, "beta_a": 4 * m, "ia_a": 4 * m, "aux_a": 8 * m, "epa_codes": m * 32,
            "alpha_b": 4 * n, "beta_b": 4 * n, "ia_b": 4 * n, "aux_b": 8 * n, "epb_codes": n * 32,
            "at": m * k, "bt": n * k, "af": m * 32 * 4, "bf": n * 32 * 4, "pa_t": m * 128, "pb_t": n * 128,
            "c": m * n * 4, "c2": m * n * 4, "u": m * n * 4, "y": m * n * 2,
            "digests": m * n, "leaves": (m // 64) * (n // 64) * 32, "tickets": (m // 16) * (n // 16) * 32,
            "rows": m * (self.row_bytes // self.segment) * 32,
            "qa": m * k, "qb": n * k, "sa": 4 * m, "sb": 4 * n, "y_plain": m * n * 2, "y_fast": m * n * 2,
            "qa_s5": m * k, "qb_s5": n * k, "y_adopt": m * n * 2, "diff": 8,
        }
        self.size = sizes
        self.b = {name: dev.alloc(s) for name, s in sizes.items()}
        self.b["xa"], self.b["xb"] = xa, xb
        self.size["xa"], self.size["xb"] = m * self.row_bytes, n * self.row_bytes

    def free(self) -> None:
        for name, p in self.b.items():
            if name not in ("xa", "xb"):
                self.dev.free(p)

    def gemm(self, name: str, a: str, b: str, M: int, N: int, grid: int | None = None, out=("c",), pa=0, pb=0, rs=0, cs=0,
             stream=None):
        if name not in self.GEMM_SMEM:
            probe = self.dev.alloc(16)
            self.dev.launch(name + "_smem", 1, 1, p=(probe,))
            self.dev.sync()
            self.GEMM_SMEM[name] = int.from_bytes(self.dev.read(probe, 4), "little")
            self.dev.free(probe)
            self.dev.ck("cuFuncSetAttribute", self.dev.fn(name), MAX_DYN, self.GEMM_SMEM[name])
        bn = 32 if name.startswith("gemm_words") else 128
        tiles = (M // BM) * (N // bn)
        ptrs = [self.b[a], self.b[b], 0, 0, 0, 0, 0, 0, 0]
        for slot, key in zip((2, 3, 4), out):
            ptrs[slot] = self.b[key] if key else 0
        ptrs[5], ptrs[6], ptrs[7], ptrs[8] = pa, pb, rs, cs
        self.dev.launch(name, min(tiles, grid or self.dev.sms), NT, self.GEMM_SMEM[name], p=ptrs, i=(M, N, self.k, RASTER),
                        stream=stream)

    def steps(self) -> list[tuple[str, str, object]]:
        d, b, m, n, k, sx = self.dev, self.b, self.m, self.n, self.k, self.suffix
        words, ct, pearlc = "gemm_words" + sx, "gemm_ct" + sx, "gemm_pearlc" + sx

        def lines(tag, key, side, factor, count, codes, t32=0, t128=0):
            norm = NORM_16 if factor == 1 else NORM_256
            return (f"lines_{tag}", lambda: d.launch("lines", cdiv(count, 128), 128, p=(key, b[codes], t32, t128),
                                                    i=(side, factor, count, norm)))

        def form_side(tag, rows, x, e_codes, f_codes, out):
            ep = f"ep{tag}_codes"
            return [(f"stats_{tag}", lambda: d.launch("stats_s5", cdiv(rows, 8), 256, p=(b[x], b[f"alpha_{tag}"], b[f"beta_{tag}"],
                                                                                       b[f"ia_{tag}"], b[f"aux_{tag}"]), i=(rows, k))),
                    (f"eprime_{tag}", lambda: d.launch("eprime", cdiv(rows, 128), 128, p=(b[e_codes], b[f"beta_{tag}"], b[ep]),
                                                       i=(rows,))),
                    (f"form_{tag}", lambda: d.launch("form_s5", (k // 128, rows // 64), 128,
                                                     p=(b[x], b[f"alpha_{tag}"], b[ep], b[f_codes], b[out]), i=(rows, k)))]

        job = [lines("eb", self.keys["line_b"], 1, 0, n, "eb_codes"),
               lines("fa", self.keys["line_b"], 0, 1, k, "fa_codes", b["faT32"]),
               lines("fb", self.keys["line_b"], 1, 1, k, "fb_codes", b["fbT32"], b["fbT128"]),
               *form_side("b", n, "xb", "eb_codes", "fb_codes", "bt"),
               ("bf", lambda: self.gemm(words, "bt", "faT32", n, 32, out=("bf",))),
               ("g", lambda: self.gemm(words, "fbT128", "faT32", 128, 32, out=("g",))),
               ("pb", lambda: d.launch("pb", cdiv(n * 32, 256), 256, p=(b["epb_codes"], 0, b["bf"], b["g"], b["pb_t"]), i=(n,)))]
        forming = [lines("ea", self.keys["line_a"], 0, 0, m, "ea_codes"),
                   *form_side("a", m, "xa", "ea_codes", "fa_codes", "at")]
        gemm = [("gemm_ct", lambda: self.gemm(ct, "at", "bt", m, n, out=("c2",)))]
        cleanup = [("af", lambda: self.gemm(words, "at", "fbT32", m, 32, out=("af",))),
                   ("pa", lambda: d.launch("pa", cdiv(m * 64, 256), 256, p=(b["epa_codes"], 0, b["af"], b["pa_t"]), i=(m,))),
                   ("gemm_pearlc", lambda: self.gemm(pearlc, "at", "bt", m, n, out=("c", "u", "y"), pa=b["pa_t"], pb=b["pb_t"],
                                                     rs=b["ia_a"], cs=b["ia_b"]))]
        segs = self.row_bytes // self.segment
        hashing = [("hash_msg", lambda: d.launch("hash_msg", cdiv(m * n // 32, 128), 128, p=(b["c"], b["u"], b["digests"]), i=(m, n))),
                   ("hash_leaf", lambda: d.launch("hash_leaf", cdiv((m // 64) * (n // 64), 64), 64, p=(b["digests"], b["leaves"]),
                                                  i=(m, n))),
                   ("tickets", lambda: d.launch("tickets", cdiv((m // 16) * (n // 16), 128), 128,
                                                p=(b["c"], b["tickets"], self.keys["jackpot"]), i=(m, n))),
                   ("hash_rows", lambda: d.launch("hash_rows_f32", cdiv(m * segs, 128), 128, p=(b["xa"], b["rows"]),
                                                  i=(m, self.row_bytes)))]
        plain_job = [("plainq_b", lambda: d.launch("plainq", n, 256, p=(b["xb"], b["qb"], b["sb"]), i=(n, k)))]
        plain = [("plainq_a", lambda: d.launch("plainq", m, 256, p=(b["xa"], b["qa"], b["sa"]), i=(m, k))),
                 ("gemm_plain", lambda: self.gemm("gemm_plain", "qa", "qb", m, n, out=(None, None, "y_plain"), rs=b["sa"], cs=b["sb"])),
                 ("gemm_fast", lambda: self.gemm("gemm_fast", "qa", "qb", m, n, out=(None, None, "y_fast"), rs=b["sa"], cs=b["sb"]))]
        adopted_job = [("quant_b", lambda: d.launch("quant_s5", cdiv(n * k // 16, 256), 256, p=(b["xb"], b["alpha_b"], b["qb_s5"]),
                                                    i=(n, k)))]
        adopted = [("quant_a", lambda: d.launch("quant_s5", cdiv(m * k // 16, 256), 256, p=(b["xa"], b["alpha_a"], b["qa_s5"]),
                                                i=(m, k))),
                   ("gemm_adopt", lambda: self.gemm("gemm_plain", "qa_s5", "qb_s5", m, n, out=(None, None, "y_adopt"), rs=b["ia_a"],
                                                    cs=b["ia_b"]))]
        tag = lambda phase, xs: [(nm, phase, f) for nm, f in xs]  # noqa: E731
        return (tag("job", job) + tag("forming", forming) + tag("gemm", gemm) + tag("cleanup", cleanup) + tag("hashing", hashing)
                + tag("plain_job", plain_job) + tag("plain", plain) + tag("adopted_job", adopted_job) + tag("adopted", adopted))

    def run_all(self) -> None:
        for _, _, f in self.steps():
            f()
        self.dev.sync()

    def diff(self, x: str, y: str) -> int:
        words = self.size[x] // 4
        self.dev.ck("cuMemsetD8_v2", u64(self.b["diff"]), 0, 8)
        self.dev.launch("diff_words", cdiv(words, 256), 256, p=(self.b[x], self.b[y], self.b["diff"]), i=(words,))
        self.dev.sync()
        return int.from_bytes(self.dev.read(self.b["diff"], 8), "little")


# ---- the check ----------------------------------------------------------------------------------------------------

PLAIN_BUFFERS = {"qa", "qb", "sa", "sb", "y_plain", "y_fast", "qa_s5", "qb_s5", "y_adopt"}
GEMM_RERUNS = (("gemm_pearlc", ("c", "u", "y")), ("gemm_ct", ("c2",)), ("gemm_plain", ("y_plain",)), ("gemm_fast", ("y_fast",)),
               ("af", ("af",)), ("bf", ("bf",)), ("gemm_adopt", ("y_adopt",)))


def compare(dev: Dev, pipe: Pipeline, name: str, want: bytes, alias: str | None = None) -> dict:
    got = dev.read(pipe.b[alias or name], len(want))
    if got == want:
        return {"ok": True, "bytes": len(want)}
    bad = [i for i in range(0, len(want), 4) if got[i:i + 4] != want[i:i + 4]]
    return {"ok": False, "bytes": len(want), "words_differing": len(bad),
            "first": [{"offset": i, "got": got[i:i + 4].hex(), "want": want[i:i + 4].hex()} for i in bad[:6]]}


def check(dev: Dev, man: dict, base: str, variants: list[str]) -> dict:
    m, n, k = man["shape"]
    keys = {name: dev.put(bytes.fromhex(h)) for name, h in man["keys"].items()}
    load = lambda d, f: open(os.path.join(base, d, f["file"]), "rb").read()  # noqa: E731
    common = man["common"]
    results = {}
    for variant in variants:
        files = man["variants"][variant]["files"]
        want = lambda name: load(variant, files[name]) if name in files else load("", common[name])  # noqa: E731
        xa, xb = dev.put(load("", common["xa"])), dev.put(load("", common["xb"]))
        pipe = Pipeline(dev, m, n, k, variant, keys, xa, xb)
        pipe.run_all()
        res = {}
        for name in [*common, *files]:
            if name in ("xa", "xb"):
                continue
            res[name] = compare(dev, pipe, name, want(name))
            if name == "c":
                res["c@gemm_ct"] = compare(dev, pipe, name, want(name), alias="c2")
        steps = {nm: fn for nm, _, fn in pipe.steps()}
        for grid in (1, 3):
            for step, outs in GEMM_RERUNS:
                for o in outs:
                    dev.ck("cuMemsetD8_v2", u64(pipe.b[o]), 0xAB, pipe.size[o])
                real = pipe.gemm

                def pinned(*a, _real=real, _grid=grid, **kw):
                    kw["grid"] = _grid
                    return _real(*a, **kw)
                pipe.gemm = pinned
                steps[step]()
                pipe.gemm = real
                dev.sync()
                for o in outs:
                    res[f"{o}@grid{grid}"] = compare(dev, pipe, o, want("c" if o == "c2" else o))
        pipe.free()
        dev.free(xa)
        dev.free(xb)
        failed = sorted(nm for nm, r in res.items() if not r["ok"])
        results[variant] = {"buffers": len(res), "failed": failed,
                            "pearl_c_ok": not [x for x in failed if x.split("@")[0] not in PLAIN_BUFFERS],
                            "plain_ok": not [x for x in failed if x.split("@")[0] in PLAIN_BUFFERS],
                            "detail": {nm: r for nm, r in res.items() if not r["ok"]}}
    for p in keys.values():
        dev.free(p)
    return results


# ---- the diagnostic timing ----------------------------------------------------------------------------------------

CALL_PHASES = ("forming", "gemm", "cleanup", "hashing")


def timing(dev: Dev, nv: Nvml, size: int, variant: str, reps: int, gates: dict) -> dict:
    m = n = k = size
    keys = {name: dev.put(hashlib.sha256(f"pearl-c-timing/{name}".encode()).digest()) for name in ("line_a", "line_b", "jackpot")}
    xa, xb = dev.alloc(m * 4 * k), dev.alloc(n * 4 * k)
    for x, rows, seed in ((xa, m, 11), (xb, n, 12)):
        dev.launch("gen_f32", cdiv(rows * k, 256), 256, p=(x,), i=(rows, k, seed))
    pipe = Pipeline(dev, m, n, k, variant, keys, xa, xb)
    pipe.run_all()
    out = measure(dev, nv, pipe, reps, gates)
    pipe.free()
    for p_ in (xa, xb, *keys.values()):
        dev.free(p_)
    return out


def measure(dev: Dev, nv: Nvml, pipe: Pipeline, reps: int, gates: dict) -> dict:
    """The timed part of `timing` on a pipeline whose kernels have run once: medians, phases, ratios, consistency."""
    m, n, k, steps = pipe.m, pipe.n, pipe.k, pipe.steps()
    before = nv.read()
    med, runs = {}, {}
    for name, phase, f in steps:
        if phase in CALL_PHASES + ("job",) and not gates["pearl_c"] or \
                phase in ("plain", "plain_job", "adopted", "adopted_job") and not gates["plain"]:
            continue
        t = dev.timed(f, reps)
        runs[name], med[name] = [round(x, 4) for x in t], statistics.median(t)
    after = nv.read()
    phases = {ph: sum(med.get(nm, 0.0) for nm, p, _ in steps if p == ph) for ph in ("job",) + CALL_PHASES}
    out = {"kind": "timing", "label": "diagnostic, unlocked unless the clock record says so", "shape": [m, n, k],
           "variant": pipe.variant, "reps": reps, "gates": gates, "ms": {nm: round(v, 4) for nm, v in med.items()}, "runs": runs,
           "clk_before": before, "clk_after": after}
    if gates["pearl_c"]:
        call = [f for nm, p, f in steps if p in ("forming", "cleanup", "hashing")]
        no_hash = [f for nm, p, f in steps if p in ("forming", "cleanup")]
        e2e = statistics.median(dev.timed(lambda: [f() for f in call], reps))
        e2e_nh = statistics.median(dev.timed(lambda: [f() for f in no_hash], reps))
        split = {"forming": phases["forming"], "gemm": med["gemm_ct"],
                 "cleanup": med["af"] + med["pa"] + med["gemm_pearlc"] - med["gemm_ct"], "hashing": phases["hashing"]}
        out.update({"split_ms": {kk: round(v, 4) for kk, v in split.items()}, "call_sum_ms": round(sum(split.values()), 4),
                    "call_e2e_ms": round(e2e, 4), "call_e2e_no_hash_ms": round(e2e_nh, 4), "job_ms": round(phases["job"], 4),
                    "pearlc_tflops": round(2 * m * n * k / med["gemm_pearlc"] / 1e9, 1)})
        out["consistency"] = {"c_ct_vs_pearlc_words_differing": pipe.diff("c2", "c")}
        dev.copy(pipe.b["c2"], pipe.b["u"], pipe.size["u"])
        [f for nm, _, f in steps if nm == "gemm_pearlc"][0]()
        out["consistency"]["u_rerun_words_differing"] = pipe.diff("c2", "u")
    if gates["plain"]:
        out["plain_call_ms"] = round(med["plainq_a"] + med["gemm_plain"], 4)
        out["plain_tflops"] = round(2 * m * n * k / med["gemm_plain"] / 1e9, 1)
        out["fast_tflops"] = round(2 * m * n * k / med["gemm_fast"] / 1e9, 1)
        out["adopted_call_ms"] = round(med["stats_a"] + med["quant_a"] + med["gemm_adopt"], 4) if "stats_a" in med else None
        if gates["pearl_c"]:
            fastest = min(med["gemm_plain"], med["gemm_fast"])
            out["ratio_vs_own_plain_gemm"] = {"call_e2e": round(e2e / fastest, 4), "no_hash_e2e": round(e2e_nh / fastest, 4),
                                              "chain_only": round(med["gemm_ct"] / fastest, 4)}
    return out


def main(argv: list[str]) -> int:
    man_path = argv[1] if len(argv) > 1 else os.path.join(HERE, "check", "check.json")
    man = json.load(open(man_path))
    base = os.path.dirname(man_path)
    image = open(os.path.join(HERE, "pearl_c_sm120.cubin"), "rb").read()
    dev = Dev(image)
    expect = os.environ.get("PEARLC_EXPECT_UUID")
    any_gpu, ungated = bool(os.environ.get("PEARLC_ANY_GPU")), bool(os.environ.get("PEARLC_UNGATED"))
    refused = [] if any_gpu else dev.identity(expect)
    if ungated and not any_gpu:
        refused.append("PEARLC_UNGATED is for dry runs on a stand-in driver (with PEARLC_ANY_GPU)")
    nv = Nvml(dev.uuid)
    env = {"kind": "env", "device": dev.name, "uuid": dev.uuid, "expect_uuid": expect, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "device_order": os.environ.get("CUDA_DEVICE_ORDER"), "count": dev.count, "cc": list(dev.cc), "sms": dev.sms,
           "cuda_driver_api": dev.driver, "nvml_driver": nv.driver, "cubin_sha256": hashlib.sha256(image).hexdigest(),
           "check_json_sha256": hashlib.sha256(open(man_path, "rb").read()).hexdigest(), "clk": nv.read(),
           "refused": refused, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    emit(env)
    if refused:
        print("REFUSED: " + "; ".join(refused), file=sys.stderr)
        return 3
    dev.open()
    env2 = {"kind": "mem", "free_total": dev.free_mem()}
    emit(env2)
    variants = os.environ.get("VARIANTS", "sm120 sm120-unpromoted").split()
    t0 = time.time()
    res = check(dev, man, base, variants)
    emit({"kind": "check", "shape": man["shape"], "seconds": round(time.time() - t0, 2), "variants": res})
    ok = all(r["pearl_c_ok"] and r["plain_ok"] for r in res.values())
    print(f"check at {man['shape']}: " + "; ".join(f"{v} {r['buffers']} buffers, failed {r['failed'] or 'none'}" for v, r in res.items()),
          file=sys.stderr)
    reps = int(os.environ.get("REPS", "5"))
    for size in [int(s) for s in os.environ.get("SHAPES", "8192").split()]:
        for variant in variants:
            gates = {"pearl_c": res[variant]["pearl_c_ok"] or ungated, "plain": res[variant]["plain_ok"] or ungated,
                     "ungated": ungated}
            if not (gates["pearl_c"] or gates["plain"]):
                continue
            t0 = time.time()
            row = timing(dev, nv, size, variant, reps, gates)
            row["seconds"] = round(time.time() - t0, 2)
            emit(row)
            if "ratio_vs_own_plain_gemm" in row:
                s, r = row["split_ms"], row["ratio_vs_own_plain_gemm"]
                print(f"{size}^3 {variant}: call {row['call_e2e_ms']:.3f} ms = {r['call_e2e']:.3f}x this build's plain FP8 GEMM "
                      f"({row['plain_tflops']} / {row['fast_tflops']} TFLOPS promoted / not); split GEMM {s['gemm']:.3f}, clean-up "
                      f"{s['cleanup']:.3f}, forming {s['forming']:.3f}, hashing {s['hashing']:.3f}; no hashing {r['no_hash_e2e']:.3f}x; "
                      f"consistency {row['consistency']}", file=sys.stderr)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main(sys.argv))
