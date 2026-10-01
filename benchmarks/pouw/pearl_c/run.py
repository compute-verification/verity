#!/usr/bin/env python3
"""Pearl-C on one H100 SXM from the prebuilt pearl_c.cubin, through the CUDA driver API (ctypes on libcuda.so.1, stdlib
only, nothing built on the pod).

1. The check.  At check.json's shape (fixture.py's), every kernel runs on the fixture's inputs and every buffer it writes is
   compared byte for byte with the fixture; the GEMMs are re-run persistent on 1 and 3 CTAs and compared again.  A forming's
   timing runs only if all of its buffers match (the gate: C~ and U bit-exact against the CPU reference), the plain
   baseline's only if its own do, and neither unless ieee_ops and ieee_q (the correctly rounded div, rcp and sqrt, the only
   `.FTZ` sequences build.sh admits) give the correctly rounded words on subnormal inputs and outputs.  PEARLC_UNGATED=1
   times regardless and says so.
2. The timing, at each of SHAPES (default 8192 and 16384, m = n = k), for forming v1 (§5), v0 (#311's, as ported) and the
   FP8 baselines, on generated inputs: each kernel launched REPS times after a warm-up, the median per kernel, the phases
   (per job; the unit's A commitment and seed_A, which forming waits for, then forming, GEMM, clean-up and hashing), the
   unit end to end with and without hashing, and consistency
   checks at that shape (C~ of gemm_ct equals gemm_pearlc's; a second gemm_pearlc gives the same U).  The baselines share
   the GEMM: plain-max (s = 448/max|x|, `plainq`) for both formings, and for v1 the adopted quantizer (the model's FP8 path
   with PoUW off: e4m3(alpha x) with v1's alpha, `stats_s5` + `quant_s5`).

One JSON object per line on stdout as each part finishes, a summary on stderr.  Exit 3: not an H100 SXM; 4: a check failed.

  python3 run.py [check.json]      (env: SHAPES="8192 16384", REPS=5, HASH_STREAM=1 (the timing's hashing on a side stream,
                                    `Pipeline`), PEARLC_LIBCUDA, PEARLC_LIBNVML)
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
H100 = "NVIDIA H100 80GB HBM3"
SM_COUNT, MAX_DYN = 16, 8
NORM_256, NORM_16 = 0x4600, 0x4400      # bf16(32 x the line norm): E's lines, and F's under v1
#: hashing format -h1's public label keys, blake3("pearl-c/h1/message") and blake3("pearl-c/h1/leaf") (`pearl_c.H1_KEYS`)
H1_KEYS = {"k_msg": "071cf46377a5d6f62e118a6c4d1bcd93d8d781af04ef0bdd84f9011381e801d1",
           "k_leaf": "0e661b01c2077b147887d1ee19aed2ddb35c2cc459cc62f6af0b05ea9f926169"}
#: the seed-A label's key, blake3("pearl-c/v0/seed-A"), under which b3_seeds derives a unit's seed_A (`pearl_c.unit_seed_a`)
SEED_KEY = "6c8abb01868eaa7a07ed62333c4c6ae75daeb48c0f4e709fe7a6d8401f2a202a"

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER
CUDA = {
    "cuInit": [u], "cuDriverGetVersion": [P(i32)], "cuDeviceGet": [P(i32), i32], "cuDeviceGetName": [C.c_char_p, i32, i32],
    "cuDeviceGetAttribute": [P(i32), i32, i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuFuncSetAttribute": [vp, i32, i32], "cuMemAlloc_v2": [P(u64), sz], "cuMemFree_v2": [u64],
    "cuMemsetD8_v2": [u64, C.c_ubyte, sz], "cuMemcpyHtoD_v2": [u64, vp, sz], "cuMemcpyDtoH_v2": [vp, u64, sz],
    "cuMemcpyDtoD_v2": [u64, u64, sz], "cuMemGetInfo_v2": [P(sz), P(sz)],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuCtxSynchronize": [],
    "cuEventCreate": [P(vp), u], "cuEventRecord": [vp, vp], "cuEventSynchronize": [vp],
    "cuEventElapsedTime": [P(C.c_float), vp, vp], "cuEventDestroy_v2": [vp], "cuGetErrorString": [i32, P(C.c_char_p)],
    "cuStreamCreate": [P(vp), u], "cuStreamWaitEvent": [vp, vp, u], "cuStreamDestroy_v2": [vp],
}
CU_STREAM_NON_BLOCKING, CU_EVENT_DISABLE_TIMING = 1, 2
NVML = {
    "nvmlInit_v2": [], "nvmlDeviceGetHandleByIndex_v2": [u, P(vp)], "nvmlSystemGetDriverVersion": [C.c_char_p, u],
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


class Nvml:
    def __init__(self):
        self.lib, self.h, self.driver = None, vp(), ""
        try:
            lib = bind(C.CDLL(os.environ.get("PEARLC_LIBNVML", "libnvidia-ml.so.1")), NVML)
        except (OSError, AttributeError):
            return
        if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByIndex_v2(0, C.byref(self.h)) == 0:
            self.lib = lib
            b = C.create_string_buffer(96)
            if lib.nvmlSystemGetDriverVersion(b, 96) == 0:
                self.driver = b.value.decode()

    def read(self) -> dict:
        sm, mem, pw, t, thr = u(), u(), u(), u(), C.c_ulonglong()
        if self.lib:
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
        self.dev = self.get("cuDeviceGet", 0)
        name = C.create_string_buffer(256)
        self.ck("cuDeviceGetName", name, 256, self.dev)
        self.name = name.value.decode()
        self.sms = self.get("cuDeviceGetAttribute", SM_COUNT, self.dev)
        self.driver = self.get("cuDriverGetVersion")
        ctx = vp()
        self.ck("cuDevicePrimaryCtxRetain", C.byref(ctx), self.dev)
        self.ck("cuCtxSetCurrent", ctx)
        self.mod = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod), cubin)
        self.fns: dict[str, vp] = {}
        self.live: set[int] = set()

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
        """Launch `name` on `stream` (None: the legacy default stream, which everything else uses)."""
        a = Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        g = tuple(grid) + (1,) * (3 - len(grid)) if isinstance(grid, (tuple, list)) else (grid, 1, 1)
        params = (vp * 1)(C.addressof(a))
        self.ck("cuLaunchKernel", self.fn(name), g[0], g[1], g[2], block, 1, 1, smem, stream, params, None)

    def stream(self) -> vp:
        """A non-blocking stream: it neither waits for nor holds up the default stream, except through `wait`."""
        s = vp()
        self.ck("cuStreamCreate", C.byref(s), CU_STREAM_NON_BLOCKING)
        return s

    def wait(self, waiter, on) -> None:
        """Work issued to stream `waiter` from here on waits for the work issued so far to stream `on`."""
        e = vp()
        self.ck("cuEventCreate", C.byref(e), CU_EVENT_DISABLE_TIMING)
        self.ck("cuEventRecord", e, on)
        self.ck("cuStreamWaitEvent", waiter, e, 0)
        self.ck("cuEventDestroy_v2", e)

    def sync(self) -> None:
        self.ck("cuCtxSynchronize")

    def timed(self, body, reps: int, stream=None) -> list[float]:
        out = []
        for _ in range(reps):
            a, b, ms = vp(), vp(), C.c_float()
            self.ck("cuEventCreate", C.byref(a), 0)
            self.ck("cuEventCreate", C.byref(b), 0)
            self.ck("cuEventRecord", a, stream)
            body()
            self.ck("cuEventRecord", b, stream)
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


def pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def depth(n: int) -> int:
    """Levels of nodes above the leaves of a frame-b3 tree over n leaves."""
    return pow2(n).bit_length() - 1


class Pipeline:
    """One (shape, forming) instance: its buffers and its kernels in order, each a (name, phase, launch) step.

    A unit starts with A's commitment: its rows' frame-b3 tree and, for unit index `unit`, the unit's seed_A over that root
    (b3_seeds, `pearl_c.unit_seed_a`, from the keys salt, root_b and seed_key) and seed_A's two subkeys (seed_subkeys): E_A's
    line key and the tickets' key.  So forming waits for A's root, as the protocol has it.  With `unit` None the keys carry
    line_a and jackpot instead (bench.py: its twin can't hash a whole A of the bench's shapes).

    The post-GEMM hashing's launches go to `hash_stream` (default: the default stream, like the rest).  On a side stream they
    fence themselves: the side stream waits for the default stream's work before the first kernel that reads C~ and U
    (hash_msg), and the default stream waits for the side stream after the tile tree, so a unit ends when its tile root is
    written."""

    GEMM_SMEM: dict[str, int] = {}

    def __init__(self, dev: Dev, m: int, n: int, k: int, forming: str, keys: dict[str, int], xa: int, xb: int,
                 hash_stream=None, unit: int | None = None):
        self.dev, self.m, self.n, self.k, self.forming, self.keys, self.hs = dev, m, n, k, forming, keys, hash_stream
        self.unit = unit
        v1 = forming == "v1"
        rowa, rowb = (4 * k, 4 * k) if v1 else (k + k // 4, k + k // 4)
        self.row_bytes = rowa
        if m % 128 or n % 128 or k % 128:
            raise ValueError(f"the kernels take m, n and k multiples of 128, not {(m, n, k)}")
        self.tiles = cdiv(m, 64) * (n // 64)
        sizes = {
            "ea_codes": m * 32, "ea_tiled": m * 32, "eb_codes": n * 32, "eb_tiled": n * 32,
            "fa_codes": k * 32, "fa_tiled": k * 32, "faT32": 32 * k, "fb_codes": k * 32, "fb_tiled": k * 32, "fbT32": 32 * k,
            "fbT128": 128 * k, "g": 128 * 32 * 4,
            "alpha_a": 4 * m, "beta_a": 4 * m, "ia_a": 4 * m, "aux_a": 8 * m,
            "alpha_b": 4 * n, "beta_b": 4 * n, "ia_b": 4 * n, "aux_b": 8 * n,
            "at": m * k, "bt": n * k, "af": m * 32 * 4, "bf": n * 32 * 4, "pa_t": m * 128, "pb_t": n * 128,
            "c": m * n * 4, "c2": m * n * 4, "u": m * n * 4, "y": m * n * 2,
            "digests": m * n, "leaves": self.tiles * 32, "tickets": (m // 16) * (n // 16) * 32,
            "tree_a": (2 * pow2(m) - 1) * 32, "tree_t": (2 * pow2(self.tiles) - 1) * 32,
            "keys_a": 32 * depth(m), "keys_t": 32 * depth(self.tiles), "seed_a": 32, "a_keys": 64,
            "qa": m * k, "qb": n * k, "sa": 4 * m, "sb": 4 * n, "y_plain": m * n * 2, "y_fast": m * n * 2, "diff": 8,
        }
        if v1:
            sizes.update({"epa_codes": m * 32, "epa_tiled": m * 32, "epb_codes": n * 32, "epb_tiled": n * 32,
                          "qa_s5": m * k, "qb_s5": n * k, "y_adopt": m * n * 2})
        else:
            sizes.update({"xa_bf16": m * k * 2, "xb_bf16": n * k * 2})
        self.size = sizes
        self.b = {name: dev.alloc(s) for name, s in sizes.items()}
        self.b["xa"], self.b["xb"] = xa, xb
        self.size["xa"], self.size["xb"] = m * rowa, n * rowb

    def free(self) -> None:
        for name, p in self.b.items():
            if name not in ("xa", "xb"):
                self.dev.free(p)

    def gemm(self, name: str, a: str, b: str, M: int, N: int, grid: int | None = None, out=("c",), pa=0, pb=0, rs=0, cs=0):
        if name not in self.GEMM_SMEM:
            probe = self.dev.alloc(16)
            self.dev.launch(name + "_smem", 1, 1, p=(probe,))
            self.dev.sync()
            self.GEMM_SMEM[name] = int.from_bytes(self.dev.read(probe, 4), "little")
            self.dev.free(probe)
            self.dev.ck("cuFuncSetAttribute", self.dev.fn(name), MAX_DYN, self.GEMM_SMEM[name])
        bn = 32 if name == "gemm_words" else 128
        tiles = (M // 128) * (N // bn)
        ptrs = [self.b[a], self.b[b], 0, 0, 0, 0, 0, 0, 0]
        for slot, key in zip((2, 3, 4), out):
            ptrs[slot] = self.b[key] if key else 0
        ptrs[5], ptrs[6], ptrs[7], ptrs[8] = pa, pb, rs, cs
        self.dev.launch(name, min(tiles, grid or self.dev.sms), 384, self.GEMM_SMEM[name], p=ptrs, i=(M, N, self.k, 8))

    def levels(self, name: str, count: int, dom: str, keys: str, s=None) -> None:
        """A frame-b3 tree over `count` leaves already at level 0 of buffer `name`, on stream s: its pads, its level keys (into
        buffer `keys`), then each level above."""
        d, base, width, lk = self.dev, self.b[name], pow2(count), self.b[keys]
        if width > count:
            d.launch("hash_pad", cdiv(width - count, 128), 128, p=(base, self.keys[dom]), i=(count, width), stream=s)
        if width > 1:
            d.launch("hash_node_keys", 1, 32, p=(lk, self.keys[dom]), i=(depth(count),), stream=s)
        off, level = 0, 0
        while width > 1:
            d.launch("hash_nodes", cdiv(width // 2, 128), 128, p=(base + 32 * off, base + 32 * (off + width), lk + 32 * level),
                     i=(level, width // 2), stream=s)
            off, width, level = off + width, width // 2, level + 1

    def fence_in(self) -> None:
        if self.hs is not None:
            self.dev.wait(self.hs, None)

    def fence_out(self) -> None:
        if self.hs is not None:
            self.dev.wait(None, self.hs)

    def steps(self) -> list[tuple[str, str, object]]:
        d, b, m, n, k, v1 = self.dev, self.b, self.m, self.n, self.k, self.forming == "v1"
        fl = 1 if v1 else 0

        def lines(tag, key, side, factor, count, codes, tiled_, t32=0, t128=0):
            norm = NORM_16 if v1 and factor == 1 else NORM_256
            return (f"lines_{tag}", lambda: d.launch("lines", cdiv(count, 128), 128, p=(key, b[codes], b[tiled_], t32, t128),
                                                    i=(side, factor, count, norm)))

        def form_side(tag, rows, x, e_codes, e_tiled, f_tiled, out):
            ep_codes, ep_tiled = f"ep{tag}_codes", f"ep{tag}_tiled"
            if v1:
                return [(f"stats_{tag}", lambda: d.launch("stats_s5", cdiv(rows, 8), 256, p=(b[x], b[f"alpha_{tag}"], b[f"beta_{tag}"],
                                                                                           b[f"ia_{tag}"], b[f"aux_{tag}"]), i=(rows, k))),
                        (f"eprime_{tag}", lambda: d.launch("eprime", cdiv(rows, 128), 128, p=(b[e_codes], b[f"beta_{tag}"], b[ep_codes],
                                                                                            b[ep_tiled]), i=(rows,))),
                        (f"form_{tag}", lambda: d.launch("form_s5", (k // 128, rows // 64), 128,
                                                         p=(b[x], b[f"alpha_{tag}"], 0, b[ep_tiled], b[f_tiled], b[out]), i=(rows, k)))]
            return [(f"norms_{tag}", lambda: d.launch("norms", cdiv(rows, 8), 256, p=(b[x], b[f"alpha_{tag}"], b[f"beta_{tag}"],
                                                                                     b[f"ia_{tag}"], b[f"aux_{tag}"]), i=(rows, k))),
                    (f"form_{tag}", lambda: d.launch("form", (k // 128, rows // 64), 128,
                                                     p=(b[x], b[f"alpha_{tag}"], b[f"beta_{tag}"], b[e_tiled], b[f_tiled], b[out]),
                                                     i=(rows, k)))]

        noise_a = "epa_codes" if v1 else "ea_codes"
        noise_b = "epb_codes" if v1 else "eb_codes"
        job = [lines("eb", self.keys["line_b"], 1, 0, n, "eb_codes", "eb_tiled"),
               lines("fa", self.keys["line_b"], 0, 1, k, "fa_codes", "fa_tiled", b["faT32"]),
               lines("fb", self.keys["line_b"], 1, 1, k, "fb_codes", "fb_tiled", b["fbT32"], b["fbT128"]),
               *form_side("b", n, "xb", "eb_codes", "eb_tiled", "fb_tiled", "bt"),
               ("bf", lambda: self.gemm("gemm_words", "bt", "faT32", n, 32, out=("bf",))),
               ("g", lambda: self.gemm("gemm_words", "fbT128", "faT32", 128, 32, out=("g",))),
               ("pb", lambda: d.launch("pb", cdiv(n * 32, 256), 256, p=(b[noise_b], b["beta_b"], b["bf"], b["g"], b["pb_t"]), i=(n, fl)))]
        root_a = b["tree_a"] + 32 * (2 * pow2(m) - 2)
        if self.unit is None:
            line_a, jackpot, seed = self.keys["line_a"], self.keys["jackpot"], []
        else:
            line_a, jackpot = b["a_keys"], b["a_keys"] + 32
            seed = [("seed_a", lambda: d.launch("b3_seeds", 1, 128, p=(root_a, b["seed_a"], self.keys["seed_key"], self.keys["salt"],
                                                                    self.keys["root_b"]), i=(1, self.unit, 0))),
                    ("a_keys", lambda: d.launch("seed_subkeys", 1, 32, p=(b["seed_a"], b["a_keys"])))]
        commit = [("hash_rows", lambda: d.launch("hash_rows", m, 256, p=(b["xa"], b["tree_a"], self.keys["dom_a"]),
                                                 i=(m, self.row_bytes))),
                  ("tree_a", lambda: self.levels("tree_a", m, "dom_a", "keys_a")), *seed]
        forming = [lines("ea", line_a, 0, 0, m, "ea_codes", "ea_tiled"),
                   *form_side("a", m, "xa", "ea_codes", "ea_tiled", "fa_tiled", "at")]
        gemm = [("gemm_ct", lambda: self.gemm("gemm_ct", "at", "bt", m, n, out=("c2",)))]
        cleanup = [("af", lambda: self.gemm("gemm_words", "at", "fbT32", m, 32, out=("af",))),
                   ("pa", lambda: d.launch("pa", cdiv(m * 64, 256), 256, p=(b[noise_a], b["beta_a"], b["af"], b["pa_t"]), i=(m, fl))),
                   ("gemm_pearlc", lambda: self.gemm("gemm_pearlc", "at", "bt", m, n, out=("c", "u", "y"), pa=b["pa_t"],
                                                     pb=b["pb_t"], rs=b["ia_a"], cs=b["ia_b"]))]
        hs = self.hs
        hashing = [("hash_msg", lambda: (self.fence_in(), d.launch("hash_msg", cdiv(m * n // 32, 128), 128,
                                                                   p=(b["c"], b["u"], b["digests"], self.keys["k_msg"]),
                                                                   i=(m, n), stream=hs))),
                   ("hash_leaf", lambda: d.launch("hash_leaf", cdiv(self.tiles, 64), 64, p=(b["digests"], b["leaves"],
                                                                                           self.keys["k_leaf"]), i=(m, n),
                                                  stream=hs)),
                   ("tickets", lambda: d.launch("tickets", cdiv((m // 16) * (n // 16), 128), 128,
                                                p=(b["c"], b["tickets"], jackpot), i=(m, n), stream=hs)),
                   ("tree_t", lambda: (d.launch("hash_tree_leaves", cdiv(self.tiles, 128), 128,
                                                p=(b["leaves"], b["tree_t"], self.keys["dom_t"]), i=(self.tiles,), stream=hs),
                                       self.levels("tree_t", self.tiles, "dom_t", "keys_t", hs), self.fence_out()))]
        plain_in_a, plain_in_b = ("xa", "xb") if v1 else ("xa_bf16", "xb_bf16")
        prep = [] if v1 else [
            ("open_a", lambda: d.launch("open_rows", cdiv(m * k // 8, 256), 256, p=(b["xa"], b["xa_bf16"]), i=(m, k))),
            ("open_b", lambda: d.launch("open_rows", cdiv(n * k // 8, 256), 256, p=(b["xb"], b["xb_bf16"]), i=(n, k)))]
        plain_job = [("plainq_b", lambda: d.launch("plainq", n, 256, p=(b[plain_in_b], b["qb"], b["sb"]), i=(n, k, fl)))]
        plain = [("plainq_a", lambda: d.launch("plainq", m, 256, p=(b[plain_in_a], b["qa"], b["sa"]), i=(m, k, fl))),
                 ("gemm_plain", lambda: self.gemm("gemm_plain", "qa", "qb", m, n, out=(None, None, "y_plain"), rs=b["sa"], cs=b["sb"])),
                 ("gemm_fast", lambda: self.gemm("gemm_fast", "qa", "qb", m, n, out=(None, None, "y_fast"), rs=b["sa"], cs=b["sb"]))]
        adopted_job = [("quant_b", lambda: d.launch("quant_s5", cdiv(n * k // 16, 256), 256, p=(b["xb"], b["alpha_b"], b["qb_s5"]),
                                                    i=(n, k)))] if v1 else []
        adopted = [("quant_a", lambda: d.launch("quant_s5", cdiv(m * k // 16, 256), 256, p=(b["xa"], b["alpha_a"], b["qa_s5"]),
                                                i=(m, k))),
                   ("gemm_adopt", lambda: self.gemm("gemm_plain", "qa_s5", "qb_s5", m, n, out=(None, None, "y_adopt"), rs=b["ia_a"],
                                                    cs=b["ia_b"]))] if v1 else []
        tag = lambda phase, xs: [(nm, phase, f) for nm, f in xs]  # noqa: E731
        return (tag("job", job) + tag("commit", commit) + tag("forming", forming) + tag("gemm", gemm) + tag("cleanup", cleanup)
                + tag("hashing", hashing)
                + tag("prep", prep) + tag("plain_job", plain_job) + tag("plain", plain) + tag("adopted_job", adopted_job)
                + tag("adopted", adopted))

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

PLAIN_BUFFERS = {"qa", "qb", "sa", "sb", "y_plain", "y_fast", "xa_bf16", "xb_bf16", "qa_s5", "qb_s5", "y_adopt"}
GEMM_RERUNS = (("gemm_pearlc", ("c", "u", "y")), ("gemm_ct", ("c2",)), ("gemm_plain", ("y_plain",)), ("gemm_fast", ("y_fast",)),
               ("af", ("af",)), ("gemm_adopt", ("y_adopt",)))


def compare(dev: Dev, pipe: Pipeline, name: str, want: bytes, alias: str | None = None) -> dict:
    got = dev.read(pipe.b[alias or name], len(want))
    if got == want:
        return {"ok": True, "bytes": len(want)}
    bad = [i for i in range(0, len(want), 4) if got[i:i + 4] != want[i:i + 4]]
    return {"ok": False, "bytes": len(want), "words_differing": len(bad),
            "first": [{"offset": i, "got": got[i:i + 4].hex(), "want": want[i:i + 4].hex()} for i in bad[:6]]}


def check_ieee(dev: Dev, man: dict, base: str) -> dict:
    """The test kernels of the only `.FTZ` sequences build.sh admits (nvcc's correctly rounded div, rcp and sqrt, ieee_ops and
    ieee_q) against the fixture's correctly rounded words, subnormal inputs and outputs included."""
    spec = man["ieee"]
    n, load = spec["count"], lambda f: open(os.path.join(base, f["file"]), "rb").read()  # noqa: E731
    x, y = dev.put(load(spec["files"]["ieee_x"])), dev.put(load(spec["files"]["ieee_y"]))
    ops, q = dev.alloc(24 * n), dev.alloc(4 * n)
    dev.launch("ieee_ops", cdiv(n, 128), 128, p=(x, y, ops), i=(n,))
    dev.launch("ieee_q", cdiv(n, 128), 128, p=(x, q), i=(n,))
    dev.sync()
    out = {}
    for name, p, size in (("ieee_ops", ops, 24 * n), ("ieee_q", q, 4 * n)):
        got, want = dev.read(p, size), load(spec["files"][name])
        bad = [i for i in range(0, size, 4) if got[i:i + 4] != want[i:i + 4]]
        out[name] = {"ok": not bad, "bytes": size, "words_differing": len(bad),
                     "first": [{"offset": i, "got": got[i:i + 4].hex(), "want": want[i:i + 4].hex()} for i in bad[:6]]}
    for p in (x, y, ops, q):
        dev.free(p)
    return out


def check(dev: Dev, man: dict, base: str) -> dict:
    m, n, k = man["shape"]
    keys = {name: dev.put(bytes.fromhex(h)) for name, h in man["keys"].items()}
    load = lambda d, f: open(os.path.join(base, d, f["file"]), "rb").read()  # noqa: E731
    ieee = check_ieee(dev, man, base)
    results = {}
    for forming, spec in man["formings"].items():
        files = spec["files"]
        xa, xb = dev.put(load(forming, files["xa"])), dev.put(load(forming, files["xb"]))
        pipe = Pipeline(dev, m, n, k, forming, keys, xa, xb, unit=man["unit"]["index"])
        pipe.run_all()
        res = {}
        for name, f in man["common"].items():
            res[name] = compare(dev, pipe, name, load("", f))
        for name, f in files.items():
            if name in ("xa", "xb"):
                continue
            want = load(forming, f)
            res[name] = compare(dev, pipe, name, want)
            if name == "c":
                res["c@gemm_ct"] = compare(dev, pipe, name, want, alias="c2")
        steps = {nm: fn for nm, _, fn in pipe.steps()}
        for grid in (1, 3):
            for step, outs in GEMM_RERUNS:
                if step not in steps:
                    continue
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
                    res[f"{o}@grid{grid}"] = compare(dev, pipe, o, load(forming, files["c" if o == "c2" else o]))
        pipe.free()
        dev.free(xa)
        dev.free(xb)
        res.update(ieee)
        failed = sorted(nm for nm, r in res.items() if not r["ok"])
        ieee_ok = all(r["ok"] for r in ieee.values())
        results[forming] = {"buffers": len(res), "failed": failed,
                            "pearl_c_ok": not [x for x in failed if x.split("@")[0] not in PLAIN_BUFFERS],
                            "plain_ok": ieee_ok and not [x for x in failed if x.split("@")[0] in PLAIN_BUFFERS],
                            "detail": {nm: r for nm, r in res.items() if not r["ok"]}}
    for p in keys.values():
        dev.free(p)
    return results


# ---- the timing ---------------------------------------------------------------------------------------------------

#: a unit's phases: A's commitment and seed_A, then E_A and A's forming, the GEMM, the clean-up, the post-GEMM hashing
UNIT_PHASES = ("commit", "forming", "gemm", "cleanup", "hashing")
#: the timing's unit index: seed_A is derived from A's root on the device, so no stand-in seed keys E_A
TIMING_UNIT = 0


def timing(dev: Dev, nv: Nvml, size: int, forming: str, reps: int, gates: dict) -> dict:
    m = n = k = size
    keys = {name: dev.put(hashlib.sha256(f"pearl-c-timing/{name}".encode()).digest())
            for name in ("salt", "root_b", "line_b", "dom_a", "dom_t")}
    keys.update({name: dev.put(bytes.fromhex(h)) for name, h in (*H1_KEYS.items(), ("seed_key", SEED_KEY))})
    rowb = 4 * k if forming == "v1" else k + k // 4
    xa, xb = dev.alloc(m * rowb), dev.alloc(n * rowb)
    for x, rows, seed in ((xa, m, 11), (xb, n, 12)):
        if forming == "v1":
            dev.launch("gen_f32", cdiv(rows * k, 256), 256, p=(x,), i=(rows, k, seed))
        else:
            dev.launch("gen_fp10", cdiv(rows * rowb // 4, 256), 256, p=(x,), i=(rows, k, seed))
    dev.sync()
    hs = dev.stream() if os.environ.get("HASH_STREAM") else None
    pipe = Pipeline(dev, m, n, k, forming, keys, xa, xb, hash_stream=hs, unit=TIMING_UNIT)
    pipe.run_all()
    out = measure(dev, nv, pipe, reps, gates)
    out["hash_stream"] = hs is not None
    pipe.free()
    for p_ in (xa, xb, *keys.values()):
        dev.free(p_)
    if hs is not None:
        dev.ck("cuStreamDestroy_v2", hs)
    return out


def measure(dev: Dev, nv: Nvml, pipe: Pipeline, reps: int, gates: dict) -> dict:
    """The timed part of `timing` on a pipeline whose kernels have run once: medians, phases, ratios, consistency."""
    m, n, k, forming, steps = pipe.m, pipe.n, pipe.k, pipe.forming, pipe.steps()
    before = nv.read()
    med, runs = {}, {}
    for name, phase, f in steps:
        if phase in UNIT_PHASES + ("job",) and not gates["pearl_c"] or \
                phase in ("plain", "plain_job", "adopted", "adopted_job") and not gates["plain"]:
            continue
        t = dev.timed(f, reps, pipe.hs if phase == "hashing" else None)
        runs[name], med[name] = [round(x, 4) for x in t], statistics.median(t)
    after = nv.read()
    phases = {ph: sum(med.get(nm, 0.0) for nm, p, _ in steps if p == ph) for ph in ("job",) + UNIT_PHASES}
    out = {"kind": "timing", "shape": [m, n, k], "forming": forming, "reps": reps, "gates": gates,
           "ms": {nm: round(v, 4) for nm, v in med.items()}, "runs": runs, "clk_before": before, "clk_after": after}
    if gates["pearl_c"]:
        unit = [f for nm, p, f in steps if p in ("commit", "forming", "cleanup", "hashing")]
        no_hash = [f for nm, p, f in steps if p in ("forming", "cleanup")]
        e2e = statistics.median(dev.timed(lambda: [f() for f in unit], reps))
        e2e_nh = statistics.median(dev.timed(lambda: [f() for f in no_hash], reps))
        split = {"commit": phases["commit"], "forming": phases["forming"], "gemm": med["gemm_ct"],
                 "cleanup": med["af"] + med["pa"] + med["gemm_pearlc"] - med["gemm_ct"], "hashing": phases["hashing"]}
        out.update({"split_ms": {kk: round(v, 4) for kk, v in split.items()}, "unit_sum_ms": round(sum(split.values()), 4),
                    "unit_e2e_ms": round(e2e, 4), "unit_e2e_no_hash_ms": round(e2e_nh, 4), "job_ms": round(phases["job"], 4)})
        out["consistency"] = {"c_ct_vs_pearlc_words_differing": pipe.diff("c2", "c")}
        dev.copy(pipe.b["c2"], pipe.b["u"], pipe.size["u"])
        [f for nm, _, f in steps if nm == "gemm_pearlc"][0]()
        out["consistency"]["u_rerun_words_differing"] = pipe.diff("c2", "u")
    if gates["plain"]:
        out["plain_unit_ms"] = round(med["plainq_a"] + med["gemm_plain"], 4)
        out["fast_unit_ms"] = round(med["plainq_a"] + med["gemm_fast"], 4)
        out["plain_tflops"] = round(2 * m * n * k / med["gemm_plain"] / 1e9, 1)
        out["fast_tflops"] = round(2 * m * n * k / med["gemm_fast"] / 1e9, 1)
        if "gemm_adopt" in med and "stats_a" in med:
            out["adopted_unit_ms"] = round(med["stats_a"] + med["quant_a"] + med["gemm_adopt"], 4)
        if gates["pearl_c"]:
            for key, base, job in (("ratio_vs_plain", out["plain_unit_ms"], med["plainq_b"]),
                                   ("ratio_vs_adopted", out.get("adopted_unit_ms"), med.get("stats_b", 0) + med.get("quant_b", 0))):
                if base:
                    out[key] = {"unit_e2e": round(out["unit_e2e_ms"] / base, 4), "unit_sum": round(out["unit_sum_ms"] / base, 4),
                                "no_hash_e2e": round(out["unit_e2e_no_hash_ms"] / base, 4),
                                "chain_only": round(med["gemm_ct"] / med["gemm_plain"], 4),
                                "with_job": round((out["unit_e2e_ms"] + out["job_ms"]) / (base + job), 4)}
    return out


def main(argv: list[str]) -> int:
    man_path = argv[1] if len(argv) > 1 else os.path.join(HERE, "check", "check.json")
    man = json.load(open(man_path))
    base = os.path.dirname(man_path)
    image = open(os.path.join(HERE, "pearl_c.cubin"), "rb").read()
    nv = Nvml()
    dev = Dev(image)
    env = {"kind": "env", "device": dev.name, "sms": dev.sms, "cuda_driver_api": dev.driver, "nvml_driver": nv.driver,
           "cubin_sha256": hashlib.sha256(image).hexdigest(), "check_json_sha256": hashlib.sha256(open(man_path, "rb").read()).hexdigest(),
           "mem_free_total": dev.free_mem(), "clk": nv.read(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    emit(env)
    if dev.name != H100 and not os.environ.get("PEARLC_ANY_GPU"):
        print(f"REFUSED: GPU is {dev.name!r}, not an H100 SXM ({H100})", file=sys.stderr)
        return 3
    t0 = time.time()
    res = check(dev, man, base)
    emit({"kind": "check", "shape": man["shape"], "seconds": round(time.time() - t0, 2), "formings": res})
    ungated = bool(os.environ.get("PEARLC_UNGATED"))
    ok = all(r["pearl_c_ok"] and r["plain_ok"] for r in res.values())
    print(f"check at {man['shape']}: " + "; ".join(f"{f} {r['buffers']} buffers, failed {r['failed'] or 'none'}" for f, r in res.items()),
          file=sys.stderr)
    reps = int(os.environ.get("REPS", "5"))
    for size in [int(s) for s in os.environ.get("SHAPES", "8192 16384").split()]:
        for forming in ("v1", "v0"):
            gates = {"pearl_c": res[forming]["pearl_c_ok"] or ungated, "plain": res[forming]["plain_ok"] or ungated,
                     "ungated": ungated}
            if not (gates["pearl_c"] or gates["plain"]):
                continue
            t0 = time.time()
            row = timing(dev, nv, size, forming, reps, gates)
            row["seconds"] = round(time.time() - t0, 2)
            emit(row)
            if "ratio_vs_plain" in row:
                s, r = row["split_ms"], row["ratio_vs_plain"]
                print(f"{size}^3 {forming}: unit {row['unit_e2e_ms']:.3f} ms end to end = {r['unit_e2e']:.3f}x plain FP8 "
                      f"({row['plain_unit_ms']:.3f} ms, {row['plain_tflops']} TFLOPS); split GEMM {s['gemm']:.3f}, clean-up "
                      f"{s['cleanup']:.3f}, A's commitment and seed {s['commit']:.3f}, forming {s['forming']:.3f}, hashing {s['hashing']:.3f}; no hashing {r['no_hash_e2e']:.3f}x; "
                      f"consistency {row['consistency']}" + (f"; {row['ratio_vs_adopted']['unit_e2e']:.3f}x the adopted quantizer's "
                      f"{row['adopted_unit_ms']:.3f} ms" if "ratio_vs_adopted" in row else ""), file=sys.stderr)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main(sys.argv))
