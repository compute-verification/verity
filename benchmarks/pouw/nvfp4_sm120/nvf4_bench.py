#!/usr/bin/env python3
"""Bench for mainloop_nvf4.cuh on one RTX PRO 6000 Blackwell (sm_120), through the CUDA driver API (ctypes on libcuda.so.1).

nvf4_bench.cubin is nvf4_bench.cu built off the card:

  nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin -o nvf4_bench.cubin nvf4_bench.cu

0. Identity, before any launch: one visible device (CUDA_VISIBLE_DEVICES a single index), compute capability 12.0, an
   "NVIDIA RTX PRO 6000 Blackwell", its UUID equal to NVF4_EXPECT_UUID (the lease's GPU_LEASE_UUID).  NVML by that UUID.
1. Operands per scale range (SCALES, default "0 1"): random E2M1 codes; UE4M3 scale bytes, every code 0x00-0x7E (0) or
   0x30-0x3F (1), never bit 7 or 0x7F.
2. Gates, at each of SMALL's shapes and at SHAPE (m = n = k):
   model   nv_ref_words (one warp per m16 x n8 block, fragments read from global memory) against BLACKWELL_SM120_NVF4
           (verity.ml.kernels.tc.block_scaled_batch chained from +0): every word at shapes of at most FULL_MODEL coordinates,
           MODEL_SAMPLES coordinates (every row mod 16 and column mod 8) otherwise.  Needs numpy and verity importable.
   words   each of KERNELS' FP32-word kernel (_w) against nv_ref_words, word for word: full grid, 1 CTA, 3 CTAs, then
           GATE_REPEATS more full-grid runs.
    bf16    each BF16 kernel (_b) against nv_ref_words rounded to BF16 (nv_to_bf16) at the words gate's grids (full, 1 CTA,
           3 CTAs, GATE_REPEATS more full), and at full grid against each CUTLASS kernel's D and cuBLASLt's (HARNESS_LIB,
           the harness's libpouw_harness.so), element for element.
   plain   each NVFP4 entry of each PLAIN_LIB (a library with the registry's ph_cutlass_* ABI alone, nvf4_plain.cu built with
           -DNVF4_PLAIN_ABI) against nv_ref_words rounded to BF16, once, and GATE_REPEATS more times at SHAPE.
   A kernel or PLAIN_LIB entry is timed only if every gate it is in passed at every shape and scale range, and the references
   passed theirs.
3. Diagnostic timing (not the headline; GPU 6's harness times baselines): at SHAPE on scale range TIME_SCALES, the passed BF16
   kernels, the CUTLASS kernels, the passed PLAIN_LIB entries and cuBLASLt's fastest candidate, REPS rounds round-robin (events
   around one launch), the median and TFLOPS; NVML clocks before and after.  STEADY=N: then batches of N back to back per
   kernel, NVML during.

  python3 nvf4_bench.py CUBIN   (env: NVF4_EXPECT_UUID, SHAPE=8192 (or several: each gated, then each timed),
                                 SMALL="256x128x128 512x256x1024 768x384x640",
                                 KERNELS="nv256 nv256_o", SCALES="0 1", TIME_SCALES=1, GATE_REPEATS=3, REPS=20,
                                 RASTER=8, HARNESS_LIB, CUTLASS="cutlass3x_nvfp4_256x128x128_coop ...", LT=1,
                                 LT2_LIB (the harness's libpouw_lt2.so: its cuBLASLt too, as "lt2"), PLAIN_LIB,
                                 PLAIN_ENTRIES (only these of PLAIN_LIB's entries),
                                 MODEL_SAMPLES=4096, FULL_MODEL=131072, STEADY, STEADY_ROUNDS=4, ONLY_CHECK,
                                 PROFILE=1: after timing, one launch of each timed kernel between cuProfilerStart and
                                 cuProfilerStop, for ncu --profile-from-start off)
One JSON object per line on stdout, a summary on stderr.  Exit 3: refused (identity); 4: a gate failed; 0 otherwise.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
import random
import statistics
import sys
import time

NAME_PREFIX = "NVIDIA RTX PRO 6000 Blackwell"
SM_COUNT, MAX_DYN, CC_MAJOR, CC_MINOR = 16, 8, 75, 76
UINT8, SWIZZLE_64B, L2_128B = 0, 2, 2
NVFP4 = 5  # the harness's kind

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER
CUDA = {
    "cuInit": [u], "cuDriverGetVersion": [P(i32)], "cuDeviceGetCount": [P(i32)], "cuDeviceGet": [P(i32), i32],
    "cuDeviceGetName": [C.c_char_p, i32, i32], "cuDeviceGetUuid_v2": [C.c_char_p, i32],
    "cuDeviceGetAttribute": [P(i32), i32, i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuFuncSetAttribute": [vp, i32, i32], "cuMemAlloc_v2": [P(u64), sz], "cuMemFree_v2": [u64],
    "cuMemsetD8_v2": [u64, C.c_ubyte, sz], "cuMemcpyHtoD_v2": [u64, vp, sz], "cuMemcpyDtoH_v2": [vp, u64, sz],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuCtxSynchronize": [],
    "cuEventCreate": [P(vp), u], "cuEventRecord": [vp, vp], "cuEventSynchronize": [vp], "cuEventQuery": [vp],
    "cuEventElapsedTime": [P(C.c_float), vp, vp], "cuEventDestroy_v2": [vp], "cuGetErrorString": [i32, P(C.c_char_p)],
    "cuTensorMapEncodeTiled": [vp, i32, u, vp, P(u64), P(u64), P(u), P(u), i32, i32, i32, i32],
    "cuProfilerStart": [], "cuProfilerStop": [],
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


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class Nvml:
    """NVML on the device of one UUID; reads nothing (and says so) if NVML or the UUID isn't there."""

    def __init__(self, uuid: str):
        self.lib, self.h, self.driver = None, vp(), ""
        try:
            lib = bind(C.CDLL("libnvidia-ml.so.1"), NVML)
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
        self.cu = bind(C.CDLL("libcuda.so.1"), CUDA)
        self.ck("cuInit", 0)
        self.count = self.get("cuDeviceGetCount")
        self.dev = self.get("cuDeviceGet", 0)
        name, raw = C.create_string_buffer(256), C.create_string_buffer(16)
        self.ck("cuDeviceGetName", name, 256, self.dev)
        self.ck("cuDeviceGetUuid_v2", raw, self.dev)
        h = raw.raw[:16].hex()
        self.name, self.uuid = name.value.decode(), f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
        self.sms = self.get("cuDeviceGetAttribute", SM_COUNT, self.dev)
        self.cc = (self.get("cuDeviceGetAttribute", CC_MAJOR, self.dev), self.get("cuDeviceGetAttribute", CC_MINOR, self.dev))
        self.driver = self.get("cuDriverGetVersion")
        self.cubin, self.mod, self.fns, self.smem = cubin, None, {}, {}

    def identity(self, expect: str | None, env: str = "NVF4_EXPECT_UUID") -> list[str]:
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
        if not expect:
            bad.append(f"{env} is not set (the lease's GPU_LEASE_UUID)")
        elif self.uuid != expect:
            bad.append(f"UUID {self.uuid} is not the expected {expect}")
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
        return p.value

    def free(self, p: int) -> None:
        self.ck("cuMemFree_v2", u64(p))

    def put(self, data: bytes) -> int:
        p = self.alloc(len(data))
        self.ck("cuMemcpyHtoD_v2", u64(p), C.c_char_p(data), len(data))
        return p

    def read(self, p: int, n: int) -> bytes:
        buf = C.create_string_buffer(n)
        self.ck("cuMemcpyDtoH_v2", buf, u64(p), n)
        return buf.raw

    def fill(self, p: int, byte: int, n: int) -> None:
        self.ck("cuMemsetD8_v2", u64(p), byte, n)

    def launch(self, name: str, grid: int, block: int, smem: int = 0, p=(), i=()) -> None:
        a = Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        params = (vp * 1)(C.addressof(a))
        self.ck("cuLaunchKernel", self.fn(name), grid, 1, 1, block, 1, 1, smem, None, params, None)

    def sync(self) -> None:
        self.ck("cuCtxSynchronize")

    def event(self) -> vp:
        e = vp()
        self.ck("cuEventCreate", C.byref(e), 0)
        return e

    def timed(self, body) -> float:
        a, b, ms = self.event(), self.event(), C.c_float()
        self.ck("cuEventRecord", a, None)
        body()
        self.ck("cuEventRecord", b, None)
        self.ck("cuEventSynchronize", b)
        self.ck("cuEventElapsedTime", C.byref(ms), a, b)
        self.cu.cuEventDestroy_v2(a)
        self.cu.cuEventDestroy_v2(b)
        return ms.value


class LtCandidate(C.Structure):
    _fields_ = [("algo", C.c_uint64 * 8), ("workspace", C.c_size_t), ("waves", C.c_float), ("algo_id", i32), ("tile", i32),
                ("stages", i32), ("splitk", i32), ("reduction", i32), ("swizzle", i32), ("custom", i32),
                ("inner_shape", i32), ("cluster_shape", i32)]


class Harness:
    """The harness's CUTLASS registry and cuBLASLt calls (libpouw_harness.so), on this process's primary context, for the
    kernels of one harness kind (NVFP4 by default).  A registry alone (PLAIN_LIB) has no cuBLASLt; libpouw_lt2.so (LT2_LIB)
    has cuBLASLt alone."""

    def __init__(self, path: str, kind: int = NVFP4):
        L = self.lib = C.CDLL(path)
        self.kind = kind
        if hasattr(L, "ph_last_error"):
            L.ph_last_error.restype = C.c_char_p
        if hasattr(L, "ph_cutlass_count"):
            L.ph_cutlass_describe.argtypes = [i32, P(C.c_char_p), P(i32), P(i32)]
            L.ph_cutlass_query.argtypes = [i32, i32, i32, i32, P(sz), P(sz), P(sz)]
            L.ph_cutlass_prepare.argtypes = [i32, i32, i32, i32, vp, vp, vp, vp, vp, vp, sz, vp, P(vp)]
            L.ph_cutlass_launch.argtypes = [vp, vp]
            L.ph_cutlass_release.argtypes = [vp]
        if hasattr(L, "ph_lt_plan_create"):
            L.ph_lt_plan_create.argtypes = [i32, i32, i32, i32, sz, i32, P(vp)]
            L.ph_lt_plan_destroy.argtypes = [vp]
            L.ph_lt_candidates.argtypes = [vp, vp, vp, i32, P(LtCandidate), P(i32)]
            L.ph_lt_run.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp]
        self.sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()
        self.index = {}
        for j in range(L.ph_cutlass_count() if hasattr(L, "ph_cutlass_count") else 0):
            name, kd, swap = C.c_char_p(), i32(), i32()
            if L.ph_cutlass_describe(j, C.byref(name), C.byref(kd), C.byref(swap)) == 0 and kd.value == kind:
                self.index[name.value.decode()] = j

    def err(self, what: str, r: int) -> RuntimeError:
        last = self.lib.ph_last_error() if hasattr(self.lib, "ph_last_error") else b""
        return RuntimeError(f"{what} {r}: {(last or b'').decode()}")

    def sf_bytes(self, m: int, n: int, k: int) -> tuple[int, int]:
        """The scale bytes a kernel of this kind reads: NVFP4's per 16, none for the per-tensor FP8 kind."""
        return (m * k // 16, n * k // 16) if self.kind == NVFP4 else (0, 0)

    def cutlass(self, dev: Dev, name: str, m: int, n: int, k: int, ops: dict, d: int):
        """A launcher of CUTLASS kernel `name` writing D = A B^T (BF16) to d, and its record."""
        j = self.index[name]
        ws, sfa, sfb = sz(), sz(), sz()
        r = self.lib.ph_cutlass_query(j, m, n, k, C.byref(ws), C.byref(sfa), C.byref(sfb))
        if r:
            raise self.err(f"ph_cutlass_query {name}", r)
        if (sfa.value, sfb.value) != self.sf_bytes(m, n, k):
            raise RuntimeError(f"{name}: scale bytes {sfa.value}, {sfb.value}, not {self.sf_bytes(m, n, k)}")
        wsp = dev.alloc(ws.value) if ws.value else 0
        h = vp()
        r = self.lib.ph_cutlass_prepare(j, m, n, k, vp(ops["a"]), vp(ops["b"]), vp(ops["sfa"]), vp(ops["sfb"]), vp(d),
                                        vp(wsp or None), ws.value, None, C.byref(h))
        if r:
            raise self.err(f"ph_cutlass_prepare {name}", r)

        def run() -> None:
            r = self.lib.ph_cutlass_launch(h, None)
            if r:
                raise self.err(f"ph_cutlass_launch {name}", r)

        def release() -> None:
            self.lib.ph_cutlass_release(h)
            if wsp:
                dev.free(wsp)

        return run, release, {"workspace": ws.value}

    def lt(self, dev: Dev, m: int, n: int, k: int, ops: dict, d: int, cands: int = 8):
        """cuBLASLt's GEMM of this kind (NVFP4: VEC16_UE4M3 scales; FP8: per-tensor scales; BF16 out): the heuristic's first
        candidates, each timed briefly, the fastest kept."""
        ws_bytes = 64 << 20
        ws, plan = dev.alloc(ws_bytes), vp()
        r = self.lib.ph_lt_plan_create(self.kind, m, n, k, ws_bytes, 0, C.byref(plan))
        if r:
            raise self.err("ph_lt_plan_create", r)
        cs, got = (LtCandidate * cands)(), i32()
        r = self.lib.ph_lt_candidates(plan, vp(ops["sfa"]), vp(ops["sfb"]), cands, cs, C.byref(got))
        if r:
            raise self.err("ph_lt_candidates", r)

        def run_j(j: int) -> None:
            r = self.lib.ph_lt_run(plan, C.byref(cs[j]), vp(ops["a"]), vp(ops["b"]), vp(d), vp(ops["sfa"]), vp(ops["sfb"]),
                                   vp(ws), None)
            if r:
                raise self.err("ph_lt_run", r)

        tried, best = [], None
        for j in range(got.value):
            if cs[j].workspace > ws_bytes:
                continue
            run_j(j)
            dev.sync()
            t = statistics.median(dev.timed(lambda j=j: run_j(j)) for _ in range(5))
            tried.append({"i": j, "ms": round(t, 4), **{f: getattr(cs[j], f) for f in ("algo_id", "tile", "stages", "splitk",
                                                                                        "swizzle", "custom", "inner_shape")}})
            if best is None or t < tried[best]["ms"]:
                best = len(tried) - 1
        if best is None:
            raise RuntimeError(f"cuBLASLt: no kind {self.kind} candidate")
        j = tried[best]["i"]

        def release() -> None:
            self.lib.ph_lt_plan_destroy(plan)
            dev.free(ws)

        return (lambda: run_j(j)), release, {"candidates": got.value, "tried": tried, "best": best}


class Bench:
    """NVFP4's operands, kernels and model gate; fp8_bench.py's subclass swaps them for FP8's.  `lts` are the cuBLASLt
    baselines, (name, Harness), set by main."""

    KIND = NVFP4
    KERNELS = "nv256 nv256_o"
    CUTLASS = "cutlass3x_nvfp4_256x128x128_coop cutlass3x_nvfp4_128x128x128_coop"
    UUID_ENV = "NVF4_EXPECT_UUID"

    def __init__(self, dev: Dev):
        self.dev = dev
        self.raster = int(os.environ.get("RASTER", "8"))
        self.lts: list[tuple[str, Harness]] = []

    def model_gate(self, ops: dict, ref: int, m: int, n: int, k: int, samples: int, full: int, seed: int) -> dict:
        return model_gate(self.dev, ops, ref, m, n, k, samples, full, seed)

    @staticmethod
    def model_ok(rec: dict) -> bool:
        return rec.get("words_differing", 0) == 0

    def tmap(self, ptr: int, k: int, rows: int, box_rows: int) -> bytes:
        """A CUtensorMap over a row-major rows x (k / 2) byte matrix: box_rows x 64-byte boxes, SWIZZLE_64B."""
        raw = C.create_string_buffer(256)
        at = (C.addressof(raw) + 63) & ~63
        dims, strides = (u64 * 2)(k // 2, rows), (u64 * 1)(k // 2)
        box, estr = (u * 2)(64, box_rows), (u * 2)(1, 1)
        self.dev.ck("cuTensorMapEncodeTiled", vp(at), UINT8, 2, vp(ptr), dims, strides, box, estr, 0, SWIZZLE_64B, L2_128B, 0)
        return C.string_at(at, 128)

    def operands(self, m: int, n: int, k: int, scales: int, seed: int) -> dict:
        dev = self.dev
        ops = {"a": dev.alloc(m * k // 2), "b": dev.alloc(n * k // 2), "sfa": dev.alloc(m * k // 16), "sfb": dev.alloc(n * k // 16)}
        for j, (key, words) in enumerate((("a", m * k // 8), ("b", n * k // 8))):
            dev.launch("nv_gen_codes", cdiv(words, 256), 256, p=(ops[key],), i=(words, seed + j))
        for j, (key, words) in enumerate((("sfa", m * k // 64), ("sfb", n * k // 64))):
            dev.launch("nv_gen_scales", cdiv(words, 256), 256, p=(ops[key],), i=(words, seed + 2 + j, scales))
        ops["maps"] = dev.put(self.tmap(ops["a"], k, m, 256) + self.tmap(ops["b"], k, n, 128))
        dev.sync()
        return ops

    def free(self, bufs: dict) -> None:
        for p in bufs.values():
            self.dev.free(p)

    def prep(self, name: str) -> int:
        if name not in self.dev.smem:
            probe = self.dev.alloc(16)
            self.dev.launch(name + "_smem", 1, 1, p=(probe,))
            self.dev.sync()
            self.dev.smem[name] = int.from_bytes(self.dev.read(probe, 4), "little")
            self.dev.free(probe)
            self.dev.ck("cuFuncSetAttribute", self.dev.fn(name), MAX_DYN, self.dev.smem[name])
        return self.dev.smem[name]

    def gemm(self, name: str, ops: dict, c: int, m: int, n: int, k: int, grid: int | None = None, stamps: int = 0) -> None:
        """Header kernel `name` (nv256_w, nv256_o_b, nv256_t, ...): TM from the name, 8 consumer warps a 256-row tile, a
        producer warpgroup; a _t kernel's stamps to `stamps`."""
        smem = self.prep(name)
        tm = int(name[2:5])
        tiles = (m // tm) * (n // 128)
        g = min(tiles, grid or self.dev.sms)
        self.dev.launch(name, g, 32 * 2 * (tm // 64) + 128, smem, p=[ops["sfa"], ops["sfb"], c, stamps] + [0] * 6 + [ops["maps"]],
                        i=(m, n, k, self.raster))

    def ref(self, ops: dict, c: int, m: int, n: int, k: int) -> None:
        warps = (m // 16) * (n // 8)
        self.dev.launch("nv_ref_words", cdiv(warps * 32, 256), 256, p=(ops["a"], ops["b"], ops["sfa"], ops["sfb"], c), i=(m, n, k))

    def diff(self, x: int, y: int, count: int, width: int) -> int:
        counter = self.dev.alloc(8)
        self.dev.launch("nv_diff", cdiv(count, 256), 256, p=(x, y, counter), i=(count, width))
        self.dev.sync()
        bad = int.from_bytes(self.dev.read(counter, 8), "little")
        self.dev.free(counter)
        return bad


def model_gate(dev: Dev, ops: dict, ref: int, m: int, n: int, k: int, samples: int, full: int, seed: int) -> dict:
    """nv_ref_words against BLACKWELL_SM120_NVF4 on the host: every coordinate if m n <= full, else `samples` of them, with
    every row mod 16 and column mod 8."""
    try:
        import numpy as np
        from verity.ml.kernels.tc import block_scaled_batch, chain_batch
        from verity.ml.circuits.silicon.models import BLACKWELL_SM120_NVF4 as model
    except ImportError as e:
        return {"skipped": f"{type(e).__name__}: {e}"}
    t0 = time.time()
    if m * n <= full:
        rows, cols = np.divmod(np.arange(m * n, dtype=np.int64), n)
    else:
        rng = random.Random(seed)
        idx = range(samples)
        rows = np.array([16 * rng.randrange(m // 16) + i % 16 for i in idx], np.int64)
        cols = np.array([8 * rng.randrange(n // 8) + (i // 16) % 8 for i in idx], np.int64)

    def codes(p: int, rows_total: int, which: np.ndarray) -> np.ndarray:
        raw = np.frombuffer(dev.read(p, rows_total * k // 2), np.uint8).reshape(rows_total, k // 2)
        uniq, inv = np.unique(which, return_inverse=True)
        sel = raw[uniq]
        out = np.empty((len(uniq), k), np.uint8)
        out[:, 0::2], out[:, 1::2] = sel & 0x0F, sel >> 4
        return out, inv

    def scales(p: int, rows_total: int, which: np.ndarray) -> np.ndarray:
        raw = np.frombuffer(dev.read(p, rows_total * k // 16), np.uint8)
        cc = np.arange(k // 16, dtype=np.int64)
        r = which[:, None]
        off = ((r // 128) * (k // 64) + cc[None, :] // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + cc[None, :] % 4
        return raw[off]

    ac, ai = codes(ops["a"], m, rows)
    bc, bi = codes(ops["b"], n, cols)
    got = np.frombuffer(dev.read(ref, m * n * 4), np.uint32)[rows * n + cols]
    want = np.empty(len(rows), np.uint32)
    declined = np.zeros(len(rows), bool)
    chunk = 32768
    ua, ub = np.unique(rows), np.unique(cols)
    sa_rows, sb_rows = scales(ops["sfa"], m, ua), scales(ops["sfb"], n, ub)
    for lo in range(0, len(rows), chunk):
        sl = slice(lo, lo + chunk)
        a_codes, b_codes = ac[ai[sl]], bc[bi[sl]]
        a_sc, b_sc = sa_rows[np.searchsorted(ua, rows[sl])], sb_rows[np.searchsorted(ub, cols[sl])]
        w, d = chain_batch(lambda acc, x, y, sx, sy: block_scaled_batch(model, acc, x, y, sx, sy), model.k, k // model.k,
                           a_codes, b_codes, a_sc, b_sc, per_step=(model.k, model.k, model.scales_per_step, model.scales_per_step))
        want[sl], declined[sl] = w, d
    bad = np.nonzero((got != want) & ~declined)[0]
    return {"model": model.name, "coordinates": int(len(rows)), "full": bool(m * n <= full), "declined": int(declined.sum()),
            "words_differing": int(len(bad)), "first": [[int(rows[i]), int(cols[i]), f"{int(got[i]):08x}", f"{int(want[i]):08x}"]
                                                         for i in bad[:8]], "seconds": round(time.time() - t0, 2)}


def registry(bench: Bench, harness: Harness | None, plains: list[Harness]) -> list[tuple[str, Harness, bool]]:
    """(name, library, plain) of each registry baseline: HARNESS_LIB's kernels named in CUTLASS (default bench.CUTLASS), then
    each PLAIN_LIB entry of the bench's kind (those PLAIN_ENTRIES names, if set)."""
    out = [(nm, harness, False) for nm in os.environ.get("CUTLASS", bench.CUTLASS).split()] if harness else []
    only = os.environ.get("PLAIN_ENTRIES", "").split()
    return out + [(nm, p, True) for p in plains for nm in sorted(p.index) if not only or nm in only]


def operand_seed(m: int, n: int, k: int, scales: int) -> int:
    return 1000 * scales + (m ^ (n << 1) ^ (k << 2)) % 997


def gates(bench: Bench, harness: Harness | None, plains: list[Harness], m: int, n: int, k: int, scales: int, kernels: list[str],
          repeats: int, model_args: tuple[int, int] | None) -> tuple[dict, dict | None]:
    """Every gate at one shape and scale range, and the operands (for timing)."""
    dev = bench.dev
    t0 = time.time()
    ops = bench.operands(m, n, k, scales, seed=operand_seed(m, n, k, scales))
    ref, got, ref16, d16 = dev.alloc(m * n * 4), dev.alloc(m * n * 4), dev.alloc(m * n * 2), dev.alloc(m * n * 2)
    bench.ref(ops, ref, m, n, k)
    dev.launch("nv_to_bf16", cdiv(m * n // 2, 256), 256, p=(ref, ref16), i=(m * n,))
    dev.sync()
    out: dict = {"shape": [m, n, k], "scales": scales}
    if model_args:
        out["model"] = bench.model_gate(ops, ref, m, n, k, *model_args, seed=m + n + k + scales)
        out["model_ok"] = bench.model_ok(out["model"])
    words: dict = {}
    for name in kernels:
        rec = {}
        for label, grid, reps in (("full", None, 1), ("1", 1, 1), ("3", 3, 1), ("repeats", None, repeats)):
            worst = 0
            for _ in range(reps):
                dev.fill(got, 0xAB, m * n * 4)
                bench.gemm(name + "_w", ops, got, m, n, k, grid=grid)
                dev.sync()
                worst = max(worst, bench.diff(got, ref, m * n, 4))
            rec[label] = worst
        words[name] = rec
    out["words"] = words
    bf16: dict = {}
    others: dict = {}
    lts = dict(bench.lts)
    for name, lib, plain in registry(bench, harness, plains) + [(nm, lib, False) for nm, lib in bench.lts]:
        buf = dev.alloc(m * n * 2)
        try:
            run, release, rec = lib.lt(dev, m, n, k, ops, buf) if name in lts else lib.cutlass(dev, name, m, n, k, ops, buf)
            worst = 0
            for _ in range(1 + (repeats if plain else 0)):
                dev.fill(buf, 0xAB, m * n * 2)
                run()
                dev.sync()
                worst = max(worst, bench.diff(buf, ref16, m * n, 2))
            release()
            others[name] = {"buf": buf, "vs_ref": worst, **({"lt": rec} if name in lts else {})}
        except (RuntimeError, KeyError) as e:
            dev.free(buf)
            others[name] = {"error": str(e)}
    for name in kernels:
        grids = {}
        for label, grid, reps in (("full", None, 1), ("1", 1, 1), ("3", 3, 1), ("repeats", None, repeats)):
            worst = 0
            for _ in range(reps):
                dev.fill(d16, 0xAB, m * n * 2)
                bench.gemm(name + "_b", ops, d16, m, n, k, grid=grid)
                dev.sync()
                worst = max(worst, bench.diff(d16, ref16, m * n, 2))
            grids[label] = worst
        dev.fill(d16, 0xAB, m * n * 2)
        bench.gemm(name + "_b", ops, d16, m, n, k)
        dev.sync()
        rec = {"vs_ref": max(max(grids.values()), bench.diff(d16, ref16, m * n, 2)), "grids": grids}
        for o, v in others.items():
            if "buf" in v:
                rec["vs_" + o] = bench.diff(d16, v["buf"], m * n, 2)
        bf16[name] = rec
    out["bf16"] = bf16
    out["baselines"] = {o: {kk: vv for kk, vv in v.items() if kk != "buf"} for o, v in others.items()}
    for v in others.values():
        if "buf" in v:
            dev.free(v["buf"])
    for p in (ref, got, ref16, d16):
        dev.free(p)
    out["seconds"] = round(time.time() - t0, 2)
    return out, ops


def passed(g: dict, name: str) -> bool:
    """Every word equals the reference's, and every BF16 element the reference's rounded and each baseline's that is itself
    the reference's (a baseline that isn't is reported, not gated on: the model is the oracle)."""
    exact = [o for o, v in g["baselines"].items() if v.get("vs_ref") == 0]
    rec = g["bf16"][name]
    return (g.get("model_ok", True) and all(v == 0 for v in g["words"][name].values()) and rec["vs_ref"] == 0
            and all(rec["vs_" + o] == 0 for o in exact))


def timing(bench: Bench, harness: Harness | None, plains: list[Harness], nv: Nvml, m: int, n: int, k: int, ops: dict,
           kernels: list[str], plain_ok: set[str], reps: int) -> tuple[dict, dict, list]:
    dev = bench.dev
    d = dev.alloc(m * n * 2)
    bodies, info, releases = {}, {}, []
    for name, lib, plain in registry(bench, harness, plains):
        if plain and name not in plain_ok:
            continue
        try:
            run, release, rec = lib.cutlass(dev, name, m, n, k, ops, d)
            bodies[name], info[name] = run, rec
            releases.append(release)
        except (RuntimeError, KeyError) as e:
            info[name] = {"error": str(e)}
    for name, lib in bench.lts:
        try:
            run, release, rec = lib.lt(dev, m, n, k, ops, d)
            bodies[name], info[name] = run, rec
            releases.append(release)
        except RuntimeError as e:
            info[name] = {"error": str(e)}
    for name in kernels:
        bodies[name + "_b"] = lambda name=name: bench.gemm(name + "_b", ops, d, m, n, k)
    for body in bodies.values():
        body()
    dev.sync()
    before = nv.read()
    runs = {nm: [] for nm in bodies}
    for _ in range(reps):
        for nm, body in bodies.items():
            runs[nm].append(dev.timed(body))
    after = nv.read()
    med = {nm: statistics.median(t) for nm, t in runs.items()}
    row = {"kind": "timing", "label": "Measured diagnostic (one GPU, events around one launch, round-robin, clocks as recorded)",
           "shape": [m, n, k], "reps": reps, "raster": bench.raster, "baselines": info,
           "ms": {nm: round(v, 5) for nm, v in med.items()}, "tflops": {nm: round(2 * m * n * k / v / 1e9, 1) for nm, v in med.items()},
           "runs": {nm: [round(x, 5) for x in t] for nm, t in runs.items()}, "clk_before": before, "clk_after": after}
    return row, bodies, releases + [lambda: dev.free(d)]


def steady(dev: Dev, nv: Nvml, m: int, n: int, k: int, bodies: dict, rounds: int, batch: int) -> dict:
    rows = {nm: [] for nm in bodies}
    for _ in range(rounds):
        for nm, body in bodies.items():
            a, b = dev.event(), dev.event()
            dev.ck("cuEventRecord", a, None)
            for _ in range(batch):
                body()
            dev.ck("cuEventRecord", b, None)
            samples = []
            while dev.cu.cuEventQuery(b) == 600:  # CUDA_ERROR_NOT_READY
                samples.append(nv.read())
                time.sleep(0.01)
            dev.ck("cuEventSynchronize", b)
            ms = C.c_float()
            dev.ck("cuEventElapsedTime", C.byref(ms), a, b)
            dev.cu.cuEventDestroy_v2(a)
            dev.cu.cuEventDestroy_v2(b)
            late = [s for s in samples[len(samples) // 2:] if "sm_mhz" in s] or [{"sm_mhz": 0, "power_w": 0, "throttle": "?"}]
            mhz, per = statistics.mean(s["sm_mhz"] for s in late), ms.value / batch
            rows[nm].append({"ms": round(per, 5), "sm_mhz": round(mhz, 1), "power_w": round(statistics.mean(s["power_w"] for s in late), 1),
                             "throttle": sorted({s["throttle"] for s in late}), "samples": len(late), "mcycles": round(per * mhz / 1e3, 4)})
    summary = {nm: {key: round(statistics.median(r[key] for r in rs), 4) for key in ("ms", "sm_mhz", "power_w", "mcycles")}
               for nm, rs in rows.items()}
    return {"kind": "steady", "label": "Measured diagnostic (one GPU, batches back to back, round-robin, NVML in each batch's second half)",
            "shape": [m, n, k], "rounds": rounds, "batch": batch, "summary": summary, "rows": rows}


def stamps(bench: Bench, m: int, n: int, k: int, ops: dict, kernels: list[str], out_dir: str) -> dict:
    """Each kernel's _t once at (m, n, k), its D checked against _b's: per consumer warp and tile, SM cycles in the mainloop
    and in the epilogue, the lag of the second 128 rows' warp behind the first's on the same SMSP (warps w and w + 4) at
    the mainloop's end, the cycles both of them spend in their epilogues at once, and the spread of the CTAs' last
    epilogue ends (globaltimer, ns).  Raw stamps to out_dir/<kernel>.stamps.npy."""
    import numpy as np
    dev, T, ncw = bench.dev, 16, 8
    tiles = (m // 256) * (n // 128)
    grid = min(tiles, dev.sms)
    size = grid * ncw * T * 4 * 8
    buf, d1, d2 = dev.alloc(size), dev.alloc(m * n * 2), dev.alloc(m * n * 2)
    per = np.array([len(range(c, tiles, grid)) for c in range(grid)])
    valid = np.arange(T)[None, :] < per[:, None]
    q = lambda x: [round(float(v), 1) for v in np.percentile(x, [10, 50, 90])] if len(x) else []
    out = {}
    for nm in kernels:
        dev.fill(buf, 0, size)
        bench.gemm(nm + "_b", ops, d1, m, n, k)
        bench.gemm(nm + "_t", ops, d2, m, n, k, stamps=buf)
        dev.sync()
        raw = np.frombuffer(dev.read(buf, size), np.uint64).reshape(grid, ncw, T, 4).astype(np.int64)
        if out_dir:
            np.save(os.path.join(out_dir, f"{nm}.stamps.npy"), raw)
        v = np.broadcast_to(valid[:, None, :], raw.shape[:3])
        later = v & (np.arange(T)[None, None, :] > 0)
        main, epi = raw[..., 1] - raw[..., 0], raw[..., 2] - raw[..., 1]
        lo, hi = raw[:, :4], raw[:, 4:]
        both = np.maximum(0, np.minimum(lo[..., 2], hi[..., 2]) - np.maximum(lo[..., 1], hi[..., 1]))
        pv, pl = valid[:, None, :].repeat(4, 1), later[:, :4]
        ends = raw[np.arange(grid), :, per - 1, 3].max(axis=1)
        out[nm] = {"same_as_b": bench.diff(d1, d2, m * n, 2), "tiles": [int(per.min()), int(per.max())],
                   "main_cycles_p10_50_90": q(main[later]), "first_main_cycles": q(main[:, :, 0].ravel()),
                   "epilogue_cycles": q(epi[v]), "lag_cycles": q((hi[..., 1] - lo[..., 1])[pl]),
                   "both_in_epilogue_cycles": q(both[pv]),
                   "tile_period_cycles": q((raw[:, :, 1:, 0] - raw[:, :, :-1, 0])[later[:, :, 1:]]),
                   "cta_end_spread_ns": int(ends.max() - ends.min()),
                   "cta_end_ns_by_tiles": {int(t): q(ends[per == t] - ends.min()) for t in np.unique(per)}}
    for p in (buf, d1, d2):
        dev.free(p)
    return {"kind": "stamps", "label": "Measured diagnostic (one launch each, SM clock64 per warp)", "shape": [m, n, k],
            "kernels": out}


def plan() -> tuple[list[int], list[tuple[int, int, int]], list[int]]:
    """SHAPE's sizes, every gated shape (SMALL's, then SHAPE's cubes) and the scale ranges."""
    sizes = [int(x) for x in os.environ.get("SHAPE", "8192").split()]
    shapes = [tuple(int(x) for x in t.split("x")) for t in os.environ.get("SMALL", "256x128x128 512x256x1024 768x384x640").split()]
    return sizes, shapes + [(s, s, s) for s in sizes], [int(x) for x in os.environ.get("SCALES", "0 1").split()]


def main(argv: list[str], bench_cls: type = Bench) -> int:
    image = open(argv[1], "rb").read()
    dev = Dev(image)
    expect = os.environ.get(bench_cls.UUID_ENV)
    refused = dev.identity(expect, bench_cls.UUID_ENV)
    nv = Nvml(dev.uuid)
    emit({"kind": "env", "device": dev.name, "uuid": dev.uuid, "expect_uuid": expect, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "count": dev.count, "cc": list(dev.cc), "sms": dev.sms, "cuda_driver_api": dev.driver, "nvml_driver": nv.driver,
          "cubin_sha256": hashlib.sha256(image).hexdigest(), "clk": nv.read(), "refused": refused,
          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if refused:
        print("REFUSED: " + "; ".join(refused), file=sys.stderr)
        return 3
    dev.open()
    bench = bench_cls(dev)
    kind = bench_cls.KIND
    harness = Harness(os.environ["HARNESS_LIB"], kind) if os.environ.get("HARNESS_LIB") else None
    lt2 = Harness(os.environ["LT2_LIB"], kind) if os.environ.get("LT2_LIB") else None
    plains = [Harness(p, kind) for p in os.environ.get("PLAIN_LIB", "").split()]
    bench.lts = ([("lt", harness)] if harness and os.environ.get("LT", "1") == "1" and hasattr(harness.lib, "ph_lt_plan_create")
                 else []) + ([("lt2", lt2)] if lt2 else [])
    for h, role in ([(harness, "harness")] if harness else []) + ([(lt2, "lt2")] if lt2 else []) + [(p, "plain") for p in plains]:
        emit({"kind": role, "lib_sha256": h.sha256, "harness_kind": kind, "kernels": sorted(h.index)})
    plain_all = {nm for nm, _, plain in registry(bench, None, plains) if plain}
    plain_ok = set(plain_all)
    kernels = os.environ.get("KERNELS", bench_cls.KERNELS).split()
    sizes, shapes, ranges = plan()
    time_scales = int(os.environ.get("TIME_SCALES", "1"))
    repeats = int(os.environ.get("GATE_REPEATS", "3"))
    model_args = (int(os.environ.get("MODEL_SAMPLES", "4096")), int(os.environ.get("FULL_MODEL", "131072")))
    ok = {nm: True for nm in kernels}
    refs_ok, keep = True, {}
    for sc in ranges:
        for (m, n, k) in shapes:
            big = m == n == k and m in sizes
            g, ops = gates(bench, harness, plains, m, n, k, sc, kernels, repeats if big else 1, model_args)
            emit({"kind": "gates", **g})
            if not g.get("model_ok", True):
                refs_ok = False
            for nm in kernels:
                ok[nm] &= passed(g, nm)
            plain_ok = {nm for nm in plain_ok if g["baselines"].get(nm, {}).get("vs_ref") == 0}
            print(f"gates {m}x{n}x{k} scales {sc}: model {g.get('model', {}).get('words_differing', 'skipped')}, "
                  f"words {g['words']}, bf16 {g['bf16']}, baselines {g['baselines']}", file=sys.stderr)
            if big and sc == time_scales:
                keep[m] = ops
            else:
                bench.free(ops)
    gated = [nm for nm in kernels if ok[nm] and refs_ok]
    plain_ok = plain_ok if refs_ok else set()
    all_ok = len(gated) == len(kernels) and plain_ok == plain_all
    emit({"kind": "gated", "passed": gated, "plain_passed": sorted(plain_ok), "refs_ok": refs_ok})
    print(f"passed: {gated}, plain {sorted(plain_ok)} (refs_ok {refs_ok})", file=sys.stderr)
    if os.environ.get("ONLY_CHECK") or not keep:
        return 0 if all_ok else 4
    for s, ops in keep.items():
        row, bodies, releases = timing(bench, harness, plains, nv, s, s, s, ops, gated, plain_ok, int(os.environ.get("REPS", "20")))
        row["gated"] = gated
        emit(row)
        print(f"timing {s}^3 (Measured diagnostic): "
              + ", ".join(f"{nm} {row['ms'][nm]:.4f} ms {row['tflops'][nm]} TFLOPS" for nm in row["ms"]), file=sys.stderr)
        if os.environ.get("STEADY"):
            st = steady(dev, nv, s, s, s, bodies, int(os.environ.get("STEADY_ROUNDS", "4")), int(os.environ["STEADY"]))
            emit(st)
            print(f"steady {s}^3 (Measured diagnostic): {st['summary']}", file=sys.stderr)
        if os.environ.get("STAMPS") and gated:
            st = stamps(bench, s, s, s, ops, gated, os.environ["STAMPS"])
            emit(st)
            print(f"stamps {s}^3 (Measured diagnostic): {json.dumps(st['kernels'])}", file=sys.stderr)
        if os.environ.get("PROFILE"):   # one launch of each, between cuProfilerStart and Stop (ncu --profile-from-start off)
            dev.ck("cuProfilerStart")
            for nm, body in bodies.items():
                body()
                dev.sync()
            dev.ck("cuProfilerStop")
            emit({"kind": "profiled", "shape": [s, s, s], "kernels": list(bodies)})
        for r in releases:
            r()
        bench.free(ops)
    return 0 if all_ok else 4


if __name__ == "__main__":
    sys.exit(main(sys.argv))
