#!/usr/bin/env python3
"""Pearl-C on one H100 or H200 (sm_90a) from the prebuilt pearl_c_sm90.cubin, through the CUDA driver API (ctypes on
libcuda.so.1, stdlib only, nothing built on the card).

0. Identity, before anything runs: exactly one visible device (CUDA_VISIBLE_DEVICES a single index), compute capability
   9.0, an "NVIDIA H100" or "NVIDIA H200" name, and its UUID equal to PEARLC_EXPECT_UUID.  NVML is addressed by that UUID,
   never by index (NVML ignores CUDA_VISIBLE_DEVICES).  A mismatch exits 3 before any launch.
1. The check.  At check.json's shape (fixture.py's), every kernel runs on the fixture's inputs and every buffer it writes is
   compared byte for byte with the fixture, for the variant `h100` (pearl-c-h100-v1-h3: G = 4, hashing format -h3, SHA-512
   throughout); the GEMMs, with gemm_pearlc_s (y alone), are re-run on all, 1 and 3 CTAs and compared again
   (`<buffer>@<step>/<tile>/grid<n>`; after af, pa and its P_A too: at bm64, WORDS_WIN's windows and pa_w), at each of the
   three tiles (bm128: 128 x 128, n64: 128 x 64, bm64: 64 x 64), and
   at 64 x 64 on LIVE_ROWS real rows (`check_live`): the real rows the fixture's, the rest as the memset left them.
   hash_decode.cuh's served launches (whole, split, and the whole's stamp twins: `check_pd`) write every output they own
   over a poison (0xA5) and must match; the no-write control must not.  So do A's commit launches past them under each chain (`check_chains`: "pl", hash_prefill.cuh's, and "h3"),
   and over each of CHAIN_WIDE_ROWS' rows (two-byte ranks and indices, prefill's M) "pl" must write "h3"'s bytes.  A
   variant is timed only if all of its buffers match (C~ and U bit-exact against the CPU reference), the plain baselines
   only if theirs do.  Exit 4 if any check fails.
2. Diagnostic timing (not the headline: the stock FP8 block GEMM is timed beside the call by its own job), at each of
   SHAPES (default 8192, m = n = k; or MxNxK, m padded to 64 rows): each kernel REPS times after a warm-up, its median, the
   phases (the weight side; the call's commitment of A, forming, GEMM, clean-up and hashing), the call end to end with and
   without its hashes (commit and hashing; without, the chain is gemm_pearlc_s), and consistency checks (C~ of gemm_ct
   equals gemm_pearlc's; a second gemm_pearlc gives the same U; gemm_pearlc_s's y equals gemm_pearlc's; form_s5_scalar,
   timed beside form_s5, writes form_s5's codes).  Every operand is in natural order along k (pearl_c_sm90.cu).  The
   timing's keys and domain ids are stand-ins of the same sizes (diagnostic only; the arm's calls run on a real epoch).
   NVML's clocks, power and throttle reasons before and after.

One JSON object per line on stdout as each part finishes, a summary on stderr.

  python3 run.py [check.json]   (env: PEARLC_EXPECT_UUID, SHAPES="8192", REPS=5, VARIANTS="h100", PEARLC_CHAIN="pl",
                                 PEARLC_LIBCUDA, PEARLC_LIBNVML, PEARLC_LIVE=0 for decode's 64 x 64 GEMMs storing all m rows,
                                 PEARLC_WORDS_WIN=0 for A'F_B on one 64 x 32 CTA and pa at 64-row tiles,
                                 PEARLC_FORM_EP=0 for A's E' in a launch of its own, PEARLC_WORDS_FORM=0 for A'F_B's
                                 windows in one of their own, PEARLC_PDL=0 for PDL_KERNELS launched plainly,
                                 PEARLC_GRID_BALANCE=0 for persistent GEMMs on every SM they may take,
                                 PEARLC_ONE_COMMIT=0 for pd_commit and pd_tree in the shape's pd variant)

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
NAME_PREFIXES = ("NVIDIA H100", "NVIDIA H200")
SM_COUNT, MAX_DYN, CC_MAJOR, CC_MINOR, MAX_SHARED_OPTIN, SHARED_SIZE = 16, 8, 75, 76, 97, 1
NORM_256, NORM_16 = 0x4600, 0x4400      # bf16(32 x the line norm): E's lines, and F's under v1
BM, BM_D, BN_N64, WORDS_N, RASTER = 128, 64, 64, 32, 8
PEEL_ROW = 128                  # the peel factors' bytes a row (64 BF16 words)
VARIANT_G = {"h100": 4}
#: the 64 x 64 GEMMs store the real rows alone (i4 = m_real: a filler row's C~, U and y are left as the buffer holds them);
#: PEARLC_LIVE=0 stores all m rows, the same real rows
LIVE = os.environ.get("PEARLC_LIVE", "1") != "0"
#: a promotion window's depth (pearl_c_sm90.cu's BK)
WINDOW = 128
#: with 64-row tiles and an af_win buffer, A'F_B is a CTA a window (gemm_words_win_d), which pa_w adds up in gemm_words_d's
#: order; PEARLC_WORDS_WIN=0 runs gemm_words_d and pa
WORDS_WIN = os.environ.get("PEARLC_WORDS_WIN", "1") != "0"
#: A's E' in its forming's launch (form_s5_ep, the step `form_ep_a`); PEARLC_FORM_EP=0 runs eprime and form_s5
FORM_EP = os.environ.get("PEARLC_FORM_EP", "1") != "0"
#: under FORM_EP and WORDS_WIN, A'F_B's windows in that launch too (form_s5_epw, the step `form_epw_a`), from a cubin with it;
#: PEARLC_WORDS_FORM=0 runs gemm_words_win_d
WORDS_FORM = os.environ.get("PEARLC_WORDS_FORM", "1") != "0"
#: a persistent GEMM takes the fewest CTAs that keep its rounds of tiles (decode's gate_up: 448 tiles, four rounds on 112 CTAs
#: as on 131), leaving the rest of the SMs to the side lanes; a tile's bytes don't depend on its CTA.  PEARLC_GRID_BALANCE=0
#: takes every SM it may
GRID_BALANCE = os.environ.get("PEARLC_GRID_BALANCE", "1") != "0"
#: pd_commit and pd_tree launch a shape's variant without PD_GROUP, which only pd_tiles reads (the same code and bytes), so
#: every call's A commitment runs one function, its code warm (pearl_c_sm120's run.py makes the same cut).
#: PEARLC_ONE_COMMIT=0 launches the shape's variant
ONE_COMMIT = os.environ.get("PEARLC_ONE_COMMIT", "1") != "0"
#: PEARLC_PDL=0 launches PDL_KERNELS and PDL_ENTRY_KERNELS (below PD_SERVED) plainly
PDL = os.environ.get("PEARLC_PDL", "1") != "0"
PROGRAMMATIC_STREAM_SERIALIZATION, PDL_MARK, PDL_ENTRY_MARK = 6, "pdl_waits", "pdl_waits_entry"
#: a split pipeline's pd_commit may read a call's BF16 rows in its FP32 words' place (`Pipeline.reads_bf16`), from a cubin
#: with BF16_ROWS_MARK, so the call's widen_a can run beside it; PEARLC_BF16_ROWS=0 says no
BF16_ROWS = os.environ.get("PEARLC_BF16_ROWS", "1") != "0"
BF16_ROWS_MARK = "pd_reads_bf16"
TMAP_UINT8, TMAP_SWIZZLE_128B, TMAP_L2_128B, TMAP_BYTES = 0, 3, 2, 128
#: every GEMM's operand pair, a set of four tensor maps (A, B, P_A, P_B) each at `tma` + MAP_SET * slot: the chain's (P_A, P_B
#: the peel's), the words' (A'F_B, B~F_A, G) and the baselines' (A and B again for the tail, which they don't load)
MAPS = {("at", "bt"): 0, ("at", "fbT32"): 1, ("bt", "faT32"): 2, ("fbT128", "faT32"): 3, ("qa", "qb"): 4, ("qa_s5", "qb_s5"): 5}
MAP_SET = 4 * TMAP_BYTES

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER


class LaunchAttr(C.Structure):
    """cuda.h's CUlaunchAttribute with an int value (CUlaunchAttributeValue is a 64-byte union)."""
    _fields_ = [("id", i32), ("pad", C.c_char * 4), ("value", i32), ("rest", C.c_char * 60)]


class LaunchConfig(C.Structure):
    """cuda.h's CUlaunchConfig."""
    _fields_ = [("grid", u * 3), ("block", u * 3), ("smem", u), ("stream", vp), ("attrs", P(LaunchAttr)), ("n_attrs", u)]


CUDA = {
    "cuInit": [u], "cuDriverGetVersion": [P(i32)], "cuDeviceGetCount": [P(i32)], "cuDeviceGet": [P(i32), i32],
    "cuDeviceGetName": [C.c_char_p, i32, i32], "cuDeviceGetUuid_v2": [C.c_char_p, i32],
    "cuDeviceGetAttribute": [P(i32), i32, i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuFuncSetAttribute": [vp, i32, i32], "cuFuncGetAttribute": [P(i32), i32, vp], "cuMemAlloc_v2": [P(u64), sz],
    "cuMemFree_v2": [u64],
    "cuMemsetD8_v2": [u64, C.c_ubyte, sz], "cuMemcpyHtoD_v2": [u64, vp, sz], "cuMemcpyDtoH_v2": [vp, u64, sz],
    "cuMemcpyDtoD_v2": [u64, u64, sz], "cuMemcpyDtoDAsync_v2": [u64, u64, sz, vp], "cuMemGetInfo_v2": [P(sz), P(sz)],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuLaunchKernelEx": [P(LaunchConfig), vp, P(vp), P(vp)],
    "cuCtxSynchronize": [], "cuEventCreate": [P(vp), u], "cuEventRecord": [vp, vp], "cuEventSynchronize": [vp],
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
        self.stream = None               # where launch() and copy_async() enqueue unless told otherwise (a harness's stream)
        self.pdl = False                 # PDL_KERNELS launched programmatically (`open`: PDL, and the cubin has PDL_MARK)
        self.pdl_names = frozenset()     # the launches that are: PDL_KERNELS, and PDL_ENTRY_KERNELS with PDL_ENTRY_MARK

    def identity(self, expect_uuid: str | None) -> list[str]:
        """What stops the run before any context or launch: the reasons, empty if this is the expected device."""
        bad = []
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.isdigit():
            bad.append(f"CUDA_VISIBLE_DEVICES={visible!r} is not a single index")
        if self.count != 1:
            bad.append(f"{self.count} devices visible, not 1")
        if self.cc != (9, 0):
            bad.append(f"compute capability {self.cc}, not (9, 0)")
        if not self.name.startswith(NAME_PREFIXES):
            bad.append(f"device {self.name!r} is not one of {NAME_PREFIXES}")
        if not expect_uuid:
            bad.append("PEARLC_EXPECT_UUID is not set (the node's UUID for this index)")
        elif self.uuid != expect_uuid:
            bad.append(f"UUID {self.uuid} is not the expected {expect_uuid}")
        return bad

    def open(self) -> None:
        ctx = vp()
        self.ck("cuDevicePrimaryCtxRetain", C.byref(ctx), self.dev)
        self.ck("cuCtxSetCurrent", ctx)
        self.mod = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod), self.cubin)
        mark = vp()
        self.pdl = PDL and self.cu.cuModuleGetFunction(C.byref(mark), self.mod, PDL_MARK.encode()) == 0
        entry = self.pdl and self.cu.cuModuleGetFunction(C.byref(mark), self.mod, PDL_ENTRY_MARK.encode()) == 0
        self.pdl_names = (PDL_KERNELS | PDL_ENTRY_KERNELS if entry else PDL_KERNELS) if self.pdl else frozenset()

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

    def has(self, name: str) -> bool:
        """Whether the cubin has the kernel `name` (an older cubin may not)."""
        f = vp()
        return name in self.fns or self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()) == 0

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

    def copy_async(self, dst: int, src: int, n: int, stream=None) -> None:
        self.ck("cuMemcpyDtoDAsync_v2", u64(dst), u64(src), n, self.stream if stream is None else stream)

    def launch(self, name: str, grid, block: int, smem: int = 0, p=(), i=(), stream=None, pdl: bool = True) -> None:
        """`name` on `stream` (else `stream`'s attribute); one of `pdl_names` programmatically unless `pdl` is False."""
        a = Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        g = tuple(grid) + (1,) * (3 - len(grid)) if isinstance(grid, (tuple, list)) else (grid, 1, 1)
        params, s = (vp * 1)(C.addressof(a)), self.stream if stream is None else stream
        if pdl and name in self.pdl_names:
            attr = LaunchAttr(id=PROGRAMMATIC_STREAM_SERIALIZATION, value=1)
            cfg = LaunchConfig((u * 3)(*g), (u * 3)(block, 1, 1), smem, s.value if isinstance(s, vp) else s, C.pointer(attr), 1)
            self.ck("cuLaunchKernelEx", C.byref(cfg), self.fn(name), params, None)
        else:
            self.ck("cuLaunchKernel", self.fn(name), g[0], g[1], g[2], block, 1, 1, smem, s, params, None)

    def sync(self) -> None:
        self.ck("cuCtxSynchronize")

    def tmap(self, ptr: int, depth: int, rows: int, box_rows: int = BM) -> bytes:
        """A CUtensorMap over a row-major (rows x depth) byte matrix as pearl_c_sm90.cu's GEMMs load it: UINT8, rank 2,
        boxes of box_rows x 128 bytes, SWIZZLE_128B, L2 promotion 128B, no fill."""
        f = self.cu.cuTensorMapEncodeTiled
        f.argtypes, f.restype = [vp, i32, u, vp, P(u64), P(u64), P(u), P(u), i32, i32, i32, i32], i32
        raw = C.create_string_buffer(TMAP_BYTES + 64)
        at = (C.addressof(raw) + 63) & ~63
        dims, strides = (u64 * 2)(depth, rows), (u64 * 1)(depth)
        box, estr = (u * 2)(TMAP_BYTES, box_rows), (u * 2)(1, 1)
        self.ck("cuTensorMapEncodeTiled", vp(at), TMAP_UINT8, 2, vp(ptr), dims, strides, box, estr, 0, TMAP_SWIZZLE_128B,
                TMAP_L2_128B, 0)
        return C.string_at(at, TMAP_BYTES)

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


TILE = 64                                        # the leaf's tile
#: a tree node's bytes: SHA-512's
NB = 64
#: s512_rows' most segments a row (hash_h3.cuh's RS_MAX_SEGS), 1 KiB each
RS_MAX_SEGS = 128
#: hash_fast.cuh's blocks: the segment kernels' (RS_FAST_NT, as built) and s512_folds' (RS_FOLD_NT)
RS_FAST_NT, RS_FOLD_NT = int(os.environ.get("PEARLC_RS_FAST_NT", "128")), 32
#: their scratch, the segment digests of A's rows, B's and the tiles'
SEGDIG = ("segdig_a", "segdig_b", "segdig_t")
#: the hm96 salt nonces (`audit.HM96_WEIGHTS`, `audit.HM96_TRANSCRIPT`; an activation tree's is its unit index) and the widest
#: level s512_tree_small (64 threads) takes
HM96_WEIGHTS, HM96_TRANSCRIPT, S512_SMALL = 1 << 62, 1 << 63, 128


#: hash_decode.cuh's launches at decode, the variants pearl_c_sm90.cu builds (its PD_VARIANT_LIST): pd_commit, A's commitment
#: in one launch, a CTA a real row (A's pads; each row's digest, hm96 leaf, seed, line key and E_A line; then on the last CTA
#: to finish A's tree, seed_A, line_a and the filler rows' E_A lines), and pd_tiles, the tile leaves.  pd_commit takes at most
#: PD_MAX_ROWS rows and k of at most PD_MAX_K (its 192 threads: two warps of pairs a 32 segments, and two more); its paired
#: compressions take PD_PAIR_SMEM bytes of dynamic shared memory, and PD_GROUP's tiles PD_G a CTA (hash_decode.cuh's)
PD_SERVED, PD_MAX_ROWS, PD_MAX_K, PD_PAIR_SMEM, PD_G, PD_WHOLE, PD_GROUP = (851, 979), 256, 64 * 256, 2 * 8 * 3080, 4, 64, 128
#: each served variant's PD_STAMP twin (the same code, its phases' global timer and SM clock into p13, `Pipeline.stamps`),
#: and the stamps' layout: PD_SLOTS (timer, clock) pairs of u64 a row, a row a pd_commit CTA and the tail's after them
PD_STAMP, PD_SLOTS = 32, 16
PD_STAMPED = tuple(v | PD_STAMP for v in PD_SERVED)
#: the main lane's launches that may start behind the launch before them on their stream (programmatic stream serialization,
#: cuLaunchKernelEx), each kernel waiting at its entry for that launch to complete (pearl_c_sm90.cu's pdl_entry, and
#: hash_decode.cuh's pd_commit under PD_PDL), so they read and write what they did (gemm_pearlc_d's producer waits only
#: before P_A, which alone of what it reads its launch before, pa_w or pa, writes): only from a cubin whose kernels wait (its
#: kernel PDL_MARK, `Dev.pdl`).  PDL_ENTRY_KERNELS, a call's widen_a behind the model's launch before it and its pd_commit
#: behind widen_a (each resident when the one before completes), only from a cubin that also has PDL_ENTRY_MARK, since an
#: older cubin's widen_a and pd_commit don't wait
PDL_KERNELS = frozenset({"form_s5", "form_s5_ep", "form_s5_epw", "gemm_words_win_d", "pa_w", "gemm_pearlc_d"})
PDL_ENTRY_KERNELS = frozenset({"widen_a", *(f"pd_commit_{v}" for v in (*PD_SERVED, *PD_STAMPED))})
PD_STAMP_BYTES = 16 * PD_SLOTS * (PD_MAX_ROWS + 1)
#: hash_prefill.cuh's launches past decode's rows (`Pipeline`'s chain "pl"), in the variant pearl_c_sm90.cu builds them
#: (PL_INSTANCE): pl_leaves, A's level 0 at PL_ROWS rows a CTA; pl_levels, PL_LEVELS levels a launch; pl_seed_a; pl_seeds,
#: PL_SEEDS rows a CTA.  Ranks and indices below PL_MAX_ROWS; "h3" is hash_h3.cuh's launches (hm_leaves, s512_frame_leaves,
#: s512_nodes, s512_tree_small, s512_seeds)
PL_VARIANT, PL_ROWS, PL_LEVELS, PL_SEEDS, PL_MAX_ROWS, CHAINS = 851, 16, 7, 32, 1 << 16, ("pl", "h3")


def pd_variant(m_real: int, n: int, k: int) -> int | None:
    """The pd variant a decode shape takes: 979 (PD_GROUP's tiles, four a CTA) where the tile grid is wide (n at least 16,384)
    or every row block is full, else 851, the fastest at Llama-3.1-8B's linears at m16 = 32 and 128; None outside
    pd_commit's rows and k."""
    if m_real > PD_MAX_ROWS or k > PD_MAX_K:
        return None
    return 979 if n >= 16384 or m_real % TILE == 0 else 851


def pd_setup(dev) -> None:
    """Once per module: hash_decode.cuh's label states (pd_heads) and hash_fast.cuh's header tables (s512_heads), the
    dynamic shared memory of each served pd_commit and pd_tree and of their stamp twins (`pd_smem`, `pd_tree_smem`), with
    PD_WHOLE all a CTA may take beside its static, so no side lane's CTA shares its SMs, and pl_levels' (its two pairs')."""
    if getattr(dev, "pd_smem", None):
        return
    if not getattr(dev, "rs_heads", False):
        dev.launch("s512_heads", cdiv(RS_MAX_SEGS * (RS_MAX_SEGS + 1) // 2 + RS_MAX_SEGS + 1, 128), 128)
        dev.rs_heads = True
    dev.launch("pd_heads", 1, 32)
    optin, smem = dev.get("cuDeviceGetAttribute", MAX_SHARED_OPTIN, dev.dev), {"pd_commit": {}, "pd_tree": {}}
    for v in (*PD_SERVED, *PD_STAMPED):
        for f in smem:
            fn = dev.fn(f"{f}_{v}")
            smem[f][v] = optin - dev.get("cuFuncGetAttribute", SHARED_SIZE, fn) if v & PD_WHOLE else PD_PAIR_SMEM
            if smem[f][v] < PD_PAIR_SMEM:
                raise RuntimeError(f"{f}_{v} gets {smem[f][v]} bytes of dynamic shared memory, not its {PD_PAIR_SMEM}")
            dev.ck("cuFuncSetAttribute", fn, MAX_DYN, smem[f][v])
    dev.ck("cuFuncSetAttribute", dev.fn(f"pl_levels_{PL_VARIANT}"), MAX_DYN, PD_PAIR_SMEM)
    dev.sync()
    dev.pd_smem, dev.pd_tree_smem = smem["pd_commit"], smem["pd_tree"]


def nonce_args(nonce: int) -> tuple[int, int]:
    """A 64-bit nonce as hm_leaves' i1, i2 (each an int32 slot, its bits)."""
    lo, hi = nonce & 0xFFFFFFFF, nonce >> 32
    return tuple(v - (1 << 32) if v >> 31 else v for v in (lo, hi))


def pow2(n: int) -> int:
    """A tree's level-0 width over n leaves."""
    return 1 << max(n - 1, 0).bit_length()


def depth(n: int) -> int:
    return pow2(n).bit_length() - 1


def gemm_tile(name: str) -> tuple[int, int]:
    """A GEMM kernel's CTA tile (rows, columns): gemm_words 128 x 32 and gemm_words_d 64 x 32; the _d kernels 64 x 64, the
    _n64 ones 128 x 64, else 128 x 128."""
    rows = BM_D if name.endswith("_d") else BM
    if name.startswith("gemm_words"):
        return (rows, WORDS_N)
    return (rows, BM_D if name.endswith("_d") else BN_N64 if name.endswith("_n64") else BM)


def gemm_threads(name: str) -> int:
    """A GEMM's block: a consumer warpgroup per 64 rows of its tile, then the producer warpgroup."""
    return 128 * (gemm_tile(name)[0] // 64 + 1)


#: B's widening (pearl_c_sm90.cu's widen_*), the one step from B's stored form to the FP32 rows its tree commits: "bf16"
#: (exact) or "fp8_block" (E4M3 codes times their 128 x 128 block's FP32 scale, each product RN to FP32)
WIDEN = {"bf16": "widen_bf16", "fp8_block": "widen_fp8_block"}
#: the zero rows `check_widen` has widen_a fill past the BF16 rows
WIDEN_A_PAD = 3


def widen_sizes(kind: str, rows: int, k: int) -> tuple[int, int]:
    """B's stored form's bytes: (source, scales)."""
    return (2 * rows * k, 0) if kind == "bf16" else (rows * k, 4 * cdiv(rows, 128) * cdiv(k, 128))


def widen(dev, kind: str, src: int, scales: int, out: int, rows: int, k: int) -> None:
    """B's widening into out (rows x k FP32 words) from its stored form at src (and its block scales at `scales`)."""
    if k % 8:
        raise ValueError(f"the widening takes k a multiple of 8, not {k}")
    dev.launch(WIDEN[kind], cdiv(rows * k // 8, 256), 256, p=(src, scales, out), i=(rows, k))


def widen_a(dev, src: int, out: int, rows: int, filled: int, k: int, stream=None, pdl: bool = True) -> None:
    """A served call's A words (widen_a): its `rows` BF16 rows at src widened into out (FP32), then out's rows `rows` to
    `filled` - 1 zero, in one launch; with `pdl` False plainly, as off the main lane, where the launch before it on its
    stream isn't the one it follows."""
    if k % 8 or filled < rows:
        raise ValueError(f"widen_a takes k a multiple of 8 and filled >= rows, not k {k}, rows {rows}, filled {filled}")
    dev.launch("widen_a", cdiv(filled * k // 8, 256), 256, p=(src, 0, out), i=(rows, k, filled), stream=stream,
               **({} if pdl else {"pdl": False}))


def tile_suffix(bm: int, bn: int) -> str:
    """The chain's and the baselines' kernel suffix for a bm x bn tile."""
    return {(BM, BM): "", (BM, BN_N64): "_n64", (BM_D, BM_D): "_d"}[(bm, bn)]


class Pipeline:
    """One (shape, variant) instance: its buffers and its kernels in order, each a (name, phase, launch) step.  Phases:
    `epoch` (B's row tree, whose root the epoch's salt and seed_B follow), `unit` (the pads of A's tree and the tile tree,
    which depend on their domains alone), `job` (the weight side, once per weight and epoch), then the call: `commit` (A's
    row tree, seed_A with E_A's line key from its root, and each real row's seed: `seed_a`, `seeds_a`), `forming`, `gemm`
    (gemm_ct, the chain alone, for the split), `cleanup` and `hashing` (the tile leaves and the tile tree); and the
    baselines' `plain_job`, `plain`, `adopted_job`, `adopted`.  A caller that gives pd's counter (`pd_count`, 16 zero bytes
    it never writes) gets hash_decode.cuh's launches at decode (`pd`, `pd_variant`'s variant): `commit_a`, pd_commit, does
    every step of `FUSED` (the commit phase, A's pads and E_A's lines) in one launch, and `hash_leaf` is pd_tiles; the bytes
    are the same.  With `split_a` the commitment is two launches of the same bytes: `rows_a`, pd_commit in mode 1 (A's pads,
    each real row's digest, leaf, seed, line key and E_A line), then `tree_seed_a`, pd_tree (A's tree, seed_A, line_a and the
    padding rows' E_A lines under it), which only the padding rows' products read after it.  With `stamps` (a device buffer
    of PD_STAMP_BYTES) pd_commit and pd_tree run as their PD_STAMP twins.  Past pd, `chain` (default PEARLC_CHAIN, else "pl") picks A's commit launches after its row digests:
    "pl", hash_prefill.cuh's (`pl_steps`: pl_leaves, pl_levels, pl_seed_a and pl_seeds, at most PL_MAX_ROWS rows; `pl`),
    or "h3", hash_h3.cuh's (`row_tree`'s, then s512_seeds); the steps' names and bytes are the same.  The chain and the
    baselines run on 128 x 128 tiles when m and n are multiples of 128, on 128 x 64 when only m is, else on 64 x 64 (`bm`
    and `bn` force a tile); the 32-column words on 128 x 32 tiles where their rows are a multiple of 128 (A'F_B's only at a
    128-row chain tile), else 64 x 32.  Every GEMM takes its operands by the tensor maps in the pipeline's own `tma`
    (MAPS), each map's box the tile's.  gemm_pearlc stores C~, U and y.

    The hashing is format -h3, SHA-512 throughout (hash_h3.cuh): `rowdig_<tree>` first, s512_rows' SHA-512 row digest of
    each row (`pearl_c.row_digest`), then the `audit.HM96` trees (hm_leaves over the row digests under the prover's salt
    key, s512_frame_leaves, s512_pads, s512_nodes and s512_tree_small; 64-byte nodes), seed_A and each real row's seed with
    their line keys (s512_seeds; `pearl_c.unit_seed_a`, `pearl_c.row_seed`), E_A's lines (`lines_ea`: row i < m_real's line
    0 under its own line key, `pearl_c.PearlC.encode`'s E_A), and the tile leaves over C~ and U themselves
    (s512_tile_leaves, `pearl_c.leaf`) in an HM96 tree under the transcript's nonce.  `sha` (default PEARLC_SHA, else
    "fast") picks how the row digests and tile leaves run: "fast", hash_fast.cuh's two launches each (a thread a segment over
    every row or tile, then a thread a row or tile for the fold; scratch `segdig_<tree>`), or "ref", s512_rows (a CTA a row)
    and s512_tile_leaves (a CTA a tile).  Both write the same bytes.  A caller that owns the buffers gets the fast kernels
    for the trees whose `segdig_<tree>` it gives (`sizes` has them) and the reference kernels for the rest.

    `widen` (kind, source, scales), when given, makes B's FP32 rows from its stored form first (`widen_b`, the epoch's
    first step: WIDEN), so xb is its output.  `keys` are device pointers: `line_b` (E_B's and the F lines' key), the 64-byte domain ids `dom_a`, `dom_b`, `dom_t`,
    `unit` (the unit's 104 bytes that s512_seeds reads: salt ‖ root_B's 64 bytes ‖ u64 index), `salt_key` (the prover's hm96
    salt key) and `hm` (`audit.hm96_kernel_constants`).  `weight` is B's weight index, its tree's salt nonce HM96_WEIGHTS +
    weight.  E_A's line key is the pipeline's `line_a` buffer, written by `seed_a` (the key of the rows past m_real, the
    kernels' padding, which nothing commits)."""

    GEMM_SMEM: dict[str, int] = {}
    #: the steps one launch does: under pd, `commit_a` (pd_commit) for A's pads, row digests, tree, seed_A, row seeds and E_A's
    #: lines, and with `split_a`, `rows_a` (pd_commit's mode 1) and `tree_seed_a` (pd_tree) for the same steps (the padding
    #: rows' E_A lines only with `fillers`); under FORM_EP, `form_ep_a` (form_s5_ep) for A's E' and forming, and under
    #: `words_form`, `form_epw_a` (form_s5_epw) for those and A'F_B's windows
    FUSED = {"commit_a": ("pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "lines_ea"),
             "rows_a": ("pad_a", "rowdig_a", "hash_rows_a", "seeds_a", "lines_ea"), "tree_seed_a": ("tree_a", "seed_a"),
             "form_ep_a": ("eprime_a", "form_a"), "form_epw_a": ("eprime_a", "form_a", "af")}
    #: SMs a launch running beside the steps holds whole (pd_tree's CTA under PD_WHOLE), which the persistent GEMMs leave
    #: it: a GEMM CTA with no SM free would start only when that launch ends.  A tile's bytes don't depend on its CTA
    beside = 0
    #: pd_tiles takes PD_GROUP's tiles, PD_G a CTA, at every pd shape (the same leaves on a quarter of the CTAs): a whole
    #: capture's tile hashing beside A's tree, whose CTA and pd_commit's each hold an SM whole
    tiles_group = False
    #: where `reads_bf16`, a call's BF16 rows (address, rows: contiguous, k halves each) that pd_commit's mode 1 (`rows_a`)
    #: reads in xa's place, its rows past them zero, so xa (their widening) needn't be written before it: the same words
    rows_bf16: tuple[int, int] | None = None

    def __init__(self, dev: Dev, m: int, n: int, k: int, variant: str, keys: dict[str, int], xa: int, xb: int,
                 bufs: dict[str, int] | None = None, m_real: int | None = None, bm: int | None = None, weight: int = 0,
                 sha: str | None = None, bn: int | None = None, widen: tuple[str, int, int] | None = None,
                 chain: str | None = None, split_a: bool = False, fillers: bool = True):
        """`bufs` names the buffers when the caller owns them (only those its steps touch); `m_real` rows of A are the
        matmul's, committed and hashed, the rest filler that only the products see.  With `split_a`, `fillers` False leaves
        the filler rows' E_A lines as the buffer holds them: pd_tree then writes A's levels, seed_A and line_a alone, nothing
        the forming reads."""
        self.dev, self.m, self.n, self.k, self.variant, self.keys = dev, m, n, k, variant, keys
        self.widen, self.split_a, self.fillers, self.stamps = widen, split_a, fillers, 0
        self.sha = sha or os.environ.get("PEARLC_SHA", "fast")
        if self.sha not in ("fast", "ref"):
            raise ValueError(f"sha is 'fast' or 'ref', not {self.sha!r}")
        self.chain = chain or os.environ.get("PEARLC_CHAIN", "pl")
        if self.chain not in CHAINS:
            raise ValueError(f"chain is one of {CHAINS}, not {self.chain!r}")
        self.m_real = m if m_real is None else m_real
        self.g = VARIANT_G[variant]
        self.bm = bm or (BM if m % BM == 0 else BM_D)
        self.bn = bn or (BM if self.bm == BM and n % BM == 0 else BN_N64 if self.bm == BM else BM_D)
        self.tile = tile_suffix(self.bm, self.bn)
        if m % self.bm or n % self.bn or k % 256 or k < 256:
            raise ValueError(f"the kernels take m a multiple of {self.bm}, n of {self.bn} and k of 256, not {(m, n, k)}")
        if 4 * k > RS_MAX_SEGS * 1024:
            raise ValueError(f"s512_rows takes rows of at most {RS_MAX_SEGS} 1 KiB segments (k <= {RS_MAX_SEGS * 256}), not k = {k}")
        self.nb, self.weight = NB, weight
        self.row_bytes = 4 * k
        self.tiles = cdiv(self.m_real, TILE) * (n // TILE)
        sizes = self.sizes(m, n, k, self.m_real)
        self.size = sizes
        self.owned = bufs is None
        self.b = {name: dev.alloc(s) for name, s in sizes.items()} if self.owned else dict(bufs)
        self.allocated: list[str] = []
        if not self.owned:
            self.b.pop("tma", None)
            self.b["tma"] = dev.alloc(sizes["tma"])
            self.allocated.append("tma")
        self.b["xa"], self.b["xb"] = xa, xb
        self.size["xa"], self.size["xb"] = m * self.row_bytes, n * self.row_bytes
        self.pd = pd_variant(self.m_real, n, k) if self.sha == "fast" and "pd_count" in self.b else None
        self.pl = self.pd is None and self.chain == "pl" and self.m_real <= PL_MAX_ROWS
        if self.pd is not None or self.pl:
            pd_setup(dev)
        if self.sha == "fast" and any(x in self.b for x in SEGDIG) and not getattr(dev, "rs_heads", False):
            dev.launch("s512_heads", cdiv(RS_MAX_SEGS * (RS_MAX_SEGS + 1) // 2 + RS_MAX_SEGS + 1, 128), 128)
            dev.sync()
            dev.rs_heads = True
        self.boxes: dict[tuple[str, str], tuple[int, int]] = {}
        self.encode_maps()

    def words_rows(self, rows: int, small: bool = False) -> int:
        """The words GEMM's tile rows for an operand of `rows` rows: 128 where it divides them (and not `small`), else 64."""
        return BM if rows % BM == 0 and not small else BM_D

    def encode_maps(self) -> None:
        """The tensor maps into the pipeline's own `tma` (a caller's is never shared: the maps hold these operands'
        addresses), a set per operand pair of MAPS whose buffers the pipeline has, each box the rows its GEMM's tile takes:
        P_A and P_B the peel's for the chain's pair, A and B again for the others (which load no tail).  Encoded here, once,
        so no launch writes a map (a captured graph replays the launches alone)."""
        d, b, m, n, k = self.dev, self.b, self.m, self.n, self.k
        rows = {"at": m, "bt": n, "qa": m, "qb": n, "qa_s5": m, "qb_s5": n, "fbT32": WORDS_N, "faT32": WORDS_N, "fbT128": BM}
        box = {("at", "bt"): (self.bm, self.bn), ("qa", "qb"): (self.bm, self.bn), ("qa_s5", "qb_s5"): (self.bm, self.bn),
               ("at", "fbT32"): (self.words_rows(m, self.bm == BM_D), WORDS_N), ("bt", "faT32"): (self.words_rows(n), WORDS_N),
               ("fbT128", "faT32"): (BM, WORDS_N)}
        for (x, y), slot in MAPS.items():
            if x in b and y in b:
                bx, by = box[(x, y)]
                ab = d.tmap(b[x], k, rows[x], bx) + d.tmap(b[y], k, rows[y], by)
                peel = slot == 0 and "pa_t" in b and "pb_t" in b
                tail = d.tmap(b["pa_t"], PEEL_ROW, m, bx) + d.tmap(b["pb_t"], PEEL_ROW, n, by) if peel else ab
                d.ck("cuMemcpyHtoD_v2", u64(b["tma"] + MAP_SET * slot), C.c_char_p(ab + tail), MAP_SET)
                self.boxes[(x, y)] = (bx, by)

    @staticmethod
    def sizes(m: int, n: int, k: int, m_real: int | None = None) -> dict[str, int]:
        m_real = m if m_real is None else m_real
        tiles = cdiv(m_real, TILE) * (n // TILE)
        return {
            "ea_codes": m * 32, "eb_codes": n * 32, "fa_codes": k * 32, "faT32": 32 * k, "fb_codes": k * 32, "fbT32": 32 * k,
            "fbT128": 128 * k, "g": 128 * 32 * 4,
            "alpha_a": 4 * m, "beta_a": 4 * m, "ia_a": 4 * m, "aux_a": 8 * m, "epa_codes": m * 32,
            "alpha_b": 4 * n, "beta_b": 4 * n, "ia_b": 4 * n, "aux_b": 8 * n, "epb_codes": n * 32,
            "at": m * k, "bt": n * k, "at2": max(m, n) * k, "af": m * 32 * 4, "af_win": m * 32 * 4 * (k // WINDOW),
            "bf": n * 32 * 4, "pa_t": m * 128, "pb_t": n * 128,
            "c": m * n * 4, "c2": m * n * 4, "u": m * n * 4, "y": m * n * 2, "y2": m * n * 2, "tma": MAP_SET * len(MAPS),
            "line_a": 32, "seed_a": 32, "seeds_a": 32 * m_real, "lkeys_a": 32 * m_real, "rowdig_a": 64 * m_real, "rowdig_b": 64 * n,
            "leaves": tiles * NB,
            **{f"tree_{t}": NB * (2 * pow2(count) - 1) for t, count in (("a", m_real), ("b", n), ("t", tiles))},
            "qa": m * k, "qb": n * k, "sa": 4 * m, "sb": 4 * n, "y_plain": m * n * 2, "y_fast": m * n * 2,
            "qa_s5": m * k, "qb_s5": n * k, "y_adopt": m * n * 2, "diff": 8,
            "segdig_a": 64 * (k // 256) * m_real, "segdig_b": 64 * (k // 256) * n, "segdig_t": 64 * 32 * tiles,
        }

    def root(self, tag: str) -> int:
        """The device address of a tree's root, its last node."""
        return self.b[f"tree_{tag}"] + self.size[f"tree_{tag}"] - NB

    def levels(self, tag: str, count: int, dom: str | None = None) -> None:
        """A tree's levels above level 0 (leaves and pads written): s512_nodes a level at a time down to S512_SMALL nodes,
        then s512_tree_small for the rest in one launch, under the domain id `dom` (default dom_<tag>)."""
        d, base, di = self.dev, self.b[f"tree_{tag}"], self.keys[dom or f"dom_{tag}"]
        width, off, level = pow2(count), 0, 0
        while width > S512_SMALL:
            d.launch("s512_nodes", cdiv(width // 2, 64), 64, p=(base + NB * off, base + NB * (off + width), di),
                     i=(level, width // 2))
            off, width, level = off + width, width // 2, level + 1
        if width > 1:
            d.launch("s512_tree_small", 1, 64, p=(base + NB * off, di), i=(width, level))

    def tree_setup(self, tag: str, count: int, dom: str) -> list[tuple[str, object]]:
        """A tree's pads (s512_pads): its domain's alone, so once per unit."""
        d, b, width = self.dev, self.b, pow2(count)
        if width == count:
            return []
        return [(f"pad_{tag}", lambda: d.launch("s512_pads", cdiv(width - count, 64), 64, p=(b[f"tree_{tag}"], self.keys[dom]),
                                                i=(count, width)))]

    def row_tree(self, tag: str, x: str, count: int, dom: str) -> list[tuple[str, object]]:
        """A tree over `count` FP32 rows of x: their row digests (s512_rows, a CTA a row), the HM96 leaves over them, then
        its levels.  A's salt nonce is its unit index, read from `unit`; B's HM96_WEIGHTS + weight."""
        b = self.b
        nonce = (0, 0) if tag == "a" else nonce_args(HM96_WEIGHTS + self.weight)
        block = self.keys["unit"] + 96 if tag == "a" else 0
        return [(f"rowdig_{tag}", lambda: self.row_digests(b[x], b[f"rowdig_{tag}"], b.get(f"segdig_{tag}", 0), count)),
                (f"hash_rows_{tag}", lambda: Pipeline.hm_level0(self, tag, b[f"rowdig_{tag}"], count, dom, block, nonce)),
                (f"tree_{tag}", lambda: self.levels(tag, count, dom))]

    def pl_steps(self) -> list[tuple[str, object]]:
        """A's commit steps under chain "pl": its row digests (`row_tree`'s), then hash_prefill.cuh's launches under the h3
        chain's names: `hash_rows_a` A's level 0 (pl_leaves, its hm96 nonce read from `unit` as hm_leaves'), `tree_a` its
        levels (`pl_levels`), `seed_a` seed_A with line_a from the root (pl_seed_a), `seeds_a` each real row's seed and line
        key (pl_seeds)."""
        d, b, kk, mr, v, half = self.dev, self.b, self.keys, self.m_real, PL_VARIANT, PD_PAIR_SMEM // 2
        return [self.row_tree("a", "xa", mr, "dom_a")[0],
                ("hash_rows_a", lambda: d.launch(f"pl_leaves_{v}", cdiv(mr, PL_ROWS), 128, half,
                                                 p=(b["rowdig_a"], b["tree_a"], kk["salt_key"], kk["hm"], kk["unit"], kk["dom_a"]),
                                                 i=(mr,))),
                ("tree_a", self.pl_levels),
                ("seed_a", lambda: d.launch(f"pl_seed_a_{v}", 1, 64, half, p=(self.root("a"), kk["unit"], b["seed_a"], b["line_a"]))),
                ("seeds_a", lambda: d.launch(f"pl_seeds_{v}", cdiv(mr, PL_SEEDS), 64, half,
                                             p=(b["rowdig_a"], b["seeds_a"], b["lkeys_a"], kk["unit"]), i=(mr,)))]

    def pl_levels(self) -> None:
        """A's levels above level 0 under "pl" (leaves and pads written): pl_levels, PL_LEVELS levels a launch (the last
        launch the rest), a CTA a node of the launch's top level."""
        d, base, width, off, level = self.dev, self.b["tree_a"], pow2(self.m_real), 0, 0
        while width > 1:
            lv = min(PL_LEVELS, width.bit_length() - 1)
            d.launch(f"pl_levels_{PL_VARIANT}", width >> lv, 128, PD_PAIR_SMEM, p=(base + NB * off, self.keys["dom_a"]),
                     i=(width, level, lv))
            off += sum(width >> q for q in range(lv))
            width, level = width >> lv, level + lv

    def row_digests(self, x: int, out: int, scratch: int, count: int) -> None:
        """The sha512/row-seg/v1 digests of `count` FP32 rows at x into out: s512_rows, or with a scratch buffer
        s512_row_segs then s512_folds."""
        d, segs = self.dev, self.k // 256
        if self.sha == "ref" or not scratch:
            d.launch("s512_rows", count, 128, p=(x, out), i=(count, self.k))
            return
        d.launch("s512_row_segs", cdiv(count * segs, RS_FAST_NT), RS_FAST_NT, p=(x, scratch), i=(count, self.k))
        d.launch("s512_folds", cdiv(count, RS_FOLD_NT), RS_FOLD_NT, p=(scratch, out), i=(count, segs, count, segs))

    def tile_leaves(self) -> None:
        """The tile leaves over C~ and U into `leaves`: s512_tile_leaves, or with `segdig_t` s512_tile_segs then s512_folds
        (the full row blocks' tiles at 32 segments, the last block's at its rows / 2)."""
        d, b, mr, n = self.dev, self.b, self.m_real, self.n
        if self.sha == "ref" or not b.get("segdig_t"):
            d.launch("s512_tile_leaves", self.tiles, 32, p=(b["c"], b["u"], b["leaves"]), i=(mr, n))
            return
        full, last = mr // TILE * (n // TILE), mr % TILE // 2
        d.launch("s512_tile_segs", cdiv(32 * full + n // TILE * last, RS_FAST_NT), RS_FAST_NT, p=(b["c"], b["u"], b["segdig_t"]),
                 i=(mr, n))
        d.launch("s512_folds", cdiv(self.tiles, RS_FOLD_NT), RS_FOLD_NT, p=(b["segdig_t"], b["leaves"]), i=(self.tiles, 32, full, last))

    @property
    def pd_a(self) -> int:
        """The pd variant A's commitment launches (pd_commit, pd_tree): `pd`, without PD_GROUP under ONE_COMMIT."""
        return self.pd & ~PD_GROUP if ONE_COMMIT else self.pd

    def reads_bf16(self) -> bool:
        """Whether `rows_a` may read a call's BF16 rows (`rows_bf16`): a split pd pipeline, BF16_ROWS, and a cubin whose
        pd_commit reads them (BF16_ROWS_MARK)."""
        return (BF16_ROWS and self.pd is not None and self.split_a
                and getattr(self.dev, "has", lambda _: False)(BF16_ROWS_MARK))

    def pd_commit(self, mode: int = 0) -> None:
        """A's commitment under pd (`FUSED`'s steps), pd_commit in mode 0: a CTA a real row, E_A's lines for all m rows (row
        i < m_real's under its own line key, the rest under line_a), its counter left zero.  Mode 1 (`rows_a`) stops after A's
        level 0 and the rows' lines, and `pd_tree` does the rest; it reads `rows_bf16` where they are set."""
        d, b, kk, v, s = self.dev, self.b, self.keys, self.pd_a | (PD_STAMP if self.stamps else 0), self.k // 256
        rows = self.rows_bf16 if mode == 1 else None
        if rows is not None and not self.reads_bf16():
            raise RuntimeError("rows_bf16 is set on a pipeline whose pd_commit doesn't read BF16 rows (`reads_bf16`)")
        d.launch(f"pd_commit_{v}", self.m_real, 32 * (cdiv(s, 32) * 2 + 2), d.pd_smem[v],
                 p=(b["xa"], b["rowdig_a"], b["tree_a"], b["seeds_a"], b["lkeys_a"], kk["unit"], kk["dom_a"], kk["salt_key"],
                    kk["hm"], b["pd_count"], b["seed_a"], b["line_a"], b["ea_codes"], self.stamps,
                    *((0, rows[0]) if rows else ())),
                 i=(self.m_real, self.k, mode, self.m, *((0, rows[1]) if rows else ())))

    def pd_tree(self) -> None:
        """`split_a`'s `tree_seed_a`: pd_tree, one CTA, A's levels above level 0, seed_A, line_a, and with `fillers` the
        padding rows' E_A lines under it (rows m_real..m - 1, where there are any)."""
        d, b, kk, v = self.dev, self.b, self.keys, self.pd_a | (PD_STAMP if self.stamps else 0)
        ea = b["ea_codes"] if self.m_real < self.m and self.fillers else 0
        d.launch(f"pd_tree_{v}", 1, 128, d.pd_tree_smem[v],
                 p=(0, 0, b["tree_a"], 0, 0, kk["unit"], kk["dom_a"], 0, 0, 0, b["seed_a"], b["line_a"], ea, self.stamps),
                 i=(self.m_real, 0, 0, self.m))

    def pd_tiles(self) -> None:
        """The tile leaves under pd: pd_tiles, a CTA a tile (32 threads), or with PD_GROUP (or `tiles_group`) a CTA of PD_G
        tiles of a row block."""
        d, b, v, mr, n = self.dev, self.b, self.pd | (PD_GROUP if self.tiles_group else 0), self.m_real, self.n
        p = (b["c"], b["u"], b["leaves"])
        if v & PD_GROUP:
            d.launch(f"pd_tiles_{v}", cdiv(mr, TILE) * cdiv(n // TILE, PD_G), 32 * PD_G, PD_PAIR_SMEM // 2, p=p, i=(mr, n))
        else:
            d.launch(f"pd_tiles_{v}", self.tiles, 32, p=p, i=(mr, n))

    def hm_level0(self, tag: str, x: int, count: int, dom: str, block: int = 0, nonce: tuple[int, int] = (0, 0)) -> None:
        """An `audit.HM96` tree's level 0 over `count` 64-byte values at x: hm_leaves under the salt key, the nonce read at
        `block` (A's unit index) or given, then s512_frame_leaves in place."""
        d, tree, kk = self.dev, self.b[f"tree_{tag}"], self.keys
        d.launch("hm_leaves", cdiv(count, 64), 64, p=(x, tree, kk["salt_key"], kk["hm"], 0, block), i=(count, *nonce))
        d.launch("s512_frame_leaves", cdiv(count, 64), 64, p=(tree, kk[dom]), i=(0, count))

    def tree_leaves(self, tag: str, values: int, count: int, dom: str) -> None:
        """A tree's level 0 over `count` 64-byte values at device address `values` (the tile leaves): `hm_level0` under the
        transcript's nonce."""
        Pipeline.hm_level0(self, tag, values, count, dom, nonce=nonce_args(HM96_TRANSCRIPT))

    def form_alt_launch(self, tag: str, kernel: str) -> None:
        """`kernel` (one of FORM_ALTS) over one side's rows into at2."""
        b, rows = self.b, self.m if tag == "a" else self.n
        self.dev.launch(kernel, (self.k // 128, rows // 64), 128,
                        p=(b[f"x{tag}"], b[f"alpha_{tag}"], b[f"ep{tag}_codes"], b[f"f{tag}_codes"], b["at2"]), i=(rows, self.k))

    def form_alt(self, tag: str, kernel: str) -> int:
        """`kernel` over one side's rows into at2, and the words where form_s5's codes (at or bt) differ from it."""
        self.form_alt_launch(tag, kernel)
        self.dev.sync()
        return self.diff(f"{tag}t", "at2")

    def free(self) -> None:
        for name, p in self.b.items():
            if self.owned and name not in ("xa", "xb") or name in self.allocated:
                self.dev.free(p)

    def gemm(self, name: str, a: str, b: str, M: int, N: int, grid: int | None = None, out=("c",), pa=0, pb=0, rs=0, cs=0,
             dg=0, key=0, live: int | None = None):
        """`live`: the rows from 0 whose products are stored (i4; 0 all M), by default m_real at 64 x 64 tiles (LIVE)."""
        if name not in self.GEMM_SMEM:
            probe = self.dev.alloc(16)
            self.dev.launch(name + "_smem", 1, 1, p=(probe,))
            self.dev.sync()
            self.GEMM_SMEM[name] = int.from_bytes(self.dev.read(probe, 4), "little")
            self.dev.free(probe)
            self.dev.ck("cuFuncSetAttribute", self.dev.fn(name), MAX_DYN, self.GEMM_SMEM[name])
        tm, bn = gemm_tile(name)
        if self.boxes.get((a, b)) != (tm, bn):
            raise ValueError(f"{name} takes {tm} x {bn} tiles, and ({a}, {b})'s maps are {self.boxes.get((a, b))}")
        tiles = (M // tm) * (N // bn) * (self.k // WINDOW if name.startswith("gemm_words_win") else 1)
        if live is None:
            live = self.m_real if LIVE and (tm, bn) == (BM_D, BM_D) and self.m_real < M else 0
        ptrs = [self.b[a], self.b[b], 0, 0, 0, 0, 0, 0, 0, dg, self.b["tma"] + MAP_SET * MAPS[(a, b)], key]
        for slot, x in zip((2, 3, 4), out):
            ptrs[slot] = self.b[x] if x else 0
        ptrs[5], ptrs[6], ptrs[7], ptrs[8] = pa, pb, rs, cs
        ctas = min(tiles, grid or self.dev.sms - self.beside)
        if GRID_BALANCE and grid is None:
            ctas = cdiv(tiles, cdiv(tiles, ctas))
        self.dev.launch(name, ctas, gemm_threads(name), self.GEMM_SMEM[name], p=ptrs, i=(M, N, self.k, RASTER, live))

    def extra(self) -> dict[str, object]:
        """The chain's launches by name, beside `steps`' choice: `gemm_pearlc_cu` (C~, U and y stored) and `gemm_pearlc_s`
        (y alone: the call without hashing); under `words_win`, A'F_B's windows both ways, `af` (gemm_words_win_d over A's
        codes) and, from a cubin with it, `form_epw_a` (form_s5_epw: A's E', codes and windows in one launch)."""
        d, b, m, n, k, tile = self.dev, self.b, self.m, self.n, self.k, self.tile
        peel = lambda: {"pa": b["pa_t"], "pb": b["pb_t"], "rs": b["ia_a"], "cs": b["ia_b"]}  # noqa: E731
        out = {"gemm_pearlc_cu": lambda: self.gemm(f"gemm_pearlc{tile}", "at", "bt", m, n, out=("c", "u", "y"), **peel()),
               "gemm_pearlc_s": lambda: self.gemm(f"gemm_pearlc_s{tile}", "at", "bt", m, n, out=(None, None, "y"), **peel())}
        if self.words_win():
            out["af"] = lambda: self.gemm("gemm_words_win_d", "at", "fbT32", m, WORDS_N, out=("af_win",))
            if getattr(d, "has", lambda _: False)("form_s5_epw"):
                out["form_epw_a"] = lambda: d.launch("form_s5_epw", (k // 128, m // 64), 128,
                                                     p=(b["xa"], b["alpha_a"], b["ea_codes"], b["fa_codes"], b["at"], b["beta_a"],
                                                        b["epa_codes"], b["fbT32"], b["af_win"]), i=(m, k))
        return out

    def words_win(self) -> bool:
        """Whether A'F_B runs a CTA a window (WORDS_WIN): on 64-row tiles, where the buffers hold its windows."""
        return WORDS_WIN and self.bm == BM_D and "af_win" in self.b

    def words_form(self) -> bool:
        """Whether A's forming launch makes A'F_B's windows too (WORDS_FORM, `form_epw_a`): under FORM_EP and `words_win`,
        from a cubin with form_s5_epw."""
        return WORDS_FORM and FORM_EP and self.words_win() and getattr(self.dev, "has", lambda _: False)("form_s5_epw")

    def words(self, a: str, b: str, M: int, out: str, small: bool = False) -> None:
        """The FP32 words of an (M x K) by (32 x K) product: on 128 x 32 tiles where 128 divides M (and not `small`), else
        64 x 32."""
        self.gemm("gemm_words" + ("" if self.words_rows(M, small) == BM else "_d"), a, b, M, WORDS_N, out=(out,))

    def commit_steps(self) -> list[tuple[str, object]]:
        """The commit phase's steps: pd's `commit_a` (with `split_a` its `rows_a` and `tree_seed_a`), chain "pl"'s
        (`pl_steps`), or chain "h3"'s (`row_tree`'s, then s512_seeds for seed_A with line_a and for each real row's seed and
        line key)."""
        d, b, kk, mr = self.dev, self.b, self.keys, self.m_real
        if self.pd is not None and self.split_a:
            return [("rows_a", lambda: self.pd_commit(1)), ("tree_seed_a", self.pd_tree)]
        if self.pd is not None:
            return [("commit_a", self.pd_commit)]
        if self.pl:
            return self.pl_steps()
        return self.row_tree("a", "xa", mr, "dom_a") + [
            ("seed_a", lambda: d.launch("s512_seeds", 1, 128, p=(self.root("a"), b["seed_a"], b["line_a"], kk["unit"]), i=(1, 0, 0))),
            ("seeds_a", lambda: d.launch("s512_seeds", cdiv(mr, 128), 128, p=(b["rowdig_a"], b["seeds_a"], b["lkeys_a"], kk["unit"]),
                                         i=(mr, 0, 1)))]

    def steps(self) -> list[tuple[str, str, object]]:
        d, b, m, n, k, tile = self.dev, self.b, self.m, self.n, self.k, self.tile
        ct, chain = f"gemm_ct{tile}", self.extra()

        def lines(tag, key, side, factor, count, codes, t32=0, t128=0):
            norm = NORM_16 if factor == 1 else NORM_256
            return (f"lines_{tag}", lambda: d.launch("lines", cdiv(count, 128), 128, p=(key, b[codes], t32, t128),
                                                    i=(side, factor, count, norm)))

        def form_side(tag, rows, x, e_codes, f_codes, out, fused=False):
            ep = f"ep{tag}_codes"
            stats = (f"stats_{tag}", lambda: d.launch("stats_s5", cdiv(rows, 8), 256, i=(rows, k),
                                                      p=(b[x], b[f"alpha_{tag}"], b[f"beta_{tag}"], b[f"ia_{tag}"], b[f"aux_{tag}"])))
            if fused:
                return [stats, (f"form_ep_{tag}", lambda: d.launch("form_s5_ep", (k // 128, rows // 64), 128,
                                                                   p=(b[x], b[f"alpha_{tag}"], b[e_codes], b[f_codes], b[out],
                                                                      b[f"beta_{tag}"], b[ep]), i=(rows, k)))]
            return [stats,
                    (f"eprime_{tag}", lambda: d.launch("eprime", cdiv(rows, 128), 128, p=(b[e_codes], b[f"beta_{tag}"], b[ep]),
                                                       i=(rows,))),
                    (f"form_{tag}", lambda: d.launch("form_s5", (k // 128, rows // 64), 128,
                                                     p=(b[x], b[f"alpha_{tag}"], b[ep], b[f_codes], b[out]), i=(rows, k)))]

        job = [lines("eb", self.keys["line_b"], 1, 0, n, "eb_codes"),
               lines("fa", self.keys["line_b"], 0, 1, k, "fa_codes", b["faT32"]),
               lines("fb", self.keys["line_b"], 1, 1, k, "fb_codes", b["fbT32"], b["fbT128"]),
               *form_side("b", n, "xb", "eb_codes", "fb_codes", "bt"),
               ("bf", lambda: self.words("bt", "faT32", n, "bf")),
               ("g", lambda: self.words("fbT128", "faT32", 128, "g")),
               ("pb", lambda: d.launch("pb", cdiv(n * 32, 256), 256, p=(b["epb_codes"], 0, b["bf"], b["g"], b["pb_t"]), i=(n,)))]
        mr, tiles = self.m_real, self.tiles
        ea = ("lines_ea", lambda: d.launch("lines", cdiv(m, 128), 128, p=(b["line_a"], b["ea_codes"], 0, 0, b["lkeys_a"]),
                                           i=(0, 0, m, NORM_256, mr)))
        pd = self.pd is not None
        forming = [ea] * (not pd) + form_side("a", m, "xa", "ea_codes", "fa_codes", "at", fused=FORM_EP)
        gemm = [("gemm_ct", lambda: self.gemm(ct, "at", "bt", m, n, out=("c2",)))]
        if self.words_win():
            pa_w = ("pa", lambda: d.launch("pa_w", cdiv(m * 64, 256), 256, p=(b["epa_codes"], b["af_win"], b["af"], b["pa_t"]),
                                           i=(m, k // WINDOW)))
            if self.words_form():
                forming[-1], cleanup = ("form_epw_a", chain["form_epw_a"]), [pa_w]
            else:
                cleanup = [("af", chain["af"]), pa_w]
        else:
            cleanup = [("af", lambda: self.words("at", "fbT32", m, "af", small=self.bm == BM_D)),
                       ("pa", lambda: d.launch("pa", cdiv(m * 64, 256), 256, p=(b["epa_codes"], 0, b["af"], b["pa_t"]), i=(m,)))]
        cleanup.append(("gemm_pearlc", chain["gemm_pearlc_cu"]))
        widen_b = [("widen_b", lambda: widen(d, self.widen[0], self.widen[1], self.widen[2], b["xb"], n, k))] if self.widen else []
        epoch = widen_b + self.tree_setup("b", n, "dom_b") + self.row_tree("b", "xb", n, "dom_b")
        unit = self.tree_setup("a", mr, "dom_a") * (not pd) + self.tree_setup("t", tiles, "dom_t")
        commit = self.commit_steps()
        hashing = [("hash_leaf", self.pd_tiles if pd else self.tile_leaves),
                   ("tree_leaves_t", lambda: self.tree_leaves("t", b["leaves"], tiles, "dom_t")),
                   ("tree_t", lambda: self.levels("t", tiles))]
        plain_job = [("plainq_b", lambda: d.launch("plainq", n, 256, p=(b["xb"], b["qb"], b["sb"]), i=(n, k)))]
        plain = [("plainq_a", lambda: d.launch("plainq", m, 256, p=(b["xa"], b["qa"], b["sa"]), i=(m, k))),
                 ("gemm_plain", lambda: self.gemm(f"gemm_plain{tile}", "qa", "qb", m, n, out=(None, None, "y_plain"), rs=b["sa"],
                                                  cs=b["sb"])),
                 ("gemm_fast", lambda: self.gemm(f"gemm_fast{tile}", "qa", "qb", m, n, out=(None, None, "y_fast"), rs=b["sa"],
                                                 cs=b["sb"]))]
        adopted_job = [("quant_b", lambda: d.launch("quant_s5", cdiv(n * k // 16, 256), 256, p=(b["xb"], b["alpha_b"], b["qb_s5"]),
                                                    i=(n, k)))]
        adopted = [("quant_a", lambda: d.launch("quant_s5", cdiv(m * k // 16, 256), 256, p=(b["xa"], b["alpha_a"], b["qa_s5"]),
                                                i=(m, k))),
                   ("gemm_adopt", lambda: self.gemm(f"gemm_plain{tile}", "qa_s5", "qb_s5", m, n, out=(None, None, "y_adopt"),
                                                    rs=b["ia_a"], cs=b["ia_b"]))]
        tag = lambda phase, xs: [(nm, phase, f) for nm, f in xs]  # noqa: E731
        return (tag("epoch", epoch) + tag("unit", unit) + tag("job", job) + tag("commit", commit) + tag("forming", forming)
                + tag("gemm", gemm) + tag("cleanup", cleanup) + tag("hashing", hashing) + tag("plain_job", plain_job)
                + tag("plain", plain) + tag("adopted_job", adopted_job) + tag("adopted", adopted))

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
#: form_s5's gate (the one-code cast), run into at2 and held to form_s5's codes
FORM_ALTS = ("form_s5_scalar",)
GEMM_RERUNS = (("gemm_pearlc", ("c", "u", "y")), ("gemm_ct", ("c2",)), ("gemm_plain", ("y_plain",)), ("gemm_fast", ("y_fast",)),
               ("af", ("af",)), ("bf", ("bf",)), ("gemm_adopt", ("y_adopt",)), ("gemm_pearlc_s", ("y",)))
ALIASES = {"c2": "c"}
#: a rerun step's followers at each tile, each with what it writes: pa after the words (under WORDS_WIN it adds up their
#: windows into af)
RERUN_AFTER = {"af": {"pa": "pa_t"}}
#: the tiles the check re-runs every GEMM at, (tag, bm, bn): the fixture's m and n are multiples of 128, so each is forced
TILES = (("bm128", BM, BM), ("n64", BM, BN_N64), ("bm64", BM_D, BM_D))


def compare(dev: Dev, pipe: Pipeline, name: str, want: bytes, alias: str | None = None) -> dict:
    got = dev.read(pipe.b[alias or name], len(want))
    if got == want:
        return {"ok": True, "bytes": len(want)}
    bad = [i for i in range(0, len(want), 4) if got[i:i + 4] != want[i:i + 4]]
    return {"ok": False, "bytes": len(want), "words_differing": len(bad),
            "first": [{"offset": i, "got": got[i:i + 4].hex(), "want": want[i:i + 4].hex()} for i in bad[:6]]}


#: what pd's launches write (pd_commit: A's commitment and E_A's lines; pd_tiles: the tile leaves)
PD_OUTPUTS = ("rowdig_a", "tree_a", "seed_a", "line_a", "seeds_a", "lkeys_a", "ea_codes", "leaves")
POISON = 0xA5


def check_pd(dev: Dev, pipe: Pipeline, want) -> dict:
    """hash_decode.cuh's served launches on `pipe`'s buffers (its C~ and U computed), each variant of PD_SERVED (`commit_a`,
    then `hash_leaf`; also `split_a`'s `rows_a` and `tree_seed_a` in its place, ``/split``) and each stamp twin of
    PD_STAMPED (``pd_V``, with stamps), after every output in PD_OUTPUTS is poisoned, against the fixture's bytes; the
    counter left zero.  The no-write control, the poison with no launch, must fail every output."""
    count, stamps, res = dev.alloc(16), dev.alloc(PD_STAMP_BYTES), {}
    make = lambda split: Pipeline(dev, pipe.m, pipe.n, pipe.k, pipe.variant, pipe.keys, pipe.b["xa"], pipe.b["xb"],  # noqa: E731
                                  bufs={**pipe.b, "pd_count": count}, m_real=pipe.m_real, bm=BM_D, weight=pipe.weight,
                                  split_a=split)
    whole = make(False)
    if whole.pd is None:
        whole.free()
        dev.free(count)
        dev.free(stamps)
        return {"pd@domain": {"ok": False, "why": f"{(pipe.m_real, pipe.n, pipe.k)} is outside pd's domain"}}
    forms = [(None, None, ())]
    split = make(True)
    forms += [(whole, (v, at, ""), ("commit_a", "hash_leaf")) for at in (0, stamps) for v in PD_SERVED]
    forms += [(split, (v, 0, "/split"), ("rows_a", "tree_seed_a", "hash_leaf")) for v in PD_SERVED]
    for p_, form, names in forms:
        for x in PD_OUTPUTS:
            dev.ck("cuMemsetD8_v2", u64(pipe.b[x]), POISON, pipe.size[x])
        if p_ is not None:
            v, p_.stamps, tag = form
            p_.pd = v
            steps = {nm: fn for nm, _, fn in p_.steps()}
            for nm in names:
                steps[nm]()
            p_.stamps = 0
        dev.sync()
        label = None if p_ is None else f"pd_{v | (PD_STAMP if form[1] else 0)}{tag}"
        for x in PD_OUTPUTS:
            r = compare(dev, pipe, x, want(x))
            if label is None:
                res[f"{x}@pd/no-write"] = {"ok": not r["ok"], "bytes": r["bytes"]}
            else:
                res[f"{x}@{label}"] = r
        if label is not None:
            res[f"pd_count@{label}"] = {"ok": dev.read(count, 16) == bytes(16), "bytes": 16}
    for p_ in {id(f[0]): f[0] for f in forms if f[0] is not None}.values():
        p_.free()
    dev.free(count)
    dev.free(stamps)
    return res


#: what A's commit phase writes past pd (each chain's steps), and the real rows of the chains' comparisons past 256 rows:
#: ranks and indices of two bytes (A's level 0 from rank 256, level 1 from index 256), and prefill's M (glm53.PREFILL_M),
#: whose trees take pl_levels' widest splits
CHAIN_OUTPUTS, CHAIN_WIDE_ROWS = ("rowdig_a", "tree_a", "seed_a", "line_a", "seeds_a", "lkeys_a"), (600, 2048, 8192)


def run_commit(pipe: Pipeline) -> None:
    """A's pads, then the pipeline's commit phase."""
    for _, f in pipe.tree_setup("a", pipe.m_real, "dom_a") + pipe.commit_steps():
        f()


def check_chains(dev: Dev, pipe: Pipeline, want) -> dict:
    """A's commitment past pd under each chain of CHAINS on `pipe`'s buffers (`run_commit`), every output in CHAIN_OUTPUTS
    poisoned first, against the fixture's bytes; the no-write control, the poison with no launch, must fail every output.
    Then over each of CHAIN_WIDE_ROWS generated rows of A (its own buffers), "pl"'s bytes against "h3"'s, every output
    poisoned before each chain."""
    res = {}
    for chain in (None, *CHAINS):
        for x in CHAIN_OUTPUTS:
            dev.ck("cuMemsetD8_v2", u64(pipe.b[x]), POISON, pipe.size[x])
        if chain is not None:
            p_ = Pipeline(dev, pipe.m, pipe.n, pipe.k, pipe.variant, pipe.keys, pipe.b["xa"], pipe.b["xb"], bufs=pipe.b,
                          m_real=pipe.m_real, weight=pipe.weight, chain=chain)
            run_commit(p_)
            dev.sync()
            p_.free()
        dev.sync()
        for x in CHAIN_OUTPUTS:
            r = compare(dev, pipe, x, want(x))
            if chain is None:
                res[f"{x}@chain/no-write"] = {"ok": not r["ok"], "bytes": r["bytes"]}
            else:
                res[f"{x}@chain_{chain}"] = r
    n, k = pipe.n, pipe.k
    for mr in CHAIN_WIDE_ROWS:
        m = cdiv(mr, BM_D) * BM_D
        sizes = Pipeline.sizes(m, n, k, mr)
        bufs = {x: dev.alloc(sizes[x]) for x in (*CHAIN_OUTPUTS, "segdig_a")}
        xa = dev.put(hashlib.shake_256(b"pearl-c-sm90/chains").digest(4 * mr * k) + bytes(4 * (m - mr) * k))
        got = {}
        for chain in CHAINS:
            for x in CHAIN_OUTPUTS:
                dev.ck("cuMemsetD8_v2", u64(bufs[x]), POISON, sizes[x])
            p_ = Pipeline(dev, m, n, k, pipe.variant, pipe.keys, xa, 0, bufs=bufs, m_real=mr, chain=chain)
            run_commit(p_)
            dev.sync()
            p_.free()
            got[chain] = {x: dev.read(bufs[x], sizes[x]) for x in CHAIN_OUTPUTS}
        for x in CHAIN_OUTPUTS:
            pl, h3 = got["pl"][x], got["h3"][x]
            res[f"{x}@chain_pl/{mr}_rows"] = {
                "ok": pl == h3 and h3 != bytes([POISON]) * sizes[x], "bytes": sizes[x],
                "words_differing": 0 if pl == h3 else sum(pl[i:i + 4] != h3[i:i + 4] for i in range(0, sizes[x], 4))}
        for p_ in (*bufs.values(), xa):
            dev.free(p_)
    return res


def check_widen(dev: Dev, man: dict, load) -> dict:
    """Each of WIDEN's kinds on the fixture's stored B (`widen`: its rows, k and files), its output poisoned first, against
    the fixture's FP32 rows; and widen_a on the BF16 rows, filled with WIDEN_A_PAD zero rows past them."""
    spec, res = man["widen"], {}
    rows, k = spec["shape"]
    src, want = dev.put(load("", spec["files"]["src_bf16"])), load("", spec["files"]["xb_bf16"]) + bytes(4 * k * WIDEN_A_PAD)
    out = dev.alloc(len(want))
    dev.ck("cuMemsetD8_v2", u64(out), POISON, len(want))
    widen_a(dev, src, out, rows, rows + WIDEN_A_PAD, k)
    dev.sync()
    got = dev.read(out, len(want))
    res["xa@widen_a"] = {"ok": got == want, "bytes": len(want),
                         "words_differing": sum(got[i:i + 4] != want[i:i + 4] for i in range(0, len(want), 4))}
    dev.free(src)
    dev.free(out)
    for kind in WIDEN:
        src = dev.put(load("", spec["files"][f"src_{kind}"]))
        scales = dev.put(load("", spec["files"]["scales"])) if kind == "fp8_block" else 0
        want = load("", spec["files"][f"xb_{kind}"])
        out = dev.alloc(len(want))
        dev.ck("cuMemsetD8_v2", u64(out), POISON, len(want))
        widen(dev, kind, src, scales, out, rows, k)
        dev.sync()
        got = dev.read(out, len(want))
        res[f"xb_{kind}@widen"] = {"ok": got == want, "bytes": len(want),
                                   "words_differing": sum(got[i:i + 4] != want[i:i + 4] for i in range(0, len(want), 4))}
        for p_ in (src, scales, out):
            dev.free(p_)
    return res


#: the real rows of `check_live`'s GEMMs: a row block with real rows past a warp's first eight, then whole filler blocks
LIVE_ROWS, LIVE_FILL = 40, 0xAB


def check_live(dev: Dev, pipe: Pipeline, want) -> dict:
    """The A side's GEMMs of GEMM_RERUNS (all but bf, over B's rows, and af under `words_win`, whose windows and pa_w store
    every row) at 64 x 64 on LIVE_ROWS real rows (i4): each output's real rows against the fixture's, and its filler rows as
    LIVE_FILL's memset left them."""
    res = {}
    p_ = Pipeline(dev, pipe.m, pipe.n, pipe.k, pipe.variant, pipe.keys, pipe.b["xa"], pipe.b["xb"], bufs=pipe.b, bm=BM_D,
                  bn=BM_D, weight=pipe.weight)
    steps = {**p_.extra(), **{nm: fn for nm, _, fn in p_.steps()}}
    real = p_.gemm
    p_.gemm = lambda *a, **kw: real(*a, **{**kw, "live": LIVE_ROWS})
    for step, outs in GEMM_RERUNS:
        if step == "bf" or step == "af" and p_.words_win():
            continue
        for o in outs:
            dev.ck("cuMemsetD8_v2", u64(pipe.b[o]), LIVE_FILL, pipe.size[o])
        steps[step]()
        dev.sync()
        for o in outs:
            w = want(ALIASES.get(o, o))
            got, cut = dev.read(pipe.b[o], len(w)), LIVE_ROWS * (len(w) // pipe.m)
            res[f"{o}@{step}/live{LIVE_ROWS}"] = {"ok": got[:cut] == w[:cut] and got[cut:] == bytes([LIVE_FILL]) * (len(w) - cut),
                                                  "bytes": len(w)}
    p_.gemm = real
    p_.free()
    return res


def check_form_words(dev: Dev, pipe: Pipeline, want) -> dict:
    """form_s5_epw (`form_epw_a`), where the cubin has it, at 64-row tiles: A's E', codes and A'F_B's windows from one launch,
    the windows added up by pa_w, each output poisoned first: E', the codes, AF and P_A against the fixture's."""
    p_ = Pipeline(dev, pipe.m, pipe.n, pipe.k, pipe.variant, pipe.keys, pipe.b["xa"], pipe.b["xb"], bufs=pipe.b, bm=BM_D,
                  bn=BM_D, weight=pipe.weight)
    extra, res = p_.extra(), {}
    if "form_epw_a" in extra:
        outs = ("epa_codes", "at", "af", "pa_t")
        for o in (*outs, "af_win"):
            dev.ck("cuMemsetD8_v2", u64(pipe.b[o]), POISON, pipe.size[o])
        extra["form_epw_a"]()
        {nm: fn for nm, _, fn in p_.steps()}["pa"]()
        dev.sync()
        res = {f"{o}@form_s5_epw": compare(dev, pipe, o, want(o)) for o in outs}
    p_.free()
    return res


def check(dev: Dev, man: dict, base: str, variants: list[str]) -> dict:
    m, n, k = man["shape"]
    load = lambda d, f: open(os.path.join(base, d, f["file"]), "rb").read()  # noqa: E731
    keys = {name: dev.put(bytes.fromhex(h)) for name, h in man["keys"].items()}
    keys.update({name: dev.put(load("", f)) for name, f in man.get("key_files", {}).items()})
    common = man["common"]
    results = {}
    for variant in variants:
        files = man["variants"][variant]["files"]
        want = lambda name: load(variant, files[name]) if name in files else load("", common[name])  # noqa: E731
        xa, xb = dev.put(load("", common["xa"])), dev.put(load("", common["xb"]))
        pipe = Pipeline(dev, m, n, k, variant, keys, xa, xb, weight=man["weight"])
        pipe.run_all()
        res = {}
        for name in [*common, *files]:
            if name in ("xa", "xb"):
                continue
            res[name] = compare(dev, pipe, name, want(name))
            if name == "c":
                res["c@gemm_ct"] = compare(dev, pipe, name, want(name), alias="c2")
        res.update(check_pd(dev, pipe, want))
        res.update(check_chains(dev, pipe, want))
        res.update(check_widen(dev, man, load))
        for tag in ("a", "b"):
            for kernel in FORM_ALTS:
                pipe.form_alt(tag, kernel)
                res[f"{tag}t@{kernel}"] = compare(dev, pipe, f"{tag}t", want(f"{tag}t"), alias="at2")
        for tag, bm, bn in TILES:
            p_ = Pipeline(dev, m, n, k, variant, keys, xa, xb, bufs=pipe.b, bm=bm, bn=bn, weight=man["weight"])
            steps = {**p_.extra(), **{nm: fn for nm, _, fn in p_.steps()}}
            for grid in (None, 1, 3) if tag == "bm128" else (None, 3):
                for step, outs in GEMM_RERUNS:
                    outs = (*outs, *RERUN_AFTER.get(step, {}).values())
                    for o in outs:
                        dev.ck("cuMemsetD8_v2", u64(pipe.b[o]), 0xAB, pipe.size[o])
                    real = p_.gemm

                    def pinned(*a, _real=real, _grid=grid, **kw):
                        kw["grid"] = _grid
                        return _real(*a, **kw)
                    p_.gemm = pinned
                    steps[step]()
                    for after in RERUN_AFTER.get(step, {}):
                        steps[after]()
                    p_.gemm = real
                    dev.sync()
                    for o in outs:
                        res[f"{o}@{step}/{tag}/grid{grid or 'all'}"] = compare(dev, pipe, o, want(ALIASES.get(o, o)))
            p_.free()
        res.update(check_live(dev, pipe, want))
        res.update(check_form_words(dev, pipe, want))
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

CALL_PHASES = ("commit", "forming", "gemm", "cleanup", "hashing")
TIMING_KEYS = {"line_b": 32, "dom_a": 64, "dom_b": 64, "dom_t": 64, "unit": 104, "salt_key": 32, "hm": 128 + 192 * 256 * 64}


def parse_shape(s: str) -> tuple[int, int, int]:
    """SHAPES' entries: "8192" (m = n = k) or "MxNxK"."""
    parts = [int(x) for x in s.lower().split("x")]
    return (parts[0],) * 3 if len(parts) == 1 else tuple(parts)


def timing(dev: Dev, nv: Nvml, shape: tuple[int, int, int], variant: str, reps: int, gates: dict) -> dict:
    m_real, n, k = shape
    m = cdiv(m_real, BM_D) * BM_D
    keys = {name: dev.put(hashlib.shake_256(f"pearl-c-timing/{name}".encode()).digest(size)) for name, size in TIMING_KEYS.items()}
    xa, xb = dev.alloc(m * 4 * k), dev.alloc(n * 4 * k)
    for x, rows, seed in ((xa, m_real, 11), (xb, n, 12)):
        dev.launch("gen_f32", cdiv(rows * k, 256), 256, p=(x,), i=(rows, k, seed))
    pipe = Pipeline(dev, m, n, k, variant, keys, xa, xb, m_real=m_real)
    pipe.run_all()
    out = measure(dev, nv, pipe, reps, gates)
    if gates["pearl_c"]:
        for kind in WIDEN:
            src_n, sc_n = widen_sizes(kind, n, k)
            src, sc, dst = dev.alloc(src_n), dev.alloc(sc_n), dev.alloc(4 * n * k)
            t = dev.timed(lambda kind=kind, src=src, sc=sc, dst=dst: widen(dev, kind, src, sc, dst, n, k), reps)
            out["runs"][f"widen_{kind}"], out["ms"][f"widen_{kind}"] = [round(x, 4) for x in t], round(statistics.median(t), 4)
            for p_ in (src, sc, dst):
                dev.free(p_)
    pipe.free()
    for p_ in (xa, xb, *keys.values()):
        dev.free(p_)
    return out


def measure(dev: Dev, nv: Nvml, pipe: Pipeline, reps: int, gates: dict) -> dict:
    """The timed part of `timing` on a pipeline whose kernels have run once: medians, phases, ratios, consistency."""
    m, n, k, steps, extra = pipe.m, pipe.n, pipe.k, pipe.steps(), pipe.extra()
    before = nv.read()
    med, runs = {}, {}
    for name, phase, f in steps:
        if phase in CALL_PHASES + ("epoch", "unit", "job") and not gates["pearl_c"] or \
                phase in ("plain", "plain_job", "adopted", "adopted_job") and not gates["plain"]:
            continue
        t = dev.timed(f, reps)
        runs[name], med[name] = [round(x, 4) for x in t], statistics.median(t)
    if gates["pearl_c"]:
        for tag in ("a", "b"):
            for kernel in FORM_ALTS:
                t = dev.timed(lambda tag=tag, kernel=kernel: pipe.form_alt_launch(tag, kernel), reps)
                nm = f"form_{tag}{kernel.removeprefix('form_s5')}"
                runs[nm], med[nm] = [round(x, 4) for x in t], statistics.median(t)
        for name, f in extra.items():
            if name not in med:
                t = dev.timed(f, reps)
                runs[name], med[name] = [round(x, 4) for x in t], statistics.median(t)
    after = nv.read()
    phases = {ph: sum(med.get(nm, 0.0) for nm, p, _ in steps if p == ph) for ph in ("job",) + CALL_PHASES}
    out = {"kind": "timing", "label": "diagnostic, unlocked unless the clock record says so", "shape": [m, n, k],
           "variant": pipe.variant, "reps": reps, "gates": gates,
           "ms": {nm: round(v, 4) for nm, v in med.items()}, "runs": runs, "clk_before": before, "clk_after": after}
    if gates["pearl_c"]:
        call = [f for nm, p, f in steps if p in ("commit", "forming", "cleanup", "hashing")]
        y_only = extra.get("gemm_pearlc_s")
        no_hash = [y_only if y_only and nm == "gemm_pearlc" else f for nm, p, f in steps if p in ("forming", "cleanup")]
        e2e = statistics.median(dev.timed(lambda: [f() for f in call], reps))
        e2e_nh = statistics.median(dev.timed(lambda: [f() for f in no_hash], reps))
        chain = med["gemm_pearlc"]
        split = {"commit": phases["commit"], "forming": phases["forming"], "gemm": med["gemm_ct"],
                 "cleanup": phases["cleanup"] - med["gemm_ct"], "hashing": phases["hashing"]}
        out.update({"split_ms": {kk: round(v, 4) for kk, v in split.items()}, "call_sum_ms": round(sum(split.values()), 4),
                    "call_e2e_ms": round(e2e, 4), "call_e2e_no_hash_ms": round(e2e_nh, 4), "job_ms": round(phases["job"], 4),
                    "pearlc_tflops": round(2 * m * n * k / chain / 1e9, 1)})
        extra["gemm_pearlc_cu"]()
        out["consistency"] = {"c_ct_vs_pearlc_words_differing": pipe.diff("c2", "c")}
        dev.copy(pipe.b["c2"], pipe.b["u"], pipe.size["u"])
        extra["gemm_pearlc_cu"]()
        out["consistency"]["u_rerun_words_differing"] = pipe.diff("c2", "u")
        dev.copy(pipe.b["y2"], pipe.b["y"], pipe.size["y"])
        if y_only:
            y_only()
            out["consistency"]["y_s_vs_pearlc_words_differing"] = pipe.diff("y", "y2")
        out["consistency"].update({f"{tag}t_form_s5_vs_{kernel.removeprefix('form_s5_')}_words_differing": pipe.form_alt(tag, kernel)
                                   for tag in ("a", "b") for kernel in FORM_ALTS})
    if gates["plain"] and "gemm_plain" in med:
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
    image = open(os.path.join(HERE, "pearl_c_sm90.cubin"), "rb").read()
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
    env2 = {"kind": "mem", "free_total": dev.free_mem(), "pdl": dev.pdl}
    emit(env2)
    variants = os.environ.get("VARIANTS", "h100").split()
    t0 = time.time()
    res = check(dev, man, base, variants)
    emit({"kind": "check", "shape": man["shape"], "seconds": round(time.time() - t0, 2), "variants": res})
    ok = all(r["pearl_c_ok"] and r["plain_ok"] for r in res.values())
    print(f"check at {man['shape']}: " + "; ".join(f"{v} {r['buffers']} buffers, failed {r['failed'] or 'none'}" for v, r in res.items()),
          file=sys.stderr)
    reps = int(os.environ.get("REPS", "5"))
    for shape in [parse_shape(s) for s in os.environ.get("SHAPES", "8192").split()]:
        for variant in variants:
            gates = {"pearl_c": res[variant]["pearl_c_ok"] or ungated, "plain": res[variant]["plain_ok"] or ungated,
                     "ungated": ungated}
            if not (gates["pearl_c"] or gates["plain"]):
                continue
            t0 = time.time()
            row = timing(dev, nv, shape, variant, reps, gates)
            row["seconds"] = round(time.time() - t0, 2)
            row["m_real"] = shape[0]
            ok = ok and not any(row.get("consistency", {}).values())
            emit(row)
            label = "x".join(map(str, shape))
            if "split_ms" in row:
                print(f"{label} {variant}: call {row['call_e2e_ms']:.3f} ms, no hashing {row['call_e2e_no_hash_ms']:.3f} ms; split "
                      f"{row['split_ms']}", file=sys.stderr)
            if "ratio_vs_own_plain_gemm" in row:
                s, r = row["split_ms"], row["ratio_vs_own_plain_gemm"]
                print(f"{label} {variant}: call {row['call_e2e_ms']:.3f} ms = {r['call_e2e']:.3f}x this build's plain FP8 GEMM "
                      f"({row['plain_tflops']} / {row['fast_tflops']} TFLOPS promoted / not); split commit {s['commit']:.3f}, GEMM "
                      f"{s['gemm']:.3f}, clean-up {s['cleanup']:.3f}, forming {s['forming']:.3f}, hashing {s['hashing']:.3f}; no "
                      f"hashing {r['no_hash_e2e']:.3f}x; "
                      f"consistency {row['consistency']}", file=sys.stderr)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main(sys.argv))
