#!/usr/bin/env python3
"""Pearl-C on one RTX PRO 6000 Blackwell (sm_120) from the prebuilt pearl_c_sm120.cubin, through the CUDA driver API
(ctypes on libcuda.so.1, stdlib only, nothing built on the card).

0. Identity, before anything runs: exactly one visible device (CUDA_VISIBLE_DEVICES a single index), compute capability
   12.0, an "NVIDIA RTX PRO 6000 Blackwell" name, and its UUID equal to PEARLC_EXPECT_UUID (server.md's table).  NVML is
   addressed by that UUID, never by index (NVML ignores CUDA_VISIBLE_DEVICES).  A mismatch exits 3 before any launch.
1. The check.  At check.json's shape (fixture.py's), every kernel runs on the fixture's inputs and every buffer it writes is
   compared byte for byte with the fixture, for each variant (`sm120`, G = 4, and `sm120-unpromoted`), hashing format -h3
   (SHA-512 throughout); the GEMMs, with gemm_pearlc_s (y alone), are re-run on all, 1 and 3 CTAs and compared again
   (`<buffer>@<step>/<tile>/grid<n>`), and the chain's 64 x 64 kernels (decode's) too, with decode's split kernels (words_k,
   stats_r), its chain on 128-deep stages, and the ones they replace (`Pipeline.replaced`).  A variant is timed only if all
   of its buffers match (C~ and U bit-exact against the CPU reference), the plain baselines only if theirs do.  Exit 4 if
   any check fails.
2. Diagnostic timing (not the headline: GPU 6's harness times the arm against autotuned baselines), at each of SHAPES
   (default 8192, m = n = k; or MxNxK, m padded to 64 rows): each kernel REPS times after a warm-up, its median, the phases
   (the weight side; the call's commitment of A, forming, GEMM, clean-up and hashing), the call end to end with and without
   its hashes (commit and hashing; without, the chain is gemm_pearlc_s where it runs), and consistency checks (C~ of gemm_ct
   equals gemm_pearlc's; a second gemm_pearlc gives the same U; gemm_pearlc_s's y equals gemm_pearlc's; form_s5_scalar and
   form_s5_r144, timed beside form_s5, and the RATE kernels write form_s5's codes; at decode, words_k, stats_r and the
   chain on 128-deep stages write the words of the kernels they replace, timed beside them as af_warps, stats_a_warps
   and gemm_pearlc_bk64, and the chain on the real rows writes their C~, U and y as it does over all m rows, timed beside it
   as gemm_pearlc_full; at decode every comparison is of the real rows, the only ones a consumer reads).
   `form_rates` prices the forming loop per code in the mix it runs in, in units of one dense E4M3 MAC, and the packed cast
   within it (form_rate less form_rate_nocast; at the 144-byte row stride, form_rate_r144 less form_rate_nocast_r144).
   form_s5's codes and the lines' transposed F lines are in K32 order (pearl_c_sm120.cu); the check's fixture holds them in
   it, and every product over k against the unpermuted operands'.  The timing's keys and domain ids are stand-ins of the same sizes (diagnostic
   only; the arm's calls run on a real epoch).  NVML's clocks, power and throttle reasons before and after.

One JSON object per line on stdout as each part finishes, a summary on stderr.

  python3 run.py [check.json]   (env: PEARLC_EXPECT_UUID, SHAPES="8192", REPS=5, VARIANTS="sm120 sm120-unpromoted",
                                 PEARLC_LIBCUDA, PEARLC_LIBNVML, PEARLC_SPLIT=0 for decode on words and stats_s5,
                                 PEARLC_LIVE=0 for decode's 64 x 64 GEMMs over all m rows,
                                 PEARLC_DEEP=0 for decode's chain on 64-deep stages)

Dry runs on a stand-in driver only: PEARLC_ANY_GPU=1 skips the identity check, and PEARLC_UNGATED=1 (refused without it)
times regardless of the check and marks every row ungated.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
import statistics
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NAME_PREFIX = "NVIDIA RTX PRO 6000 Blackwell"
SM_COUNT, MAX_DYN, CC_MAJOR, CC_MINOR, MAX_SHARED_OPTIN, SHARED_SIZE = 16, 8, 75, 76, 97, 1
NORM_256, NORM_16 = 0x4600, 0x4400      # bf16(32 x the line norm): E's lines, and F's under v1
NT, BM, BM_D, RASTER = 256, 128, 64, 8
#: a caller with more real rows than this (a prefill's) pads them to BM rather than BM_D, so the chain runs on the mainloop:
#: a served prefill step's chains took 18.9 ms at 640 rows there against 31.7 ms at 656 rows on 64 x 64 tiles
#: (r20261024-171936-c297); at fewer rows the mainloop's 128 x 128 tiles leave SMs idle
MAINLOOP_ROWS = 512
ML_NT = 384                  # mainloop_sm120.cuh's block: a TMA producer warpgroup and 8 consumer warps
WK_WARPS, WK_CL = 16, 4      # words_k's block (warps) and cluster (blocks) per m16 x n8 tile (pearl_c_sm120.cu)
SR_ROWS, SR_MAX_K = 256, 32768   # stats_r (a block a row) for at most SR_ROWS rows of k <= SR_MAX_K, else stats_s5
#: decode's split kernels (words_k, stats_r); PEARLC_SPLIT=0 runs the kernels they replace (words, stats_s5), the same words
SPLIT = os.environ.get("PEARLC_SPLIT", "1") != "0"
#: the 64 x 64 GEMMs on the real rows alone (i4 = m_real: a warp of filler rows skips its products, stores and A's loads);
#: PEARLC_LIVE=0 runs them over all m rows, the same real rows
LIVE = os.environ.get("PEARLC_LIVE", "1") != "0"
#: decode's chain (gemm_pearlc_d) on 128-deep stages with L2's 256-byte fetch; PEARLC_DEEP=0 runs the 64-deep one it
#: replaces (gemm_pearlc4_d), the same words
DEEP = os.environ.get("PEARLC_DEEP", "1") != "0"
#: B~'s prefetch into L2 ahead of a decode call's chain (`Pipeline.prefetch_b`): prefetch_l2's form (PEARLC_PREFETCH_FORM: 0 a
#: line a thread, 1 the same evict_last, 2 a 16 KiB bulk prefetch a thread, the default: served b8's whole step at an ITL p50
#: of 20.61 ms against 21.42 in form 0 and 21.85 without it, r20261024-193651-3ef1), its blocks, and B~'s leading bytes it
#: takes (PEARLC_PREFETCH_MB, 0 all of them)
PREFETCH_FORM = int(os.environ.get("PEARLC_PREFETCH_FORM", "2"))
PREFETCH_CTAS = int(os.environ.get("PEARLC_PREFETCH_CTAS", "64"))
PREFETCH_MB = int(os.environ.get("PEARLC_PREFETCH_MB", "0"))
PREFETCH_FORMS = (0, 1, 2)
VARIANT_G = {"sm120": 4, "sm120-unpromoted": 0}
TMAP_UINT8, TMAP_SWIZZLE_128B, TMAP_L2_128B, TMAP_BYTES = 0, 3, 2, 128
#: the mainloop GEMMs' operand pairs, a set of four tensor maps (A, B, P_A, P_B) each at `tma` + ML_SET * slot
ML_MAPS = {("at", "bt"): 0, ("qa", "qb"): 1, ("qa_s5", "qb_s5"): 2}
ML_SET = 4 * TMAP_BYTES

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER
CUDA = {
    "cuInit": [u], "cuDriverGetVersion": [P(i32)], "cuDeviceGetCount": [P(i32)], "cuDeviceGet": [P(i32), i32],
    "cuDeviceGetName": [C.c_char_p, i32, i32], "cuDeviceGetUuid_v2": [C.c_char_p, i32],
    "cuDeviceGetAttribute": [P(i32), i32, i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuFuncSetAttribute": [vp, i32, i32], "cuFuncGetAttribute": [P(i32), i32, vp], "cuMemAlloc_v2": [P(u64), sz],
    "cuMemFree_v2": [u64],
    "cuMemsetD8_v2": [u64, C.c_ubyte, sz], "cuMemcpyHtoD_v2": [u64, vp, sz], "cuMemcpyDtoH_v2": [vp, u64, sz],
    "cuMemcpyDtoD_v2": [u64, u64, sz], "cuMemcpyDtoDAsync_v2": [u64, u64, sz, vp], "cuMemGetInfo_v2": [P(sz), P(sz)],
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


#: Args' bytes, p's slots unsigned and i's signed, as Args' fields
ARGS, NO_ARGS = struct.Struct("<16Q16i"), (0,) * 16


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
        # launch()'s: one Args and its params array, rewritten at every launch (cuLaunchKernel copies the parameters before
        # it returns, and launches come from one thread, as `stream` already assumes), and cuLaunchKernel without argtypes,
        # its pointers passed as c_void_p (the streams' cached in `streams`) and its sizes as C ints
        self.args = Args()
        self.params = (vp * 1)(C.addressof(self.args))
        self.cu_launch = self.cu["cuLaunchKernel"]
        self.cu_launch.restype = i32
        self.streams: dict[int | None, vp] = {}

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
            self.fail(f, r)

    def fail(self, f: str, r: int):
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

    def copy_async(self, dst: int, src: int, n: int, stream=None) -> None:
        self.ck("cuMemcpyDtoDAsync_v2", u64(dst), u64(src), n, self.stream if stream is None else stream)

    def launch(self, name: str, grid, block: int, smem: int = 0, p=(), i=(), stream=None) -> None:
        try:
            ARGS.pack_into(self.args, 0, *p, *NO_ARGS[len(p):], *i, *NO_ARGS[len(i):])
        except struct.error:             # an int past its slot's range: its low bits, as ctypes' slots store it
            ARGS.pack_into(self.args, 0, *(v & 0xFFFFFFFFFFFFFFFF for v in p), *NO_ARGS[len(p):],
                           *((v & 0xFFFFFFFF ^ 0x80000000) - 0x80000000 for v in i), *NO_ARGS[len(i):])
        g0, g1, g2 = tuple(grid) + (1,) * (3 - len(grid)) if isinstance(grid, (tuple, list)) else (grid, 1, 1)
        raw = self.stream if stream is None else stream
        s = raw if isinstance(raw, vp) else self.streams.get(raw)   # a harness's own CUstream object (pouw_decode's lanes)
        if s is None:
            s = self.streams[raw] = vp(raw)
        f = self.fns.get(name) or self.fn(name)
        r = self.cu_launch(f, g0, g1, g2, block, 1, 1, smem, s, self.params, None)
        if r:
            self.fail("cuLaunchKernel", r)

    def sync(self) -> None:
        self.ck("cuCtxSynchronize")

    def tmap(self, ptr: int, depth: int, rows: int, box_rows: int = BM) -> bytes:
        """A CUtensorMap over a row-major (rows x depth) byte matrix as mainloop_sm120.cuh loads it: UINT8, rank 2, boxes of
        box_rows x 128 bytes, SWIZZLE_128B, L2 promotion 128B, no fill."""
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


#: hash_decode.cuh's launches at decode, the variants pearl_c_sm120.cu builds (its PD_VARIANT_LIST): pd_commit, A's commitment
#: in one launch, a CTA a real row (A's pads; each row's digest, hm96 leaf, seed, line key and E_A line; then on the last CTA
#: to finish A's tree, seed_A, line_a and the filler rows' E_A lines), and pd_tiles, the tile leaves.  pd_commit takes at most
#: PD_MAX_ROWS rows and k of at most PD_MAX_K (its 192 threads: two warps of pairs a 32 segments, and two more); its paired
#: compressions take PD_PAIR_SMEM bytes of dynamic shared memory, and PD_GROUP's tiles PD_G a CTA (hash_decode.cuh's)
PD_SERVED, PD_MAX_ROWS, PD_MAX_K, PD_PAIR_SMEM, PD_G, PD_WHOLE, PD_GROUP = (851, 979), 256, 64 * 256, 2 * 8 * 3080, 4, 64, 128


def pd_variant(m_real: int, n: int, k: int) -> int | None:
    """The pd variant a decode shape's tiles take (its pd_commit is that without PD_GROUP, `Pipeline.pd_commit`): 979
    (PD_GROUP's tiles, four a CTA) where the tile grid is wide (n at least 16,384)
    or every row block is full, else 851, the fastest at Llama-3.1-8B's linears at m16 = 32 and 128 (runs
    r20261018-225110-fbef and r20261018-230602-5632); None outside pd_commit's rows and k."""
    if m_real > PD_MAX_ROWS or k > PD_MAX_K:
        return None
    return 979 if n >= 16384 or m_real % TILE == 0 else 851


def pd_setup(dev) -> None:
    """Once per module: hash_decode.cuh's label states (pd_heads) and hash_fast.cuh's header tables (s512_heads), and the
    dynamic shared memory of each served pd_commit and pd_tree (`pd_smem`, `pd_tree_smem`), with PD_WHOLE all a CTA may
    take beside its static, so no side lane's CTA shares its SMs."""
    if getattr(dev, "pd_smem", None):
        return
    if not getattr(dev, "rs_heads", False):
        dev.launch("s512_heads", cdiv(RS_MAX_SEGS * (RS_MAX_SEGS + 1) // 2 + RS_MAX_SEGS + 1, 128), 128)
        dev.rs_heads = True
    dev.launch("pd_heads", 1, 32)
    optin, smem = dev.get("cuDeviceGetAttribute", MAX_SHARED_OPTIN, dev.dev), {"pd_commit": {}, "pd_tree": {}}
    for v in sorted({x & ~PD_GROUP for x in PD_SERVED}):
        for f in smem:
            fn = dev.fn(f"{f}_{v}")
            smem[f][v] = optin - dev.get("cuFuncGetAttribute", SHARED_SIZE, fn) if v & PD_WHOLE else PD_PAIR_SMEM
            if smem[f][v] < PD_PAIR_SMEM:
                raise RuntimeError(f"{f}_{v} gets {smem[f][v]} bytes of dynamic shared memory, not its {PD_PAIR_SMEM}")
            dev.ck("cuFuncSetAttribute", fn, MAX_DYN, smem[f][v])
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
    """A GEMM kernel's CTA tile (rows, columns): 64 x 64 for the _d kernels, 128 x 32 for gemm_words, else 128 x 128."""
    if name.startswith("gemm_words"):
        return (BM, 32)
    return (BM_D, BM_D) if name.endswith(("_d", "_d_u")) else (BM, BM)


def is_ml(name: str) -> bool:
    """The 128 x 128 GEMMs, which run on the mainloop (ML_GEMM): ML_NT threads, operands by tensor map."""
    return gemm_tile(name) == (BM, BM)


class Pipeline:
    """One (shape, variant) instance: its buffers and its kernels in order, each a (name, phase, launch) step.  Phases:
    `epoch` (B's row tree, whose root the epoch's salt and seed_B follow), `unit` (the pads of A's tree and the tile tree,
    which depend on their domains alone), `job` (the weight side, once per weight and epoch), then the call: `commit` (A's
    row tree, seed_A with E_A's line key from its root, and each real row's seed: `seed_a`, `seeds_a`), `forming`, `gemm`
    (gemm_ct, the chain alone, for the split), `cleanup` and `hashing` (the tile leaves and the tile tree); and the
    baselines' `plain_job`, `plain`, `adopted_job`, `adopted` (at m a multiple of 128 only).  A caller that gives pd's
    counter (`pd_count`, 16 zero bytes it never writes) gets hash_decode.cuh's launches at decode (`pd`, `pd_variant`'s
    variant): `commit_a`, pd_commit, does every step of `FUSED` (the commit phase, A's pads and E_A's lines) in one launch,
    and `hash_leaf` is pd_tiles; the bytes are the same.  With `split_a` the commitment is two launches of the same bytes:
    `rows_a`, pd_commit in mode 1 (A's pads, each real row's digest, leaf, seed, line key and E_A line), then
    `tree_seed_a`, pd_tree (A's tree, seed_A, line_a and the padding rows' E_A lines under it), which only the padding
    rows' products read after it; outside pd, the same two steps over the unsplit kernels (`FUSED_GENERIC`, A's level 0
    in `tree_seed_a`).  The chain runs on 128 x 128
    tiles when m is a multiple of 128, else on 64 x 64 (`bm` forces either); the 32-column words on 128 x 32 CTA tiles, but
    A'F_B at 64 x 64 on the `words` kernel's warp tiles.  The 128 x 128 GEMMs run on the mainloop, their operands by the
    tensor maps in `tma` (ML_MAPS).  gemm_pearlc stores C~, U and y.

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

    `keys` are device pointers: `line_b` (E_B's and the F lines' key), the 64-byte domain ids `dom_a`, `dom_b`, `dom_t`,
    `unit` (the unit's 104 bytes that s512_seeds reads: salt ‖ root_B's 64 bytes ‖ u64 index), `salt_key` (the prover's hm96
    salt key) and `hm` (`audit.hm96_kernel_constants`).  `weight` is B's weight index, its tree's salt nonce HM96_WEIGHTS +
    weight.  E_A's line key is the pipeline's `line_a` buffer, written by `seed_a` (the key of the rows past m_real, the
    kernels' padding, which nothing commits)."""

    GEMM_SMEM: dict[str, int] = {}
    #: the steps one launch does under pd: `commit_a` (pd_commit) for A's pads, row digests, tree, seed_A, row seeds and E_A's
    #: lines; with `split_a`, `rows_a` (pd_commit's mode 1) and `tree_seed_a` (pd_tree) for the same steps (the padding rows'
    #: E_A lines only with `fillers`)
    FUSED = {"commit_a": ("pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "lines_ea"),
             "rows_a": ("pad_a", "rowdig_a", "hash_rows_a", "seeds_a", "lines_ea"), "tree_seed_a": ("tree_a", "seed_a")}
    #: `split_a` outside pd (the pipeline's own `FUSED` there), the same kernels as the unsplit steps: ``rows_a`` A's pads,
    #: row digests, each real row's seed and line key and the real rows' E_A lines (`lines` reading no line_a), then
    #: ``tree_seed_a`` A's hm96 level 0 over the row digests, its levels, seed_A and line_a (with `fillers`, every row's E_A
    #: line again, the real rows' the same bytes)
    FUSED_GENERIC = {"rows_a": ("pad_a", "rowdig_a", "seeds_a", "lines_ea"), "tree_seed_a": ("hash_rows_a", "tree_a", "seed_a")}
    #: SMs a launch running beside the steps holds whole (pd_tree's CTA under PD_WHOLE), which the persistent GEMMs leave
    #: it: a GEMM CTA with no SM free would start only when that launch ends.  A tile's bytes don't depend on its CTA
    beside = 0

    def __init__(self, dev: Dev, m: int, n: int, k: int, variant: str, keys: dict[str, int], xa: int, xb: int,
                 bufs: dict[str, int] | None = None, m_real: int | None = None, bm: int | None = None, weight: int = 0,
                 sha: str | None = None, split_a: bool = False, fillers: bool = True):
        """`bufs` names the buffers when the caller owns them (only those its steps touch); `m_real` rows of A are the
        matmul's, committed and hashed, the rest filler that only the products see (and the 64 x 64 GEMMs skip, LIVE).
        With `split_a`, `fillers` False leaves the filler rows' E_A lines as the buffer holds them: ``tree_seed_a`` (pd_tree,
        or outside pd `FUSED_GENERIC`'s) then writes A's tree, seed_A and line_a alone, nothing the forming reads."""
        self.dev, self.m, self.n, self.k, self.variant, self.keys = dev, m, n, k, variant, keys
        self.split_a, self.fillers = split_a, fillers
        self.sha = sha or os.environ.get("PEARLC_SHA", "fast")
        if self.sha not in ("fast", "ref"):
            raise ValueError(f"sha is 'fast' or 'ref', not {self.sha!r}")
        self.m_real = m if m_real is None else m_real
        self.g = VARIANT_G[variant]
        self.suffix = "" if self.g else "_u"
        self.bm = bm or (BM if m % BM == 0 else BM_D)
        if m % self.bm or n % BM or k % 1024:
            raise ValueError(f"the kernels take m a multiple of {self.bm}, n of 128 and k of 1024, not {(m, n, k)}")
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
        self.b["xa"], self.b["xb"] = xa, xb
        self.size["xa"], self.size["xb"] = m * self.row_bytes, n * self.row_bytes
        self.pd = pd_variant(self.m_real, n, k) if self.sha == "fast" and "pd_count" in self.b else None
        if self.pd is not None:
            pd_setup(dev)
        elif split_a:
            self.FUSED = {**Pipeline.FUSED, **Pipeline.FUSED_GENERIC}
        if self.sha == "fast" and any(x in self.b for x in SEGDIG) and not getattr(dev, "rs_heads", False):
            dev.launch("s512_heads", cdiv(RS_MAX_SEGS * (RS_MAX_SEGS + 1) // 2 + RS_MAX_SEGS + 1, 128), 128)
            dev.sync()
            dev.rs_heads = True
        if self.bm == BM:
            self.encode_maps()

    def encode_maps(self) -> None:
        """The tensor maps into `tma`, a set per operand pair of ML_MAPS whose buffers the pipeline has: P_A and P_B the
        peel's for the chain's pair, A and B again for the plain GEMMs' (which load no tail).  A caller's buffers without
        `tma` get the pipeline's own, freed with it: the maps hold these operands' addresses, so no other pipeline may share
        them."""
        d, b, m, n, k = self.dev, self.b, self.m, self.n, self.k
        for (x, y), slot in ML_MAPS.items():
            if x in b and y in b:
                if "tma" not in b:
                    b["tma"] = d.alloc(self.size["tma"])
                    self.allocated.append("tma")
                ab = d.tmap(b[x], k, m) + d.tmap(b[y], k, n)
                tail = d.tmap(b["pa_t"], 128, m) + d.tmap(b["pb_t"], 128, n) if slot == 0 else ab
                d.ck("cuMemcpyHtoD_v2", u64(b["tma"] + ML_SET * slot), C.c_char_p(ab + tail), ML_SET)

    @staticmethod
    def sizes(m: int, n: int, k: int, m_real: int | None = None) -> dict[str, int]:
        m_real = m if m_real is None else m_real
        tiles = cdiv(m_real, TILE) * (n // TILE)
        return {
            "ea_codes": m * 32, "eb_codes": n * 32, "fa_codes": k * 32, "faT32": 32 * k, "fb_codes": k * 32, "fbT32": 32 * k,
            "fbT128": 128 * k, "g": 128 * 32 * 4,
            "alpha_a": 4 * m, "beta_a": 4 * m, "ia_a": 4 * m, "aux_a": 8 * m, "epa_codes": m * 32,
            "alpha_b": 4 * n, "beta_b": 4 * n, "ia_b": 4 * n, "aux_b": 8 * n, "epb_codes": n * 32,
            "at": m * k, "bt": n * k, "at2": max(m, n) * k, "af": m * 32 * 4, "bf": n * 32 * 4, "pa_t": m * 128, "pb_t": n * 128,
            "c": m * n * 4, "c2": m * n * 4, "u": m * n * 4, "y": m * n * 2, "y2": m * n * 2, "tma": ML_SET * len(ML_MAPS),
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
        # read at launch: a ring's sets each hold their call's keys, and the lanes point `keys` at the set's before each call
        block = lambda: self.keys["unit"] + 96 if tag == "a" else 0  # noqa: E731
        return [(f"rowdig_{tag}", lambda: self.row_digests(b[x], b[f"rowdig_{tag}"], b.get(f"segdig_{tag}", 0), count)),
                (f"hash_rows_{tag}", lambda: Pipeline.hm_level0(self, tag, b[f"rowdig_{tag}"], count, dom, block(), nonce)),
                (f"tree_{tag}", lambda: self.levels(tag, count, dom))]

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

    def pd_commit(self, mode: int = 0) -> None:
        """A's commitment under pd (`FUSED`'s steps), pd_commit in mode 0: a CTA a real row, E_A's lines for all m rows (row
        i < m_real's under its own line key, the rest under line_a), its counter left zero.  Mode 1 (`rows_a`) stops after
        A's level 0 and the rows' lines, and `pd_tree` does the rest.  Every shape's on one kernel, the variant without
        PD_GROUP (a bit pd_commit doesn't read): the decode step's other commit kernel had its code fetched cold once a
        layer (979's at gate_up, 110.3 µs against 851's 98.3 at qkv, r20261024-104625-7b57)."""
        d, b, kk, v, s = self.dev, self.b, self.keys, self.pd & ~PD_GROUP, self.k // 256
        d.launch(f"pd_commit_{v}", self.m_real, 32 * (cdiv(s, 32) * 2 + 2), d.pd_smem[v],
                 p=(b["xa"], b["rowdig_a"], b["tree_a"], b["seeds_a"], b["lkeys_a"], kk["unit"], kk["dom_a"], kk["salt_key"],
                    kk["hm"], b["pd_count"], b["seed_a"], b["line_a"], b["ea_codes"], 0), i=(self.m_real, self.k, mode, self.m))

    def pd_tree(self) -> None:
        """`split_a`'s `tree_seed_a`: pd_tree, one CTA, A's levels above level 0, seed_A, line_a, and with `fillers` the
        padding rows' E_A lines under it (rows m_real..m - 1, where there are any); `pd_commit`'s variant."""
        d, b, kk, v = self.dev, self.b, self.keys, self.pd & ~PD_GROUP
        ea = b["ea_codes"] if self.m_real < self.m and self.fillers else 0
        d.launch(f"pd_tree_{v}", 1, 128, d.pd_tree_smem[v],
                 p=(0, 0, b["tree_a"], 0, 0, kk["unit"], kk["dom_a"], 0, 0, 0, b["seed_a"], b["line_a"], ea, 0),
                 i=(self.m_real, 0, 0, self.m))

    def pd_tiles(self) -> None:
        """The tile leaves under pd: pd_tiles, a CTA a tile (32 threads), or with PD_GROUP a CTA of PD_G tiles of a row block."""
        d, b, v, mr, n = self.dev, self.b, self.pd, self.m_real, self.n
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
        """`live`: the rows from 0 the product is wanted for (i4; 0 all M), by default m_real at 64 x 64 tiles (LIVE)."""
        tm, bn = tile = gemm_tile(name)
        if live is None:
            live = self.m_real if LIVE and tile == (BM_D, BM_D) and self.m_real < M else 0
        if name not in self.GEMM_SMEM:
            probe = self.dev.alloc(16)
            self.dev.launch(name + "_smem", 1, 1, p=(probe,))
            self.dev.sync()
            self.GEMM_SMEM[name] = int.from_bytes(self.dev.read(probe, 4), "little")
            self.dev.free(probe)
            self.dev.ck("cuFuncSetAttribute", self.dev.fn(name), MAX_DYN, self.GEMM_SMEM[name])
        tiles = (M // tm) * (N // bn)
        ptrs = [self.b[a], self.b[b], 0, 0, 0, 0, 0, 0, 0, dg, 0, key]
        for slot, x in zip((2, 3, 4), out):
            ptrs[slot] = self.b[x] if x else 0
        ptrs[5], ptrs[6], ptrs[7], ptrs[8] = pa, pb, rs, cs
        block = NT
        if tile == (BM, BM):
            ptrs[10], block = self.b["tma"] + ML_SET * ML_MAPS[(a, b)], ML_NT
        self.dev.launch(name, min(tiles, grid or self.dev.sms - self.beside), block, self.GEMM_SMEM[name], p=ptrs,
                        i=(M, N, self.k, RASTER, live))

    def peel(self) -> dict[str, int]:
        b = self.b
        return {"pa": b["pa_t"], "pb": b["pb_t"], "rs": b["ia_a"], "cs": b["ia_b"]}

    def extra(self) -> dict[str, object]:
        """The chain's launches by name, beside `steps`' choice: `gemm_pearlc_cu` (C~, U and y stored), and at 128 x 128 tiles
        `gemm_pearlc_s` (y alone: the call without hashing)."""
        m, n, sx = self.m, self.n, self.suffix
        tile = "_d" if self.bm == BM_D else ""
        stages = "" if DEEP or not tile else "4"
        out = {"gemm_pearlc_cu": lambda: self.gemm(f"gemm_pearlc{stages}{tile}{sx}", "at", "bt", m, n, out=("c", "u", "y"),
                                                   **self.peel())}
        if self.bm == BM:
            out["gemm_pearlc_s"] = lambda: self.gemm("gemm_pearlc_s" + sx, "at", "bt", m, n, out=(None, None, "y"), **self.peel())
        else:
            out.update(self.replaced())
        return out

    def replaced(self) -> dict[str, object]:
        """The kernels the split ones replace at decode, on the same buffers: `af_warps`, A'F_B on `words` (a warp a tile over
        all of K) where `af` runs words_k, `stats_a_warps`, stats_s5 (a warp a row) where `stats_a` runs stats_r, and
        `gemm_pearlc_full`, the chain over all m rows where `gemm_pearlc` runs on the real ones (LIVE), `gemm_pearlc_bk64`,
        the chain on 64-deep stages (gemm_pearlc4_d) where `gemm_pearlc` runs the 128-deep ones (DEEP), and
        `gemm_pearlc_prefetched`, the chain after B~'s prefetch in each form (`prefetch_b`), which writes nothing."""
        d, b, m, k, out = self.dev, self.b, self.m, self.k, {}
        if self.g and SPLIT:
            out["af_warps"] = lambda: d.launch("words", m // 16, 128, p=(b["at"], b["fbT32"], b["af"]), i=(m, 32, k))
        if self.stats_kernel(m) == "stats_r":
            out["stats_a_warps"] = lambda: self.stats("a", m, "xa", "stats_s5")
        chain = f"gemm_pearlc{'' if DEEP else '4'}_d{self.suffix}"
        if LIVE and self.m_real < m:
            out["gemm_pearlc_full"] = lambda: self.gemm(chain, "at", "bt", m, self.n, out=("c", "u", "y"), live=m, **self.peel())
        if DEEP:
            out["gemm_pearlc_bk64"] = lambda: self.gemm("gemm_pearlc4_d" + self.suffix, "at", "bt", m, self.n,
                                                       out=("c", "u", "y"), **self.peel())

        def prefetched():
            for f in PREFETCH_FORMS:
                self.prefetch_b(f)
            self.gemm(chain, "at", "bt", m, self.n, out=("c", "u", "y"), **self.peel())
        out["gemm_pearlc_prefetched"] = prefetched
        return out

    def prefetch_b(self, form: int | None = None, limit: int | None = None) -> None:
        """B~'s leading `limit` bytes (default PREFETCH_MB's, all n·k at 0) into L2 ahead of the chain, which reads them all:
        prefetch_l2 in `form` (default PREFETCH_FORM) on PREFETCH_CTAS blocks.  It writes nothing."""
        size = self.n * self.k
        limit = PREFETCH_MB << 20 if limit is None else limit
        size = min(size, limit) if limit else size
        self.dev.launch("prefetch_l2", PREFETCH_CTAS, 256, p=(self.b["bt"],),
                        i=(size - size % 128, PREFETCH_FORM if form is None else form))

    def real(self, x: str) -> int:
        """The bytes of buffer x's real rows (its first m_real of m), the only ones a consumer reads."""
        return self.size[x] * self.m_real // self.m

    def stats_kernel(self, rows: int) -> str:
        return "stats_r" if SPLIT and rows <= SR_ROWS and self.k <= SR_MAX_K else "stats_s5"

    def stats(self, tag: str, rows: int, x: str, kernel: str | None = None) -> None:
        """A side's row statistics (`stats_<tag>`) over its FP32 rows x: stats_s5, a warp a row, or for few rows stats_r, a
        block a row; the same words."""
        b, kernel = self.b, kernel or self.stats_kernel(rows)
        self.dev.launch(kernel, rows if kernel == "stats_r" else cdiv(rows, 8), 256,
                        p=(b[x], b[f"alpha_{tag}"], b[f"beta_{tag}"], b[f"ia_{tag}"], b[f"aux_{tag}"]), i=(rows, self.k))

    def words(self, a: str, b: str, M: int, out: str, warps: bool = False) -> None:
        """The FP32 words of an (M x K) by (32 x K) product: on 128 x 32 CTA tiles, or with `warps` (decode's few rows) on
        words_k, a cluster of WK_CL blocks of WK_WARPS warps a m16 x n8 tile, K split among them (unpromoted, on `words_u`,
        a warp a tile)."""
        if M % BM == 0 and not warps:
            self.gemm("gemm_words" + self.suffix, a, b, M, 32, out=(out,))
        elif self.g and SPLIT:
            self.dev.launch("words_k", M // 16 * 4 * WK_CL, 32 * WK_WARPS, p=(self.b[a], self.b[b], self.b[out]),
                            i=(M, 32, self.k))
        else:
            self.dev.launch("words" + self.suffix, M // 16, 128, p=(self.b[a], self.b[b], self.b[out]), i=(M, 32, self.k))

    def steps(self) -> list[tuple[str, str, object]]:
        d, b, m, n, k, sx = self.dev, self.b, self.m, self.n, self.k, self.suffix
        tile = "_d" if self.bm == BM_D else ""
        ct, chain = f"gemm_ct{tile}{sx}", self.extra()

        def lines(tag, key, side, factor, count, codes, t32=0, t128=0):
            norm = NORM_16 if factor == 1 else NORM_256
            return (f"lines_{tag}", lambda: d.launch("lines", cdiv(count, 128), 128, p=(key, b[codes], t32, t128),
                                                    i=(side, factor, count, norm)))

        def form_side(tag, rows, x, e_codes, f_codes, out):
            ep = f"ep{tag}_codes"
            return [(f"stats_{tag}", lambda: self.stats(tag, rows, x)),
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
        mr, kk, tiles = self.m_real, self.keys, self.tiles
        ea = ("lines_ea", lambda: d.launch("lines", cdiv(m, 128), 128, p=(b["line_a"], b["ea_codes"], 0, 0, b["lkeys_a"]),
                                           i=(0, 0, m, NORM_256, mr)))
        pd = self.pd is not None
        forming = [ea] * (not pd and not self.split_a) + form_side("a", m, "xa", "ea_codes", "fa_codes", "at")
        gemm = [("gemm_ct", lambda: self.gemm(ct, "at", "bt", m, n, out=("c2",)))]
        cleanup = [("af", lambda: self.words("at", "fbT32", m, "af", warps=self.bm == BM_D)),
                   ("pa", lambda: d.launch("pa", cdiv(m * 64, 256), 256, p=(b["epa_codes"], 0, b["af"], b["pa_t"]), i=(m,))),
                   ("gemm_pearlc", chain["gemm_pearlc_cu"])]
        epoch = self.tree_setup("b", n, "dom_b") + self.row_tree("b", "xb", n, "dom_b")
        unit = self.tree_setup("a", mr, "dom_a") * (not pd and not self.split_a) + self.tree_setup("t", tiles, "dom_t")
        split = [("rows_a", lambda: self.pd_commit(1)), ("tree_seed_a", self.pd_tree)]
        rows, level0, levels = self.row_tree("a", "xa", mr, "dom_a")
        seed_a = ("seed_a", lambda: d.launch("s512_seeds", 1, 128, p=(self.root("a"), b["seed_a"], b["line_a"], kk["unit"]),
                                             i=(1, 0, 0)))
        seeds_a = ("seeds_a", lambda: d.launch("s512_seeds", cdiv(mr, 128), 128,
                                               p=(b["rowdig_a"], b["seeds_a"], b["lkeys_a"], kk["unit"]), i=(mr, 0, 1)))
        own = ("lines_ea", lambda: d.launch("lines", cdiv(mr, 128), 128, p=(0, b["ea_codes"], 0, 0, b["lkeys_a"]),
                                            i=(0, 0, mr, NORM_256, mr)))
        generic = [("rows_a", lambda: [f() for _, f in self.tree_setup("a", mr, "dom_a") + [rows, seeds_a, own]]),
                   ("tree_seed_a", lambda: [f() for _, f in [level0, levels, seed_a] + [ea] * (self.fillers and mr < m)])]
        commit = ((split if self.split_a else [("commit_a", self.pd_commit)]) if pd else generic if self.split_a
                  else [rows, level0, levels, seed_a, seeds_a])
        hashing = [("hash_leaf", self.pd_tiles if pd else self.tile_leaves),
                   ("tree_leaves_t", lambda: self.tree_leaves("t", b["leaves"], tiles, "dom_t")),
                   ("tree_t", lambda: self.levels("t", tiles))]
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
        out = (tag("epoch", epoch) + tag("unit", unit) + tag("job", job) + tag("commit", commit) + tag("forming", forming)
               + tag("gemm", gemm) + tag("cleanup", cleanup) + tag("hashing", hashing))
        if m % BM == 0:
            out += tag("plain_job", plain_job) + tag("plain", plain) + tag("adopted_job", adopted_job) + tag("adopted", adopted)
        return out

    def run_all(self) -> None:
        for _, _, f in self.steps():
            f()
        self.dev.sync()

    def diff(self, x: str, y: str) -> int:
        return self.diff_at(self.b[x], self.b[y], self.size[x])

    def diff_at(self, x: int, y: int, n: int) -> int:
        """Words differing between n bytes at device addresses x and y."""
        words = n // 4
        self.dev.ck("cuMemsetD8_v2", u64(self.b["diff"]), 0, 8)
        self.dev.launch("diff_words", cdiv(words, 256), 256, p=(x, y, self.b["diff"]), i=(words,))
        self.dev.sync()
        return int.from_bytes(self.dev.read(self.b["diff"], 8), "little")

    def splits(self) -> dict[str, int]:
        """Each split kernel against the one it replaces (SPLITS), on this pipeline's inputs: the replaced step's outputs
        copied into c2, the split step run, the words differing per output in its real rows (`real`)."""
        steps, alt, out = {nm: f for nm, _, f in self.steps()}, self.extra(), {}
        for step, ref, outs in SPLITS:
            if ref not in alt:
                continue
            alt[ref]()
            at = 0
            for o in outs:
                self.dev.copy(self.b["c2"] + at, self.b[o], self.size[o])
                at += self.size[o]
            steps[step]()
            at = 0
            for o in outs:
                out[f"{o}_{step}_vs_{ref}_words_differing"] = self.diff_at(self.b[o], self.b["c2"] + at, self.real(o))
                at += self.size[o]
        return out


# ---- the check ----------------------------------------------------------------------------------------------------

PLAIN_BUFFERS = {"qa", "qb", "sa", "sb", "y_plain", "y_fast", "qa_s5", "qb_s5", "y_adopt"}
#: form_s5's gate (the one-code cast) and its 144-byte-stride twin, each run into at2 and held to form_s5's codes
FORM_ALTS = ("form_s5_scalar", "form_s5_r144")
GEMM_RERUNS = (("gemm_pearlc", ("c", "u", "y")), ("gemm_ct", ("c2",)), ("gemm_plain", ("y_plain",)), ("gemm_fast", ("y_fast",)),
               ("af", ("af",)), ("bf", ("bf",)), ("gemm_adopt", ("y_adopt",)), ("gemm_pearlc_s", ("y",)))
ALIASES = {"c2": "c"}
STATS_A = ("alpha_a", "beta_a", "ia_a", "aux_a")
#: the 64 x 64 pipeline's: the chain, A'F_B on words_k, and the kernels the split ones replace (`Pipeline.replaced`)
SMALL_RERUNS = GEMM_RERUNS[:2] + (("af", ("af",)), ("af_warps", ("af",)), ("stats_a_warps", STATS_A),
                                  ("gemm_pearlc_bk64", ("c", "u", "y")), ("gemm_pearlc_prefetched", ("c", "u", "y")))
#: each split kernel's step at decode, the step it replaces (`Pipeline.replaced`) and their outputs, held word for word (the
#: chain's one output at a time: c2, where the replaced step's land, holds one m x n FP32 product)
SPLITS = (("af", "af_warps", ("af",)), ("stats_a", "stats_a_warps", STATS_A),
          *(("gemm_pearlc", alt, (o,)) for alt in ("gemm_pearlc_full", "gemm_pearlc_bk64") for o in ("c", "u", "y")))


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
    then `hash_leaf`; also `split_a`'s `rows_a` and `tree_seed_a` in its place, ``/split``) after every output in
    PD_OUTPUTS is poisoned, against the fixture's bytes; the counter left zero.  The no-write control, the poison with no
    launch, must fail every output."""
    count, res = dev.alloc(16), {}
    make = lambda split: Pipeline(dev, pipe.m, pipe.n, pipe.k, pipe.variant, pipe.keys, pipe.b["xa"], pipe.b["xb"],  # noqa: E731
                                  bufs={**pipe.b, "pd_count": count}, m_real=pipe.m_real, bm=BM_D, weight=pipe.weight,
                                  split_a=split)
    whole = make(False)
    if whole.pd is None:
        dev.free(count)
        return {"pd@domain": {"ok": False, "why": f"{(pipe.m_real, pipe.n, pipe.k)} is outside pd's domain"}}
    split = make(True)
    forms = [(None, None, "", ())] + [(whole, v, "", ("commit_a", "hash_leaf")) for v in PD_SERVED] + [
        (split, v, "/split", ("rows_a", "tree_seed_a", "hash_leaf")) for v in PD_SERVED]
    for p_, v, tag, names in forms:
        for x in PD_OUTPUTS:
            dev.ck("cuMemsetD8_v2", u64(pipe.b[x]), POISON, pipe.size[x])
        if p_ is not None:
            p_.pd = v
            steps = {nm: fn for nm, _, fn in p_.steps()}
            for nm in names:
                steps[nm]()
        dev.sync()
        for x in PD_OUTPUTS:
            r = compare(dev, pipe, x, want(x))
            if p_ is None:
                res[f"{x}@pd/no-write"] = {"ok": not r["ok"], "bytes": r["bytes"]}
            else:
                res[f"{x}@pd_{v}{tag}"] = r
        if p_ is not None:
            res[f"pd_count@pd_{v}{tag}"] = {"ok": dev.read(count, 16) == bytes(16), "bytes": 16}
    dev.free(count)
    return res


def check_split(dev: Dev, pipe: Pipeline, want) -> dict:
    """`split_a` outside pd (`Pipeline.FUSED_GENERIC`) on `pipe`'s buffers: ``rows_a`` then ``tree_seed_a`` after every
    output of A's commitment (PD_OUTPUTS but the leaves) is poisoned, against the fixture's bytes (``@split``)."""
    split = Pipeline(dev, pipe.m, pipe.n, pipe.k, pipe.variant, pipe.keys, pipe.b["xa"], pipe.b["xb"], bufs=pipe.b,
                     m_real=pipe.m_real, bm=BM_D, weight=pipe.weight, split_a=True)
    outs = [x for x in PD_OUTPUTS if x != "leaves"]
    for x in outs:
        dev.ck("cuMemsetD8_v2", u64(pipe.b[x]), POISON, pipe.size[x])
    steps = {nm: fn for nm, _, fn in split.steps()}
    for nm in ("rows_a", "tree_seed_a"):
        steps[nm]()
    dev.sync()
    return {f"{x}@split": compare(dev, pipe, x, want(x)) for x in outs}


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
        res.update(check_split(dev, pipe, want))
        for tag in ("a", "b"):
            for kernel in FORM_ALTS:
                pipe.form_alt(tag, kernel)
                res[f"{tag}t@{kernel}"] = compare(dev, pipe, f"{tag}t", want(f"{tag}t"), alias="at2")
        small = Pipeline(dev, m, n, k, variant, keys, xa, xb, bufs=pipe.b, bm=BM_D, weight=man["weight"])
        for p_, grids, tag in ((pipe, (None, 1, 3), ""), (small, (None, 3), "bm64")):
            steps = {**p_.extra(), **{nm: fn for nm, _, fn in p_.steps()}}
            for grid in grids:
                for step, outs in SMALL_RERUNS if p_ is small else GEMM_RERUNS:
                    if step not in steps:
                        continue
                    for o in outs:
                        dev.ck("cuMemsetD8_v2", u64(pipe.b[o]), 0xAB, pipe.size[o])
                    real = p_.gemm

                    def pinned(*a, _real=real, _grid=grid, **kw):
                        kw["grid"] = _grid
                        return _real(*a, **kw)
                    p_.gemm = pinned
                    steps[step]()
                    p_.gemm = real
                    dev.sync()
                    for o in outs:
                        res[f"{o}@{step}/{tag or 'bm128'}/grid{grid or 'all'}"] = compare(dev, pipe, o, want(ALIASES.get(o, o)))
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
    pipe.free()
    for p_ in (xa, xb, *keys.values()):
        dev.free(p_)
    return out


#: form_body's RATE kernels (pearl_c_sm120.cu): form_s5's loop, form_s5_scalar's, form_s5's with the cast replaced by a
#: float's bits, and its atoms, F's loads and stores alone; form_s5's loop and its castless one again with the codes staged at
#: a 144-byte row stride (form_s5_r144's); each run at FORM_REPEATS repeats per block
FORM_RATES = ("form_rate", "form_rate_scalar", "form_rate_nocast", "form_rate_atoms", "form_rate_r144", "form_rate_nocast_r144")
FORM_REPEATS = (64, 2112)
FORM_BLOCKS_PER_SM = 2      # form_s5's occupancy: 45,056 bytes of static shared memory a block, of the SM's 100 KB
MACS_PER_SM_CLOCK = 1024    # one unit, a dense E4M3 mma.sync MAC: m16n8k32 is 4,096 MACs per 4 SM clocks


def form_rates(dev: Dev, nv: Nvml, pipe: Pipeline, reps: int) -> dict:
    """The forming loop's price per code in the mix it runs in, in units (one dense E4M3 MAC): each RATE kernel on B's side,
    2 blocks per SM of 64 x 128 codes each, timed at both FORM_REPEATS; the slope over the repeats' codes at the NVML SM
    clock.  form_rate and form_rate_scalar must write bt's codes on their tiles (the gate); the cast's price in the mix is
    form_rate's less form_rate_nocast's."""
    b, n, k = pipe.b, pipe.n, pipe.k
    blocks, per_row = FORM_BLOCKS_PER_SM * dev.sms, k // 128
    if blocks > (n // 64) * per_row:
        return {"skipped": f"{blocks} blocks need more than B's {(n // 64) * per_row} tiles"}
    full, rem = divmod(blocks, per_row)
    span = 64 * (full + (rem > 0)) * k
    want = dev.read(b["bt"], span)
    out: dict = {"blocks": blocks, "repeats": list(FORM_REPEATS), "codes_per_repeat": blocks * 64 * 128}
    for name in FORM_RATES:
        def launch(r, name=name):
            dev.launch(name, blocks, 128, p=(b["xb"], b["alpha_b"], b["epb_codes"], b["fb_codes"], b["at2"]), i=(n, k, r))
        row: dict = {}
        if name in ("form_rate", "form_rate_scalar", "form_rate_r144"):
            dev.ck("cuMemsetD8_v2", u64(b["at2"]), 0xAB, span)
            launch(3)
            dev.sync()
            got = dev.read(b["at2"], span)
            spans = [(0, 64 * full * k)] + [(64 * full * k + r * k, 64 * full * k + r * k + 128 * rem) for r in range(64 * (rem > 0))]
            row["codes_differing"] = sum(x != y for s, e in spans for x, y in zip(got[s:e], want[s:e]))
        before = nv.read()
        t = {r: dev.timed(lambda r=r: launch(r), reps) for r in FORM_REPEATS}
        after = nv.read()
        mhz = [c["sm_mhz"] for c in (before, after) if "sm_mhz" in c]
        lo, hi = (statistics.median(t[r]) for r in FORM_REPEATS)
        row.update({"ms": {str(r): [round(x, 5) for x in t[r]] for r in FORM_REPEATS}, "sm_mhz": mhz})
        if mhz and hi > lo:
            codes = out["codes_per_repeat"] * (FORM_REPEATS[1] - FORM_REPEATS[0])
            row["units_per_code"] = round((hi - lo) * 1e-3 * statistics.mean(mhz) * 1e6 * dev.sms * MACS_PER_SM_CLOCK / codes, 3)
        out[name] = row
    u_ = {nm: out[nm].get("units_per_code") for nm in FORM_RATES}
    if u_["form_rate"] is not None and u_["form_rate_nocast"] is not None:
        out["cast_units_per_code"] = {"packed": round(u_["form_rate"] - u_["form_rate_nocast"], 3)}
        if u_["form_rate_scalar"] is not None:
            out["cast_units_per_code"]["scalar_loop_less_nocast"] = round(u_["form_rate_scalar"] - u_["form_rate_nocast"], 3)
    if u_["form_rate_r144"] is not None and u_["form_rate_nocast_r144"] is not None:
        out.setdefault("cast_units_per_code", {})["packed_r144"] = round(u_["form_rate_r144"] - u_["form_rate_nocast_r144"], 3)
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
                 "cleanup": med["af"] + med["pa"] + chain - med["gemm_ct"], "hashing": phases["hashing"]}
        out.update({"split_ms": {kk: round(v, 4) for kk, v in split.items()}, "call_sum_ms": round(sum(split.values()), 4),
                    "call_e2e_ms": round(e2e, 4), "call_e2e_no_hash_ms": round(e2e_nh, 4), "job_ms": round(phases["job"], 4),
                    "pearlc_tflops": round(2 * m * n * k / chain / 1e9, 1)})
        extra["gemm_pearlc_cu"]()
        out["consistency"] = {"c_ct_vs_pearlc_words_differing": pipe.diff_at(pipe.b["c2"], pipe.b["c"], pipe.real("c"))}
        dev.copy(pipe.b["c2"], pipe.b["u"], pipe.size["u"])
        extra["gemm_pearlc_cu"]()
        out["consistency"]["u_rerun_words_differing"] = pipe.diff_at(pipe.b["c2"], pipe.b["u"], pipe.real("u"))
        dev.copy(pipe.b["y2"], pipe.b["y"], pipe.size["y"])
        if y_only:
            y_only()
            out["consistency"]["y_s_vs_pearlc_words_differing"] = pipe.diff("y", "y2")
        out["consistency"].update({f"{tag}t_form_s5_vs_{kernel.removeprefix('form_s5_')}_words_differing": pipe.form_alt(tag, kernel)
                                   for tag in ("a", "b") for kernel in FORM_ALTS})
        out["consistency"].update(pipe.splits())
        out["form_rate"] = rate = form_rates(dev, nv, pipe, reps)
        out["consistency"].update({f"bt_{nm}_codes_differing": rate[nm]["codes_differing"]
                                   for nm in ("form_rate", "form_rate_scalar", "form_rate_r144") if nm in rate})
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
