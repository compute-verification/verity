#!/usr/bin/env python3
"""Pearl-C on one RTX PRO 6000 Blackwell (sm_120) from the prebuilt pearl_c_sm120.cubin, through the CUDA driver API
(ctypes on libcuda.so.1, stdlib only, nothing built on the card).

0. Identity, before anything runs: exactly one visible device (CUDA_VISIBLE_DEVICES a single index), compute capability
   12.0, an "NVIDIA RTX PRO 6000 Blackwell" name, and its UUID equal to PEARLC_EXPECT_UUID (server.md's table).  NVML is
   addressed by that UUID, never by index (NVML ignores CUDA_VISIBLE_DEVICES).  A mismatch exits 3 before any launch.
1. The check.  At check.json's shape (fixture.py's), every kernel runs on the fixture's inputs and every buffer it writes is
   compared byte for byte with the fixture, for each variant (`sm120`, G = 4, and `sm120-unpromoted`), hashing format -h3
   (SHA-512 throughout); the GEMMs, with gemm_pearlc_s (y alone), are re-run on all, 1 and 3 CTAs and compared again
   (`<buffer>@<step>/<tile>/grid<n>`), and the chain's 64 x 64 kernels (decode's) too.  A variant is timed only if all of its
   buffers match (C~ and U bit-exact against the CPU reference), the plain baselines only if theirs do.  Exit 4 if any check
   fails.
2. Diagnostic timing (not the headline: GPU 6's harness times the arm against autotuned baselines), at each of SHAPES
   (default 8192, m = n = k; or MxNxK, m padded to 64 rows): each kernel REPS times after a warm-up, its median, the phases
   (the weight side; the call's commitment of A, forming, GEMM, clean-up and hashing), the call end to end with and without
   its hashes (commit and hashing; without, the chain is gemm_pearlc_s where it runs), and consistency checks (C~ of gemm_ct
   equals gemm_pearlc's; a second gemm_pearlc gives the same U; gemm_pearlc_s's y equals gemm_pearlc's; form_s5_scalar and
   form_s5_r144, timed beside form_s5, and the RATE kernels write form_s5's codes).
   `form_rates` prices the forming loop per code in the mix it runs in, in units of one dense E4M3 MAC, and the packed cast
   within it (form_rate less form_rate_nocast; at the 144-byte row stride, form_rate_r144 less form_rate_nocast_r144).
   form_s5's codes and the lines' transposed F lines are in K32 order (pearl_c_sm120.cu); the check's fixture holds them in
   it, and every product over k against the unpermuted operands'.  The timing's keys and domain ids are stand-ins of the same sizes (diagnostic
   only; the arm's calls run on a real epoch).  NVML's clocks, power and throttle reasons before and after.

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
NT, BM, BM_D, RASTER = 256, 128, 64, 8
ML_NT = 384                  # mainloop_sm120.cuh's block: a TMA producer warpgroup and 8 consumer warps
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
    "cuFuncSetAttribute": [vp, i32, i32], "cuMemAlloc_v2": [P(u64), sz], "cuMemFree_v2": [u64],
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

    def copy_async(self, dst: int, src: int, n: int, stream=None) -> None:
        self.ck("cuMemcpyDtoDAsync_v2", u64(dst), u64(src), n, self.stream if stream is None else stream)

    def launch(self, name: str, grid, block: int, smem: int = 0, p=(), i=(), stream=None) -> None:
        a = Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        g = tuple(grid) + (1,) * (3 - len(grid)) if isinstance(grid, (tuple, list)) else (grid, 1, 1)
        params = (vp * 1)(C.addressof(a))
        self.ck("cuLaunchKernel", self.fn(name), g[0], g[1], g[2], block, 1, 1, smem, self.stream if stream is None else stream,
                params, None)

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
    baselines' `plain_job`, `plain`, `adopted_job`, `adopted` (at m a multiple of 128 only).  The chain runs on 128 x 128
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

    def __init__(self, dev: Dev, m: int, n: int, k: int, variant: str, keys: dict[str, int], xa: int, xb: int,
                 bufs: dict[str, int] | None = None, m_real: int | None = None, bm: int | None = None, weight: int = 0,
                 sha: str | None = None):
        """`bufs` names the buffers when the caller owns them (only those its steps touch); `m_real` rows of A are the
        matmul's, committed and hashed, the rest filler that only the products see."""
        self.dev, self.m, self.n, self.k, self.variant, self.keys = dev, m, n, k, variant, keys
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
        d, b = self.dev, self.b
        nonce = (0, 0) if tag == "a" else nonce_args(HM96_WEIGHTS + self.weight)
        block = self.keys["unit"] + 96 if tag == "a" else 0
        return [(f"rowdig_{tag}", lambda: self.row_digests(b[x], b[f"rowdig_{tag}"], b.get(f"segdig_{tag}", 0), count)),
                (f"hash_rows_{tag}", lambda: Pipeline.hm_level0(self, tag, b[f"rowdig_{tag}"], count, dom, block, nonce)),
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
             dg=0, key=0):
        if name not in self.GEMM_SMEM:
            probe = self.dev.alloc(16)
            self.dev.launch(name + "_smem", 1, 1, p=(probe,))
            self.dev.sync()
            self.GEMM_SMEM[name] = int.from_bytes(self.dev.read(probe, 4), "little")
            self.dev.free(probe)
            self.dev.ck("cuFuncSetAttribute", self.dev.fn(name), MAX_DYN, self.GEMM_SMEM[name])
        tm, bn = gemm_tile(name)
        tiles = (M // tm) * (N // bn)
        ptrs = [self.b[a], self.b[b], 0, 0, 0, 0, 0, 0, 0, dg, 0, key]
        for slot, x in zip((2, 3, 4), out):
            ptrs[slot] = self.b[x] if x else 0
        ptrs[5], ptrs[6], ptrs[7], ptrs[8] = pa, pb, rs, cs
        block = NT
        if is_ml(name):
            ptrs[10], block = self.b["tma"] + ML_SET * ML_MAPS[(a, b)], ML_NT
        self.dev.launch(name, min(tiles, grid or self.dev.sms), block, self.GEMM_SMEM[name], p=ptrs, i=(M, N, self.k, RASTER))

    def extra(self) -> dict[str, object]:
        """The chain's launches by name, beside `steps`' choice: `gemm_pearlc_cu` (C~, U and y stored), and at 128 x 128 tiles
        `gemm_pearlc_s` (y alone: the call without hashing)."""
        b, m, n, sx = self.b, self.m, self.n, self.suffix
        tile = "_d" if self.bm == BM_D else ""
        peel = lambda: {"pa": b["pa_t"], "pb": b["pb_t"], "rs": b["ia_a"], "cs": b["ia_b"]}  # noqa: E731
        out = {"gemm_pearlc_cu": lambda: self.gemm(f"gemm_pearlc{tile}{sx}", "at", "bt", m, n, out=("c", "u", "y"), **peel())}
        if self.bm == BM:
            out["gemm_pearlc_s"] = lambda: self.gemm("gemm_pearlc_s" + sx, "at", "bt", m, n, out=(None, None, "y"), **peel())
        return out

    def words(self, a: str, b: str, M: int, out: str, warps: bool = False) -> None:
        """The FP32 words of an (M x K) by (32 x K) product: on 128 x 32 CTA tiles, or with `warps` one warp per m16 x n8
        tile, a block per m16 row."""
        if M % BM == 0 and not warps:
            self.gemm("gemm_words" + self.suffix, a, b, M, 32, out=(out,))
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
               ("bf", lambda: self.words("bt", "faT32", n, "bf")),
               ("g", lambda: self.words("fbT128", "faT32", 128, "g")),
               ("pb", lambda: d.launch("pb", cdiv(n * 32, 256), 256, p=(b["epb_codes"], 0, b["bf"], b["g"], b["pb_t"]), i=(n,)))]
        mr, kk, tiles = self.m_real, self.keys, self.tiles
        ea = ("lines_ea", lambda: d.launch("lines", cdiv(m, 128), 128, p=(b["line_a"], b["ea_codes"], 0, 0, b["lkeys_a"]),
                                           i=(0, 0, m, NORM_256, mr)))
        forming = [ea, *form_side("a", m, "xa", "ea_codes", "fa_codes", "at")]
        gemm = [("gemm_ct", lambda: self.gemm(ct, "at", "bt", m, n, out=("c2",)))]
        cleanup = [("af", lambda: self.words("at", "fbT32", m, "af", warps=self.bm == BM_D)),
                   ("pa", lambda: d.launch("pa", cdiv(m * 64, 256), 256, p=(b["epa_codes"], 0, b["af"], b["pa_t"]), i=(m,))),
                   ("gemm_pearlc", chain["gemm_pearlc_cu"])]
        epoch = self.tree_setup("b", n, "dom_b") + self.row_tree("b", "xb", n, "dom_b")
        unit = self.tree_setup("a", mr, "dom_a") + self.tree_setup("t", tiles, "dom_t")
        commit = self.row_tree("a", "xa", mr, "dom_a") + [
            ("seed_a", lambda: d.launch("s512_seeds", 1, 128, p=(self.root("a"), b["seed_a"], b["line_a"], kk["unit"]), i=(1, 0, 0))),
            ("seeds_a", lambda: d.launch("s512_seeds", cdiv(mr, 128), 128, p=(b["rowdig_a"], b["seeds_a"], b["lkeys_a"], kk["unit"]),
                                         i=(mr, 0, 1)))]
        hashing = [("hash_leaf", self.tile_leaves),
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
        words = self.size[x] // 4
        self.dev.ck("cuMemsetD8_v2", u64(self.b["diff"]), 0, 8)
        self.dev.launch("diff_words", cdiv(words, 256), 256, p=(self.b[x], self.b[y], self.b["diff"]), i=(words,))
        self.dev.sync()
        return int.from_bytes(self.dev.read(self.b["diff"], 8), "little")


# ---- the check ----------------------------------------------------------------------------------------------------

PLAIN_BUFFERS = {"qa", "qb", "sa", "sb", "y_plain", "y_fast", "qa_s5", "qb_s5", "y_adopt"}
#: form_s5's gate (the one-code cast) and its 144-byte-stride twin, each run into at2 and held to form_s5's codes
FORM_ALTS = ("form_s5_scalar", "form_s5_r144")
GEMM_RERUNS = (("gemm_pearlc", ("c", "u", "y")), ("gemm_ct", ("c2",)), ("gemm_plain", ("y_plain",)), ("gemm_fast", ("y_fast",)),
               ("af", ("af",)), ("bf", ("bf",)), ("gemm_adopt", ("y_adopt",)), ("gemm_pearlc_s", ("y",)))
ALIASES = {"c2": "c"}
#: the 64 x 64 pipeline's: the chain, and A'F_B on the words kernel
SMALL_RERUNS = GEMM_RERUNS[:2] + (("af", ("af",)),)


def compare(dev: Dev, pipe: Pipeline, name: str, want: bytes, alias: str | None = None) -> dict:
    got = dev.read(pipe.b[alias or name], len(want))
    if got == want:
        return {"ok": True, "bytes": len(want)}
    bad = [i for i in range(0, len(want), 4) if got[i:i + 4] != want[i:i + 4]]
    return {"ok": False, "bytes": len(want), "words_differing": len(bad),
            "first": [{"offset": i, "got": got[i:i + 4].hex(), "want": want[i:i + 4].hex()} for i in bad[:6]]}


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
