"""fp4_emulation_gpu: build, SASS-gate, check and time the NVFP4-atom emulation routes of ``fp4_emulation.cu`` on sm_120a.

    python benchmarks/pouw/kernels/fp4_emulation_gpu.py build     # no GPU: nvcc, the SASS gate, and a CPU dry run of the packing
    python benchmarks/pouw/kernels/fp4_emulation_gpu.py run [--expect-uuid GPU-...] [--family gaussian] [--seconds 0.3] [--reps 5]

The routes (``fp4_emulation.cu``'s header has the table): native and native_hot (the OMMA atom, from +0 and from a hot
accumulator H), imma_exact (IMMA group dots and the atom's align-add on the CUDA cores), imma_fast (IMMA and an exact int64
accumulator), imma_floor (the hot tweak's per-atom floor), bf16_fold and bf16_floor (E2M1 x UE4M3 folded into BF16 on HMMA),
dp4a and lut (CUDA cores only).  ``build`` also dry-runs every integer route's arithmetic in numpy from the packed words, so the
constants the kernels read (E0, the scale factors, H) are checked before the card is.

``run`` needs the card.  Per route it
  1. checks words: short launches (``--check-atoms`` atoms per chain) on every warp.  Each route's words must equal its own
     semantics replayed on the CPU (``impl_mismatches`` = 0, the kernel-correctness gate); for native and native_hot that
     semantics is the pinned ``BLACKWELL_SM120_NVF4`` itself (the card-equals-model gate).  ``matches_native`` is the share of
     words equal to the atom's (native's, or native_hot's for the hot routes): the emulation question.  Every chain repeats
     one 64-wide atom, so its accumulator grows coherently and truncates sooner than a real row's: read ``matches_native``
     at the short check length, and the CPU study (``fp4_emulation.py routes``) for real-length chains;
  2. times a one-wave launch of about ``--seconds``, ``--reps`` times: logical NVFP4 MACs per second and per SM per clock, and
     A = route rate / native rate (A < 1: slower than the native atom).

Under ``research run`` (``$RESEARCH_RUN_DIR`` set) it writes a ``research/result/v0.1`` document.  Operands are
register-resident, so every rate is an upper bound for a GEMM built on that route.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_result  # noqa: E402  benchmarks/pouw/run_result.py
from fp4_emulation import (  # noqa: E402  benchmarks/pouw/fp4_emulation.py
    Problem, family, halves, quantize_pearl_c4, ue4m3_dyadic, floor_log2, f32_rz, atoms,
)

SOURCE = HERE / "fp4_emulation.cu"
ARCH = "sm_120a"
ROUTES = ("native", "native_hot", "imma_exact", "imma_fast", "imma_floor", "bf16_fold", "bf16_floor", "dp4a", "lut")
HOT = {"native_hot", "imma_floor", "bf16_floor"}
PEAKS = ("omma", "imma16", "hmma16", "dp4a")
#: (NSETS = NACC, words per set) per route, as ``Shape<R>`` in the kernel (``emu_route_shape`` checks it on the card).
SHAPE = {"native": (8, 8), "native_hot": (8, 8), "imma_exact": (2, 32), "imma_fast": (4, 28), "imma_floor": (4, 28),
         "bf16_fold": (4, 24), "bf16_floor": (4, 24), "dp4a": (2, 80), "lut": (1, 48)}
#: The SASS gate: (opcode prefix, count per kernel) -- one atom per chain per loop body, nothing merged or hoisted.
SASS_EXPECT = {"native": ("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X", 8), "native_hot": ("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X", 8),
               "imma_exact": ("IMMA.16816.S8.S8", 16), "imma_fast": ("IMMA.16816.S8.S8", 16),
               "imma_floor": ("IMMA.16816.S8.S8", 16), "bf16_fold": ("HMMA.16816.F32.BF16", 16),
               "bf16_floor": ("HMMA.16816.F32.BF16", 16), "dp4a": ("IDP.4A.S8.S8", 128), "lut": ("LDS", 256)}
PEAK_EXPECT = {"omma": ("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X", 32), "imma16": ("IMMA.16816.S8.S8", 32),
               "hmma16": ("HMMA.16816.F32.BF16", 32), "dp4a": ("IDP.4A.S8.S8", 128)}
#: MACs per warp-wide instruction of each peak: OMMA m16n8k64, IMMA / HMMA m16n8k16, IDP4A 4 per lane x 32 lanes.
PEAK_MACS = {"omma": 16 * 8 * 64, "imma16": 16 * 8 * 16, "hmma16": 16 * 8 * 16, "dp4a": 32 * 4}
TILES = 16          # distinct 16 x 8 tiles; warp w reads tile w % TILES
HOT_BITS = 12       # the tweak's calibration: the floor grid at 2^-12 of the atom's sum
LANE = np.arange(32)
G, T = LANE >> 2, LANE & 3


# ---- operands ---------------------------------------------------------------------------------------------------------

def make_pool(fam: str, seed: int, delta: float = 0.25):
    """TILES tiles of 8 atoms each, cut from Pearl-C4 operand rows of k = 8,192 (so alpha0 and the scales are a long row's)."""

    rng = np.random.default_rng(seed)
    ca, sa = quantize_pearl_c4(family(fam, "A", 16 * TILES, 8192, rng), delta, rng)
    cb, sb = quantize_pearl_c4(family(fam, "B", 8 * TILES, 8192, rng), delta, rng)
    k = 64 * 8
    return (ca[:, :k].reshape(TILES, 16, k), sa[:, :k // 16].reshape(TILES, 16, k // 16),
            cb[:, :k].reshape(TILES, 8, k), sb[:, :k // 16].reshape(TILES, 8, k // 16))


def _pack(vals: np.ndarray, bits: int) -> np.ndarray:
    """Pack the last axis (32 / bits fields, lowest first) into uint32."""

    n = 32 // bits
    v = (vals.astype(np.int64) & ((1 << bits) - 1)).reshape(*vals.shape[:-1], -1, n)
    return (v << (bits * np.arange(n))).sum(axis=-1).astype(np.uint32)


def _bf16(h: np.ndarray, ma: np.ndarray, ea: np.ndarray) -> np.ndarray:
    """BF16 bits of (h / 2) * ma * 2^ea, exact (at most 6 significant bits)."""

    v = np.abs(h) * ma
    fl = floor_log2(np.maximum(v, 1))
    frac = np.where(fl <= 7, (v << np.clip(7 - fl, 0, 7)) & 0x7F, 0)
    bits = ((fl + ea - 1 + 127) << 7) | frac
    return np.where(v == 0, 0, bits | np.where(h < 0, 1 << 15, 0)).astype(np.int64)


def lane_constants(route: str, pool, nsets: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per tile and lane: E0 of the 4 words, the scale factors' shifts, and H (bits) for the hot routes."""

    A, SA, B, SB = pool
    _, ea = ue4m3_dyadic(SA[:, :, :4 * nsets])
    _, eb = ue4m3_dyadic(SB[:, :, :4 * nsets])
    rows = G[:, None] + 8 * (np.arange(4)[None, :] >> 1)          # [32, 4]: the row of word o
    cols = 2 * T[:, None] + (np.arange(4)[None, :] & 1)
    emin_a, emin_b = ea.min(axis=2), eb.min(axis=2)                # [tiles, 16], [tiles, 8]
    e0 = emin_a[:, rows] + emin_b[:, cols] - 2                     # [tiles, 32, 4]
    h = np.zeros(e0.shape, np.int64)
    if route in HOT:
        sums = []
        for c in range(nsets):
            _, mant, x, _ = atoms(A[:, :, 64 * c:64 * c + 64].reshape(-1, 64), SA[:, :, 4 * c:4 * c + 4].reshape(-1, 4),
                                  B[:, :, 64 * c:64 * c + 64].reshape(-1, 64), SB[:, :, 4 * c:4 * c + 4].reshape(-1, 4))
            s = (mant.astype(np.float64) * np.exp2(x)).sum(axis=0)          # [16 tiles, 8 tiles] -- keep the diagonal blocks
            s = s.reshape(TILES, 16, TILES, 8)[np.arange(TILES), :, np.arange(TILES), :]
            sums.append(np.abs(s))
        big = np.max(np.stack(sums), axis=0)[:, rows, cols]       # [tiles, 32, 4]: the largest atom keeps a check in H's binade
        eh = np.floor(np.log2(np.maximum(big, 2.0 ** -60))).astype(np.int64) + 23 - HOT_BITS
        eh = np.maximum(eh, e0 + 23)
        h = ((eh + 127) << 23) | (1 << 22)
    return e0, emin_a, emin_b, h


def pack(route: str, pool) -> tuple[np.ndarray, np.ndarray]:
    """Per tile: the route's fragments [tiles, 32, NSETS, WORDS] and lane constants [tiles, 32, 8]."""

    A, SA, B, SB = pool
    nsets, words = SHAPE[route]
    e0, emin_a, emin_b, h = lane_constants(route, pool, nsets)
    frags = np.zeros((TILES, 32, nsets, words), np.uint32)
    ma_all, ea_all = ue4m3_dyadic(SA)
    mb_all, eb_all = ue4m3_dyadic(SB)
    for s in range(nsets):
        a, b = A[:, :, 64 * s:64 * s + 64], B[:, :, 64 * s:64 * s + 64]
        sa, sb = SA[:, :, 4 * s:4 * s + 4], SB[:, :, 4 * s:4 * s + 4]
        ma, ea = ma_all[:, :, 4 * s:4 * s + 4], ea_all[:, :, 4 * s:4 * s + 4]
        mb, eb = mb_all[:, :, 4 * s:4 * s + 4], eb_all[:, :, 4 * s:4 * s + 4]
        ha, hb = halves(a), halves(b)
        j4, j8 = np.arange(4), np.arange(8)
        # scale factors ma * 2^(ea - emin_row): rows g, g+8 x 4 blocks, cols 2t, 2t+1 x 4 blocks
        fa = ma * np.exp2(ea - emin_a[:, :, None]).astype(np.int64)
        fb = mb * np.exp2(eb - emin_b[:, :, None]).astype(np.int64)
        if fa.max() >= 1 << 15 or fb.max() >= 1 << 15:
            raise ValueError("a row's block-scale spread exceeds the kernel's 15-bit factors")
        FA = np.concatenate([fa[:, G, :], fa[:, G + 8, :]], axis=2)            # [tiles, 32, 8]
        FB = np.concatenate([fb[:, 2 * T, :], fb[:, 2 * T + 1, :]], axis=2)
        x = []
        if route in ("native", "native_hot"):
            for r in range(4):
                row = G + 8 * (r & 1)
                k0 = 8 * T + 32 * (r >> 1)
                x.append(_pack(a[:, row[:, None], k0[:, None] + j8], 4)[..., 0])
            for r in range(2):
                x.append(_pack(b[:, G[:, None], (8 * T + 32 * r)[:, None] + j8], 4)[..., 0])
            x.append(_pack(sa[:, 8 * (LANE & 1) + (LANE >> 2), :], 8)[..., 0])
            x.append(_pack(sb[:, LANE >> 2, :], 8)[..., 0])
        elif route.startswith("imma"):
            def s8_frags(va, vb):
                """m16n8k16 s8: per block, A rows g and g + 8 (k = 4 t + j), then per block B (k = 4 t + j, column g)."""
                fr = []
                for blk in range(4):
                    kk = 16 * blk + 4 * T[:, None] + j4
                    fr += [_pack(va[:, G[:, None], kk], 8)[..., 0], _pack(va[:, G[:, None] + 8, kk], 8)[..., 0]]
                return fr + [_pack(vb[:, G[:, None], 16 * blk + 4 * T[:, None] + j4], 8)[..., 0] for blk in range(4)]

            x += s8_frags(ha, hb)
            if route == "imma_exact":
                x += s8_frags(((a & 7) != 0).astype(np.int64), ((b & 7) != 0).astype(np.int64))
                ha_s = ma | ((ea + 16) << 8)
                hb_s = mb | ((eb + 16) << 8)
                for rr in (G, G + 8):
                    x.append(_pack(ha_s[:, rr, 0:2], 16)[..., 0])
                    x.append(_pack(ha_s[:, rr, 2:4], 16)[..., 0])
                for cc in (2 * T, 2 * T + 1):
                    x.append(_pack(hb_s[:, cc, 0:2], 16)[..., 0])
                    x.append(_pack(hb_s[:, cc, 2:4], 16)[..., 0])
            else:
                x += [FA[..., i].astype(np.uint32) for i in range(8)] + [FB[..., i].astype(np.uint32) for i in range(8)]
        elif route.startswith("bf16"):
            va = _bf16(ha, np.repeat(ma, 16, axis=2), np.repeat(ea, 16, axis=2))
            vb = _bf16(hb, np.repeat(mb, 16, axis=2), np.repeat(eb, 16, axis=2))
            j2 = np.arange(2)
            for blk in range(4):
                for rr, k8 in ((G, 0), (G + 8, 0), (G, 8), (G + 8, 8)):
                    x.append(_pack(va[:, rr[:, None], 16 * blk + 2 * T[:, None] + k8 + j2], 16)[..., 0])
                for k8 in (0, 8):
                    x.append(_pack(vb[:, G[:, None], 16 * blk + 2 * T[:, None] + k8 + j2], 16)[..., 0])
        elif route == "dp4a":
            for rr in (G, G + 8):
                bytes_ = ha[:, rr, :].reshape(TILES, 32, 16, 4)
                x += [_pack(bytes_[:, :, w, :], 8) for w in range(16)]
            for cc in (2 * T, 2 * T + 1):
                bytes_ = hb[:, cc, :].reshape(TILES, 32, 16, 4)
                x += [_pack(bytes_[:, :, w, :], 8) for w in range(16)]
            x = [v.reshape(TILES, 32) for v in x]
            x += [FA[..., i].astype(np.uint32) for i in range(8)] + [FB[..., i].astype(np.uint32) for i in range(8)]
        elif route == "lut":
            for rr in (G, G + 8):
                nib = a[:, rr, :].reshape(TILES, 32, 8, 8)
                x += [_pack(nib[:, :, w, :], 4).reshape(TILES, 32) for w in range(8)]
            for cc in (2 * T, 2 * T + 1):
                nib = b[:, cc, :].reshape(TILES, 32, 8, 8)
                x += [_pack(nib[:, :, w, :], 4).reshape(TILES, 32) for w in range(8)]
            x += [FA[..., i].astype(np.uint32) for i in range(8)] + [FB[..., i].astype(np.uint32) for i in range(8)]
        if len(x) != words:
            raise AssertionError(f"{route}: packed {len(x)} words per set, the kernel reads {words}")
        frags[:, :, s, :] = np.stack(x, axis=-1)
    lanec = np.concatenate([e0.astype(np.int64) & 0xFFFFFFFF, h], axis=2).astype(np.uint32)
    return frags, lanec


# ---- what each route should output (CPU, on the pinned model) -------------------------------------------------------

def expected(route: str, pool, lanec: np.ndarray, atoms_per_chain: int) -> tuple[np.ndarray, np.ndarray]:
    """[tiles, NACC, 128] words of the route's own semantics, and of the atom it emulates (native, or native from H)."""

    A, SA, B, SB = pool
    nsets, _ = SHAPE[route]
    L = atoms_per_chain
    own = np.zeros((TILES, nsets, 128), np.uint32)
    atom = np.zeros((TILES, nsets, 128), np.uint32)
    rows = G[:, None] + 8 * (np.arange(4)[None, :] >> 1)
    cols = 2 * T[:, None] + (np.arange(4)[None, :] & 1)
    for p in range(TILES):
        hmat = np.zeros((16, 8), np.uint32)
        hmat[rows, cols] = lanec[p, :, 4:8]
        for c in range(nsets):
            prob = Problem.from_codes(np.tile(A[p, :, 64 * c:64 * c + 64], (1, L)), np.tile(SA[p, :, 4 * c:4 * c + 4], (1, L)),
                                      np.tile(B[p, :, 64 * c:64 * c + 64], (1, L)), np.tile(SB[p, :, 4 * c:4 * c + 4], (1, L)))
            zero = np.zeros((16, 8), np.uint32)
            ref = prob.native(hmat) if route in HOT else prob.native(zero)
            if route in ("native", "native_hot", "imma_exact"):
                mine = ref
            elif route in ("imma_fast", "dp4a", "lut"):
                mine = prob.exact()[0]
            elif route == "imma_floor":
                mine = prob.step_floor(hmat)
            elif route == "bf16_fold":
                mine = prob.bf16_fold(zero)
            else:
                mine = prob.bf16_floor(hmat)
            own[p, c] = mine[rows, cols].reshape(128)
            atom[p, c] = ref[rows, cols].reshape(128)
    return own, atom


def dry_run(route: str, pool, frags: np.ndarray, lanec: np.ndarray, L: int) -> int:
    """The integer routes' arithmetic in numpy from the packed words (group dots from the logical operands): mismatches
    against ``expected``'s own semantics.  -1 for routes whose arithmetic is a tensor-core instruction's (checked on the card)."""

    if route not in ("imma_fast", "imma_floor", "dp4a", "lut"):
        return -1
    A, SA, B, SB = pool
    nsets, _ = SHAPE[route]
    fa_off, fb_off = {"imma_fast": (12, 20), "imma_floor": (12, 20), "dp4a": (64, 72), "lut": (32, 40)}[route]
    own, _ = expected(route, pool, lanec, L)
    e0 = lanec[..., :4].astype(np.int32).astype(np.int64)
    eh = ((lanec[..., 4:8].astype(np.int64) >> 23) & 0xFF) - 127
    bad = 0
    for c in range(nsets):
        gd, _, _, _ = atoms(A[:, :, 64 * c:64 * c + 64].reshape(-1, 64), SA[:, :, 4 * c:4 * c + 4].reshape(-1, 4),
                            B[:, :, 64 * c:64 * c + 64].reshape(-1, 64), SB[:, :, 4 * c:4 * c + 4].reshape(-1, 4))
        gd = gd.reshape(4, TILES, 16, TILES, 8)[:, np.arange(TILES), :, np.arange(TILES), :]   # [tiles, 4, 16, 8]
        st = np.zeros((TILES, 32, 4), np.int64)
        for o in range(4):
            r, q = G + 8 * (o >> 1), 2 * T + (o & 1)
            for blk in range(4):
                ab = frags[:, :, c, fa_off + 4 * (o >> 1) + blk].astype(np.int64) * frags[:, :, c, fb_off + 4 * (o & 1) + blk]
                st[:, :, o] += gd[:, blk][:, r, q] * ab
        if route == "imma_floor":
            gsh = eh - 23 - e0
            acc = (3 << (eh - 1 - e0)) + L * ((st >> gsh) << gsh)
        else:
            acc = L * st
        got = f32_rz(acc, e0)
        bad += int((got.reshape(TILES, 128) != own[:, c]).sum())
    return bad


# ---- build and SASS gate ------------------------------------------------------------------------------------------------

def nvcc_bin() -> str | None:
    for c in (shutil.which("nvcc"), "/usr/local/cuda-12.9/bin/nvcc", "/usr/local/cuda/bin/nvcc"):
        if c and Path(c).is_file():
            return c
    return None


def build(out_dir: Path) -> dict:
    nvcc = nvcc_bin()
    if nvcc is None:
        raise FileNotFoundError("nvcc not found (PATH, /usr/local/cuda-12.9/bin or /usr/local/cuda/bin)")
    so = out_dir / f"libfp4_emulation_{ARCH}.so"
    cmd = [nvcc, "-O3", f"-gencode=arch=compute_{ARCH[3:]},code={ARCH}", "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(SOURCE)]
    subprocess.run(cmd, check=True)
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return {"so": str(so), "nvcc": ver[-1].strip(), "cmd": " ".join(cmd), "sha256": hashlib.sha256(so.read_bytes()).hexdigest(),
            "sass": sass_gate(so, Path(nvcc).parent)}


def sass_gate(so: Path, bindir: Path) -> dict:
    """cuobjdump -sass: per kernel, the count of its route's instruction; the gate is exactly one atom per chain per body."""

    env = dict(os.environ, PATH=f"{bindir}:{os.environ.get('PATH', '')}")
    text = subprocess.run([str(bindir / "cuobjdump"), "-sass", str(so)], capture_output=True, text=True, check=True, env=env).stdout
    got, ok = {}, True
    for fn in re.split(r"\n\s*Function : ", text)[1:]:
        m = re.search(r"(route|peak)_kernelILi(\d+)E", fn.split("\n", 1)[0])
        if not m:
            continue
        name = ROUTES[int(m.group(2))] if m.group(1) == "route" else "peak_" + PEAKS[int(m.group(2))]
        op, want = SASS_EXPECT[name] if m.group(1) == "route" else PEAK_EXPECT[name[5:]]
        n = sum(1 for line in fn.splitlines() if re.search(r"/\*[0-9a-f]{4,5}\*/\s+(@!?U?P[T\d]+\s+)?" + re.escape(op) + r"[\s.]", line))
        got[name] = {"opcode": op, "count": n, "expected": want}
        ok &= n == want
    return {"ok": ok and len(got) == len(ROUTES) + len(PEAKS), "kernels": got}


# ---- the card -----------------------------------------------------------------------------------------------------------

class Card:
    def __init__(self, so: str):
        self.lib = ctypes.CDLL(so)
        sms, ma, mi = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        uuid = ctypes.create_string_buffer(16)
        rc = self.lib.emu_device(ctypes.byref(sms), ctypes.byref(ma), ctypes.byref(mi), uuid)
        if rc:
            raise RuntimeError(f"emu_device: cuda error {rc}")
        u = uuid.raw.hex()
        self.sms, self.cc = sms.value, (ma.value, mi.value)
        self.uuid = f"GPU-{u[:8]}-{u[8:12]}-{u[12:16]}-{u[16:20]}-{u[20:]}"

    def shape(self, r: int) -> tuple[int, int, int]:
        a, b, c = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self.lib.emu_route_shape(r, ctypes.byref(a), ctypes.byref(b), ctypes.byref(c))
        return a.value, b.value, c.value

    def blocks_per_sm(self, r: int, wpb: int) -> int:
        n = ctypes.c_int()
        rc = self.lib.emu_occupancy(r, wpb, ctypes.byref(n))
        if rc:
            raise RuntimeError(f"emu_occupancy: cuda error {rc}")
        return max(n.value, 1)

    def run(self, r: int, frags: np.ndarray, lanec: np.ndarray, nblocks: int, wpb: int, iters: int):
        nsets, nacc, words = self.shape(r)
        nw = nblocks * wpb
        out = np.zeros(nw * nacc * 128, np.uint32)
        cyc = np.zeros(nblocks, np.int64)
        ms = ctypes.c_float()
        f = np.ascontiguousarray(frags, np.uint32)
        lc = np.ascontiguousarray(lanec, np.uint32)
        rc = self.lib.emu_run(r, f.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(f.size), lc.ctypes.data_as(ctypes.c_void_p),
                              nblocks, wpb, ctypes.c_longlong(iters), out.ctypes.data_as(ctypes.c_void_p),
                              cyc.ctypes.data_as(ctypes.c_void_p), ctypes.byref(ms))
        if rc:
            raise RuntimeError(f"emu_run({ROUTES[r]}): cuda error {rc}")
        return out.reshape(nw, nacc, 128), cyc, ms.value

    def peak(self, p: int, frags: np.ndarray, nblocks: int, wpb: int, iters: int):
        cyc = np.zeros(nblocks, np.int64)
        ms = ctypes.c_float()
        f = np.ascontiguousarray(frags, np.uint32)
        rc = self.lib.emu_peak(p, f.ctypes.data_as(ctypes.c_void_p), nblocks, wpb, ctypes.c_longlong(iters),
                               cyc.ctypes.data_as(ctypes.c_void_p), ctypes.byref(ms))
        if rc:
            raise RuntimeError(f"emu_peak({PEAKS[p]}): cuda error {rc}")
        return cyc, ms.value


def per_warp(x: np.ndarray, nwarps: int) -> np.ndarray:
    return x[np.arange(nwarps) % TILES]


def timed(card: Card, fn, seconds: float, reps: int, macs_per_iter: float, nblocks: int) -> dict:
    """Calibrate iters to about `seconds`, then `reps` launches: median MAC/s, and the SM clock from clock64."""

    iters = 64
    while True:
        cyc, ms = fn(iters)
        if ms >= 1000 * seconds / 4 or iters >= 1 << 30:
            break
        iters *= 4
    iters = max(int(iters * 1000 * seconds / max(ms, 1e-3)), 1)
    rates, clocks = [], []
    for _ in range(reps):
        cyc, ms = fn(iters)
        rates.append(macs_per_iter * iters / (ms / 1000))
        clocks.append(float(np.median(cyc)) / (ms * 1000))           # MHz
    mhz = statistics.median(clocks)
    rate = statistics.median(rates)
    return {"iters": iters, "mac_per_s": rate, "mac_per_s_all": rates, "sm_mhz": mhz,
            "mac_per_sm_per_clock": rate / (card.sms * mhz * 1e6)}


def cmd_build(args) -> dict:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    b = build(out)
    pool = make_pool(args.family, args.seed)
    dry = {}
    for route in ROUTES:
        frags, lanec = pack(route, pool)
        dry[route] = {"words_per_set": int(frags.shape[-1])}
        for L in args.check_atoms:
            dry[route][f"dry_mismatches_L{L}"] = dry_run(route, pool, frags, lanec, L)
    ok = b["sass"]["ok"] and all(v[f"dry_mismatches_L{L}"] in (0, -1) for v in dry.values() for L in args.check_atoms)
    return {"ok": ok, "build": b, "dry_run": dry}


def cmd_run(args) -> dict:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    b = build(out)
    card = Card(b["so"])
    if args.expect_uuid and card.uuid != args.expect_uuid:
        raise SystemExit(f"refusing: the card is {card.uuid}, expected {args.expect_uuid}")
    pool = make_pool(args.family, args.seed)
    res = {"card": {"uuid": card.uuid, "sms": card.sms, "cc": list(card.cc)}, "build": b, "family": args.family,
           "seed": args.seed, "hot_bits": HOT_BITS, "routes": {}, "peaks": {}}
    wpb = args.warps_per_block
    for r, route in enumerate(ROUTES):
        nsets, nacc, words = card.shape(r)
        if (nsets, words) != SHAPE[route] or nsets != nacc:
            raise RuntimeError(f"{route}: the kernel's shape {nsets, nacc, words} is not the driver's {SHAPE[route]}")
        frags, lanec = pack(route, pool)
        nblocks = card.sms * card.blocks_per_sm(r, wpb)
        nw = nblocks * wpb
        fw, lw = per_warp(frags, nw), per_warp(lanec, nw)
        entry = {"nsets": nsets, "blocks": nblocks, "warps_per_block": wpb, "check": {}}
        for L in args.check_atoms:
            words_out, _, _ = card.run(r, fw, lw, nblocks, wpb, L)
            own, atom = expected(route, pool, lanec, L)
            tile = np.arange(nw) % TILES
            entry["check"][f"L{L}"] = {"words": int(words_out.size),
                                       "impl_mismatches": int((words_out != own[tile]).sum()),
                                       "matches_native": float((words_out == atom[tile]).mean())}
        macs = nw * nacc * 16 * 8 * 64
        entry["time"] = timed(card, lambda it: card.run(r, fw, lw, nblocks, wpb, it)[1:], args.seconds, args.reps, macs, nblocks)
        res["routes"][route] = entry
        print(route, json.dumps({k: entry[k] for k in ("check",)}), f"{entry['time']['mac_per_s'] / 1e12:.1f} TMAC/s", flush=True)
    base = res["routes"]["native"]["time"]["mac_per_s"]
    for route, e in res["routes"].items():
        e["A"] = e["time"]["mac_per_s"] / base
    rng = np.random.default_rng(args.seed)
    nat_frags, _ = pack("native", pool)
    bf_frags, _ = pack("bf16_fold", pool)
    for p, peak in enumerate(PEAKS):
        nblocks = card.sms * card.blocks_per_sm(-1 - p, wpb)
        nw = nblocks * wpb
        f = np.zeros((TILES, 32, 32), np.uint32)
        if peak == "omma":
            f[:, :, :] = nat_frags[:, :, :4, :].reshape(TILES, 32, 32)
        elif peak == "hmma16":
            f[:, :, :] = np.tile(bf_frags[:, :, 0, :8], (1, 1, 4))
        else:
            f[:, :, :] = rng.integers(0, 1 << 32, (TILES, 32, 32), dtype=np.uint64).astype(np.uint32) & 0x0F0F0F0F
        fw = per_warp(f, nw)
        macs = nw * PEAK_EXPECT[peak][1] * PEAK_MACS[peak]              # instructions per body per warp x MACs per instruction
        res["peaks"][peak] = timed(card, lambda it: card.peak(p, fw, nblocks, wpb, it), args.seconds, args.reps, macs, nblocks)
    ok = b["sass"]["ok"] and all(c["impl_mismatches"] == 0 for e in res["routes"].values() for c in e["check"].values())
    res["ok"] = ok
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--family", default="gaussian")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--check-atoms", type=lambda s: [int(x) for x in s.split(",")], default=[4, 32])
    ap.add_argument("--seconds", type=float, default=0.3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warps-per-block", type=int, default=4)
    ap.add_argument("--expect-uuid", default=None)
    ap.add_argument("--out-dir", default=os.environ.get("RESEARCH_RUN_DIR", "/tmp/fp4_emulation"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = cmd_build(args) if args.cmd == "build" else cmd_run(args)
    out = Path(args.out or Path(args.out_dir) / f"fp4_emulation_{args.cmd}.json")
    out.write_text(json.dumps(res, indent=1, default=float))
    if args.cmd == "run":
        meas = [(f"A_{r}", e["A"], "ratio") for r, e in res["routes"].items()]
        meas += [(f"tmac_per_s_{r}", e["time"]["mac_per_s"] / 1e12, "TMAC/s") for r, e in res["routes"].items()]
        run_result.write(out, res["ok"], "fp4_emulation_gpu",
                         "SASS gate; every route's words equal its own CPU semantics; native equals BLACKWELL_SM120_NVF4", meas)
    print(json.dumps({"ok": res["ok"], "out": str(out)}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
