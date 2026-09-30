"""The sm_120 FP8 step check on the GPU (``fp8_check.cu``): the capture's words checked where they are made, the CPU re-checking a
hash-drawn sample and every word the GPU check flags.

    python tools/tc_probe_fp4/fp8_gpucheck.py --phase gate --out $RESEARCH_RUN_DIR --lease-via-fill OWNER
    python tools/tc_probe_fp4/fp8_gpucheck.py --phase fill-gpu --unit DIR --prebuilt SO      (a gpus=1 fill job; exit 99 to continue)
    python tools/tc_probe_fp4/fp8_gpucheck.py --phase fill-verify --unit DIR                 (its gpus=0 half)

The gate, before any fill word counts: the SASS gate (each fused kernel holds exactly one mma.sync of its kind and no
floating-point instruction, the model kernel neither), then under one fill lease the GPU model's words of every candidate spec on
every tile of the existing step captures (``fp8-capture2``) and every step of the existing chain captures (``fp8-chain``, replayed
from the device's previous words), and small fused launches with every tile sampled and one word corrupted.  After the lease, on
the CPU: those words against ``fp8.evaluate`` and the core kernel (``kernels.group_sum_total_batch``), bit for bit; the GPU's
verdict on each capture against the CPU verify's (``probe_results.json``); the fused launches' operands against the numpy port of
the generator, their per-tile mismatch counts against the CPU models, and the corrupted word flagged.  Any mismatch fails the gate.

Fill units (``plan.json`` in a unit directory): one kind (e4m3, e5m2, e4m3e5m2: fused step launches over the generated families;
chain: fused E4M3 chains) and seed; launch ``i`` of family ``f`` covers tiles ``[i * L, (i + 1) * L)`` (chains: ``[i * C, (i + 1) *
C)``) and writes ``L_<f>_<i>.npz`` whole, so a preempted job restarts at the first missing launch.  Launch 0 of each family
corrupts one word (the negative control the verify phase must find flagged).  ``fill-verify`` regenerates every sampled tile and
flagged tile from its hash, checks the dumped operands against the regeneration, and re-derives each sampled tile's per-spec
mismatch counts and each flagged word with the CPU models.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fp8 as F8

ARCH = F8.ARCH
SOURCE = "fp8_check.cu"
TAB = 4096
MAXM = 4
FLAG_WORDS, META_WORDS, OPS_BYTES = 6, 3 + MAXM, 512 + 256 + 512
KEPT_SAMPLED, KEPT_FLAGGED = 1, 2
WPB = 8
TABLE_SEED = 20261001
SLOT_B, SLOT_C1, SLOT_C2, SLOT_ROW, SLOT_COL, SLOT_K0, SLOT_SAMPLE, SLOT_RES = 512, 768, 896, 1024, 1040, 1056, 1100, 1200
CHAIN_C0 = 0xFFFFFFFF
FILL_OUT = Path(os.environ.get("FP8_FILL_OUT", "/workspace/pouw/fill-out"))
KINDS = {"e4m3": (0, "e4m3", "e4m3", "f8_e4m3"), "e5m2": (1, "e5m2", "e5m2", "f8_e5m2"), "e4m3e5m2": (2, "e4m3", "e5m2", "f8_e4m3e5m2"),
         "chain": (3, "e4m3", "e4m3", "f8_e4m3")}
CONTROLS = {"primary": F8.SM120, "(32,) w25 f-133": F8.spec((32,), 25, -133), "(16,16) w26 f-133": F8.spec((16, 16), 26, -133),
            "ada (16,16) w14 f-139": F8.spec((16, 16), 14, -139), "hopper (32,) w14 f-139": F8.spec((32,), 14, -139)}
CHAIN_MODELS = {"blackwell_sm120_e4m3_m16n8k32": (F8.TCM.BLACKWELL_SM120_E4M3_M16N8K32, F8.SM120),
                "ada_e4m3_m16n8k32": (F8.TCM.ADA_E4M3_M16N8K32, CONTROLS["ada (16,16) w14 f-139"]),
                "hopper_e4m3_k32": (F8.TCM.HOPPER_E4M3_K32, CONTROLS["hopper (32,) w14 f-139"])}
SASS_EXPECT = {"17fused_step_kernelILi0E": ["QMMA.16832.F32.E4M3.E4M3"], "17fused_step_kernelILi1E": ["QMMA.16832.F32.E5M2.E5M2"],
               "17fused_step_kernelILi2E": ["QMMA.16832.F32.E4M3.E5M2"], "18fused_chain_kernel": ["QMMA.16832.F32.E4M3.E4M3"],
               "18model_tiles_kernel": []}
_WATCHED = re.compile(r"^(?:[A-Z]*MMA|F2FP|F2F|F2I|I2F|FADD|FFMA|FMUL|FMNMX|FSET|FSETP|FCHK|FRND|FSEL|MUFU|DADD|DFMA|DMUL|DSETP|HADD2|HFMA2|"
                      r"HMUL2|HMNMX2|HSETP2|LDL|STL)(?:\.|$)")


# --- the hash and the generator (numpy port of fp8_check.cu) ---------------------------------------------------------------------

def _u32(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x)).astype(np.uint32)


def fmix32(x) -> np.ndarray:
    x = _u32(x).copy()
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x85EBCA6B)
    x ^= x >> np.uint32(13)
    x *= np.uint32(0xC2B2AE35)
    x ^= x >> np.uint32(16)
    return x


def mix(x) -> np.ndarray:
    return fmix32(_u32(x) * np.uint32(0x9E3779B1) + np.uint32(0x7F4A7C15))


def tile_key(key, lo, hi) -> np.ndarray:
    return mix(mix(_u32(key) ^ _u32(lo)) ^ _u32(hi))


def hs(tk, slot) -> np.ndarray:
    return mix(_u32(tk) ^ _u32(slot))


def below(x, n) -> np.ndarray:
    return ((_u32(x).astype(np.uint64) * np.asarray(n, np.uint64)) >> np.uint64(32)).astype(np.uint32)


def family_key(seed: int, kind: str, fam_id: int) -> int:
    return int(mix(mix(mix(_u32(seed & 0xFFFFFFFF)) ^ _u32(seed >> 32)) ^ _u32((KINDS[kind][0] << 8) | fam_id))[0])


def gen_tiles(fam: dict, tabs: np.ndarray, lo, hi) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiles (lo[x], hi[x]) of a generated family -> A u8 (n,16,32), Bt u8 (n,8,32), C u32 (n,16,8): fp8_check.cu's gen_a, gen_b,
    gen_c."""
    tk = tile_key(fam["key"], lo, hi)[:, None, None]
    ta, tb, ts = tabs[0], tabs[1], tabs[2]
    i, j, k = np.arange(16)[:, None], np.arange(8)[:, None], np.arange(32)[None, :]
    am, cm = fam["amode"], fam["cmode"]
    slot_a, slot_b = (i * 32 + k).astype(np.uint32), (SLOT_B + j * 32 + k).astype(np.uint32)
    k0 = (hs(tk, SLOT_K0) & np.uint32(31)).astype(np.int64)
    if am == 1:
        x = hs(tk, SLOT_RES + slot_a)
        A = np.where(((x >> np.uint32(12)) & np.uint32(255)) < 38, ts[x & np.uint32(TAB - 1)], ta[hs(tk, i * 32 + (k & ~1)) & np.uint32(TAB - 1)])
        Bt = tb[hs(tk, SLOT_B + j * 32 + (k & ~1)) & np.uint32(TAB - 1)] ^ ((k & 1) << 7).astype(np.uint8)
    elif am == 2:
        base = ta[hs(tk, i * 32 + np.where(k < 16, k, k - 16)) & np.uint32(TAB - 1)]
        A = np.where(k < 16, base, base ^ np.uint8(0x80))
        Bt = np.broadcast_to(tb[hs(tk, SLOT_B + j * 32) & np.uint32(TAB - 1)], (tk.shape[0], 8, 32))
    elif am == 3:
        A = np.where(k == k0, ta[hs(tk, slot_a) & np.uint32(TAB - 1)], 0)
        Bt = np.where(k == k0, tb[hs(tk, slot_b) & np.uint32(TAB - 1)], 0)
    elif am == 4:
        ea, eb, mna, mnb = fam["ea"], fam["eb"], fam["min_a"], fam["min_b"]
        la, lb = max(ea - 15, mna), max(eb - 15, mnb)
        e = la + below(hs(tk, SLOT_ROW + i), ea - 10 - la + 1).astype(np.int64)
        A = ts[e - mna] | ((hs(tk, slot_a) >> np.uint32(31)) << np.uint32(7)).astype(np.uint8)
        A = np.where(k == 0, ts[ea - mna], A)
        e = lb + below(hs(tk, SLOT_COL + j), eb - 10 - lb + 1).astype(np.int64)
        Bt = np.where(k == 0, ts[64 + eb - mnb], np.broadcast_to(ts[64 + e - mnb], (tk.shape[0], 8, 32)))
    elif am == 5:
        r = hs(tk, SLOT_ROW + i)
        x = hs(tk, slot_a)
        spike = (k == (r & np.uint32(31))) | (k == ((r >> np.uint32(5)) & np.uint32(31)))
        A = np.where(spike, ts[x & np.uint32(TAB - 1)], ta[x & np.uint32(TAB - 1)])
        Bt = tb[hs(tk, slot_b) & np.uint32(TAB - 1)]
    else:
        A = ta[hs(tk, slot_a) & np.uint32(TAB - 1)]
        Bt = tb[hs(tk, slot_b) & np.uint32(TAB - 1)]
    ci, cj = np.arange(16)[:, None], np.arange(8)[None, :]
    h1, h2 = hs(tk, SLOT_C1 + ci * 8 + cj), hs(tk, SLOT_C2 + ci * 8 + cj)
    sign = np.uint32(F8.SIGN)
    if cm == 1:
        C = (h2 & sign) | ((fam["c_lo"] + below(h2 << np.uint32(1), fam["c_hi"] - fam["c_lo"] + 1)).astype(np.uint32) << np.uint32(23)) | \
            (h1 & np.uint32(0x7FFFFF))
    elif cm == 2:
        e = np.where((h2 >> np.uint32(30)) & np.uint32(1), 1 + below(h2 << np.uint32(2), 27), 0).astype(np.uint32)
        m = h1 & np.uint32(0x7FFFFF)
        C = (h1 & sign) | (e << np.uint32(23)) | np.where(m == 0, np.uint32(1), m)
    elif cm == 3:
        C = np.asarray(fam["clist"], np.uint32)[h1 & np.uint32(7)]
    elif cm == 4:
        C = (h2 & sign) | ((fam["ea"] + fam["eb"] - 18 - below(h1, 13).astype(np.int64) + 127).astype(np.uint32) << np.uint32(23))
    elif cm == 5:
        C = h1 & sign
    else:
        C = np.zeros(h1.shape, np.uint32)
    return np.ascontiguousarray(A, np.uint8), np.ascontiguousarray(Bt, np.uint8), np.ascontiguousarray(C, np.uint32)


def gen_chain_c0(fam: dict, tabs: np.ndarray, chains) -> np.ndarray:
    return gen_tiles(fam, tabs, chains, np.full(np.size(chains), CHAIN_C0, np.uint32))[2]


def decode_table(fa: str, fb: str) -> tuple[np.ndarray, int]:
    """fp8.operands of every code of both formats, packed as fp8_check.cu reads them, and the product shift 25 - 2S."""
    siga, sigb = F8.FMT[fa][1] + 1, F8.FMT[fb][1] + 1
    S = max(siga, sigb)
    out = []
    for fmt, sig in ((fa, siga), (fb, sigb)):
        nan, inf, g, u, s = F8.operands(fmt, np.arange(256))
        out.append((g << (S - sig)) | ((u + sig - 1 + 64) << 8) | (s << 16) | (nan.astype(np.int64) << 17) | (inf.astype(np.int64) << 18))
    return np.concatenate(out).astype(np.uint32), F8.PRODUCT_WIDTH - 2 * S


# --- families ----------------------------------------------------------------------------------------------------------------------

STEP_FAMILIES = ["g_randn_zero_acc", "g_randn_random_acc", "g_uniform_bits", "g_subnormal", "g_cancellation", "g_mixed_magnitude",
                 "g_near_overflow", "g_acc_floor", "g_acc_floor_cancel", "g_subnormal_isolated", "g_wide_spread"]
FLOOR_FAMILIES = ["g_acc_floor", "g_acc_floor_cancel", "g_subnormal_isolated", "g_wide_spread"]
CHAIN_FAMILIES = ["c_gauss", "c_gauss_acc", "c_pos", "c_unit", "c_absorb", "c_spikes"]
ISOLATED_ACC = [0, F8.SIGN, 0x3F800000, 0xBF800000, 0x00000001, 0x80000001, 0x0D000000, 0x8D000000]
ABSORB_ACC = [int(v) for v in F8._f32(np.array([2.0**20, 2.0**20 - 64, -(2.0**20), 2.0**19] * 2, np.float32))]


def family(kind: str, name: str, seed: int) -> tuple[dict, np.ndarray]:
    """A generated family: its parameters (fp8_check.cu's Fam) and its code tables u8 (3, TAB), drawn once per (kind, family) from
    TABLE_SEED (each table is 4096 draws of the family's distribution; the unit seed only keys the hash)."""
    fam_id = (CHAIN_FAMILIES if kind == "chain" else STEP_FAMILIES).index(name)
    _, fa, fb, _ = KINDS[kind]
    FA, FB = F8.TPF[fa], F8.TPF[fb]
    rng = np.random.default_rng([TABLE_SEED, KINDS[kind][0], fam_id])
    n = TAB
    fam = {"name": name, "kind": kind, "key": family_key(seed, kind, fam_id), "amode": 0, "cmode": 0, "c_lo": 0, "c_hi": 0,
           "ea": FA.max_pow2_exp, "eb": FB.max_pow2_exp, "min_a": FA.min_pow2_exp, "min_b": FB.min_pow2_exp, "clist": [0] * 8}
    ta = tb = ts = np.zeros(n, np.uint8)
    N = lambda: rng.standard_normal(n)
    if name in ("g_randn_zero_acc", "g_randn_random_acc"):
        ta, tb = FA.from_f32(N()), FB.from_f32(N())
        if name == "g_randn_random_acc":
            fam.update(cmode=1, c_lo=107, c_hi=147)
    elif name in ("g_uniform_bits", "g_mixed_magnitude"):
        ta, tb = FA.random_finite_bits(rng, n), FB.random_finite_bits(rng, n)
        fam.update(cmode=1, c_lo=1, c_hi=254) if name == "g_uniform_bits" else fam.update(cmode=1, c_lo=67, c_hi=187)
    elif name == "g_subnormal":
        ta, tb = FA.random_finite_bits(rng, n, 0, 1), FB.random_finite_bits(rng, n, 0, 1)
        fam.update(cmode=1, c_lo=0, c_hi=3)
    elif name == "g_cancellation":
        ta, tb, ts = FA.from_f32(N() * 2.0**8), FB.from_f32(N() * 2.0**8), FA.from_f32(N() * 2.0**-6)
        fam.update(amode=1, cmode=1, c_lo=119, c_hi=125)
    elif name == "g_near_overflow":
        ta = FA.random_finite_bits(rng, n, FA.finite_exp_max - 3, FA.finite_exp_max)
        tb = FB.random_finite_bits(rng, n, FB.finite_exp_max - 3, FB.finite_exp_max)
        fam.update(cmode=1, c_lo=250, c_hi=254)
    elif name == "g_acc_floor":
        ta = np.where(rng.random(n) < 0.5, FA.pow2(FA.min_pow2_exp), 0).astype(np.uint8)
        ta |= (rng.integers(0, 2, n) << 7).astype(np.uint8) * (ta != 0)
        tb = np.full(n, FB.pow2(FB.min_pow2_exp), np.uint8)
        fam.update(cmode=2)
    elif name == "g_acc_floor_cancel":
        ta, tb = FA.random_finite_bits(rng, n), FB.random_finite_bits(rng, n) & 0x7F
        fam.update(amode=2, cmode=2)
    elif name == "g_subnormal_isolated":
        ta = rng.integers(1, 1 << F8.FMT[fa][1], n).astype(np.uint8) | (rng.integers(0, 2, n) << 7).astype(np.uint8)
        tb = np.where(rng.random(n) < 0.5, rng.integers(1, 1 << F8.FMT[fb][1], n), FB.random_finite_bits(rng, n)).astype(np.uint8)
        fam.update(amode=3, cmode=3, clist=ISOLATED_ACC)
    elif name == "g_wide_spread":
        ts = np.zeros(n, np.uint8)
        for off, F in ((0, FA), (64, FB)):
            for e in range(F.min_pow2_exp, F.max_pow2_exp + 1):
                ts[off + e - F.min_pow2_exp] = F.pow2(e)
        fam.update(amode=4, cmode=4)
    elif name in ("c_gauss", "c_gauss_acc"):
        ta, tb = FA.from_f32(N() * 16), FB.from_f32(N() * 16)
        if name == "c_gauss_acc":
            fam.update(cmode=1, c_lo=117, c_hi=157)
    elif name == "c_pos":
        ta, tb = FA.random_finite_bits(rng, n) & 0x7F, FB.random_finite_bits(rng, n) & 0x7F
    elif name == "c_unit":
        ta, tb = np.full(n, FA.pow2(0), np.uint8), np.full(n, FB.pow2(0), np.uint8)
    elif name == "c_absorb":
        ta, tb = np.full(n, FA.pow2(-3), np.uint8), np.full(n, FB.pow2(-3), np.uint8)
        fam.update(cmode=3, clist=ABSORB_ACC)
    elif name == "c_spikes":
        ta, ts, tb = FA.from_f32(N() * 4), FA.from_f32(N() * 4 * 256), FB.from_f32(np.abs(N()) * 4)
        fam.update(amode=5)
    else:
        raise ValueError(name)
    return fam, np.stack([np.asarray(t, np.uint8) for t in (ta, tb, ts)])


def tables_digest(tabs: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(tabs).tobytes()).hexdigest()


# --- the library ------------------------------------------------------------------------------------------------------------------

class CSpec(ctypes.Structure):
    _fields_ = [("ngroups", ctypes.c_int), ("width", ctypes.c_int), ("floor", ctypes.c_int)]


class CModels(ctypes.Structure):
    _fields_ = [("n", ctypes.c_int), ("s", CSpec * MAXM)]


class CFam(ctypes.Structure):
    _fields_ = [("key", ctypes.c_uint32), *((f, ctypes.c_int) for f in ("amode", "cmode", "c_lo", "c_hi", "ea", "eb", "min_a", "min_b")),
                ("clist", ctypes.c_uint32 * 8)]


class CHostOut(ctypes.Structure):
    _fields_ = [(f, ctypes.c_void_p) for f in ("counts", "nflag", "flags", "nsamp", "meta", "words", "ops")] + \
               [(f, ctypes.c_uint32) for f in ("flag_cap", "samp_cap", "ndump")]


def cspec(sp: dict) -> CSpec:
    g = sp["groups"]
    if sp.get("post_scale") or len(set(g)) != 1 or sum(g) != 32 or len(g) not in (1, 2, 4):
        raise ValueError(f"spec {sp} is not ported to the GPU check")
    return CSpec(len(g), sp["width"], sp["floor"])


def cmodels(specs: list[dict]) -> CModels:
    m = CModels()
    m.n = len(specs)
    for x, sp in enumerate(specs):
        m.s[x] = cspec(sp)
    return m


def cfam(fam: dict) -> CFam:
    return CFam(fam["key"], *(fam[k] for k in ("amode", "cmode", "c_lo", "c_hi", "ea", "eb", "min_a", "min_b")), (ctypes.c_uint32 * 8)(*fam["clist"]))


_ptr = F8._ptr


class GpuCheck:
    """ctypes over libfp8_check: the model on host tiles (GPU or CPU), the generator on the CPU, and the fused launches."""

    info = F8.Device.info
    _check = F8.Device._check

    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        for name in ("gc_model_tiles", "gc_host_model_tiles", "gc_host_gen", "gc_fused_step", "gc_fused_chain", "device_info"):
            getattr(self.lib, name).restype = ctypes.c_int

    def model_tiles(self, sp: dict, fa: str, fb: str, A, Bt, C, SFA=None, SFB=None, host: bool = False) -> np.ndarray:
        dec, pshift = decode_table(fa, fb)
        n = A.shape[0]
        A, Bt, C = (np.ascontiguousarray(x.reshape(n, -1), t) for x, t in ((A, np.uint8), (Bt, np.uint8), (C, np.uint32)))
        scaled = SFA is not None
        SFA = np.ascontiguousarray(np.full((n, 16), 127, np.uint8) if SFA is None else SFA, np.uint8).reshape(n, 16)
        SFB = np.ascontiguousarray(np.full((n, 8), 127, np.uint8) if SFB is None else SFB, np.uint8).reshape(n, 8)
        out = np.zeros((n, 16, 8), np.uint32)
        fn = self.lib.gc_host_model_tiles if host else self.lib.gc_model_tiles
        s = cspec(sp)
        step = 1 << 18
        for a in range(0, n, step):
            b = min(n, a + step)
            o = np.zeros((b - a, 16, 8), np.uint32)
            xs = [np.ascontiguousarray(x[a:b]) for x in (A, Bt, C, SFA, SFB)]
            self._check(fn(ctypes.byref(s), _ptr(dec), pshift, *(_ptr(x) for x in xs[:3]), _ptr(xs[3]), _ptr(xs[4]), int(scaled), _ptr(o), b - a),
                        "model_tiles")
            out[a:b] = o
        return out

    def host_gen(self, fam: dict, tabs: np.ndarray, lo, hi):
        lo, hi = _u32(lo), _u32(hi)
        n = lo.size
        A, Bt, C = np.zeros((n, 16, 32), np.uint8), np.zeros((n, 8, 32), np.uint8), np.zeros((n, 16, 8), np.uint32)
        f, t = cfam(fam), np.ascontiguousarray(tabs, np.uint8)
        self._check(self.lib.gc_host_gen(ctypes.byref(f), _ptr(t), _ptr(lo), _ptr(hi), n, _ptr(A), _ptr(Bt), _ptr(C)), "host_gen")
        return A, Bt, C

    @staticmethod
    def _buffers(flag_cap: int, samp_cap: int, ndump: int, per: int) -> tuple[dict, CHostOut]:
        b = {"counts": np.zeros(3 + 2 * MAXM, np.uint64), "nflag": np.zeros(1, np.uint32), "flags": np.zeros((flag_cap, FLAG_WORDS), np.uint32),
             "nsamp": np.zeros(1, np.uint32), "meta": np.zeros((samp_cap, META_WORDS), np.uint32), "words": np.zeros((samp_cap, per), np.uint32),
             "ops": np.zeros((max(ndump, 1), OPS_BYTES), np.uint8)}
        h = CHostOut(*(b[k].ctypes.data for k in ("counts", "nflag", "flags", "nsamp", "meta", "words", "ops")), flag_cap, samp_cap, ndump)
        return b, h

    @staticmethod
    def _result(b: dict, ms: float, nspecs: int, flag_cap: int, samp_cap: int, ndump: int) -> dict:
        nf, ns = int(b["nflag"][0]), int(b["nsamp"][0])
        c = b["counts"]
        return {"tiles": int(c[0]), "elements": int(c[1]), "gated": int(c[2]),
                "mismatches": [int(c[3 + 2 * m]) for m in range(nspecs)], "record_only": [int(c[4 + 2 * m]) for m in range(nspecs)],
                "nflag": nf, "nsamp": ns, "flag_overflow": nf > flag_cap, "sample_overflow": ns > samp_cap,
                "flags": b["flags"][:min(nf, flag_cap)].copy(), "meta": b["meta"][:min(ns, samp_cap)].copy(),
                "words": b["words"][:min(ns, samp_cap)].copy(), "ops": b["ops"][:min(ns, samp_cap, ndump)].copy(), "device_ms": ms}

    def fused_step(self, kind: str, fam: dict, tabs, specs: list[dict], t0: int, blocks: int, per_warp: int, smask: int, corrupt: int = -1,
                   flag_cap: int = 4096, samp_cap: int = 4096, ndump: int = 16) -> dict:
        _, fa, fb, _ = KINDS[kind]
        dec, pshift = decode_table(fa, fb)
        b, h = self._buffers(flag_cap, samp_cap, ndump, 128)
        f, m, t, ms = cfam(fam), cmodels(specs), np.ascontiguousarray(tabs, np.uint8), ctypes.c_float()
        self._check(self.lib.gc_fused_step(KINDS[kind][0], ctypes.byref(f), ctypes.byref(m), pshift, _ptr(t), _ptr(dec), ctypes.c_ulonglong(t0),
                                           blocks, ctypes.c_uint(per_warp), ctypes.c_uint(smask), ctypes.c_ulonglong(corrupt & (2**64 - 1)),
                                           ctypes.byref(h), ctypes.byref(ms)), f"fused_step {kind} {fam['name']}")
        return self._result(b, ms.value, len(specs), flag_cap, samp_cap, ndump)

    def fused_chain(self, fam: dict, tabs, specs: list[dict], c0: int, nchains: int, nsteps: int, smask: int, corrupt: int = -1,
                    rec_chain: int | None = None, flag_cap: int = 4096, samp_cap: int = 4096, ndump: int = 16) -> dict:
        dec, pshift = decode_table("e4m3", "e4m3")
        b, h = self._buffers(flag_cap, samp_cap, ndump, 256)
        rec = np.zeros((nsteps + 1, 16, 8), np.uint32) if rec_chain is not None else None
        f, m, t, ms = cfam(fam), cmodels(specs), np.ascontiguousarray(tabs, np.uint8), ctypes.c_float()
        self._check(self.lib.gc_fused_chain(ctypes.byref(f), ctypes.byref(m), pshift, _ptr(t), _ptr(dec), ctypes.c_uint(c0), ctypes.c_uint(nchains),
                                            ctypes.c_uint(nsteps), ctypes.c_uint(smask), ctypes.c_ulonglong(corrupt & (2**64 - 1)),
                                            ctypes.c_uint(0xFFFFFFFF if rec_chain is None else rec_chain),
                                            _ptr(rec) if rec is not None else None, ctypes.byref(h), ctypes.byref(ms)), f"fused_chain {fam['name']}")
        out = self._result(b, ms.value, len(specs), flag_cap, samp_cap, ndump)
        out["rec"] = rec
        return out


def build(out: Path) -> tuple[Path, str]:
    return F8.P4.build(ARCH, out, SOURCE)


def sass_gate(so: Path, out: Path | None = None) -> dict:
    """``cuobjdump -sass``: per kernel the watched opcodes (tensor-core, floating-point, local memory), except HFMA2 Rx, -RZ, RZ, imm
    (the compiler's constant move); ok iff each kernel's multiset equals SASS_EXPECT and no other kernel exists."""
    cuobjdump = str(Path(F8.P4.nvcc_bin()).parent / "cuobjdump")
    text = subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    found: dict[str, list[str]] = {}
    fn = None
    for line in text.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            fn = next((k for k in SASS_EXPECT if k in m.group(1)), m.group(1))
            found[fn] = []
            continue
        if fn and "*/" in line:
            body = line.split("*/", 1)[1].split(";")[0].strip()
            toks = body.split()
            if toks and toks[0].startswith("@"):
                toks = toks[1:]
            if not toks or not _WATCHED.match(toks[0]):
                continue
            if toks[0] == "HFMA2" and re.match(r"^R\d+, -RZ, RZ, ", " ".join(toks[1:])):
                continue
            found[fn].append(toks[0])
    kernels = {k: {"opcodes": sorted(found.get(k, [])), "expected": sorted(v), "ok": sorted(found.get(k, [])) == sorted(v)} for k, v in SASS_EXPECT.items()}
    extra = sorted(set(found) - set(SASS_EXPECT))
    if out is not None:
        (out / "sass_fp8_check.txt").write_text(text)
    return {"ok": all(r["ok"] for r in kernels.values()) and not extra, "kernels": kernels, "unexpected_kernels": extra,
            "sass_sha256": hashlib.sha256(text.encode()).hexdigest()}


# --- CPU references -----------------------------------------------------------------------------------------------------------------

def cpu_words(specs: list[dict], fa: str, fb: str, A, Bt, C) -> np.ndarray:
    """fp8.groupsum of tiles under each spec (no scales) -> (len(specs), n, 16, 8); the products are formed once per chunk."""
    n = A.shape[0]
    out = np.zeros((len(specs), n * 128), np.uint32)
    for s in range(0, n, 512):
        e = min(n, s + 512)
        a, b, acc = F8.expand(A[s:e], Bt[s:e], C[s:e])
        p = F8.products(fa, a, fb, b)
        for x, sp in enumerate(specs):
            out[x, s * 128:e * 128] = F8.groupsum(sp, acc, p)
    return out.reshape(len(specs), n, 16, 8)


def gated_mask(fa: str, fb: str, A, Bt, C, primary) -> np.ndarray:
    return F8._finite_domain(A, Bt, C, fa, fb, primary).reshape(primary.shape)


# --- the gate ---------------------------------------------------------------------------------------------------------------------

def _json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _passed(path: Path, word: str = "passed") -> bool:
    try:
        return json.loads(path.read_text()).get("validation") == word
    except (OSError, ValueError):
        return False


def discover(root: Path) -> list[dict]:
    """The existing captures: every fp8-capture2 step unit whose GPU phase passed, every fp8-chain unit whose GPU phase passed, each
    with the CPU verify's verdict when it has one."""
    units = []
    for d in sorted((root / "fp8-capture2").glob("die*/*")):
        g = d / F8.STEP_GPU_JSON
        if d.is_dir() and _passed(g, "gpu phase passed"):
            instr = json.loads(g.read_text())["instruction"]
            units.append({"row": "step", "dir": str(d), "name": f"{d.parent.name}/{d.name}", "instruction": instr,
                          "cpu_verified": _passed(d / "probe_results.json")})
    for d in sorted((root / "fp8-chain").glob("die*/*")):
        g = d / "chain_gpu.json"
        if d.is_dir() and _passed(g, "gpu phase passed"):
            w = json.loads(g.read_text())
            units.append({"row": "chain", "dir": str(d), "name": f"{d.parent.name}/{d.name}", "seed": w["seed"], "steps": w["steps"],
                          "chain_tiles": w["chain_tiles"], "families": w.get("families"), "cpu_verified": _passed(d / "probe_results.json")})
    return units


def step_candidates(instr: str) -> dict[str, dict]:
    return {c: sp for c, sp in F8.candidates(instr).items() if not sp.get("post_scale")}


def _prep(job: tuple[dict, str]) -> str:
    """CPU, before the lease: a unit's arrays into the scratch directory as .npy (step units: the capture's tiles, decompressed;
    chain units: the operands regenerated from the seed, as fp8.verify_chain does), so the lease only reads and computes."""
    u, scratch = job
    d = Path(scratch) / u["name"].replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    if (d / "prep.done").exists():
        return u["name"]
    if u["row"] == "step":
        with np.load(Path(u["dir"]) / f"tiles_{u['instruction']}.npz") as data:
            for key in data.files:
                np.save(d / f"{key}.npy", data[key])
    else:
        fams = set(u["families"]) if u.get("families") else None
        for name, A, Bt, C0 in F8.chain_families(np.random.default_rng(u["seed"]), u["chain_tiles"], u["steps"]):
            if fams and name not in fams:
                continue
            for k, v in (("A", A), ("Bt", Bt), ("C0", C0)):
                np.save(d / f"{name}.{k}.npy", np.ascontiguousarray(v))
    (d / "prep.done").write_text("ok\n")
    return u["name"]


def _step_arrays(d: Path, fname: str, keys: str = "A Bt C SFA SFB") -> list[np.ndarray]:
    return [np.load(d / f"{fname}.{k}.npy") for k in keys.split()]


GATE_WINDOW = 64  # chain steps at each end of every chain family whose words the CPU recomputes directly


def gate_gpu(args) -> int:
    """In the lease: identity and SASS gates, then the GPU model's words of every existing capture into the scratch directory (step
    units: every ported candidate; chain units: counts against the device and the two end windows), then the fused smoke launches.
    Restartable per unit; exits 99 when --budget-s is spent with units left."""
    t0 = time.time()
    out, scratch = Path(args.out), Path(args.scratch)
    out.mkdir(parents=True, exist_ok=True)
    so = Path(args.prebuilt)
    res: dict = {"phase": "gate-gpu", "prebuilt": str(so), "so_sha256": hashlib.sha256(so.read_bytes()).hexdigest()}
    dev = GpuCheck(so)
    ident, failures = F8.identity_gate(dev, args.expect_gpu_uuid)
    res["identity"] = ident
    sass = sass_gate(so)
    res["sass"] = sass
    if not sass["ok"]:
        failures.append("SASS gate failed")
    if failures:
        res.update(failures=failures, validation="failed")
        _json(out / "gate_gpu.json", res)
        return 1
    units = json.loads(Path(args.units).read_text())
    for u in units:
        if time.time() - t0 > args.budget_s:
            res.update(failures=[], validation="continue")
            _json(out / "gate_gpu.json", res)
            return 99
        d = scratch / u["name"].replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        if (d / "gpu.done").exists():
            continue
        t = time.time()
        rec: dict = {}
        if u["row"] == "step":
            instr = u["instruction"]
            _, fa, fb, _, _ = F8.TILES[instr]
            scaled = instr == "mxf8_e4m3"
            cands = list(step_candidates(instr).items())
            fams = sorted(p.name[: -len(".A.npy")] for p in d.glob("*.A.npy"))
            for fname in fams:
                A, Bt, C, SFA, SFB = _step_arrays(d, fname)
                words = np.stack([dev.model_tiles(sp, fa, fb, A, Bt, C, SFA if scaled else None, SFB if scaled else None) for _, sp in cands])
                np.save(d / f"{fname}.gpu.npy", words)
            rec = {"families": fams, "candidates": [c for c, _ in cands]}
        else:
            words = np.load(Path(u["dir"]) / F8.CHAIN_WORDS)
            fams = [f.name.split(".")[0] for f in sorted(d.glob("*.C0.npy"))]
            rec = {"families": {}}
            for fname in fams:
                A, Bt, C0 = (np.load(d / f"{fname}.{k}.npy") for k in ("A", "Bt", "C0"))
                D = words[fname]
                tiles, steps = A.shape[:2]
                prev = np.concatenate([C0.reshape(tiles, 1, 16, 8), D[:, :-1]], axis=1)
                win = sorted(set(range(min(GATE_WINDOW, steps))) | set(range(max(0, steps - GATE_WINDOW), steps)))
                fam_rec, wins = {}, []
                for mname, (_, sp) in CHAIN_MODELS.items():
                    g = dev.model_tiles(sp, "e4m3", "e4m3", A.reshape(-1, 16, 32), Bt.reshape(-1, 8, 32), prev.reshape(-1, 16, 8))
                    g = g.reshape(tiles, steps, 16, 8)
                    fam_rec[mname] = int(np.count_nonzero(g != D))
                    wins.append(g[:, win])
                np.save(d / f"{fname}.gpuwin.npy", np.stack(wins))
                rec["families"][fname] = {"step_mismatches": fam_rec, "window_steps": win, "tiles": tiles, "steps": steps}
        rec["seconds"] = round(time.time() - t, 3)
        _json(d / "gpu.json", rec)
        (d / "gpu.done").write_text("ok\n")
    smoke = out / "smoke"
    smoke.mkdir(exist_ok=True)
    if not (smoke / "smoke.done").exists():
        res["smoke"] = run_smoke(dev, smoke, args.seed)
        (smoke / "smoke.done").write_text("ok\n")
    if not (out / "timing.json").exists():
        _json(out / "timing.json", run_timing(dev, args.seed))
    res.update(failures=[], validation="gpu phase passed", seconds=round(time.time() - t0, 3))
    _json(out / "gate_gpu.json", res)
    return 0


SMOKE_BLOCKS, SMOKE_PER_WARP, SMOKE_CORRUPT = 4, 4, 5
SMOKE_CHAINS, SMOKE_STEPS, SMOKE_CHAIN_CORRUPT = 8, 64, (1, 7)  # (chain, step)


def run_smoke(dev: GpuCheck, out: Path, seed: int) -> dict:
    """Fused launches small enough that the CPU checks all of them: every tile (chain step) sampled with its operands, one word
    corrupted, all control specs; saved for the CPU half of the gate."""
    specs = list(CONTROLS.values())[:MAXM]
    info = {}
    n = SMOKE_BLOCKS * WPB * SMOKE_PER_WARP
    for kind in ("e4m3", "e5m2", "e4m3e5m2"):
        for name in STEP_FAMILIES:
            fam, tabs = family(kind, name, seed)
            r = dev.fused_step(kind, fam, tabs, specs, 0, SMOKE_BLOCKS, SMOKE_PER_WARP, 0, SMOKE_CORRUPT, samp_cap=n, ndump=n)
            np.savez(out / f"{kind}.{name}.npz", **{k: v for k, v in r.items() if isinstance(v, np.ndarray)},
                     summary=json.dumps({k: v for k, v in r.items() if not isinstance(v, np.ndarray)}))
            info[f"{kind}.{name}"] = {k: r[k] for k in ("tiles", "mismatches", "record_only", "nflag", "nsamp")}
    for name in CHAIN_FAMILIES:
        fam, tabs = family("chain", name, seed)
        ns = SMOKE_CHAINS * SMOKE_STEPS
        r = dev.fused_chain(fam, tabs, specs, 0, SMOKE_CHAINS, SMOKE_STEPS, 0, (SMOKE_CHAIN_CORRUPT[0] << 32) | SMOKE_CHAIN_CORRUPT[1], rec_chain=0,
                            samp_cap=ns, ndump=ns)
        np.savez(out / f"chain.{name}.npz", **{k: v for k, v in r.items() if isinstance(v, np.ndarray)},
                 summary=json.dumps({k: v for k, v in r.items() if not isinstance(v, np.ndarray)}))
        info[f"chain.{name}"] = {k: r[k] for k in ("tiles", "mismatches", "record_only", "nflag", "nsamp")}
    return info


def run_timing(dev: GpuCheck, seed: int) -> dict:
    """Device time of production-sized fused launches (cudaEvent), 1 and 4 specs: the throughput the fill is sized by."""
    t = {"sms": dev.info()["sms"]}
    fam, tabs = family("e4m3", "g_randn_random_acc", seed + 1)
    blocks = 2 * t["sms"]
    for nspec in (1, 2, 4):
        specs = list(CONTROLS.values())[:nspec]
        for per_warp in (64, 512):
            r = dev.fused_step("e4m3", fam, tabs, specs, 1 << 40, blocks, per_warp, (1 << 16) - 1)
            t[f"step.specs{nspec}.pw{per_warp}"] = {"tiles": r["tiles"], "device_ms": r["device_ms"], "tiles_per_s": r["tiles"] / (r["device_ms"] / 1e3),
                                                     "mismatches": r["mismatches"], "nsamp": r["nsamp"]}
    fam, tabs = family("chain", "c_gauss", seed + 1)
    for nspec in (1, 2):
        specs = list(CONTROLS.values())[:nspec]
        r = dev.fused_chain(fam, tabs, specs, 1 << 20, 16 * t["sms"], 1024, (1 << 16) - 1)
        t[f"chain.specs{nspec}"] = {"steps": r["tiles"], "device_ms": r["device_ms"], "steps_per_s": r["tiles"] / (r["device_ms"] / 1e3),
                                    "mismatches": r["mismatches"]}
    return t


def _cmp_step_unit(job: tuple[dict, str]) -> dict:
    """CPU: a step unit's GPU words against fp8.evaluate (every ported candidate) and the core kernel (the registered primary), and
    the GPU's verdict on the device words against the CPU verify's."""
    u, scratch = job
    d = Path(scratch) / u["name"].replace("/", "__")
    instr = u["instruction"]
    _, fa, fb, _, _ = F8.TILES[instr]
    cands = step_candidates(instr)
    names = list(cands)
    primary = F8.PRIMARY[instr][0]
    core = {"f8_e4m3": F8.TCM.BLACKWELL_SM120_E4M3_M16N8K32, "legacy_e4m3": F8.TCM.BLACKWELL_SM120_E4M3_M16N8K32,
            "f8_e5m2": F8.TCM.BLACKWELL_SM120_E5M2_M16N8K32}.get(instr)
    out = {"unit": u["name"], "instruction": instr, "elements": 0, "gpu_vs_cpu": {c: 0 for c in names}, "gpu_vs_core": 0 if core else None,
           "gpu_on_device": {c: {"gated": 0, "record_only": 0} for c in names}, "first": []}
    for fname in json.loads((d / "gpu.json").read_text())["families"]:
        A, Bt, C, D, SFA, SFB = _step_arrays(d, fname, "A Bt C D SFA SFB")
        g = np.load(d / f"{fname}.gpu.npy")
        out["elements"] += int(D.size)
        prim = None
        for x, c in enumerate(names):
            w = F8.evaluate(instr, cands[c], A, Bt, C, SFA, SFB)
            bad = g[x] != w
            out["gpu_vs_cpu"][c] += int(bad.sum())
            for r in np.argwhere(bad)[: max(0, 5 - len(out["first"]))]:
                out["first"].append({"family": fname, "candidate": c, "at": [int(v) for v in r], "gpu": f"{int(g[x][tuple(r)]):08x}",
                                     "cpu": f"{int(w[tuple(r)]):08x}"})
            if c == primary:
                prim = w
        gate = gated_mask(fa, fb, A, Bt, C, prim)
        for x, c in enumerate(names):
            dbad = g[x] != D
            out["gpu_on_device"][c]["gated"] += int((dbad & gate).sum())
            out["gpu_on_device"][c]["record_only"] += int((dbad & ~gate).sum())
        if core is not None:
            out["gpu_vs_core"] += int(np.count_nonzero(F8.step_model(core, A, Bt, C) != g[names.index(primary)]))
    pr = Path(u["dir"]) / "probe_results.json"
    if u["cpu_verified"] and pr.exists():
        cap = json.loads(pr.read_text())["capture"]
        want = {c: {"gated": sum(f["mismatches"][c] for f in cap["families"].values()),
                    "record_only": sum(f["record_only_mismatches"][c] for f in cap["families"].values())} for c in names}
        out["cpu_verify_on_device"] = want
        out["verdicts_agree"] = want == out["gpu_on_device"]
    shutil.rmtree(d, ignore_errors=True)
    return out


def _cmp_chain_unit(job: tuple[dict, str]) -> dict:
    """CPU: a chain unit's GPU step replays against the core kernel's words on both end windows (every model), and its step
    mismatch counts against the CPU verify's."""
    u, scratch = job
    d = Path(scratch) / u["name"].replace("/", "__")
    g = json.loads((d / "gpu.json").read_text())
    words = np.load(Path(u["dir"]) / F8.CHAIN_WORDS)
    out = {"unit": u["name"], "window_words": 0, "gpu_vs_cpu_window": {m: 0 for m in CHAIN_MODELS}, "gpu_on_device": {m: 0 for m in CHAIN_MODELS}}
    for fname, fr in g["families"].items():
        A, Bt, C0 = (np.load(d / f"{fname}.{k}.npy") for k in ("A", "Bt", "C0"))
        D = words[fname]
        tiles = A.shape[0]
        prev = np.concatenate([C0.reshape(tiles, 1, 16, 8), D[:, :-1]], axis=1)
        win = fr["window_steps"]
        gw = np.load(d / f"{fname}.gpuwin.npy")
        for x, (mname, (model, _)) in enumerate(CHAIN_MODELS.items()):
            w = F8.step_model(model, A[:, win].reshape(-1, 16, 32), Bt[:, win].reshape(-1, 8, 32), prev[:, win].reshape(-1, 16, 8))
            out["gpu_vs_cpu_window"][mname] += int(np.count_nonzero(w.reshape(tiles, len(win), 16, 8) != gw[x]))
            out["gpu_on_device"][mname] += fr["step_mismatches"][mname]
        out["window_words"] += tiles * len(win) * 128
    pr = Path(u["dir"]) / "probe_results.json"
    if u["cpu_verified"] and pr.exists():
        cap = json.loads(pr.read_text())["capture"]
        want = {m: sum(f["step_mismatches"][m] for f in cap["families"].values()) for m in CHAIN_MODELS}
        out["cpu_verify_on_device"] = want
        out["verdicts_agree"] = want == out["gpu_on_device"]
    shutil.rmtree(d, ignore_errors=True)
    return out


def launch_tiles(kind: str, first: int, count: int, steps: int = 0):
    """(lo, hi) chunks of every tile a launch covers: step tiles first .. first + count - 1; chain steps (chain, step) of chains
    first .. first + count - 1, step < steps."""
    if kind == "chain":
        for c in range(first, first + count, 64):
            cs = np.arange(c, min(first + count, c + 64), dtype=np.uint32)
            yield np.repeat(cs, steps), np.tile(np.arange(steps, dtype=np.uint32), cs.size)
        return
    for a in range(first, first + count, 1 << 21):
        t = np.arange(a, min(first + count, a + (1 << 21)), dtype=np.uint64)
        yield (t & np.uint64(0xFFFFFFFF)).astype(np.uint32), (t >> np.uint64(32)).astype(np.uint32)


def _key(lo, hi) -> np.ndarray:
    return (np.asarray(hi, np.uint64) << np.uint64(32)) | np.asarray(lo, np.uint64)


def hash_sample(fam: dict, smask: int, chunks) -> np.ndarray:
    """The sorted keys (hi << 32 | lo) of the tiles the hash samples among the chunks'."""
    keys = [np.zeros(0, np.uint64)]
    for lo, hi in chunks:
        m = (hs(tile_key(fam["key"], lo, hi), SLOT_SAMPLE) & np.uint32(smask)) == 0
        keys.append(_key(lo[m], hi[m]))
    return np.sort(np.concatenate(keys))


def check_launch(kind: str, fam: dict, tabs: np.ndarray, specs: list[dict], r: dict, summ: dict, smask: int, chunks,
                 corrupt: tuple[int, int] | None) -> dict:
    """CPU: one fused launch.  The kept tiles are exactly the hash's sample over the launch plus the flagged words' tiles; their
    dumped operands are the numpy generator's; per kept tile, every spec's mismatch count is the CPU models'; the flags are exactly
    the CPU primary's mismatches on the kept tiles (device word, primary word and gate included) and add up to the launch's
    primary counts; the negative control (``corrupt`` = (lo, hi) of its tile, element 0) is flagged."""
    _, fa, fb, _ = KINDS[kind]
    meta, words, ops, flags = r["meta"], r["words"], r["ops"], r["flags"]
    fails = []
    if summ["sample_overflow"] or summ["flag_overflow"]:
        fails.append(f"buffer overflow: {summ['nsamp']} kept tiles, {summ['nflag']} flags")
    lo, hi, why = meta[:, 0], meta[:, 1], meta[:, 2]
    key, n = _key(lo, hi), len(meta)
    fkey = _key(flags[:, 0], flags[:, 1])
    sample = hash_sample(fam, smask, chunks)
    if not summ["sample_overflow"] and not np.array_equal(np.sort(key[(why & KEPT_SAMPLED) != 0]), sample):
        fails.append(f"the kept sample ({int(np.count_nonzero(why & KEPT_SAMPLED))} tiles) is not the hash's ({sample.size})")
    if not summ["sample_overflow"] and not np.array_equal(np.sort(key[(why & KEPT_FLAGGED) != 0]), np.unique(fkey)):
        fails.append("the tiles kept as flagged are not the flagged words' tiles")
    if np.any(why == 0) or np.unique(key).size != n:
        fails.append("a kept tile has no reason, or is kept twice")
    A, Bt, C = gen_tiles(fam, tabs, lo, hi)
    if kind == "chain":
        C = np.ascontiguousarray(words[:, :128].reshape(-1, 16, 8))
    D = words[:, -128:].reshape(-1, 16, 8)
    nd = ops.shape[0]
    port = {"A": int(np.count_nonzero(ops[:, :512].reshape(-1, 16, 32) != A[:nd])),
            "Bt": int(np.count_nonzero(ops[:, 512:768].reshape(-1, 8, 32) != Bt[:nd])),
            "C": int(np.count_nonzero(np.ascontiguousarray(ops[:, 768:]).view(np.uint32).reshape(-1, 16, 8) != C[:nd]))}
    if any(port.values()):
        fails.append(f"dumped operands differ from the numpy generator's: {port}")
    cpu = cpu_words(specs, fa, fb, A, Bt, C) if n else np.zeros((len(specs), 0, 16, 8), np.uint32)
    bad = (cpu != D[None]).reshape(len(specs), n, 128)
    tile_m = bad.sum(-1).T
    if not np.array_equal(tile_m, meta[:, 3:3 + len(specs)].astype(np.int64)):
        fails.append("kept tiles' per-spec mismatch counts differ from the CPU models'")
    gate = gated_mask(fa, fb, A, Bt, C, cpu[0]).reshape(n, 128) if n else np.zeros((0, 128), bool)
    t_i, e_i = np.nonzero(bad[0])
    Dw, Pw = D.reshape(n, 128), cpu[0].reshape(n, 128)
    cpu_flags = sorted(zip(key[t_i].tolist(), e_i.tolist(), Dw[t_i, e_i].tolist(), Pw[t_i, e_i].tolist(), gate[t_i, e_i].astype(int).tolist()))
    gpu_flags = sorted(zip(fkey.tolist(), flags[:, 2].tolist(), flags[:, 3].tolist(), flags[:, 4].tolist(), flags[:, 5].tolist()))
    if cpu_flags != gpu_flags:
        fails.append(f"the flags ({len(gpu_flags)}) are not the CPU primary's mismatches on the kept tiles ({len(cpu_flags)})")
    ctl = int(_key(*corrupt)) if corrupt is not None else None
    ctl_flags = [f for f in gpu_flags if f[0] == ctl and f[1] == 0]
    if corrupt is not None and not ctl_flags:
        fails.append(f"the negative control (tile {corrupt}, element 0) was not flagged")
    real = [f for f in gpu_flags if not (f[0] == ctl and f[1] == 0)]
    if not summ["flag_overflow"] and (sum(f[4] for f in gpu_flags) != summ["mismatches"][0] or
                                      sum(1 - f[4] for f in gpu_flags) != summ["record_only"][0]):
        fails.append(f"the flags do not add up to the primary's counts {summ['mismatches'][0]} + {summ['record_only'][0]}")
    sampled = (why & KEPT_SAMPLED) != 0
    return {"kept": n, "sampled_tiles": int(sampled.sum()), "sampled_words": int(sampled.sum()) * 128,
            "cpu_sample_mismatches": [int(v) for v in bad[:, sampled].sum((1, 2))], "port_mismatches": port, "flags": len(gpu_flags),
            "negative_control_flagged": bool(ctl_flags) if corrupt is not None else None,
            "primary_gated_beyond_control": sum(f[4] for f in real), "primary_record_only_beyond_control": sum(1 - f[4] for f in real),
            "first_real": [[f"{v:x}" for v in f] for f in real[:5]], "failures": fails}


def check_recorded_chain(fam: dict, tabs: np.ndarray, chain: int, rec: np.ndarray, kept: dict | None = None) -> dict:
    """CPU: a chain whose every word the launch kept (rec = C0, then each step's device word): C0 is the generator's, and every step
    is the core kernel's registered model on the device's previous word; the launch's kept steps of this chain agree with rec."""
    fails = []
    steps = rec.shape[0] - 1
    if not np.array_equal(gen_chain_c0(fam, tabs, [chain])[0], rec[0]):
        fails.append("the recorded chain's C0 differs from the generator's")
    bad = 0
    for s0 in range(0, steps, 1024):
        s1 = min(steps, s0 + 1024)
        a, b, _ = gen_tiles(fam, tabs, np.full(s1 - s0, chain, np.uint32), np.arange(s0, s1, dtype=np.uint32))
        bad += int(np.count_nonzero(F8.step_model(F8.TCM.BLACKWELL_SM120_E4M3_M16N8K32, a, b, rec[s0:s1]) != rec[s0 + 1:s1 + 1]))
    if bad:
        fails.append(f"the recorded chain: {bad} step words differ from the core kernel on the device's previous words")
    if kept is not None:
        m = kept["meta"][:, 0] == chain
        s = kept["meta"][m, 1].astype(np.int64)
        w = kept["words"][m]
        if np.count_nonzero(w[:, :128].reshape(-1, 16, 8) != rec[s]) or np.count_nonzero(w[:, 128:].reshape(-1, 16, 8) != rec[s + 1]):
            fails.append("the recorded chain differs from the launch's kept steps of it")
    return {"chain": chain, "steps": steps, "replay_mismatches": bad, "failures": fails}


def check_smoke_file(path: Path, seed: int) -> dict:
    """CPU: one fused smoke launch (smask 0: every tile kept, with its operands) through check_launch; chains also replay chain 0."""
    kind, name = path.stem.split(".")
    fam, tabs = family(kind, name, seed)
    with np.load(path) as z:
        r = {k: z[k] for k in z.files}
    summ = json.loads(str(r["summary"]))
    specs = list(CONTROLS.values())[:MAXM]
    if kind == "chain":
        out = check_launch(kind, fam, tabs, specs, r, summ, 0, launch_tiles(kind, 0, SMOKE_CHAINS, SMOKE_STEPS), SMOKE_CHAIN_CORRUPT)
        out["recorded_chain"] = check_recorded_chain(fam, tabs, 0, r["rec"], r)
        out["failures"] += out["recorded_chain"]["failures"]
        want = SMOKE_CHAINS * SMOKE_STEPS
    else:
        want = SMOKE_BLOCKS * WPB * SMOKE_PER_WARP
        out = check_launch(kind, fam, tabs, specs, r, summ, 0, launch_tiles(kind, 0, want), (SMOKE_CORRUPT, 0))
    if out["kept"] != want or r["ops"].shape[0] != want:
        out["failures"].append(f"{out['kept']} tiles kept with {r['ops'].shape[0]} operand dumps, not every tile ({want})")
    return {"launch": path.stem, **out}


def gate(args) -> int:
    """The gate run: build and SASS-gate, prepare the chain operands, the GPU half under fill leases, then the CPU half."""
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"schema": "tc-probe-fp4/fp8-gpucheck/v1", "phase": "gate", "seed": args.seed}
    so, nvcc = build(out)
    res["nvcc"] = nvcc
    res["sass"] = sass_gate(so, out)
    if not res["sass"]["ok"]:
        return finish_gate(out, res, ["SASS gate failed"], t0)
    host = GpuCheck(so)
    res["host_port"] = host_port_check(host, args.seed)
    if res["host_port"]["failures"]:
        return finish_gate(out, res, res["host_port"]["failures"], t0)
    units = discover(Path(args.capture_root))
    if args.max_units:
        units = units[: args.max_units]
    res["units"] = {"step": sum(u["row"] == "step" for u in units), "chain": sum(u["row"] == "chain" for u in units),
                    "cpu_verified": sum(u["cpu_verified"] for u in units)}
    scratch = Path(args.scratch or (FILL_OUT / "fp8-gpucheck" / "gate-scratch" / (os.environ.get("RESEARCH_RUN_ID") or str(os.getpid()))))
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        return _gate_units(args, out, res, so, units, scratch, t0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _gate_units(args, out: Path, res: dict, so: Path, units: list[dict], scratch: Path, t0: float) -> int:
    failures: list[str] = []
    (out / "units.json").write_text(json.dumps(units, indent=1))
    t = time.time()
    with Pool(args.workers) as pool:
        pool.map(_prep, [(u, str(scratch)) for u in units], chunksize=1)
    res["prep_seconds"] = round(time.time() - t, 3)
    sub = out / "lease"
    child = [sys.executable, str(Path(__file__).resolve()), "--phase", "gate-gpu", "--out", str(sub), "--prebuilt", str(so), "--scratch", str(scratch),
             "--units", str(out / "units.json"), "--seed", str(args.seed), "--budget-s", str(args.budget_s)]
    import w1 as W1
    leases = []
    for n in range(args.max_leases):
        rc, fill = W1.fill_lease(child, args.lease_via_fill, args.lease_max_min, sub,
                                 job=f"fp8gc-{os.environ.get('RESEARCH_RUN_ID') or os.getpid()}-{n}.sh")
        leases.append({"rc": rc, **{k: v for k, v in fill.items() if k != "events"}})
        if rc != 99:
            break
    res["leases"] = leases
    g = json.loads((sub / "gate_gpu.json").read_text()) if (sub / "gate_gpu.json").exists() else {"failures": ["no gate_gpu.json"], "validation": "failed"}
    res["identity"], res["gpu_phase"] = g.get("identity"), {k: g.get(k) for k in ("validation", "seconds", "so_sha256", "smoke")}
    if g.get("validation") != "gpu phase passed":
        return finish_gate(out, res, list(g.get("failures") or []) + [f"GPU phase: {g.get('validation')} (last lease exit {leases[-1]['rc']})"], t0)
    t = time.time()
    jobs = [(u, str(scratch)) for u in units]
    with Pool(args.workers) as pool:
        steps = pool.map(_cmp_step_unit, [j for j in jobs if j[0]["row"] == "step"], chunksize=1)
        chains = pool.map(_cmp_chain_unit, [j for j in jobs if j[0]["row"] == "chain"], chunksize=1)
        smoke = pool.starmap(check_smoke_file, [(p, args.seed) for p in sorted((sub / "smoke").glob("*.npz"))])
    res["cpu_seconds"] = round(time.time() - t, 3)
    res["step_units"], res["chain_units"], res["smoke"] = steps, chains, smoke
    res["timing"] = json.loads((sub / "timing.json").read_text()) if (sub / "timing.json").exists() else None
    tot = lambda rows, key: {k: sum(r[key][k] for r in rows) for k in (rows[0][key] if rows else {})}
    res["totals"] = {"step_elements": sum(r["elements"] for r in steps), "step_gpu_vs_cpu": sum(sum(r["gpu_vs_cpu"].values()) for r in steps),
                     "step_gpu_vs_core": sum(r["gpu_vs_core"] or 0 for r in steps),
                     "step_verdicts_compared": sum("verdicts_agree" in r for r in steps), "step_verdicts_agree": sum(bool(r.get("verdicts_agree")) for r in steps),
                     "chain_window_words": sum(r["window_words"] for r in chains), "chain_gpu_vs_cpu_window": tot(chains, "gpu_vs_cpu_window"),
                     "chain_gpu_on_device": tot(chains, "gpu_on_device"),
                     "chain_verdicts_compared": sum("verdicts_agree" in r for r in chains), "chain_verdicts_agree": sum(bool(r.get("verdicts_agree")) for r in chains),
                     "chain_step_words_replayed": sum(sum(f["tiles"] * f["steps"] * 128 for f in json.loads(Path(u["dir"], "chain_gpu.json").read_text())["device"].values())
                                                      for u in units if u["row"] == "chain"),
                     "smoke_launches": len(smoke), "smoke_failures": sum(bool(s["failures"]) for s in smoke),
                     "smoke_words": sum(s["kept"] * 128 for s in smoke), "smoke_negative_controls_flagged": sum(bool(s["negative_control_flagged"]) for s in smoke),
                     "smoke_primary_gated_beyond_control": sum(s["primary_gated_beyond_control"] for s in smoke),
                     "smoke_primary_record_only_beyond_control": sum(s["primary_record_only_beyond_control"] for s in smoke)}
    T = res["totals"]
    if T["step_gpu_vs_cpu"] or T["step_gpu_vs_core"]:
        failures.append(f"step captures: {T['step_gpu_vs_cpu']} GPU words differ from fp8.evaluate, {T['step_gpu_vs_core']} from the core kernel")
    if any(T["chain_gpu_vs_cpu_window"].values()):
        failures.append(f"chain windows: GPU words differ from the core kernel's {T['chain_gpu_vs_cpu_window']}")
    if T["step_verdicts_agree"] != T["step_verdicts_compared"] or T["chain_verdicts_agree"] != T["chain_verdicts_compared"]:
        failures.append("the GPU's verdict on a capture differs from the CPU verify's")
    if T["chain_gpu_on_device"].get("blackwell_sm120_e4m3_m16n8k32"):
        failures.append(f"the GPU primary flags {T['chain_gpu_on_device']['blackwell_sm120_e4m3_m16n8k32']} chain step words")
    for s in smoke:
        failures += [f"smoke {s['launch']}: {f}" for f in s["failures"]]
    return finish_gate(out, res, failures, t0)


def host_port_check(dev: GpuCheck, seed: int) -> dict:
    """CPU, before any lease: the library's host build of the model and the generator against fp8.groupsum and the numpy port, on
    every generated family (the device build is the same source; the gate's GPU half checks it on silicon)."""
    fails, rows = [], {}
    lo = np.arange(24, dtype=np.uint32) * np.uint32(2654435761)
    hi = np.arange(24, dtype=np.uint32)
    for kind in ("e4m3", "e5m2", "e4m3e5m2", "chain"):
        _, fa, fb, _ = KINDS[kind]
        for name in CHAIN_FAMILIES if kind == "chain" else STEP_FAMILIES:
            fam, tabs = family(kind, name, seed)
            A, Bt, C = gen_tiles(fam, tabs, lo, hi)
            hA, hB, hC = dev.host_gen(fam, tabs, lo, hi)
            gen_bad = int(np.count_nonzero(A != hA) + np.count_nonzero(Bt != hB) + np.count_nonzero(C != hC))
            cpu = cpu_words(list(CONTROLS.values()), fa, fb, A, Bt, C)
            model_bad = {c: int(np.count_nonzero(dev.model_tiles(sp, fa, fb, A, Bt, C, host=True) != cpu[x])) for x, (c, sp) in enumerate(CONTROLS.items())}
            if kind == "chain":
                model_bad |= {m: int(np.count_nonzero(dev.model_tiles(sp, fa, fb, A, Bt, C, host=True) != F8.step_model(model, A, Bt, C)))
                              for m, (model, sp) in CHAIN_MODELS.items()}
            rows[f"{kind}.{name}"] = {"generator": gen_bad, "model": model_bad}
            if gen_bad or any(model_bad.values()):
                fails.append(f"host port {kind}.{name}: generator {gen_bad}, model {model_bad}")
    rng = np.random.default_rng(seed)
    for name in ("g_randn_random_acc", "g_uniform_bits", "g_subnormal", "g_acc_floor"):
        fam, tabs = family("e4m3", name, seed)
        A, Bt, C = gen_tiles(fam, tabs, lo, hi)
        SFA, SFB = (np.where(rng.random((24, w)) < 0.05, 0xFF, rng.integers(112, 143, (24, w))).astype(np.uint8) for w in (16, 8))
        bad = {c: int(np.count_nonzero(dev.model_tiles(sp, "e4m3", "e4m3", A, Bt, C, SFA, SFB, host=True) != F8.evaluate("mxf8_e4m3", sp, A, Bt, C, SFA, SFB)))
               for c, sp in step_candidates("mxf8_e4m3").items()}
        rows[f"mxf8.{name}"] = {"model": bad}
        if any(bad.values()):
            fails.append(f"host port mxf8.{name}: model {bad}")
    return {"families": rows, "failures": fails}


def finish_gate(out: Path, res: dict, failures: list[str], t0: float) -> int:
    res.update(failures=failures, validation="passed" if not failures else "failed", seconds=round(time.time() - t0, 3))
    _json(out / "gate_results.json", res)
    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id:
        T, tm = res.get("totals") or {}, res.get("timing") or {}
        meas = [{"name": "wall", "value": res["seconds"], "unit": "s"}]
        for k, v in tm.items():
            if isinstance(v, dict) and "tiles_per_s" in v:
                meas.append({"name": f"throughput.{k}", "value": round(v["tiles_per_s"]), "unit": "tiles/s", "label": "Measured, locked-2100 clocks"})
            if isinstance(v, dict) and "steps_per_s" in v:
                meas.append({"name": f"throughput.{k}", "value": round(v["steps_per_s"]), "unit": "chain steps/s", "label": "Measured, locked-2100 clocks"})
        doc = {"schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": run_id, "instruction": "sm120.mma.m16n8k32.f8.gpucheck",
               "row": "gpucheck-gate", "arch": ARCH, "sku": (res.get("identity") or {}).get("device"), "gpu_uuid": (res.get("identity") or {}).get("gpu_uuid"),
               "uuid_checked": (res.get("identity") or {}).get("uuid_checked"), "driver": (res.get("identity") or {}).get("driver"), "nvcc": res.get("nvcc"),
               "sass_opcodes": {k: r["opcodes"] for k, r in (res.get("sass") or {}).get("kernels", {}).items()},
               "sass_sha256": (res.get("sass") or {}).get("sass_sha256"), "so_sha256": (res.get("gpu_phase") or {}).get("so_sha256"),
               "counts": T, "model": "blackwell_sm120_e4m3_m16n8k32", "model_agrees": not failures, "seed": res.get("seed"),
               "workload_fingerprint": {"probe": "tools/tc_probe_fp4/fp8_gpucheck.py", "phase": "gate", "seed": res.get("seed"), "units": res.get("units")},
               "validation": {"status": res["validation"], "checker": "tools/tc_probe_fp4/fp8_gpucheck.py --phase gate: GPU model words vs "
                              "fp8.evaluate and kernels.group_sum_total_batch on the existing captures; fused launches vs the numpy generator "
                              "and the CPU models", "detail": "; ".join(failures) or "every GPU word equals the CPU's", "evidence": ["gate_results.json"]},
               "measurements": meas, "artifacts": sorted(p.name for p in out.iterdir() if p.suffix in (".json", ".txt") and p.name != "result.json")}
        _json(out / "result.json", doc)
    print(json.dumps({"phase": "gate", "validation": res["validation"], "failures": failures[:10], "seconds": res["seconds"]}))
    return 0 if not failures else 1


# --- fill units -------------------------------------------------------------------------------------------------------------------

def launch_path(unit: Path, fam: str, i: int) -> Path:
    return unit / f"L_{fam}_{i:05d}.npz"


def launch_span(plan: dict, i: int) -> tuple[int, int]:
    """(first, count) of the tiles (chain: the chains) launch i covers."""
    n = plan["chains"] if plan["kind"] == "chain" else plan["blocks"] * WPB * plan["per_warp"]
    return i * n, n


def launch_control(plan: dict, i: int) -> tuple[int, int] | None:
    """(lo, hi) of the tile whose element 0 launch 0 corrupts (the negative control): step tile first + 17, chain first + 3's step 11."""
    if i:
        return None
    first, _ = launch_span(plan, i)
    return (first + 3, 11) if plan["kind"] == "chain" else ((first + 17) & 0xFFFFFFFF, (first + 17) >> 32)


def fill_gpu(args) -> int:
    """A gpus=1 fill chunk: the gates, then the unit's missing launches in order until --budget-s is spent (exit 99) or none is left
    (exit 0)."""
    t0 = time.time()
    unit = Path(args.unit)
    plan = json.loads((unit / "plan.json").read_text())
    so = Path(args.prebuilt)
    dev = GpuCheck(so)
    ident, failures = F8.identity_gate(dev, args.expect_gpu_uuid)
    sass = sass_gate(so)
    if not sass["ok"]:
        failures.append("SASS gate failed")
    chunk = {"start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu_uuid": ident.get("gpu_uuid"), "failures": failures}
    if failures:
        _json(unit / f"chunk_failed_{int(t0)}.json", {**chunk, "identity": ident, "sass": sass})
        print(json.dumps(chunk))
        return 1
    specs = [CONTROLS[c] for c in plan["specs"]]
    done = 0
    for name in plan["families"]:
        fam, tabs = family(plan["kind"], name, plan["seed"])
        if tables_digest(tabs) != plan["tables_sha256"][name]:
            print(f"tables of {name} differ from the plan's", file=sys.stderr)
            return 1
        for i in range(plan["launches"]):
            p = launch_path(unit, name, i)
            if p.exists():
                continue
            if time.time() - t0 > args.budget_s:
                print(json.dumps({**chunk, "launches": done, "left": True}))
                return 99
            first, count = launch_span(plan, i)
            ctl = launch_control(plan, i)
            if plan["kind"] == "chain":
                r = dev.fused_chain(fam, tabs, specs, first, count, plan["steps"], plan["smask"], (ctl[0] << 32) | ctl[1] if ctl else -1,
                                    rec_chain=first, samp_cap=plan["samp_cap"], ndump=plan["ndump"])
            else:
                r = dev.fused_step(plan["kind"], fam, tabs, specs, first, plan["blocks"], plan["per_warp"], plan["smask"],
                                   (ctl[1] << 32) | ctl[0] if ctl else -1, samp_cap=plan["samp_cap"], ndump=plan["ndump"])
            arrays = {k: v for k, v in r.items() if isinstance(v, np.ndarray)}
            summary = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)} | {"family": name, "launch": i, "gpu_uuid": ident.get("gpu_uuid")}
            tmp = p.with_name(p.name + ".tmp")
            with open(tmp, "wb") as f:
                np.savez(f, summary=json.dumps(summary), **arrays)
            os.replace(tmp, p)
            done += 1
    print(json.dumps({**chunk, "launches": done, "left": False}))
    return 0


def _verify_launch(job: tuple[str, str]) -> dict:
    """CPU: one launch file of a fill unit through check_launch (and, for chains, the replay of the chain it recorded)."""
    unit, path = Path(job[0]), Path(job[1])
    plan = json.loads((unit / "plan.json").read_text())
    kind = plan["kind"]
    with np.load(path) as z:
        r = {k: z[k] for k in z.files}
    s = json.loads(str(r["summary"]))
    name, i = s["family"], s["launch"]
    fam, tabs = family(kind, name, plan["seed"])
    specs = [CONTROLS[c] for c in plan["specs"]]
    first, count = launch_span(plan, i)
    out = check_launch(kind, fam, tabs, specs, r, s, plan["smask"], launch_tiles(kind, first, count, plan.get("steps", 0)), launch_control(plan, i))
    want = count * plan["steps"] if kind == "chain" else count
    if s["tiles"] != want:
        out["failures"].append(f"the launch covered {s['tiles']} tiles, not {want}")
    if kind == "chain":
        out["recorded_chain"] = check_recorded_chain(fam, tabs, first, r["rec"], r)
        out["failures"] += out["recorded_chain"]["failures"]
    return {"launch": path.name, "family": name, "i": i, **{k: s.get(k) for k in ("tiles", "elements", "gated", "mismatches", "record_only",
                                                                                  "device_ms", "gpu_uuid")}, **out}


def fill_verify(args) -> int:
    """A gpus=0 fill chunk: every launch file without a verify record, in worker processes, until --budget-s is spent (exit 99)."""
    t0 = time.time()
    unit = Path(args.unit)
    todo = [p for p in sorted(unit.glob("L_*.npz")) if not (unit / f"V_{p.stem}.json").exists()]
    with Pool(args.workers) as pool:
        for rec in pool.imap(_verify_launch, [(str(unit), str(p)) for p in todo]):
            _json(unit / f"V_{Path(rec['launch']).stem}.json", rec)
            if time.time() - t0 > args.budget_s:
                break
    plan = json.loads((unit / "plan.json").read_text())
    want = [launch_path(unit, f, i) for f in plan["families"] for i in range(plan["launches"])]
    left = [p for p in want if not (unit / f"V_{p.stem}.json").exists()]
    if left:
        print(json.dumps({"unit": str(unit), "left": len(left)}))
        return 99
    recs = [json.loads((unit / f"V_{p.stem}.json").read_text()) for p in want]
    fails = [f"{r['launch']}: {f}" for r in recs for f in r["failures"]]
    tot = {k: sum(r[k] for r in recs) for k in ("tiles", "elements", "gated", "kept", "sampled_tiles", "sampled_words", "flags",
                                               "primary_gated_beyond_control", "primary_record_only_beyond_control")}
    for k in ("mismatches", "record_only", "cpu_sample_mismatches"):
        tot[k] = dict(zip(plan["specs"], (sum(r[k][m] for r in recs) for m in range(len(plan["specs"])))))
    tot["device_seconds"] = round(sum(r["device_ms"] for r in recs) / 1e3, 3)
    tot["recorded_chain_words"] = sum(r["recorded_chain"]["steps"] * 128 for r in recs if r.get("recorded_chain"))
    controls = [r["negative_control_flagged"] for r in recs if r["negative_control_flagged"] is not None]
    if tot["primary_gated_beyond_control"]:
        fails.append(f"the primary flags {tot['primary_gated_beyond_control']} gated words beyond the negative controls")
    summary = {"unit": str(unit), "plan": plan, "totals": tot, "negative_controls": len(controls), "negative_controls_flagged": sum(controls),
               "failures": fails, "validation": "passed" if not fails else "failed"}
    _json(unit / "unit_verify.json", summary)
    print(json.dumps({"unit": str(unit), "validation": summary["validation"], "failures": fails[:5]}))
    return 0 if summary["validation"] == "passed" else 1


def make_plan(unit: Path, kind: str, families: list[str], seed: int, launches: int, specs: list[str], smask_bits: int, blocks: int = 376,
              per_warp: int = 512, chains: int = 3008, steps: int = 32768, samp_cap: int = 8192, ndump: int = 8) -> dict:
    plan = {"kind": kind, "families": families, "seed": seed, "launches": launches, "specs": specs, "smask": (1 << smask_bits) - 1,
            "samp_cap": samp_cap, "ndump": ndump, "tables_sha256": {f: tables_digest(family(kind, f, seed)[1]) for f in families}}
    if kind == "chain":
        plan.update(chains=chains, steps=steps)
    else:
        plan.update(blocks=blocks, per_warp=per_warp)
    unit.mkdir(parents=True, exist_ok=True)
    _json(unit / "plan.json", plan)
    return plan


# --- CLI --------------------------------------------------------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase", required=True, choices=["compile", "gate", "gate-gpu", "fill-gpu", "fill-verify"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--unit", default=None, help="fill phases: the unit directory holding plan.json")
    ap.add_argument("--seed", type=int, default=20261001, help="gate: the fused smoke launches' seed")
    ap.add_argument("--prebuilt", default=None)
    ap.add_argument("--expect-gpu-uuid", default=os.environ.get("GPU_LEASE_UUID"))
    ap.add_argument("--capture-root", default=str(FILL_OUT), help="gate: where fp8-capture2/ and fp8-chain/ are")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--units", default=None, help="gate-gpu: units.json")
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget-s", type=float, default=330.0, help="GPU chunks: stop starting work after this long and exit 99")
    ap.add_argument("--lease-via-fill", default=None, help="gate: the fill owner the GPU half runs under")
    ap.add_argument("--lease-max-min", type=int, default=8)
    ap.add_argument("--max-leases", type=int, default=12)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for k in ("out", "unit", "scratch", "prebuilt", "units", "capture_root"):
        if getattr(args, k):
            setattr(args, k, os.path.expandvars(getattr(args, k)))
    if args.phase == "compile":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        so, nvcc = build(out)
        sass = sass_gate(so, out)
        _json(out / "compile.json", {"nvcc": nvcc, "sass": sass, "so": so.name, "so_sha256": hashlib.sha256(so.read_bytes()).hexdigest()})
        print(json.dumps({"phase": "compile", "sass_ok": sass["ok"]}))
        return 0 if sass["ok"] else 1
    if args.phase == "gate":
        if not args.lease_via_fill:
            print("fp8_gpucheck.py: --phase gate needs --lease-via-fill OWNER", file=sys.stderr)
            return 2
        return gate(args)
    if args.phase == "gate-gpu":
        return gate_gpu(args)
    if args.phase == "fill-gpu":
        return fill_gpu(args)
    return fill_verify(args)


if __name__ == "__main__":
    sys.exit(main())
