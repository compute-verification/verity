#!/usr/bin/env python3
"""sm_120 FP8 rows of the Blackwell tensor-core probe (the RTX PRO 6000 FP8 capture): the gaps the single-step capture of lane
vllm-sm120-tc-gemm (``tools/tc_probe`` ``sm120.mma.m16n8k32.e4m3``/``.e5m2``, model ``BLACKWELL_SM120_E4M3_M16N8K32``) leaves.

    python tools/tc_probe_fp4/fp8.py --row ROW --out $RESEARCH_RUN_DIR --expect-gpu-uuid GPU-... [--seed S] [--n-random N]

Rows (kernels in ``mma_fp8.cu``, built ``-gencode arch=compute_120a,code=sm_120a``):
    compile   build + SASS gate only (no GPU): every kernel holds exactly its intended tensor-core / F2FP / FADD instructions
    identity  build + SASS + device gates only, and no result.json: the UUID check in front of another probe in the same run
              (``--out $RESEARCH_RUN_DIR/identity``)
    step      one mma.sync per tile over families, against a candidate grid (``--instruction``: f8_e4m3, legacy_e4m3, f8_e5m2,
              f8_e4m3e5m2, mxf8_e4m3); the primary model is the registered sm_120 step where one exists, else the hypothesis named
    chain     dependent E4M3 k32 chains in registers up to K = 32 * --steps (2^16 at 2048), every step's word replayed from the
              device's previous word, plus freezing facts measured on the device words (exact steps, frozen steps, error vs exact)
    mixed     FP8 atom -> BF16 peel chains (#453's mixed chain) on one accumulator: patterns qb, bq, qqqqb x 8 and random
    cvt       cvt.rn.satfinite.e4m3x2.f32 and .e5m2x2.f32 over a structured grid + random words, against the H100-measured cast
              (``verity.ml.tc.cast``) and its E5M2 analogue
    fadd      add.rn.f32 and add.rn.ftz.f32 over structured pairs (subnormals, ties, cancellation, overflow) + random words

Gates, in order, before anything is measured: the device (exactly one visible, its UUID equal to ``--expect-gpu-uuid``, which
defaults to ``gpu-lease``'s ``GPU_LEASE_UUID``), the SASS
gate, the exact small-integer layout check of every kernel the row uses.  A failed gate writes the gate report and no
measurements.  Non-finite results (NaN / inf operands or accumulators, sums leaving binary32, NaN adds and casts) are recorded and
never gate.  ``--fake-silicon`` replaces the device with the row's primary model (a CPU dry run; refused under research run) and
``--corrupt-word I`` flips one bit of the I-th output word of the row's first capture: the run must then fail.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tc_probe"))
_VERITY_SRC = HERE.parents[1] / "packages" / "verity" / "src"
if _VERITY_SRC.is_dir() and str(_VERITY_SRC) not in sys.path:
    sys.path.insert(0, str(_VERITY_SRC))

import probe as P4  # noqa: E402  (build, nvcc_bin)
import tc_probe as TP  # noqa: E402  (the FP8 families and encoders of lane vllm-sm120-tc-gemm's capture)
from verity.ml.kernels import _bitlen, _pack_finite, group_sum_total_batch  # noqa: E402
from verity.ml.tc import models as TCM  # noqa: E402
from verity.ml.tc.cast import f32_to_e4m3_sat_word  # noqa: E402
from verity.ml.tc.term import PRODUCT_WIDTH  # noqa: E402

ARCH = "sm_120a"
SOURCE = "mma_fp8.cu"
SEED_DEFAULT = 20260930
MAX_TILES = 1 << 14
NAN_WORD, INF_WORD, SIGN = 0x7FFFFFFF, 0x7F800000, 0x80000000

TILES = {  # --instruction: kernel selector, operand formats, PTX, instruction id
    "f8_e4m3": (0, "e4m3", "e4m3", "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32", "sm120.mma.m16n8k32.e4m3.f8f6f4"),
    "legacy_e4m3": (3, "e4m3", "e4m3", "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", "sm120.mma.m16n8k32.e4m3"),
    "f8_e5m2": (1, "e5m2", "e5m2", "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e5m2.f32", "sm120.mma.m16n8k32.e5m2.f8f6f4"),
    "f8_e4m3e5m2": (2, "e4m3", "e5m2", "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e5m2.f32", "sm120.mma.m16n8k32.e4m3.e5m2"),
    "mxf8_e4m3": (4, "e4m3", "e4m3", "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0",
                  "sm120.mma.m16n8k32.e4m3.mxf8"),
}
ROW_ID = {"chain": "sm120.mma.m16n8k32.e4m3.chain", "mixed": "sm120.mma.e4m3-bf16.mixed-chain",
          "cvt": "sm120.cvt.rn.satfinite.f8x2.f32", "fadd": "sm120.add.rn.f32"}
SASS_EXPECT = {  # kernel -> its exact multiset of tensor-core / conversion / FP32-add instructions
    "f8_e4m3_tiles_kernel": ["QMMA.16832.F32.E4M3.E4M3"],
    "legacy_e4m3_tiles_kernel": ["QMMA.16832.F32.E4M3.E4M3"],
    "f8_e5m2_tiles_kernel": ["QMMA.16832.F32.E5M2.E5M2"],
    "f8_e4m3e5m2_tiles_kernel": ["QMMA.16832.F32.E4M3.E5M2"],
    "mxf8_e4m3_tiles_kernel": ["QMMA.SF.16832.F32.E4M3.E4M3.E8"],
    "e4m3_chain_kernel": ["QMMA.16832.F32.E4M3.E4M3"],
    "mixed_chain_kernel": ["HMMA.16816.F32.BF16", "QMMA.16832.F32.E4M3.E4M3"],
    "cvt_e4m3x2_kernel": ["F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C"],
    "cvt_e5m2x2_kernel": ["F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C"],
    "fadd_rn_kernel": ["FADD"],
    "fadd_rn_ftz_kernel": ["FADD.FTZ"],
}
_WATCHED = re.compile(r"^(?:[A-Z]*MMA|F2FP|F2F|FADD|FFMA|FMUL|FMNMX)(?:\.|$)")
FMT = {"e4m3": (4, 3, 7, "e4m3"), "e5m2": (5, 2, 15, "ieee"), "bf16": (8, 7, 127, "ieee")}
TPF = {"e4m3": TP.E4M3, "e5m2": TP.E5M2, "bf16": TP.BF16}


# --- build and SASS gate ------------------------------------------------------------------------------------------------------------


def sass_gate(so: Path, out: Path | None = None) -> dict:
    """``cuobjdump -sass``: per kernel, the watched opcodes (predicates stripped); ok iff each kernel's multiset equals SASS_EXPECT."""
    cuobjdump = str(Path(P4.nvcc_bin()).parent / "cuobjdump")
    text = subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    found: dict[str, list[str]] = {}
    fn = None
    for line in text.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            fn = next((k for k in SASS_EXPECT if f"{len(k)}{k}E" in m.group(1)), m.group(1))
            found[fn] = []
            continue
        if fn and "*/" in line:
            body = line.split("*/", 1)[1].strip()
            toks = body.split()
            if toks and toks[0].startswith("@"):
                toks = toks[1:]
            if toks and _WATCHED.match(toks[0]):
                found[fn].append(toks[0])
    kernels = {k: {"opcodes": sorted(found.get(k, [])), "expected": sorted(v), "ok": sorted(found.get(k, [])) == sorted(v)}
               for k, v in SASS_EXPECT.items()}
    if out is not None:
        (out / "sass_mma_fp8.txt").write_text(text)
    return {"ok": all(r["ok"] for r in kernels.values()), "kernels": kernels, "sass_sha256": hashlib.sha256(text.encode()).hexdigest()}


# --- the device ---------------------------------------------------------------------------------------------------------------------


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


class Device:
    """ctypes wrapper over libmma_fp8: tiles, chains, casts, adds; raw bits in and out."""

    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        for name in ("fp8_tiles", "e4m3_chain", "mixed_chain", "cvt_x2", "fadd_f32", "device_info"):
            getattr(self.lib, name).restype = ctypes.c_int

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in {what}")

    def info(self) -> dict:
        major, minor, sms = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        uuid = (ctypes.c_ubyte * 16)()
        self._check(self.lib.device_info(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(sms), name, 256, uuid), "device_info")
        b = bytes(uuid).hex()
        return {"device": name.value.decode(), "compute_capability": f"{major.value}.{minor.value}", "sms": sms.value,
                "gpu_uuid": f"GPU-{b[0:8]}-{b[8:12]}-{b[12:16]}-{b[16:20]}-{b[20:32]}"}

    def tiles(self, instr: str, A, Bt, C, SFA=None, SFB=None) -> np.ndarray:
        which = TILES[instr][0]
        n = A.shape[0]
        A = np.ascontiguousarray(A, np.uint8).reshape(n, 16, 32)
        Bt = np.ascontiguousarray(Bt, np.uint8).reshape(n, 8, 32)
        C = np.ascontiguousarray(C, np.uint32).reshape(n, 16, 8)
        SFA = np.ascontiguousarray(np.full((n, 16), 127, np.uint8) if SFA is None else SFA, np.uint8).reshape(n, 16)
        SFB = np.ascontiguousarray(np.full((n, 8), 127, np.uint8) if SFB is None else SFB, np.uint8).reshape(n, 8)
        D = np.zeros((n, 16, 8), np.uint32)
        for s in range(0, n, MAX_TILES):
            e = min(n, s + MAX_TILES)
            d = np.zeros((e - s, 16, 8), np.uint32)
            a, b, c, sa, sb = (np.ascontiguousarray(x[s:e]) for x in (A, Bt, C, SFA, SFB))
            self._check(self.lib.fp8_tiles(which, _ptr(a), _ptr(b), _ptr(sa), _ptr(sb), _ptr(c), _ptr(d), e - s), instr)
            D[s:e] = d
        return D

    def e4m3_chain(self, A, Bt, C0) -> np.ndarray:
        n, steps = A.shape[:2]
        A, Bt, C0 = (np.ascontiguousarray(x) for x in (A.astype(np.uint8), Bt.astype(np.uint8), C0.astype(np.uint32)))
        D = np.zeros((n, steps, 16, 8), np.uint32)
        self._check(self.lib.e4m3_chain(_ptr(A), _ptr(Bt), _ptr(C0), _ptr(D), n, steps), "e4m3_chain")
        return D

    def mixed_chain(self, kinds, A8, B8, A16, B16, C0) -> np.ndarray:
        n, steps = kinds.shape
        arrs = [np.ascontiguousarray(x) for x in (A8.astype(np.uint8), B8.astype(np.uint8), A16.astype(np.uint16), B16.astype(np.uint16),
                                                   kinds.astype(np.uint8), C0.astype(np.uint32))]
        D = np.zeros((n, steps, 16, 8), np.uint32)
        self._check(self.lib.mixed_chain(*(_ptr(x) for x in arrs), _ptr(D), n, steps), "mixed_chain")
        return D

    def cvt(self, fmt: str, X) -> np.ndarray:
        X = np.ascontiguousarray(X, np.uint32)
        Y = np.zeros(X.size // 2, np.uint16)
        self._check(self.lib.cvt_x2(0 if fmt == "e4m3" else 1, _ptr(X), _ptr(Y), Y.size), f"cvt {fmt}")
        return Y.view(np.uint8).copy()

    def fadd(self, ftz: bool, X, Y) -> np.ndarray:
        X, Y = np.ascontiguousarray(X, np.uint32), np.ascontiguousarray(Y, np.uint32)
        Z = np.zeros(X.size, np.uint32)
        self._check(self.lib.fadd_f32(int(ftz), _ptr(X), _ptr(Y), _ptr(Z), X.size), "fadd")
        return Z


class FakeDevice:
    """The row's primary model standing in for silicon (the CPU dry run); ``corrupt`` flips one bit of one output word once."""

    def __init__(self, corrupt: int | None = None):
        self.corrupt = corrupt
        self.armed = False  # run() arms the corruption once the gates have passed, so it lands in the capture

    def _maybe_corrupt(self, out: np.ndarray) -> np.ndarray:
        if self.corrupt is not None and self.armed:
            flat = out.reshape(-1)
            flat[self.corrupt % flat.size] ^= 1
            self.corrupt = None
        return out

    def info(self) -> dict:
        return {"device": "fake-silicon", "compute_capability": "12.0", "sms": 0, "gpu_uuid": "GPU-fake"}

    def tiles(self, instr, A, Bt, C, SFA=None, SFB=None):
        _, fa, fb, _, _ = TILES[instr]
        n = A.shape[0]
        a, b, acc = expand(A, Bt, C)
        scale = None if instr != "mxf8_e4m3" else scale_expand(SFA, SFB, n)
        d = groupsum(PRIMARY[instr][1], acc, products(fa, a, fb, b, scale))
        return self._maybe_corrupt(d.reshape(n, 16, 8))

    def e4m3_chain(self, A, Bt, C0):
        n, steps = A.shape[:2]
        D = np.zeros((n, steps, 16, 8), np.uint32)
        acc = C0.reshape(n, 16, 8)
        for s in range(steps):
            acc = step_model(TCM.BLACKWELL_SM120_E4M3_M16N8K32, A[:, s], Bt[:, s], acc)
            D[:, s] = acc
        return self._maybe_corrupt(D)

    def mixed_chain(self, kinds, A8, B8, A16, B16, C0):
        n, steps = kinds.shape
        D = np.zeros((n, steps, 16, 8), np.uint32)
        acc = C0.reshape(n, 16, 8)
        for s in range(steps):
            q = step_model(TCM.BLACKWELL_SM120_E4M3_M16N8K32, A8[:, s], B8[:, s], acc)
            b = step_model(TCM.HOPPER_BF16_M16N8K16, A16[:, s], B16[:, s], acc)
            acc = np.where((kinds[:, s] == 0)[:, None, None], q, b)
            D[:, s] = acc
        return self._maybe_corrupt(D)

    def cvt(self, fmt, X):
        return self._maybe_corrupt(cvt_ref(fmt, X))

    def fadd(self, ftz, X, Y):
        return self._maybe_corrupt(fadd_ref(X, Y, ftz))


# --- models -------------------------------------------------------------------------------------------------------------------------


def expand(A, Bt, C):
    """Tiles -> per element (tile, i, j) row-major: a (N,32) = A[t,i,:], b (N,32) = Bt[t,j,:], acc (N,) = C[t,i,j]."""
    n, k = A.shape[0], A.shape[-1]
    a = np.broadcast_to(A.reshape(n, 16, 1, k), (n, 16, 8, k)).reshape(-1, k)
    b = np.broadcast_to(Bt.reshape(n, 1, 8, k), (n, 16, 8, k)).reshape(-1, k)
    return a, b, C.reshape(-1).astype(np.uint32)


def scale_expand(SFA, SFB, n):
    """UE8M0 scale bytes -> per element (N,) exponent 2^(sfa - 127) * 2^(sfb - 127), NaN (0xFF) as a flag in the high bits."""
    sa = np.broadcast_to(SFA.reshape(n, 16, 1).astype(np.int64), (n, 16, 8)).reshape(-1)
    sb = np.broadcast_to(SFB.reshape(n, 1, 8).astype(np.int64), (n, 16, 8)).reshape(-1)
    return np.where((sa == 0xFF) | (sb == 0xFF), 1 << 40, sa + sb - 254)


def operands(fmt: str, w):
    eb, mb, bias, rule = FMT[fmt]
    w = np.asarray(w).astype(np.int64)
    e = (w >> mb) & ((1 << eb) - 1)
    m = w & ((1 << mb) - 1)
    ones = e == (1 << eb) - 1
    if rule == "e4m3":
        nan, inf = ones & (m == (1 << mb) - 1), np.zeros(w.shape, bool)
    else:
        nan, inf = ones & (m != 0), ones & (m == 0)
    sig = m | ((e != 0).astype(np.int64) << mb)
    return nan, inf, sig, np.maximum(e, 1) - bias - mb, (w >> (eb + mb)) & 1


def products(fa: str, a, fb: str, b, scale=None):
    """Exact products as ``term.Format.product`` Terms, vectorized: magnitude (sa' * sb') << (PRODUCT_WIDTH - 2S) with both
    significands widened to S = max significand bits, so 1.0 x 1.0 sits at bit 23 whatever the formats (E4M3 x E5M2 is aligned as
    E4M3 x E4M3 is); ``scale`` (N,) adds 2^scale to every product of an element (UE8M0; >= 2^40 marks a NaN scale)."""
    na, ia, ga, ua, sa = operands(fa, a)
    nb, ib, gb, ub, sb = operands(fb, b)
    siga, sigb = FMT[fa][1] + 1, FMT[fb][1] + 1
    S = max(siga, sigb)
    nan = na | nb | (ia & (gb == 0) & ~nb) | (ib & (ga == 0) & ~na)
    if scale is not None:
        nan = nan | (scale >= (1 << 40))[:, None]
    inf = (ia | ib) & ~nan
    fin = ~(nan | inf)
    mag = np.where(fin, ((ga << (S - siga)) * (gb << (S - sigb))) << (PRODUCT_WIDTH - 2 * S), 0)
    exp = ua + ub + siga + sigb - 2
    if scale is not None:
        exp = exp + np.where(scale >= (1 << 40), 0, scale)[:, None]
    return mag, exp, (sa ^ sb) == 1, nan, inf


def groupsum(spec: dict, acc, prods) -> np.ndarray:
    """A GroupSum step (``kernels.group_sum_total_batch``'s arithmetic) over given products: ``spec`` = groups, width, floor and
    ``post_scale`` (the scale applied to each group's product sum before the accumulator joins: the alternative to pre-scaled products)."""
    p_mag, p_exp, p_neg, p_nan, p_inf = prods
    groups, width, floor = spec["groups"], spec["width"], spec["floor"]
    acc = np.asarray(acc).astype(np.uint32).astype(np.int64)
    a_field, a_frac = (acc >> 23) & 0xFF, acc & 0x7FFFFF
    state = np.where(a_field == 0xFF, np.where(a_frac != 0, 1, 2), 0)
    neg = (acc >> 31) == 1
    mag = np.where(state == 0, np.where(a_field == 0, a_frac, a_frac | (1 << 23)), 0)
    exp = np.where(a_field == 0, -126, a_field - 127)
    rescale = width - 24
    start = 0
    for size in groups:
        sl = slice(start, start + size)
        start += size
        pos_inf = ((state == 2) & ~neg) | (p_inf[:, sl] & ~p_neg[:, sl]).any(-1)
        neg_inf = ((state == 2) & neg) | (p_inf[:, sl] & p_neg[:, sl]).any(-1)
        is_nan = (state == 1) | p_nan[:, sl].any(-1) | (pos_inf & neg_inf)
        is_inf = ~is_nan & (pos_inf | neg_inf)
        mags = np.concatenate([mag[:, None], p_mag[:, sl]], axis=1)
        exps = np.concatenate([exp[:, None], p_exp[:, sl]], axis=1)
        negs = np.concatenate([neg[:, None], p_neg[:, sl]], axis=1)
        live = mags != 0
        top = np.maximum(np.where(live, exps, floor).max(axis=1), floor)
        scaled = mags << rescale if rescale >= 0 else mags >> -rescale
        aligned = np.where(live, scaled >> np.clip(top[:, None] - exps, 0, 63), 0)
        total = np.where(negs, -aligned, aligned).sum(axis=1)
        s_neg, s_mag = total < 0, np.abs(total)
        bl = _bitlen(s_mag)
        s_exp = top + bl - width
        s_mag = np.where(bl > width, s_mag >> np.clip(bl - width, 0, 63), s_mag << np.clip(width - bl, 0, 63))
        s_mag >>= np.clip(-126 - s_exp, 0, 63)
        s_exp = np.maximum(s_exp, -126)
        s_mag = s_mag >> rescale if rescale >= 0 else s_mag << -rescale
        over = ~is_nan & ~is_inf & (s_mag != 0) & (s_exp > 127)
        new_state = np.where(is_nan, 1, np.where(is_inf | over, 2, 0))
        neg = np.where(is_inf, neg_inf, s_neg & (s_mag != 0))
        mag = np.where(new_state == 0, s_mag, 0)
        exp, state = s_exp, new_state
    word = _pack_finite(neg, mag, np.minimum(exp, 127))
    word = np.where(state == 2, np.where(neg, SIGN | INF_WORD, INF_WORD), word)
    return np.where(state == 1, NAN_WORD, word).astype(np.uint32)


def post_scale_groupsum(spec: dict, acc, fa, a, fb, b, scale) -> np.ndarray:
    """Alternative block-scaled datapath: the unscaled products summed alone (a zero accumulator), the FP32 sum scaled by 2^scale,
    then added to the accumulator as one term."""
    s = groupsum(spec, np.zeros_like(acc), products(fa, a, fb, b)).astype(np.int64)
    fin = (s & 0x7F800000) != 0x7F800000
    field, frac = (s >> 23) & 0xFF, s & 0x7FFFFF
    mag = np.where(fin, np.where(field == 0, frac, frac | (1 << 23)), 0)[:, None]
    exp = (np.where(field == 0, -126, field - 127) + np.where(scale >= (1 << 40), 0, scale))[:, None]
    nan = ((s & 0x7FFFFFFF) > 0x7F800000)[:, None] | (scale >= (1 << 40))[:, None]
    inf = (~fin)[:, None] & ~nan
    return groupsum({**spec, "groups": (1,)}, acc, (mag, exp, ((s >> 31) == 1)[:, None], nan, inf))


def step_model(model, A, Bt, C) -> np.ndarray:
    """A registered GroupSum on tiles (the core batch kernel)."""
    n = A.shape[0]
    a, b, acc = expand(A, Bt, C)
    return group_sum_total_batch(model, acc, a, b)[0].reshape(n, 16, 8)


def spec(groups, width, floor, post_scale=False) -> dict:
    return {"groups": tuple(groups), "width": width, "floor": floor, "post_scale": post_scale}


SM120 = spec((32,), 26, -133)
PRIMARY = {  # --instruction -> (name, spec): the registered step for the unscaled rows, the hypothesis for the others
    "f8_e4m3": ("blackwell_sm120_e4m3_m16n8k32", SM120),
    "legacy_e4m3": ("blackwell_sm120_e4m3_m16n8k32", SM120),
    "f8_e5m2": ("blackwell_sm120_e5m2_m16n8k32", SM120),
    "f8_e4m3e5m2": ("HYPOTHESIS sm120 (32,) w26 f-133, e4m3 x e5m2 products", SM120),
    "mxf8_e4m3": ("HYPOTHESIS sm120 (32,) w26 f-133, products pre-scaled by 2^(sfa+sfb-254)", SM120),
}


def candidates(instr: str) -> dict[str, dict]:
    """The candidate grid the row discriminates: the primary first, then +-1 in width and group structure, Ada's and Hopper's 14-bit
    FP8 adders, and floors above the band [-inf, -124] that FP32 accumulators cannot see into (block-scaled: products reach it)."""
    c = {PRIMARY[instr][0]: PRIMARY[instr][1],
         "(32,) w25 f-133": spec((32,), 25, -133), "(32,) w27 f-133": spec((32,), 27, -133),
         "(16,16) w26 f-133": spec((16, 16), 26, -133), "(8,8,8,8) w26 f-133": spec((8, 8, 8, 8), 26, -133),
         "ada (16,16) w14 f-139": spec((16, 16), 14, -139), "hopper (32,) w14 f-139": spec((32,), 14, -139),
         "(32,) w26 f-123": spec((32,), 26, -123)}
    if instr == "mxf8_e4m3":
        for f in (-129, -128, -126, -124):
            c[f"(32,) w26 f{f} pre-scaled"] = spec((32,), 26, f)
        c["(32,) w26 f-133 post-scaled"] = spec((32,), 26, -133, post_scale=True)
    return c


def evaluate(instr: str, sp: dict, A, Bt, C, SFA, SFB) -> np.ndarray:
    _, fa, fb, _, _ = TILES[instr]
    n = A.shape[0]
    out = np.zeros(n * 128, np.uint32)
    for s in range(0, n, 512):
        e = min(n, s + 512)
        a, b, acc = expand(A[s:e], Bt[s:e], C[s:e])
        scale = scale_expand(SFA[s:e], SFB[s:e], e - s) if instr == "mxf8_e4m3" else None
        if sp.get("post_scale"):
            out[s * 128:e * 128] = post_scale_groupsum(sp, acc, fa, a, fb, b, scale)
        else:
            out[s * 128:e * 128] = groupsum(sp, acc, products(fa, a, fb, b, scale))
    return out.reshape(n, 16, 8)


# --- scalar references --------------------------------------------------------------------------------------------------------------

CVT = {"e4m3": (3, 7, 448.0), "e5m2": (2, 15, 57344.0)}  # mantissa bits, bias, largest finite


def cvt_ref(fmt: str, X) -> np.ndarray:
    """cvt.rn.satfinite.<fmt>.f32 per element: round to nearest even on the format's grid (subnormals included), |x| rounding above
    the largest finite (and +-inf) -> +-largest finite, NaN -> 0x7F, a zero result keeps the sign (``cast.f32_to_e4m3_sat_word``)."""
    mb, bias, maxfin = CVT[fmt]
    X = np.asarray(X, np.uint32)
    x = X.view(np.float32).astype(np.float64)
    sign = (X >> 31).astype(np.uint8)
    with np.errstate(all="ignore"):
        a = np.abs(x)
        finite = np.isfinite(a)
        _, e = np.frexp(np.where(finite, a, 1.0))
        emin = 1 - bias
        q = np.ldexp(1.0, np.maximum(e - 1, emin) - mb)
        r = np.where(finite, np.rint(np.where(finite, a, 0.0) / q) * q, np.inf)
        r = np.where(r > maxfin, maxfin, r)
        _, er = np.frexp(np.where(r > 0, r, 1.0))
        er = er - 1
        sub = r < 2.0**emin
        field = np.where(sub, 0, er + bias)
        mant = np.where(sub, r / 2.0 ** (emin - mb), r / np.ldexp(1.0, er - mb) - 2**mb)
    code = (sign << 7) | (field.astype(np.int64) << mb).astype(np.uint8) | mant.astype(np.int64).astype(np.uint8)
    return np.where(np.isnan(x), 0x7F, code).astype(np.uint8)


def _flush(w: np.ndarray) -> np.ndarray:
    return np.where((w & 0x7F800000) == 0, w & SIGN, w).astype(np.uint32)


def fadd_ref(X, Y, ftz: bool) -> np.ndarray:
    """add.rn.f32 (IEEE, subnormals kept) / add.rn.ftz.f32 (subnormal inputs and results -> zero of the same sign); NaN results as
    the device's canonical 0x7FFFFFFF."""
    X, Y = np.asarray(X, np.uint32), np.asarray(Y, np.uint32)
    if ftz:
        X, Y = _flush(X), _flush(Y)
    with np.errstate(all="ignore"):
        z = (X.view(np.float32) + Y.view(np.float32)).view(np.uint32)
    if ftz:
        z = _flush(z)
    return np.where((z & 0x7FFFFFFF) > 0x7F800000, NAN_WORD, z).astype(np.uint32)


def host_self_check() -> None:
    """The host's float32 add keeps subnormals (no FTZ/DAZ set in this process), which fadd_ref relies on."""
    tiny = np.array([1], np.uint32).view(np.float32)
    if (tiny + np.float32(0)).view(np.uint32)[0] != 1:
        raise RuntimeError("host float32 arithmetic flushes subnormals; fadd_ref would be wrong")


# --- families -----------------------------------------------------------------------------------------------------------------------


def _f32(x) -> np.ndarray:
    return np.asarray(x, np.float32).view(np.uint32)


def _tiny_acc(rng, shape) -> np.ndarray:
    """FP32 accumulators aimed at the adder floor: subnormals and normals in [2^-126, 2^-100), random low bits, random sign."""
    n = int(np.prod(shape))
    e = np.where(rng.random(n) < 0.5, 0, rng.integers(1, 28, n)).astype(np.uint32)
    m = rng.integers(1, 1 << 23, n).astype(np.uint32)
    s = rng.integers(0, 2, n).astype(np.uint32)
    return ((s << 31) | (e << 23) | m).reshape(shape)


def fp8_extra_families(rng, fa: str, fb: str, n: int):
    """This row's families on top of the capture's: (name, A, Bt, C, SFA, SFB); scales None = unit."""
    FA, FB = TPF[fa], TPF[fb]
    # tiny / subnormal accumulators with zero products, with the smallest products (random count and sign), with exactly cancelling pairs
    A, Bt = np.zeros((n, 16, 32), np.uint8), np.zeros((n, 8, 32), np.uint8)
    third = n // 3
    A[third:2 * third, :, :] = np.where(rng.random((third, 16, 32)) < 0.5, FA.pow2(FA.min_pow2_exp), 0)
    A[third:2 * third] |= (rng.integers(0, 2, (third, 16, 32)) << 7).astype(np.uint8) * (A[third:2 * third] != 0)
    Bt[third:2 * third] = FB.pow2(FB.min_pow2_exp)
    pa = FA.random_finite_bits(rng, (n - 2 * third, 16, 16))
    A[2 * third:, :, :16], A[2 * third:, :, 16:] = pa, pa ^ 0x80
    Bt[2 * third:] = FB.random_finite_bits(rng, (n - 2 * third, 8, 1)) & 0x7F
    yield "acc_floor", A, Bt, _tiny_acc(rng, (n, 16, 8)), None, None
    # isolated subnormal products (one live k per tile): does the MMA flush subnormal operands?  C in {+0, -0, +-tiny, +-1}
    A, Bt = np.zeros((n, 16, 32), np.uint8), np.zeros((n, 8, 32), np.uint8)
    k0 = rng.integers(0, 32, n)
    sub_a = rng.integers(1, 1 << FMT[fa][1], (n, 16)).astype(np.uint8) | (rng.integers(0, 2, (n, 16)) << 7).astype(np.uint8)
    norm_b = FB.random_finite_bits(rng, (n, 8))
    b_sub = rng.integers(1, 1 << FMT[fb][1], (n, 8)).astype(np.uint8)
    use_sub_b = rng.random((n, 1)) < 0.5
    A[np.arange(n), :, k0] = sub_a
    Bt[np.arange(n), :, k0] = np.where(use_sub_b, b_sub, norm_b)
    accs = np.array([0, SIGN, 0x3F800000, 0xBF800000, 0x00000001, 0x80000001, 0x0D000000, 0x8D000000], np.uint32)
    yield "subnormal_isolated", A, Bt, accs[rng.integers(0, len(accs), (n, 16, 8))], None, None
    # wide spread: product 0 at 2^(Ea + Eb) (both operands' largest power of two), the other 31 at 2^-20 .. 2^-30 of it (a row
    # exponent times a column exponent, random signs), and an accumulator 2^-18 .. 2^-30 of it: the adder's width, row by row
    ea, eb = FA.max_pow2_exp, FB.max_pow2_exp
    ra = rng.integers(max(ea - 15, FA.min_pow2_exp), ea - 9, (n, 16))
    cb = rng.integers(max(eb - 15, FB.min_pow2_exp), eb - 9, (n, 8))
    A = np.array([FA.pow2(int(e)) for e in ra.reshape(-1)], np.uint8).reshape(n, 16, 1).repeat(32, -1)
    A ^= (rng.integers(0, 2, (n, 16, 32)) << 7).astype(np.uint8)
    Bt = np.array([FB.pow2(int(e)) for e in cb.reshape(-1)], np.uint8).reshape(n, 8, 1).repeat(32, -1)
    A[:, :, 0], Bt[:, :, 0] = FA.pow2(ea), FB.pow2(eb)
    C = _f32(np.exp2(ea + eb - rng.integers(18, 31, (n, 16, 8)).astype(np.float64)) * np.where(rng.random((n, 16, 8)) < 0.5, 1, -1))
    yield "wide_spread", A, Bt, C, None, None


def mxf8_families(rng, n: int):
    """Block-scaled only: scales over the whole UE8M0 range, scales that push products onto the floor candidates, NaN scales."""
    E = TP.E4M3
    A, Bt = E.random_finite_bits(rng, (n, 16, 32)), E.random_finite_bits(rng, (n, 8, 32))
    C = TP.random_finite_fp32_bits(rng, n * 128, 60, 200).reshape(n, 16, 8)
    yield "scale_sweep", A, Bt, C, rng.integers(0, 255, (n, 16)).astype(np.uint8), rng.integers(0, 255, (n, 8)).astype(np.uint8)
    # products of 2^-18 .. 2^-14 times 2^(sfa+sfb-254) in 2^[-154, -96]: they land on the grids 2^(floor-25) of every floor
    # candidate, and sums of <= 32 of them straddle 2^-149
    A = np.where(rng.random((n, 16, 32)) < 0.7, E.pow2(E.min_pow2_exp), E.random_finite_bits(rng, (n, 16, 32), 1, 3)).astype(np.uint8)
    Bt = np.full((n, 8, 32), E.pow2(E.min_pow2_exp), np.uint8)
    sfa = rng.integers(50, 80, (n, 16)).astype(np.uint8)
    sfb = rng.integers(50, 80, (n, 8)).astype(np.uint8)
    C = np.where(rng.random((n, 16, 8)) < 0.5, 0, _tiny_acc(rng, (n, 16, 8))).astype(np.uint32)
    yield "scale_floor", A, Bt, C, sfa, sfb
    A, Bt = E.random_finite_bits(rng, (n, 16, 32)), E.random_finite_bits(rng, (n, 8, 32))
    sfa = rng.integers(100, 150, (n, 16)).astype(np.uint8)
    sfa[:, ::5] = 0xFF
    yield "scale_nan", A, Bt, np.zeros((n, 16, 8), np.uint32), sfa, rng.integers(100, 150, (n, 8)).astype(np.uint8)


def step_families(rng, instr: str, n_random: int):
    _, fa, fb, _, _ = TILES[instr]
    if fa == fb:
        for name, A, Bt, C, _meta in TP.families(rng, n_random, TPF[fa], 16, 32):
            if instr == "mxf8_e4m3":  # the capture's families under unit scales for half the tiles, scales in 2^[-10, 10] for the rest
                n = A.shape[0]
                sfa = np.where(np.arange(n)[:, None] < n // 2, 127, rng.integers(117, 138, (n, 16))).astype(np.uint8)
                sfb = np.where(np.arange(n)[:, None] < n // 2, 127, rng.integers(117, 138, (n, 8))).astype(np.uint8)
                yield name, A, Bt, C, sfa, sfb
            else:
                yield name, A, Bt, C, None, None
    else:  # mixed formats: operands drawn in each format
        FA, FB = TPF[fa], TPF[fb]
        n = n_random
        yield "randn_zero_acc", FA.from_f32(rng.standard_normal((n, 16, 32))), FB.from_f32(rng.standard_normal((n, 8, 32))), \
            np.zeros((n, 16, 8), np.uint32), None, None
        yield "randn_random_acc", FA.from_f32(rng.standard_normal((n, 16, 32))), FB.from_f32(rng.standard_normal((n, 8, 32))), \
            TP.random_finite_fp32_bits(rng, n * 128, 100, 150).reshape(n, 16, 8), None, None
        yield "uniform_bits", FA.random_finite_bits(rng, (n, 16, 32)), FB.random_finite_bits(rng, (n, 8, 32)), \
            TP.random_finite_fp32_bits(rng, n * 128, 1, 254).reshape(n, 16, 8), None, None
        yield "subnormal", FA.random_finite_bits(rng, (n, 16, 32), 0, 1), FB.random_finite_bits(rng, (n, 8, 32), 0, 1), \
            np.zeros((n, 16, 8), np.uint32), None, None
    yield from fp8_extra_families(rng, fa, fb, max(96, n_random // 16))
    if instr == "mxf8_e4m3":
        yield from mxf8_families(rng, max(96, n_random // 8))


def chain_families(rng, tiles: int, steps: int):
    """(name, A (tiles, steps, 16, 32), Bt (tiles, steps, 8, 32), C0 (tiles, 16, 8)) for the dependent E4M3 chains."""
    E = TP.E4M3
    sh_a, sh_b = (tiles, steps, 16, 32), (tiles, steps, 8, 32)
    zero = np.zeros((tiles, 16, 8), np.uint32)
    yield "chain_gauss", E.from_f32(rng.standard_normal(sh_a) * 16), E.from_f32(rng.standard_normal(sh_b) * 16), zero
    acc = TP.random_finite_fp32_bits(rng, tiles * 128, 117, 157).reshape(tiles, 16, 8)
    yield "chain_gauss_acc", E.from_f32(rng.standard_normal(sh_a) * 16), E.from_f32(rng.standard_normal(sh_b) * 16), acc
    yield "chain_pos", E.random_finite_bits(rng, sh_a) & 0x7F, E.random_finite_bits(rng, sh_b) & 0x7F, zero
    yield "chain_unit", np.full(sh_a, E.pow2(0), np.uint8), np.full(sh_b, E.pow2(0), np.uint8), zero
    # 32 products of 2^-6 per step against accumulators at 2^20 (the 26-bit window's lsb is 2^-5: every product truncates away),
    # 2^20 - 64 (grows by 0.5 a step until it reaches 2^20 at step 128) and -2^20 (opposite sign)
    c0 = np.array([2.0**20, 2.0**20 - 64, -(2.0**20), 2.0**19], np.float32)
    yield "chain_absorb", np.full(sh_a, E.pow2(-3), np.uint8), np.full(sh_b, E.pow2(-3), np.uint8), \
        _f32(c0[rng.integers(0, 4, (tiles, 16, 8))])
    # Gaussian background with two aligned spikes (x 2^8) per k32 row: small products next to a large running sum
    a = rng.standard_normal(sh_a) * 4
    idx = rng.integers(0, 32, (tiles, steps, 16, 2))
    np.put_along_axis(a, idx, np.take_along_axis(a, idx, -1) * 256, -1)
    yield "chain_spikes", E.from_f32(a), E.from_f32(np.abs(rng.standard_normal(sh_b)) * 4), zero


def mixed_families(rng, tiles: int):
    """(name, kinds (tiles, steps) 0 = e4m3 k32 / 1 = bf16 k16, A8, B8, A16, B16, C0): the FP8 atom -> BF16 peel chains."""
    E, B = TP.E4M3, TP.BF16
    acc_rand = lambda: TP.random_finite_fp32_bits(rng, tiles * 128, 110, 160).reshape(tiles, 16, 8)
    acc_zero = lambda: np.where(rng.random((tiles, 16, 8)) < 0.5, 0, SIGN).astype(np.uint32)
    for name, pattern, acc in (("mixed_qb", "qb", acc_rand), ("mixed_qb_zero_acc", "qb", acc_zero), ("mixed_bq", "bq", acc_rand),
                               ("mixed_g4", "qqqqb" * 8, acc_rand), ("mixed_random", None, acc_rand)):
        steps = 64 if pattern is None else len(pattern)
        kinds = (rng.random((tiles, steps)) < 0.5).astype(np.uint8) if pattern is None else \
            np.broadcast_to(np.array([c == "b" for c in pattern], np.uint8), (tiles, steps)).copy()
        A8 = E.from_f32(rng.standard_normal((tiles, steps, 16, 32)) * 8)
        B8 = E.from_f32(rng.standard_normal((tiles, steps, 8, 32)) * 8)
        A16 = B.from_f32(rng.standard_normal((tiles, steps, 16, 16)) * np.exp2(rng.integers(-12, 4, (tiles, steps, 16, 1))))
        B16 = B.from_f32(rng.standard_normal((tiles, steps, 8, 16)))
        yield name, kinds, A8, B8, A16, B16, acc()


def cvt_inputs(rng) -> dict[str, np.ndarray]:
    """Every exponent field x sign x the top 11 mantissa bits, each with low 12 bits in {0, 1, 0x7ff, 0x800, 0x801, 0xfff} (so every
    guard / round / sticky pattern of both formats), plus uniformly random words."""
    top = np.arange(1 << 11, dtype=np.uint32) << 12
    low = np.array([0, 1, 0x7FF, 0x800, 0x801, 0xFFF], np.uint32)
    exps = np.arange(256, dtype=np.uint32) << 23
    grid = (exps[:, None, None] | top[None, :, None] | low[None, None, :]).reshape(-1)
    grid = np.concatenate([grid, grid | SIGN])
    return {"grid": grid, "random": rng.integers(0, 1 << 32, 1 << 21, dtype=np.uint64).astype(np.uint32)}


def fadd_inputs(rng) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n = 1 << 20
    rw = lambda k, lo=0, hi=254: TP.random_finite_fp32_bits(rng, k, lo, hi)
    out = {"random": (rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32), rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32))}
    x = rw(n)
    d = rng.integers(0, 32, n).astype(np.uint32)
    y = (x & SIGN) ^ (rng.integers(0, 2, n).astype(np.uint32) << 31)
    y |= np.clip((x >> 23 & 0xFF).astype(np.int64) - d, 0, 254).astype(np.uint32) << 23 | rng.integers(0, 1 << 23, n).astype(np.uint32)
    out["near"] = (x, y)
    k = 1 << 18
    sub = lambda: (rng.integers(0, 2, k).astype(np.uint32) << 31) | rng.integers(1, 1 << 23, k).astype(np.uint32)
    small = rw(k, 1, 8)
    out["subnormal"] = (np.concatenate([sub(), sub(), small]), np.concatenate([sub(), rw(k), small ^ SIGN ^ rng.integers(0, 4, k).astype(np.uint32)]))
    x = rw(k, 1, 253)
    e = (x >> 23 & 0xFF).astype(np.int64)
    half = np.clip(e - 24, 0, 254).astype(np.uint32) << 23  # half an ulp of x (a power of two), then +- the smallest perturbation
    out["ties"] = (np.concatenate([x, x, x]), np.concatenate([half, half | 1, (half | 1) ^ SIGN]))
    big = rw(k, 250, 254)
    out["overflow"] = (big, big ^ (rng.integers(0, 2, k).astype(np.uint32) << 31) | rng.integers(0, 2, k).astype(np.uint32))
    sp = np.array([0, SIGN, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 1, SIGN | 1, 0x7F7FFFFF, 0xFF7FFFFF,
                   0x00800000, 0x80800000, 0x3F800000], np.uint32)
    out["specials"] = (np.repeat(sp, len(sp)), np.tile(sp, len(sp)))
    return out


# --- gates --------------------------------------------------------------------------------------------------------------------------


def _layout_values(rng, shape, fmt: str) -> tuple[np.ndarray, np.ndarray]:
    """Codes of values in {0, +-0.5, +-1, +-2, +-4} (exact in every format and every candidate) and their float values."""
    vals = rng.choice(np.array([0, 0.5, -0.5, 1, -1, 2, -2, 4, -4]), size=shape)
    return TPF[fmt].from_f32(vals), vals


def layout_gate(dev, row: str, instr: str | None, rng) -> dict:
    """Exact small integers through every kernel the row uses: D must equal the exact matmul (any candidate agrees), so a wrong
    fragment layout fails here and nothing is measured."""
    checks = {}
    n = 64
    if row == "step":
        _, fa, fb, _, _ = TILES[instr]
        A, va = _layout_values(rng, (n, 16, 32), fa)
        Bt, vb = _layout_values(rng, (n, 8, 32), fb)
        sfa = rng.integers(124, 131, (n, 16)).astype(np.uint8) if instr == "mxf8_e4m3" else np.full((n, 16), 127, np.uint8)
        sfb = rng.integers(124, 131, (n, 8)).astype(np.uint8) if instr == "mxf8_e4m3" else np.full((n, 8), 127, np.uint8)
        want = np.einsum("tik,tjk->tij", va, vb) * np.exp2(sfa.astype(float) - 127)[:, :, None] * np.exp2(sfb.astype(float) - 127)[:, None, :]
        got = dev.tiles(instr, A, Bt, np.zeros((n, 16, 8), np.uint32), sfa, sfb).view(np.float32)
        checks[instr] = int(np.count_nonzero(got != want.astype(np.float32)))
    elif row == "chain":
        A, va = _layout_values(rng, (n, 4, 16, 32), "e4m3")
        Bt, vb = _layout_values(rng, (n, 4, 8, 32), "e4m3")
        want = np.cumsum(np.einsum("tsik,tsjk->tsij", va, vb), axis=1)
        got = dev.e4m3_chain(A, Bt, np.zeros((n, 16, 8), np.uint32)).view(np.float32)
        checks["e4m3_chain"] = int(np.count_nonzero(got != want.astype(np.float32)))
    elif row == "mixed":
        steps = 4
        kinds = np.tile(np.array([0, 1, 1, 0], np.uint8), (n, 1))
        A8, va8 = _layout_values(rng, (n, steps, 16, 32), "e4m3")
        B8, vb8 = _layout_values(rng, (n, steps, 8, 32), "e4m3")
        A16, va16 = _layout_values(rng, (n, steps, 16, 16), "bf16")
        B16, vb16 = _layout_values(rng, (n, steps, 8, 16), "bf16")
        q = np.einsum("tsik,tsjk->tsij", va8, vb8)
        b = np.einsum("tsik,tsjk->tsij", va16, vb16)
        want = np.cumsum(np.where((kinds == 0)[:, :, None, None], q, b), axis=1)
        got = dev.mixed_chain(kinds, A8, B8, A16, B16, np.zeros((n, 16, 8), np.uint32)).view(np.float32)
        checks["mixed_chain"] = int(np.count_nonzero(got != want.astype(np.float32)))
    elif row == "cvt":
        x = np.array([1.0, -2.0, 0.5, 448.0, -448.0, 0.0, 3.0, 0.015625], np.float32).view(np.uint32)
        for fmt in ("e4m3", "e5m2"):
            checks[f"cvt_{fmt}"] = int(np.count_nonzero(dev.cvt(fmt, x) != TPF[fmt].from_f32(x.view(np.float32))))
    elif row == "fadd":
        x = np.array([1.0, 2.0, -3.0, 0.5], np.float32).view(np.uint32)
        y = np.array([2.0, -2.0, 1.0, 0.25], np.float32).view(np.uint32)
        want = (x.view(np.float32) + y.view(np.float32)).view(np.uint32)
        for ftz in (False, True):
            checks[f"fadd_ftz{int(ftz)}"] = int(np.count_nonzero(dev.fadd(ftz, x, y) != want))
    return {"ok": all(v == 0 for v in checks.values()), "mismatches": checks}


BOOTSTRAP = {"name": "RTX PRO 6000", "sms": 188, "min_driver": 575}  # server.md's bootstrap checks for the card
SMI_FIELDS = ("name,driver_version,vbios_version,pci.bus_id,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,"
              "clocks_event_reasons.active")


def smi_query(uuid: str, fields: str, nounits: bool = False) -> dict | None:
    """``nvidia-smi --query-gpu`` for one device; when the whole query fails (a field this driver doesn't know), field by field."""
    fmt = "--format=csv,noheader" + (",nounits" if nounits else "")

    def q(f: str) -> list[str] | None:
        p = subprocess.run(["nvidia-smi", f"--id={uuid}", f"--query-gpu={f}", fmt], capture_output=True, text=True)
        return [s.strip() for s in p.stdout.split(",")] if p.returncode == 0 else None

    names = fields.split(",")
    vals = q(fields)
    if vals is not None and len(vals) == len(names):
        return dict(zip(names, vals))
    out = {n: (v or [None])[0] for n in names for v in [q(n)]}
    return out if any(v is not None for v in out.values()) else None


def identity_gate(dev, expect_uuid: str | None, bootstrap: dict | None = None) -> tuple[dict, list[str]]:
    boot = BOOTSTRAP if bootstrap is None else bootstrap
    info = dev.info()
    failures = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and ("," in visible or not visible.strip()):
        failures.append(f"CUDA_VISIBLE_DEVICES={visible!r} is not a single device")
    if expect_uuid and info["gpu_uuid"].lower() != expect_uuid.lower():
        failures.append(f"device UUID {info['gpu_uuid']} != --expect-gpu-uuid {expect_uuid}")
    if info["gpu_uuid"] != "GPU-fake":
        info["nvidia_smi"] = smi_query(info["gpu_uuid"], SMI_FIELDS)
        info["driver"] = (info["nvidia_smi"] or {}).get("driver_version")
        if info["compute_capability"] != "12.0":
            failures.append(f"compute capability {info['compute_capability']} is not sm_120")
        if boot.get("name") and boot["name"] not in info["device"]:
            failures.append(f"device {info['device']!r} is not an {boot['name']}")
        if boot.get("sms") and info["sms"] != boot["sms"]:
            failures.append(f"{info['sms']} SMs, not {boot['sms']}")
        major = int(str(info["driver"] or "0").split(".")[0] or 0)
        if boot.get("min_driver") and major < boot["min_driver"]:
            failures.append(f"driver {info['driver']} is older than {boot['min_driver']}")
    info.update({"hostname": platform.node(), "cuda_visible_devices": visible, "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
                 "expect_gpu_uuid": expect_uuid, "uuid_checked": bool(expect_uuid), "bootstrap": boot})
    return info, failures


# --- rows ---------------------------------------------------------------------------------------------------------------------------


def _hexrow(x, width) -> str:
    return "".join(f"{int(v):0{width}x}" for v in np.asarray(x).reshape(-1))


def _finite_domain(A, Bt, C, fa, fb, primary_out) -> np.ndarray:
    """Per element: accumulator and every operand finite, and the primary model's word finite (else record-only)."""
    a, b, acc = expand(A, Bt, C)
    ok = ((acc & 0x7F800000) != 0x7F800000) & TPF[fa].is_finite(a).all(-1) & TPF[fb].is_finite(b).all(-1)
    return ok & ((primary_out.reshape(-1) & 0x7F800000) != 0x7F800000)


def row_step(dev, args, out: Path) -> dict:
    instr = args.instruction
    _, fa, fb, _, _ = TILES[instr]
    rng = np.random.default_rng(args.seed)
    cands = candidates(instr)
    primary = PRIMARY[instr][0]
    fams, samples, first = {}, {}, []
    arrays = {}
    for name, A, Bt, C, SFA, SFB in step_families(rng, instr, args.n_random):
        if args.families and name not in args.families:
            continue
        n = A.shape[0]
        SFA = np.full((n, 16), 127, np.uint8) if SFA is None else SFA
        SFB = np.full((n, 8), 127, np.uint8) if SFB is None else SFB
        D = dev.tiles(instr, A, Bt, C, SFA, SFB)
        preds = {c: evaluate(instr, sp, A, Bt, C, SFA, SFB) for c, sp in cands.items()}
        if instr in ("f8_e4m3", "legacy_e4m3", "f8_e5m2"):  # the core kernel on the registered model must equal this row's arithmetic
            core = TCM.BLACKWELL_SM120_E4M3_M16N8K32 if fa == "e4m3" else TCM.BLACKWELL_SM120_E5M2_M16N8K32
            if not np.array_equal(step_model(core, A, Bt, C), preds[primary]):
                raise RuntimeError(f"fp8.groupsum disagrees with kernels.group_sum_total_batch on family {name}")
        gated = _finite_domain(A, Bt, C, fa, fb, preds[primary])
        d = D.reshape(-1)
        rec = {"elements": int(d.size), "gated_elements": int(gated.sum()), "mismatches": {}, "record_only_mismatches": {}}
        for c, p in preds.items():
            bad = d != p.reshape(-1)
            rec["mismatches"][c] = int((bad & gated).sum())
            rec["record_only_mismatches"][c] = int((bad & ~gated).sum())
            if c == primary:
                for r in np.flatnonzero(bad & gated)[:5]:
                    t, i, j = r // 128, (r // 8) % 16, r % 8
                    first.append({"family": name, "tile": int(t), "i": int(i), "j": int(j), "c": f"{int(C[t, i, j]):08x}",
                                  "a": _hexrow(A[t, i], 2), "b": _hexrow(Bt[t, j], 2), "sfa": int(SFA[t, i]), "sfb": int(SFB[t, j]),
                                  "hardware": f"{int(d[r]):08x}", "expected": f"{int(p.reshape(-1)[r]):08x}"})
        pick = rng.choice(d.size, size=min(64, d.size), replace=False)
        samples[name] = [{"c": f"{int(C.reshape(-1)[r]):08x}", "a": _hexrow(A[r // 128, (r // 8) % 16], 2), "b": _hexrow(Bt[r // 128, r % 8], 2),
                          "sfa": int(SFA[r // 128, (r // 8) % 16]), "sfb": int(SFB[r // 128, r % 8]), "d": f"{int(d[r]):08x}"} for r in pick]
        fams[name] = rec
        arrays[name] = dict(A=A, Bt=Bt, C=C, D=D, SFA=SFA, SFB=SFB)
    np.savez_compressed(out / f"tiles_{instr}.npz", **{f"{f}.{k}": v for f, d in arrays.items() for k, v in d.items()})
    totals = {c: sum(f["mismatches"][c] for f in fams.values()) for c in cands}
    return {"families": fams, "candidates": {c: {**sp, "groups": list(sp["groups"])} for c, sp in cands.items()}, "primary": primary,
            "total_mismatches": totals, "first_mismatches": first, "samples": samples,
            "elements": sum(f["elements"] for f in fams.values()), "gated_elements": sum(f["gated_elements"] for f in fams.values()),
            "agrees": {c: v == 0 for c, v in totals.items()}}


def _chain_facts(A, Bt, C0, D) -> dict:
    """Freezing facts from the device words alone (E4M3 products are exact in float64, and so is their k32 sum): per doubling
    window (n/2, n] of steps, the fraction of exact steps (D_s == D_{s-1} + sum p exactly), of frozen steps (sum p != 0 and
    D_s == D_{s-1}), and at n the error |D_n - exact_n| relative to |C0| + sum |p| (median, max)."""
    tiles, steps = A.shape[:2]
    P = np.zeros((tiles, steps, 16, 8))
    Pabs = np.zeros((tiles, steps, 16, 8))
    for s0 in range(0, steps, 64):
        prod = TP.E4M3.decode(A[:, s0:s0 + 64])[:, :, :, None, :] * TP.E4M3.decode(Bt[:, s0:s0 + 64])[:, :, None, :, :]
        P[:, s0:s0 + 64], Pabs[:, s0:s0 + 64] = prod.sum(-1), np.abs(prod).sum(-1)
    d = D.view(np.float32).astype(np.float64)
    prev = np.concatenate([C0.view(np.float32).astype(np.float64).reshape(tiles, 1, 16, 8), d[:, :-1]], axis=1)
    s = prev + P
    bb = s - prev
    err = (prev - (s - bb)) + (P - bb)  # TwoSum: the exact sum is s + err
    exact = (err == 0) & (s.astype(np.float32).astype(np.float64) == s) & (d == s)
    frozen = (P != 0) & (d == prev)
    c0 = C0.view(np.float32).astype(np.float64)
    total, comp, mass = c0.copy(), np.zeros_like(c0), np.abs(c0)
    facts, n, lo = [], 1, 0
    for step in range(steps):
        y = P[:, step] - comp
        t = total + y
        comp = (t - total) - y
        total = t
        mass = mass + Pabs[:, step]
        if step + 1 == n:
            rel = np.abs(d[:, step] - total) / np.where(mass > 0, mass, 1.0)
            facts.append({"steps": n, "k": 32 * n, "exact_steps": float(exact[:, lo:n].mean()), "frozen_steps": float(frozen[:, lo:n].mean()),
                          "rel_err_median": float(np.median(rel)), "rel_err_max": float(rel.max()),
                          "unchanged_since_half": float((d[:, step] == d[:, max(n // 2 - 1, 0)]).mean()) if n > 1 else None})
            lo, n = n, n * 2
    return {"per_doubling": facts, "exact_steps_all": float(exact.mean()), "frozen_steps_all": float(frozen.mean())}


def row_chain(dev, args, out: Path) -> dict:
    rng = np.random.default_rng(args.seed)
    model, controls = TCM.BLACKWELL_SM120_E4M3_M16N8K32, {"ada_e4m3_m16n8k32": TCM.ADA_E4M3_M16N8K32, "hopper_e4m3_k32": TCM.HOPPER_E4M3_K32}
    fams, first, arrays = {}, [], {}
    for name, A, Bt, C0 in chain_families(rng, args.chain_tiles, args.steps):
        if args.families and name not in args.families:
            continue
        D = dev.e4m3_chain(A, Bt, C0)
        tiles, steps = A.shape[:2]
        prev = np.concatenate([C0.reshape(tiles, 1, 16, 8), D[:, :-1]], axis=1)
        rec = {"tiles": tiles, "steps": steps, "k": 32 * steps, "elements": int(D.size), "step_mismatches": {}}
        for mname, m in {model.name: model, **controls}.items():
            bad = 0
            for s0 in range(0, steps, 64):
                s1 = min(steps, s0 + 64)
                pred = step_model(m, A[:, s0:s1].reshape(-1, 16, 32), Bt[:, s0:s1].reshape(-1, 8, 32), prev[:, s0:s1].reshape(-1, 16, 8))
                miss = pred.reshape(tiles, s1 - s0, 16, 8) != D[:, s0:s1]
                bad += int(miss.sum())
                if m is model:
                    for t, s, i, j in np.argwhere(miss)[: max(0, 5 - sum(1 for f in first if f["family"] == name))]:
                        first.append({"family": name, "tile": int(t), "step": int(s0 + s), "i": int(i), "j": int(j),
                                      "c": f"{int(prev[t, s0 + s, i, j]):08x}", "hardware": f"{int(D[t, s0 + s, i, j]):08x}",
                                      "expected": f"{int(pred.reshape(tiles, s1 - s0, 16, 8)[t, s, i, j]):08x}"})
            rec["step_mismatches"][mname] = bad
        acc = C0.reshape(tiles, 16, 8)
        diverged = None
        for s in range(steps):  # the free-running fold of the model from C0
            acc = step_model(model, A[:, s], Bt[:, s], acc)
            if diverged is None and not np.array_equal(acc, D[:, s]):
                diverged = s
        rec["free_run_first_divergent_step"] = diverged
        rec["facts"] = _chain_facts(A, Bt, C0, D)
        fams[name] = rec
        arrays[name] = dict(C0=C0, D_final=D[:, -1], D_pow2=D[:, [2**j - 1 for j in range(steps.bit_length()) if 2**j <= steps]])
    np.savez_compressed(out / "chains_e4m3.npz", **{f"{f}.{k}": v for f, d in arrays.items() for k, v in d.items()})
    totals = {m: sum(f["step_mismatches"][m] for f in fams.values()) for m in (model.name, *controls)}
    free_ok = all(f["free_run_first_divergent_step"] is None for f in fams.values())
    return {"families": fams, "primary": model.name, "total_mismatches": totals, "first_mismatches": first,
            "elements": sum(f["elements"] for f in fams.values()), "gated_elements": sum(f["elements"] for f in fams.values()),
            "agrees": {m: v == 0 for m, v in totals.items()} | {"free_run": free_ok}}


def row_mixed(dev, args, out: Path) -> dict:
    rng = np.random.default_rng(args.seed)
    q_models = {"blackwell_sm120_e4m3_m16n8k32": TCM.BLACKWELL_SM120_E4M3_M16N8K32, "ada_e4m3_m16n8k32": TCM.ADA_E4M3_M16N8K32}
    b_models = {"hopper_bf16_m16n8k16": TCM.HOPPER_BF16_M16N8K16, "ampere_bf16_m16n8k16": TCM.AMPERE_BF16_M16N8K16}
    pairs = {"sm120_e4m3+hopper_bf16": ("blackwell_sm120_e4m3_m16n8k32", "hopper_bf16_m16n8k16"),
             "sm120_e4m3+ampere_bf16": ("blackwell_sm120_e4m3_m16n8k32", "ampere_bf16_m16n8k16"),
             "ada_e4m3+hopper_bf16": ("ada_e4m3_m16n8k32", "hopper_bf16_m16n8k16")}
    primary = "sm120_e4m3+hopper_bf16"
    fams, first, arrays = {}, [], {}
    for name, kinds, A8, B8, A16, B16, C0 in mixed_families(rng, args.chain_tiles):
        if args.families and name not in args.families:
            continue
        D = dev.mixed_chain(kinds, A8, B8, A16, B16, C0)
        tiles, steps = kinds.shape
        prev = np.concatenate([C0.reshape(tiles, 1, 16, 8), D[:, :-1]], axis=1)
        flat = lambda x, *s: x.reshape(-1, *s)
        qp = {m: step_model(v, flat(A8, 16, 32), flat(B8, 8, 32), flat(prev, 16, 8)).reshape(tiles, steps, 16, 8) for m, v in q_models.items()}
        bp = {m: step_model(v, flat(A16, 16, 16), flat(B16, 8, 16), flat(prev, 16, 8)).reshape(tiles, steps, 16, 8) for m, v in b_models.items()}
        isq = (kinds == 0)[:, :, None, None]
        rec = {"tiles": tiles, "steps": steps, "fp8_steps": int((kinds == 0).sum()), "bf16_steps": int((kinds == 1).sum()),
               "elements": int(D.size), "step_mismatches": {}}
        for pname, (qm, bm) in pairs.items():
            pred = np.where(isq, qp[qm], bp[bm])
            miss = pred != D
            rec["step_mismatches"][pname] = int(miss.sum())
            if pname == primary:
                for t, s, i, j in np.argwhere(miss)[:5]:
                    first.append({"family": name, "tile": int(t), "step": int(s), "kind": "bf16" if kinds[t, s] else "e4m3",
                                  "c": f"{int(prev[t, s, i, j]):08x}", "hardware": f"{int(D[t, s, i, j]):08x}", "expected": f"{int(pred[t, s, i, j]):08x}"})
        acc = C0.reshape(tiles, 16, 8)
        diverged = None
        for s in range(steps):
            q = step_model(q_models["blackwell_sm120_e4m3_m16n8k32"], A8[:, s], B8[:, s], acc)
            b = step_model(b_models["hopper_bf16_m16n8k16"], A16[:, s], B16[:, s], acc)
            acc = np.where((kinds[:, s] == 0)[:, None, None], q, b)
            if diverged is None and not np.array_equal(acc, D[:, s]):
                diverged = s
        rec["free_run_first_divergent_step"] = diverged
        fams[name] = rec
        arrays[name] = dict(kinds=kinds, C0=C0, D=D)
    np.savez_compressed(out / "chains_mixed.npz", **{f"{f}.{k}": v for f, d in arrays.items() for k, v in d.items()})
    totals = {p: sum(f["step_mismatches"][p] for f in fams.values()) for p in pairs}
    free_ok = all(f["free_run_first_divergent_step"] is None for f in fams.values())
    return {"families": fams, "primary": primary, "total_mismatches": totals, "first_mismatches": first,
            "elements": sum(f["elements"] for f in fams.values()), "gated_elements": sum(f["elements"] for f in fams.values()),
            "agrees": {p: v == 0 for p, v in totals.items()} | {"free_run": free_ok}}


def row_cvt(dev, args, out: Path) -> dict:
    rng = np.random.default_rng(args.seed)
    inputs = cvt_inputs(rng)
    probe_x = np.concatenate([inputs["random"][:20000], inputs["grid"][:: max(1, inputs["grid"].size // 20000)]])
    core = np.array([f32_to_e4m3_sat_word(int(u)) for u in probe_x], np.uint8)
    if not np.array_equal(core, cvt_ref("e4m3", probe_x)):
        raise RuntimeError("fp8.cvt_ref disagrees with verity.ml.tc.cast.f32_to_e4m3_sat_word")
    fams, first, arrays = {}, [], {}
    for fmt in ("e4m3", "e5m2"):
        for name, X in inputs.items():
            X = X[: X.size // 2 * 2]
            Y = dev.cvt(fmt, X)
            ref = cvt_ref(fmt, X)
            nan_in = (X & 0x7FFFFFFF) > 0x7F800000
            bad = Y != ref
            key = f"{fmt}.{name}"
            fams[key] = {"elements": int(X.size), "gated_elements": int((~nan_in).sum()), "mismatches": {"cvt_ref": int((bad & ~nan_in).sum())},
                         "record_only_mismatches": {"cvt_ref": int((bad & nan_in).sum())},
                         "nan_outputs": sorted({f"{int(v):02x}" for v in np.unique(Y[nan_in])})}
            for r in np.flatnonzero(bad & ~nan_in)[:5]:
                first.append({"family": key, "x": f"{int(X[r]):08x}", "hardware": f"{int(Y[r]):02x}", "expected": f"{int(ref[r]):02x}"})
            arrays[key] = Y
    np.savez_compressed(out / "cvt.npz", **arrays)
    total = sum(f["mismatches"]["cvt_ref"] for f in fams.values())
    return {"families": fams, "primary": "cvt_ref", "total_mismatches": {"cvt_ref": total}, "first_mismatches": first,
            "elements": sum(f["elements"] for f in fams.values()), "gated_elements": sum(f["gated_elements"] for f in fams.values()),
            "agrees": {"cvt_ref": total == 0}, "core_crosscheck_inputs": int(probe_x.size)}


def row_fadd(dev, args, out: Path) -> dict:
    host_self_check()
    rng = np.random.default_rng(args.seed)
    fams, first, arrays = {}, [], {}
    for name, (X, Y) in fadd_inputs(rng).items():
        for ftz in (False, True):
            Z = dev.fadd(ftz, X, Y)
            refs = {"rn": fadd_ref(X, Y, False), "rn_ftz": fadd_ref(X, Y, True)}
            want = "rn_ftz" if ftz else "rn"
            nan_out = refs[want] == NAN_WORD
            key = f"{want}.{name}"
            fams[key] = {"elements": int(X.size), "gated_elements": int((~nan_out).sum()),
                         "mismatches": {r: int(((Z != v) & ~nan_out).sum()) for r, v in refs.items()},
                         "record_only_mismatches": {r: int(((Z != v) & nan_out).sum()) for r, v in refs.items()},
                         "nan_outputs": sorted({f"{int(v):08x}" for v in np.unique(Z[nan_out])})}
            for r in np.flatnonzero((Z != refs[want]) & ~nan_out)[:5]:
                first.append({"family": key, "x": f"{int(X[r]):08x}", "y": f"{int(Y[r]):08x}", "hardware": f"{int(Z[r]):08x}",
                              "expected": f"{int(refs[want][r]):08x}"})
            arrays[key] = Z
    np.savez_compressed(out / "fadd.npz", **arrays)
    totals = {k: sum(f["mismatches"][k.split(".")[0]] for n, f in fams.items() if n.startswith(k + ".")) for k in ("rn", "rn_ftz")}
    return {"families": fams, "primary": "rn and rn_ftz on their own kernels", "total_mismatches": totals, "first_mismatches": first,
            "elements": sum(f["elements"] for f in fams.values()), "gated_elements": sum(f["gated_elements"] for f in fams.values()),
            "agrees": {k: v == 0 for k, v in totals.items()}}


ROWS = {"step": row_step, "chain": row_chain, "mixed": row_mixed, "cvt": row_cvt, "fadd": row_fadd}


# --- main ---------------------------------------------------------------------------------------------------------------------------


def run(args) -> tuple[int, dict]:
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results: dict = {"schema": "tc-probe-fp4/fp8/v1", "row": args.row, "instruction": args.instruction if args.row == "step" else None,
                     "seed": args.seed, "fake_silicon": bool(args.fake_silicon)}
    failures: list[str] = []
    so, nvcc = None, None
    if not args.fake_silicon or P4.nvcc_bin() and Path(P4.nvcc_bin()).exists():
        so, nvcc = P4.build(ARCH, out, SOURCE)
        results["nvcc"] = nvcc
        results["sass"] = sass_gate(so, out)
        if not results["sass"]["ok"]:
            failures.append("SASS gate: " + ", ".join(k for k, r in results["sass"]["kernels"].items() if not r["ok"]))
    else:
        results["sass"] = {"ok": None, "detail": "skipped: fake silicon without nvcc"}
    if args.row == "compile" or failures:
        results["seconds"] = round(time.time() - t0, 3)
        return finish(out, args, results, failures, identity=None)
    dev = FakeDevice(args.corrupt_word) if args.fake_silicon else Device(so)
    identity, id_fail = identity_gate(dev, args.expect_gpu_uuid)
    results["identity"] = identity
    failures += id_fail
    if args.row == "identity":
        results["seconds"] = round(time.time() - t0, 3)
        return finish(out, args, results, failures, identity)
    if not failures:
        results["layout"] = layout_gate(dev, args.row, args.instruction, np.random.default_rng(1))
        if not results["layout"]["ok"]:
            failures.append(f"layout check: {results['layout']['mismatches']}")
    if failures:
        results["seconds"] = round(time.time() - t0, 3)
        return finish(out, args, results, failures, identity)
    if isinstance(dev, FakeDevice):
        dev.armed = True
    res = ROWS[args.row](dev, args, out)
    results["capture"] = {k: v for k, v in res.items() if k != "samples"}
    (out / "capture_sample.json").write_text(json.dumps(res.get("samples", {}), indent=1))
    primary_ok = res["agrees"].get(res["primary"], all(res["agrees"].values())) if args.row != "fadd" else all(res["agrees"].values())
    if args.row in ("chain", "mixed"):
        primary_ok = primary_ok and res["agrees"]["free_run"]
    if not primary_ok:
        failures.append(f"primary {res['primary']}: {res['total_mismatches']} mismatches")
    results["seconds"] = round(time.time() - t0, 3)
    return finish(out, args, results, failures, identity)


def finish(out: Path, args, results: dict, failures: list[str], identity: dict | None) -> tuple[int, dict]:
    results["failures"] = failures
    results["validation"] = "passed" if not failures else "failed"
    (out / "probe_results.json").write_text(json.dumps(results, indent=1, default=str))
    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id and not args.fake_silicon and args.row != "identity":
        cap = results.get("capture", {})
        instr = TILES[args.instruction][4] if args.row == "step" else ROW_ID.get(args.row, "sm120.fp8.compile")
        doc = {"schema": "research/result/v0.1", "kind": "tc-compile-check/v1" if args.row == "compile" else "tc-evidence/v1", "run_id": run_id,
               "instruction": instr, "row": args.row, "ptx": TILES[args.instruction][3] if args.row == "step" else None, "arch": ARCH,
               "sku": (identity or {}).get("device"), "driver": (identity or {}).get("driver"), "gpu_uuid": (identity or {}).get("gpu_uuid"),
               "uuid_checked": (identity or {}).get("uuid_checked"), "nvcc": results.get("nvcc"),
               "sass_opcodes": {k: r["opcodes"] for k, r in results.get("sass", {}).get("kernels", {}).items()},
               "sass_sha256": results.get("sass", {}).get("sass_sha256"),
               "families": {k: {kk: vv for kk, vv in f.items() if kk in ("elements", "gated_elements", "mismatches", "step_mismatches", "k")}
                            for k, f in cap.get("families", {}).items()},
               "counts": {"elements": cap.get("elements", 0), "gated_elements": cap.get("gated_elements", 0), "mismatches": cap.get("total_mismatches", {})},
               "model": cap.get("primary"), "model_agrees": (not failures) if cap else None, "agreement": cap.get("agrees", {}),
               "first_mismatches": cap.get("first_mismatches", [])[:20], "seed": args.seed, "n": args.n_random,
               "steps": args.steps if args.row == "chain" else None,
               "workload_fingerprint": {"probe": "tools/tc_probe_fp4/fp8.py", "row": args.row, "instruction": instr, "seed": args.seed,
                                        "n_random": args.n_random, "steps": args.steps, "chain_tiles": args.chain_tiles},
               "validation": {"status": "passed" if not failures else "failed", "checker": "tc_probe_fp4/fp8.py: identity, SASS, layout gates; "
                              "hardware words vs verity.ml.tc models and the scalar references", "detail": "; ".join(failures) or "every gated word reproduced",
                              "evidence": ["probe_results.json"]},
               "measurements": [{"name": "wall", "value": results.get("seconds"), "unit": "s"}],
               "artifacts": sorted(p.name for p in out.iterdir() if p.suffix in (".json", ".npz", ".txt") and p.name != "result.json")}
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1, default=str))
        os.replace(tmp, out / "result.json")
    return (0 if not failures else 1), results


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--row", required=True, choices=["compile", "identity", *ROWS])
    ap.add_argument("--instruction", default="f8_e4m3", choices=sorted(TILES), help="the step row's tile instruction")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--n-random", type=int, default=4096, help="step row: tiles of the largest family (128 elements each)")
    ap.add_argument("--steps", type=int, default=2048, help="chain row: dependent k32 steps per chain (2048 = K 2^16)")
    ap.add_argument("--chain-tiles", type=int, default=16, help="chain and mixed rows: tiles (chains of 128 elements) per family")
    ap.add_argument("--families", default=None, type=lambda s: set(filter(None, s.split(","))), help="comma list (default all)")
    ap.add_argument("--expect-gpu-uuid", default=os.environ.get("GPU_LEASE_UUID"),
                    help="the device's UUID (default: gpu-lease's GPU_LEASE_UUID); a mismatch stops the run")
    ap.add_argument("--fake-silicon", action="store_true", help="the primary model stands in for the device (CPU dry run; never evidence)")
    ap.add_argument("--corrupt-word", type=int, default=None, help="with --fake-silicon: flip one bit of this output word once; the run must fail")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.fake_silicon and os.environ.get("RESEARCH_RUN_ID"):
        print("fp8.py: --fake-silicon is a dry run and never evidence; refused under research run", file=sys.stderr)
        return 2
    if args.corrupt_word is not None and not args.fake_silicon:
        print("fp8.py: --corrupt-word needs --fake-silicon", file=sys.stderr)
        return 2
    code, results = run(args)
    print(json.dumps({"row": args.row, "validation": results["validation"], "failures": results["failures"],
                      "mismatches": results.get("capture", {}).get("total_mismatches"), "seconds": results.get("seconds")}))
    return code


if __name__ == "__main__":
    sys.exit(main())
