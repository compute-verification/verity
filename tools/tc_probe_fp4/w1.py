#!/usr/bin/env python3
"""W1 price microbenchmarks on sm_120 (the RTX PRO 6000): issue rates per SM per clock of the dense FP8 atom (f32 and f16
accumulation), the 2:4-sparse E4M3 atom (mma.sp m16n8k64), the block-scaled mxf8f6f4 and nvf4 atoms, BF16 and FP16 mma.sync, FP32
FADD and FFMA, the packed add.rn.f32x2 and the same adds as plain C pairs, the E4M3 cast and the E2M1 cast (three loops, whose split
gives the cast's own cost), the E4M3 cast-and-store of Pearl-C's sm_120 forming kernel (one code per convert, two per 16-bit store)
beside its packed replacement (four codes per 32-bit store) and the packed words through 64- and 128-bit stores (at conflict-free row
strides and at the port's), each fed by one FADD per code or by no per-code input, and the prices W1 reads off them.

    python tools/tc_probe_fp4/w1.py --out $RESEARCH_RUN_DIR --expect-gpu-uuid GPU-... [--reps 10] [--target-cycles 134217728]
                                    [--ops qmma_e4m3_f32,qmma_sp_e4m3] [--prebuilt libw1_prices_sm_120a.so] [--lease-max-min 10]
                                    [--smi-every 5]

``--lease-max-min M`` builds and SASS-gates here, then runs the rest under ``gpu-lease 1 --wait --max-min M`` on the built library.

The unit is W1's: one dense E4M3 MAC with FP32 accumulation, i.e. 1 / (the qmma_e4m3_f32 rate per SM per clock).  A price is that
rate over an op's rate in its own unit: per MAC for the atoms (the sparse atom per logical MAC, 16 x 8 x 64 an instruction, and per
nonzero product, half of them), per register written for FADD / FFMA, per result word (and per
instruction, two words) for the packed add, per code (and per instruction, two codes) for the cast, per code stored for the
cast-and-store loops.  Per-clock rates come from clock64() around each block's loop, so they don't move with the SM clock; the
clock itself is recorded per rep as cycles over event time, and by an ``nvidia-smi --id`` sample before the first rep round and after
every ``--smi-every``-th and the last (not every rep: the GPU idles through each sample), and results carry ``--clocks`` as
their clock label (unlocked unless the operator says otherwise, e.g. locked-2100 on node 2).

Gates, in order, before anything is timed: the device (``fp8.identity_gate``: one visible device, its UUID, sm_120); the SASS gate
(each op's loop body holds exactly its instruction, NACC x UNROLL MMAs or 64 scalar instructions, and for the cast-and-store loops
exactly their FADD feed, support and stores too; its full loop composition is recorded); the numeric gate (all-ones operands give
exact integers for every atom and FADD / FFMA, and for the packed and paired adds 1 in the low word and 2 in the high, so a swapped
half shows; the casts are checked against ``fp8.cvt_ref`` and ``e2m1_ref`` on the timing data, the cast-and-store loops on the words
they leave in shared memory, and an op whose expected words are all zero fails as vacuous); one block per
SM (distinct %smid).  A failed gate writes no measurements.  ``--fake-silicon``
replaces the device with a model that returns the gate's words and cycles from the Public datasheet rates (a CPU dry run of the
plumbing, refused under research run); ``--corrupt-word I`` flips one bit of one gate word, and the run must then fail.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fp8 as F8  # noqa: E402  (identity gate, cast reference)
import probe as P4  # noqa: E402  (build, nvcc_bin)

SOURCE = "w1_prices.cu"
THREADS, NACC, UNROLL, NCHAIN, SUNROLL = 512, 4, 4, 8, 8
WARPS = THREADS // 32
GATE_ITERS = 8

# op index in w1_prices.cu -> name, loop opcode, instructions per loop per warp, work per warp instruction, work unit
OPS = [
    ("qmma_e4m3_f32", "QMMA.16832.F32.E4M3.E4M3", NACC * UNROLL, 16 * 8 * 32, "MAC"),
    ("qmma_e4m3_f16", "QMMA.16832.F16.E4M3.E4M3", NACC * UNROLL, 16 * 8 * 32, "MAC"),
    ("qmma_mxf8_e4m3", "QMMA.SF.16832.F32.E4M3.E4M3.E8", NACC * UNROLL, 16 * 8 * 32, "MAC"),
    ("omma_nvf4", "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X", NACC * UNROLL, 16 * 8 * 64, "MAC"),
    ("hmma_bf16_f32", "HMMA.16816.F32.BF16", NACC * UNROLL, 16 * 8 * 16, "MAC"),
    ("hmma_f16_f16", "HMMA.16816.F16", NACC * UNROLL, 16 * 8 * 16, "MAC"),
    ("fadd", "FADD", NCHAIN * SUNROLL, 32, "register"),
    ("ffma", "FFMA", NCHAIN * SUNROLL, 32, "register"),
    ("f2fp_e4m3x2", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 2 * NCHAIN * 4, 64, "code"),
    ("f2fp_e4m3x2_u16", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 2 * NCHAIN * 4, 64, "code"),
    # sm_120a has no FADD2: each add.rn.f32x2 is two FADDs (sm_100a's ptxas gives one FADD2), and plain C pairs stay FADDs
    ("fadd_f32x2", "FADD", 2 * NCHAIN * SUNROLL, 32, "word"),
    ("fadd_pairs", "FADD", 2 * NCHAIN * SUNROLL, 32, "word"),
    ("f2fp_e2m1x2", "F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C", 2 * NCHAIN * 4, 64, "code"),
    ("f2fp_e2m1x2_u8", "F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C", 2 * NCHAIN * 4, 64, "code"),
    ("f2fp_e2m1x2_x2", "F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C", 4 * NCHAIN * 4, 64, "code"),
    # 2:4 sparse along k: 16 x 8 x 64 logical MACs per instruction, half of them nonzero products
    ("qmma_sp_e4m3", "QMMA.SP.16864.F32.E4M3.E4M3", NACC * UNROLL, 16 * 8 * 64, "logical MAC"),
    # the E4M3 cast-and-store of form_s5 (pearl_c_sm120.cu at 952e3ecc) and its packed replacement, 2,048 codes per warp iteration
    ("cs_port_fadd", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 64, 32, "code"),
    ("cs_port_nogen", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 64, 32, "code"),
    ("cs_packed_fadd", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 32, 64, "code"),
    ("cs_packed_nogen", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 32, 64, "code"),
    # the packed words through wider stores: STS.64 (rows 160 bytes apart) and STS.128 (192), then both at the port's 144
    ("cs_w64_fadd", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 32, 64, "code"),
    ("cs_w64_nogen", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 32, 64, "code"),
    ("cs_w128_fadd", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 32, 64, "code"),
    ("cs_w128_nogen", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 32, 64, "code"),
    ("cs_w64_nogen_r144", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 32, 64, "code"),
    ("cs_w128_nogen_r144", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 32, 64, "code"),
]
UNIT_OP = "qmma_e4m3_f32"
_KERNEL = ["qmma_e4m3_f32_rate", "qmma_e4m3_f16_rate", "qmma_mxf8_rate", "omma_nvf4_rate", "hmma_bf16_f32_rate", "hmma_f16_f16_rate",
           "fadd_rate", "ffma_rate", "f2fp_rate", "f2fp_u16_rate", "fadd_f32x2_rate", "fadd_pairs_rate",
           "f2fp_e2m1_rate", "f2fp_e2m1_u8_rate", "f2fp_e2m1_x2_rate", "qmma_sp_e4m3_rate",
           "cs_port_fadd_rate", "cs_port_nogen_rate", "cs_packed_fadd_rate", "cs_packed_nogen_rate",
           "cs_w64_fadd_rate", "cs_w64_nogen_rate", "cs_w128_fadd_rate", "cs_w128_nogen_rate",
           "cs_w64_nogen_r144_rate", "cs_w128_nogen_r144_rate"]  # w1_prices.cu's KERNELS, in op order
SP_OP = 15
CS_OPS = tuple(range(16, 26))
CS_FADD = (16, 18, 20, 22)  # fed by one FADD per code; the others need no per-code input
NCS = 64  # codes per thread per loop iteration
#: the rest of each cast-and-store loop body, exactly (the watched opcodes and these; FADD.FTZ would be a different opcode)
LOOP_EXTRA = {16: {"FADD": 64, "LOP3.LUT": 32, "IMAD": 32, "STS.U16": 32}, 17: {"LOP3.LUT": 32, "IMAD": 32, "STS.U16": 32},
              18: {"FADD": 64, "STS": 16}, 19: {"STS": 16}, 20: {"FADD": 64, "STS.64": 8}, 21: {"STS.64": 8},
              22: {"FADD": 64, "STS.128": 4}, 23: {"STS.128": 4}, 24: {"STS.64": 8}, 25: {"STS.128": 4}}
#: the metadata nibbles mma.sp::ordered_metadata accepts for 2:4: kept indices i0 < i1 as (i1 << 2) | i0
SP_META_VALID = np.array([0b0100, 0b1000, 0b1001, 0b1100, 0b1101, 0b1110], np.uint32)
E2M1_OPS = (12, 13, 14)
E2M1_NK = {12: 2 * NCHAIN, 13: 2 * NCHAIN, 14: 4 * NCHAIN}  # constants per lane in each E2M1 loop
_WATCHED = re.compile(r"^(?:[A-Z]*MMA|F2FP|F2F|FADD2?|FFMA2?|FMUL2?|UF2FP|UFADD|UFFMA)(?:\.|$)")
# Public (datasheet) dense rates per SM per clock, the fake device's cycles: RTX PRO 6000 Blackwell, 188 SMs.  Estimated, never output.
_FAKE_RATE = {"qmma_e4m3_f32": 1024, "qmma_e4m3_f16": 1024, "qmma_mxf8_e4m3": 1024, "omma_nvf4": 2048, "hmma_bf16_f32": 512,
              "hmma_f16_f16": 512, "fadd": 128, "ffma": 128, "f2fp_e4m3x2": 64, "f2fp_e4m3x2_u16": 64, "fadd_f32x2": 128,
              "fadd_pairs": 128, "f2fp_e2m1x2": 64, "f2fp_e2m1x2_u8": 64, "f2fp_e2m1x2_x2": 64, "qmma_sp_e4m3": 2048,
              "cs_port_fadd": 32, "cs_port_nogen": 43, "cs_packed_fadd": 72, "cs_packed_nogen": 128, "cs_w64_fadd": 100,
              "cs_w64_nogen": 128, "cs_w128_fadd": 100, "cs_w128_nogen": 128, "cs_w64_nogen_r144": 128, "cs_w128_nogen_r144": 128}


# --- SASS gate ----------------------------------------------------------------------------------------------------------------------


def loop_bodies(text: str) -> dict[str, collections.Counter]:
    """Per rate kernel: the opcode multiset of its one backward-branch loop (predicates stripped)."""
    out: dict[str, collections.Counter] = {}
    for chunk in re.split(r"\n\s+Function : ", text)[1:]:
        fn = chunk.split("\n", 1)[0]
        name = next((k for k in _KERNEL if f"{len(k)}{k}E" in fn), None)
        if name is None:
            continue
        ins = []
        for line in chunk.splitlines():
            m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)(.*?);", line)
            if m:
                ins.append((int(m.group(1), 16), m.group(2), m.group(3)))
        loops = []
        for addr, op, rest in ins:
            t = re.search(r"0x([0-9a-f]+)", rest)
            if op.startswith("BRA") and t and int(t.group(1), 16) < addr:
                loops.append((int(t.group(1), 16), addr))
        if len(loops) == 1:
            lo, hi = loops[0]
            out[name] = collections.Counter(op for addr, op, _ in ins if lo <= addr <= hi)
        else:
            out[name] = collections.Counter({"<loops>": len(loops)})
    return out


def sass_gate(so: Path, out: Path | None) -> dict:
    cuobjdump = str(Path(P4.nvcc_bin()).parent / "cuobjdump")
    text = subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    if out is not None:
        (out / "sass_w1_prices.txt").write_text(text)
    bodies = loop_bodies(text)
    kernels = {}
    for i, (name, opcode, per_loop, _, _) in enumerate(OPS):
        body = bodies.get(_KERNEL[i], collections.Counter())
        expected = {opcode: per_loop, **LOOP_EXTRA.get(i, {})}
        watched = {k: v for k, v in body.items() if _WATCHED.match(k) or k in expected}
        kernels[name] = {"kernel": _KERNEL[i], "loop": dict(sorted(body.items())), "expected": expected, "ok": watched == expected}
    return {"ok": all(k["ok"] for k in kernels.values()), "kernels": kernels}


# --- devices ------------------------------------------------------------------------------------------------------------------------


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


class Device(F8.Device):
    """ctypes wrapper over libw1_prices; ``info()`` is fp8.Device's (the same ``device_info`` entry point)."""

    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        for name in ("w1_run", "w1_device", "device_info"):
            getattr(self.lib, name).restype = ctypes.c_int
        sms, smem, khz = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        rc = self.lib.w1_device(ctypes.byref(sms), ctypes.byref(smem), ctypes.byref(khz))
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in w1_device")
        self.sms, self.smem_per_sm, self.clock_khz = sms.value, smem.value, khz.value
        assert self.lib.w1_nops() == len(OPS) and self.lib.w1_threads() == THREADS

    def run(self, op: int, iters: int, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        blocks = self.sms
        rows = np.ascontiguousarray(rows, np.uint32).reshape(32 * 8)
        chk = np.zeros(blocks * THREADS * 16, np.uint32)
        cycles, sm, ms = np.zeros(blocks, np.int64), np.zeros(blocks, np.int32), ctypes.c_float()
        rc = self.lib.w1_run(op, blocks, self.smem_per_sm // 2 + 1024, iters, _ptr(rows), _ptr(chk), _ptr(cycles), _ptr(sm), ctypes.byref(ms))
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in w1_run({OPS[op][0]})")
        return chk.reshape(blocks, THREADS, 16), cycles, sm, float(ms.value)


class FakeDevice:
    """Gate words from the host models, cycles from the Public rates at 2,400 MHz; ``corrupt`` flips one gate word once."""

    sms, smem_per_sm, clock_khz = 4, 102400, 2400000

    def __init__(self, corrupt: int | None = None):
        self.corrupt = corrupt

    def run(self, op, iters, rows):
        name, _, per_loop, work, _ = OPS[op]
        gate = iters == GATE_ITERS  # timed launches' words are never read
        chk = np.broadcast_to(expected_words(op, iters, rows) if gate else np.zeros((THREADS, 16), np.uint32), (self.sms, THREADS, 16)).copy()
        if self.corrupt is not None:
            chk.reshape(-1)[self.corrupt % chk.size] ^= 1
            self.corrupt = None
        cyc = WARPS * iters * per_loop * work / _FAKE_RATE[name]
        cycles = np.full(self.sms, int(cyc), np.int64)
        return chk, cycles, np.arange(self.sms, dtype=np.int32), cyc / (self.clock_khz * 1e3) * 1e3


# --- operands and the numeric gate --------------------------------------------------------------------------------------------------


def _rep(code: int, bits: int) -> int:
    return int(sum(code << (bits * i) for i in range(32 // bits)))


GATE_ROW = {  # per op: the all-ones row (a0..a3, b0, b1, scale A, scale B), or the scalar ops' x0, y, m
    0: [_rep(0x38, 8)] * 6 + [0, 0], 1: [_rep(0x38, 8)] * 6 + [0, 0], 2: [_rep(0x38, 8)] * 6 + [127, 127],
    3: [_rep(0x2, 4)] * 6 + [_rep(0x38, 8)] * 2, 4: [_rep(0x3F80, 16)] * 6 + [0, 0], 5: [_rep(0x3C00, 16)] * 6 + [0, 0],
    6: [0, 0x3F800000, 0x3F800000] + [0] * 5, 7: [0, 0x3F800000, 0x3F800000] + [0] * 5,
    10: [0, 0, 0x3F800000, 0x40000000] + [0] * 4, 11: [0, 0, 0x3F800000, 0x40000000] + [0] * 4,  # low half adds 1, high half 2
    SP_OP: [_rep(0x38, 8)] * 6 + [_rep(0b0100, 4), 0],  # every group keeps its first two: 32 products of 1 per output
}


def _f16_bits(x: float) -> int:
    return int(np.float16(x).view(np.uint16))


def f2fp_model(iters: int, rows: np.ndarray, packed: bool = True) -> np.ndarray:
    """h[lane][8] of f2fp_rate (packed): per iteration and constant x_k, chain k % 8 adds the words P(f, x_k) | P(x_k, f) << 16 and
    the same for g; of f2fp_u16_rate: chain k % 8 adds P(f, x_k) + P(x_k, f) and chain (k + 1) % 8 the same for g.  Mod 2^32, with
    P(hi, lo) = code(hi) << 8 | code(lo)."""
    rows = np.asarray(rows, np.uint32).reshape(32, 8)
    h = np.zeros((32, NCHAIN), np.uint32)
    xs = rows[:, 1:2] + (np.arange(2 * NCHAIN, dtype=np.uint32) << 20)
    cx = F8.cvt_ref("e4m3", xs).astype(np.uint32)
    for it in range(iters):
        for s, off in ((0, 0), (1, 1 << 21)):
            c = F8.cvt_ref("e4m3", rows[:, 0] + np.uint32(it + off)).astype(np.uint32)[:, None]
            if packed:
                v, s = ((c << 8) | cx) | (((cx << 8) | c) << 16), 0
            else:
                v = ((c << 8) | cx) + ((cx << 8) | c)
            for k in range(2 * NCHAIN):
                h[:, (k + s) % NCHAIN] += v[:, k]
    return h


E2M1_GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float64)


def e2m1_ref(words) -> np.ndarray:
    """cvt.rn.satfinite.e2m1x2.f32's code of each finite FP32 word: the nearest point of the E2M1 grid, a tie to the even code,
    |x| > 6 to 6, the sign kept (a negative that rounds to zero is 0x8): the rule GPU 4's ``cast.ref_e2m1`` matched bit for bit on
    1,041,112 elements (r20260930-072940-33b5)."""
    w = np.asarray(words, np.uint32)
    a = np.abs(w.view(np.float32).astype(np.float64))
    mids = (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2
    idx = np.searchsorted(mids, a, side="left")
    tie = (idx < len(mids)) & (a == mids[np.minimum(idx, len(mids) - 1)])
    idx = np.minimum(np.where(tie & (idx % 2 == 1), idx + 1, idx), 7)
    return (idx.astype(np.uint32) | ((w >> 31) << 3)).astype(np.uint32)


def f2fp_e2m1_model(iters: int, rows: np.ndarray, nk: int, packed: bool) -> np.ndarray:
    """h[lane][8] of the E2M1 loops, mod 2^32, with Q(hi, lo) = code(hi) << 4 | code(lo) and x_k = base + k * 2^25 / nk: packed,
    chain k % 8 adds the word Q(f, x_k) | Q(x_k, f) << 8 | Q(g, x_k) << 16 | Q(x_k, g) << 24; zero-extended, chain k % 8 adds
    Q(f, x_k) + Q(x_k, f) and chain (k + 1) % 8 the same for g."""
    rows = np.asarray(rows, np.uint32).reshape(32, 8)
    h = np.zeros((32, NCHAIN), np.uint32)
    cx = e2m1_ref(rows[:, 1:2] + np.arange(nk, dtype=np.uint32) * np.uint32((1 << 25) // nk))
    for it in range(iters):
        cf, cg = (e2m1_ref(rows[:, 0] + np.uint32(it + off))[:, None] for off in (0, 1 << 21))
        if packed:
            w = ((cf << 4) | cx) | (((cx << 4) | cf) << 8) | (((cg << 4) | cx) << 16) | (((cx << 4) | cg) << 24)
            for k in range(nk):
                h[:, k % NCHAIN] += w[:, k]
        else:
            vf, vg = ((cf << 4) | cx) + ((cx << 4) | cf), ((cg << 4) | cx) + ((cx << 4) | cg)
            for k in range(nk):
                h[:, k % NCHAIN] += vf[:, k]
                h[:, (k + 1) % NCHAIN] += vg[:, k]
    return h


def cs_values(base: np.ndarray, n: int) -> np.ndarray:
    """(32, n) FP32 words of w1_prices.cu's cs_values: base + j * 2^20, the sign flipped on odd j."""
    j = np.arange(n, dtype=np.uint32)
    return ((np.asarray(base, np.uint32)[:, None] + (j << 20)) ^ ((j & 1) << 31)).astype(np.uint32)


def cs_model(op: int, iters: int, rows: np.ndarray) -> np.ndarray:
    """(32, 16) chk words of a cast-and-store loop, each four staged codes in store order (byte i of word w = code 4w + i): the codes
    of the FADD chains after ``iters`` adds (ops 16, 18, 20, 22), of the constants (17), or constant 2w, f, g, constant 2w + 1 with
    the last iteration's f and g (19, 21, 23-25: the wide stores stage op 19's words)."""
    rows = np.asarray(rows, np.uint32).reshape(32, 8)

    def code(words):
        return F8.cvt_ref("e4m3", words).astype(np.uint32)

    if op in CS_FADD:
        x, y = cs_values(rows[:, 0], NCS).view(np.float32).copy(), rows[:, 1:2].view(np.float32)
        for _ in range(iters):
            x = (x + y).astype(np.float32)
        c = code(x.view(np.uint32))
    elif op == 17:
        c = code(cs_values(rows[:, 1], NCS))
    else:
        cx = code(cs_values(rows[:, 1], NCS // 2))
        f = code(rows[:, 0] + np.uint32(iters - 1))[:, None]
        g = code(rows[:, 0] + np.uint32(iters - 1) + np.uint32(1 << 21))[:, None]
        c = np.stack([cx[:, 0::2], np.repeat(f, 16, 1), np.repeat(g, 16, 1), cx[:, 1::2]], -1).reshape(32, NCS)
    c = c.reshape(32, 16, 4)
    return (c[..., 0] | (c[..., 1] << 8) | (c[..., 2] << 16) | (c[..., 3] << 24)).astype(np.uint32)


def expected_words(op: int, iters: int, rows: np.ndarray) -> np.ndarray:
    """(THREADS, 16) chk words of one block: all-ones MMA accumulators are k * UNROLL * iters, FADD / FFMA chains SUNROLL * iters."""
    name, _, _, work, _ = OPS[op]
    w = np.zeros((THREADS, 16), np.uint32)
    if op in (0, 2, 3, 4, SP_OP):
        k = work // 128 // (2 if op == SP_OP else 1)
        w[:] = np.float32(k * UNROLL * iters).view(np.uint32)
    elif op in (1, 5):
        k = work // 128
        v = _f16_bits(k * UNROLL * iters)
        w.reshape(THREADS, NACC, 4)[:, :, :2] = v | (v << 16)
    elif op in (6, 7):
        w[:, :NCHAIN] = np.float32(SUNROLL * iters).view(np.uint32)
    elif op in (10, 11):
        w[:, 0:2 * NCHAIN:2] = np.float32(SUNROLL * iters).view(np.uint32)
        w[:, 1:2 * NCHAIN:2] = np.float32(2 * SUNROLL * iters).view(np.uint32)
    elif op in E2M1_OPS:
        w[:, :NCHAIN] = np.tile(f2fp_e2m1_model(iters, rows, E2M1_NK[op], packed=op != 13), (THREADS // 32, 1))
    elif op in CS_OPS:
        w[:] = np.tile(cs_model(op, iters, rows), (THREADS // 32, 1))
    else:
        w[:, :NCHAIN] = np.tile(f2fp_model(iters, rows, packed=op == 8), (THREADS // 32, 1))
    return w


def timing_rows(op: int, rng) -> np.ndarray:
    """Random per-lane operands that keep every accumulator finite over the timed loop: FP8 / FP4 / BF16 / FP16 codes of magnitude
    about 1 with random signs (the sums are random walks), unit-order scales, FADD adds of about 2^-10, FFMA's contraction m < 1."""
    r = np.zeros((32, 8), np.uint32)
    if op in (0, 1, 2, SP_OP):
        b = (rng.integers(0, 2, (32, 6, 4)) << 7) | (rng.integers(6, 9, (32, 6, 4)) << 3) | rng.integers(0, 8, (32, 6, 4))
        r[:, :6] = (b << (8 * np.arange(4))).sum(-1)
        r[:, 6:] = rng.integers(126, 129, (32, 2))
        if op == SP_OP:  # word 6: random valid metadata, one nibble per group
            r[:, 6] = (SP_META_VALID[rng.integers(0, 6, (32, 8))] << (4 * np.arange(8, dtype=np.uint32))).sum(-1)
    elif op == 3:
        b = rng.integers(0, 16, (32, 6, 8))
        r[:, :6] = (b << (4 * np.arange(8))).sum(-1)
        s = rng.integers(0x34, 0x3C, (32, 2, 4))
        r[:, 6:] = (s << (8 * np.arange(4))).sum(-1)
    elif op in (4, 5):
        bias, mb = (127, 7) if op == 4 else (15, 10)
        e = rng.integers(bias - 1, bias + 2, (32, 6, 2))
        b = (rng.integers(0, 2, (32, 6, 2)) << 15) | (e << mb) | rng.integers(0, 1 << mb, (32, 6, 2))
        r[:, :6] = (b << (16 * np.arange(2))).sum(-1)
    elif op in (6, 7):
        r[:, 0] = np.float32(rng.uniform(1, 2, 32)).view(np.uint32)
        r[:, 1] = np.float32(rng.uniform(2**-10, 2**-9, 32) * rng.choice([-1, 1], 32)).view(np.uint32)
        r[:, 2] = np.float32(1 - rng.uniform(2**-12, 2**-11, 32)).view(np.uint32)
    elif op in (10, 11):
        r[:, :2] = np.float32(rng.uniform(1, 2, (32, 2))).view(np.uint32)
        r[:, 2:4] = np.float32(rng.uniform(2**-10, 2**-9, (32, 2)) * rng.choice([-1, 1], (32, 2))).view(np.uint32)
    elif op in E2M1_OPS:  # f over every E2M1 code and past saturation, the constants over [0.25, 8); random signs
        r[:, 0] = np.float32(rng.uniform(0.25, 8, 32) * rng.choice([-1, 1], 32)).view(np.uint32)
        r[:, 1] = np.float32(rng.uniform(0.25, 0.5, 32) * rng.choice([-1, 1], 32)).view(np.uint32)
    elif op in CS_OPS:  # the 64 values of a lane span eight binades from a base in [2^-12, 4): subnormal codes to saturation
        base = np.float32(2.0 ** rng.uniform(-12, 2, 32)).view(np.uint32)
        if op in CS_FADD:  # chains' base, and y: about 2^-13, a random sign
            r[:, 0] = base
            r[:, 1] = np.float32(rng.uniform(2**-14, 2**-13, 32) * rng.choice([-1, 1], 32)).view(np.uint32)
        else:  # f's start in [0.25, 0.5) (g two E4M3 ulps up), the constants' base
            r[:, 0] = np.float32(rng.uniform(0.25, 0.5, 32)).view(np.uint32)
            r[:, 1] = base
    else:
        r[:, 0] = np.float32(rng.uniform(1, 2, 32)).view(np.uint32)
        r[:, 1] = np.float32(rng.uniform(0.25, 0.5, 32)).view(np.uint32)
    return r.astype(np.uint32)


def numeric_gate(dev, rng, ops=None) -> tuple[dict, dict[int, np.ndarray]]:
    rows_t = {op: timing_rows(op, rng) for op in range(len(OPS))}  # every op's draw, so a subset times the same data
    res = {}
    for op, (name, *_) in enumerate(OPS):
        if ops is not None and op not in ops:
            continue
        rows = np.array(GATE_ROW[op], np.uint32)[None].repeat(32, 0) if op in GATE_ROW else rows_t[op]
        chk, _, sm, _ = dev.run(op, GATE_ITERS, rows)
        want = expected_words(op, GATE_ITERS, rows)
        diff = np.argwhere(chk != want[None])
        ref = "e2m1_ref" if op in E2M1_OPS else "fp8.cvt_ref"
        res[name] = {"iters": GATE_ITERS, "data": "all-ones" if op in GATE_ROW else f"timing rows vs {ref}",
                     "mismatches": int(len(diff)), "vacuous": not want[:, :NCHAIN].any(),
                     "distinct_sms": int(len(set(sm.tolist()))), "blocks": int(len(sm))}
        if len(diff):
            b, t, w = (int(v) for v in diff[0])
            res[name]["first_mismatch"] = {"block": b, "thread": t, "word": w, "got": f"{int(chk[b, t, w]):08x}", "want": f"{int(want[t, w]):08x}"}
    ok = all(r["mismatches"] == 0 and not r["vacuous"] and r["distinct_sms"] == r["blocks"] for r in res.values())
    return {"ok": ok, "ops": res}, rows_t


# --- timing -------------------------------------------------------------------------------------------------------------------------


def smi_sample(uuid: str | None) -> dict | None:
    if not uuid or uuid == "GPU-fake":
        return None
    return F8.smi_query(uuid, "clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active", nounits=True)


def timing(dev, rows_t: dict[int, np.ndarray], reps: int, target_cycles: int, uuid: str | None, ops=None, smi_every: int = 5) -> dict:
    """``reps`` rounds over the ops, interleaved.  nvidia-smi is sampled before the first round and after every ``smi_every``-th and
    the last (a sample is an nvidia-smi process, and the GPU idles through it); every rep records its own clock as cycles over
    event time."""
    ops = sorted(range(len(OPS)) if ops is None else ops)
    iters = {}
    for op in ops:
        dev.run(op, 64, rows_t[op])  # warm-up
        _, cyc, _, _ = dev.run(op, 256, rows_t[op])
        iters[op] = int(min(1 << 24, max(256, round(256 * target_cycles / max(1, int(np.median(cyc)))))))
    per = {OPS[op][0]: [] for op in ops}
    smi = [{"before_round": 0, **(smi_sample(uuid) or {})}]
    for rep in range(reps):
        for op in ops:
            name, _, per_loop, work, _ = OPS[op]
            _, cyc, sm, ms = dev.run(op, iters[op], rows_t[op])
            per_block = WARPS * iters[op] * per_loop * work
            rate = per_block / cyc.astype(np.float64)
            per[name].append({"rep": rep, "iters": iters[op], "rate_median": float(np.median(rate)), "rate_min": float(rate.min()),
                              "rate_max": float(rate.max()), "cycles_median": int(np.median(cyc)), "ms": ms,
                              "sm_mhz_from_cycles": float(np.max(cyc) / (ms * 1e3)) if ms > 0 else None,
                              "distinct_sms": int(len(set(sm.tolist()))), "blocks": int(len(sm)),
                              "throughput_per_s": per_block * len(cyc) / (ms * 1e-3) if ms > 0 else None})
        if (rep + 1) % smi_every == 0 or rep == reps - 1:
            smi.append({"after_round": rep, **(smi_sample(uuid) or {})})
    return {"iters": {OPS[op][0]: n for op, n in iters.items()}, "reps": per, "smi": smi}


def prices(tim: dict) -> dict:
    """Median over reps of the per-rep median rate, and W1's prices against the unit (qmma_e4m3_f32 per SM per clock)."""
    rate = {name: float(np.median([r["rate_median"] for r in reps])) for name, reps in tim["reps"].items()}
    spread = {name: [float(min(r["rate_median"] for r in reps)), float(max(r["rate_median"] for r in reps))] for name, reps in tim["reps"].items()}
    unit = rate[UNIT_OP]
    out = {"unit": f"1 / {UNIT_OP} MAC rate per SM per clock", "unit_rate_per_sm_per_clock": unit, "rates_per_sm_per_clock": rate,
           "rate_spread_over_reps": spread, "price_units": {}}
    for name, _, _, _, per in OPS:
        if name in rate:
            out["price_units"][f"{name} per {per}"] = unit / rate[name]
    for name, per, n in (("f2fp_e4m3x2", "instruction (2 codes)", 2), ("fadd_f32x2", "instruction (2 words)", 2),
                         ("f2fp_e2m1x2", "instruction (2 codes)", 2), ("qmma_sp_e4m3", "nonzero product", 2)):
        if name in rate:
            out["price_units"][f"{name} per {per}"] = n * unit / rate[name]
    if "qmma_sp_e4m3" in rate:  # instructions per SM per clock, sparse over dense (both 16 per loop)
        out["sp_over_dense_instructions"] = rate["qmma_sp_e4m3"] / (2 * unit)
    if "qmma_e4m3_f16" in rate:
        out["f16_over_f32_accumulate"] = rate["qmma_e4m3_f16"] / unit
    return out


def split(pr: dict, loops: dict[str, dict], ops=E2M1_OPS) -> dict:
    """F2FP's own cost from three loops (Derived): each loop's SM-clocks per warp per iteration, tau = per_loop * work / rate, solved
    exactly as tau = a F + b S + c U over its SASS counts: F the cast, U the uniform-datapath and branch instructions, S the rest.
    The alone price per code is the unit rate times a / 64."""
    rows, tau = [], []
    detail = {}
    for op in ops:
        name, opcode, per_loop, work, _ = OPS[op]
        body = loops[name]
        f = body.get(opcode, 0)
        u = sum(v for k, v in body.items() if k.startswith(("U", "BRA")))
        s = sum(body.values()) - f - u
        t = per_loop * work / pr["rates_per_sm_per_clock"][name]
        rows.append([f, s, u])
        tau.append(t)
        detail[name] = {"F": f, "S": s, "U": u, "tau": t}
    try:
        a, b, c = (float(v) for v in np.linalg.solve(np.array(rows, float), np.array(tau)))
    except np.linalg.LinAlgError:
        return {"ok": False, "detail": "singular loop counts", "loops": detail}
    work = OPS[ops[0]][3]
    return {"ok": True, "model": "tau = a F + b S + c U, SM-clocks per warp per loop iteration", "loops": detail,
            "a_per_cast": a, "b_per_support": b, "c_per_uniform": c, "cast_alone_codes_per_sm_per_clock": work / a,
            "cast_alone_price_per_code": pr["unit_rate_per_sm_per_clock"] * a / work}


# --- main ---------------------------------------------------------------------------------------------------------------------------


def op_indices(names: str | None) -> list[int]:
    """``--ops`` -> op indices in OPS order, the unit op always among them; an unknown name is an error."""
    if not names:
        return list(range(len(OPS)))
    idx = {name: i for i, (name, *_) in enumerate(OPS)}
    want = {s.strip() for s in names.split(",") if s.strip()} | {UNIT_OP}
    unknown = sorted(want - set(idx))
    if unknown:
        raise SystemExit(f"w1.py: unknown --ops {unknown}; known: {sorted(idx)}")
    return sorted(idx[n] for n in want)


def run(args) -> tuple[int, dict]:
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res: dict = {"schema": "tc-probe-fp4/w1/v1", "fake_silicon": bool(args.fake_silicon), "seed": args.seed, "reps": args.reps,
                 "target_cycles": args.target_cycles, "clocks": args.clocks}
    failures: list[str] = []
    ops = op_indices(args.ops)
    res["ops"] = [OPS[op][0] for op in ops]
    so = None
    if args.prebuilt or not args.fake_silicon or Path(P4.nvcc_bin()).exists():
        so, res["nvcc"] = (Path(args.prebuilt), f"prebuilt {args.prebuilt}") if args.prebuilt else P4.build(F8.ARCH, out, SOURCE)
        res["sass"] = sass_gate(so, out)
        if not res["sass"]["ok"]:
            failures.append("SASS gate: " + ", ".join(k for k, r in res["sass"]["kernels"].items() if not r["ok"]))
    else:
        res["sass"] = {"ok": None, "detail": "skipped: fake silicon without nvcc"}
    if args.compile_only or failures:
        return finish(out, args, res, failures, None, t0)
    if args.lease_max_min:
        return lease_child(out, args, res, so, t0)
    if args.fake_silicon:
        dev, ident = FakeDevice(args.corrupt_word), {"device": "fake-silicon", "compute_capability": "12.0", "gpu_uuid": "GPU-fake"}
    else:
        dev = Device(so)
        ident, id_fail = F8.identity_gate(dev, args.expect_gpu_uuid)
        failures += id_fail
        ident.update({"sms": dev.sms, "smem_per_sm": dev.smem_per_sm, "clock_khz_attr": dev.clock_khz})
    res["identity"] = ident
    if failures:
        return finish(out, args, res, failures, ident, t0)
    rng = np.random.default_rng(args.seed)
    res["numeric_gate"], rows_t = numeric_gate(dev, rng, ops)
    if not res["numeric_gate"]["ok"]:
        failures.append("numeric gate: " + ", ".join(k for k, r in res["numeric_gate"]["ops"].items()
                                                     if r["mismatches"] or r["vacuous"] or r["distinct_sms"] != r["blocks"]))
        return finish(out, args, res, failures, ident, t0)
    res["timing"] = timing(dev, rows_t, args.reps, args.target_cycles, ident.get("gpu_uuid"), ops, args.smi_every)
    res["prices"] = prices(res["timing"])
    if res["sass"].get("kernels") and set(E2M1_OPS) <= set(ops):
        res["prices"]["e2m1_split"] = split(res["prices"], {k: r["loop"] for k, r in res["sass"]["kernels"].items()})
    return finish(out, args, res, failures, ident, t0)


def lease_child(out: Path, args, res: dict, so: Path, t0: float) -> tuple[int, dict]:
    """The GPU part under ``gpu-lease 1 --wait --max-min M``: a child run on the library already built and SASS-gated here, so the
    lease holds the GPU only for the identity and numeric gates and the timing.  Its results become this run's."""
    sub = out / "lease"
    child = [sys.executable, str(Path(__file__).resolve()), "--out", str(sub), "--prebuilt", str(so), "--seed", str(args.seed),
             "--reps", str(args.reps), "--target-cycles", str(args.target_cycles), "--clocks", args.clocks, "--smi-every", str(args.smi_every)]
    if args.ops:
        child += ["--ops", args.ops]
    env = {k: v for k, v in os.environ.items() if k != "RESEARCH_RUN_ID"}
    t = time.time()
    rc = subprocess.run(["gpu-lease", "1", "--wait", "--max-min", str(args.lease_max_min), "--", *child], env=env, check=False).returncode
    res["lease"] = {"rc": rc, "seconds": round(time.time() - t, 3), "max_min": args.lease_max_min}
    try:
        got = json.loads((sub / "w1_results.json").read_text())
    except (OSError, ValueError):
        return finish(out, args, res, [f"the leased child wrote no results (exit {rc})"], None, t0)
    for k in ("identity", "numeric_gate", "timing", "prices"):
        if k in got:
            res[k] = got[k]
    failures = list(got.get("failures", [])) or ([f"the leased child exited {rc}"] if rc else [])
    return finish(out, args, res, failures, res.get("identity"), t0)


def finish(out: Path, args, res: dict, failures: list[str], ident: dict | None, t0: float) -> tuple[int, dict]:
    res["failures"], res["validation"] = failures, ("passed" if not failures else "failed")
    res["seconds"] = round(time.time() - t0, 3)
    (out / "w1_results.json").write_text(json.dumps(res, indent=1, default=str))
    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id and not args.fake_silicon:
        pr = res.get("prices") or {}
        label = "Measured, " + res["clocks"] + " clocks"
        meas = [{"name": f"rate.{k}", "value": v, "unit": f"{OPS[[o[0] for o in OPS].index(k)][4]}/SM/clock", "label": label}
                for k, v in (pr.get("rates_per_sm_per_clock") or {}).items()]
        meas += [{"name": f"price.{k}", "value": v, "unit": "W1 units", "label": label} for k, v in (pr.get("price_units") or {}).items()]
        if "f16_over_f32_accumulate" in pr:
            meas.append({"name": "ratio.qmma_e4m3_f16_over_f32", "value": pr["f16_over_f32_accumulate"], "unit": "1", "label": label})
        if "sp_over_dense_instructions" in pr:
            meas.append({"name": "ratio.qmma_sp_e4m3_over_dense_instructions", "value": pr["sp_over_dense_instructions"], "unit": "1",
                         "label": label})
        sp = pr.get("e2m1_split") or {}
        if sp.get("ok"):
            derived = "Derived (three-loop split of Measured " + res["clocks"] + " rates over their SASS counts)"
            meas += [{"name": f"split.e2m1.{k}", "value": sp[k], "unit": u, "label": derived}
                     for k, u in (("a_per_cast", "SM-clocks/warp instruction"), ("b_per_support", "SM-clocks/warp instruction"),
                                  ("c_per_uniform", "SM-clocks/warp instruction"), ("cast_alone_price_per_code", "W1 units"))]
        doc = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": run_id, "arch": F8.ARCH,
               "sku": (ident or {}).get("device"), "driver": (ident or {}).get("driver"), "gpu_uuid": (ident or {}).get("gpu_uuid"),
               "uuid_checked": (ident or {}).get("uuid_checked"), "nvcc": res.get("nvcc"), "clocks": res["clocks"],
               "sass_loops": {k: r["loop"] for k, r in (res.get("sass") or {}).get("kernels", {}).items()},
               "measurements": meas + [{"name": "wall", "value": res["seconds"], "unit": "s"}],
               "workload_fingerprint": {"probe": "tools/tc_probe_fp4/w1.py", "seed": args.seed, "reps": args.reps,
                                        "target_cycles": args.target_cycles, "compile_only": args.compile_only, "ops": res["ops"]},
               "validation": {"status": res["validation"], "checker": "tc_probe_fp4/w1.py: identity, SASS loop, numeric and one-block-per-SM gates",
                              "detail": "; ".join(failures) or "every gate passed", "evidence": ["w1_results.json"]},
               "artifacts": sorted(p.name for p in out.iterdir() if p.suffix in (".json", ".txt") and p.name != "result.json")}
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1, default=str))
        os.replace(tmp, out / "result.json")
    return (0 if not failures else 1), res


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=F8.SEED_DEFAULT)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--target-cycles", type=int, default=1 << 27, help="cycles per timed launch (about 50 ms at 2.6 GHz)")
    ap.add_argument("--smi-every", type=int, default=5, help="an nvidia-smi sample before the first rep round and after every Nth and the last")
    ap.add_argument("--compile-only", action="store_true", help="build + SASS gate only (no GPU)")
    ap.add_argument("--clocks", default="unlocked", help="the clock state the operator set, as results label it (e.g. locked-2100)")
    ap.add_argument("--ops", default=None, help=f"comma list of op names to gate and time (default all; {UNIT_OP}, the unit, always)")
    ap.add_argument("--prebuilt", default=None, help="an already built libw1_prices .so (the SASS gate still reads it)")
    ap.add_argument("--lease-max-min", type=int, default=None,
                    help="build and SASS-gate here, then run the GPU part under gpu-lease 1 --wait --max-min M")
    ap.add_argument("--expect-gpu-uuid", default=os.environ.get("GPU_LEASE_UUID"),
                    help="the device's UUID (default: gpu-lease's GPU_LEASE_UUID); a mismatch stops the run")
    ap.add_argument("--fake-silicon", action="store_true", help="host models stand in for the device (CPU dry run; never evidence)")
    ap.add_argument("--corrupt-word", type=int, default=None, help="with --fake-silicon: flip one bit of one gate word; the run must fail")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.fake_silicon and os.environ.get("RESEARCH_RUN_ID"):
        print("w1.py: --fake-silicon is a dry run and never evidence; refused under research run", file=sys.stderr)
        return 2
    if args.corrupt_word is not None and not args.fake_silicon:
        print("w1.py: --corrupt-word needs --fake-silicon", file=sys.stderr)
        return 2
    if args.smi_every < 1:
        print("w1.py: --smi-every must be at least 1", file=sys.stderr)
        return 2
    if args.lease_max_min and (args.fake_silicon or args.prebuilt):
        print("w1.py: --lease-max-min builds here and leases a real GPU; not with --fake-silicon or --prebuilt", file=sys.stderr)
        return 2
    code, res = run(args)
    print(json.dumps({"validation": res["validation"], "failures": res["failures"], "prices": (res.get("prices") or {}).get("price_units"),
                      "seconds": res["seconds"]}))
    return code


if __name__ == "__main__":
    sys.exit(main())
