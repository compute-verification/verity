#!/usr/bin/env python3
"""W1 price microbenchmarks on sm_120 (the RTX PRO 6000): issue rates per SM per clock of the dense FP8 atom (f32 and f16
accumulation), the 2:4-sparse E4M3 atom (mma.sp m16n8k64), the block-scaled mxf8f6f4 and nvf4 atoms, BF16 and FP16 mma.sync, FP32
FADD and FFMA, the packed add.rn.f32x2 and the same adds as plain C pairs, the E4M3 cast and the E2M1 cast (three loops, whose split
gives the cast's own cost), the E4M3 cast-and-store of Pearl-C's sm_120 forming kernel (one code per convert, two per 16-bit store)
beside its packed replacement (four codes per 32-bit store) and the packed words through 64- and 128-bit stores (at conflict-free row
strides and at the port's), each fed by one FADD per code or by no per-code input, the E4M3-to-FP16 convert cvt.rn.f16x2.e4m3x2
(alone, and with one LOP3 per two or per one convert, so the loops' differences test that the costs add), FP16 mma.sync with FP32
accumulation (the leaf beside the E4M3 atom), the 16-bit adds (HADD2 alone, add.rn.f16x2 as ptxas splits it over HADD2 and HFMA2,
HFMA2, the widening HADD2.F32 and the mixed FHADD, each in SM-clocks per warp instruction and as a fraction of FADD's rate), their
co-issue (the leaf's 16 MMAs with N = 4 ... 256 HADD2 per loop iteration: both rates per N, each as a fraction of its loop alone),
the FP8 v1 rewrite's per-MMA instruction mix (64 HMMA.16816.F32 a loop iteration among its packed-half adds, FADDs, ldmatrix,
converts and a shared-memory round trip, 8 warps an SM at 232-240 registers: t_mix / t_solo against the leaf), and the prices W1
reads off them.

    python tools/tc_probe_fp4/w1.py --out $RESEARCH_RUN_DIR --expect-gpu-uuid GPU-... [--reps 10] [--target-cycles 134217728]
                                    [--ops qmma_e4m3_f32,qmma_sp_e4m3] [--prebuilt libw1_prices_sm_120a.so] [--lease-max-min 10]
                                    [--lease-via-fill OWNER] [--smi-every 5]

``--lease-max-min M`` builds and SASS-gates here, then runs the rest under ``gpu-lease 1 --wait --max-min M`` on the built library;
with ``--lease-via-fill OWNER`` that part is instead one GPU fill job of OWNER's on node 2's fill runner (``gpus=1 prio=10``), which
this run queues and waits for, so it preempts no one's fill chunk and still records its results here.

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
exactly their FADD feed, support and stores too; the FP16 converts' and 16-bit adds' loops are pinned whole, the mix loops whole
but their MOVs, with no spill or FTZ in the kernel and their registers in the window; its full loop composition is recorded); the
numeric gate (the mix loops against ``mix_model`` in exact integers on per-lane rows of the FP16 1/4 grid; all-ones operands give
exact integers for every atom and FADD / FFMA, and for the packed and paired adds 1 in the low word and 2 in the high, so a swapped
half shows; the casts are checked against ``fp8.cvt_ref`` and ``e2m1_ref`` on the timing data, the cast-and-store loops on the words
they leave in shared memory, the FP16 converts against the registered decode ``verity.ml.tc.term.decode_e4m3`` on their chains
and on every 16-bit input, where a NaN code's half must be an FP16 NaN and its bits are recorded, the 16-bit adds' chains on the
timing data against ``h16_chain``, one rounding per step of an exact float64 result, and an op whose expected words are all zero
fails as vacuous); one block per SM (distinct %smid).  Every gated launch starts from buffers poisoned with 0xA5, and the
words an op doesn't write must still read 0xA5A5A5A5; each op's no-write negative control (no kernel launched) must be rejected, or
the gate fails.  A failed gate writes no measurements.  ``--fake-silicon``
replaces the device with a model that returns the gate's words and cycles from the Public datasheet rates (a CPU dry run of the
plumbing, refused under research run); ``--corrupt-word I`` flips one bit of one gate word, and the run must then fail.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import functools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from verity.ml.tc.term import decode_e4m3

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
    # E4M3 to FP16, two codes a convert: chains alone, then with one LOP3 per two converts and per convert
    ("f2fp_f16_e4m3", "F2FP.F16.E4M3.UNPACK_B", NCHAIN * SUNROLL, 64, "code"),
    ("f2fp_f16_e4m3_half", "F2FP.F16.E4M3.UNPACK_B", NCHAIN * SUNROLL, 64, "code"),
    ("f2fp_f16_e4m3_xor", "F2FP.F16.E4M3.UNPACK_B", NCHAIN * SUNROLL, 64, "code"),
    ("hmma_f16_f32", "HMMA.16816.F32", NACC * UNROLL, 16 * 8 * 16, "MAC"),
    # 16-bit adds, one register (two FP16 halves, or one FP32 word) written per instruction
    ("hadd2", "HADD2", NCHAIN * SUNROLL, 32, "register"),
    ("add_f16x2", "HADD2", NCHAIN * SUNROLL, 32, "register"),
    ("hfma2", "HFMA2", NCHAIN * SUNROLL, 32, "register"),
    ("hadd2_f32", "HADD2.F32", NCHAIN * SUNROLL, 32, "register"),
    ("fhadd", "FHADD", NCHAIN * SUNROLL, 32, "register"),
]
#: FP16 MMA / HADD2 co-issue: hmma_f16_f32's 16 MMAs with N HADD2 per loop iteration (the rate is the MMAs', per MAC)
COISSUE_N = (4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256)
COISSUE_OPS = tuple(range(len(OPS), len(OPS) + len(COISSUE_N)))
OPS += [(f"coissue_h{n}", "HMMA.16816.F32", NACC * UNROLL, 16 * 8 * 16, "MAC") for n in COISSUE_N]
HADD2_PER_LOOP = dict(zip(COISSUE_OPS, COISSUE_N))
#: the FP8 v1 rewrite's instruction mix per FP16 MMA (bc-3006c44a's rewrite-mix.md), one block of MIX_WARPS warps per SM: per warp and
#: loop iteration 64 HMMA.16816.F32 among (packed-half adds, FADD, ldmatrix, E4M3-to-FP16 converts), w1_prices.cu's MixP
MIX = {"r1_lb_flat": (120, 76, 30, 3), "r1_lb_br": (120, 248, 30, 3), "r2_lb_flat": (352, 84, 28, 3), "r2_lb_br": (352, 144, 28, 3),
       "r2_con_flat": (992, 84, 28, 3), "h0_f0": (0, 0, 0, 0), "h0_f17": (0, 68, 0, 0),
       **{f"h{n}_f17": (4 * n, 68, 28, 0) for n in (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256)}}
MIX_OPS = tuple(range(len(OPS), len(OPS) + len(MIX)))
OPS += [(f"mix_{name}", "HMMA.16816.F32", 64, 16 * 8 * 16, "MAC") for name in MIX]
MIX_COUNTS = dict(zip(MIX_OPS, MIX.values()))
MIX_WARPS = 8
#: the rewrite's registers per thread, which __maxnreg__(240) and each kernel's ballast hold; h256_f17 and r2_con_flat land at 226
#: with two ballast words and spill with four, so their floor is lower
MIX_REG = (232, 240)
MIX_REG_FLOOR = {"mix_h256_f17": 224, "mix_r2_con_flat": 224}
UNIT_OP = "qmma_e4m3_f32"
_KERNEL = ["qmma_e4m3_f32_rate", "qmma_e4m3_f16_rate", "qmma_mxf8_rate", "omma_nvf4_rate", "hmma_bf16_f32_rate", "hmma_f16_f16_rate",
           "fadd_rate", "ffma_rate", "f2fp_rate", "f2fp_u16_rate", "fadd_f32x2_rate", "fadd_pairs_rate",
           "f2fp_e2m1_rate", "f2fp_e2m1_u8_rate", "f2fp_e2m1_x2_rate", "qmma_sp_e4m3_rate",
           "cs_port_fadd_rate", "cs_port_nogen_rate", "cs_packed_fadd_rate", "cs_packed_nogen_rate",
           "cs_w64_fadd_rate", "cs_w64_nogen_rate", "cs_w128_fadd_rate", "cs_w128_nogen_rate",
           "cs_w64_nogen_r144_rate", "cs_w128_nogen_r144_rate", "f2fp_f16_rate", "f2fp_f16_half_rate",
           "f2fp_f16_xor_rate", "hmma_f16_f32_rate", "hadd2_rate", "add_f16x2_rate", "hfma2_rate", "hadd2_f32_rate",
           "fhadd_rate", *(f"coissue_h{n}_rate" for n in COISSUE_N), *(f"mix_{name}_rate" for name in MIX)]  # w1_prices.cu's KERNELS
SP_OP = 15
CS_OPS = tuple(range(16, 26))
CS_FADD = (16, 18, 20, 22)  # fed by one FADD per code; the others need no per-code input
NCS = 64  # codes per thread per loop iteration
#: the rest of each cast-and-store loop body, exactly (the watched opcodes and these; FADD.FTZ would be a different opcode)
LOOP_EXTRA = {16: {"FADD": 64, "LOP3.LUT": 32, "IMAD": 32, "STS.U16": 32}, 17: {"LOP3.LUT": 32, "IMAD": 32, "STS.U16": 32},
              18: {"FADD": 64, "STS": 16}, 19: {"STS": 16}, 20: {"FADD": 64, "STS.64": 8}, 21: {"STS.64": 8},
              22: {"FADD": 64, "STS.128": 4}, 23: {"STS.128": 4}, 24: {"STS.64": 8}, 25: {"STS.128": 4},
              27: {"LOP3.LUT": 32}, 28: {"LOP3.LUT": 64}}
F16_OPS = (26, 27, 28)
F16_FEED = {26: 0, 27: 1, 28: 2}  # the lane's constant XORed in before no convert, every other one, every one
H16_OPS = (30, 31, 32, 33, 34)
#: loops whose opcode isn't one instruction: ptxas lowers half of add.rn.f16x2 to HFMA2 (x * 1 + y)
LOOP_EXACT = {31: {"HADD2": 32, "HFMA2": 32}}
E4M3_NAN = (0x7F, 0xFF)
POISON = 0xA5A5A5A5
#: chk words each op's kernel writes, a prefix of its 16; the rest must keep w1_run's poison
WRITTEN = {op: 16 for op in range(len(OPS))} | {op: NCHAIN for op in (6, 7, 8, 9, 12, 13, 14, *H16_OPS)}
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
              "cs_w64_nogen": 128, "cs_w128_fadd": 100, "cs_w128_nogen": 128, "cs_w64_nogen_r144": 128, "cs_w128_nogen_r144": 128,
              "f2fp_f16_e4m3": 128, "f2fp_f16_e4m3_half": 96, "f2fp_f16_e4m3_xor": 64, "hmma_f16_f32": 512, "hadd2": 64,
              "add_f16x2": 128, "hfma2": 64, "hadd2_f32": 64, "fhadd": 128,
              # the co-issue loops as if the HADD2 pipe (0.5 SM-clocks each) overlapped the MMAs' 64 SM-clocks per warp iteration
              **{f"coissue_h{n}": 16 * 2048 / max(64, 0.5 * n) for n in COISSUE_N},
              # the mixes as if HADD2 (0.5 SM-clocks) and FADD (0.25) overlapped the MMAs' 256 SM-clocks per warp iteration
              **{f"mix_{name}": 64 * 2048 / max(256, 0.5 * nh + 0.25 * nf) for name, (nh, nf, _, _) in MIX.items()}}


def op_warps(op: int) -> int:
    """Warps per block: the mix kernels run MIX_WARPS (w1_prices.cu's MIX_THREADS), the rest WARPS."""
    return MIX_WARPS if op in MIX_OPS else WARPS


# --- SASS gate ----------------------------------------------------------------------------------------------------------------------


def _kernel_name(mangled: str) -> str | None:
    return next((k for k in _KERNEL if f"{len(k)}{k}E" in mangled), None)


def kernel_instructions(text: str) -> dict[str, list[tuple[int, str, str]]]:
    """Per rate kernel: its instructions as (address, opcode with its predicate stripped, operands)."""
    out = {}
    for chunk in re.split(r"\n\s+Function : ", text)[1:]:
        name = _kernel_name(chunk.split("\n", 1)[0])
        if name is None:
            continue
        ins = []
        for line in chunk.splitlines():
            m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)(.*?);", line)
            if m:
                ins.append((int(m.group(1), 16), m.group(2), m.group(3)))
        out[name] = ins
    return out


def res_usage(text: str) -> dict[str, dict[str, int]]:
    """Per rate kernel: REG, STACK and LOCAL from ``cuobjdump -res-usage``."""
    out = {}
    for m in re.finditer(r"Function (\S+):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:\d+ LOCAL:(\d+)", text):
        name = _kernel_name(m.group(1))
        if name is not None:
            out[name] = {"REG": int(m.group(2)), "STACK": int(m.group(3)), "LOCAL": int(m.group(4))}
    return out


def loop_bodies(text: str) -> dict[str, collections.Counter]:
    """Per rate kernel: the opcode multiset of its one backward-branch loop (predicates stripped)."""
    out: dict[str, collections.Counter] = {}
    for name, ins in kernel_instructions(text).items():
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
    usage = res_usage(subprocess.run([cuobjdump, "-res-usage", str(so)], capture_output=True, text=True, check=True).stdout)
    if out is not None:
        (out / "sass_w1_prices.txt").write_text(text)
    bodies, ins = loop_bodies(text), kernel_instructions(text)
    kernels = {}
    for i, (name, opcode, per_loop, _, _) in enumerate(OPS):
        body = bodies.get(_KERNEL[i], collections.Counter())
        expected = LOOP_EXACT.get(i) or {opcode: per_loop, **LOOP_EXTRA.get(i, {})}
        watched = {k: v for k, v in body.items() if _WATCHED.match(k) or k in expected}
        if i in F16_OPS + H16_OPS:  # the whole loop but its uniform-datapath counter and branch
            watched = {k: v for k, v in body.items() if not k.startswith(("U", "BRA"))}
        if i in COISSUE_OPS:  # the whole loop but those and the NOPs ptxas spaces the MMAs with, recorded in the loop
            expected = {opcode: per_loop, "HADD2": HADD2_PER_LOOP[i]}
            watched = {k: v for k, v in body.items() if not k.startswith(("U", "BRA", "NOP"))}
        rec = {"kernel": _KERNEL[i], "loop": dict(sorted(body.items()))}
        ok = True
        if i in MIX_OPS:  # the whole loop but those and the register copies (MOV), recorded; no spill or FTZ in the kernel
            expected = mix_loop(i)
            watched = {k: v for k, v in body.items() if not (k.startswith(("U", "BRA")) or k in ("NOP", "MOV"))}
            reg, lo = usage.get(_KERNEL[i], {}), MIX_REG_FLOOR.get(name, MIX_REG[0])
            bad = sorted({op for _, op, _ in ins.get(_KERNEL[i], []) if ".FTZ" in op or op.startswith(("LDL", "STL"))})
            rec |= {"registers": reg, "register_window": [lo, MIX_REG[1]], "forbidden_in_kernel": bad, "movs": body.get("MOV", 0)}
            ok = not bad and reg.get("STACK") == 0 and reg.get("LOCAL") == 0 and lo <= reg.get("REG", -1) <= MIX_REG[1]
        kernels[name] = rec | {"expected": expected, "ok": ok and watched == expected}
    return {"ok": all(k["ok"] for k in kernels.values()), "kernels": kernels}


def mix_loop(op: int) -> dict[str, int]:
    """A mix loop's instructions besides the uniform datapath, branch, NOPs and MOVs: its 64 MMAs, NH HADD2, NF FADD, NC converts,
    the ldmatrix (x4 for the A pool, x2 for the B pool) and 4 STS / LDS (the promotions' shared-memory round trip) when NF > 0."""
    nh, nf, nl, nc = MIX_COUNTS[op]
    want = {"HMMA.16816.F32": 64, "HADD2": nh, "FADD": nf, "F2FP.F16.E4M3.UNPACK_B": nc, "LDSM.16.M88.4": (nl + 1) // 2,
            "LDSM.16.M88.2": nl // 2, "STS": 4 * (nf > 0), "LDS": 4 * (nf > 0)}
    return {k: v for k, v in want.items() if v}


# --- devices ------------------------------------------------------------------------------------------------------------------------


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


class Device(F8.Device):
    """ctypes wrapper over libw1_prices; ``info()`` is fp8.Device's (the same ``device_info`` entry point)."""

    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        for name in ("w1_run", "w1_device", "device_info", "w1_op_threads"):
            getattr(self.lib, name).restype = ctypes.c_int
        sms, smem, khz = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        rc = self.lib.w1_device(ctypes.byref(sms), ctypes.byref(smem), ctypes.byref(khz))
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in w1_device")
        self.sms, self.smem_per_sm, self.clock_khz = sms.value, smem.value, khz.value
        assert self.lib.w1_nops() == len(OPS) and self.lib.w1_threads() == THREADS
        assert all(self.lib.w1_op_threads(op) == 32 * op_warps(op) for op in range(len(OPS)))

    def run(self, op: int, iters: int, rows: np.ndarray, negative: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """One launch; ``negative`` is the no-write control: w1_run poisons the buffers and launches nothing."""
        blocks = self.sms
        rows = np.ascontiguousarray(rows, np.uint32).reshape(32 * 8)
        chk = np.zeros(blocks * THREADS * 16, np.uint32)
        cycles, sm, ms = np.zeros(blocks, np.int64), np.zeros(blocks, np.int32), ctypes.c_float()
        rc = self.lib.w1_run(op, blocks, self.smem_per_sm // 2 + 1024, iters, _ptr(rows), _ptr(chk), _ptr(cycles), _ptr(sm), ctypes.byref(ms),
                             int(negative))
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in w1_run({OPS[op][0]})")
        return chk.reshape(blocks, THREADS, 16), cycles, sm, float(ms.value)


class FakeDevice:
    """Gate words from the host models, cycles from the Public rates at 2,400 MHz; ``corrupt`` flips one gate word once."""

    sms, smem_per_sm, clock_khz = 4, 102400, 2400000

    def __init__(self, corrupt: int | None = None):
        self.corrupt = corrupt

    def run(self, op, iters, rows, negative=False):
        name, _, per_loop, work, _ = OPS[op]
        if negative:
            poisoned = np.full(self.sms, POISON, np.uint32)
            return (np.full((self.sms, THREADS, 16), POISON, np.uint32), np.repeat(poisoned, 2).view(np.int64), poisoned.view(np.int32), 0.0)
        gate = iters == GATE_ITERS  # timed launches' words are never read
        chk = gate_words(op, iters, rows, self.sms)[0] if gate else np.full((self.sms, THREADS, 16), POISON, np.uint32)
        if self.corrupt is not None:
            chk.reshape(-1)[self.corrupt % chk.size] ^= 1
            self.corrupt = None
        cyc = op_warps(op) * iters * per_loop * work / _FAKE_RATE[name]
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
    29: [_rep(0x3C00, 16)] * 6 + [0, 0],
    **{op: [_rep(0x3C00, 16)] * 6 + [0, 0] for op in COISSUE_OPS},  # words 6 and 7, the HADD2 chains', from the timing rows
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


def _e4m3_f16_bits() -> np.ndarray:
    """(256,) uint16: each E4M3 code's value under the registered decode (``verity.ml.tc.term.decode_e4m3``) as FP16 bits, exact for
    every finite code (|x| <= 448, and the least subnormal 2^-9 is an FP16 normal); a NaN code's entry is 0x7FFF, which the gate
    never compares (the decode names no bits for NaN)."""
    out = np.zeros(256, np.uint16)
    for c in range(256):
        d = decode_e4m3(c)
        if isinstance(d, str):
            out[c] = 0x7FFF
            continue
        sign, exp, sig = d
        v = np.float16(sig * 2.0**exp)
        assert float(v) == sig * 2.0**exp
        out[c] = int(v.view(np.uint16)) | sign << 15
    return out


E4M3_F16 = _e4m3_f16_bits()


def f16_decode(s) -> tuple[np.ndarray, np.ndarray]:
    """cvt.rn.f16x2.e4m3x2 of 16-bit inputs under the registered decode (the high code to the high half), and the mask of the halves
    whose code is NaN."""
    s = np.asarray(s, np.uint32) & np.uint32(0xFFFF)
    hi, lo = s >> 8, s & np.uint32(0xFF)
    words = (E4M3_F16[hi].astype(np.uint32) << 16) | E4M3_F16[lo]
    nan = np.where(np.isin(hi, E4M3_NAN), np.uint32(0xFFFF0000), np.uint32(0)) | np.where(np.isin(lo, E4M3_NAN), np.uint32(0xFFFF), np.uint32(0))
    return words.astype(np.uint32), nan.astype(np.uint32)


def f16_chains(op: int, iters: int, rows: np.ndarray) -> np.ndarray:
    """(32, NCHAIN) chain words of an FP16-convert loop after ``iters`` iterations of SUNROLL converts: chain j starts at row word j,
    and each convert reads the chain's low half, XORed first with row word 7 where the op feeds it.  A chain that would read a NaN
    code raises: the decode names no bits for it, and timing_rows never draws one."""
    rows = np.asarray(rows, np.uint32).reshape(32, 8)
    x, y = rows.copy(), rows[:, 7:8]
    for _ in range(iters):
        for u in range(SUNROLL):
            v = x ^ y if F16_FEED[op] == 2 or (F16_FEED[op] == 1 and u % 2 == 0) else x
            x, nan = f16_decode(v)
            if nan.any():
                raise ValueError(f"{OPS[op][0]}: a chain reads a NaN code at step {u}; these rows can't be gated")
    return x


def f16_words(op: int, iters: int, rows: np.ndarray, blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """(blocks, THREADS, 16) chk words of an FP16-convert op and their NaN mask: words 0..7 of thread tid the converts of the 16-bit
    inputs 8 * tid + w (every input once per 8,192 threads), words 8..15 the chains."""
    tid = np.arange(blocks * THREADS, dtype=np.uint32)[:, None]
    w, nan = np.zeros((blocks * THREADS, 16), np.uint32), np.zeros((blocks * THREADS, 16), np.uint32)
    w[:, :8], nan[:, :8] = f16_decode(8 * tid + np.arange(8, dtype=np.uint32))
    w[:, 8:] = np.tile(f16_chains(op, iters, rows), (blocks * THREADS // 32, 1))
    return w.reshape(blocks, THREADS, 16), nan.reshape(blocks, THREADS, 16)


def _exact_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a + b in float64, which must be exact (TwoSum's error zero, no infinity or NaN), so that one rounding of it to FP16 or FP32
    is the device's round-to-nearest-even; rows that would need more raise."""
    s = a + b
    bb = s - a
    if not (np.isfinite(s).all() and ((a - (s - bb)) + (b - bb) == 0).all()):
        raise ValueError("a 16-bit chain's float64 step is inexact or not finite; these rows can't be gated")
    return s


def _halves(w) -> np.ndarray:
    """(n,) f16x2 words -> (n, 2) float64 values, the low half first."""
    w = np.asarray(w, np.uint32)
    return np.stack([w & 0xFFFF, w >> 16], -1).astype(np.uint16).view(np.float16).astype(np.float64)


def _f16x2(h: np.ndarray) -> np.ndarray:
    """(n, 2) values, each rounded once to FP16 -> (n,) f16x2 words."""
    b = np.asarray(h).astype(np.float16).view(np.uint16).astype(np.uint32)
    return (b[:, 0] | (b[:, 1] << 16)).astype(np.uint32)


def h16_chain(op: int, iters: int, rows: np.ndarray) -> np.ndarray:
    """(32,) chain words of a 16-bit add loop after ``iters`` iterations of SUNROLL steps from row word 0, with y row word 1 and m
    row word 2 as f16x2: HADD2 adds y's high half to both halves (30), add.rn.f16x2 and HFMA2 act per half (31: x + y, 32:
    x * m + y), HADD2.F32 widens the FP32 chain's high half, read as FP16 (33), FHADD adds y's low half to the FP32 chain (34)."""
    rows = np.asarray(rows, np.uint32).reshape(32, 8)
    return h16_steps(op, iters * SUNROLL, rows[:, 0], rows[:, 1], rows[:, 2])


def h16_steps(op: int, n: int, x0: np.ndarray, y: np.ndarray, m: np.ndarray) -> np.ndarray:
    """h16_chain's chains after ``n`` steps from the words x0, with addend y and multiplier m."""
    x0, y, m = (np.asarray(v, np.uint32) for v in (x0, y, m))
    if op == 33:
        x = x0.copy()
        for _ in range(n):
            x = (x >> 16).astype(np.uint16).view(np.float16).astype(np.float32).view(np.uint32)
        return x
    if op == 34:
        x, yl = x0.view(np.float32).astype(np.float64), _halves(y)[:, 0]
        for _ in range(n):
            x = _exact_sum(x, yl).astype(np.float32).astype(np.float64)
        return x.astype(np.float32).view(np.uint32)
    x, yh, mh = _halves(x0), _halves(y), _halves(m)
    if op == 30:
        yh = yh[:, 1:].repeat(2, 1)
    for _ in range(n):
        x = _exact_sum(x * mh if op == 32 else x, yh).astype(np.float16).astype(np.float64)  # an FP16 product is exact in float64
    return _f16x2(x)


def coissue_chains(op: int, iters: int, rows: np.ndarray) -> np.ndarray:
    """(32, NCHAIN) HADD2 chains of a co-issue loop: HADD2 number h of an iteration steps chain h % NCHAIN, so chain j takes
    len(range(j, N, NCHAIN)) steps an iteration, from row word 6 with the addend row word 7's high half."""
    rows = np.asarray(rows, np.uint32).reshape(32, 8)
    steps = [len(range(j, HADD2_PER_LOOP[op], NCHAIN)) for j in range(NCHAIN)]
    by_steps = {s: h16_steps(30, iters * s, rows[:, 6], rows[:, 7], rows[:, 7]) for s in set(steps)}
    return np.stack([by_steps[s] for s in steps], 1)


def mix_params(op: int) -> dict:
    """w1_prices.cu's MixP: NA A-form block-adds (4 HADD2 each) and NB B-form ones (2 each), chain u's shares na[u] and nb[u], the
    pools of NSA x4 and NSB x2 sub-blocks, M merges an iteration, and whether each group's MMAs restart from zero."""
    nh, nf, nl, nc = MIX_COUNTS[op]
    na_ = 2 * (nh // 12)
    nb_ = (nh - 4 * na_) // 2
    return {"NH": nh, "NF": nf, "NL": nl, "NC": nc, "NA": na_, "NB": nb_, "NSA": (nl + 1) // 2 if nl else 1, "NSB": nl // 2 if nl else 1,
            "M": (nf - 64) // 4 if nf else 0, "restart": nh > 0 and nf > 0,
            "na": [(u + 1) * na_ // 4 - u * na_ // 4 for u in range(4)], "nb": [(u + 1) * nb_ // 4 - u * nb_ // 4 for u in range(4)]}


@functools.cache
def mix_schedule(op: int) -> tuple[tuple[tuple, ...], ...]:
    """A mix loop iteration's 64 slots, each its steps in mix_slot's order: ("ldsm", role, t, matrices) loads sub-block t of the A
    pool (role 0, an x4) or the B pool (1, an x2), register q from the warp's matrix matrices[q]; ("cvt", t) converts word 1 of B
    sub-block t; ("add", role, c, lhs, rhs) is a block-add on chain c's form (role 0: words 0..3 from the A pool, 1: words 4 and 5
    from the B pool), X = pool[lhs] + pool[rhs], or X + pool[rhs] when lhs is None; ("mma", c, zero) is chain c's MMA on its
    form, from zero or from acc[c]; ("merge", c) is acc[c] += acc[(c + 2) % 4]; ("promote", c, q) is tot[4q + i] += acc[c][i]."""
    p = mix_params(op)
    slots = []
    for k in range(64):
        c, w, r = k % 4, k % 16, k // 4
        steps: list[tuple] = []
        for ld in range(k * p["NL"] // 64, (k + 1) * p["NL"] // 64):
            j = ld // 2
            steps.append(("ldsm", 0, j % p["NSA"], tuple(4 * (j % 4) + q for q in range(4))) if ld % 2 == 0 else
                         ("ldsm", 1, j % p["NSB"], tuple(2 * (j % 8) + q for q in range(2))))
        for i in range(p["NC"]):
            j = i * p["NSB"] // p["NC"]
            if ((2 * j + 2) * 64 + p["NL"] - 1) // p["NL"] - 1 == k:  # the slot that loads B sub-block j (ldmatrix 2j + 1)
                steps.append(("cvt", j))
        for role, n, ns in ((0, p["na"][c], p["NSA"]), (1, p["nb"][c], p["NSB"])):
            for a in range((r * n + 15) // 16, ((r + 1) * n + 15) // 16):
                steps.append(("add", role, c, c % ns if a == 0 else None, (4 * (a + 1) + c) % ns))
        steps.append(("mma", c, p["restart"] and w < 4))
        if p["NF"] and 4 <= w < 12:
            e = 8 * (k // 16) + w - 4
            steps += [("merge", c)] * ((e + 1) * p["M"] // 32 - e * p["M"] // 32)
        elif p["NF"] and w >= 12:
            steps.append(("promote", c, 4 * (k // 16) + c))
        slots.append(tuple(steps))
    return tuple(slots)


def _quarters(words) -> np.ndarray:
    """f16x2 words -> (..., 2) int64, each half times 4, the low half first; a half off the 1/4 grid (or not finite) raises."""
    h = _halves(words) * 4
    if not (np.isfinite(h).all() and (h == np.rint(h)).all()):
        raise ValueError("a mix operand is off the FP16 1/4 grid; these rows can't be gated")
    return h.astype(np.int64)


def _bounded(v: np.ndarray, limit: int, what: str) -> np.ndarray:
    if np.abs(v).max(initial=0) > limit:
        raise ValueError(f"mix_model: {what} leaves the range where every order of rounding is exact; these rows can't be gated")
    return v


F16_QUARTERS, F32_SIXTEENTHS = 1 << 11, (1 << 24) - 1  # |4x| <= 2^11 is exact in FP16, |16x| < 2^24 in FP32


def mma16816(x: np.ndarray, acc: np.ndarray | None) -> np.ndarray:
    """One warp's HMMA.16816.F32 in exact integers: x (6, 32, 2) the lanes' form halves times 4 (words 0..3 A, 4 and 5 B), acc
    (4, 32) their accumulators times 16 or None for zero; returns D's (4, 32).  Lane 4g + t holds A[g][2t, 2t + 1] (a0),
    A[g + 8][..] (a1), A[g][2t + 8, 2t + 9] (a2), A[g + 8][..] (a3), B[2t, 2t + 1][g] (b0), B[2t + 8, 2t + 9][g] (b1), D[g][2t, 2t + 1]
    (c0, c1) and D[g + 8][..] (c2, c3), the low half the lower index.  The sum of the products' and C's magnitudes stays under 2^20,
    all on the 1/16 grid, so every order and alignment of the accumulation is exact."""
    q = x.reshape(6, 8, 8)
    a, b = np.block([[q[0], q[2]], [q[1], q[3]]]), np.concatenate([q[4].T, q[5].T])
    cm = np.zeros((16, 8), np.int64) if acc is None else np.concatenate([acc[0:2].T.reshape(8, 8), acc[2:4].T.reshape(8, 8)])
    _bounded(np.abs(a) @ np.abs(b) + np.abs(cm), F32_SIXTEENTHS, "an MMA")
    d = a @ b + cm
    return np.concatenate([d[:8].reshape(32, 2).T, d[8:].reshape(32, 2).T])


def mix_model(op: int, iters: int, rows: np.ndarray) -> np.ndarray:
    """(32, 32) chk words per lane of a mix loop after ``iters`` iterations of mix_schedule, in exact integers: words 0..15 acc[c][i]
    at 4c + i (the ballast's XOR into word 0 is zero), 16 + q the sum tot[q] + tot[q + 16] + tot[q + 32] + tot[q + 48].  Matrix m
    of a warp is its lanes' row words m % 8, their halves swapped for m >= 8, so ldmatrix hands each lane its own words; the forms
    start at row words 0..5 and the pools and sums at 0.  Rows whose values leave the FP16 1/4 grid, or the exact range of any step,
    raise (mix_gate_rows keeps them in)."""
    p = mix_params(op)
    rows = np.asarray(rows, np.uint32).reshape(32, 8)
    tile = [rows[:, m] if m < 8 else (rows[:, m - 8] >> np.uint32(16)) | (rows[:, m - 8] << np.uint32(16)) for m in range(16)]
    pools = [np.zeros((p["NSA"], 4, 32), np.uint32), np.zeros((p["NSB"], 2, 32), np.uint32)]
    x = np.stack([_quarters(rows[:, :6].T)] * 4)
    acc, tot = np.zeros((4, 4, 32), np.int64), np.zeros((64, 32), np.int64)
    for _ in range(iters):
        for slot in mix_schedule(op):
            for step in slot:
                if step[0] == "ldsm":
                    pools[step[1]][step[2]] = [tile[m] for m in step[3]]
                elif step[0] == "cvt":
                    words, nan = f16_decode(pools[1][step[1], 1])
                    if nan.any():
                        raise ValueError("mix_model: a convert reads a NaN code; these rows can't be gated")
                    pools[1][step[1], 1] = words
                elif step[0] == "add":
                    _, role, c, lhs, rhs = step
                    off, nr = (0, 4) if role == 0 else (4, 2)
                    left = x[c, off:off + nr] if lhs is None else _quarters(pools[role][lhs])
                    x[c, off:off + nr] = _bounded(left + _quarters(pools[role][rhs]), F16_QUARTERS, "a packed-half add")
                elif step[0] == "mma":
                    acc[step[1]] = mma16816(x[step[1]], None if step[2] else acc[step[1]])
                elif step[0] == "merge":
                    c = step[1]
                    acc[c] = _bounded(acc[c] + acc[(c + 2) % 4], F32_SIXTEENTHS, "a merge")
                else:
                    _, c, q = step
                    tot[4 * q:4 * q + 4] = _bounded(tot[4 * q:4 * q + 4] + acc[c], F32_SIXTEENTHS, "a promotion")
    s = tot[:16]
    for j in (1, 2, 3):
        s = _bounded(s + tot[16 * j:16 * j + 16], F32_SIXTEENTHS, "the totals' sum")
    words = np.concatenate([acc.reshape(16, 32), s]).T / 16
    return words.astype(np.float32).view(np.uint32)


#: the gate's FP16 halves (0.25, 0.5, 0.75, 1, with random signs): on the 1/4 grid, and so is the E4M3 value of each high byte (0.75,
#: 1, 1.25, 1.5), which is what a convert reads of a low half (its low byte, 0x00, converts to zero); at most 3 the widest mixes
#: reach 2^21 sixteenths, past FP32's exact range, and these keep them under half of it
MIX_GATE_HALVES = np.array([0x3400, 0x3800, 0x3A00, 0x3C00], np.uint32)


def mix_gate_rows(rng) -> np.ndarray:
    """(32, 8) per-lane row words for the mix gate: every half from MIX_GATE_HALVES with a random sign."""
    h = MIX_GATE_HALVES[rng.integers(0, len(MIX_GATE_HALVES), (32, 8, 2))] | (rng.integers(0, 2, (32, 8, 2)).astype(np.uint32) << 15)
    return (h[..., 0] | (h[..., 1] << 16)).astype(np.uint32)


def gate_words(op: int, iters: int, rows: np.ndarray, blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """(blocks, THREADS, 16) chk words each block must hold, the words the op doesn't write at POISON, and the mask of FP16 halves
    that decode a NaN code (see gate_mismatches)."""
    if op in F16_OPS:
        want, nan = f16_words(op, iters, rows, blocks)
    else:
        want = np.broadcast_to(expected_words(op, iters, rows), (blocks, THREADS, 16)).copy()
        nan = np.zeros_like(want)
    want[..., WRITTEN[op]:] = POISON
    return want, nan


def gate_mismatches(chk: np.ndarray, want: np.ndarray, nan: np.ndarray) -> np.ndarray:
    """Where chk fails: every bit outside the NaN halves equal to want's, and each NaN half an FP16 NaN (exponent all ones,
    fraction nonzero), whatever its bits."""
    bad = (chk & ~nan) != (want & ~nan)
    for sh in (0, 16):
        half = (chk >> np.uint32(sh)) & np.uint32(0xFFFF)
        is_nan = ((half & 0x7C00) == 0x7C00) & ((half & 0x3FF) != 0)
        bad |= (((nan >> np.uint32(sh)) & np.uint32(0xFFFF)) != 0) & ~is_nan
    return bad


def f16_nan_outputs(chk: np.ndarray) -> dict:
    """The FP16 bits the device wrote for each NaN code, read off the table words (0..7) of every block."""
    blocks = chk.shape[0]
    s = (8 * np.arange(blocks * THREADS, dtype=np.uint32)[:, None] + np.arange(8, dtype=np.uint32)) & np.uint32(0xFFFF)
    tab = chk.reshape(-1, 16)[:, :8]
    out = {}
    for code in E4M3_NAN:
        vals = np.concatenate([(tab >> 16)[(s >> 8) == code], (tab & 0xFFFF)[(s & 0xFF) == code]])
        out[f"0x{code:02x}"] = sorted(f"{int(v):04x}" for v in np.unique(vals))
    return out


def expected_words(op: int, iters: int, rows: np.ndarray) -> np.ndarray:
    """(THREADS, 16) chk words of one block (block 0 for the FP16 converts, whose table words differ by block): all-ones MMA
    accumulators are k * UNROLL * iters, FADD / FFMA chains SUNROLL * iters."""
    name, _, _, work, _ = OPS[op]
    if op in F16_OPS:
        return f16_words(op, iters, rows, 1)[0][0]
    w = np.zeros((THREADS, 16), np.uint32)
    if op in (0, 2, 3, 4, SP_OP, 29):
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
    elif op in H16_OPS:
        w[:, :NCHAIN] = np.tile(h16_chain(op, iters, rows)[:, None], (THREADS // 32, NCHAIN))
    elif op in COISSUE_OPS:  # even warps the all-ones accumulators, odd warps the chains and then the poison
        w[:] = np.float32(work // 128 * UNROLL * iters).view(np.uint32)
        odd = (np.arange(THREADS) // 32) % 2 == 1
        w[odd, :NCHAIN] = np.tile(coissue_chains(op, iters, rows), (THREADS // 64, 1))
        w[odd, NCHAIN:] = POISON
    elif op in MIX_OPS:  # MIX_WARPS identical warps, each thread t writing 32 words: rows 2t and 2t + 1
        w[:] = np.tile(mix_model(op, iters, rows).reshape(64, 16), (THREADS // 64, 1))
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
    elif op in (4, 5, 29, *COISSUE_OPS):
        bias, mb = (127, 7) if op == 4 else (15, 10)
        e = rng.integers(bias - 1, bias + 2, (32, 6, 2))
        b = (rng.integers(0, 2, (32, 6, 2)) << 15) | (e << mb) | rng.integers(0, 1 << mb, (32, 6, 2))
        r[:, :6] = (b << (16 * np.arange(2))).sum(-1)
        if op in COISSUE_OPS:  # hadd2's chains: from [1, 2), y about 2^-10, random signs
            r[:, 6] = _f16x2(rng.uniform(1, 2, (32, 2)) * rng.choice([-1, 1], (32, 2)))
            r[:, 7] = _f16x2(rng.uniform(2**-10, 2**-9, (32, 2)) * rng.choice([-1, 1], (32, 2)))
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
    elif op in F16_OPS:
        # A result's low half is the FP16 image of an E4M3 value: low byte 0x00 or 0x80, high byte <= 0x5F apart from the sign.  So
        # with y's (word 7's) low byte not 0x7F either way round and its high byte under 0x20 apart from the sign, no convert after
        # the first reads a NaN code; the chain starts (words 0..6) are drawn so the first doesn't either, as they are and XORed
        # with y.  Chain 7 starts at y.
        lo = rng.integers(0, 0x7F, 32) | (rng.integers(0, 2, 32) << 7)
        hi = rng.integers(0, 0x20, 32) | (rng.integers(0, 2, 32) << 7)
        r[:, 7] = (rng.integers(0, 1 << 16, 32) << 16) | (hi << 8) | lo
        y16 = r[:, 7:8] & 0xFFFF

        def has_nan(v):
            return np.isin(v & 0xFF, E4M3_NAN) | np.isin((v >> 8) & 0xFF, E4M3_NAN)

        r[:, :7] = rng.integers(0, 1 << 32, (32, 7), dtype=np.uint64)
        bad = has_nan(r[:, :7] & 0xFFFF) | has_nan((r[:, :7] & 0xFFFF) ^ y16)
        while bad.any():
            r[:, :7][bad] = rng.integers(0, 1 << 32, int(bad.sum()), dtype=np.uint64)
            bad = has_nan(r[:, :7] & 0xFFFF) | has_nan((r[:, :7] & 0xFFFF) ^ y16)
    elif op in H16_OPS:
        # FP16 chains from [1, 2) and FP32 chains from [0.25, 4), random signs; adds of about 2^-10 with random signs, so FP16 chains
        # stall below 8 in magnitude and FP32 chains below 2^16; HFMA2's contraction m = 1 - k 2^-11, k in 1..8, pulls toward
        # y / (1 - m), under 4.  Every step of h16_chain is then exact in float64 before its one rounding.
        sign = rng.choice([-1, 1], (32, 2))
        if op in (33, 34):
            r[:, 0] = np.float32(rng.uniform(0.25, 4, 32) * sign[:, 0]).view(np.uint32)
        else:
            r[:, 0] = _f16x2(rng.uniform(1, 2, (32, 2)) * sign)
        r[:, 1] = _f16x2(rng.uniform(2**-10, 2**-9, (32, 2)) * rng.choice([-1, 1], (32, 2)))
        r[:, 2] = _f16x2(1 - rng.integers(1, 9, (32, 2)) * 2.0**-11)
    elif op in MIX_OPS:
        # FP16 halves in [0.5, 2) with random signs and fraction bit 6 clear: a convert reads a low half's bytes, the high one
        # E4M3 [1, 1.875] and the low one an exponent field under 8 (|x| < 2), so never a NaN code, and the forms stay under 2^8
        b = (rng.integers(0, 2, (32, 8, 2)) << 15) | (rng.integers(14, 16, (32, 8, 2)) << 10) | (rng.integers(0, 1 << 10, (32, 8, 2)) & ~(1 << 6))
        r[:] = (b << (16 * np.arange(2))).sum(-1)
    else:
        r[:, 0] = np.float32(rng.uniform(1, 2, 32)).view(np.uint32)
        r[:, 1] = np.float32(rng.uniform(0.25, 0.5, 32)).view(np.uint32)
    return r.astype(np.uint32)


def numeric_gate(dev, rng, ops=None) -> tuple[dict, dict[int, np.ndarray]]:
    rows_t = {op: timing_rows(op, rng) for op in range(len(OPS))}  # every op's draw, so a subset times the same data
    rows_mix = mix_gate_rows(rng)
    res = {}
    for op, (name, *_) in enumerate(OPS):
        if ops is not None and op not in ops:
            continue
        rows = np.array(GATE_ROW[op], np.uint32)[None].repeat(32, 0) if op in GATE_ROW else rows_mix if op in MIX_OPS else rows_t[op]
        if op in COISSUE_OPS:
            rows[:, 6:] = rows_t[op][:, 6:]
        chk, _, sm, _ = dev.run(op, GATE_ITERS, rows)
        want, nan = gate_words(op, GATE_ITERS, rows, chk.shape[0])
        diff = np.argwhere(gate_mismatches(chk, want, nan))
        ref = ("e2m1_ref" if op in E2M1_OPS else "verity.ml.tc.term.decode_e4m3" if op in F16_OPS
               else "h16_chain (one rounding per step of an exact float64 result)" if op in H16_OPS + COISSUE_OPS else "fp8.cvt_ref")
        data = f"timing rows vs {ref}" if op not in GATE_ROW else "all-ones"
        if op in COISSUE_OPS:
            data = f"all-ones MMA on even warps; odd warps' HADD2 chains on timing rows vs {ref}, their words 8..15 the poison"
        if op in MIX_OPS:
            data = "per-lane FP16 halves on the 1/4 grid (mix_gate_rows) vs mix_model, exact integers"
        res[name] = {"iters": GATE_ITERS, "data": data,
                     "mismatches": int(len(diff)), "vacuous": not want[..., :NCHAIN].any(),
                     "distinct_sms": int(len(set(sm.tolist()))), "blocks": int(len(sm)),
                     "unwritten_words_per_thread": 16 - WRITTEN[op]}
        if op in F16_OPS:
            res[name]["inputs_converted"] = int(min(1 << 16, chk.shape[0] * THREADS * 8))
            res[name]["nan_outputs"] = f16_nan_outputs(chk)
        if len(diff):
            b, t, w = (int(v) for v in diff[0])
            res[name]["first_mismatch"] = {"block": b, "thread": t, "word": w, "got": f"{int(chk[b, t, w]):08x}", "want": f"{int(want[b, t, w]):08x}"}
        nchk, _, nsm, _ = dev.run(op, GATE_ITERS, rows, negative=True)
        n_bad, n_sms = int(gate_mismatches(nchk, want, nan).sum()), len(set(nsm.tolist()))
        res[name]["negative_control"] = {"writes": "none: no kernel launched, chk, cycles and sm keep their 0xA5 poison",
                                         "mismatches": n_bad, "distinct_sms": n_sms, "rejected": n_bad > 0 or n_sms != len(nsm)}
    ok = all(r["mismatches"] == 0 and not r["vacuous"] and r["distinct_sms"] == r["blocks"] and r["negative_control"]["rejected"]
             for r in res.values())
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
            per_block = op_warps(op) * iters[op] * per_loop * work
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
                         ("f2fp_e2m1x2", "instruction (2 codes)", 2), ("qmma_sp_e4m3", "nonzero product", 2),
                         ("f2fp_f16_e4m3", "instruction (2 codes)", 2)):
        if name in rate:
            out["price_units"][f"{name} per {per}"] = n * unit / rate[name]
    f16 = [OPS[op] for op in F16_OPS]
    if all(o[0] in rate for o in f16):  # SM-clocks per warp per loop iteration; each step adds 32 LOP3 to the same 64 converts
        tau = [o[2] * o[3] / rate[o[0]] for o in f16]
        out["f16_lop3_steps"] = {"sm_clocks_per_warp_iteration": dict(zip((o[0] for o in f16), tau)),
                                 "first_32_lop3": tau[1] - tau[0], "second_32_lop3": tau[2] - tau[1]}
    scalar = {name: rate[name] for name in ("fadd", *(OPS[op][0] for op in H16_OPS)) if name in rate}
    if set(scalar) - {"fadd"}:  # every one of these writes one register per warp lane per instruction
        out["fp16_scalar"] = {name: {"sm_clocks_per_warp_instruction": 32 / r,
                                     **({"fraction_of_fadd_rate": r / rate["fadd"]} if "fadd" in rate else {})}
                              for name, r in scalar.items()}
    co = [(0, "hmma_f16_f32")] + [(HADD2_PER_LOOP[op], OPS[op][0]) for op in COISSUE_OPS]
    if any(name in rate for _, name in co[1:]):  # n = 0 is the leaf's own loop; HADD2's rate is the MMAs' times n 32 / (16 2048)
        mma0, h_alone = rate.get("hmma_f16_f32"), rate.get("hadd2")
        tab = {}
        for n, name in co:
            if name not in rate:
                continue
            mma = rate[name]
            h = mma * n * 32 / (NACC * UNROLL * 2048)
            row = {"hadd2_per_iteration": n, "mma_macs_per_sm_per_clock": mma, "hadd2_registers_per_sm_per_clock": h,
                   "sm_clocks_per_warp_iteration": NACC * UNROLL * 2048 / mma}
            if mma0:
                row["mma_fraction_of_alone"] = mma / mma0
            if h_alone:
                row["hadd2_fraction_of_alone"] = h / h_alone
            if mma0 and h_alone:
                row["utilization_sum"] = mma / mma0 + h / h_alone
            tab[name] = row
        out["coissue"] = tab
    mix = [op for op in MIX_OPS if OPS[op][0] in rate]
    if mix:  # t_mix / t_solo: the leaf's MAC rate (hmma_f16_f32, its 16 MMAs alone, 16 warps) over the mix's; and over mix_h0_f0's
        solo, alone = rate.get("hmma_f16_f32"), rate.get("mix_h0_f0")
        tab = {}
        for op in mix:
            name, (nh, nf, nl, nc) = OPS[op][0], MIX_COUNTS[op]
            row = {"per_16_hmma": {"hadd2": nh / 4, "fadd": nf / 4, "ldmatrix": nl / 4, "cvt": nc / 4, "sts_lds_pairs": int(nf > 0)},
                   "mma_macs_per_sm_per_clock": rate[name], "sm_clocks_per_warp_iteration": 64 * 2048 / rate[name]}
            if solo:
                row["t_mix_over_t_solo"] = solo / rate[name]
            if alone:
                row["t_mix_over_t_h0_f0"] = alone / rate[name]
            tab[name] = row
        out["rewrite_mix"] = tab
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
                                                     if r["mismatches"] or r["vacuous"] or r["distinct_sms"] != r["blocks"]
                                                     or not r["negative_control"]["rejected"]))
        return finish(out, args, res, failures, ident, t0)
    res["timing"] = timing(dev, rows_t, args.reps, args.target_cycles, ident.get("gpu_uuid"), ops, args.smi_every)
    res["prices"] = prices(res["timing"])
    if res["sass"].get("kernels") and set(E2M1_OPS) <= set(ops):
        res["prices"]["e2m1_split"] = split(res["prices"], {k: r["loop"] for k, r in res["sass"]["kernels"].items()})
    return finish(out, args, res, failures, ident, t0)


FILL_DIR = Path(os.environ.get("FILL_DIR", "/workspace/pouw/fill"))
FILL_POLL_S = 5


def _fill_events(job: str) -> list[dict]:
    events = []
    for line in (FILL_DIR / "events.jsonl").read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("job") == job:
            events.append(e)
    return events


def fill_lease(child: list[str], owner: str, max_min: int, sub: Path, wait_min: int = 90, sleep=time.sleep) -> tuple[int, dict]:
    """Run ``child`` as one GPU fill job of ``owner`` on the fill runner in FILL_DIR (``gpus=1 prio=10``, one chunk of at most
    ``max_min`` minutes, any GPU) and wait until the runner moves it to done/ or failed/.  The script is staged as a dotfile, which
    the runner skips, then renamed into queue/.  The runner leases the GPU (``gpu-lease 1 --preemptible``, which sets
    GPU_LEASE_UUID); a preempted chunk is killed with its script, requeued and rerun from the top, which the child can, since it
    only gates and times.  The script writes the child's exit status into ``sub`` and exits 0, because the runner retries any other
    exit once: a failed gate must fail this run, not be retried into a pass.  A job still queued after ``wait_min`` minutes is
    withdrawn (exit 124).  Returns the child's exit status (the runner's if the child never finished), with the job's events; its
    logs are copied into ``sub``."""
    job = f"w1-{os.environ.get('RESEARCH_RUN_ID') or os.getpid()}.sh"
    queue = FILL_DIR / "queue"
    stage = queue / f".{job}"
    sub.mkdir(parents=True, exist_ok=True)
    rc_file = sub / "fill_child_rc.txt"
    rc_file.unlink(missing_ok=True)
    stage.write_text("\n".join(["#!/bin/bash", f"# fill: owner={owner} gpus=1 max_min={max_min} cpus=8 project=pous prio=10",
                                f"cd {shlex.quote(os.getcwd())} || exit 1", " ".join(shlex.quote(c) for c in child),
                                f"echo $? > {shlex.quote(str(rc_file))}", "exit 0"]) + "\n")
    stage.chmod(0o755)
    os.replace(stage, queue / job)
    waited = 0
    while not any((FILL_DIR / d / job).exists() for d in ("done", "failed")):
        if waited >= wait_min * 60 and (queue / job).exists():
            (queue / job).unlink(missing_ok=True)
            return 124, {"job": job, "withdrawn": f"still queued after {wait_min} min", "events": _fill_events(job)}
        sleep(FILL_POLL_S)
        waited += FILL_POLL_S
    for _ in range(12):  # the runner moves a finished job before it appends the event
        events = _fill_events(job)
        if events and events[-1]["event"] in ("done", "failed"):
            break
        sleep(FILL_POLL_S)
    for log in sorted((FILL_DIR / "logs").glob(f"{job}.*.log")):
        shutil.copy(log, sub / f"fill-{log.name}")
    try:
        return int(rc_file.read_text()), {"job": job, "events": events}
    except (OSError, ValueError):
        rc = next((e["rc"] for e in reversed(events) if isinstance(e.get("rc"), int)), 1)
        return rc or 1, {"job": job, "events": events, "child_rc": "not written: the child never finished"}


def lease_child(out: Path, args, res: dict, so: Path, t0: float) -> tuple[int, dict]:
    """The GPU part under ``gpu-lease 1 --wait --max-min M``: a child run on the library already built and SASS-gated here, so the
    lease holds the GPU only for the identity and numeric gates and the timing.  Its results become this run's."""
    sub = out / "lease"
    child = [sys.executable, str(Path(__file__).resolve()), "--out", str(sub), "--prebuilt", str(so), "--seed", str(args.seed),
             "--reps", str(args.reps), "--target-cycles", str(args.target_cycles), "--clocks", args.clocks, "--smi-every", str(args.smi_every)]
    if args.ops:
        child += ["--ops", args.ops]
    t = time.time()
    if args.lease_via_fill:
        rc, fill = fill_lease(child, args.lease_via_fill, args.lease_max_min, sub)
        res["lease"] = {"rc": rc, "seconds": round(time.time() - t, 3), "max_min": args.lease_max_min, "via": "fill", **fill}
    else:
        env = {k: v for k, v in os.environ.items() if k != "RESEARCH_RUN_ID"}
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
        st = pr.get("f16_lop3_steps") or {}
        if st:
            derived = "Derived (differences of Measured " + res["clocks"] + " loop periods)"
            meas += [{"name": f"f16_lop3_steps.{k}", "value": st[k], "unit": "SM-clocks/warp iteration", "label": derived}
                     for k in ("first_32_lop3", "second_32_lop3")]
        derived = "Derived (from Measured " + res["clocks"] + " rates)"
        for name, v in (pr.get("fp16_scalar") or {}).items():
            meas += [{"name": f"fp16_scalar.{name}.{k}", "value": x, "unit": u, "label": derived}
                     for k, u in (("sm_clocks_per_warp_instruction", "SM-clocks/warp instruction"), ("fraction_of_fadd_rate", "1"))
                     if (x := v.get(k)) is not None]
        for name, v in (pr.get("coissue") or {}).items():
            meas += [{"name": f"coissue.{name}.{k}", "value": x, "unit": u, "label": derived}
                     for k, u in (("hadd2_registers_per_sm_per_clock", "register/SM/clock"), ("mma_fraction_of_alone", "1"),
                                  ("hadd2_fraction_of_alone", "1"), ("utilization_sum", "1"))
                     if (x := v.get(k)) is not None]
        for name, v in (pr.get("rewrite_mix") or {}).items():
            meas += [{"name": f"rewrite_mix.{name}.{k}", "value": x, "unit": "1", "label": derived}
                     for k in ("t_mix_over_t_solo", "t_mix_over_t_h0_f0") if (x := v.get(k)) is not None]
        doc = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": run_id, "arch": F8.ARCH,
               "sku": (ident or {}).get("device"), "driver": (ident or {}).get("driver"), "gpu_uuid": (ident or {}).get("gpu_uuid"),
               "uuid_checked": (ident or {}).get("uuid_checked"), "nvcc": res.get("nvcc"), "clocks": res["clocks"],
               "sass_loops": {k: r["loop"] for k, r in (res.get("sass") or {}).get("kernels", {}).items()},
               "measurements": meas + [{"name": "wall", "value": res["seconds"], "unit": "s"}],
               "workload_fingerprint": {"probe": "tools/tc_probe_fp4/w1.py", "seed": args.seed, "reps": args.reps,
                                        "target_cycles": args.target_cycles, "compile_only": args.compile_only, "ops": res["ops"]},
               "validation": {"status": res["validation"],
                              "checker": "tc_probe_fp4/w1.py: identity, SASS loop, numeric (poisoned buffers, a rejected no-write "
                                         "negative control per op) and one-block-per-SM gates",
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
    ap.add_argument("--lease-via-fill", default=None, metavar="OWNER",
                    help="with --lease-max-min: run the GPU part as one fill job of OWNER (gpus=1 prio=10) on node 2's fill runner and wait")
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
    if args.lease_via_fill and not args.lease_max_min:
        print("w1.py: --lease-via-fill needs --lease-max-min (the fill job's max_min)", file=sys.stderr)
        return 2
    code, res = run(args)
    print(json.dumps({"validation": res["validation"], "failures": res["failures"], "prices": (res.get("prices") or {}).get("price_units"),
                      "seconds": res["seconds"]}))
    return code


if __name__ == "__main__":
    sys.exit(main())
