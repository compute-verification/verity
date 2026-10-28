"""F1 (`fp4-tile-only/sm120`): every multiplier on sm_120a priced per nonzero product against dense NVFP4.

The FP4-tile model claims that no sm_120 instruction multiplies faster than the NVFP4 m16n8k64
tile per nonzero product.  This tool measures that claim over every ``mma.sync`` / ``mma.sp::ordered_metadata`` kind, shape,
type and sparsity ptxas accepts for sm_120a, and over the CUDA-core multipliers, each in the same process as dense NVFP4:

    python benchmarks/tc_probe_fp4/f1_rates.py build --out DIR                  # ptxas acceptance of every form + the timed .so + SASS gates
    python benchmarks/tc_probe_fp4/f1_rates.py run --build DIR --family fp4 --out DIR2 --expect-uuid GPU-... --locked-mhz 2100
    python benchmarks/tc_probe_fp4/f1_rates.py scan-libs --out DIR [--libs a.so,b.so]   # every sm_120 opcode in NVIDIA's libraries
    python benchmarks/tc_probe_fp4/f1_rates.py collect --build DIR --runs D1,D2,... [--libscan DIR] --out DIR3   # the price table

``build`` compiles each form alone (``nvcc -cubin``) and records whether ptxas accepts it and what its timed loop lowers to, then
builds one shared library of the accepted forms (``f1_<form>`` kernels; ``FORMS`` below is the whole list, ``FAMILIES`` the
chunks).  ``run`` times one family: each form's kernel is ``NSETS`` register-resident operand sets rotating through ``NACC``
accumulator chains (tensor forms: one warp instruction each, 16 warps per SM) or ``SIMT_CH`` chains x ``SIMT_U`` dependent steps
per thread (CUDA-core forms: every thread, the SM's full occupancy), interleaved round-robin with dense NVFP4 (``nvf4``), whose
nonzero-product rate is the unit.  A form's price is ``rate(nvf4) / rate(form)`` in FP4 slots per nonzero product (the model's
table: dense NVFP4 1, FP8 2, BF16 4, TF32 8, FFMA about 17), where a sparse form's nonzero products are half its logical MACs.

Gates, before any timing (a failed gate writes no measurements): the device UUID and compute capability 12.0; the build's SASS
gate (each timed loop holds its form's instruction count, and no kernel in the library has an FP32 ``FADD``/``FFMA``/``FMUL``
with ``.FTZ``; the build passes no ``-ftz``/fast-math flag); every kernel's outputs finite and bit-identical
across two launches; every launch one wave; and, after timing, every kernel's outputs finite at its timed length (a failure there
publishes no measurements either).  ``--corrupt-word`` flips one gate word: the run must then fail.  A form ptxas
emulates (s4, u4, b1 and FP16 m8n8k4 on sm_120a: one ``CALL`` per PTX instruction to a subroutine of unpacking and other MMAs)
passes the SASS gate as ``emulated``, with its callee's MMAs recorded, and is timed as that route: its price is the emulation's.

Operands are uniform random codes of each form's types (small magnitudes where the accumulator is FP16, so no chain reaches inf);
the sparse forms carry real 2:4 metadata (every nibble one of the six index pairs ordered_metadata accepts, drawn per lane and per
operand set; 1:2 for TF32), and ``sp_nvf4_meta_const`` repeats OMMA.SF.SP with one constant nibble.
A product narrower than an E2M1 product (``s4``, ``u4``, ``b1``) is priced per its own product and flagged: it can't
hold one E2M1 x E2M1 product (12 x 12 as integers after the x2), so it is not a like-for-like multiplier.  Labels: every rate
and price is Measured; the per-E2M1-product prices of the narrow forms are Derived from them.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import ctypes
import dataclasses
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402
from verity_config.settings import SHARED  # noqa: E402

ARCH = "sm_120a"
SCHEMA = "pouw-rtxpro/f1-rates-sm120/v1"
NSETS, NACC = 4, 8
BODY = NSETS * NACC
SIMT_CH, SIMT_U = 8, 16
TENSOR_TPB, SIMT_TPB = 128, 256
#: the unit: dense NVFP4, and the model's claim is price >= 1 for every multiplier
REF = "nvf4"
#: sparse metadata nibbles mma.sp::ordered_metadata accepts: index pairs p0 < p1 of four as (p1 << 2) | p0; TF32's 1:2 pairs
META_24 = (0b0100, 0b1000, 0b1001, 0b1100, 0b1101, 0b1110)
META_12 = (0b0100, 0b1110)
NVCC_FLAGS = ["-O2", "-std=c++17"]  # no -ftz / --use_fast_math; the build refuses them anyway
FTZ_RE = re.compile(r"\b(FADD|FFMA|FMUL)(\.[A-Z0-9_]+)*\.FTZ\b")
MMA_RE = re.compile(r"MMA")
MULT_RE = re.compile(r"MMA|FMA|MUL|MAD|IDP|DP[24]A|POPC")


@dataclasses.dataclass(frozen=True)
class Form:
    """One tensor-core instruction: ``ptx`` with register-count operands ``a`` / ``b`` / ``acc`` as (constraint, count)."""
    name: str
    family: str
    ptx: str
    mnk: tuple[int, int, int]           # logical shape
    a: tuple[str, int]
    b: tuple[str, int]
    acc: tuple[str, int]                # 'f' f32, 'h' packed f16x2 (constraint r), 'r' s32, 'd' f64
    fa: str
    fb: str
    sparse: str | None = None           # '2:4' | '1:2'
    scale: str | None = None            # 'ue8m0' | 'ue4m3'
    product: str = "e2m1-capable"       # 'e2m1-capable' | 's4' | 'u4' | 'b1'
    meta: str = "real"                  # 'real' | 'const'

    @property
    def tensor(self) -> bool:
        return True

    @property
    def copies(self) -> int:
        """Independent m x n x k products one warp instruction computes: 4 for FP16 m8n8k4 (one per quad-pair), else 1."""
        m, n, _ = self.mnk
        elems = 32 * self.acc[1] * (2 if self.acc[0] == "h" else 1)
        assert elems % (m * n) == 0, self.name
        return elems // (m * n)

    @property
    def logical_macs(self) -> int:
        m, n, k = self.mnk
        return m * n * k * self.copies

    @property
    def products(self) -> int:
        return self.logical_macs // (2 if self.sparse else 1)

    def words_per_set(self) -> int:
        w = lambda t: (2 if t[0] == "d" else 1) * t[1]  # noqa: E731
        return w(self.a) + w(self.b) + 3


@dataclasses.dataclass(frozen=True)
class SimtMultiplier:
    """One CUDA-core multiplier: ``body`` is one dependent step on chain register %0 with operands %1 (y) and %2 (z)."""
    name: str
    ptx: str
    body: str
    x: str                              # constraint of the chain: 'f' 'r' 'd' 'l'
    y: str
    z: str | None
    fmt: str                            # operand domain (op_words)
    opcode: str | tuple[str, ...]       # the SASS opcodes each step may lower to (prefix match)
    per_step: int                       # their count per step
    products: int                       # multiplications per step
    product: str = "e2m1-capable"
    family: str = "simt"

    @property
    def tensor(self) -> bool:
        return False


def _f(name, family, ptx, mnk, a, b, acc, fa, fb, **kw) -> Form:
    return Form(name, family, ptx, mnk, a, b, acc, fa, fb, **kw)


MS = "mma.sync.aligned"
SP = "mma.sp::ordered_metadata.sync.aligned"
F8 = ("e4m3", "e5m2", "e3m2", "e2m3", "e2m1")


def _forms() -> list:
    f: list = []
    R, D = "r", "d"
    # FP4: the honest tile (the unit), every other block-scaled K=64 kind, and OMMA.SF.SP on real 2:4 fragments
    f += [
        _f("nvf4", "fp4", f"{MS}.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3", (16, 8, 64),
           (R, 4), (R, 2), ("f", 4), "e2m1x8", "e2m1x8", scale="ue4m3"),
        _f("mxf4nvf4_2x_ue8m0", "fp4", f"{MS}.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
           (16, 8, 64), (R, 4), (R, 2), ("f", 4), "e2m1x8", "e2m1x8", scale="ue8m0"),
        _f("mxf4nvf4_4x_ue8m0", "fp4", f"{MS}.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
           (16, 8, 64), (R, 4), (R, 2), ("f", 4), "e2m1x8", "e2m1x8", scale="ue8m0"),
        _f("mxf4nvf4_2x_ue4m3", "fp4", f"{MS}.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
           (16, 8, 64), (R, 4), (R, 2), ("f", 4), "e2m1x8", "e2m1x8", scale="ue4m3"),
        _f("mxf4_2x", "fp4", f"{MS}.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0", (16, 8, 64),
           (R, 4), (R, 2), ("f", 4), "e2m1x8", "e2m1x8", scale="ue8m0"),
        _f("sp_nvf4", "fp4", f"{SP}.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3", (16, 8, 128),
           (R, 4), (R, 4), ("f", 4), "e2m1x8", "e2m1x8", sparse="2:4", scale="ue4m3"),
        _f("sp_nvf4_meta_const", "fp4", f"{SP}.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3",
           (16, 8, 128), (R, 4), (R, 4), ("f", 4), "e2m1x8", "e2m1x8", sparse="2:4", scale="ue4m3", meta="const"),
        _f("sp_mxf4nvf4_2x_ue8m0", "fp4", f"{SP}.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
           (16, 8, 128), (R, 4), (R, 4), ("f", 4), "e2m1x8", "e2m1x8", sparse="2:4", scale="ue8m0"),
        _f("sp_mxf4_2x", "fp4", f"{SP}.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", (16, 8, 128),
           (R, 4), (R, 4), ("f", 4), "e2m1x8", "e2m1x8", sparse="2:4", scale="ue8m0"),
    ]
    # kind::f8f6f4 and kind::mxf8f6f4 (the QMMA datapath): every same-type pair, E2M1 x E4M3, the FP16 accumulator, 2:4 sparse
    for t in F8:
        f.append(_f(f"f8f6f4_{t}", "f8f6f4", f"{MS}.m16n8k32.row.col.kind::f8f6f4.f32.{t}.{t}.f32", (16, 8, 32), (R, 4), (R, 2),
                    ("f", 4), t, t))
        f.append(_f(f"f8f6f4_{t}_f16acc", "f8f6f4", f"{MS}.m16n8k32.row.col.kind::f8f6f4.f16.{t}.{t}.f16", (16, 8, 32), (R, 4),
                    (R, 2), ("h", 2), t, t))
        f.append(_f(f"sp_f8f6f4_{t}", "f8f6f4", f"{SP}.m16n8k64.row.col.kind::f8f6f4.f32.{t}.{t}.f32", (16, 8, 64), (R, 4), (R, 4),
                    ("f", 4), t, t, sparse="2:4"))
        f.append(_f(f"mxf8f6f4_{t}", "mxf8f6f4",
                    f"{MS}.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.{t}.{t}.f32.ue8m0", (16, 8, 32), (R, 4),
                    (R, 2), ("f", 4), t, t, scale="ue8m0"))
        f.append(_f(f"sp_mxf8f6f4_{t}", "mxf8f6f4",
                    f"{SP}.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.{t}.{t}.f32.ue8m0", (16, 8, 64), (R, 4),
                    (R, 4), ("f", 4), t, t, sparse="2:4", scale="ue8m0"))
    f.append(_f("f8f6f4_e2m1_e4m3", "f8f6f4", f"{MS}.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32", (16, 8, 32), (R, 4), (R, 2),
                ("f", 4), "e2m1", "e4m3"))
    # FP8 without a kind (the Ada forms), both accumulators, m16n8k16 and 2:4 sparse
    for t in ("e4m3", "e5m2"):
        f += [
            _f(f"fp8_{t}_k32", "fp8", f"{MS}.m16n8k32.row.col.f32.{t}.{t}.f32", (16, 8, 32), (R, 4), (R, 2), ("f", 4), t, t),
            _f(f"fp8_{t}_k32_f16acc", "fp8", f"{MS}.m16n8k32.row.col.f16.{t}.{t}.f16", (16, 8, 32), (R, 4), (R, 2), ("h", 2), t, t),
            _f(f"fp8_{t}_k16", "fp8", f"{MS}.m16n8k16.row.col.f32.{t}.{t}.f32", (16, 8, 16), (R, 2), (R, 1), ("f", 4), t, t),
            _f(f"sp_fp8_{t}_k64", "fp8", f"{SP}.m16n8k64.row.col.f32.{t}.{t}.f32", (16, 8, 64), (R, 4), (R, 4), ("f", 4), t, t,
               sparse="2:4"),
            _f(f"sp_fp8_{t}_k64_f16acc", "fp8", f"{SP}.m16n8k64.row.col.f16.{t}.{t}.f16", (16, 8, 64), (R, 4), (R, 4), ("h", 2), t,
               t, sparse="2:4"),
        ]
    # FP16 and BF16 (HMMA): every shape, both accumulators, 2:4 sparse
    f += [
        _f("f16_k16", "f16", f"{MS}.m16n8k16.row.col.f32.f16.f16.f32", (16, 8, 16), (R, 4), (R, 2), ("f", 4), "f16", "f16"),
        _f("f16_k16_f16acc", "f16", f"{MS}.m16n8k16.row.col.f16.f16.f16.f16", (16, 8, 16), (R, 4), (R, 2), ("h", 2), "f16", "f16"),
        _f("f16_k8", "f16", f"{MS}.m16n8k8.row.col.f32.f16.f16.f32", (16, 8, 8), (R, 2), (R, 1), ("f", 4), "f16", "f16"),
        _f("f16_k8_f16acc", "f16", f"{MS}.m16n8k8.row.col.f16.f16.f16.f16", (16, 8, 8), (R, 2), (R, 1), ("h", 2), "f16", "f16"),
        _f("f16_m8n8k4", "f16", f"{MS}.m8n8k4.row.col.f32.f16.f16.f32", (8, 8, 4), (R, 2), (R, 2), ("f", 8), "f16", "f16"),
        _f("f16_m8n8k4_f16acc", "f16", f"{MS}.m8n8k4.row.col.f16.f16.f16.f16", (8, 8, 4), (R, 2), (R, 2), ("h", 4), "f16", "f16"),
        _f("bf16_k16", "f16", f"{MS}.m16n8k16.row.col.f32.bf16.bf16.f32", (16, 8, 16), (R, 4), (R, 2), ("f", 4), "bf16", "bf16"),
        _f("bf16_k8", "f16", f"{MS}.m16n8k8.row.col.f32.bf16.bf16.f32", (16, 8, 8), (R, 2), (R, 1), ("f", 4), "bf16", "bf16"),
        _f("sp_f16_k16", "f16", f"{SP}.m16n8k16.row.col.f32.f16.f16.f32", (16, 8, 16), (R, 2), (R, 2), ("f", 4), "f16", "f16",
           sparse="2:4"),
        _f("sp_f16_k32", "f16", f"{SP}.m16n8k32.row.col.f32.f16.f16.f32", (16, 8, 32), (R, 4), (R, 4), ("f", 4), "f16", "f16",
           sparse="2:4"),
        _f("sp_f16_k32_f16acc", "f16", f"{SP}.m16n8k32.row.col.f16.f16.f16.f16", (16, 8, 32), (R, 4), (R, 4), ("h", 2), "f16",
           "f16", sparse="2:4"),
        _f("sp_bf16_k32", "f16", f"{SP}.m16n8k32.row.col.f32.bf16.bf16.f32", (16, 8, 32), (R, 4), (R, 4), ("f", 4), "bf16", "bf16",
           sparse="2:4"),
    ]
    # TF32 (HMMA.xx.TF32): both shapes, 1:2 sparse
    f += [
        _f("tf32_k8", "tf32", f"{MS}.m16n8k8.row.col.f32.tf32.tf32.f32", (16, 8, 8), (R, 4), (R, 2), ("f", 4), "tf32", "tf32"),
        _f("tf32_k4", "tf32", f"{MS}.m16n8k4.row.col.f32.tf32.tf32.f32", (16, 8, 4), (R, 2), (R, 1), ("f", 4), "tf32", "tf32"),
        _f("sp_tf32_k8", "tf32", f"{SP}.m16n8k8.row.col.f32.tf32.tf32.f32", (16, 8, 8), (R, 2), (R, 2), ("f", 4), "tf32", "tf32",
           sparse="1:2"),
        _f("sp_tf32_k16", "tf32", f"{SP}.m16n8k16.row.col.f32.tf32.tf32.f32", (16, 8, 16), (R, 4), (R, 4), ("f", 4), "tf32", "tf32",
           sparse="1:2"),
    ]
    # integers (IMMA): int8 every shape, u8 x s8, satfinite; int4 every shape; 2:4 sparse
    f += [
        _f("s8_k32", "int", f"{MS}.m16n8k32.row.col.s32.s8.s8.s32", (16, 8, 32), (R, 4), (R, 2), ("r", 4), "b32", "b32"),
        _f("s8_k32_satfinite", "int", f"{MS}.m16n8k32.row.col.satfinite.s32.s8.s8.s32", (16, 8, 32), (R, 4), (R, 2), ("r", 4), "b32",
           "b32"),
        _f("u8s8_k32", "int", f"{MS}.m16n8k32.row.col.s32.u8.s8.s32", (16, 8, 32), (R, 4), (R, 2), ("r", 4), "b32", "b32"),
        _f("s8_k16", "int", f"{MS}.m16n8k16.row.col.s32.s8.s8.s32", (16, 8, 16), (R, 2), (R, 1), ("r", 4), "b32", "b32"),
        _f("s8_m8n8k16", "int", f"{MS}.m8n8k16.row.col.s32.s8.s8.s32", (8, 8, 16), (R, 1), (R, 1), ("r", 2), "b32", "b32"),
        _f("s4_k64", "int", f"{MS}.m16n8k64.row.col.s32.s4.s4.s32", (16, 8, 64), (R, 4), (R, 2), ("r", 4), "b32", "b32", product="s4"),
        _f("u4_k64", "int", f"{MS}.m16n8k64.row.col.s32.u4.u4.s32", (16, 8, 64), (R, 4), (R, 2), ("r", 4), "b32", "b32", product="u4"),
        _f("s4_k32", "int", f"{MS}.m16n8k32.row.col.s32.s4.s4.s32", (16, 8, 32), (R, 2), (R, 1), ("r", 4), "b32", "b32", product="s4"),
        _f("s4_m8n8k32", "int", f"{MS}.m8n8k32.row.col.s32.s4.s4.s32", (8, 8, 32), (R, 1), (R, 1), ("r", 2), "b32", "b32",
           product="s4"),
        _f("sp_s8_k32", "int", f"{SP}.m16n8k32.row.col.s32.s8.s8.s32", (16, 8, 32), (R, 2), (R, 2), ("r", 4), "b32", "b32",
           sparse="2:4"),
        _f("sp_s8_k64", "int", f"{SP}.m16n8k64.row.col.s32.s8.s8.s32", (16, 8, 64), (R, 4), (R, 4), ("r", 4), "b32", "b32",
           sparse="2:4"),
        _f("sp_s4_k64", "int", f"{SP}.m16n8k64.row.col.s32.s4.s4.s32", (16, 8, 64), (R, 2), (R, 2), ("r", 4), "b32", "b32",
           sparse="2:4", product="s4"),
        _f("sp_s4_k128", "int", f"{SP}.m16n8k128.row.col.s32.s4.s4.s32", (16, 8, 128), (R, 4), (R, 4), ("r", 4), "b32", "b32",
           sparse="2:4", product="s4"),
    ]
    # single-bit (BMMA): and.popc and xor.popc, every shape
    for op in ("and", "xor"):
        f += [
            _f(f"b1_{op}_k256", "b1", f"{MS}.m16n8k256.row.col.s32.b1.b1.s32.{op}.popc", (16, 8, 256), (R, 4), (R, 2), ("r", 4), "b32",
               "b32", product="b1"),
            _f(f"b1_{op}_k128", "b1", f"{MS}.m16n8k128.row.col.s32.b1.b1.s32.{op}.popc", (16, 8, 128), (R, 2), (R, 1), ("r", 4), "b32",
               "b32", product="b1"),
            _f(f"b1_{op}_m8n8k128", "b1", f"{MS}.m8n8k128.row.col.s32.b1.b1.s32.{op}.popc", (8, 8, 128), (R, 1), (R, 1), ("r", 2),
               "b32", "b32", product="b1"),
        ]
    # FP64 (DMMA)
    f += [
        _f("f64_m8n8k4", "f64", f"{MS}.m8n8k4.row.col.f64.f64.f64.f64", (8, 8, 4), (D, 1), (D, 1), ("d", 2), "f64", "f64"),
        _f("f64_k4", "f64", f"{MS}.m16n8k4.row.col.f64.f64.f64.f64", (16, 8, 4), (D, 2), (D, 1), ("d", 4), "f64", "f64"),
        _f("f64_k8", "f64", f"{MS}.m16n8k8.row.col.f64.f64.f64.f64", (16, 8, 8), (D, 4), (D, 2), ("d", 4), "f64", "f64"),
        _f("f64_k16", "f64", f"{MS}.m16n8k16.row.col.f64.f64.f64.f64", (16, 8, 16), (D, 8), (D, 4), ("d", 4), "f64", "f64"),
    ]
    return f


FORMS_TENSOR = _forms()

OPS = [
    SimtMultiplier("ffma_f32", "fma.rn.f32", "fma.rn.f32 %0, %0, %1, %2;", "f", "f", "f", "ffma", "FFMA", 1, 1),
    # an immediate addend: two register sources, so no three-operand register-bank conflict (the op's best form)
    SimtMultiplier("ffma_f32_imm", "fma.rn.f32 (immediate addend 2^-4)", "fma.rn.f32 %0, %0, %1, 0f3D800000;", "f", "f", None,
                   "ffma_imm", "FFMA", 1, 1),
    SimtMultiplier("fmul_f32", "mul.rn.f32", "mul.rn.f32 %0, %0, %1;", "f", "f", None, "fmul", "FMUL", 1, 1),
    SimtMultiplier("ffma_f32x2", "fma.rn.f32x2", "fma.rn.f32x2 %0, %0, %1, %2;", "l", "l", "l", "ffma2", "FFMA", 2, 2),
    SimtMultiplier("hfma2_f16", "fma.rn.f16x2", "fma.rn.f16x2 %0, %0, %1, %2;", "r", "r", "r", "hfma2", "HFMA2", 1, 2),
    SimtMultiplier("hfma2_bf16", "fma.rn.bf16x2", "fma.rn.bf16x2 %0, %0, %1, %2;", "r", "r", "r", "bfma2", "HFMA2", 1, 2),
    # ptxas lowers half of these to HFMA2 with a -0 addend (an exact multiply): both count
    SimtMultiplier("hmul2_f16", "mul.rn.f16x2", "mul.rn.f16x2 %0, %0, %1;", "r", "r", None, "hmul2", ("HMUL2", "HFMA2"), 1, 2),
    SimtMultiplier("dfma_f64", "fma.rn.f64", "fma.rn.f64 %0, %0, %1, %2;", "d", "d", "d", "dfma", "DFMA", 1, 1),
    SimtMultiplier("imad_s32", "mad.lo.s32", "mad.lo.s32 %0, %0, %1, %2;", "r", "r", "r", "int", "IMAD", 1, 1),
    SimtMultiplier("imad_s32_imm", "mad.lo.s32 (immediate addend)", "mad.lo.s32 %0, %0, %1, 12345;", "r", "r", None, "int", "IMAD",
                   1, 1),
    SimtMultiplier("imad_hi_u32", "mad.hi.u32", "mad.hi.u32 %0, %0, %1, %2;", "r", "r", "r", "int", "IMAD.HI", 1, 1),
    SimtMultiplier("imad_wide_u32", "mad.wide.u32", "{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, %0;\nmad.wide.u32 %0, lo, %1, %0;\n}",
                   "l", "r", None, "int", "IMAD.WIDE", 1, 1),
    SimtMultiplier("dp4a_s8", "dp4a.s32.s32", "dp4a.s32.s32 %0, %0, %1, %2;", "r", "r", "r", "int", "IDP.4A", 1, 4),
    SimtMultiplier("dp2a_s16s8", "dp2a.lo.s32.s32", "dp2a.lo.s32.s32 %0, %0, %1, %2;", "r", "r", "r", "int", "IDP.2A", 1, 2),
    SimtMultiplier("popc_and_b1", "and.b32 + popc.b32 + add.u32",
                   "{\n.reg .b32 t;\nand.b32 t, %0, %1;\npopc.b32 t, t;\nadd.u32 %0, %0, t;\n}", "r", "r", None, "int", "POPC", 1, 32,
                   product="b1"),
]

FORMS = FORMS_TENSOR + OPS
BY_NAME = {f.name: f for f in FORMS}
FAMILIES = {}
for _x in FORMS:
    FAMILIES.setdefault(_x.family, []).append(_x.name)
assert len(BY_NAME) == len(FORMS), "form names must be unique"


# ---------------------------------------------------------------------------------------------------------------- code generation

def _ctype(c: str) -> str:
    return {"f": "float", "h": "uint32_t", "r": "uint32_t", "d": "double", "l": "unsigned long long"}[c]


def _cons(c: str) -> str:
    return {"f": "f", "h": "r", "r": "r", "d": "d", "l": "l"}[c]


def tensor_asm(form: Form) -> str:
    """The asm statement of one MMA on chain ``c`` and operand set ``s`` (C and D are the same registers: an accumulator chain)."""
    nd, na, nb = form.acc[1], form.a[1], form.b[1]
    ops = [f"%{i}" for i in range(nd + na + nb + (1 if form.sparse else 0) + (2 if form.scale else 0))]
    d, a, b = ops[:nd], ops[nd:nd + na], ops[nd + na:nd + na + nb]
    rest = ops[nd + na + nb:]
    text = f"{form.ptx} {{{', '.join(d)}}}, {{{', '.join(a)}}}, {{{', '.join(b)}}}, {{{', '.join(d)}}}"
    if form.sparse:
        text += f", {rest.pop(0)}, 0x0"
    if form.scale:
        text += f", {{{rest[0]}}}, {{0, 0}}, {{{rest[1]}}}, {{0, 0}}"
    outs = ", ".join(f'"+{_cons(form.acc[0])}"(d[c][{j}])' for j in range(nd))
    ins = [f'"{_cons(form.a[0])}"(a[s][{j}])' for j in range(na)] + [f'"{_cons(form.b[0])}"(b[s][{j}])' for j in range(nb)]
    if form.sparse:
        ins.append('"r"(e[s])')
    if form.scale:
        ins += ['"r"(sfa[s])', '"r"(sfb[s])']
    return f'asm volatile("{text};" : {outs} : {", ".join(ins)});'


def _load(t: tuple[str, int], name: str, base: str) -> tuple[list[str], str]:
    """Statements loading operand array ``name[s][..]`` from words at ``base``; returns (lines, words expression)."""
    lines = []
    for j in range(t[1]):
        if t[0] == "d":
            lines.append(f"{name}[s][{j}] = __hiloint2double((int)f[{base} + {2 * j + 1}], (int)f[{base} + {2 * j}]);")
        else:
            lines.append(f"{name}[s][{j}] = f[{base} + {j}];")
    return lines, f"{base} + {(2 if t[0] == 'd' else 1) * t[1]}"


def tensor_kernel(form: Form) -> str:
    wps = form.words_per_set()
    ta, tb, tacc = _ctype(form.a[0]), _ctype(form.b[0]), _ctype(form.acc[0])
    la, pb = _load(form.a, "a", "o")
    lb, pe = _load(form.b, "b", "ob")
    body = []
    for j in range(NSETS):
        for c in range(NACC):
            body.append(f"        {{ const int c = {c}, s = {(c + j) % NSETS}; {tensor_asm(form)} }}")
    if form.acc[0] == "f":
        fin = "bad += !isfinite(d[c][j]); x ^= __float_as_uint(d[c][j]);"
    elif form.acc[0] == "h":
        fin = "bad += (((d[c][j] >> 10) & 0x1F) == 0x1F) + (((d[c][j] >> 26) & 0x1F) == 0x1F); x ^= d[c][j];"
    elif form.acc[0] == "d":
        fin = ("bad += !isfinite(d[c][j]); { unsigned long long u = __double_as_longlong(d[c][j]); "
               "x ^= (uint32_t)u ^ (uint32_t)(u >> 32); }")
    else:
        fin = "x ^= d[c][j];"
    nl = "\n"
    return f"""
extern "C" __global__ void __launch_bounds__({TENSOR_TPB}) f1_{form.name}(const uint32_t* __restrict__ in, long long iters,
                                                                        uint32_t* __restrict__ sink, long long* __restrict__ cycles) {{
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t* f = in + t * {NSETS * wps};
    {ta} a[{NSETS}][{form.a[1]}];
    {tb} b[{NSETS}][{form.b[1]}];
    uint32_t e[{NSETS}], sfa[{NSETS}], sfb[{NSETS}];
#pragma unroll
    for (int s = 0; s < {NSETS}; s++) {{
        const int o = s * {wps};
        {(nl + "        ").join(la)}
        const int ob = {pb};
        {(nl + "        ").join(lb)}
        const int oe = {pe};
        e[s] = f[oe]; sfa[s] = f[oe + 1]; sfb[s] = f[oe + 2];
    }}
    (void)e; (void)sfa; (void)sfb;
    {tacc} d[{NACC}][{form.acc[1]}];
#pragma unroll
    for (int c = 0; c < {NACC}; c++)
#pragma unroll
        for (int j = 0; j < {form.acc[1]}; j++) d[c][j] = 0;
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 1
    for (long long i = 0; i < iters; i++) {{
{nl.join(body)}
    }}
    __syncthreads();
    const long long t1 = clock64();
    uint32_t x = 0, bad = 0;
#pragma unroll
    for (int c = 0; c < {NACC}; c++)
#pragma unroll
        for (int j = 0; j < {form.acc[1]}; j++) {{ {fin} }}
    sink[2 * t] = x;
    sink[2 * t + 1] = bad;
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}}
"""


def _op_words(c: str | None) -> int:
    return 0 if c is None else (2 if c in ("d", "l") else 1)


def op_words_per_chain(op: SimtMultiplier) -> int:
    return _op_words(op.x) + _op_words(op.y) + _op_words(op.z)


def simt_kernel(op: SimtMultiplier) -> str:
    w = op_words_per_chain(op)

    def load(c, idx):
        if c in ("d",):
            return f"__hiloint2double((int)f[{idx} + 1], (int)f[{idx}])"
        if c == "l":
            return f"((unsigned long long)f[{idx} + 1] << 32 | f[{idx}])"
        if c == "f":
            return f"__uint_as_float(f[{idx}])"
        return f"f[{idx}]"

    ox, oy = 0, _op_words(op.x)
    oz = oy + _op_words(op.y)
    decl = [f"{_ctype(op.x)} x[{SIMT_CH}]; {_ctype(op.y)} y[{SIMT_CH}];"]
    ld = [f"x[c] = {load(op.x, f'c * {w} + {ox}')}; y[c] = {load(op.y, f'c * {w} + {oy}')};"]
    ins = f'"{_cons(op.y)}"(y[c])'
    if op.z:
        decl.append(f"{_ctype(op.z)} z[{SIMT_CH}];")
        ld.append(f"z[c] = {load(op.z, f'c * {w} + {oz}')};")
        ins += f', "{_cons(op.z)}"(z[c])'
    body_text = op.body.replace("\n", "\\n")
    step = f'asm volatile("{body_text}" : "+{_cons(op.x)}"(x[c]) : {ins});'
    if op.x == "f":
        fin = "bad += !isfinite(x[c]); acc ^= __float_as_uint(x[c]);"
    elif op.x == "d":
        fin = "bad += !isfinite(x[c]); { unsigned long long u = __double_as_longlong(x[c]); acc ^= (uint32_t)u ^ (uint32_t)(u >> 32); }"
    elif op.x == "l" and op.fmt == "ffma2":
        fin = ("{ float lo = __uint_as_float((uint32_t)x[c]), hi = __uint_as_float((uint32_t)(x[c] >> 32)); "
               "bad += !isfinite(lo) + !isfinite(hi); acc ^= (uint32_t)x[c] ^ (uint32_t)(x[c] >> 32); }")
    elif op.x == "l":
        fin = "acc ^= (uint32_t)x[c] ^ (uint32_t)(x[c] >> 32);"
    elif op.fmt in ("hfma2", "hmul2"):
        fin = "bad += (((x[c] >> 10) & 0x1F) == 0x1F) + (((x[c] >> 26) & 0x1F) == 0x1F); acc ^= x[c];"
    elif op.fmt == "bfma2":
        fin = "bad += (((x[c] >> 7) & 0xFF) == 0xFF) + (((x[c] >> 23) & 0xFF) == 0xFF); acc ^= x[c];"
    else:
        fin = "acc ^= x[c];"
    return f"""
extern "C" __global__ void __launch_bounds__({SIMT_TPB}) f1_{op.name}(const uint32_t* __restrict__ in, long long iters,
                                                                      uint32_t* __restrict__ sink, long long* __restrict__ cycles) {{
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t* f = in + t * {SIMT_CH * w};
    {' '.join(decl)}
#pragma unroll
    for (int c = 0; c < {SIMT_CH}; c++) {{ {' '.join(ld)} }}
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 1
    for (long long i = 0; i < iters; i++) {{
#pragma unroll
        for (int u = 0; u < {SIMT_U}; u++) {{
#pragma unroll
            for (int c = 0; c < {SIMT_CH}; c++) {step}
        }}
    }}
    __syncthreads();
    const long long t1 = clock64();
    uint32_t acc = 0, bad = 0;
#pragma unroll
    for (int c = 0; c < {SIMT_CH}; c++) {{ {fin} }}
    sink[2 * t] = acc;
    sink[2 * t + 1] = bad;
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}}
"""


PRELUDE = "#include <cuda_runtime.h>\n#include <stdint.h>\n#include <math.h>\n"

API = r"""
typedef void (*F1Fn)(const uint32_t*, long long, uint32_t*, long long*);
static F1Fn F1_FNS[] = {%(fns)s};
static uint32_t* g_in[64];
static uint32_t* g_sink = nullptr;
static long long* g_cyc = nullptr;
static int g_nslots = 0;

extern "C" {
int f1_nkernels(void) { return (int)(sizeof(F1_FNS) / sizeof(F1_FNS[0])); }

int f1_device(int* sms, int* max_threads_per_sm, int* major, int* minor) {
    cudaDeviceProp p;
    cudaError_t err = cudaGetDeviceProperties(&p, 0);
    if (err != cudaSuccess) return (int)err;
    *sms = p.multiProcessorCount; *max_threads_per_sm = p.maxThreadsPerMultiProcessor; *major = p.major; *minor = p.minor;
    return 0;
}

int device_identity(int* major, int* minor, char* name, int name_len) {
    cudaDeviceProp p;
    cudaError_t err = cudaGetDeviceProperties(&p, 0);
    if (err != cudaSuccess) return (int)err;
    *major = p.major; *minor = p.minor;
    int i = 0;
    for (; i < name_len - 1 && p.name[i]; ++i) name[i] = p.name[i];
    name[i] = 0;
    return 0;
}

int device_uuid(char* out, int out_len) {
    cudaDeviceProp p;
    cudaError_t err = cudaGetDeviceProperties(&p, 0);
    if (err != cudaSuccess) return (int)err;
    if (out_len < 41) return (int)cudaErrorInvalidValue;
    static const char* hex = "0123456789abcdef";
    const unsigned char* u = (const unsigned char*)p.uuid.bytes;
    int k = 0;
    out[k++] = 'G'; out[k++] = 'P'; out[k++] = 'U'; out[k++] = '-';
    for (int i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out[k++] = '-';
        out[k++] = hex[u[i] >> 4];
        out[k++] = hex[u[i] & 15];
    }
    out[k] = 0;
    return 0;
}

int has_mxf4(void) { return 1; }

int f1_alloc(int nslots, long long max_threads, int max_blocks) {
    for (int i = 0; i < g_nslots; i++) cudaFree(g_in[i]);
    if (g_sink) cudaFree(g_sink);
    if (g_cyc) cudaFree(g_cyc);
    g_nslots = 0; g_sink = nullptr; g_cyc = nullptr;
    if (nslots > 64) return -1;
    cudaError_t err = cudaMalloc(&g_sink, (size_t)max_threads * 2 * sizeof(uint32_t));
    if (err == cudaSuccess) err = cudaMalloc(&g_cyc, (size_t)max_blocks * sizeof(long long));
    for (int i = 0; i < nslots; i++) g_in[i] = nullptr;
    g_nslots = nslots;
    return (int)err;
}

int f1_upload(int slot, const uint32_t* host, long long words) {
    if (slot < 0 || slot >= g_nslots) return -1;
    if (g_in[slot]) cudaFree(g_in[slot]);
    cudaError_t err = cudaMalloc(&g_in[slot], (size_t)words * 4);
    if (err == cudaSuccess) err = cudaMemcpy(g_in[slot], host, (size_t)words * 4, cudaMemcpyHostToDevice);
    return (int)err;
}

int f1_occupancy(int k, int tpb, int* blocks_per_sm) {
    if (k < 0 || k >= f1_nkernels()) return -1;
    return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, F1_FNS[k], tpb, 0);
}

int f1_run(int k, int slot, int nblocks, int tpb, long long iters, float* ms, uint32_t* sink_out, long long* cyc_out) {
    if (k < 0 || k >= f1_nkernels() || slot < 0 || slot >= g_nslots || !g_in[slot]) return -1;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    F1_FNS[k]<<<nblocks, tpb>>>(g_in[slot], iters, g_sink, g_cyc);
    cudaEventRecord(e1);
    cudaError_t err = cudaEventSynchronize(e1);
    if (err == cudaSuccess) err = cudaGetLastError();
    cudaEventElapsedTime(ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (err == cudaSuccess && sink_out) err = cudaMemcpy(sink_out, g_sink, (size_t)nblocks * tpb * 2 * 4, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess && cyc_out) err = cudaMemcpy(cyc_out, g_cyc, (size_t)nblocks * sizeof(long long), cudaMemcpyDeviceToHost);
    return (int)err;
}

void f1_teardown(void) {
    for (int i = 0; i < g_nslots; i++) if (g_in[i]) cudaFree(g_in[i]);
    if (g_sink) cudaFree(g_sink);
    if (g_cyc) cudaFree(g_cyc);
    g_nslots = 0; g_sink = nullptr; g_cyc = nullptr;
}
}  // extern "C"
"""


def kernel_source(form) -> str:
    return tensor_kernel(form) if form.tensor else simt_kernel(form)


def library_source(names: list[str]) -> str:
    return PRELUDE + "".join(kernel_source(BY_NAME[n]) for n in names) + API % {"fns": ", ".join(f"f1_{n}" for n in names)}


# ------------------------------------------------------------------------------------------------------------------- build + SASS

def _nvcc() -> str:
    return P.nvcc_bin()


def _cuobjdump() -> str:
    return str(Path(_nvcc()).parent / "cuobjdump")


def _flags_ok(flags: list[str]) -> bool:
    return not any(re.search(r"ftz|fast-math|fast_math", x, re.I) for x in flags)


def accept_one(name: str, work: Path) -> dict:
    """ptxas acceptance of one form alone, and its timed loop's opcode histogram."""
    src, cubin = work / f"{name}.cu", work / f"{name}.cubin"
    src.write_text(PRELUDE + kernel_source(BY_NAME[name]))
    cmd = [_nvcc(), *NVCC_FLAGS, *P.arch_flags(ARCH), "-cubin", "-o", str(cubin), str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"accepted": r.returncode == 0}
    if r.returncode != 0:
        msgs = [ln.strip() for ln in (r.stderr + r.stdout).splitlines() if ln.strip()]
        rec["ptxas"] = [m for m in msgs if "error" in m.lower()][:4] or msgs[:4]
        return rec
    sass = subprocess.run([_cuobjdump(), "-sass", str(cubin)], capture_output=True, text=True, check=True).stdout
    rec["loop_opcodes"] = loops_of(sass).get(f"f1_{name}", {})
    rec["ftz_fp32"] = sorted(set(m.group(0) for m in FTZ_RE.finditer(sass)))
    return rec


def loops_of(text: str) -> dict:
    """{kernel: opcode histogram of its innermost timed loop} (w1_price.py's reader)."""
    import w1_price as W
    return W.loops_of(text)


def calls_of(text: str) -> dict:
    """{kernel: [opcode histogram of each top-level ``CALL.REL.NOINC``'s callee, from its target to the first ``RET``, with nested
    calls expanded in place]}: the subroutines ptxas emulates an MMA with (s4 / u4 / b1 and FP16 m8n8k4 on sm_120a).  A call
    inside a callee is nested, not top-level; an unresolvable or too-deep callee holds the marker opcode ``?CALL``."""
    out: dict = {}
    fn, ins = None, []

    def flush():
        if fn is None:
            return
        at = {a: i for i, (a, _) in enumerate(ins)}

        def target(toks):
            m = re.search(r"0x([0-9a-f]+)", " ".join(toks))
            return int(m.group(1), 16) if m and int(m.group(1), 16) in at else None

        def expand(t, depth) -> tuple[dict, set]:
            hist: dict = {}
            body: set = set()
            if t is None or depth > 4:
                return {"?CALL": 1}, body
            for a, toks in ins[at[t]:]:
                body.add(a)
                if _op(toks).startswith("RET"):
                    break
                hist[_op(toks)] = hist.get(_op(toks), 0) + 1
                if _op(toks) == "CALL.REL.NOINC":
                    h, b2 = expand(target(toks), depth + 1)
                    body |= b2
                    for k, v in h.items():
                        hist[k] = hist.get(k, 0) + v
            return hist, body

        sites = [(a, toks) for a, toks in ins if _op(toks) == "CALL.REL.NOINC"]
        inside: set = set()
        for _, toks in sites:
            inside |= expand(target(toks), 0)[1]
        out[fn] = [expand(target(toks), 0)[0] for a, toks in sites if a not in inside]

    for line in text.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            flush()
            fn, ins = m.group(1), []
            continue
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?);", line)
        if fn and m and m.group(2).split():
            ins.append((int(m.group(1), 16), m.group(2).split()))
    flush()
    return out


def _op(toks: list[str]) -> str:
    return toks[1] if toks[0].startswith("@") and len(toks) > 1 else toks[0]


def form_gate(form, hist: dict, calls: list | None = None) -> dict:
    """What the timed loop holds.  Tensor forms: the MMA-family instructions, a nonzero multiple of BODY (one PTX MMA may lower
    to several: ``sass_per_ptx`` > 1, recorded), or, where ptxas emulates the form, exactly BODY calls in the loop (and none
    elsewhere) to callees holding MMA instructions, whose MMAs are counted per PTX instruction (``emulated``); SIMT forms: exactly
    the expected count of the op's opcodes."""
    if form.tensor:
        mma = {k: v for k, v in hist.items() if MMA_RE.search(k)}
        n = sum(mma.values())
        other = {k: v for k, v in hist.items() if k not in mma}
        ncall = hist.get("CALL.REL.NOINC", 0)
        if not mma and ncall:
            calls = calls or []
            callee: dict = {}
            mult: dict = {}
            for h in calls:
                for k, v in h.items():
                    if MMA_RE.search(k):
                        callee[k] = callee.get(k, 0) + v
                    elif MULT_RE.search(k):
                        mult[k] = mult.get(k, 0) + v
            m = sum(callee.values())
            resolved = not any("?CALL" in h for h in calls)
            ok = ncall == BODY and len(calls) == BODY and resolved and ((m > 0 and m % BODY == 0) or (m == 0 and bool(mult)))
            return {"ok": ok, "emulated": "tensor" if m else "cuda-core", "mma_opcodes": {}, "callee_mma_opcodes": callee,
                    "callee_multiply_opcodes": mult, "calls_in_loop": ncall, "top_level_calls": len(calls),
                    "sass_per_ptx": m / BODY, "native": False, "other_opcodes": other, "callee_opcodes": calls[0] if calls else {}}
        ok = n >= BODY and n % BODY == 0
        return {"ok": ok, "emulated": False, "mma_opcodes": mma, "sass_per_ptx": n / BODY if BODY else None,
                "native": ok and n == BODY and len(mma) == 1, "other_opcodes": other}
    want = SIMT_CH * SIMT_U * form.per_step
    got = sum(v for k, v in hist.items() if k.startswith(form.opcode))
    return {"ok": got == want, "expected_opcode": form.opcode, "expected": want, "got": got, "loop_opcodes": hist}


def build(out: Path, names: list[str] | None = None, jobs: int = 8) -> dict:
    """Acceptance of every form, then the library of the accepted ones, its SASS gate and its FTZ scan; writes build.json."""
    if not _flags_ok(NVCC_FLAGS):
        raise SystemExit("refused: an -ftz / fast-math flag in NVCC_FLAGS")
    out.mkdir(parents=True, exist_ok=True)
    work = out / "accept"
    work.mkdir(exist_ok=True)
    names = names or [f.name for f in FORMS]
    with cf.ThreadPoolExecutor(jobs) as ex:
        accept = dict(zip(names, ex.map(lambda n: accept_one(n, work), names)))
    ok_names = [n for n in names if accept[n]["accepted"]]
    src = out / "f1_rates_gen.cu"
    src.write_text(library_source(ok_names))
    so = out / f"libf1_rates_{ARCH}.so"
    cmd = [_nvcc(), *NVCC_FLAGS, *P.arch_flags(ARCH), "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(src)]
    subprocess.run(cmd, check=True)
    nvcc = subprocess.run([_nvcc(), "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-2:]
    sass = subprocess.run([_cuobjdump(), "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    loops, calls = loops_of(sass), calls_of(sass)
    gates = {n: form_gate(BY_NAME[n], loops.get(f"f1_{n}", {}), calls.get(f"f1_{n}")) for n in ok_names}
    ftz = sorted(set(m.group(0) for m in FTZ_RE.finditer(sass)))
    rec = {"schema": SCHEMA + "/build", "arch": ARCH, "nvcc": " / ".join(x.strip() for x in nvcc), "nvcc_flags": NVCC_FLAGS,
           "kernels": ok_names, "so": so.name, "so_sha256": hashlib.sha256(so.read_bytes()).hexdigest(),
           "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "accept": accept, "gate": gates, "ftz_fp32": ftz,
           "ok": (not ftz) and all(g["ok"] for g in gates.values()),
           "refused": {n: accept[n].get("ptxas") for n in names if not accept[n]["accepted"]},
           "emulated": sorted(n for n, g in gates.items() if g.get("emulated")),
           "sass_opcodes_reached": sorted({k for g in gates.values() for k in (g.get("mma_opcodes") or {})}
                                          | {k for g in gates.values() for k in (g.get("callee_mma_opcodes") or {})}
                                          | {k for n in ok_names if not BY_NAME[n].tensor for k in gates[n]["loop_opcodes"]})}
    (out / "build.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


# -------------------------------------------------------------------------------------------------------------------- operands

_E4M3_FINITE = np.array([c for c in range(256) if c & 0x7F != 0x7F], np.uint32)


#: FP6 / FP4 codes (exponent bits, mantissa bits) and, for the FP16-accumulate forms, how many of the smallest magnitude codes
#: they draw (|x| <= 0.4375, 0.625, 0.5). A K=32 sum then stays within 16, at most half an FP16 ulp once |D| >= 2^15, so a
#: chain of millions of MMAs on fixed operands stalls short of inf
_NARROW = {"e3m2": (3, 2, 8), "e2m3": (2, 3, 6), "e2m1": (2, 1, 2)}
#: where kind::f8f6f4 / kind::mxf8f6f4 read a code in its 8-bit container (PTX ISA: E2M1 in the central 4 bits, E3M2 and E2M3
#: in the low 6)
_CONTAINER_SHIFT = {"e3m2": 0, "e2m3": 0, "e2m1": 2}


def _small_fp8(rng, fmt: str, shape, small: bool = False) -> np.ndarray:
    """Bytes of an FP8 value (small magnitudes) or an FP6/FP4 code (uniform, or small exponents with ``small``) in its 8-bit
    container (``_CONTAINER_SHIFT``)."""
    s = rng.integers(0, 2, size=shape, dtype=np.uint32) << 7
    if fmt == "e4m3":  # exponent field 1..5: |x| < 0.5
        return s | (rng.integers(1, 6, size=shape, dtype=np.uint32) << 3) | rng.integers(0, 8, size=shape, dtype=np.uint32)
    if fmt == "e5m2":  # exponent field 9..13
        return s | (rng.integers(9, 14, size=shape, dtype=np.uint32) << 2) | rng.integers(0, 4, size=shape, dtype=np.uint32)
    ebits, mbits, nsmall = _NARROW[fmt]
    if not small:
        return rng.integers(0, 1 << (1 + ebits + mbits), size=shape, dtype=np.uint32) << _CONTAINER_SHIFT[fmt]
    code = ((s >> 7) << (ebits + mbits)) | rng.integers(0, nsmall, size=shape, dtype=np.uint32)
    return code << _CONTAINER_SHIFT[fmt]


def _pack4(b: np.ndarray) -> np.ndarray:
    return (b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)).astype(np.uint32)


def _half_bits(rng, shape, bf16: bool, lo: int, hi: int) -> np.ndarray:
    s = rng.integers(0, 2, size=shape, dtype=np.uint32)
    if bf16:
        return (s << 15) | (rng.integers(lo, hi, size=shape, dtype=np.uint32) << 7) | rng.integers(0, 128, size=shape, dtype=np.uint32)
    return (s << 15) | (rng.integers(lo, hi, size=shape, dtype=np.uint32) << 10) | rng.integers(0, 1024, size=shape, dtype=np.uint32)


def operand_words(fmt: str, rng, shape, small: bool = False) -> np.ndarray:
    """u32 words of one operand register (or a double's two words interleaved lo, hi) in a form's input format."""
    if fmt == "e2m1x8" or fmt == "b32":
        return rng.integers(0, 2**32, size=shape, dtype=np.uint64).astype(np.uint32)
    if fmt in F8:
        return _pack4(_small_fp8(rng, fmt, shape + (4,), small))
    if fmt in ("f16", "bf16"):
        lo, hi = (10, 15) if fmt == "f16" else (122, 127)
        h = _half_bits(rng, shape + (2,), fmt == "bf16", lo, hi)
        return (h[..., 0] | (h[..., 1] << 16)).astype(np.uint32)
    if fmt == "tf32":
        v = rng.uniform(1, 2, size=shape) * np.exp2(rng.integers(-5, 0, size=shape)) * rng.choice([-1.0, 1.0], size=shape)
        return v.astype(np.float32).view(np.uint32)
    if fmt == "f64":  # float64 values: the caller views them as (lo, hi) word pairs
        return rng.uniform(1, 2, size=shape) * np.exp2(rng.integers(-5, 0, size=shape)) * rng.choice([-1.0, 1.0], size=shape)
    raise ValueError(fmt)


def _seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def _meta(rng, form: Form, shape) -> np.ndarray:
    nibs = np.array(META_12 if form.sparse == "1:2" else META_24, np.uint32)
    if form.meta == "const":
        return np.full(shape, int(nibs[0]) * 0x11111111, np.uint32)
    n = nibs[rng.integers(0, len(nibs), size=shape + (8,))]
    return (n << (4 * np.arange(8, dtype=np.uint32))).sum(axis=-1, dtype=np.uint64).astype(np.uint32)


def _scale(rng, kind: str | None, shape) -> np.ndarray:
    if kind is None:
        return np.zeros(shape, np.uint32)
    lo, hi = (0x30, 0x48) if kind == "ue4m3" else (124, 132)
    return _pack4(rng.integers(lo, hi, size=shape + (4,), dtype=np.uint32))


def _reg_words(t: tuple[str, int], fmt: str, rng, lanes: int, small: bool = False) -> np.ndarray:
    if t[0] == "d":
        v = operand_words("f64", rng, (lanes, NSETS, t[1]))
        return v.view(np.uint32).reshape(lanes, NSETS, 2 * t[1])
    return operand_words(fmt, rng, (lanes, NSETS, t[1]), small)


def tensor_inputs(form: Form, threads: int, seed: int) -> np.ndarray:
    """u32[threads][NSETS][words_per_set]: a, b, metadata, sfa, sfb per operand set per thread."""
    rng = np.random.default_rng([seed, _seed(form.name)])
    small = form.acc[0] == "h"
    a = _reg_words(form.a, form.fa, rng, threads, small)
    b = _reg_words(form.b, form.fb, rng, threads, small)
    e = _meta(rng, form, (threads, NSETS, 1)) if form.sparse else np.zeros((threads, NSETS, 1), np.uint32)
    sf = _scale(rng, form.scale, (threads, NSETS, 2))
    out = np.concatenate([a, b, e, sf], axis=-1).astype(np.uint32)
    assert out.shape[-1] == form.words_per_set()
    return out


def _f32w(v) -> np.ndarray:
    return np.asarray(v, np.float32).view(np.uint32)


def _f64w(v) -> np.ndarray:
    return np.ascontiguousarray(v, np.float64).view(np.uint32)


def simt_inputs(op: SimtMultiplier, threads: int, seed: int) -> np.ndarray:
    """u32[threads][SIMT_CH][words]: (x, y[, z]) per chain, in the op's domain: FMA chains x <- x*y + z with y in [0.5, 0.99) and z
    small (x converges, finite), multiply chains with y = +-1, integer chains uniform random words."""
    rng = np.random.default_rng([seed, 7, _seed(op.name)])
    sh = (threads, SIMT_CH)
    u = lambda: rng.integers(0, 2**32, size=sh, dtype=np.uint64).astype(np.uint32)  # noqa: E731
    pm = lambda: rng.choice([-1.0, 1.0], size=sh)  # noqa: E731
    if op.fmt == "ffma":
        cols = [_f32w(rng.uniform(1, 2, sh)), _f32w(rng.uniform(0.5, 0.99, sh)), _f32w(rng.uniform(2**-4, 2**-3, sh))]
    elif op.fmt == "ffma_imm":
        cols = [_f32w(rng.uniform(1, 2, sh)), _f32w(rng.uniform(0.5, 0.99, sh))]
    elif op.fmt == "fmul":
        cols = [_f32w(rng.uniform(1, 2, sh) * pm()), _f32w(pm())]
    elif op.fmt == "ffma2":
        two = lambda lo, hi: np.stack([_f32w(rng.uniform(lo, hi, sh)), _f32w(rng.uniform(lo, hi, sh))], -1)  # noqa: E731
        return np.concatenate([two(1, 2), two(0.5, 0.99), two(2**-4, 2**-3)], -1).astype(np.uint32)
    elif op.fmt in ("hfma2", "bfma2", "hmul2"):
        def pack(lo, hi, sign=False):
            v = rng.uniform(lo, hi, sh + (2,)) * (rng.choice([-1.0, 1.0], sh + (2,)) if sign else 1.0)
            if op.fmt == "bfma2":
                bits = (v.astype(np.float32).view(np.uint32) >> 16).astype(np.uint32)
            else:
                bits = v.astype(np.float16).view(np.uint16).astype(np.uint32)
            return (bits[..., 0] | (bits[..., 1] << 16)).astype(np.uint32)
        if op.fmt == "hmul2":
            cols = [pack(1, 2, True), pack(1, 1, True)]
        else:
            cols = [pack(1, 2), pack(0.5, 0.99), pack(2**-4, 2**-3)]
    elif op.fmt == "dfma":
        return np.concatenate([_f64w(rng.uniform(1, 2, sh + (1,))), _f64w(rng.uniform(0.5, 0.99, sh + (1,))),
                               _f64w(rng.uniform(2**-4, 2**-3, sh + (1,)))], -1).astype(np.uint32)
    else:  # integers
        cols = [u(), u()] + ([u()] if op.z else [])
        if op.x == "l":
            cols = [u(), u()] + cols[1:]
    out = np.stack(cols, -1).astype(np.uint32)
    assert out.shape[-1] == op_words_per_chain(op), (op.name, out.shape)
    return out


def inputs(form, threads: int, seed: int) -> np.ndarray:
    return tensor_inputs(form, threads, seed) if form.tensor else simt_inputs(form, threads, seed)


# ------------------------------------------------------------------------------------------------------------------------ device

class F1Device:
    def __init__(self, so: Path, kernels: list[str]):
        self.lib = lib = ctypes.CDLL(str(so))
        V, I, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        lib.f1_device.argtypes = [ctypes.POINTER(I)] * 4
        lib.f1_alloc.argtypes = [I, L, I]
        lib.f1_upload.argtypes = [I, V, L]
        lib.f1_occupancy.argtypes = [I, I, ctypes.POINTER(I)]
        lib.f1_run.argtypes = [I, I, I, I, L, ctypes.POINTER(ctypes.c_float), V, V]
        lib.device_uuid.restype = I
        lib.device_uuid.argtypes = [ctypes.c_char_p, I]
        lib.has_mxf4.restype = I
        lib.device_identity.restype = I
        lib.device_identity.argtypes = [ctypes.POINTER(I), ctypes.POINTER(I), ctypes.c_char_p, I]
        if lib.f1_nkernels() != len(kernels):
            raise RuntimeError(f"{so.name} has {lib.f1_nkernels()} kernels, build.json lists {len(kernels)}")
        self.index = {n: i for i, n in enumerate(kernels)}
        self.tcgen05 = False
        self._keep: dict = {}

    identity = P.Probe.identity
    clocks = P.Probe.clocks

    def device(self) -> dict:
        v = [ctypes.c_int() for _ in range(4)]
        rc = self.lib.f1_device(*v)
        if rc:
            raise RuntimeError(f"f1_device rc={rc}")
        return {"sms": v[0].value, "max_threads_per_sm": v[1].value, "cc": f"{v[2].value}.{v[3].value}"}

    def alloc(self, nslots: int, max_threads: int, max_blocks: int) -> None:
        rc = self.lib.f1_alloc(nslots, max_threads, max_blocks)
        if rc:
            raise RuntimeError(f"f1_alloc rc={rc}")

    def upload(self, slot: int, words: np.ndarray) -> None:
        w = np.ascontiguousarray(words, np.uint32).reshape(-1)
        self._keep[slot] = w
        rc = self.lib.f1_upload(slot, w.ctypes.data, w.size)
        if rc:
            raise RuntimeError(f"f1_upload rc={rc}")

    def occupancy(self, name: str, tpb: int) -> int:
        n = ctypes.c_int()
        rc = self.lib.f1_occupancy(self.index[name], tpb, ctypes.byref(n))
        if rc:
            raise RuntimeError(f"f1_occupancy({name}) rc={rc}")
        return n.value

    def run(self, name: str, slot: int, nblocks: int, tpb: int, iters: int, want: bool = False):
        ms = ctypes.c_float()
        sink = np.zeros(nblocks * tpb * 2, np.uint32) if want else None
        cyc = np.zeros(nblocks, np.int64)
        rc = self.lib.f1_run(self.index[name], slot, nblocks, tpb, iters, ctypes.byref(ms), sink.ctypes.data if want else None,
                             cyc.ctypes.data)
        if rc:
            raise RuntimeError(f"f1_run({name}) rc={rc}")
        return ms.value, sink, cyc

    def teardown(self) -> None:
        self.lib.f1_teardown()


class FakeF1:
    """CPU stand-in for the driver's dry run: invented rates (NVFP4 at 16 MMAs per SM per 128 clocks, everything else slower per
    product).  Its numbers are made up; ``--fake-silicon`` never runs with ``RESEARCH_RUN_ID`` set."""

    SMS, MHZ = 188, 2000.0

    def __init__(self, kernels: list[str]):
        self.index = {n: i for i, n in enumerate(kernels)}

    def identity(self) -> dict:
        return {"device": "FAKE SILICON", "compute_capability": "12.0", "driver": None, "gpu_uuid": "GPU-fake", "backend": "fake",
                "cuda_visible_devices": SHARED.CUDA_VISIBLE_DEVICES}

    def clocks(self, uuid):
        return None

    def device(self) -> dict:
        return {"sms": self.SMS, "max_threads_per_sm": 1536, "cc": "12.0"}

    def alloc(self, nslots, max_threads, max_blocks) -> None:
        pass

    def upload(self, slot, words) -> None:
        pass

    def occupancy(self, name, tpb) -> int:
        return 4 if BY_NAME[name].tensor else 6

    def _products_per_sm_clk(self, name: str) -> float:
        f = BY_NAME[name]
        if not f.tensor:
            return 128.0 * f.products / f.per_step / 2
        base = 8192 / 4  # dense NVFP4: 2,048 products per SM per clock
        return base * (1.0 if f.family == "fp4" else 0.5 if f.family in ("f8f6f4", "mxf8f6f4", "fp8", "int") else 0.25)

    def run(self, name, slot, nblocks, tpb, iters, want=False):
        f = BY_NAME[name]
        per_launch = iters * (BODY * nblocks * tpb // 32 * f.products if f.tensor else SIMT_CH * SIMT_U * nblocks * tpb * f.products)
        cycles = per_launch / self.SMS / self._products_per_sm_clk(name)
        ms = cycles / (self.MHZ * 1e3)
        sink = np.full(nblocks * tpb * 2, self.index[name], np.uint32) if want else None
        if sink is not None:
            sink[1::2] = 0
        return ms, sink, (np.linspace(0.6, 1.0, nblocks) * cycles).astype(np.int64)

    def teardown(self) -> None:
        pass


# ------------------------------------------------------------------------------------------------------------------------- run

def grid(form, d: dict, occ: int) -> dict:
    if form.tensor:
        bps = min(4, occ)
        tpb = TENSOR_TPB
    else:
        bps = min(d["max_threads_per_sm"] // SIMT_TPB, occ)
        tpb = SIMT_TPB
    return {"threads_per_block": tpb, "blocks_per_sm": bps, "blocks": d["sms"] * bps, "warps_per_sm": bps * tpb // 32}


def work_per_launch(form, g: dict, iters: int) -> tuple[float, float, float]:
    """(instructions, nonzero products, logical MACs) of one launch: warp instructions for tensor forms, thread ops for SIMT."""
    if form.tensor:
        inst = float(iters) * BODY * g["blocks"] * g["threads_per_block"] / 32
        return inst, inst * form.products, inst * form.logical_macs
    inst = float(iters) * SIMT_CH * SIMT_U * g["blocks"] * g["threads_per_block"]
    return inst, inst * form.products, inst * form.products


def clock_label(samples: list, launch_mhz: list[float], locked_mhz: float | None) -> str:
    import w1_price as W
    return W.clock_label(samples, launch_mhz, locked_mhz)


def run_family(args) -> int:
    t_all = time.time()
    bdir = Path(os.path.expandvars(args.build))
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    b = json.loads((bdir / "build.json").read_text())
    fam = [n for n in FAMILIES[args.family] if n in b["kernels"]] if args.family != "_all" else list(b["kernels"])
    names = ([REF] if REF not in fam else []) + fam
    rec: dict = {"schema": SCHEMA, "family": args.family, "forms": fam, "build": {k: b[k] for k in ("so_sha256", "source_sha256", "nvcc",
                 "nvcc_flags", "ok", "ftz_fp32")}, "refused_in_family": {n: b["refused"].get(n) for n in FAMILIES.get(args.family, [])
                                                                       if n in b["refused"]},
                 "seed": args.seed, "fake_silicon": args.fake_silicon, "corrupt_word": args.corrupt_word}
    failures: list[str] = []
    if REF not in b["kernels"]:
        failures.append(f"the unit {REF} is not in the build")
    if not b["ok"]:
        failures.append("build SASS gate failed: " + ", ".join([n for n, g in b["gate"].items() if not g["ok"]] + b["ftz_fp32"]))
    dev = FakeF1(b["kernels"]) if args.fake_silicon else F1Device(bdir / b["so"], b["kernels"])
    identity = dev.identity()
    rec["identity"] = identity
    if args.expect_uuid and not args.fake_silicon and not P.uuid_allowed(identity.get("gpu_uuid"), args.expect_uuid):
        failures.append(f"device UUID {identity.get('gpu_uuid')} != expected {args.expect_uuid}: stopped before any measurement")
    if failures:
        return finish(out, rec, failures, args, None, t_all)
    d = dev.device()
    rec["device"] = d
    if d["cc"] != "12.0":
        return finish(out, rec, [f"compute capability {d['cc']} != 12.0"], args, None, t_all)
    geo = {n: grid(BY_NAME[n], d, dev.occupancy(n, TENSOR_TPB if BY_NAME[n].tensor else SIMT_TPB)) for n in names}
    rec["grid"] = geo
    max_threads = max(g["blocks"] * g["threads_per_block"] for g in geo.values())
    dev.alloc(len(names), max_threads, max(g["blocks"] for g in geo.values()))
    for i, n in enumerate(names):
        words = inputs(BY_NAME[n], geo[n]["blocks"] * geo[n]["threads_per_block"], args.seed)
        dev.upload(i, words)
    slot = {n: i for i, n in enumerate(names)}
    gate = {}
    for n in names:
        g = geo[n]
        _, s1, _ = dev.run(n, slot[n], g["blocks"], g["threads_per_block"], 64, want=True)
        _, s2, _ = dev.run(n, slot[n], g["blocks"], g["threads_per_block"], 64, want=True)
        if args.corrupt_word and not gate:
            s2 = s2.copy()
            s2[0] ^= 1
        finite = int(s1[1::2].sum()) == 0 and int(s2[1::2].sum()) == 0
        same = bool(np.array_equal(s1, s2))
        gate[n] = {"finite": finite, "bit_identical": same, "ok": finite and same, "warps_per_sm": g["warps_per_sm"]}
    rec["output_gate"] = gate
    failures += [f"output gate {n}: finite={v['finite']} bit_identical={v['bit_identical']}" for n, v in gate.items() if not v["ok"]]
    if failures:
        dev.teardown()
        return finish(out, rec, failures, args, None, t_all)
    iters = {}
    for n in names:
        g = geo[n]
        ms, _, _ = dev.run(n, slot[n], g["blocks"], g["threads_per_block"], 1000)
        iters[n] = max(1000, int(1000 * args.target_ms / max(ms, 1e-3)))
    t0 = time.time()
    while time.time() - t0 < args.sustain_s:
        dev.run(REF, slot[REF], geo[REF]["blocks"], geo[REF]["threads_per_block"], iters[REF])
    launches: dict = {n: [] for n in names}
    samples = []
    for rep in range(args.reps):
        order = names[rep % len(names):] + names[:rep % len(names)]
        for n in order:
            g = geo[n]
            ms, _, cyc = dev.run(n, slot[n], g["blocks"], g["threads_per_block"], iters[n])
            launches[n].append({"rep": rep, "ms": ms, "cycles_max": int(cyc.max()), "sm_mhz": float(cyc.max()) / (ms * 1e3)})
        samples.append(dev.clocks(identity.get("gpu_uuid")))
    post = {}
    for n in names:
        g = geo[n]
        _, s, _ = dev.run(n, slot[n], g["blocks"], g["threads_per_block"], iters[n], want=True)
        post[n] = {"iterations": iters[n], "finite": int(s[1::2].sum()) == 0}
    dev.teardown()
    rec["timed_length_gate"] = post
    failures += [f"timed-length gate {n}: outputs not finite after {v['iterations']} iterations" for n, v in post.items() if not v["finite"]]
    if failures:
        return finish(out, rec, failures, args, None, t_all)
    per = {}
    for n in names:
        f, g, ls = BY_NAME[n], geo[n], launches[n]
        ms = statistics.median(x["ms"] for x in ls)
        mhz = statistics.median(x["sm_mhz"] for x in ls)
        inst, prod, macs = work_per_launch(f, g, iters[n])
        denom = d["sms"] * mhz * 1e6 * ms / 1e3
        per[n] = {"ptx": f.ptx, "tensor": f.tensor, "iterations": iters[n], "ms_median": ms, "ms_all": [x["ms"] for x in ls],
                  "sm_mhz_median": mhz, "sm_mhz_all": [x["sm_mhz"] for x in ls], "warps_per_sm": g["warps_per_sm"],
                  "instructions_per_sm_per_clock": inst / denom, "products_per_sm_per_clock": prod / denom,
                  "logical_macs_per_sm_per_clock": macs / denom, "products_per_instruction": prod / inst,
                  "sparse": getattr(f, "sparse", None), "product": f.product,
                  "emulated": b["gate"].get(n, {}).get("emulated") or None,
                  "sass": (b["gate"].get(n, {}).get("mma_opcodes") or b["gate"].get(n, {}).get("callee_mma_opcodes")
                           or b["gate"].get(n, {}).get("loop_opcodes"))}
    ref = per[REF]["products_per_sm_per_clock"]
    for n, v in per.items():
        v["fp4_slots_per_product"] = ref / v["products_per_sm_per_clock"]
        v["fp4_slots_per_logical_mac"] = ref / v["logical_macs_per_sm_per_clock"]
    rec.update(per_form=per, launches=launches, clock_samples=samples,
               clocks=clock_label(samples, [x["sm_mhz"] for ls in launches.values() for x in ls], args.locked_mhz),
               iterations=iters)
    return finish(out, rec, failures, args, per, t_all)


def finish(out: Path, rec: dict, failures: list[str], args, per: dict | None, t_all: float) -> int:
    rec["failures"] = failures
    rec["elapsed_s"] = time.time() - t_all
    name = f"{args.family}.json" if not args.fake_silicon else f"{args.family}.fake.json"
    tmp = out / (name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    tmp.replace(out / name)
    if failures:
        print("FAILURES", failures, flush=True)
        return 2 if any("UUID" in f for f in failures) else 1
    print(f"F1 {args.family} RECORDED ({rec.get('clocks')})", flush=True)
    for n, v in (per or {}).items():
        print(f"  {n:28s} {v['products_per_sm_per_clock']:9.1f} products/SM/clk  {v['fp4_slots_per_product']:8.3f} FP4 slots/product"
              f"  {v['sm_mhz_median']:7.1f} MHz  {v['warps_per_sm']} warps/SM", flush=True)
    return 0


# -------------------------------------------------------------------------------------------------------------------- scan-libs

#: node 2's CUDA 13.0 libraries, and DCGM's newer cuBLASLt and its cuSPARSELt (NVIDIA's sparse FP4 / FP8 GEMMs)
DEFAULT_LIBS = ("/usr/local/cuda/lib64/libcublasLt.so", "/usr/local/cuda/lib64/libcublas.so", "/usr/local/cuda/lib64/libcusparse.so",
                "/usr/local/cuda/lib64/libcufft.so", "/usr/local/cuda/lib64/libcusolver.so",
                "/usr/libexec/datacenter-gpu-manager-4/cuda13/lib/libcublasLt.so.13.4.0.1",
                "/usr/libexec/datacenter-gpu-manager-4/cuda13/lib/libcusparseLt.so.0.9.1.1")
OPCODE_LINE = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)")


def scan_lib(path: str, arch: str = "sm_120", timeout_s: float = 1200.0) -> dict:
    """Every opcode in ``path``'s ``arch`` SASS, counted (full opcode with modifiers), streamed from cuobjdump."""
    t0 = time.time()
    real = str(Path(path).resolve())
    rec = {"path": path, "resolved": real, "arch": arch}
    if not Path(real).is_file():
        rec["error"] = "missing"
        return rec
    rec["sha256"] = hashlib.sha256(Path(real).read_bytes()).hexdigest()
    proc = subprocess.Popen([_cuobjdump(), "-arch", arch, "-sass", real], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                            bufsize=1 << 20)
    counts: dict = {}
    functions = 0
    complete = True
    for line in proc.stdout:
        if "Function :" in line:
            functions += 1
            continue
        m = OPCODE_LINE.search(line)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
        if time.time() - t0 > timeout_s:
            complete = False
            proc.kill()
            break
    proc.wait()
    rec.update(functions=functions, complete=complete and proc.returncode == 0, returncode=proc.returncode, seconds=time.time() - t0,
               instructions=sum(counts.values()), opcodes=len(counts),
               multiply_opcodes=dict(sorted(((k, v) for k, v in counts.items() if MULT_RE.search(k)), key=lambda kv: -kv[1])))
    return rec


def scan_libs(args) -> int:
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    libs = [x for x in (args.libs.split(",") if args.libs else DEFAULT_LIBS) if x]
    with cf.ThreadPoolExecutor(min(len(libs), args.jobs)) as ex:
        recs = list(ex.map(lambda p: scan_lib(p, timeout_s=args.timeout_s), libs))
    seen: dict = {}
    for r in recs:
        for k, v in r.get("multiply_opcodes", {}).items():
            seen[k] = seen.get(k, 0) + v
    doc = {"schema": SCHEMA + "/libscan", "cuobjdump": _cuobjdump(), "libraries": recs, "multiply_opcodes_all": seen}
    (out / "libscan.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("LIBSCAN", [(r["path"], r.get("functions"), r.get("complete"), len(r.get("multiply_opcodes", {}))) for r in recs], flush=True)
    ok = all(r.get("complete") for r in recs if "error" not in r)
    if SHARED.RESEARCH_RUN_ID:
        res = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": SHARED.RESEARCH_RUN_ID,
               "label": "Measured (the sm_120 SASS of NVIDIA's libraries; no device)",
               "method": "cuobjdump -arch sm_120 -sass of each library, every opcode counted; the multiply-family ones kept",
               "workload_fingerprint": {"tool": "tc_rates_fp4", "version": "1", "step": "scan-libs",
                                        "libraries": {r["path"]: r.get("sha256") for r in recs}, "commit": SHARED.RESEARCH_SOURCE_COMMIT},
               "validation": {"status": "passed" if ok else "failed", "checker": "every present library's SASS dumped to the end",
                              "detail": "; ".join(f"{r['path']}: {r.get('error') or r.get('functions')}" for r in recs),
                              "evidence": ["libscan.json"]},
               "measurements": [{"name": "multiply_opcodes", "value": len(seen), "unit": "opcodes"}],
               "measurement_files": ["libscan.json"], "artifacts": []}
        (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------------- collect

def _base(op: str) -> str:
    return op.split(".")[0]


def mma_class(op: str) -> str:
    """An MMA opcode's datapath class without shape: e.g. ``OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X`` -> ``OMMA.SF.SP``."""
    parts = op.split(".")
    keep = [parts[0]] + [p for p in parts[1:] if p in ("SF", "SP")]
    return ".".join(keep)


def collect(args) -> int:
    """The price table over every recorded family (each priced against the NVFP4 of its own chunk), the claim's check, the
    ptxas-refused forms, and (with --libscan) every multiply-family opcode NVIDIA's sm_120 libraries use that no F1 form reached."""
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    bj = Path(os.path.expandvars(args.build)) / "build.json"
    b = json.loads(bj.read_text())
    kept = out / "inputs"
    kept.mkdir(exist_ok=True)
    (kept / "build.json").write_bytes(bj.read_bytes())
    rows, fams, failures = {}, {}, []
    for d in [x for x in args.runs.split(",") if x]:
        for p in sorted(Path(os.path.expandvars(d)).glob("*.json")):
            r = json.loads(p.read_text())
            if r.get("schema") != SCHEMA:
                continue
            if r["family"] in fams:
                failures.append(f"{r['family']}: recorded twice ({fams[r['family']]['file']}, {p})")
                continue
            (kept / p.name).write_bytes(p.read_bytes())
            fams[r["family"]] = {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "gpu_uuid": r["identity"].get("gpu_uuid"),
                                 "clocks": r.get("clocks"), "failures": r.get("failures"),
                                 "build_so_sha256": r["build"]["so_sha256"], "fake_silicon": r.get("fake_silicon")}
            if r.get("fake_silicon"):
                failures.append(f"{r['family']}: fake silicon (a dry run's invented numbers)")
            if r.get("failures"):
                failures.append(f"{r['family']}: {r['failures']}")
                continue
            if r["build"]["so_sha256"] != b["so_sha256"]:
                failures.append(f"{r['family']}: ran a different build ({r['build']['so_sha256'][:12]})")
            for n, v in r["per_form"].items():
                if n == REF and r["family"] != "fp4":
                    rows.setdefault(f"{REF}@{r['family']}", {**v, "family": r["family"], "gpu_uuid": fams[r["family"]]["gpu_uuid"]})
                    continue
                rows[n] = {**v, "family": r["family"], "gpu_uuid": fams[r["family"]]["gpu_uuid"]}
    missing = [f for f in FAMILIES if f not in fams]
    like = {n: v for n, v in rows.items() if v["product"] == "e2m1-capable" and not n.startswith(REF)}
    narrow = {n: v for n, v in rows.items() if v["product"] != "e2m1-capable"}
    cheaper = {n: v["fp4_slots_per_product"] for n, v in like.items() if v["fp4_slots_per_product"] < 1 - args.tolerance}
    # the plain bit-plane/limb decompositions of one E2M1 x E2M1 product (values x2: integers -12..12; Derived, not a lower bound)
    decomp = {"s4": 4, "u4": 4, "b1": 16 * 2}
    narrow_e2m1 = {n: {"fp4_slots_per_own_product": v["fp4_slots_per_product"], "own_products_per_e2m1_product_plain": decomp[v["product"]],
                       "fp4_slots_per_e2m1_product_plain": v["fp4_slots_per_product"] * decomp[v["product"]]} for n, v in narrow.items()}
    table = sorted(((n, v["fp4_slots_per_product"]) for n, v in rows.items()), key=lambda x: x[1])
    unreached = {}
    if args.libscan:
        lj = Path(os.path.expandvars(args.libscan)) / "libscan.json"
        (kept / "libscan.json").write_bytes(lj.read_bytes())
        ls = json.loads(lj.read_text())
        reached = set(b["sass_opcodes_reached"])
        reached_classes = {mma_class(o) for o in reached if MMA_RE.search(o)} | {_base(o) for o in reached}
        for op, cnt in ls["multiply_opcodes_all"].items():
            if MMA_RE.search(op):
                if mma_class(op) not in reached_classes:
                    unreached[op] = cnt
            elif _base(op) not in reached_classes and _base(op) in {"FFMA", "FMUL", "HFMA2", "HMUL2", "IMAD", "IDP", "DFMA", "DMUL",
                                                                 "FMUL2", "FFMA2", "HFMA2.MMA"}:
                unreached[op] = cnt
    doc = {"schema": SCHEMA + "/table", "unit": "one dense NVFP4 nonzero product (mma.sync kind::mxf4nvf4 4X ue4m3 m16n8k64) per SM "
                                                "per clock, measured in each family's own chunk",
           "build_so_sha256": b["so_sha256"], "families": fams, "missing_families": missing, "rows": rows,
           "sorted_fp4_slots_per_product": table, "claim": {
               "statement": "fp4-tile-only/sm120: no instruction multiplies faster than dense NVFP4 per nonzero product",
               "tolerance": args.tolerance, "like_for_like_forms": len(like), "cheaper_like_for_like": cheaper,
               "holds_over_measured": not cheaper and not missing},
           "narrow_products": narrow_e2m1, "refused_by_ptxas": b["refused"], "library_opcodes_unreached": unreached,
           "label": "Measured (rates, prices); Derived (the narrow forms' per-E2M1-product prices, from the plain decompositions)"}
    (out / "f1_table.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    ok = not failures and not missing
    meas = [{"name": f"{n}_fp4_slots_per_product", "value": v["fp4_slots_per_product"], "unit": "NVFP4 products"} for n, v in rows.items()]
    meas += [{"name": f"{n}_products_per_sm_per_clock", "value": v["products_per_sm_per_clock"], "unit": "products/(SM*cycle)"}
             for n, v in rows.items()]
    clocks = sorted({f["clocks"] for f in fams.values()})
    res = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": SHARED.RESEARCH_RUN_ID,
           "clocks": clocks[0] if len(clocks) == 1 else "mixed", "label": "Measured" if clocks == ["locked-2100"] else "Measured, " + ",".join(clocks),
           "method": "F1 instruction microbenchmarks from the fill queue (one family per chunk, each against dense NVFP4 in the same "
                     "process), collected; see f1_table.json",
           "workload_fingerprint": {"tool": "tc_rates_fp4", "version": "1", "build_so_sha256": b["so_sha256"],
                                    "families": {k: v["sha256"] for k, v in fams.items()}, "commit": SHARED.RESEARCH_SOURCE_COMMIT},
           "validation": {"status": "passed" if ok else "failed", "checker": "every family's gates (UUID, cc 12.0, SASS per loop, no FTZ FP32, "
                          "outputs finite and bit-identical, one wave) passed in its chunk, on this build", "detail": "; ".join(failures) or
                          f"{len(fams)} families, {len(rows)} rows", "evidence": ["f1_table.json"]},
           "claim": doc["claim"], "measurements": meas, "measurement_files": ["f1_table.json"], "artifacts": []}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("F1 TABLE", "ok" if ok else "FAILED", failures, missing, flush=True)
    for n, p in table:
        print(f"  {n:28s} {p:9.3f} FP4 slots per product  ({rows[n]['product']})", flush=True)
    if unreached:
        print("  library opcodes no F1 form reached:", unreached, flush=True)
    return 0 if ok else 1


def build_cmd(args) -> int:
    out = Path(os.path.expandvars(args.out))
    names = [x for x in args.forms.split(",") if x] if args.forms else None
    rec = build(out, names, args.jobs)
    print(f"F1 BUILD {'ok' if rec['ok'] else 'FAILED'}: {len(rec['kernels'])} accepted, {len(rec['refused'])} refused by ptxas, "
          f"FTZ {rec['ftz_fp32'] or 'none'}", flush=True)
    for n, g in rec["gate"].items():
        tag = "native" if g.get("native") else ("x%.0f" % g["sass_per_ptx"] if g.get("sass_per_ptx") else "")
        tag += " CALL" if g.get("emulated") else ""
        ops = g.get("mma_opcodes") or g.get("callee_mma_opcodes") or {str(g.get("expected_opcode")): g.get("got")}
        print(f"  {n:28s} {'ok ' if g['ok'] else 'BAD'} {tag:12s} {sorted(ops)}")
    for n, m in rec["refused"].items():
        print(f"  refused {n}: {(m or [''])[0][:140]}")
    res = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": SHARED.RESEARCH_RUN_ID,
           "label": "Measured (ptxas acceptance and SASS of each form; no device)",
           "method": "each F1 form compiled alone for sm_120a, then the accepted forms as one library; loop opcode histograms from cuobjdump",
           "workload_fingerprint": {"tool": "tc_rates_fp4", "version": "1", "step": "build", "so_sha256": rec["so_sha256"],
                                    "source_sha256": rec["source_sha256"], "commit": SHARED.RESEARCH_SOURCE_COMMIT},
           "validation": {"status": "passed" if rec["ok"] else "failed", "checker": "SASS per timed loop; no FADD/FFMA/FMUL .FTZ in the library",
                          "detail": f"{len(rec['kernels'])} accepted, {len(rec['refused'])} refused", "evidence": ["build.json"]},
           "measurements": [{"name": "forms_accepted", "value": len(rec["kernels"]), "unit": "forms"},
                            {"name": "forms_refused", "value": len(rec["refused"]), "unit": "forms"}],
           "measurement_files": ["build.json"], "artifacts": []}
    if SHARED.RESEARCH_RUN_ID:
        (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0 if rec["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--forms", default=None, help="comma list (default: every form)")
    b.add_argument("--jobs", type=int, default=8)
    r = sub.add_parser("run")
    r.add_argument("--build", required=True)
    r.add_argument("--family", required=True, choices=sorted(FAMILIES) + ["_all"])
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--target-ms", type=float, default=100.0)
    r.add_argument("--sustain-s", type=float, default=2.0)
    r.add_argument("--seed", type=int, default=20260930)
    r.add_argument("--expect-uuid", default=None)
    r.add_argument("--locked-mhz", type=float, default=None)
    r.add_argument("--corrupt-word", action="store_true")
    r.add_argument("--fake-silicon", action="store_true")
    s = sub.add_parser("scan-libs")
    s.add_argument("--out", required=True)
    s.add_argument("--libs", default=None)
    s.add_argument("--jobs", type=int, default=5)
    s.add_argument("--timeout-s", type=float, default=1200.0)
    c = sub.add_parser("collect")
    c.add_argument("--build", required=True)
    c.add_argument("--runs", required=True, help="comma list of directories holding <family>.json")
    c.add_argument("--libscan", default=None)
    c.add_argument("--out", required=True)
    c.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args(argv)
    if getattr(args, "fake_silicon", False) and SHARED.RESEARCH_RUN_ID:
        ap.error("--fake-silicon is a dry run and never publishes: unset RESEARCH_RUN_ID")
    return {"build": build_cmd, "run": run_family, "scan-libs": scan_libs, "collect": collect}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
