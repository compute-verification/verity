"""Hardware probe for one tensor-core instruction: one ``mma.sync`` (or ``wgmma``) per tile, arbitrary accumulators, captures in
the golden-record format, compared bit for bit against the pinned ``verity.ml.circuits.silicon.silicon`` model of that instruction.

Provenance: ported from veritor ``gpu/tensor-core-semantics/`` (``mma_tiles.cu``, ``tcs_gpu.py``,
``validate_tiles.py``: the BF16 kernel, encodings, finite-sweep families, per-element records, multiprocess model comparison,
golden sampling) and extended with the E4M3 kernel half and encodings of the same files, the FP8-specific
families (the Ada zero-product accumulator truncation: discrepancy-log entry D10 and ``verity.ml.circuits.silicon.models.ADA_E4M3_M16N8K32``;
Hawkeye's adder-width probe of ``characterize.py``; the special-value classes of ``gpu/tc-fp8/campaign.py``) and the Hopper
``wgmma`` path with the SASS confirmation of ``gpu/tc-fp8/wgmma_fp8_probe.py`` (exactly one MMA-class instruction per tile
kernel).  Only the GPU host runs this; it needs numpy and nvcc (``cuobjdump`` for ``--sass-check``).

Instructions (``--instruction``; ``--arch`` is derived from the entry and may be overridden)::

    id                          nvcc arch  kernel                       K   model(s) compared
    sm80.mma.m16n8k16.bf16      sm_80      mma_bf16_m16n8k16_tiles      16  ampere_bf16_m16n8k16 (pinned)
    sm89.mma.m16n8k16.bf16      sm_89      mma_bf16_m16n8k16_tiles      16  ada_bf16_m16n8k16 (pinned)
    sm90.mma.m16n8k16.bf16      sm_90a     mma_bf16_m16n8k16_tiles      16  hopper_bf16_m16n8k16 (pinned)
    sm89.mma.m16n8k32.e4m3      sm_89      mma_e4m3_m16n8k32_tiles      32  ada_e4m3_m16n8k32 (pinned)
    sm89.mma.m16n8k32.e5m2      sm_89      mma_e5m2_m16n8k32_tiles      32  HYPOTHESIS ada_e4m3_m16n8k32 with e5m2 products
    sm90.mma.m16n8k32.e4m3      sm_90a     mma_e4m3_m16n8k32_tiles      32  HYPOTHESES hopper_e4m3_wgmma_k32, ada_e4m3_m16n8k32 (open question: which
                                                                            agrees) and hopper_bf16_m16n8k16 x2 (see below)
    sm90.mma.m16n8k32.e5m2      sm_90a     mma_e5m2_m16n8k32_tiles      32  HYPOTHESES the three above with e5m2 products
    sm120.mma.m16n8k32.e4m3     sm_120a    mma_e4m3_m16n8k32_tiles      32  blackwell_sm120_e4m3_m16n8k32; controls ada_e4m3_m16n8k32, hopper_bf16_m16n8k16 x2
    sm120.mma.m16n8k32.e5m2     sm_120a    mma_e5m2_m16n8k32_tiles      32  blackwell_sm120_e5m2_m16n8k32; the same controls with e5m2 products
    sm90.wgmma.m64n8k32.e4m3    sm_90a     wgmma_e4m3_m64n8k32_tiles    32  hopper_e4m3_wgmma_k32 (pinned)
    sm90.wgmma.m64n8k32.e5m2    sm_90a     wgmma_e5m2_m64n8k32_tiles    32  HYPOTHESIS hopper_e4m3_wgmma_k32 with e5m2 products
    sm90.wgmma.m64n8k16.bf16    sm_90a     wgmma_bf16_m64n8k16_tiles    16  hopper_bf16_wgmma_k16 (total semantics: specials asserted);
                                       (_rs_tiles: --a-source registers)     control HYPOTHESIS ampere_bf16_m16n8k16
    sm100.mma.m16n8k16.bf16     sm_100a    mma_bf16_m16n8k16_tiles      16  hopper_bf16_m16n8k16 (sm_120's), control ampere_bf16_m16n8k16
    sm100.tcgen05.mma.bf16      sm_100a    tcgen05_bf16_m128n8k16_tiles 16  hopper_bf16_wgmma_k16 (total), control ampere_bf16_m16n8k16 (total)

A pinned model's mismatches fail ``validation``; a HYPOTHESIS model only yields agreement counts (validation ``not_run`` when no
pinned model is compared) and the raw capture is recorded regardless.  The E5M2 product (``_e5m2_product``) is a local
HYPOTHESIS pending the canonical one in ``verity.ml.circuits.silicon``.  Kernels: ``mma_tiles.cu`` (``mma.sync``, sm_80+/sm_89+) and
``wgmma_tiles.cu`` (``wgmma.mma_async``, sm_90a; B through a shared-memory descriptor, A from registers -- or, for BF16, from a
second descriptor, the SS form GEMM mainloops issue) and ``tcgen05_tiles.cu`` (``tcgen05.mma.cta_group::1.kind::f16``, sm_100a: A
and B through descriptors, the accumulator in tensor memory, M = 128; ``--no-input-d`` runs its first MMA with enable_input_d = 0,
the accumulator not read, over zero accumulators, as a GEMM mainloop's first k step).

SASS (nvcc 12.4): every tile kernel holds exactly one MMA-class instruction --
``HMMA.16816.F32.BF16`` (sm_80/sm_89/sm_90a bf16), ``QMMA.16832.F32.E4M3.E4M3`` / ``QMMA.16832.F32.E5M2.E5M2`` (sm_89 FP8),
``QGMMA.64x8x32.F32.E4M3.E4M3`` / ``QGMMA.64x8x32.F32.E5M2.E5M2`` (sm_90a wgmma) -- EXCEPT ``mma.sync.m16n8k32.{e4m3,e5m2}`` on
sm_90a, which ptxas lowers to ``F2FP.F16.<fmt>.UNPACK_B`` conversions, two ``HMMA.16816.F32`` (the FP16 tensor-core instruction,
K = 16 each, both with a ZERO accumulator: the low halves of the operand words, k mod 4 in {0, 1}, and the high halves, k mod 4
in {2, 3}), an ``FADD`` of the two partial words and an ``FADD`` with the accumulator: there is no FP8 ``mma.sync`` tensor-core
instruction on Hopper, so those two ids are not single-instruction probes.  The table records that count (``Instruction.sass_mma``
= 2; the SASS check compares against it and flags ``single_instruction: false``), and their third HYPOTHESIS is exactly that
lowering around Hopper's K=16 HMMA pipeline with exact FP8 products (``SM90_FP8_MMA_LOWERING``).

    python benchmarks/tc_probe/tc_probe.py --instruction sm89.mma.m16n8k16.bf16 --out $RESEARCH_RUN_DIR \\
        --replay-golden verity/ml/circuits/ops/tests/fixtures/golden/ada_bf16_m16n8k16.json --sweep --n-random 100000
    python benchmarks/tc_probe/tc_probe.py --instruction sm89.mma.m16n8k32.e4m3 --out $RESEARCH_RUN_DIR --sweep --sass-check
    python benchmarks/tc_probe/tc_probe.py --instruction sm90.wgmma.m64n8k32.e4m3,sm90.mma.m16n8k32.e4m3 --out DIR --compile-only

``--replay-golden`` re-executes every record of a golden file on the device (each at its own tile position) and requires the
hardware word to equal the record's ``d``.  ``--replay-instances DIR`` replays ``bench-instances/v1`` (BF16 only: the ``tu-k16``
tier as one launch, ``vu-k1536`` as 96 dependent launches with the hardware's accumulators carried) against the dataset's
expected words, after checking the arrays' sha256 against the pinned manifest (``--instances-manifest``).  ``--sweep`` runs the
finite-sweep families and compares every output element with the model(s); ``--families`` selects a comma list.  Families
(``M`` = 16 rows for ``mma``, 64 for ``wgmma``; ``n`` = ``--n-random`` tiles of ``M x 8`` elements):

    randn_zero_acc (n), randn_random_acc (n/4), uniform_bits (n/4), subnormal (n/8), cancellation (n/8), mixed_magnitude (n/8),
    near_overflow (n/16), signed_zero (256), tiny_acc_zero_products (512)               -- every format, as in the Ada validation
    zero_acc_tiny_cancellation (max(256, n/16))                                          -- BF16 only (the zero's exponent)
    zero_products_nonzero_acc (512): all products zero, random finite accumulators        -- FP8: the accumulator truncation quirk
    acc_only_lsb_walk (64 bases x 1024 low-10-bit patterns): zero products               -- FP8: which accumulator bits survive
    specials (48): NaN (E4M3) / inf and NaN (E5M2) operand and accumulator classes       -- FP8: hardware words only, never asserted
    adder_width_cancellation: +P, -P and a residual 2^ratio.P in every k slot, ratio in [-26, -5], zero and P accumulators
                                                                                          -- FP8: Hawkeye's width and grouping probe

Seeds are fixed (``--seed``, default 20260902) and recorded.  Outputs under ``--out``: ``probe_results.json`` (identity, layout
check, per-family counts per model, first mismatches with A/B/C/D words, family summaries, SASS counts), ``capture_sample.json``
(golden-format records sampled per family), ``tiles_<family>.npz`` (the full capture: A, Bt, C, D per tile) and, in a recorded
run (``RESEARCH_RUN_ID`` set), ``result.json`` with ``kind: tc-evidence/v1`` (``workload_fingerprint`` = instruction, arch, sku, driver, nvcc,
seed, n, families, models; ``tc_evidence`` = per-family counts, ``model_agrees`` per model, first mismatches; validation passed
iff every mismatch count against a pinned model and every replay is zero).  ``--compile-only`` builds the instruction's kernel
for its arch (sm_90a compiles wherever the toolkit is, whatever the GPU), runs ``--sass-check`` (cuobjdump: MMA-class
instructions per tile kernel, compared with the table's ``sass_mma``; the tile kernel's SASS is kept as ``sass_<kernel>_<arch>.txt``)
and writes ``result.json`` with ``kind: tc-compile-check/v1`` without touching the device.

``--canary`` (``canary.py``) runs the instruction's fixed canary tile set -- the first tiles of every family under the canary seed
plus every regression tile of every resolved discrepancy-log entry naming the instruction -- in under a minute, exits 1 on any
mismatch against a declared model, and marks ``result.json`` (``tc-evidence/v1``) with ``workload_fingerprint.family = canary``
and ``tc_evidence.canary``; the trust framework (``trust.py``) treats a failing canary on the anchor as an
invalidation trigger.  Run it after any driver / ptxas / toolkit change on an anchor and on a second device of the SKU.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.circuits.silicon import models as TCM
from verity.ml.circuits.silicon import silicon as S
from verity.ml.circuits.silicon import total as T

HERE = Path(__file__).resolve().parent
N = 8  # every shape here has 8 output columns
MAX_TILES_PER_CALL = 1 << 16
FIRST_MISMATCHES = 20
UNASSERTED_FAMILIES = {"specials"}  # recorded, never compared for validation -- unless the declared model is total (family_asserted)
SEED_DEFAULT = 20260902


# --- operand formats -----------------------------------------------------------------------------------------------------------------


def f32_bits(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even FP32 -> BF16 (torch's cast)."""

    u = f32_bits(x).astype(np.uint64)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def random_finite_fp32_bits(rng, n, exp_lo=1, exp_hi=254):
    e = rng.integers(exp_lo, exp_hi + 1, size=n).astype(np.uint32)
    m = rng.integers(0, 1 << 23, size=n).astype(np.uint32)
    s = rng.integers(0, 2, size=n).astype(np.uint32)
    return (s << 31) | (e << 23) | m


@dataclass(frozen=True)
class Format:
    """An operand encoding: ``bits`` wide, ``exp_bits``/``man_bits`` fields, ``bias``; E4M3 (OCP, no infinities, NaN = 0x7f/0xff),
    E5M2 (IEEE-like: exponent 31 = inf/NaN), BF16."""

    name: str
    bits: int
    exp_bits: int
    man_bits: int
    bias: int

    @property
    def dtype(self):
        return np.uint16 if self.bits == 16 else np.uint8

    @property
    def hex_width(self) -> int:
        return self.bits // 4

    @property
    def exp_mask(self) -> int:
        return (1 << self.exp_bits) - 1

    @property
    def finite_exp_max(self) -> int:
        """Largest exponent field that holds finite values (E4M3: 15 except the all-ones mantissa; others: max - 1)."""

        return self.exp_mask if self.name == "e4m3" else self.exp_mask - 1

    @property
    def min_pow2_exp(self) -> int:
        """Exponent of the smallest positive value (a subnormal with mantissa 1)."""

        return 1 - self.bias - self.man_bits

    @property
    def max_pow2_exp(self) -> int:
        """Exponent of the largest finite power of two."""

        return self.finite_exp_max - self.bias

    def is_finite(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w)
        e = (w >> self.man_bits) & self.exp_mask
        if self.name == "e4m3":
            return (w & 0x7F) != 0x7F
        return e != self.exp_mask

    def decode(self, w: np.ndarray) -> np.ndarray:
        """Word -> float64 value (NaN for non-finite encodings; +-inf of E5M2 also NaN here: not a value of the finite model)."""

        w = np.asarray(w).astype(np.int64)
        s = (w >> (self.bits - 1)) & 1
        e = (w >> self.man_bits) & self.exp_mask
        m = w & ((1 << self.man_bits) - 1)
        normal = e != 0
        val = np.where(normal, (1.0 + m / 2.0**self.man_bits) * np.exp2(e - self.bias), m * 2.0 ** (1 - self.bias - self.man_bits))
        val = np.where(s == 1, -val, val)
        return np.where(self.is_finite(w), val, np.nan)

    def pow2(self, exponent: int, negative: bool = False) -> int:
        """The word of +-2^exponent (normal or subnormal power of two)."""

        if exponent >= 1 - self.bias:
            w = (exponent + self.bias) << self.man_bits
        else:
            w = 1 << (exponent - self.min_pow2_exp)
            assert 1 <= w < (1 << self.man_bits), exponent
        w |= int(negative) << (self.bits - 1)
        assert self.is_finite(np.array(w)), (self.name, exponent)
        return w

    def from_f32(self, x: np.ndarray) -> np.ndarray:
        """Round-to-nearest-even FP32 -> word, saturating to the largest finite magnitude; -0.0 keeps its sign."""

        if self.name == "bf16":
            return to_bf16_bits(x)
        x = np.asarray(x, dtype=np.float32).astype(np.float64)
        codes, vals = _fp8_positive_table(self)
        a = np.abs(x)
        hi = np.clip(np.searchsorted(vals, a, side="left"), 0, len(vals) - 1)  # vals[hi-1] < a <= vals[hi]
        lo = np.clip(hi - 1, 0, len(vals) - 1)
        d_lo, d_hi = a - vals[lo], vals[hi] - a
        choose_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (codes[hi] % 2 == 0))  # ties to the even code; a > max -> hi (saturate)
        code = np.where(choose_hi, codes[hi], codes[lo]).astype(np.uint8)
        return (code | (np.signbit(x).astype(np.uint8) << 7)).astype(np.uint8)

    def random_finite_bits(self, rng, shape, exp_lo=None, exp_hi=None) -> np.ndarray:
        """Uniformly random sign, exponent field in [exp_lo, exp_hi] (default 1 .. finite max) and mantissa; NaN codes repaired
        to the largest finite mantissa (E4M3) -- the BF16 probe's draws."""

        n = int(np.prod(shape))
        lo, hi = (1 if exp_lo is None else exp_lo), (self.finite_exp_max if exp_hi is None else exp_hi)
        e = rng.integers(lo, hi + 1, size=n).astype(self.dtype)
        m = rng.integers(0, 1 << self.man_bits, size=n).astype(self.dtype)
        s = rng.integers(0, 2, size=n).astype(self.dtype)
        w = ((s << (self.bits - 1)) | (e << self.man_bits) | m).astype(self.dtype)
        if self.name == "e4m3":
            w[(w & 0x7F) == 0x7F] &= 0xFE
        return w.reshape(shape)


BF16 = Format("bf16", 16, 8, 7, 127)
E4M3 = Format("e4m3", 8, 4, 3, 7)
E5M2 = Format("e5m2", 8, 5, 2, 15)
FORMATS = {f.name: f for f in (BF16, E4M3, E5M2)}
_FP8_TABLES: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _fp8_positive_table(fmt: Format) -> tuple[np.ndarray, np.ndarray]:
    """The positive finite codes of an 8-bit format in increasing value order (monotone in the code) and their values."""

    if fmt.name not in _FP8_TABLES:
        codes = np.arange(128, dtype=np.int64)
        codes = codes[fmt.is_finite(codes)]
        vals = fmt.decode(codes)
        assert np.all(np.diff(vals) > 0)
        _FP8_TABLES[fmt.name] = (codes, vals)
    return _FP8_TABLES[fmt.name]


# --- the instruction table -----------------------------------------------------------------------------------------------------------


def _e5m2_product(a: int, b: int) -> S.Term:
    """HYPOTHESIS: exact product of two E5M2 bytes as a 25-bit-scaled :class:`Term`, formed like :func:`silicon.e4m3_product`
    (3x3-bit significands -> 6 bits, shifted by 19; subnormals keep their unnormalised significand at exponent -14).  Exponent
    field 31 (infinities and NaN) is outside the finite model and raises.  The canonical product is silicon-p0's to add."""

    ea, eb = (a >> 2) & 0x1F, (b >> 2) & 0x1F
    if ea == 0x1F or eb == 0x1F:
        raise S.InvalidArtifact("non-finite E5M2 operand")
    sa = (a & 0x3) | (0x4 if ea else 0)
    sb = (b & 0x3) | (0x4 if eb else 0)
    magnitude = (sa * sb) << 19
    if magnitude == 0:
        return S.ZERO
    return S.Term(((a ^ b) >> 7) & 1 == 1, (ea or 1) + (eb or 1) - 30, magnitude)


PRODUCTS: dict[str, Callable[[int, int], S.Term]] = {"e4m3": S.e4m3_product, "e5m2": _e5m2_product}


def tc_dot_with_product(pipeline: S.Pipeline, product: Callable[[int, int], S.Term], acc_bits: int, a, b) -> int:
    """:func:`silicon.tc_dot` with the products formed by ``product`` instead of the pipeline's own (hypothesis models)."""

    if len(a) != pipeline.k or len(b) != pipeline.k:
        raise S.InvalidArtifact(f"{pipeline.name} takes {pipeline.k} operand pairs")
    acc = S.fp32_term(acc_bits)
    start = 0
    for size in pipeline.groups:
        terms = [acc, *(product(a[i], b[i]) for i in range(start, start + size))]
        acc = S.group_sum(terms, width=pipeline.width, zero_exponent=pipeline.zero_exponent)
        if acc.magnitude and acc.exponent > S.FP32_MAX_EXPONENT:
            return S.pack_fp32(acc)
        start += size
    return S.pack_fp32(acc)


def fp32_add_bits(x: int, y: int) -> int:
    """IEEE FP32 round-to-nearest-even addition on bit patterns (SASS ``FADD`` without ``.FTZ``: subnormals kept)."""

    with np.errstate(all="ignore"):
        r = np.array([x], np.uint32).view(np.float32) + np.array([y], np.uint32).view(np.float32)
    return int(r.view(np.uint32)[0])


SM90_FP8_MMA_LOWERING = "2xHMMA.16816.F32+FADD"
# ptxas (12.4, sm_90a) lowers mma.sync.m16n8k32.{e4m3,e5m2} to: F2FP.F16.<fmt>.UNPACK_B of every operand word (low half and high
# half separately), HMMA.16816.F32 over the low halves with a zero accumulator, HMMA.16816.F32 over the high halves with a zero
# accumulator, FADD of the two partial words, FADD with the accumulator (sass_*.txt).  The
# low half of an operand word holds k = 4t, 4t+1 (and 16+4t, 16+4t+1), the high half k = 4t+2, 4t+3 (and 16+..): the partition
# below.  The one open assumption is that HMMA.16816.F32 with FP16 inputs has Hopper's BF16 K=16 pipeline (groups (16,), width
# 26); the FP8 values and products are exact in FP16, so the terms are the same numbers.
_SM90_LOW_K = [k for k in range(32) if k % 4 in (0, 1)]
_SM90_HIGH_K = [k for k in range(32) if k % 4 in (2, 3)]


def sm90_fp8_mma_lowered(pipeline16: S.Pipeline, product: Callable[[int, int], S.Term], acc_bits: int, a, b) -> int:
    if len(a) != 32 or len(b) != 32:
        raise S.InvalidArtifact("the sm_90a FP8 mma.sync lowering takes 32 operand pairs")
    low = tc_dot_with_product(pipeline16, product, 0, [a[k] for k in _SM90_LOW_K], [b[k] for k in _SM90_LOW_K])
    high = tc_dot_with_product(pipeline16, product, 0, [a[k] for k in _SM90_HIGH_K], [b[k] for k in _SM90_HIGH_K])
    return fp32_add_bits(acc_bits, fp32_add_bits(low, high))


@dataclass(frozen=True)
class Model:
    """A model the capture is compared with.  ``product`` names an entry of :data:`PRODUCTS` (None: the pipeline's own product,
    evaluated by :func:`silicon.tc_dot` exactly as pinned).  ``lowering`` names a multi-instruction lowering the pipeline is
    embedded in (:data:`SM90_FP8_MMA_LOWERING`).  ``total`` evaluates the pipeline's total semantics
    (:func:`verity.ml.circuits.silicon.total.tc_dot_total`: NaN, infinities and saturation included), so the ``specials`` family is asserted
    too.  ``declared`` models fail validation on any mismatch; the others are HYPOTHESES whose agreement counts are reported."""

    pipeline: S.Pipeline
    product: str | None = None
    declared: bool = True
    lowering: str | None = None
    total: bool = False

    @property
    def name(self) -> str:
        n = self.pipeline.name + (f"+{self.product}_product" if self.product else "") + (f"/{self.lowering}" if self.lowering else "")
        return n if self.declared else n + "[HYPOTHESIS]"

    def describe(self) -> dict:
        p = self.pipeline
        return {"name": self.name, "pipeline": p.name, "groups": list(p.groups), "width": p.width, "zero_exponent": p.zero_exponent,
                "product": self.product or p.dtype, "lowering": self.lowering, "declared": self.declared, "total": self.total}

    def evaluate(self, acc_bits: int, a, b) -> int:
        if self.total:
            return T.tc_dot_total(self.pipeline, acc_bits, a, b)
        if self.lowering == SM90_FP8_MMA_LOWERING:
            return sm90_fp8_mma_lowered(self.pipeline, PRODUCTS[self.product or self.pipeline.dtype], acc_bits, a, b)
        if self.product is None:
            return S.tc_dot(self.pipeline, acc_bits, a, b)
        return tc_dot_with_product(self.pipeline, PRODUCTS[self.product], acc_bits, a, b)


@dataclass(frozen=True)
class Instruction:
    id: str
    arch: str  # nvcc -arch
    tu: str  # translation unit under benchmarks/tc_probe/
    kernel: str  # extern "C" entry; the device kernel is <kernel>_kernel
    ptx: str
    fmt: Format
    m: int
    k: int
    models: tuple[Model, ...]
    sass_mma: int = 1  # MMA-class SASS instructions ptxas emits for the PTX instruction (measured; 1 = single-instruction probe)
    rs_kernel: str | None = None  # the same instruction with A from registers (``--a-source registers``); ``kernel`` reads A from shared memory
    chain: bool = False  # the TU has ``<kernel>_chain4`` (``--chain``)

    @property
    def a_sources(self) -> tuple[str, ...]:
        """``--a-source`` values: ``smem`` and ``registers`` where the table has both kernels, else the one ``kernel`` uses."""

        if self.rs_kernel is not None:
            return ("smem", "registers")
        return ("registers",) if self.tu == "wgmma_tiles.cu" else ("n/a",)

    def entry(self, a_source: str | None) -> str:
        return self.rs_kernel if a_source == "registers" and self.rs_kernel is not None else self.kernel

    def describe(self) -> dict:
        return {"id": self.id, "arch": self.arch, "tu": self.tu, "kernel": self.kernel, "rs_kernel": self.rs_kernel, "ptx": self.ptx,
                "format": self.fmt.name, "m": self.m, "n": N, "k": self.k, "sass_mma": self.sass_mma, "models": [m.describe() for m in self.models]}


def _mma(id_: str, arch: str, fmt: Format, models: tuple[Model, ...], sass_mma: int = 1) -> Instruction:
    k = 16 if fmt is BF16 else 32
    return Instruction(id_, arch, "mma_tiles.cu", f"mma_{fmt.name}_m16n8k{k}_tiles",
                       f"mma.sync.aligned.m16n8k{k}.row.col.f32.{fmt.name}.{fmt.name}.f32", fmt, 16, k, models, sass_mma)


def _wgmma(id_: str, fmt: Format, models: tuple[Model, ...]) -> Instruction:
    k = 16 if fmt is BF16 else 32
    rs = f"wgmma_{fmt.name}_m64n8k{k}_rs_tiles" if fmt is BF16 else None  # FP8: A is always in registers
    return Instruction(id_, "sm_90a", "wgmma_tiles.cu", f"wgmma_{fmt.name}_m64n8k{k}_tiles",
                       f"wgmma.mma_async.sync.aligned.m64n8k{k}.f32.{fmt.name}.{fmt.name}", fmt, 64, k, models, rs_kernel=rs, chain=rs is not None)


def _tcgen05(id_: str, models: tuple[Model, ...]) -> Instruction:
    return Instruction(id_, "sm_100a", "tcgen05_tiles.cu", "tcgen05_bf16_m128n8k16_tiles",
                       "tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc(bf16 x bf16 -> f32, M128 N16 K16, K-major), enable_input_d",
                       BF16, 128, 16, models, chain=True)


_HOPPER_E5M2 = Model(S.HOPPER_E4M3_K32, "e5m2", declared=False)
_ADA_E5M2 = Model(S.ADA_E4M3_M16N8K32, "e5m2", declared=False)
# sm_90a lowers mma.sync.m16n8k32.{e4m3,e5m2} to two HMMA.16816.F32 plus FADDs (see SM90_FP8_MMA_LOWERING): the third hypothesis
# for those ids is that lowering around Hopper's K=16 HMMA pipeline with the exact FP8 products.
_HOPPER_HMMA_X2_E4M3 = Model(S.HOPPER_BF16_M16N8K16, "e4m3", declared=False, lowering=SM90_FP8_MMA_LOWERING)
_HOPPER_HMMA_X2_E5M2 = Model(S.HOPPER_BF16_M16N8K16, "e5m2", declared=False, lowering=SM90_FP8_MMA_LOWERING)

INSTRUCTIONS: dict[str, Instruction] = {
    i.id: i
    for i in (
        _mma("sm80.mma.m16n8k16.bf16", "sm_80", BF16, (Model(S.AMPERE_BF16_M16N8K16),)),
        _mma("sm89.mma.m16n8k16.bf16", "sm_89", BF16, (Model(S.ADA_BF16_M16N8K16),)),
    _mma("sm90.mma.m16n8k16.bf16", "sm_90a", BF16, (Model(S.HOPPER_BF16_M16N8K16),)),
    # Blackwell consumer (RTX 5090): the registry pins the Hopper BF16 pipeline (fp4-5090 rejected the Ampere one at 2.5e7 elements).
    _mma("sm120.mma.m16n8k16.bf16", "sm_120a", BF16, (Model(S.HOPPER_BF16_M16N8K16),)),
    _mma("sm89.mma.m16n8k32.e4m3", "sm_89", E4M3, (Model(S.ADA_E4M3_M16N8K32),)),
        _mma("sm89.mma.m16n8k32.e5m2", "sm_89", E5M2, (_ADA_E5M2,)),
        _mma("sm90.mma.m16n8k32.e4m3", "sm_90a", E4M3,
             (Model(S.HOPPER_E4M3_K32, declared=False), Model(S.ADA_E4M3_M16N8K32, declared=False), _HOPPER_HMMA_X2_E4M3), sass_mma=2),
        _mma("sm90.mma.m16n8k32.e5m2", "sm_90a", E5M2, (_HOPPER_E5M2, _ADA_E5M2, _HOPPER_HMMA_X2_E5M2), sass_mma=2),
        # Blackwell consumer: the fitted one-group-of-32, 26-bit model; Ada's pipeline and the sm_90a two-HMMA lowering stay as controls.
        _mma("sm120.mma.m16n8k32.e4m3", "sm_120a", E4M3,
             (Model(TCM.BLACKWELL_SM120_E4M3_M16N8K32), Model(S.ADA_E4M3_M16N8K32, declared=False), _HOPPER_HMMA_X2_E4M3)),
        _mma("sm120.mma.m16n8k32.e5m2", "sm_120a", E5M2, (Model(TCM.BLACKWELL_SM120_E5M2_M16N8K32), _ADA_E5M2, _HOPPER_HMMA_X2_E5M2)),
        _wgmma("sm90.wgmma.m64n8k32.e4m3", E4M3, (Model(S.HOPPER_E4M3_K32),)),
        _wgmma("sm90.wgmma.m64n8k32.e5m2", E5M2, (_HOPPER_E5M2,)),
        # The BF16 wgmma vLLM's H100 GEMMs issue, against its total semantics (specials asserted); the Ampere
        # pipeline is the control that the tiles distinguish one group of 16 / 26 bits from two groups of 8 / 25 bits.
        _wgmma("sm90.wgmma.m64n8k16.bf16", BF16, (Model(TCM.HOPPER_BF16_WGMMA_K16, total=True), Model(S.AMPERE_BF16_M16N8K16, declared=False, total=True))),
        # Blackwell datacenter (B200): the mma.sync FA2 runs (JIT from its sm_80 PTX) and the tcgen05 k16 step cuBLAS's GEMMs issue, each
        # against Hopper's k16 pipeline (which they match on every word) with Ampere's as the control that the families tell them apart.
        _mma("sm100.mma.m16n8k16.bf16", "sm_100a", BF16, (Model(S.HOPPER_BF16_M16N8K16), Model(S.AMPERE_BF16_M16N8K16, declared=False))),
        _tcgen05("sm100.tcgen05.mma.bf16", (Model(TCM.HOPPER_BF16_WGMMA_K16, total=True),
                                            Model(S.AMPERE_BF16_M16N8K16, declared=False, total=True))),
    )
}
ARCH_DEFAULT_INSTRUCTION = {"sm_80": "sm80.mma.m16n8k16.bf16", "sm_89": "sm89.mma.m16n8k16.bf16", "sm_90a": "sm90.mma.m16n8k16.bf16"}


# --- the kernel ----------------------------------------------------------------------------------------------------------------


def cuda_bin(tool: str) -> str:
    found = shutil.which(tool)
    return found or f"/usr/local/cuda/bin/{tool}"  # the CUDA images do not put the toolkit on a non-login PATH


def build(tu: str, arch: str, out: Path) -> tuple[Path, str]:
    so = out / f"lib{Path(tu).stem}_{arch}.so"
    nvcc_bin = cuda_bin("nvcc")
    nvcc = subprocess.run([nvcc_bin, "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-2:]
    # sm_100a's tcgen05 is an `a`-feature instruction: a plain -arch=sm_100a also embeds compute_100 PTX, which refuses it
    target = ["-gencode", f"arch=compute_{arch[3:]},code={arch}"] if arch == "sm_100a" else [f"-arch={arch}"]
    subprocess.run([nvcc_bin, "-O2", *target, "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(HERE / tu)], check=True)
    ident = " / ".join(line.strip() for line in nvcc)
    Path(f"{so}.nvcc").write_text(ident + "\n")
    return so, ident


def prebuilt(path: str) -> tuple[Path, str]:
    """A library `build` produced on another host (``--library``): the path and its toolchain identity (``<path>.nvcc``) plus sha256."""
    so = Path(path)
    sidecar = Path(f"{so}.nvcc")
    ident = sidecar.read_text().strip() if sidecar.exists() else "unknown toolchain"
    return so, f"{ident} (prebuilt {so.name}, sha256 {hashlib.sha256(so.read_bytes()).hexdigest()})"


MMA_SASS = re.compile(r"\b((?:[A-Z]{1,4})?MMA(?:\.[A-Za-z0-9_.]+)?)\b")  # HMMA, QGMMA, ... and tcgen05's UTCHMMA


def sass_check(so: Path, kernel: str, out: Path | None = None, expected: int = 1) -> dict:
    """``cuobjdump -sass`` of the built library: MMA-class instructions (HMMA/QMMA/HGMMA/QGMMA/IMMA...) per device function, and
    the count in the instruction's tile kernel, which must equal ``expected`` -- one for a single-instruction probe; the table
    records the measured count where ptxas lowers a PTX instruction to several (``Instruction.sass_mma``).  With ``out`` the
    tile kernel's SASS is written to ``sass_<kernel>_<arch>.txt`` there (the evidence behind the count)."""

    proc = subprocess.run([cuda_bin("cuobjdump"), "-sass", str(so)], capture_output=True, text=True)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip()[-2000:], "kernel": kernel}
    device_kernel = f"{kernel}_kernel"
    per_function: dict[str, dict] = {}
    excerpt: list[str] = []
    current = None
    for line in proc.stdout.splitlines():
        m = re.search(r"Function\s*:\s*(\S+)", line)
        if m:
            current = m.group(1)
            per_function[current] = {"mma_count": 0, "mma": []}
        if current is not None and device_kernel in current:
            excerpt.append(line)
        if m or current is None or "/*" not in line:
            continue
        code = line.split("/*", 1)[1].split("*/", 1)[1] if line.lstrip().startswith("/*") else line  # drop the offset comment
        code = code.split("/*", 1)[0]  # drop the encoding comment
        for hit in MMA_SASS.findall(code):
            per_function[current]["mma_count"] += 1
            if hit not in per_function[current]["mma"]:
                per_function[current]["mma"].append(hit)
    tile_fns = {name: v for name, v in per_function.items() if device_kernel in name}
    count = sum(v["mma_count"] for v in tile_fns.values())
    result = {
        "ok": bool(tile_fns) and count == expected,
        "kernel": device_kernel,
        "tile_kernel_functions": tile_fns,
        "tile_kernel_mma_count": count,
        "expected_mma_count": expected,
        "single_instruction": bool(tile_fns) and count == 1,
        "functions": per_function,
        "sass_sha256": _sha256_text(proc.stdout),
    }
    if out is not None:
        arch = so.stem.rsplit("_", 2)[-2:]  # lib<tu>_sm_90a -> sm_90a
        name = f"sass_{kernel}_{'_'.join(arch)}.txt"
        (out / name).write_text("\n".join(excerpt) + "\n")
        result["sass_file"] = name
    return result


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


class Probe:
    def __init__(self, so: Path, ins: Instruction, a_source: str | None = None, kernel: str | None = None, k: int | None = None):
        """``kernel`` / ``k`` override the instruction's entry point and tile depth (the chained BF16 wgmma kernels, K = 64)."""

        self.ins = ins
        self.a_source = a_source or ins.a_sources[0]
        if self.a_source not in ins.a_sources:
            raise ValueError(f"{ins.id} has no {self.a_source!r} A operand path (has {', '.join(ins.a_sources)})")
        self.kernel = kernel or ins.entry(self.a_source)
        self.k = k or ins.k
        self.lib = ctypes.CDLL(str(so))
        fn = getattr(self.lib, self.kernel)
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.fn = fn
        ident = self.lib.device_identity
        ident.restype = ctypes.c_int
        ident.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int]
        self.descriptor: dict | None = None
        self.tcgen05 = ins.tu == "tcgen05_tiles.cu"
        if ins.tu in ("wgmma_tiles.cu", "tcgen05_tiles.cu"):
            self._set_offsets = getattr(self.lib, f"{Path(ins.tu).stem.split('_')[0]}_set_descriptor_offsets")
            self._set_offsets.restype = None
            self._set_offsets.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
            self.set_descriptor(128, 256)
        if self.tcgen05:
            self.lib.tcgen05_replica_mismatches.restype = ctypes.c_ulonglong
            self.lib.tcgen05_set_enable_input_d.restype = None
            self.lib.tcgen05_set_enable_input_d.argtypes = [ctypes.c_uint32]

    def set_descriptor(self, lbo: int, sbo: int) -> None:
        self._set_offsets(lbo, sbo)
        self.descriptor = {"lbo_bytes": lbo, "sbo_bytes": sbo, "layout": "K-major no swizzle (INTERLEAVE)"}

    def set_enable_input_d(self, enable: bool) -> None:
        """tcgen05: the first MMA's enable_input_d (False: D = A.B, the accumulator not read)."""
        self.lib.tcgen05_set_enable_input_d(1 if enable else 0)

    def identity(self) -> dict:
        major, minor = ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        rc = self.lib.device_identity(ctypes.byref(major), ctypes.byref(minor), name, 256)
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in cudaGetDeviceProperties")
        return {"device": name.value.decode(), "compute_capability": f"{major.value}.{minor.value}", **host_identity()}

    def tiles(self, A: np.ndarray, Bt: np.ndarray, C: np.ndarray) -> np.ndarray:
        """One instruction per tile.  ``A``: (n,M,K), ``Bt``: (n,8,K) operand words, ``C``: (n,M,8) FP32 words -> ``D`` (n,M,8)."""

        M, K, fmt = self.ins.m, self.k, self.ins.fmt
        n = A.shape[0]
        A = np.ascontiguousarray(A, dtype=fmt.dtype).reshape(n, M, K)
        Bt = np.ascontiguousarray(Bt, dtype=fmt.dtype).reshape(n, N, K)
        C = np.ascontiguousarray(C, dtype=np.uint32).reshape(n, M, N)
        D = np.zeros((n, M, N), dtype=np.uint32)
        for start in range(0, n, MAX_TILES_PER_CALL):
            stop = min(n, start + MAX_TILES_PER_CALL)
            a, b, c = A[start:stop], Bt[start:stop], C[start:stop]
            d = np.zeros((stop - start, M, N), dtype=np.uint32)
            rc = self.fn(a.ctypes.data, b.ctypes.data, c.ctypes.data, d.ctypes.data, stop - start)
            if rc != 0:
                raise RuntimeError(f"CUDA error {rc} in {self.kernel} (209 = no kernel image for this device: wrong --arch?)")
            if self.tcgen05 and (bad := int(self.lib.tcgen05_replica_mismatches())):
                raise RuntimeError(f"{self.kernel}: {bad} words of the instruction's columns 8..15 differ from its columns 0..7")
            D[start:stop] = d
        return D

    # kept for the BF16 replay code paths
    mma_tiles = tiles


def host_identity() -> dict:
    """``nvidia_smi`` (name, driver, compute capability of GPU 0), ``driver``, ``hostname`` and ``gpu_uuid`` -- the board's own
    identity (``nvidia-smi --query-gpu=uuid``; the trust framework's S1 wants the device, not the pod's hostname)."""

    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,uuid", "--format=csv,noheader", "-i", "0"], capture_output=True, text=True)
        text = smi.stdout.strip() if smi.returncode == 0 else None
    except OSError:
        text = None
    fields = [f.strip() for f in text.split(",")] if text else []
    uuid = fields[3] if len(fields) >= 4 and fields[3].startswith("GPU-") else None
    return {"nvidia_smi": ", ".join(fields[:3]) if fields else None, "driver": fields[1] if len(fields) >= 2 else None, "hostname": platform.node(),
            "gpu_uuid": uuid}


# --- the finite-sweep families ---------------------------------------------------------------------------------------------------------


def families(rng, n_random, fmt: Format, M: int, K: int):
    """Yield (name, A (n,M,K), Bt (n,8,K), C (n,M,8), meta) -- the nine families of the Ada validation (unchanged for BF16: the
    same draws in the same order), the zero-exponent family (BF16) and the FP8-specific families."""

    def cast(x):
        return fmt.from_f32(x.astype(np.float32))
    def zeros_acc(n):
        return np.zeros((n, M, N), np.uint32)
    fp8 = fmt.bits == 8

    # 1. Hawkeye-style: standard normal operands, zero accumulator.
    n = n_random
    yield "randn_zero_acc", cast(rng.standard_normal((n, M, K))), cast(rng.standard_normal((n, N, K))), zeros_acc(n), None

    # 2. standard normal operands, random nonzero accumulators over a wide range.
    n = n_random // 4
    acc = rng.standard_normal((n, M, N)) * np.exp2(rng.integers(-20, 21, (n, M, N)))
    yield "randn_random_acc", cast(rng.standard_normal((n, M, K))), cast(rng.standard_normal((n, N, K))), f32_bits(acc.astype(np.float32)), None

    # 3. uniformly random finite encodings for everything (all exponents).
    n = n_random // 4
    yield (
        "uniform_bits",
        fmt.random_finite_bits(rng, (n, M, K)),
        fmt.random_finite_bits(rng, (n, N, K)),
        random_finite_fp32_bits(rng, n * M * N).reshape(n, M, N),
        None,
    )

    # 4. subnormal-heavy operands, tiny/subnormal accumulators.
    n = n_random // 8
    sub_hi = 3 if fmt is BF16 else 1
    yield (
        "subnormal",
        fmt.random_finite_bits(rng, (n, M, K), 0, sub_hi),
        fmt.random_finite_bits(rng, (n, N, K), 0, sub_hi),
        random_finite_fp32_bits(rng, n * M * N, 0, 3).reshape(n, M, N),
        None,
    )

    # 5. exact cancellation: pairs of equal-magnitude opposite products, small residuals, random acc.
    n = n_random // 8
    a = rng.standard_normal((n, M, K)) * 2.0**8
    b = rng.standard_normal((n, N, K)) * 2.0**8
    a[:, :, 1::2] = a[:, :, 0::2]
    b[:, :, 1::2] = -b[:, :, 0::2]
    small = rng.standard_normal((n, M, K)) * 2.0**-6
    mask = rng.random((n, M, K)) < 0.15
    a = np.where(mask, small, a)
    acc = rng.standard_normal((n, M, N)) * 2.0**-4
    yield "cancellation", cast(a), cast(b), f32_bits(acc.astype(np.float32)), None

    # 6. mixed magnitudes: per-element random exponents over the whole operand range.
    n = n_random // 8
    yield (
        "mixed_magnitude",
        fmt.random_finite_bits(rng, (n, M, K)),
        fmt.random_finite_bits(rng, (n, N, K)),
        f32_bits((rng.standard_normal((n, M, N)) * np.exp2(rng.integers(-60, 61, (n, M, N)))).astype(np.float32)),
        None,
    )

    # 7. near FP32 overflow: accumulators near max, the largest operands (the FP8 path cannot overflow from finite inputs; the
    # family then probes the truncation of the products below a maximal accumulator).
    n = n_random // 16
    big_lo = fmt.finite_exp_max - 9 if fmt is BF16 else fmt.finite_exp_max - 3
    yield (
        "near_overflow",
        fmt.random_finite_bits(rng, (n, M, K), big_lo, fmt.finite_exp_max),
        fmt.random_finite_bits(rng, (n, N, K), big_lo, fmt.finite_exp_max),
        random_finite_fp32_bits(rng, n * M * N, 250, 254).reshape(n, M, N),
        None,
    )

    # 8. zero handling: -0 accumulators, all-zero products, cancellations to zero.
    n = 256
    a = np.zeros((n, M, K), np.float32)
    b = np.zeros((n, N, K), np.float32)
    a[n // 2 :, :, 0] = 1.0
    a[n // 2 :, :, 1] = 1.0
    b[n // 2 :, :, 0] = 3.0
    b[n // 2 :, :, 1] = -3.0
    acc = np.where(rng.random((n, M, N)) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)
    yield "signed_zero", cast(a), cast(b), acc, None

    # 9. tiny accumulators with zero products (exponent floor probe).
    n = 512
    acc = random_finite_fp32_bits(rng, n * M * N, 0, 30).reshape(n, M, N)
    yield "tiny_acc_zero_products", np.zeros((n, M, K), fmt.dtype), np.zeros((n, N, K), fmt.dtype), acc, None

    if fmt is BF16:
        # 10. +-0 accumulators with cancelling products below 2^-145: BF16 subnormal a (30% of lanes), b in
        # [2^-27, 2^-5].  Decides whether a zero accumulator's exponent enters the alignment maximum (silicon and the simulator's
        # E_ZERO zero: no; a zero carried at -126/-127 would truncate these sums differently -- see the log).
        n = max(256, n_random // 16)
        a = ((rng.integers(0, 2, (n, M, K)) << 15) | rng.integers(1, 128, (n, M, K))).astype(np.uint16)
        a = np.where(rng.random((n, M, K)) < 0.7, np.uint16(0), a)
        b = ((rng.integers(0, 2, (n, N, K)) << 15) | (rng.integers(100, 123, (n, N, K)) << 7) | rng.integers(0, 128, (n, N, K))).astype(np.uint16)
        acc = np.where(rng.random((n, M, N)) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)
        yield "zero_acc_tiny_cancellation", a, b, acc, None

    # 11-14: the accumulator / adder families of tc-probe-fp8, for every format (pin-bf16 extended them to BF16 with the
    # format's own parameters; the FP8 draws are the same, in the same order, as before).
    yield from probe_families(rng, fmt, M, K)

    if fp8:
        yield from fp8_families(rng, fmt, M, K)


def probe_families(rng, fmt: Format, M: int, K: int):
    """Families 11-14 (tc-probe-fp8 for FP8; pin-bf16 for BF16): the accumulator pass-through, the LSB walk, the special-value
    classes and Hawkeye's adder-width probe.  The accumulator families are format-independent by construction (products are
    zero); the specials table and the adder-width exponents come from the format (``SPECIAL_OPERANDS``, ``adder_width_parameters``)."""

    def zero_ops(n):
        return (np.zeros((n, M, K), fmt.dtype), np.zeros((n, N, K), fmt.dtype))

    # 11. all products zero, random finite accumulators over the whole range: the Ada quirk is that the accumulator comes back
    # truncated to the adder's 14 bits even though nothing was added (0x3f800001 -> 0x3f800000).  Half of the tiles have zero
    # B as well, half a random finite B (products +-0 * x).
    n = 512
    acc = random_finite_fp32_bits(rng, n * M * N, 1, 254).reshape(n, M, N)
    small = random_finite_fp32_bits(rng, n * M * N, 0, 40).reshape(n, M, N)
    acc = np.where(rng.random((n, M, N)) < 0.25, small, acc)
    A, Bt = zero_ops(n)
    Bt[n // 2 :] = fmt.random_finite_bits(rng, (n // 2, N, K))
    yield "zero_products_nonzero_acc", A, Bt, acc, None

    # 12. accumulator-only LSB walk: 64 base words (sign, exponent spread over the range incl. subnormals, mantissa bits 22..10)
    # each with every pattern of the low 10 bits, all products zero: exactly which low bits survive the pass-through.
    bases = 64
    patterns = 1 << 10
    fixed = [0, 1, 2, 3, 30, 100, 120, 126, 127, 128, 140, 150, 200, 240, 253, 254]
    exps = np.concatenate([np.array(fixed, np.uint32), rng.integers(1, 255, bases - len(fixed)).astype(np.uint32)])
    sign = rng.integers(0, 2, bases).astype(np.uint32)
    high = rng.integers(0, 1 << 13, bases).astype(np.uint32)
    base_words = (sign << 31) | (exps << 23) | (high << 10)
    flat = base_words[np.arange(bases * patterns) // patterns] | (np.arange(bases * patterns) % patterns).astype(np.uint32)
    n = bases * patterns // (M * N)
    A, Bt = zero_ops(n)
    yield "acc_only_lsb_walk", A, Bt, flat.reshape(n, M, N), {"bases": [f"{w:08x}" for w in base_words]}

    # 13. specials: the format's non-finite codes (E4M3: NaN 0x7f/0xff; E5M2: inf 0x7c/0xfc, NaN 0x7d..0x7f/0xfd..0xff) and
    # non-finite / extreme accumulators, one class per row (a), per column (b) and per tile (acc, fill, k position).  Recorded,
    # never asserted: the finite model has nothing to say here.
    A, Bt, C, meta = specials_tiles(fmt, M, K)
    yield "specials", A, Bt, C, meta

    # 14. Hawkeye's adder-width probe in every k slot: +P and -P at k = 0, 1 and one residual 2^ratio * P at slot s (rows vary
    # the slot and the residual's a exponent, columns the b exponent), ratio in [-26, -5] (FP8) / [-30, -3] (BF16), with a zero
    # accumulator (variant "prod") or with the +P carried by the accumulator instead of product 0 (variant "acc").  A w-bit adder
    # keeps the residual iff ratio >= -(w - 1) when it is summed with the pair (14 bits: -13; the BF16 pipelines: 25 bits -> -24
    # on Ampere/Ada, 26 bits -> -25 on Hopper); a residual in a later group survives whatever the ratio.
    A, Bt, C, meta = adder_width_tiles(fmt, M, K)
    yield "adder_width_cancellation", A, Bt, C, meta


def fp8_families(rng, fmt: Format, M: int, K: int):
    """The FP8-only families: 15. the adder-floor probe (pin-bf16, D15(b))."""

    A, Bt, C, meta = floor_probe_tiles(fmt, M, K)
    yield "fp8_floor_probe", A, Bt, C, meta


SPECIAL_ACC = {
    "+0": 0x00000000, "-0": 0x80000000, "+one": 0x3F800000, "-one": 0xBF800000, "+inf": 0x7F800000, "-inf": 0xFF800000,
    "+qnan": 0x7FC00000, "+qnan_payload": 0x7FC12345, "-qnan": 0xFFC00000, "+snan": 0x7F800001, "+max": 0x7F7FFFFF, "+sub_min": 0x00000001,
}
SPECIAL_OPERANDS = {
    "e4m3": {
        "a": ["+0:00", "-0:80", "+sub_min:01", "+sub_max:07", "+min_normal:08", "+one:38", "-one:b8", "+max:7e", "-max:fe", "+nan:7f",
              "-nan:ff", "+two:40", "-sub_min:81", "+2^8:78", "+half:30", "-two:c0"],
        "b": ["+0:00", "-0:80", "+one:38", "-one:b8", "+max:7e", "+nan:7f", "-nan:ff", "+sub_min:01"],
        "one": 0x38,
    },
    "e5m2": {
        "a": ["+0:00", "-0:80", "+sub_min:01", "+sub_max:03", "+min_normal:04", "+one:3c", "-one:bc", "+max:7b", "-max:fb", "+inf:7c",
              "-inf:fc", "+nan:7d", "+nan_q:7f", "-nan:fd", "+two:40", "+2^15:78"],
        "b": ["+0:00", "-0:80", "+one:3c", "-one:bc", "+inf:7c", "-inf:fc", "+nan:7e", "+max:7b"],
        "one": 0x3C,
    },
    # BF16 (pin-bf16): the same roles as the FP8 tables in BF16 encodings.  1.0 = 0x3F80; subnormals 0x0001..0x007F; quiet NaN 0x7FC0.
    "bf16": {
        "a": ["+0:0000", "-0:8000", "+sub_min:0001", "+sub_max:007f", "+min_normal:0080", "+one:3f80", "-one:bf80", "+max:7f7f",
              "-max:ff7f", "+inf:7f80", "-inf:ff80", "+nan:7fc0", "-nan:ffc0", "+two:4000", "-sub_min:8001", "+2^64:5f80"],
        "b": ["+0:0000", "-0:8000", "+one:3f80", "-one:bf80", "+inf:7f80", "-inf:ff80", "+nan:7fc0", "+max:7f7f"],
        "one": 0x3F80,
    },
}


def specials_tiles(fmt: Format, M: int, K: int):
    spec = SPECIAL_OPERANDS[fmt.name]
    a_cls = [(s.split(":")[0], int(s.split(":")[1], 16)) for s in spec["a"]]
    b_cls = [(s.split(":")[0], int(s.split(":")[1], 16)) for s in spec["b"]]
    fills = [("0*0", 0x00, 0x00), ("1*1", spec["one"], spec["one"])]
    positions = [0, K - 1]
    tiles = [(an, fn, kp) for an in SPECIAL_ACC for fn, _, _ in fills for kp in positions]
    n = len(tiles)
    A = np.zeros((n, M, K), fmt.dtype)
    Bt = np.zeros((n, N, K), fmt.dtype)
    C = np.zeros((n, M, N), np.uint32)
    for s, (an, fn, kp) in enumerate(tiles):
        _, fa, fb = next(f for f in fills if f[0] == fn)
        A[s] = fa
        Bt[s] = fb
        for i in range(M):
            A[s, i, kp] = a_cls[i % len(a_cls)][1]
        for j in range(N):
            Bt[s, j, kp] = b_cls[j][1]
        C[s] = SPECIAL_ACC[an]
    meta = {"tiles": [{"acc": an, "fill": fn, "k": kp} for an, fn, kp in tiles], "a_rows": [c[0] for c in a_cls], "b_cols": [c[0] for c in b_cls]}
    return A, Bt, C, meta


def adder_width_parameters(fmt: Format) -> tuple[tuple[int, ...], list[int]]:
    """``(pair exponents pa, ratios)`` of the adder-width probe.  The pair term is P = 2^(2 pa) and the residual 2^(2 pa + ratio).

    FP8 (tc-probe-fp8, unchanged): pa in {0, mid, max} -- unit, mid-range and the largest square -- ratios -26..-5 around the
    14-bit boundary (-13/-14).  BF16 (pin-bf16): the pipelines carry 25 (Ampere/Ada) or 26 (Hopper) bits, so the ratios run
    -30..-3 to bracket both boundaries with margin; pa in {-55, 0, 32, 63} keeps every 2^pe, 2^(pe+ratio) a finite FP32 (pe <= 126,
    pe + ratio >= -140, i.e. down into the subnormals, where a survived residual comes back as an exact subnormal word)."""

    hi = fmt.max_pow2_exp
    if fmt.bits == 8:
        return (0, (hi + 1) // 2, hi), list(range(-26, -4))
    return (-55, 0, 32, 63), list(range(-30, -2))


def adder_width_tiles(fmt: Format, M: int, K: int):
    lo, hi = fmt.min_pow2_exp, fmt.max_pow2_exp
    pas, ratios = adder_width_parameters(fmt)
    cases = []  # (variant, pe, ratio_base, slot_base)
    for variant in ("prod", "acc"):
        for pa in pas:  # pair exponent pe = 2 * pa
            for r0 in range(ratios[0], ratios[-1] + 1, N):
                for slot_base in range(2, K, M):
                    cases.append((variant, pa, r0, slot_base))
    A = np.zeros((len(cases), M, K), fmt.dtype)
    Bt = np.zeros((len(cases), N, K), fmt.dtype)
    C = np.zeros((len(cases), M, N), np.uint32)
    elements = []  # per tile: list of (variant, pe, ratio, slot) per (i, j) or None where the exponents do not fit
    for s, (variant, pa, r0, slot_base) in enumerate(cases):
        pe = 2 * pa
        info = [[None] * N for _ in range(M)]
        # The residual 2^(pe + ratio) = 2^up * 2^fl: the a exponent `up` is one per tile (a row is shared by the eight columns), the
        # b exponent fl = pe + ratio - up carries the column's ratio; `up` is placed so that the eight fl land mid-range.
        up = max(lo, min(hi, pe + r0 + N // 2 - (lo + hi) // 2))
        for j in range(N):
            ratio = r0 + j
            fl = pe + ratio - up
            if not (lo <= fl <= hi):
                continue
            Bt[s, j, 0] = fmt.pow2(pa)
            Bt[s, j, 1] = fmt.pow2(pa, negative=True)
            Bt[s, j, 2:] = fmt.pow2(fl)
            for i in range(M):
                slot = 2 + (slot_base - 2 + i) % (K - 2)
                if variant == "prod":
                    A[s, i, 0] = fmt.pow2(pa)
                else:
                    C[s, i, j] = f32_bits(np.array([2.0**pe], np.float32))[0]
                A[s, i, 1] = fmt.pow2(pa)
                A[s, i, slot] = fmt.pow2(up)
                info[i][j] = (variant, pe, ratio, slot)
        elements.append(info)
    return A, Bt, C, {"cases": cases, "elements": elements}


def adder_width_summary(meta: dict, D: np.ndarray) -> dict:
    """Per variant and slot: whether the residual survived (D == 2^(pe+ratio)), was truncated (D == 0) or came back as something
    else (written relative to the pair term P = 2^pe, so the three pair exponents fold), folded to the most negative surviving
    ratio and the least negative truncated one; a ratio whose verdict differs between pair exponents is listed as inconsistent."""

    table: dict[tuple, dict[int, set]] = {}
    for s, info in enumerate(meta["elements"]):
        for i, row in enumerate(info):
            for j, e in enumerate(row):
                if e is None:
                    continue
                variant, pe, ratio, slot = e
                want = int(f32_bits(np.array([2.0 ** (pe + ratio)], np.float32))[0])
                got = int(D[s, i, j])
                if got == want:
                    verdict = "survived"
                elif got == 0:
                    verdict = "truncated"
                else:
                    value = float(np.array([got], np.uint32).view(np.float32)[0])
                    if value != 0 and np.isfinite(value) and math.log2(abs(value)).is_integer():
                        verdict = f"{'-' if value < 0 else ''}2^{int(math.log2(abs(value))) - pe:+d}*P"
                    else:
                        verdict = f"{got:08x}"
                table.setdefault((variant, slot), {}).setdefault(ratio, set()).add(verdict)
    out: dict[str, dict] = {}
    for (variant, slot), by_ratios in sorted(table.items()):
        by_ratio = {r: (next(iter(v)) if len(v) == 1 else "inconsistent(" + ",".join(sorted(v)) + ")") for r, v in by_ratios.items()}
        surv = [r for r, v in by_ratio.items() if v == "survived"]
        trunc = [r for r, v in by_ratio.items() if v == "truncated"]
        other = {r: v for r, v in by_ratio.items() if v not in ("survived", "truncated")}
        out.setdefault(variant, {})[str(slot)] = {
            "min_surviving_ratio": min(surv) if surv else None,
            "max_truncated_ratio": max(trunc) if trunc else None,
            "other": {str(r): v for r, v in other.items()},
        }
    return out


def specials_summary(meta: dict, D: np.ndarray, fmt: Format) -> dict:
    words, counts = np.unique(D, return_counts=True)
    order = np.argsort(-counts)
    table = {}
    for s, t in enumerate(meta["tiles"]):
        table[f"acc={t['acc']} fill={t['fill']} k={t['k']}"] = [[f"{int(D[s, i, j]):08x}" for j in range(N)] for i in range(min(D.shape[1], len(meta["a_rows"])))]
    return {
        "distinct_output_words": {f"{int(words[i]):08x}": int(counts[i]) for i in order[:32]},
        "a_rows": meta["a_rows"],
        "b_cols": meta["b_cols"],
        "words_by_tile": table,
    }


FLOOR_PROBE_RANGE = list(range(-152, -125))  # the floors the H100 fit left open (D15(b)): every one is evaluated at run time

# accumulator classes of the floor probe, one per column: (name, FP32 word)
FLOOR_PROBE_ACC = [
    ("+0", 0x00000000), ("+2^-149", 0x00000001), ("-2^-149", 0x80000001), ("+2^-126", 0x00800000),
    ("-max_sub", 0x807FFFFF), ("+2^-140", 0x00000200), ("+2^-139", 0x00000400), ("+2^-133", 0x00010000),
]
FLOOR_PROBE_COUNTS = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32]


def floor_probe_tiles(fmt: Format, M: int, K: int):
    """D15(b): tiles on which an adder floor f in ``FLOOR_PROBE_RANGE`` could show, if it can show at all.  Rows and columns
    are read by ``floor_probe_summary`` through ``meta['tiles']``:

    acc_bits          zero products; every single accumulator bit +-2^b, b in 0..23 (2^-149 .. 2^-126), then random subnormal
                      words (exponent field 0), then random tiny normals (fields 1..3); one copy with random finite B (+-0 * x)
    min_products      n same-sign minimal products p = 2^(2 min_pow2) (E4M3: 2^-18; E5M2: 2^-32), n in FLOOR_PROBE_COUNTS by row,
                      against the tiny accumulator classes ``FLOOR_PROBE_ACC`` by column; positive and negative variants
    cancelling_pair   +p - p against the same tiny accumulators (does the accumulator come back, or the pair's zero?)
    product_window    one product p against accumulators p * 2^-k, k in 0..23 (the adder's window below a sub-unit product)

    Deterministic (a private generator seeded by the format): a pure function of (fmt, M, K)."""

    prng = np.random.default_rng(0xD15B + fmt.bits)
    a_min, b_min = fmt.pow2(fmt.min_pow2_exp), fmt.pow2(fmt.min_pow2_exp)
    a_min_neg = fmt.pow2(fmt.min_pow2_exp, negative=True)
    p_exp = 2 * fmt.min_pow2_exp
    tiles: list[dict] = []
    As, Bts, Cs = [], [], []

    def designed_acc(n_words: int) -> np.ndarray:
        words = []
        for b in range(24):
            words += [1 << b, (1 << 31) | (1 << b)]
        rest = n_words - len(words)
        subs = prng.integers(0, 1 << 23, rest // 2).astype(np.uint32) | (prng.integers(0, 2, rest // 2).astype(np.uint32) << 31)
        tiny = random_finite_fp32_bits(prng, rest - rest // 2, 1, 3)
        return np.concatenate([np.array(words, np.uint32), subs, tiny])[:n_words]

    # acc_bits: two zero-product tiles and one with random finite B.
    for variant in ("zero_b", "zero_b", "random_b"):
        A = np.zeros((M, K), fmt.dtype)
        Bt = np.zeros((N, K), fmt.dtype) if variant == "zero_b" else fmt.random_finite_bits(prng, (N, K))
        C = designed_acc(M * N).reshape(M, N)
        As.append(A); Bts.append(Bt); Cs.append(C)
        tiles.append({"kind": "acc_bits", "variant": variant, "expect": "D == C"})

    # min_products: row i has n_i minimal products of one sign, column j the accumulator class.
    for sign, a_word in (("+", a_min), ("-", a_min_neg)):
        A = np.zeros((M, K), fmt.dtype)
        Bt = np.zeros((N, K), fmt.dtype)
        C = np.zeros((M, N), np.uint32)
        counts = []
        for i in range(M):
            n_i = FLOOR_PROBE_COUNTS[i % len(FLOOR_PROBE_COUNTS)]
            counts.append(n_i)
            A[i, :n_i] = a_word
        Bt[:, :] = b_min
        for j, (_, w) in enumerate(FLOOR_PROBE_ACC):
            C[:, j] = w
        As.append(A); Bts.append(Bt); Cs.append(C)
        expect = [int(f32_bits(np.array([(-1 if sign == "-" else 1) * n * 2.0**p_exp], np.float32))[0]) for n in counts]
        tiles.append({"kind": "min_products", "sign": sign, "counts": counts, "acc_cols": [a for a, _ in FLOOR_PROBE_ACC],
                      "expect_rows": [f"{w:08x}" for w in expect], "expect": "D == n * p regardless of the accumulator"})

    # cancelling_pair: +p at k=0, -p at k=1.
    A = np.zeros((M, K), fmt.dtype)
    Bt = np.zeros((N, K), fmt.dtype)
    C = np.zeros((M, N), np.uint32)
    A[:, 0] = a_min
    A[:, 1] = a_min_neg
    Bt[:, 0] = b_min
    Bt[:, 1] = b_min
    for j, (_, w) in enumerate(FLOOR_PROBE_ACC):
        C[:, j] = w
    As.append(A); Bts.append(Bt); Cs.append(C)
    tiles.append({"kind": "cancelling_pair", "acc_cols": [a for a, _ in FLOOR_PROBE_ACC], "expect": "D == +0 (the accumulator is below the pair's window)"})

    # product_window: one product p in row i (k = 0), accumulator p * 2^-k with k = (i * N + j) % 24.
    A = np.zeros((M, K), fmt.dtype)
    Bt = np.zeros((N, K), fmt.dtype)
    C = np.zeros((M, N), np.uint32)
    A[:, 0] = a_min
    Bt[:, 0] = b_min
    ks = [[(i * N + j) % 24 for j in range(N)] for i in range(M)]
    for i in range(M):
        for j in range(N):
            C[i, j] = f32_bits(np.array([2.0 ** (p_exp - ks[i][j])], np.float32))[0]
    As.append(A); Bts.append(Bt); Cs.append(C)
    tiles.append({"kind": "product_window", "k": ks, "expect": "D == p + p*2^-k while k is inside the adder's window, else p"})

    meta = {"tiles": tiles, "product_exponent": p_exp, "floors": FLOOR_PROBE_RANGE}
    return np.stack(As), np.stack(Bts), np.stack(Cs), meta


def floor_probe_summary(meta: dict, D: np.ndarray) -> dict:
    """Per tile kind: how many words match the designed expectation (independent of any model), and the product_window's
    survive/truncate boundary in k."""

    out: dict[str, Any] = {}
    p_exp = meta["product_exponent"]
    for s, t in enumerate(meta["tiles"]):
        d = D[s]
        if t["kind"] == "acc_bits":
            out.setdefault("acc_bits", []).append({"variant": t["variant"], "elements": int(d.size)})
        elif t["kind"] == "min_products":
            expect = np.array([int(w, 16) for w in t["expect_rows"]], np.uint32)[:, None]
            out[f"min_products{t['sign']}"] = {"elements": int(d.size), "elements_equal_to_n_p": int((d == expect).sum()),
                                               "distinct_words_per_row_max": int(max(len(set(d[i].tolist())) for i in range(d.shape[0])))}
        elif t["kind"] == "cancelling_pair":
            out["cancelling_pair"] = {"elements": int(d.size), "elements_equal_to_plus_zero": int((d == 0).sum()),
                                      "words_by_acc": {a: sorted({f"{int(w):08x}" for w in d[:, j]}) for j, a in enumerate(t["acc_cols"])}}
        elif t["kind"] == "product_window":
            p = 2.0**p_exp
            surv, trunc = [], []
            for i, row in enumerate(t["k"]):
                for j, k in enumerate(row):
                    want_sum = int(f32_bits(np.array([p + p * 2.0**-k], np.float32))[0])
                    want_p = int(f32_bits(np.array([p], np.float32))[0])
                    got = int(d[i, j])
                    (surv if got == want_sum else trunc if got == want_p else []).append(k)
            out["product_window"] = {"max_surviving_k": max(surv) if surv else None, "min_truncated_k": min(trunc) if trunc else None,
                                     "unclassified": int(d.size - len(surv) - len(trunc))}
    return out


# --- the models, per tile, in processes ----------------------------------------------------------------------------------------------

_MODELS: list[Model] = []


def family_asserted(ins: Instruction, name: str) -> bool:
    """Whether a family's mismatches count for validation: every family, except ``specials`` unless every declared model is total
    (a hypothesis's counts are recorded per family either way)."""

    return name not in UNASSERTED_FAMILIES or all(m.total for m in ins.models if m.declared)


def model_descs(ins: Instruction) -> list[tuple]:
    """The picklable form of ``ins.models`` the worker processes rebuild them from (:func:`_init`)."""

    return [(m.pipeline.name, m.product, m.declared, m.lowering, m.total) for m in ins.models]


def _init(model_descs):
    global _MODELS
    _MODELS = [Model(TCM.MODELS[pname], prod, declared, lowering, total) for pname, prod, declared, lowering, total in model_descs]


def _eval_tiles(args):
    A, Bt, C = args
    n, M, _ = A.shape
    outs = [np.zeros((n, M, N), np.uint32) for _ in _MODELS]
    errs = [np.zeros((n, M, N), np.bool_) for _ in _MODELS]
    for t in range(n):
        a_rows = [A[t, i].tolist() for i in range(M)]
        b_rows = [Bt[t, j].tolist() for j in range(N)]
        for i in range(M):
            for j in range(N):
                acc = int(C[t, i, j])
                for m, model in enumerate(_MODELS):
                    try:
                        outs[m][t, i, j] = model.evaluate(acc, a_rows[i], b_rows[j])
                    except Exception:  # noqa: BLE001 - domain errors are counted, not raised
                        errs[m][t, i, j] = True
    return outs, errs


def model_eval(pool, A, Bt, C, procs, worker=_eval_tiles):
    n = A.shape[0]
    chunks = max(1, min(n // 64 + 1, procs * 8))
    idx = np.array_split(np.arange(n), chunks)
    parts = pool.map(worker, [(A[i], Bt[i], C[i]) for i in idx])
    if worker not in (_eval_tiles, _eval_chain_tiles):  # the alternative evaluators return one (words, errors) pair
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    n_models = len(parts[0][0])
    return [(np.concatenate([p[0][m] for p in parts]), np.concatenate([p[1][m] for p in parts])) for m in range(n_models)]


def _eval_zero_alternatives(args):
    """For zero accumulators: the word the source-close evaluator gives when the zero is carried at exponent -126 / -127 instead of
    E_ZERO (the two arrays returned).  Elements with a nonzero accumulator get the ordinary result in both."""

    from verity_numerical.reference import hawkeye_close as H

    A, Bt, C = args
    n, M, _ = A.shape
    out = [np.zeros((n, M, N), np.uint32) for _ in range(2)]
    for t in range(n):
        for i in range(M):
            for j in range(N):
                acc_bits = int(C[t, i, j])
                for k, zero_exp in enumerate((-126, -127)):
                    if acc_bits & 0x7FFFFFFF:
                        g = H.fp32_to_gfloat(acc_bits)
                    else:
                        g = H.Gfloat(bool(acc_bits >> 31), zero_exp, 0)
                    for start in (0, 8):
                        g = H.group_sum([g] + [H.multiply_bfloat16_to_gfloat(int(A[t, i, l]), int(Bt[t, j, l])) for l in range(start, start + 8)])
                    out[k][t, i, j] = H.gfloat_to_float_bits(g)
    return out[0], out[1]


def _eval_floor_alternatives(args):
    """The declared pipeline's words at every floor of ``_FLOORS`` (set by ``_init_floors``): one (n, M, 8) array per floor."""

    A, Bt, C = args
    n, M, _ = A.shape
    outs = np.zeros((len(_FLOOR_MODELS), n, M, N), np.uint32)
    for t in range(n):
        a_rows = [A[t, i].tolist() for i in range(M)]
        b_rows = [Bt[t, j].tolist() for j in range(N)]
        for i in range(M):
            for j in range(N):
                acc = int(C[t, i, j])
                for f, model in enumerate(_FLOOR_MODELS):
                    outs[f, t, i, j] = model.evaluate(acc, a_rows[i], b_rows[j])
    return outs, np.zeros((n, M, N), np.bool_)


_FLOOR_MODELS: list[Model] = []


def _init_floors(desc, floors):
    global _FLOOR_MODELS
    import dataclasses

    pname, prod, declared, lowering = desc
    base = TCM.MODELS[pname]
    _FLOOR_MODELS = [Model(dataclasses.replace(base, name=f"{base.name}@floor{f}", zero_exponent=f), prod, declared, lowering) for f in floors]


def floor_alternatives(pool, model: Model, A, Bt, C, D, declared_words, procs: int, floors: list[int]) -> dict:
    """D15(b): the declared pipeline evaluated with ``zero_exponent`` set to every floor in ``floors`` on the floor-probe tiles;
    per floor, how many words differ from the pinned floor's and from the hardware's.  All zeros = no floor in the range is
    observable on these tiles (the structural reason is in discrepancy-log entry D15: with any nonzero term the group's maximum
    exponent is at least -126, the FP32 subnormal grid, above every floor here; with no nonzero term the sum is zero regardless)."""

    desc = (model.pipeline.name, model.product, model.declared, model.lowering)
    with mp.Pool(min(procs, 8), initializer=_init_floors, initargs=(desc, floors)) as fpool:
        n = A.shape[0]
        chunks = max(1, min(n, 8))
        idx = np.array_split(np.arange(n), chunks)
        parts = fpool.map(_eval_floor_alternatives, [(A[i], Bt[i], C[i]) for i in idx])
    words = np.concatenate([p[0] for p in parts], axis=1)
    per_floor = {}
    for f, floor in enumerate(floors):
        per_floor[str(floor)] = {"words_differing_from_pinned_floor": int(np.count_nonzero(words[f] != declared_words)),
                                 "mismatches_vs_hardware": int(np.count_nonzero(words[f] != D))}
    return {"pinned_floor": model.pipeline.zero_exponent, "floors": floors, "elements": int(D.size), "per_floor": per_floor,
            "all_floors_identical": all(v["words_differing_from_pinned_floor"] == 0 for v in per_floor.values()),
            "all_floors_match_hardware": all(v["mismatches_vs_hardware"] == 0 for v in per_floor.values())}


# --- records ----------------------------------------------------------------------------------------------------------------------


def hexrec(acc, a, b, d, width: int = 4) -> dict:
    return {
        "acc": f"{int(acc):08x}",
        "a": "".join(f"{int(x):0{width}x}" for x in a),
        "b": "".join(f"{int(x):0{width}x}" for x in b),
        "d": f"{int(d):08x}",
    }


def words(hex_text: str, width: int = 4) -> list[int]:
    return [int(hex_text[i : i + width], 16) for i in range(0, len(hex_text), width)]


def layout_check(probe: Probe, ins: Instruction) -> dict:
    """Exact small-integer tiles: D must be C + A.Bt computed exactly (every sum fits the adder without any truncation)."""

    rng = np.random.default_rng(1)
    n = 64
    M, K, fmt = ins.m, ins.k, ins.fmt
    lim, acc_lim = (8, 1000) if fmt is BF16 else (3, 8)  # BF16 sums < 2^24 in a 25-bit adder; FP8 sums < 2^9 in a 14-bit one
    a = rng.integers(-lim, lim + 1, (n, M, K)).astype(np.float32)
    b = rng.integers(-lim, lim + 1, (n, N, K)).astype(np.float32)
    c = rng.integers(-acc_lim, acc_lim + 1, (n, M, N)).astype(np.float32)
    want = f32_bits((c + np.einsum("nik,njk->nij", a, b)).astype(np.float32))
    candidates = [None] if probe.descriptor is None else [(128, 256), (256, 128)]
    for cand in candidates:
        if cand is not None:
            probe.set_descriptor(*cand)
        got = probe.tiles(fmt.from_f32(a), fmt.from_f32(b), f32_bits(c))
        if np.array_equal(want, got):
            return {"status": "passed", "elements": n * M * N, "descriptor": probe.descriptor}
    raise RuntimeError(f"fragment layout check failed: {probe.kernel} does not compute C + A.Bt on exact integer tiles "
                       f"({int(np.count_nonzero(want != got))} of {n * M * N} words differ; descriptor {probe.descriptor})")


def special_all_finite_records(path: Path) -> list[dict]:
    """The all-finite records (finite acc, finite a and b words) of a ``tc-total-2026-09-07/<device>/mma_special.reduced.json.gz``
    file, in the golden-record shape: the ``cases`` list and the ``pos`` list (one measured word each)."""

    import gzip

    data = json.loads(gzip.open(path).read())
    assert data["instruction"] == "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
    out = []
    for rec in data["cases"]:
        out.append({"acc": rec["acc"], "a": rec["a"], "b": rec["b"], "d": rec["d"], "family": f"cases/{rec['family']}"})
    for rec in data["pos"]:
        assert len(rec["d"]) == 1, rec["name"]
        out.append({"acc": rec["acc"], "a": rec["a"], "b": rec["b"], "d": rec["d"][0], "family": "pos"})
    finite = []
    for rec in out:
        if (int(rec["acc"], 16) >> 23) & 0xFF == 0xFF:
            continue
        if any((w >> 7) & 0xFF == 0xFF for w in (*words(rec["a"]), *words(rec["b"]))):
            continue
        finite.append(rec)
    return finite


def hopper_capture_records(path: Path) -> list[dict]:
    """Every record of veritor's H100 BF16 wgmma capture (``tc-hopper-2026-09-07/special.reduced.json.gz``: 14 families, the
    non-finite ones included), in the golden-record shape with ``family`` set."""

    import gzip

    data = json.loads(gzip.open(path).read())
    ins = data["kernel_identity"]["instruction"]
    assert ins.startswith("wgmma.mma_async.sync.aligned.m64n") and ins.endswith("k16.f32.bf16.bf16"), ins
    return [{"acc": r["acc"], "a": r["a"], "b": r["b"], "d": r["d"], "family": fam} for fam, recs in data["families"].items() for r in recs]


def replay_golden(probe: Probe, path: Path, records: list[dict] | None = None, pipeline: str | None = None,
                  model: Model | None = None) -> dict:
    """Every record at its own tile position (r % M, (r // M) % 8), the rest of the tile zero.  With ``model``, the hardware word
    is also compared with the model's (``model_mismatches``), and both counts are broken down by the records' ``family``."""

    ins = probe.ins
    M, K, fmt = ins.m, ins.k, ins.fmt
    if records is None:
        data = json.loads(path.read_text())
        records, pipeline = data["records"], data["pipeline"]
        if data.get("k", K) != K or data.get("operand_bits", fmt.bits) != fmt.bits:
            raise RuntimeError(f"{path} holds k={data.get('k')} / {data.get('operand_bits')}-bit records; {ins.id} takes k={K} / {fmt.bits}-bit")
    n = len(records)
    A = np.zeros((n, M, K), fmt.dtype)
    Bt = np.zeros((n, N, K), fmt.dtype)
    C = np.zeros((n, M, N), np.uint32)
    for r, rec in enumerate(records):
        i, j = r % M, (r // M) % N
        A[r, i] = words(rec["a"], fmt.hex_width)
        Bt[r, j] = words(rec["b"], fmt.hex_width)
        C[r, i, j] = int(rec["acc"], 16)
    D = probe.tiles(A, Bt, C)
    mism = []
    model_mism = []
    by_family: dict[str, dict] = {}
    other_nonzero = 0
    for r, rec in enumerate(records):
        i, j = r % M, (r // M) % N
        hw = int(D[r, i, j])
        fam = by_family.setdefault(rec.get("family", "records"), {"records": 0, "mismatches": 0, "model_mismatches": 0})
        fam["records"] += 1
        if hw != int(rec["d"], 16):
            mism.append({**rec, "hardware": f"{hw:08x}"})
            fam["mismatches"] += 1
        if model is not None:
            try:
                want = model.evaluate(int(rec["acc"], 16), words(rec["a"], fmt.hex_width), words(rec["b"], fmt.hex_width))
            except Exception:  # noqa: BLE001 - a domain error is a mismatch
                want = None
            if want != hw:
                model_mism.append({**rec, "hardware": f"{hw:08x}", "model": "error" if want is None else f"{want:08x}"})
                fam["model_mismatches"] += 1
        # every other output has a zero accumulator and a zero row or column: +0, except that a zero times a non-finite operand
        # is NaN, so row i reads the canonical NaN where a[i] holds an infinity or NaN, and column j where b[j] does
        rest = D[r].copy()
        rest[i, j] = 0
        if not fmt.is_finite(A[r, i]).all():
            rest[i, :] ^= np.where(np.arange(N) != j, np.uint32(T.FP32_CANONICAL_NAN), np.uint32(0))
        if not fmt.is_finite(Bt[r, j]).all():
            rest[:, j] ^= np.where(np.arange(M) != i, np.uint32(T.FP32_CANONICAL_NAN), np.uint32(0))
        other_nonzero += int(np.count_nonzero(rest))
    out = {
        "file": str(path),
        "pipeline": pipeline,
        "records": n,
        "nonzero_accumulators": int(sum(1 for rec in records if int(rec["acc"], 16) & 0x7FFFFFFF)),
        "mismatches": len(mism),
        "first_mismatches": mism[:FIRST_MISMATCHES],
        "other_positions_nonzero": other_nonzero,
    }
    if model is not None:
        out.update(model=model.name, model_mismatches=len(model_mism), first_model_mismatches=model_mism[:FIRST_MISMATCHES])
    if len(by_family) > 1:
        out["families"] = by_family
    return out


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replay_instances(probe: Probe, root: Path, manifest_path: Path | None = None) -> dict:
    """Replay ``bench-instances/v1`` on the device (BF16 m16n8k16 only): the ``tu-k16`` tier as one launch (instance ``r`` at tile
    ``r``, position ``(r % 16, (r // 16) % 8)``, the rest of the tile zero) and the ``vu-k1536`` tier as 96 dependent launches (all
    instances per launch; the hardware's D of step ``s`` is the C of step ``s + 1``).  Every outgoing accumulator word is compared
    with the dataset's ``tu-k16.out.u32`` / ``vu-k1536.acc.u32``; the final accumulator, cast FP32 -> BF16 RNE on the CPU (the
    epilogue is not a tensor-core operation), with ``vu-k1536.y.u16``.  Before anything runs, every array's sha256 is checked
    against ``manifest_path`` (default ``<root>/manifest.json``): a replay of arrays other than the pinned ones is refused."""

    from verity.ml.circuits.silicon.cast import f32_to_bf16_word
    from verity_numerical.bench import instances as I

    if probe.ins.fmt is not BF16 or probe.ins.m != 16:
        raise RuntimeError("--replay-instances is a BF16 m16n8k16 replay")
    K = 16
    manifest_path = manifest_path or root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digests = {"manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path), "checked": 0, "mismatched": []}
    for tier in ("tu-k16", "vu-k1536"):
        for name, rec in manifest["tiers"][tier]["files"].items():
            digests["checked"] += 1
            if _sha256(root / name) != rec["sha256"]:
                digests["mismatched"].append(name)
    if digests["mismatched"]:
        raise RuntimeError(f"instance arrays under {root} differ from {manifest_path}: {digests['mismatched']}")

    report: dict = {"dataset": manifest["dataset"], "root": str(root), "digests": digests}

    # tu-k16: one launch
    tu = I.load("tu-k16", root)
    n = tu.n
    A = np.zeros((n, 16, K), np.uint16)
    Bt = np.zeros((n, 8, K), np.uint16)
    C = np.zeros((n, 16, 8), np.uint32)
    a = np.frombuffer(tu.a.tobytes(), dtype=np.uint16).reshape(n, K)
    b = np.frombuffer(tu.b.tobytes(), dtype=np.uint16).reshape(n, K)
    c = np.frombuffer(tu.c.tobytes(), dtype=np.uint32)
    want = np.frombuffer(tu.out.tobytes(), dtype=np.uint32)
    rows = np.arange(n) % 16
    cols = (np.arange(n) // 16) % 8
    A[np.arange(n), rows] = a
    Bt[np.arange(n), cols] = b
    C[np.arange(n), rows, cols] = c
    t0 = time.time()
    D = probe.tiles(A, Bt, C)
    got = D[np.arange(n), rows, cols]
    rest = D.copy()
    rest[np.arange(n), rows, cols] = 0
    bad = np.nonzero(got != want)[0]
    report["tu-k16"] = {
        "instances": int(n),
        "mismatches": int(bad.size),
        "other_positions_nonzero": int(np.count_nonzero(rest)),
        "first_mismatches": [{"id": int(r), **hexrec(c[r], a[r], b[r], got[r]), "expected": f"{int(want[r]):08x}"} for r in bad[:5]],
        "gpu_seconds": round(time.time() - t0, 2),
    }
    print("replay tu-k16", json.dumps({k: v for k, v in report["tu-k16"].items() if k != "first_mismatches"}), flush=True)

    # vu-k1536: 96 dependent launches
    vu = I.load("vu-k1536", root)
    n, steps = vu.n, vu.K // K
    x = np.frombuffer(vu.x.tobytes(), dtype=np.uint16).reshape(n, steps, K)
    w = np.frombuffer(vu.w.tobytes(), dtype=np.uint16).reshape(n, steps, K)
    acc = np.frombuffer(vu.acc.tobytes(), dtype=np.uint32).reshape(n, steps)
    y = np.frombuffer(vu.y.tobytes(), dtype=np.uint16)
    rows = np.arange(n) % 16
    cols = (np.arange(n) // 16) % 8
    carried = np.zeros(n, np.uint32)  # c_0 = +0
    per_step = []
    first = []
    other_nonzero = 0
    t0 = time.time()
    for s in range(steps):
        A = np.zeros((n, 16, K), np.uint16)
        Bt = np.zeros((n, 8, K), np.uint16)
        C = np.zeros((n, 16, 8), np.uint32)
        A[np.arange(n), rows] = x[:, s]
        Bt[np.arange(n), cols] = w[:, s]
        C[np.arange(n), rows, cols] = carried
        D = probe.tiles(A, Bt, C)
        got = D[np.arange(n), rows, cols]
        rest = D.copy()
        rest[np.arange(n), rows, cols] = 0
        other_nonzero += int(np.count_nonzero(rest))
        bad = np.nonzero(got != acc[:, s])[0]
        per_step.append(int(bad.size))
        for r in bad[: max(0, 5 - len(first))]:
            first.append({"id": int(r), "step": s, **hexrec(carried[r], x[r, s], w[r, s], got[r]), "expected": f"{int(acc[r, s]):08x}"})
        carried = got.copy()  # the hardware's own accumulator carries on (mismatches propagate, as they would in a real chain)
    y_hw = np.array([f32_to_bf16_word(int(v)) for v in carried], dtype=np.uint16)
    y_bad = np.nonzero(y_hw != y)[0]
    report["vu-k1536"] = {
        "instances": int(n),
        "steps": int(steps),
        "accumulator_words_compared": int(n * steps),
        "accumulator_mismatches": int(sum(per_step)),
        "steps_with_mismatches": {str(s): m for s, m in enumerate(per_step) if m},
        "y_compared": int(n),
        "y_mismatches": int(y_bad.size),
        "first_y_mismatches": [{"id": int(r), "hardware": f"{int(y_hw[r]):04x}", "expected": f"{int(y[r]):04x}"} for r in y_bad[:5]],
        "other_positions_nonzero": other_nonzero,
        "first_mismatches": first,
        "epilogue": "f32_to_bf16_word (RNE) applied on the CPU to the hardware's final accumulator",
        "gpu_seconds": round(time.time() - t0, 2),
    }
    print("replay vu-k1536", json.dumps({k: v for k, v in report["vu-k1536"].items() if k not in ("first_mismatches", "first_y_mismatches")}), flush=True)
    return report


# --- the sweep ---------------------------------------------------------------------------------------------------------------------


def sweep(probe: Probe, ins: Instruction, args, out: Path, identity: dict) -> tuple[dict, dict]:
    M, K, fmt = ins.m, ins.k, ins.fmt
    only = set(args.families.split(",")) if args.families else None
    rng = np.random.default_rng(args.seed)
    model_names = [m.name for m in ins.models]
    capture = {
        "instruction": ins.id,
        "pipeline": ins.models[0].pipeline.name,
        "models": model_names,
        "dtype": fmt.name,
        "k": K,
        "m": M,
        "operand_bits": fmt.bits,
        "note": f"acc/d are FP32 bit patterns (hex); a/b are {K} {fmt.name} words (hex) as measured on the device in identity",
        "identity": identity,
        "records": [],
    }
    fams: dict = {}
    with mp.Pool(args.procs, initializer=_init, initargs=(model_descs(ins),)) as pool:
        for name, A, Bt, C, meta in families(rng, args.n_random, fmt, M, K):
            if args.no_input_d:
                C = np.zeros_like(C)
            n_sample = min(args.golden_per_family, A.shape[0] * M * N)
            if only is not None and name not in only:
                rng.choice(A.shape[0] * M * N, size=n_sample, replace=False)  # the skipped family's sampling draw: --families leaves the others' tiles unchanged
                continue
            t0 = time.time()
            D = probe.tiles(A, Bt, C)
            t_gpu = time.time() - t0
            t0 = time.time()
            results = model_eval(pool, A, Bt, C, args.procs)
            t_model = time.time() - t0
            asserted = family_asserted(ins, name)
            fam = {
                "tiles": int(A.shape[0]),
                "elements": int(A.shape[0]) * M * N,
                "nonzero_accumulators": int(np.count_nonzero(C & 0x7FFFFFFF)),
                "asserted": asserted,
                "models": {},
                "gpu_seconds": round(t_gpu, 2),
                "model_seconds": round(t_model, 1),
                "first_mismatches": [],
            }
            any_mism = np.zeros(D.shape, np.bool_)
            for mname, (model, err) in zip(model_names, results):
                mism = (model != D) | err
                any_mism |= mism
                fam["models"][mname] = {"mismatches": int(mism.sum()), "domain_errors": int(err.sum())}
            # the legacy bench-result/v1 field names (trust.py reads them): the first (declared) model's count
            fam["mismatches_vs_model"] = fam["models"][model_names[0]]["mismatches"]
            fam["model_domain_errors"] = fam["models"][model_names[0]]["domain_errors"]
            for t, i, j in list(zip(*np.nonzero(any_mism)))[:FIRST_MISMATCHES]:
                rec = {"tile": int(t), "i": int(i), "j": int(j), **hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j], fmt.hex_width), "model": {}}
                for mname, (model, err) in zip(model_names, results):
                    rec["model"][mname] = "error" if err[t, i, j] else f"{int(model[t, i, j]):08x}"
                fam["first_mismatches"].append(rec)
            if name == "zero_acc_tiny_cancellation":
                model = results[0][0]
                alt126, alt127 = model_eval(pool, A, Bt, C, args.procs, worker=_eval_zero_alternatives)
                fam["zero_exponent_probe"] = {
                    "elements_where_a_zero_at_-126_differs_from_model": int(np.count_nonzero(alt126 != model)),
                    "hardware_eq_zero_at_-126_where_it_differs": int(np.count_nonzero((alt126 != model) & (alt126 == D))),
                    "elements_where_a_zero_at_-127_differs_from_model": int(np.count_nonzero(alt127 != model)),
                    "hardware_eq_zero_at_-127_where_it_differs": int(np.count_nonzero((alt127 != model) & (alt127 == D))),
                }
            if name == "adder_width_cancellation":
                fam["summary"] = adder_width_summary(meta, D)
            if name == "specials":
                fam["summary"] = specials_summary(meta, D, fmt)
            if name == "acc_only_lsb_walk":
                surviving = np.zeros(32, np.int64)
                for bit in range(32):
                    surviving[bit] = int(np.count_nonzero(((D >> bit) & 1) == ((C >> bit) & 1)))
                fam["summary"] = {"bases": meta["bases"], "elements": int(C.size),
                                  "elements_where_bit_b_of_D_equals_bit_b_of_C": {str(b): int(surviving[b]) for b in range(32)},
                                  "elements_with_D_eq_C": int(np.count_nonzero(D == C))}
            if name == "zero_products_nonzero_acc":
                fam["summary"] = {"elements_with_D_eq_C": int(np.count_nonzero(D == C)),
                                  "elements_with_D_ne_C": int(np.count_nonzero(D != C)),
                                  "elements_with_D_eq_C_masked_to_14_bits": int(np.count_nonzero(D == (C & ~np.uint32(0x3FF))))}
            if name == "fp8_floor_probe":
                fam["summary"] = floor_probe_summary(meta, D)
                fam["floor_alternatives"] = floor_alternatives(pool, ins.models[0], A, Bt, C, D, results[0][0], args.procs, meta["floors"])
            np.savez_compressed(out / f"tiles_{name}.npz", A=A, Bt=Bt, C=C, D=D)
            flat = rng.choice(A.shape[0] * M * N, size=n_sample, replace=False)
            for f in sorted(flat.tolist()):
                t, i, j = f // (M * N), (f % (M * N)) // N, f % N
                capture["records"].append({**hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j], fmt.hex_width), "family": name})
            fams[name] = fam
            print(name, json.dumps({k: v for k, v in fam.items() if k not in ("first_mismatches", "summary")}), flush=True)
    totals = {
        mname: sum(f["models"][mname]["mismatches"] for f in fams.values() if f["asserted"]) for mname in model_names
    }
    report = {
        "seed": args.seed,
        "n_random": args.n_random,
        "families": fams,
        "total_elements": sum(f["elements"] for f in fams.values()),
        "total_asserted_elements": sum(f["elements"] for f in fams.values() if f["asserted"]),
        "total_nonzero_accumulators": sum(f["nonzero_accumulators"] for f in fams.values()),
        "total_mismatches_by_model": totals,
        "model_agrees": {mname: totals[mname] == 0 for mname in model_names},
        "total_mismatches_vs_model": totals[model_names[0]],
    }
    (out / "capture_sample.json").write_text(json.dumps(capture, separators=(",", ":")))
    print("sweep TOTAL", report["total_asserted_elements"], "asserted elements,", json.dumps(totals), "mismatches", flush=True)
    return report, capture


# --- chained wgmma (BF16) ---------------------------------------------------------------------------------------------------------

CHAIN_STEPS = 4


def chain_kernel(probe: Probe) -> str:
    """The chained entry of the probe's kernel: four wgmma back to back in one commit group (``wgmma_tiles.cu``, STEPS = 4)."""

    return probe.kernel.replace("_tiles", f"_chain{CHAIN_STEPS}_tiles")


def _eval_chain_tiles(args):
    """``_eval_tiles`` for K = 16 * CHAIN_STEPS tiles: each model's step folded over the k16 slices in ascending order."""

    A, Bt, C = args
    n, M, K = A.shape
    outs = [np.zeros((n, M, N), np.uint32) for _ in _MODELS]
    errs = [np.zeros((n, M, N), np.bool_) for _ in _MODELS]
    w = K // CHAIN_STEPS
    for t in range(n):
        a_rows = [A[t, i].tolist() for i in range(M)]
        b_rows = [Bt[t, j].tolist() for j in range(N)]
        for i in range(M):
            for j in range(N):
                for m, model in enumerate(_MODELS):
                    acc = int(C[t, i, j])
                    try:
                        for s in range(CHAIN_STEPS):
                            acc = model.evaluate(acc, a_rows[i][s * w:(s + 1) * w], b_rows[j][s * w:(s + 1) * w])
                        outs[m][t, i, j] = acc
                    except Exception:  # noqa: BLE001 - domain errors are counted, not raised
                        errs[m][t, i, j] = True
    return outs, errs


def chain_sweep(so: Path, base: Probe, args, out: Path) -> dict:
    """Every sweep family at K = 64 through the chained kernel (one layout check first), against each model's step folded over the
    four k16 slices: whether back-to-back wgmma on one accumulator equal the chain of single steps, bit for bit.  ``base`` is the
    single-instruction probe whose layout check fixed the descriptor offsets."""

    ins, a_source = base.ins, base.a_source
    probe = Probe(so, ins, a_source, kernel=chain_kernel(base), k=CHAIN_STEPS * ins.k)
    probe.set_descriptor(base.descriptor["lbo_bytes"], base.descriptor["sbo_bytes"])
    M, K, fmt = ins.m, probe.k, ins.fmt
    rng = np.random.default_rng(1)
    a = rng.integers(-8, 9, (64, M, K)).astype(np.float32)
    b = rng.integers(-8, 9, (64, N, K)).astype(np.float32)
    c = rng.integers(-1000, 1001, (64, M, N)).astype(np.float32)
    want = f32_bits((c + np.einsum("nik,njk->nij", a, b)).astype(np.float32))
    if probe.tcgen05:
        probe.set_enable_input_d(True)
    got = probe.tiles(fmt.from_f32(a), fmt.from_f32(b), f32_bits(c))
    if not np.array_equal(want, got):
        raise RuntimeError(f"chain layout check failed: {probe.kernel} ({int(np.count_nonzero(want != got))} words differ)")
    if probe.tcgen05:
        probe.set_enable_input_d(not args.no_input_d)
    model_names = [m.name for m in ins.models]
    fams: dict = {}
    rng = np.random.default_rng(args.seed + CHAIN_STEPS)
    with mp.Pool(args.procs, initializer=_init, initargs=(model_descs(ins),)) as pool:
        for name, A, Bt, C, _meta in families(rng, args.chain_n_random, fmt, M, K):
            if args.no_input_d:
                C = np.zeros_like(C)
            D = probe.tiles(A, Bt, C)
            results = model_eval(pool, A, Bt, C, args.procs, worker=_eval_chain_tiles)
            fam = {"tiles": int(A.shape[0]), "elements": int(A.shape[0]) * M * N, "asserted": family_asserted(ins, name), "models": {},
                   "first_mismatches": []}
            any_mism = np.zeros(D.shape, np.bool_)
            for mname, (model, err) in zip(model_names, results):
                mism = (model != D) | err
                any_mism |= mism
                fam["models"][mname] = {"mismatches": int(mism.sum()), "domain_errors": int(err.sum())}
            for t, i, j in list(zip(*np.nonzero(any_mism)))[:FIRST_MISMATCHES]:
                rec = {"tile": int(t), "i": int(i), "j": int(j), **hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j], fmt.hex_width), "model": {}}
                for mname, (model, err) in zip(model_names, results):
                    rec["model"][mname] = "error" if err[t, i, j] else f"{int(model[t, i, j]):08x}"
                fam["first_mismatches"].append(rec)
            np.savez_compressed(out / f"tiles_chain{CHAIN_STEPS}_{name}.npz", A=A, Bt=Bt, C=C, D=D)
            fams[name] = fam
            print(f"chain{CHAIN_STEPS}", name, json.dumps({k: v for k, v in fam.items() if k != "first_mismatches"}), flush=True)
    totals = {m: sum(f["models"][m]["mismatches"] for f in fams.values() if f["asserted"]) for m in model_names}
    return {"kernel": probe.kernel, "steps": CHAIN_STEPS, "k": K, "a_source": a_source, "seed": args.seed + CHAIN_STEPS,
            "n_random": args.chain_n_random, "layout_check": {"status": "passed", "elements": int(want.size)}, "families": fams,
            "total_elements": sum(f["elements"] for f in fams.values()),
            "total_asserted_elements": sum(f["elements"] for f in fams.values() if f["asserted"]),
            "total_mismatches_by_model": totals, "model_agrees": {m: totals[m] == 0 for m in model_names}}


# --- main --------------------------------------------------------------------------------------------------------------------------


def write_result(out: Path, doc: dict) -> None:
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, out / "result.json")


def compile_only(ids: list[str], out: Path, args) -> int:
    """Build each instruction's kernel for its arch and run the SASS check; no device needed."""

    results: dict = {"schema": "wp1/tc-probe/v0", "mode": "compile-only", "identity": host_identity(), "instructions": {}}
    ok = True
    for id_ in ids:
        ins = INSTRUCTIONS[id_]
        arch = args.arch or ins.arch
        t0 = time.time()
        try:
            so, nvcc = build(ins.tu, arch, out)
            entry = {"instruction": ins.describe(), "arch": arch, "nvcc": nvcc, "so": so.name, "compiled": True, "seconds": round(time.time() - t0, 1)}
            entry["sass"] = sass_check(so, ins.kernel, out, ins.sass_mma)
            ok = ok and entry["sass"]["ok"]
            if ins.rs_kernel is not None:
                entry["sass_rs"] = sass_check(so, ins.rs_kernel, out, ins.sass_mma)
                ok = ok and entry["sass_rs"]["ok"]
        except subprocess.CalledProcessError as exc:
            entry = {"instruction": ins.describe(), "arch": arch, "compiled": False, "error": str(exc), "seconds": round(time.time() - t0, 1)}
            ok = False
        results["instructions"][id_] = entry
        print(id_, json.dumps({k: v for k, v in entry.items() if k not in ("instruction", "sass")}),
              json.dumps({k: v for k, v in entry.get("sass", {}).items() if k in ("ok", "kernel", "tile_kernel_mma_count", "error")}), flush=True)
    (out / "probe_results.json").write_text(json.dumps(results, indent=1))
    from verity_config.settings import SHARED  # here, not at the top: verity/ml/circuits/ops' tests import this module's tiles
    run_id = SHARED.RESEARCH_RUN_ID
    if run_id:
        checks = {id_: {"compiled": e["compiled"], "sass_ok": e.get("sass", {}).get("ok", False), "mma_count": e.get("sass", {}).get("tile_kernel_mma_count"),
                        "expected_mma_count": e.get("sass", {}).get("expected_mma_count"), "single_instruction": e.get("sass", {}).get("single_instruction"),
                        "mma": sorted({m for f in e.get("sass", {}).get("tile_kernel_functions", {}).values() for m in f["mma"]})}
                  for id_, e in results["instructions"].items()}
        write_result(out, {
            "schema": "research/result/v0.1",
            "run_id": run_id,
            "kind": "tc-compile-check/v1",
            "workload_fingerprint": {"probe": "benchmarks/tc_probe/tc_probe.py", "mode": "compile-only", "instructions": ids,
                                     "archs": {id_: e["arch"] for id_, e in results["instructions"].items()},
                                     "nvcc": next((e.get("nvcc") for e in results["instructions"].values() if e.get("nvcc")), None),
                                     "hostname": results["identity"]["hostname"]},
            "validation": {"status": "passed" if ok else "failed",
                           "checker": "nvcc compile + cuobjdump SASS (MMA-class instructions per tile kernel == the table's sass_mma; 1 = single instruction)",
                           "detail": json.dumps(checks), "evidence": ["probe_results.json"]},
            "measurements": [{"name": f"{id_}.tile_kernel_mma_count", "value": c["mma_count"], "unit": "instructions"} for id_, c in checks.items()],
            "measurement_files": ["probe_results.json"],
            "artifacts": [e["sass"]["sass_file"] for e in results["instructions"].values() if e.get("sass", {}).get("sass_file")],
        })
    print("COMPILE CHECK", "GREEN" if ok else "FAILED", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default=None, help="instruction id (see the table); comma list with --compile-only")
    ap.add_argument("--arch", default=None, help="override the instruction's nvcc arch (sm_80, sm_89, sm_90a); without --instruction selects the BF16 mma of that arch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay-golden", default=None, metavar="JSON")
    ap.add_argument("--replay-special", default=None, metavar="JSON.GZ", help="mma_special.reduced.json.gz: its all-finite records (BF16)")
    ap.add_argument("--replay-capture", default=None, metavar="JSON.GZ", help="tc-hopper-2026-09-07/special.reduced.json.gz: every record (all 14 "
                    "families, non-finite included) against its recorded word and the declared model")
    ap.add_argument("--also-model", action="append", default=[], metavar="NAME[:total]",
                    help="compare a further verity.ml.circuits.silicon.models model as a HYPOTHESIS (':total': its total semantics, which also asserts the "
                         "specials family and serves --replay-capture); repeatable")
    ap.add_argument("--a-source", default=None, choices=("smem", "registers"), help="the A operand path where the instruction has both (BF16 wgmma: "
                    "smem = SS, the GEMM form, default; registers = RS, FA3's P.V form)")
    ap.add_argument("--replay-instances", default=None, metavar="DIR", help="bench-instances/v1 directory with the built arrays: replay tu-k16 and vu-k1536 (BF16)")
    ap.add_argument("--instances-manifest", default=None, metavar="JSON", help="the pinned manifest to check the arrays against (default DIR/manifest.json)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--chain", action="store_true", help="BF16 wgmma: every family at K = 64 through four back-to-back wgmma in one commit group, "
                    "against the model's step folded over the four k16 slices")
    ap.add_argument("--chain-n-random", type=int, default=2000, help="--n-random of the chained families")
    ap.add_argument("--no-input-d", action="store_true", help="tcgen05: the first MMA runs with enable_input_d = 0 (the accumulator not read) "
                    "and every family's accumulators are +0, so the models see what the hardware computes")
    ap.add_argument("--n-random", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--golden-per-family", type=int, default=60)
    ap.add_argument("--families", default=None, help="comma list (default all)")
    ap.add_argument("--sass-check", action="store_true", help="cuobjdump -sass the built library and count MMA-class instructions in the tile kernel")
    ap.add_argument("--compile-only", action="store_true", help="build + SASS check only (any GPU or none); --instruction may be a comma list")
    ap.add_argument("--library", default=None, help="load this library (a --compile-only build of the same instruction and arch, e.g. on a "
                    "host with the CUDA toolkit) instead of compiling; its toolchain identity is read from <library>.nvcc")
    ap.add_argument("--canary", action="store_true", help="the fixed canary tile set (canary.py): the first tiles of every family + every resolved "
                    "discrepancy-log regression tile; < 1 min; exit 1 on any mismatch; result.json is tc-evidence/v1 with family=canary")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    if args.instruction is None:
        if args.arch not in ARCH_DEFAULT_INSTRUCTION:
            ap.error("--instruction is required (or --arch sm_80/sm_89/sm_90a for that arch's BF16 mma)")
        args.instruction = ARCH_DEFAULT_INSTRUCTION[args.arch]
    ids = args.instruction.split(",")
    for id_ in ids:
        if id_ not in INSTRUCTIONS:
            ap.error(f"unknown instruction {id_!r}; known: {', '.join(INSTRUCTIONS)}")
    if args.compile_only:
        return compile_only(ids, out, args)
    if len(ids) != 1:
        ap.error("one instruction per capture run")
    ins = INSTRUCTIONS[ids[0]]
    arch = args.arch or ins.arch
    if args.also_model:
        import dataclasses

        extra = []
        for spec in args.also_model:
            name, _, mode = spec.partition(":")
            if name not in TCM.MODELS or mode not in ("", "total"):
                ap.error(f"--also-model {spec!r}: a verity.ml.circuits.silicon.models name, optionally ':total'")
            extra.append(Model(TCM.MODELS[name], declared=False, total=mode == "total"))
        ins = dataclasses.replace(ins, models=ins.models + tuple(extra))

    so, nvcc = prebuilt(args.library) if args.library else build(ins.tu, arch, out)
    results: dict = {"schema": "wp1/tc-probe/v0", "instruction": ins.describe()}
    if args.sass_check:
        results["sass"] = sass_check(so, ins.entry(args.a_source), out, ins.sass_mma)
        print("sass", json.dumps({k: v for k, v in results["sass"].items() if k in ("ok", "kernel", "tile_kernel_mma_count", "tile_kernel_functions", "error")}), flush=True)
    probe = Probe(so, ins, args.a_source)
    identity = probe.identity()
    identity["nvcc"] = nvcc
    identity["arch"] = arch
    identity["a_source"] = probe.a_source
    identity["kernel"] = probe.kernel
    results["identity"] = identity
    results["layout_check"] = layout_check(probe, ins)
    if args.no_input_d:
        if not probe.tcgen05:
            ap.error("--no-input-d is a tcgen05 mode (sm100.tcgen05.mma.bf16)")
        probe.set_enable_input_d(False)
        identity["enable_input_d"] = 0
    results["pipeline"] = ins.models[0].pipeline.name  # the legacy field trust.py reads
    print("device", json.dumps(identity), flush=True)
    print("layout_check", json.dumps(results["layout_check"]), flush=True)

    if args.replay_golden:
        results["golden_replay"] = replay_golden(probe, Path(args.replay_golden))
        print("golden_replay", json.dumps({k: v for k, v in results["golden_replay"].items() if k != "first_mismatches"}), flush=True)

    if args.replay_special:
        path = Path(args.replay_special)
        results["special_replay"] = replay_golden(probe, path, special_all_finite_records(path), ins.models[0].pipeline.name)
        print("special_replay", json.dumps({k: v for k, v in results["special_replay"].items() if k != "first_mismatches"}), flush=True)

    if args.replay_capture:
        path = Path(args.replay_capture)
        replay_model = next((m for m in ins.models if m.total), ins.models[0])  # the non-finite records need a total model
        results["capture_replay"] = replay_golden(probe, path, hopper_capture_records(path), "veritor tc-hopper-2026-09-07", replay_model)
        print("capture_replay", json.dumps({k: v for k, v in results["capture_replay"].items() if not k.startswith("first_")}), flush=True)

    if args.replay_instances:
        results["instances_replay"] = replay_instances(probe, Path(args.replay_instances),
                                                       Path(args.instances_manifest) if args.instances_manifest else None)

    if args.sweep:
        results["sweep"], _ = sweep(probe, ins, args, out, identity)

    if args.chain:
        if not ins.chain:
            ap.error("--chain needs an instruction with the chained kernels (sm90.wgmma.m64n8k16.bf16, sm100.tcgen05.mma.bf16)")
        results["chain_sweep"] = chain_sweep(so, probe, args, out)
        print("chain TOTAL", results["chain_sweep"]["total_asserted_elements"], "asserted elements,",
              json.dumps(results["chain_sweep"]["total_mismatches_by_model"]), "mismatches", flush=True)

    canary_plan = None
    if args.canary:
        import canary as CANARY

        canary_plan = CANARY.canary_plan(ins.id)
        results["canary"] = CANARY.run_canary(probe, ins, canary_plan, out, args.procs)
        print("canary", json.dumps({k: results["canary"][k] for k in ("elements", "asserted_elements", "total_mismatches_by_model", "model_agrees", "digest", "seconds")},
                                   ), flush=True)
        for r in results["canary"]["replays"]:
            print("canary_replay", json.dumps({k: v for k, v in r.items() if k != "first_mismatches"}), flush=True)

    results["seconds"] = round(time.time() - t_all, 1)
    declared = [m.name for m in ins.models if m.declared]
    failures = []
    findings = []  # hypothesis disagreements: reported, not failures
    if "canary" in results:
        failures += results["canary"]["failures"]
        for mname, count in results["canary"]["total_mismatches_by_model"].items():
            if count and mname not in declared:
                findings.append(f"canary mismatches vs {mname}={count}")
    for replay in ("golden_replay", "special_replay", "capture_replay"):
        if replay in results and (results[replay]["mismatches"] or results[replay]["other_positions_nonzero"] or results[replay].get("model_mismatches")):
            failures.append(f"{replay} mismatches={results[replay]['mismatches']} other_positions_nonzero={results[replay]['other_positions_nonzero']}"
                            + (f" model_mismatches={results[replay]['model_mismatches']}" if "model_mismatches" in results[replay] else ""))
    if "sweep" in results:
        for mname, count in results["sweep"]["total_mismatches_by_model"].items():
            if count and mname in declared:
                failures.append(f"sweep mismatches vs {mname}={count}")
            elif count:
                findings.append(f"sweep mismatches vs {mname}={count}")
    if "chain_sweep" in results:
        for mname, count in results["chain_sweep"]["total_mismatches_by_model"].items():
            if count and mname in declared:
                failures.append(f"chain{CHAIN_STEPS} mismatches vs {mname}={count}")
            elif count:
                findings.append(f"chain{CHAIN_STEPS} mismatches vs {mname}={count}")
    if "instances_replay" in results:
        ir = results["instances_replay"]
        tu, vu = ir["tu-k16"], ir["vu-k1536"]
        if tu["mismatches"] or tu["other_positions_nonzero"]:
            failures.append(f"instances tu-k16 mismatches={tu['mismatches']} other_positions_nonzero={tu['other_positions_nonzero']}")
        if vu["accumulator_mismatches"] or vu["y_mismatches"] or vu["other_positions_nonzero"]:
            failures.append(f"instances vu-k1536 accumulator_mismatches={vu['accumulator_mismatches']} y_mismatches={vu['y_mismatches']} "
                            f"other_positions_nonzero={vu['other_positions_nonzero']}")
    if "sass" in results and not results["sass"]["ok"]:
        failures.append(f"sass tile_kernel_mma_count={results['sass'].get('tile_kernel_mma_count')} expected={ins.sass_mma}")
    if "sass" in results and results["sass"]["ok"] and not results["sass"].get("single_instruction", True):
        findings.append(f"not single-instruction: {results['sass']['tile_kernel_mma_count']} MMA-class SASS instructions per tile kernel (table sass_mma={ins.sass_mma})")
    results["failures"] = failures
    results["hypothesis_findings"] = findings
    (out / "probe_results.json").write_text(json.dumps(results, indent=1))

    from verity_config.settings import SHARED  # here, not at the top: verity/ml/circuits/ops' tests import this module's tiles
    run_id = SHARED.RESEARCH_RUN_ID
    if run_id:
        import canary as CANARY  # the label contract of tc-evidence/v1 lives there (no-op if --canary imported it already)

        checked_something = any(k in results for k in ("golden_replay", "special_replay", "capture_replay", "instances_replay", "chain_sweep")) or (("sweep" in results or "canary" in results) and declared)
        status = "failed" if failures else ("passed" if checked_something else "not_run")
        fingerprint = {
            "probe": "benchmarks/tc_probe/tc_probe.py",
            "instruction": ins.id,
            "ptx": ins.ptx,
            "arch": arch,
            "a_source": probe.a_source,
            "kernel": probe.kernel,
            "sku": identity["device"],
            "device": identity["device"],
            "compute_capability": identity["compute_capability"],
            "driver": identity["driver"],
            "gpu_uuid": identity.get("gpu_uuid"),
            "hostname": identity.get("hostname"),
            "nvcc": nvcc,
            "seed": args.seed,
            "n": args.n_random,
            "n_random": args.n_random,
            "families": list(results["sweep"]["families"]) if "sweep" in results else [],
            "models": [m.describe() for m in ins.models],
            "pipeline": ins.models[0].pipeline.name,
        }
        if "instances_replay" in results:
            fingerprint["instances"] = {"dataset": results["instances_replay"]["dataset"], "manifest_sha256": results["instances_replay"]["digests"]["manifest_sha256"]}
        if "canary" in results:
            fingerprint["family"] = "canary"  # the trust framework's family label for canary artifacts
            fingerprint["canary"] = {"seed": canary_plan.seed, "digest": results["canary"]["digest"], "families": list(canary_plan.families),
                                     "replays": list(canary_plan.replays), "regression": canary_plan.to_dict()["regression"]}
        evidence = {
            "instruction": ins.id,
            "arch": arch,
            "a_source": probe.a_source,
            "sku": identity["device"],
            "driver": identity["driver"],
            "gpu_uuid": identity.get("gpu_uuid"),
            "hostname": identity.get("hostname"),
            "nvcc": nvcc,
            "seed": args.seed,
            "n": args.n_random,
            "layout_check": results["layout_check"],
            "models": {m.name: {"declared": m.declared, "pipeline": m.pipeline.name, "product": m.product or m.pipeline.dtype, "lowering": m.lowering,
                                "total": m.total}
                       for m in ins.models},
        }
        measurements = [{"name": "wall", "value": results["seconds"], "unit": "s"}]
        if "sweep" in results:
            sw = results["sweep"]
            evidence["families"] = {
                name: {"elements": f["elements"], "asserted": f["asserted"], "mismatches": {m: v["mismatches"] for m, v in f["models"].items()},
                       "domain_errors": {m: v["domain_errors"] for m, v in f["models"].items()}, "first_mismatches": f["first_mismatches"]}
                for name, f in sw["families"].items()
            }
            evidence["model_agrees"] = sw["model_agrees"]
            evidence["total_mismatches_by_model"] = sw["total_mismatches_by_model"]
            measurements.append({"name": "elements", "value": sw["total_elements"], "unit": "elements"})
            for mname, count in sw["total_mismatches_by_model"].items():
                measurements.append({"name": f"mismatches[{mname}]", "value": count, "unit": "elements"})
            for name, f in sw["families"].items():
                measurements.append({"name": f"{name}.elements", "value": f["elements"], "unit": "elements"})
                for m, v in f["models"].items():
                    measurements.append({"name": f"{name}.mismatches[{m}]", "value": v["mismatches"], "unit": "elements"})
        for replay in ("golden_replay", "special_replay", "capture_replay"):
            if replay in results:
                evidence[replay] = dict(results[replay])
                measurements.append({"name": f"{replay}.records", "value": results[replay]["records"], "unit": "records"})
                measurements.append({"name": f"{replay}.mismatches", "value": results[replay]["mismatches"], "unit": "records"})
        if "chain_sweep" in results:
            ch = results["chain_sweep"]
            evidence["chain"] = {**{k: v for k, v in ch.items() if k != "families"},
                                 "families": {name: {"elements": f["elements"], "asserted": f["asserted"],
                                                     "mismatches": {m: v["mismatches"] for m, v in f["models"].items()},
                                                     "first_mismatches": f["first_mismatches"]} for name, f in ch["families"].items()}}
            measurements.append({"name": f"chain{CHAIN_STEPS}.elements", "value": ch["total_elements"], "unit": "elements"})
            for mname, count in ch["total_mismatches_by_model"].items():
                measurements.append({"name": f"chain{CHAIN_STEPS}.mismatches[{mname}]", "value": count, "unit": "elements"})
        if "sass" in results:
            evidence["sass"] = {k: v for k, v in results["sass"].items() if k != "functions"}
            measurements.append({"name": "tile_kernel_mma_count", "value": results["sass"].get("tile_kernel_mma_count"), "unit": "instructions"})
        if "canary" in results:
            cn = results["canary"]
            evidence["canary"] = {k: v for k, v in cn.items() if k != "families"}
            evidence["canary"]["families"] = {name: {k: v for k, v in f.items()} for name, f in cn["families"].items()}
            if "model_agrees" not in evidence:  # a canary alone: its verdict is the artifact's model_agrees
                evidence["model_agrees"] = cn["model_agrees"]
                evidence["total_mismatches_by_model"] = cn["total_mismatches_by_model"]
            measurements.append({"name": "canary.elements", "value": cn["elements"], "unit": "elements"})
            for mname, count in cn["total_mismatches_by_model"].items():
                measurements.append({"name": f"canary.mismatches[{mname}]", "value": count, "unit": "elements"})
            for r in cn["replays"]:
                measurements.append({"name": f"canary.replay[{Path(r['file']).name}].mismatches", "value": r["mismatches"], "unit": "records"})
            evidence["labels"] = CANARY.result_labels(ins, cn, identity)
        elif "sweep" in results:
            evidence["labels"] = CANARY.evidence_labels(ins, results["sweep"], identity, evidence="sweep")
        evidence["hypothesis_findings"] = findings
        artifacts = []
        if "sweep" in results:
            artifacts = ["capture_sample.json"] + [f"tiles_{name}.npz" for name in results["sweep"]["families"]]
        if "chain_sweep" in results:
            artifacts += [f"tiles_chain{CHAIN_STEPS}_{name}.npz" for name in results["chain_sweep"]["families"]]
        if "canary" in results:
            artifacts.append("canary_tiles.npz")
        if results.get("sass", {}).get("sass_file"):
            artifacts.append(results["sass"]["sass_file"])
        checker = "tc_probe vs " + (", ".join(declared) if declared else "HYPOTHESIS models only (" + ", ".join(m.name for m in ins.models) + ")")
        detail = "; ".join(failures) or ("every hardware word reproduced" if status == "passed" else "no pinned model compared; agreement counts in tc_evidence")
        if findings:
            detail += " | hypothesis findings: " + "; ".join(findings)
        write_result(out, {
            "schema": "research/result/v0.1",
            "run_id": run_id,
            "kind": "tc-evidence/v1",
            "workload_fingerprint": fingerprint,
            "validation": {"status": status, "checker": checker, "detail": detail, "evidence": ["probe_results.json"]},
            "measurements": measurements,
            "measurement_files": ["probe_results.json"],
            "artifacts": artifacts,
            "tc_evidence": evidence,
        })
        if evidence.get("labels"):
            print("labels (apply to the result artifact once `research run` has stored it):", flush=True)
            for cmd in CANARY.label_commands("<artifact-id>", evidence["labels"], SHARED.RESEARCH_LANE or "<lane>"):
                print("   ", cmd, flush=True)
    print("FAILURES" if failures else "ALL HARDWARE WORDS REPRODUCED" if not findings else "RECORDED (hypothesis findings)",
          failures or findings, f"{results['seconds']}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
