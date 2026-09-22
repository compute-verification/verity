"""Hardware probe for one tensor-core instruction: one ``mma.sync`` (or ``wgmma``) per tile, arbitrary accumulators, captures in
the golden-record format, compared bit for bit against the pinned ``verity.ml.tc.silicon`` model of that instruction.

Provenance: ported for lane WP1 from veritor ``gpu/tensor-core-semantics/`` (``mma_tiles.cu``, ``tcs_gpu.py``,
``validate_tiles.py``: the BF16 kernel, encodings, finite-sweep families, per-element records, multiprocess model comparison,
golden sampling) and extended for lane tc-probe-fp8 with the E4M3 kernel half and encodings of the same files, the FP8-specific
families (the Ada zero-product accumulator truncation of ``docs/hardware-semantics.md``, Hawkeye's adder-width probe of
``characterize.py``, the special-value classes of ``gpu/tc-fp8/campaign.py``) and the Hopper ``wgmma`` path with the SASS
confirmation of ``gpu/tc-fp8/wgmma_fp8_probe.py`` (exactly one MMA-class instruction per tile kernel).  Only the GPU host runs
this; it needs numpy and nvcc (``cuobjdump`` for ``--sass-check``).

Instructions (``--instruction``; ``--arch`` is derived from the entry and may be overridden)::

    id                          nvcc arch  kernel                       K   model(s) compared
    sm80.mma.m16n8k16.bf16      sm_80      mma_bf16_m16n8k16_tiles      16  ampere_bf16_m16n8k16 (pinned)
    sm89.mma.m16n8k16.bf16      sm_89      mma_bf16_m16n8k16_tiles      16  ada_bf16_m16n8k16 (pinned)
    sm90.mma.m16n8k16.bf16      sm_90a     mma_bf16_m16n8k16_tiles      16  hopper_bf16_m16n8k16 (pinned)
    sm89.mma.m16n8k32.e4m3      sm_89      mma_e4m3_m16n8k32_tiles      32  ada_e4m3_m16n8k32 (pinned)
    sm89.mma.m16n8k32.e5m2      sm_89      mma_e5m2_m16n8k32_tiles      32  HYPOTHESIS ada_e4m3_m16n8k32 with e5m2 products
    sm90.mma.m16n8k32.e4m3      sm_90a     mma_e4m3_m16n8k32_tiles      32  HYPOTHESES hopper_e4m3_wgmma_k32 and ada_e4m3_m16n8k32 (open question: which agrees)
    sm90.mma.m16n8k32.e5m2      sm_90a     mma_e5m2_m16n8k32_tiles      32  HYPOTHESES both of the above with e5m2 products
    sm90.wgmma.m64n8k32.e4m3    sm_90a     wgmma_e4m3_m64n8k32_tiles    32  hopper_e4m3_wgmma_k32 (pinned)
    sm90.wgmma.m64n8k32.e5m2    sm_90a     wgmma_e5m2_m64n8k32_tiles    32  HYPOTHESIS hopper_e4m3_wgmma_k32 with e5m2 products

A pinned model's mismatches fail ``validation``; a HYPOTHESIS model only yields agreement counts (validation ``not_run`` when no
pinned model is compared) and the raw capture is recorded regardless.  The E5M2 product (``_e5m2_product``) is a local
HYPOTHESIS pending the canonical one in ``verity.ml.tc``.  Kernels: ``mma_tiles.cu`` (``mma.sync``, sm_80+/sm_89+) and
``wgmma_tiles.cu`` (``wgmma.mma_async``, sm_90a; A from registers, B through a shared-memory descriptor).

    python tools/tc_probe/tc_probe.py --instruction sm89.mma.m16n8k16.bf16 --out $RESEARCH_RUN_DIR \\
        --replay-golden packages/verity/tests/ml/fixtures/golden/ada_bf16_m16n8k16.json --sweep --n-random 100000
    python tools/tc_probe/tc_probe.py --instruction sm89.mma.m16n8k32.e4m3 --out $RESEARCH_RUN_DIR --sweep --sass-check
    python tools/tc_probe/tc_probe.py --instruction sm90.wgmma.m64n8k32.e4m3,sm90.mma.m16n8k32.e4m3 --out DIR --compile-only

``--replay-golden`` re-executes every record of a golden file on the device (each at its own tile position) and requires the
hardware word to equal the record's ``d``.  ``--replay-instances DIR`` replays ``bench-instances/v1`` (BF16 only: the ``tu-k16``
tier as one launch, ``vu-k1536`` as 96 dependent launches with the hardware's accumulators carried) against the dataset's
expected words, after checking the arrays' sha256 against the pinned manifest (``--instances-manifest``).  ``--sweep`` runs the
finite-sweep families and compares every output element with the model(s); ``--families`` selects a comma list.  Families
(``M`` = 16 rows for ``mma``, 64 for ``wgmma``; ``n`` = ``--n-random`` tiles of ``M x 8`` elements):

    randn_zero_acc (n), randn_random_acc (n/4), uniform_bits (n/4), subnormal (n/8), cancellation (n/8), mixed_magnitude (n/8),
    near_overflow (n/16), signed_zero (256), tiny_acc_zero_products (512)               -- every format, as in the Ada validation
    zero_acc_tiny_cancellation (max(256, n/16))                                          -- BF16 only (WP1: the zero's exponent)
    zero_products_nonzero_acc (512): all products zero, random finite accumulators        -- FP8: the accumulator truncation quirk
    acc_only_lsb_walk (64 bases x 1024 low-10-bit patterns): zero products               -- FP8: which accumulator bits survive
    specials (48): NaN (E4M3) / inf and NaN (E5M2) operand and accumulator classes       -- FP8: hardware words only, never asserted
    adder_width_cancellation: +P, -P and a residual 2^ratio.P in every k slot, ratio in [-26, -5], zero and P accumulators
                                                                                          -- FP8: Hawkeye's width and grouping probe

Seeds are fixed (``--seed``, default 20260902) and recorded.  Outputs under ``--out``: ``probe_results.json`` (identity, layout
check, per-family counts per model, first mismatches with A/B/C/D words, family summaries, SASS counts), ``capture_sample.json``
(golden-format records sampled per family), ``tiles_<family>.npz`` (the full capture: A, Bt, C, D per tile) and, under
``research run``, ``result.json`` with ``kind: tc-evidence/v1`` (``workload_fingerprint`` = instruction, arch, sku, driver, nvcc,
seed, n, families, models; ``tc_evidence`` = per-family counts, ``model_agrees`` per model, first mismatches; validation passed
iff every mismatch count against a pinned model and every replay is zero).  ``--compile-only`` builds the instruction's kernel
for its arch (sm_90a compiles wherever the toolkit is, whatever the GPU), runs ``--sass-check`` (cuobjdump: MMA-class
instructions per tile kernel, exactly one expected) and writes ``result.json`` with ``kind: tc-compile-check/v1`` without
touching the device.  The BF16 path is unchanged from WP1: the same families, the same random draws, the same words.
"""

from __future__ import annotations

import argparse
import ctypes
import json
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

import numpy as np

from verity.ml.tc import silicon as S

HERE = Path(__file__).resolve().parent
N = 8  # every shape here has 8 output columns
MAX_TILES_PER_CALL = 1 << 16
FIRST_MISMATCHES = 20
UNASSERTED_FAMILIES = {"specials"}  # recorded, never compared for validation
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
        to the largest finite mantissa (E4M3) -- the draws are the ones of the WP1 BF16 probe, unchanged."""

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


PRODUCTS: dict[str, Callable[[int, int], S.Term]] = {"e5m2": _e5m2_product}


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


@dataclass(frozen=True)
class Model:
    """A model the capture is compared with.  ``product`` names an entry of :data:`PRODUCTS` (None: the pipeline's own product,
    evaluated by :func:`silicon.tc_dot` exactly as pinned).  ``declared`` models fail validation on any mismatch; the others are
    HYPOTHESES whose agreement counts are reported."""

    pipeline: S.Pipeline
    product: str | None = None
    declared: bool = True

    @property
    def name(self) -> str:
        n = self.pipeline.name + (f"+{self.product}_product" if self.product else "")
        return n if self.declared else n + "[HYPOTHESIS]"

    def describe(self) -> dict:
        p = self.pipeline
        return {"name": self.name, "pipeline": p.name, "groups": list(p.groups), "width": p.width, "zero_exponent": p.zero_exponent,
                "product": self.product or p.dtype, "declared": self.declared}


@dataclass(frozen=True)
class Instruction:
    id: str
    arch: str  # nvcc -arch
    tu: str  # translation unit under tools/tc_probe/
    kernel: str  # extern "C" entry; the device kernel is <kernel>_kernel
    ptx: str
    fmt: Format
    m: int
    k: int
    models: tuple[Model, ...]

    def describe(self) -> dict:
        return {"id": self.id, "arch": self.arch, "tu": self.tu, "kernel": self.kernel, "ptx": self.ptx, "format": self.fmt.name,
                "m": self.m, "n": N, "k": self.k, "models": [m.describe() for m in self.models]}


def _mma(id_: str, arch: str, fmt: Format, models: tuple[Model, ...]) -> Instruction:
    k = 16 if fmt is BF16 else 32
    return Instruction(id_, arch, "mma_tiles.cu", f"mma_{fmt.name}_m16n8k{k}_tiles",
                       f"mma.sync.aligned.m16n8k{k}.row.col.f32.{fmt.name}.{fmt.name}.f32", fmt, 16, k, models)


def _wgmma(id_: str, fmt: Format, models: tuple[Model, ...]) -> Instruction:
    return Instruction(id_, "sm_90a", "wgmma_tiles.cu", f"wgmma_{fmt.name}_m64n8k32_tiles",
                       f"wgmma.mma_async.sync.aligned.m64n8k32.f32.{fmt.name}.{fmt.name}", fmt, 64, 32, models)


_HOPPER_E5M2 = Model(S.HOPPER_E4M3_K32, "e5m2", declared=False)
_ADA_E5M2 = Model(S.ADA_E4M3_M16N8K32, "e5m2", declared=False)

INSTRUCTIONS: dict[str, Instruction] = {
    i.id: i
    for i in (
        _mma("sm80.mma.m16n8k16.bf16", "sm_80", BF16, (Model(S.AMPERE_BF16_M16N8K16),)),
        _mma("sm89.mma.m16n8k16.bf16", "sm_89", BF16, (Model(S.ADA_BF16_M16N8K16),)),
        _mma("sm90.mma.m16n8k16.bf16", "sm_90a", BF16, (Model(S.HOPPER_BF16_M16N8K16),)),
        _mma("sm89.mma.m16n8k32.e4m3", "sm_89", E4M3, (Model(S.ADA_E4M3_M16N8K32),)),
        _mma("sm89.mma.m16n8k32.e5m2", "sm_89", E5M2, (_ADA_E5M2,)),
        _mma("sm90.mma.m16n8k32.e4m3", "sm_90a", E4M3, (Model(S.HOPPER_E4M3_K32, declared=False), Model(S.ADA_E4M3_M16N8K32, declared=False))),
        _mma("sm90.mma.m16n8k32.e5m2", "sm_90a", E5M2, (_HOPPER_E5M2, _ADA_E5M2)),
        _wgmma("sm90.wgmma.m64n8k32.e4m3", E4M3, (Model(S.HOPPER_E4M3_K32),)),
        _wgmma("sm90.wgmma.m64n8k32.e5m2", E5M2, (_HOPPER_E5M2,)),
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
    subprocess.run([nvcc_bin, "-O2", f"-arch={arch}", "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(HERE / tu)], check=True)
    return so, " / ".join(line.strip() for line in nvcc)


MMA_SASS = re.compile(r"\b((?:[A-Z]{1,2})?MMA(?:\.[A-Za-z0-9_.]+)?)\b")


def sass_check(so: Path, kernel: str) -> dict:
    """``cuobjdump -sass`` of the built library: MMA-class instructions (HMMA/QMMA/HGMMA/QGMMA/IMMA...) per device function, and
    the count in the instruction's tile kernel, which must be exactly one for the probe to be single-instruction."""

    proc = subprocess.run([cuda_bin("cuobjdump"), "-sass", str(so)], capture_output=True, text=True)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip()[-2000:], "kernel": kernel}
    per_function: dict[str, dict] = {}
    current = None
    for line in proc.stdout.splitlines():
        m = re.search(r"Function\s*:\s*(\S+)", line)
        if m:
            current = m.group(1)
            per_function[current] = {"mma_count": 0, "mma": []}
            continue
        if current is None or "/*" not in line:
            continue
        code = line.split("/*", 1)[1].split("*/", 1)[1] if line.lstrip().startswith("/*") else line  # drop the offset comment
        code = code.split("/*", 1)[0]  # drop the encoding comment
        for hit in MMA_SASS.findall(code):
            per_function[current]["mma_count"] += 1
            if hit not in per_function[current]["mma"]:
                per_function[current]["mma"].append(hit)
    device_kernel = f"{kernel}_kernel"
    tile_fns = {name: v for name, v in per_function.items() if device_kernel in name}
    count = sum(v["mma_count"] for v in tile_fns.values())
    return {
        "ok": bool(tile_fns) and count == 1,
        "kernel": device_kernel,
        "tile_kernel_functions": tile_fns,
        "tile_kernel_mma_count": count,
        "functions": per_function,
        "sass_sha256": _sha256_text(proc.stdout),
    }


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


class Probe:
    def __init__(self, so: Path, ins: Instruction):
        self.ins = ins
        self.lib = ctypes.CDLL(str(so))
        fn = getattr(self.lib, ins.kernel)
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.fn = fn
        ident = self.lib.device_identity
        ident.restype = ctypes.c_int
        ident.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int]
        self.descriptor: dict | None = None
        if ins.tu == "wgmma_tiles.cu":
            setter = self.lib.wgmma_set_descriptor_offsets
            setter.restype = None
            setter.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
            self.set_descriptor(128, 256)

    def set_descriptor(self, lbo: int, sbo: int) -> None:
        self.lib.wgmma_set_descriptor_offsets(lbo, sbo)
        self.descriptor = {"lbo_bytes": lbo, "sbo_bytes": sbo, "layout": "K-major no swizzle (INTERLEAVE)"}

    def identity(self) -> dict:
        major, minor = ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        rc = self.lib.device_identity(ctypes.byref(major), ctypes.byref(minor), name, 256)
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in cudaGetDeviceProperties")
        return {"device": name.value.decode(), "compute_capability": f"{major.value}.{minor.value}", **host_identity()}

    def tiles(self, A: np.ndarray, Bt: np.ndarray, C: np.ndarray) -> np.ndarray:
        """One instruction per tile.  ``A``: (n,M,K), ``Bt``: (n,8,K) operand words, ``C``: (n,M,8) FP32 words -> ``D`` (n,M,8)."""

        M, K, fmt = self.ins.m, self.ins.k, self.ins.fmt
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
                raise RuntimeError(f"CUDA error {rc} in {self.ins.kernel} (209 = no kernel image for this device: wrong --arch?)")
            D[start:stop] = d
        return D

    # kept for the BF16 replay code paths
    mma_tiles = tiles


def host_identity() -> dict:
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
        text = smi.stdout.strip() if smi.returncode == 0 else None
    except OSError:
        text = None
    fields = [f.strip() for f in text.split(",")] if text else []
    return {"nvidia_smi": text, "driver": fields[1] if len(fields) >= 2 else None, "hostname": platform.node()}


# --- the finite-sweep families ---------------------------------------------------------------------------------------------------------


def families(rng, n_random, fmt: Format, M: int, K: int):
    """Yield (name, A (n,M,K), Bt (n,8,K), C (n,M,8), meta) -- the nine families of the Ada validation (unchanged for BF16: the
    same draws in the same order), the WP1 zero-exponent family (BF16) and the FP8-specific families."""

    cast = lambda x: fmt.from_f32(x.astype(np.float32))
    zeros_acc = lambda n: np.zeros((n, M, N), np.uint32)
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
        # 10. (WP1 addition) +-0 accumulators with cancelling products below 2^-145: BF16 subnormal a (30% of lanes), b in
        # [2^-27, 2^-5].  Decides whether a zero accumulator's exponent enters the alignment maximum (silicon and the simulator's
        # E_ZERO zero: no; a zero carried at -126/-127 would truncate these sums differently -- see the log).
        n = max(256, n_random // 16)
        a = ((rng.integers(0, 2, (n, M, K)) << 15) | rng.integers(1, 128, (n, M, K))).astype(np.uint16)
        a = np.where(rng.random((n, M, K)) < 0.7, np.uint16(0), a)
        b = ((rng.integers(0, 2, (n, N, K)) << 15) | (rng.integers(100, 123, (n, N, K)) << 7) | rng.integers(0, 128, (n, N, K))).astype(np.uint16)
        acc = np.where(rng.random((n, M, N)) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)
        yield "zero_acc_tiny_cancellation", a, b, acc, None

    if fp8:
        yield from fp8_families(rng, fmt, M, K)


def fp8_families(rng, fmt: Format, M: int, K: int):
    """The FP8-specific families (tc-probe-fp8)."""

    zero_ops = lambda n: (np.zeros((n, M, K), fmt.dtype), np.zeros((n, N, K), fmt.dtype))

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
    # the slot and the residual's a exponent, columns the b exponent), ratio in [-26, -5], with a zero accumulator (variant
    # "prod") or with the +P carried by the accumulator instead of product 0 (variant "acc").  A 14-bit adder keeps the residual
    # iff ratio >= -13 when it is summed with the pair; a residual in a later group survives whatever the ratio.
    A, Bt, C, meta = adder_width_tiles(fmt, M, K)
    yield "adder_width_cancellation", A, Bt, C, meta


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


def adder_width_tiles(fmt: Format, M: int, K: int):
    lo, hi = fmt.min_pow2_exp, fmt.max_pow2_exp
    ratios = list(range(-26, -4))
    cases = []  # (variant, pe, ratio_base, slot_base)
    for variant in ("prod", "acc"):
        for pa in (0, (hi + 1) // 2, hi):  # pair exponent pe = 2 * pa: unit, mid, the largest square
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
        for j in range(N):
            ratio = r0 + j
            # residual 2^(pe + ratio) = 2^up * 2^fl with both exponents inside the format
            target = pe + ratio
            up = max(lo, min(hi, target // 2))
            fl = target - up
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
    else, folded to the most negative surviving ratio and the least negative truncated one."""

    table: dict[tuple, dict[int, str]] = {}
    for s, info in enumerate(meta["elements"]):
        for i, row in enumerate(info):
            for j, e in enumerate(row):
                if e is None:
                    continue
                variant, pe, ratio, slot = e
                want = int(f32_bits(np.array([2.0 ** (pe + ratio)], np.float32))[0])
                got = int(D[s, i, j])
                verdict = "survived" if got == want else ("truncated" if got == 0 else f"{got:08x}")
                table.setdefault((variant, slot), {})
                prev = table[(variant, slot)].get(ratio)
                table[(variant, slot)][ratio] = verdict if prev in (None, verdict) else f"inconsistent({prev},{verdict})"
    out: dict[str, dict] = {}
    for (variant, slot), by_ratio in sorted(table.items()):
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


# --- the models, per tile, in processes ----------------------------------------------------------------------------------------------

_MODELS: list[tuple[str, S.Pipeline, Callable | None]] = []


def _init(model_descs):
    global _MODELS
    _MODELS = [(name, S.PIPELINES[pname], PRODUCTS[prod] if prod else None) for name, pname, prod in model_descs]


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
                for m, (_, pipe, product) in enumerate(_MODELS):
                    try:
                        if product is None:
                            outs[m][t, i, j] = S.tc_dot(pipe, acc, a_rows[i], b_rows[j])
                        else:
                            outs[m][t, i, j] = tc_dot_with_product(pipe, product, acc, a_rows[i], b_rows[j])
                    except Exception:  # noqa: BLE001 - domain errors are counted, not raised
                        errs[m][t, i, j] = True
    return outs, errs


def model_eval(pool, A, Bt, C, procs, worker=_eval_tiles):
    n = A.shape[0]
    chunks = max(1, min(n // 64 + 1, procs * 8))
    idx = np.array_split(np.arange(n), chunks)
    parts = pool.map(worker, [(A[i], Bt[i], C[i]) for i in idx])
    if worker is not _eval_tiles:
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
    raise RuntimeError(f"fragment layout check failed: {ins.kernel} does not compute C + A.Bt on exact integer tiles "
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


def replay_golden(probe: Probe, path: Path, records: list[dict] | None = None, pipeline: str | None = None) -> dict:
    """Every record at its own tile position (r % M, (r // M) % 8), the rest of the tile zero."""

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
    other_nonzero = 0
    for r, rec in enumerate(records):
        i, j = r % M, (r // M) % N
        if int(D[r, i, j]) != int(rec["d"], 16):
            mism.append({**rec, "hardware": f"{int(D[r, i, j]):08x}"})
        rest = D[r].copy()
        rest[i, j] = 0
        other_nonzero += int(np.count_nonzero(rest))  # zero rows times anything, zero accumulator: every other output is +0
    return {
        "file": str(path),
        "pipeline": pipeline,
        "records": n,
        "nonzero_accumulators": int(sum(1 for rec in records if int(rec["acc"], 16) & 0x7FFFFFFF)),
        "mismatches": len(mism),
        "first_mismatches": mism[:FIRST_MISMATCHES],
        "other_positions_nonzero": other_nonzero,
    }


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

    from verity.ml.tc.cast import f32_to_bf16_word
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
    descs = [(m.name, m.pipeline.name, m.product) for m in ins.models]
    with mp.Pool(args.procs, initializer=_init, initargs=(descs,)) as pool:
        for name, A, Bt, C, meta in families(rng, args.n_random, fmt, M, K):
            if only is not None and name not in only:
                continue
            t0 = time.time()
            D = probe.tiles(A, Bt, C)
            t_gpu = time.time() - t0
            t0 = time.time()
            results = model_eval(pool, A, Bt, C, args.procs)
            t_model = time.time() - t0
            asserted = name not in UNASSERTED_FAMILIES
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
            # compatibility with the WP1 field names: the first (declared) model's count
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
            np.savez_compressed(out / f"tiles_{name}.npz", A=A, Bt=Bt, C=C, D=D)
            flat = rng.choice(A.shape[0] * M * N, size=min(args.golden_per_family, A.shape[0] * M * N), replace=False)
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
            entry["sass"] = sass_check(so, ins.kernel)
            ok = ok and entry["sass"]["ok"]
        except subprocess.CalledProcessError as exc:
            entry = {"instruction": ins.describe(), "arch": arch, "compiled": False, "error": str(exc), "seconds": round(time.time() - t0, 1)}
            ok = False
        results["instructions"][id_] = entry
        print(id_, json.dumps({k: v for k, v in entry.items() if k not in ("instruction", "sass")}),
              json.dumps({k: v for k, v in entry.get("sass", {}).items() if k in ("ok", "kernel", "tile_kernel_mma_count", "error")}), flush=True)
    (out / "probe_results.json").write_text(json.dumps(results, indent=1))
    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id:
        checks = {id_: {"compiled": e["compiled"], "sass_ok": e.get("sass", {}).get("ok", False), "mma_count": e.get("sass", {}).get("tile_kernel_mma_count"),
                        "mma": sorted({m for f in e.get("sass", {}).get("tile_kernel_functions", {}).values() for m in f["mma"]})}
                  for id_, e in results["instructions"].items()}
        write_result(out, {
            "schema": "research/result/v0.1",
            "run_id": run_id,
            "kind": "tc-compile-check/v1",
            "workload_fingerprint": {"probe": "tools/tc_probe/tc_probe.py", "mode": "compile-only", "instructions": ids,
                                     "archs": {id_: e["arch"] for id_, e in results["instructions"].items()},
                                     "nvcc": next((e.get("nvcc") for e in results["instructions"].values() if e.get("nvcc")), None),
                                     "hostname": results["identity"]["hostname"]},
            "validation": {"status": "passed" if ok else "failed", "checker": "nvcc compile + cuobjdump SASS (exactly one MMA-class instruction per tile kernel)",
                           "detail": json.dumps(checks), "evidence": ["probe_results.json"]},
            "measurements": [{"name": f"{id_}.tile_kernel_mma_count", "value": c["mma_count"], "unit": "instructions"} for id_, c in checks.items()],
            "measurement_files": ["probe_results.json"],
            "artifacts": [],
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
    ap.add_argument("--replay-instances", default=None, metavar="DIR", help="bench-instances/v1 directory with the built arrays: replay tu-k16 and vu-k1536 (BF16)")
    ap.add_argument("--instances-manifest", default=None, metavar="JSON", help="the pinned manifest to check the arrays against (default DIR/manifest.json)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-random", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--golden-per-family", type=int, default=60)
    ap.add_argument("--families", default=None, help="comma list (default all)")
    ap.add_argument("--sass-check", action="store_true", help="cuobjdump -sass the built library and count MMA-class instructions in the tile kernel")
    ap.add_argument("--compile-only", action="store_true", help="build + SASS check only (any GPU or none); --instruction may be a comma list")
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

    so, nvcc = build(ins.tu, arch, out)
    results: dict = {"schema": "wp1/tc-probe/v0", "instruction": ins.describe()}
    if args.sass_check:
        results["sass"] = sass_check(so, ins.kernel)
        print("sass", json.dumps({k: v for k, v in results["sass"].items() if k in ("ok", "kernel", "tile_kernel_mma_count", "tile_kernel_functions", "error")}), flush=True)
    probe = Probe(so, ins)
    identity = probe.identity()
    identity["nvcc"] = nvcc
    identity["arch"] = arch
    results["identity"] = identity
    results["layout_check"] = layout_check(probe, ins)
    results["pipeline"] = ins.models[0].pipeline.name  # WP1 field
    print("device", json.dumps(identity), flush=True)
    print("layout_check", json.dumps(results["layout_check"]), flush=True)

    if args.replay_golden:
        results["golden_replay"] = replay_golden(probe, Path(args.replay_golden))
        print("golden_replay", json.dumps({k: v for k, v in results["golden_replay"].items() if k != "first_mismatches"}), flush=True)

    if args.replay_special:
        path = Path(args.replay_special)
        results["special_replay"] = replay_golden(probe, path, special_all_finite_records(path), ins.models[0].pipeline.name)
        print("special_replay", json.dumps({k: v for k, v in results["special_replay"].items() if k != "first_mismatches"}), flush=True)

    if args.replay_instances:
        results["instances_replay"] = replay_instances(probe, Path(args.replay_instances),
                                                       Path(args.instances_manifest) if args.instances_manifest else None)

    if args.sweep:
        results["sweep"], _ = sweep(probe, ins, args, out, identity)

    results["seconds"] = round(time.time() - t_all, 1)
    declared = [m.name for m in ins.models if m.declared]
    failures = []
    findings = []  # hypothesis disagreements: reported, not failures
    for replay in ("golden_replay", "special_replay"):
        if replay in results and (results[replay]["mismatches"] or results[replay]["other_positions_nonzero"]):
            failures.append(f"{replay} mismatches={results[replay]['mismatches']} other_positions_nonzero={results[replay]['other_positions_nonzero']}")
    if "sweep" in results:
        for mname, count in results["sweep"]["total_mismatches_by_model"].items():
            if count and mname in declared:
                failures.append(f"sweep mismatches vs {mname}={count}")
            elif count:
                findings.append(f"sweep mismatches vs {mname}={count}")
    if "instances_replay" in results:
        ir = results["instances_replay"]
        tu, vu = ir["tu-k16"], ir["vu-k1536"]
        if tu["mismatches"] or tu["other_positions_nonzero"]:
            failures.append(f"instances tu-k16 mismatches={tu['mismatches']} other_positions_nonzero={tu['other_positions_nonzero']}")
        if vu["accumulator_mismatches"] or vu["y_mismatches"] or vu["other_positions_nonzero"]:
            failures.append(f"instances vu-k1536 accumulator_mismatches={vu['accumulator_mismatches']} y_mismatches={vu['y_mismatches']} "
                            f"other_positions_nonzero={vu['other_positions_nonzero']}")
    if "sass" in results and not results["sass"]["ok"]:
        failures.append(f"sass tile_kernel_mma_count={results['sass'].get('tile_kernel_mma_count')}")
    results["failures"] = failures
    results["hypothesis_findings"] = findings
    (out / "probe_results.json").write_text(json.dumps(results, indent=1))

    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id:
        checked_something = any(k in results for k in ("golden_replay", "special_replay", "instances_replay")) or ("sweep" in results and declared)
        status = "failed" if failures else ("passed" if checked_something else "not_run")
        fingerprint = {
            "probe": "tools/tc_probe/tc_probe.py",
            "instruction": ins.id,
            "ptx": ins.ptx,
            "arch": arch,
            "sku": identity["device"],
            "device": identity["device"],
            "compute_capability": identity["compute_capability"],
            "driver": identity["driver"],
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
        evidence = {
            "instruction": ins.id,
            "arch": arch,
            "sku": identity["device"],
            "driver": identity["driver"],
            "nvcc": nvcc,
            "seed": args.seed,
            "n": args.n_random,
            "layout_check": results["layout_check"],
            "models": {m.name: {"declared": m.declared, "pipeline": m.pipeline.name, "product": m.product or m.pipeline.dtype} for m in ins.models},
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
        for replay in ("golden_replay", "special_replay"):
            if replay in results:
                evidence[replay] = {k: v for k, v in results[replay].items() if k != "first_mismatches"}
                evidence[replay]["first_mismatches"] = results[replay]["first_mismatches"]
                measurements.append({"name": f"{replay}.records", "value": results[replay]["records"], "unit": "records"})
                measurements.append({"name": f"{replay}.mismatches", "value": results[replay]["mismatches"], "unit": "records"})
        if "sass" in results:
            evidence["sass"] = {k: v for k, v in results["sass"].items() if k != "functions"}
            measurements.append({"name": "tile_kernel_mma_count", "value": results["sass"].get("tile_kernel_mma_count"), "unit": "instructions"})
        evidence["hypothesis_findings"] = findings
        artifacts = []
        if "sweep" in results:
            artifacts = ["capture_sample.json"] + [f"tiles_{name}.npz" for name in results["sweep"]["families"]]
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
    print("FAILURES" if failures else "ALL HARDWARE WORDS REPRODUCED" if not findings else "RECORDED (hypothesis findings)",
          failures or findings, f"{results['seconds']}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
