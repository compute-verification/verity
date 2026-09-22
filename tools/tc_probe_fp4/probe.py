"""Blackwell FP4 tensor-core probe (lanes fp4-5090, fp4-re): one ``mma.sync`` per tile on an RTX 5090, raw bits captured and
compared bit for bit against ``verity.ml.tc.models`` (``BLACKWELL_SM120_NVF4`` / ``BLACKWELL_SM120_MXF4``, the primary model and
the validation gate) and against ``model5.py`` (autoproof's five-way align-add, kept as the control whose D16/D17/D18 mismatches
the verity rule must remove).  Families under ``families_fp4_re`` are lane fp4-re's accumulator families; ``specials`` is
record-only (NaN / inf accumulators, sums leaving binary32: the hardware words are summarised, no model is asserted).

    python tools/tc_probe_fp4/probe.py --instruction nvf4 --out $RESEARCH_RUN_DIR --sass-check \
        --replay-autoproof $RESEARCH_RUN_DIR/inputs/probe3_1783788526.json --sweep --n-random 4096

Instructions (``--instruction``): ``nvf4`` = ``mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col
.f32.e2m1.e2m1.f32.ue4m3`` (id ``sm120.mma.m16n8k64.e2m1.nvf4``); ``mxf4`` = the same with ``.scale_vec::2X ... .ue8m0``
(``sm120.mma.m16n8k64.e2m1.mxf4``, capture + model5-with-ue8m0-shift as a hypothesis, no hot clip); ``f8f6f4`` = the unscaled
``mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32`` (``sm120.mma.m16n8k32.e2m1.f8f6f4``, autoproof probe2
replay only); ``bf16`` = ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` (``sm120.mma.m16n8k16.bf16``), compared with
``verity.ml.tc.silicon.tc_dot`` under BOTH ``AMPERE_BF16_M16N8K16`` and ``HOPPER_BF16_M16N8K16`` -- which agrees is the finding.

The kernel (``mma_fp4.cu``) is built with ``-arch=sm_120a``: ptxas refuses ``.block_scale`` / ``.kind::f8f6f4`` on plain
``sm_120`` ("not supported on .target sm_120"), the arch-specific target is required.  ``--sass-check`` runs ``cuobjdump -sass``
and records the number of MMA instructions per tile kernel (must be exactly one each; the run fails otherwise).

``--replay-autoproof FILE`` replays every ``sample_cases`` record of autoproof's ``probes/results/probe{2,3}_*.json`` (read-only
source repo, shipped with ``--send``): each dot at its own tile position (r % 16, (r // 16) % 8), the rest of the tile zero, a zero
accumulator, K/64 (probe3) or K/32 (probe2) dependent launches with the hardware's D fed back as C (ascending K, as autoproof
chained), and requires the hardware word to equal ``silicon_c_bits`` -- the reproduction gate.  Any mismatch is a finding: it is
recorded, nothing is tuned.

``--sweep`` runs the families (fixed seed) and compares every output element with model5: random operands with random scales and
random accumulators; random with zero accumulators; uniform encodings (codes, scales and accumulator bits all uniform); the full
UE4M3 scale-code sweep (every valid code 0x00..0x7E, one and both operands); the hot-group-clip regime (one live group with both
scale codes in 0x78..0x7E, the other groups on subnormal scales with bits at 2^-20, tiny or zero accumulators); all-hot and
one-operand-hot controls; large cancellation (two groups cancel exactly at the top of the window while small terms sit below
E1-41: the window-width probe); accumulator-window probe (large accumulators over tiny terms: W2); tiny accumulators; signed zero.

MEASURED 2026-09-22 on vy-5090 (RTX 5090, driver 570.153.02, nvcc 12.8.93; runs r20260922-182011-88f8 nvf4,
r20260922-182132-d6db f8f6f4, r20260922-182548-e654 mxf4, r20260922-182911-1f10 bf16):
  * autoproof replay: probe3 1601/1601 (K=64/128/256) and probe2 1400/1400 dots reproduced bit for bit -- model5 IS the silicon
    for zero-initialised ascending-K chains, on a different RTX 5090 and a different kernel.
  * nvf4 sweep, 1,736,704 elements, 263,781 hot firings: every family with a +0 accumulator agrees with model5 (0 mismatches over
    hot_clip_regime 131,072 / random_zero_acc 262,144 / scale sweeps / acc_window / tiny_acc).  Two regimes autoproof never probed
    (its accumulators were always +0 or the running sum, i.e. the largest addend) disagree: (1) a -0 accumulator with all-zero
    products returns +0 on silicon, model5 keeps -0 (16,328/16,328 such elements); (2) an incoming accumulator of the OPPOSITE
    sign to the result whose msb lies 37..42 binades below E1 -- at the alignment window's edge -- comes out 1 ulp (up to 8 in the
    cancellation family) larger in magnitude than model5's truncated-toward-zero sum (12,790 of 1,507,328 nonzero-acc elements,
    100% opposite-sign).  Neither a narrower accumulator window nor an anchor computed from scale exponents explains all of them;
    the accumulator's alignment path is the open question, not the 42/36-bit windows, the RZ normalisation or the hot clip.
  * mxf4 (ue8m0, .2X): accepted by ptxas on sm_120a (SASS OMMA.SF...E8); model5 with 4 groups of 16 under a per-32 shift is the
    better structure (2 groups of 32: 32,691/32,768 wrong in large_cancellation vs 126) but not exact: 798/131,072 zero-acc
    mismatches.  Capture only; hypothesis open.
  * bf16 m16n8k16 on the RTX 5090: HOPPER_BF16_M16N8K16 reproduces 204,800/204,800 words; AMPERE_BF16_M16N8K16 fails 48,569.
    The consumer Blackwell part keeps the Hopper BF16 pipeline (one 16-wide group, 26-bit alignment).

Outputs under ``--out``: ``probe_results.json`` (identity, sass counts, replay and per-family counts, first mismatches),
``capture_sample.json`` (records sampled per family: acc, a/b codes, scales, d as hex), ``tiles_<family>.npz`` (the full capture),
``result.json`` (``kind: tc-evidence/v1``; meta per store README §11: instruction, arch, sku, driver, nvcc, families, counts,
model, model_agrees, first_mismatches (<= 20), seed, n; validation passes iff every replay and model mismatch count is zero).
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
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_VERITY_SRC = HERE.parents[1] / "packages" / "verity" / "src"  # the bf16 comparison imports verity.ml.tc.silicon from the shipped tree
if _VERITY_SRC.is_dir() and str(_VERITY_SRC) not in sys.path:
    sys.path.insert(0, str(_VERITY_SRC))
import model5  # noqa: E402

MAX_TILES_PER_CALL = 1 << 14

# Families whose accumulators or sums lie outside every model's domain (NaN / inf accumulators, sums leaving binary32): the
# hardware words are recorded and summarised, they never gate validation.
RECORD_ONLY_FAMILIES = {"specials"}

PTX = {
    "nvf4": "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
    "mxf4": "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
    "f8f6f4": "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32",
    "bf16": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
}
INSTRUCTION_ID = {
    "nvf4": "sm120.mma.m16n8k64.e2m1.nvf4",
    "mxf4": "sm120.mma.m16n8k64.e2m1.mxf4",
    "f8f6f4": "sm120.mma.m16n8k32.e2m1.f8f6f4",
    "bf16": "sm120.mma.m16n8k16.bf16",
}
KERNEL = {"nvf4": "nvfp4_tiles_kernel", "mxf4": "mxfp4_tiles_kernel", "f8f6f4": "fp4raw_tiles_kernel", "bf16": "bf16_tiles_kernel"}
K_OF = {"nvf4": 64, "mxf4": 64, "f8f6f4": 32, "bf16": 16}
NBLK = {"nvf4": 4, "mxf4": 2, "f8f6f4": 0, "bf16": 0}
SCALE_KIND = {"nvf4": "ue4m3", "mxf4": "ue8m0", "f8f6f4": "none"}
UE4M3_VALID = np.array([c for c in range(0x80) if c != 0x7F], dtype=np.uint8)  # 127 codes; 0x7F is NaN


# --- build -----------------------------------------------------------------------------------------------------------------------


def nvcc_bin() -> str:
    return shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"


def build(arch: str, out: Path) -> tuple[Path, str]:
    so = out / f"libmma_fp4_{arch}.so"
    nvcc = subprocess.run([nvcc_bin(), "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-2:]
    subprocess.run([nvcc_bin(), "-O2", f"-arch={arch}", "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(HERE / "mma_fp4.cu")], check=True)
    return so, " / ".join(line.strip() for line in nvcc)


def sass_check(so: Path) -> dict:
    """MMA instructions per tile kernel from ``cuobjdump -sass``: {kernel: {count, opcodes}}."""
    cuobjdump = str(Path(nvcc_bin()).parent / "cuobjdump")
    text = subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    out: dict = {}
    fn = None
    for line in text.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            fn = m.group(1)
            for k in KERNEL.values():
                if k in fn:
                    fn = k
            out[fn] = {"mma_count": 0, "opcodes": []}
            continue
        if fn and re.search(r"\b[A-Z]*MMA\b|\bOMMA\.|\bQMMA\.|\bHMMA\.", line):
            op = line.split("*/")[1].split()[0] if "*/" in line else line.strip()
            out[fn]["mma_count"] += 1
            out[fn]["opcodes"].append(op)
    return out


class Probe:
    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        P = ctypes.c_void_p
        for name, argc in (("mma_nvfp4_tiles", 6), ("mma_mxfp4_tiles", 6), ("mma_fp4raw_tiles", 4), ("mma_bf16_tiles", 4)):
            fn = getattr(self.lib, name)
            fn.restype = ctypes.c_int
            fn.argtypes = [P] * argc + [ctypes.c_int]
        self.lib.has_mxf4.restype = ctypes.c_int
        ident = self.lib.device_identity
        ident.restype = ctypes.c_int
        ident.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int]

    def identity(self) -> dict:
        major, minor = ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        rc = self.lib.device_identity(ctypes.byref(major), ctypes.byref(minor), name, 256)
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in cudaGetDeviceProperties")
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,uuid", "--format=csv,noheader"], capture_output=True, text=True)
        fields = [s.strip() for s in smi.stdout.strip().split(",")] if smi.returncode == 0 else []
        return {"device": name.value.decode(), "compute_capability": f"{major.value}.{minor.value}",
                "nvidia_smi": smi.stdout.strip() if smi.returncode == 0 else None,
                "driver": fields[1] if len(fields) > 1 else None, "gpu_uuid": fields[3] if len(fields) > 3 else None,
                "hostname": platform.node(), "has_mxf4": bool(self.lib.has_mxf4())}

    def tiles(self, instr: str, A, Bt, C, SFA=None, SFB=None) -> np.ndarray:
        """One MMA per tile.  ``A`` (n,16,K) / ``Bt`` (n,8,K) codes (u8; bf16: u16 words), ``SFA`` (n,16,nblk) / ``SFB`` (n,8,nblk) u8,
        ``C`` (n,16,8) FP32 words -> ``D`` (n,16,8)."""
        n = A.shape[0]
        k, nblk = K_OF[instr], NBLK[instr]
        dt = np.uint16 if instr == "bf16" else np.uint8
        A = np.ascontiguousarray(A, dtype=dt).reshape(n, 16, k)
        Bt = np.ascontiguousarray(Bt, dtype=dt).reshape(n, 8, k)
        C = np.ascontiguousarray(C, dtype=np.uint32).reshape(n, 16, 8)
        if nblk:
            SFA = np.ascontiguousarray(SFA, dtype=np.uint8).reshape(n, 16, nblk)
            SFB = np.ascontiguousarray(SFB, dtype=np.uint8).reshape(n, 8, nblk)
        D = np.zeros((n, 16, 8), dtype=np.uint32)
        for s in range(0, n, MAX_TILES_PER_CALL):
            e = min(n, s + MAX_TILES_PER_CALL)
            d = np.zeros((e - s, 16, 8), dtype=np.uint32)
            a, b, c = A[s:e], Bt[s:e], C[s:e]
            if instr == "nvf4":
                rc = self.lib.mma_nvfp4_tiles(a.ctypes.data, b.ctypes.data, SFA[s:e].ctypes.data, SFB[s:e].ctypes.data, c.ctypes.data, d.ctypes.data, e - s)
            elif instr == "mxf4":
                rc = self.lib.mma_mxfp4_tiles(a.ctypes.data, b.ctypes.data, SFA[s:e].ctypes.data, SFB[s:e].ctypes.data, c.ctypes.data, d.ctypes.data, e - s)
            elif instr == "f8f6f4":
                rc = self.lib.mma_fp4raw_tiles(a.ctypes.data, b.ctypes.data, c.ctypes.data, d.ctypes.data, e - s)
            else:
                rc = self.lib.mma_bf16_tiles(a.ctypes.data, b.ctypes.data, c.ctypes.data, d.ctypes.data, e - s)
            if rc != 0:
                raise RuntimeError(f"CUDA error {rc} in the {instr} tile kernel")
            D[s:e] = d
        return D


# --- encodings ---------------------------------------------------------------------------------------------------------------------


def f32_bits(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def random_f32_bits(rng, shape, exp_lo=1, exp_hi=254) -> np.ndarray:
    n = int(np.prod(shape))
    e = rng.integers(exp_lo, exp_hi + 1, size=n).astype(np.uint32)
    m = rng.integers(0, 1 << 23, size=n).astype(np.uint32)
    s = rng.integers(0, 2, size=n).astype(np.uint32)
    return ((s << 31) | (e << 23) | m).reshape(shape)


def e2m1_encode_ints(v: np.ndarray) -> np.ndarray:
    """Small integers in {-6..6} that are E2M1 values (0, +-1, +-2, +-3, +-4, +-6) -> codes."""
    table = {0: 0, 1: 2, 2: 4, 3: 5, 4: 6, 6: 7}
    out = np.zeros(v.shape, np.uint8)
    for val, code in table.items():
        out[v == val] = code
        out[v == -val] = code | 8 if val else 0
    return out


def ue4m3_pow2(e_field: int) -> int:
    """UE4M3 code for 2^(e_field-7) (mantissa field 0): the exact power-of-two scales of the layout check."""
    return e_field << 3


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = f32_bits(x).astype(np.uint64)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def random_bf16_bits(rng, shape, exp_lo=1, exp_hi=254) -> np.ndarray:
    n = int(np.prod(shape))
    e = rng.integers(exp_lo, exp_hi + 1, size=n).astype(np.uint16)
    m = rng.integers(0, 128, size=n).astype(np.uint16)
    s = rng.integers(0, 2, size=n).astype(np.uint16)
    return ((s << 15) | (e << 7) | m).reshape(shape)


# --- layout checks (exact small integers: any fragment / scale-lane error shows up) -----------------------------------------------


def layout_check(probe: Probe, instr: str) -> str:
    rng = np.random.default_rng(1)
    n = 64
    k = K_OF[instr]
    if instr == "bf16":
        a = rng.integers(-8, 9, (n, 16, k)).astype(np.float32)
        b = rng.integers(-8, 9, (n, 8, k)).astype(np.float32)
        c = rng.integers(-1000, 1001, (n, 16, 8)).astype(np.float32)
        want = f32_bits(c + np.einsum("nik,njk->nij", a, b))
        got = probe.tiles(instr, to_bf16_bits(a), to_bf16_bits(b), f32_bits(c))
    else:
        vals = np.array([0, 1, 2, 3, 4, 6, -1, -2, -3, -4, -6])
        a = vals[rng.integers(0, len(vals), (n, 16, k))]
        b = vals[rng.integers(0, len(vals), (n, 8, k))]
        c = rng.integers(-1000, 1001, (n, 16, 8)).astype(np.float64)
        nblk = NBLK[instr]
        if nblk:
            blk = k // nblk
            # distinct power-of-two scales per (row, block) and (col, block) in 2^-2..2^3: any lane/byte misrouting changes the
            # result, and every exact sum stays below 24 significant bits (|term| < 2^17, lowest bit 2^-4) so FP32 holds it exactly
            # (a wider range made 3/8192 mxf4 elements need > 24 bits: RNE reference vs RZ hardware, a false "layout" failure)
            if instr == "nvf4":
                ea = rng.integers(5, 11, (n, 16, nblk))
                eb = rng.integers(5, 11, (n, 8, nblk))
                sfa, sfb = (ea << 3).astype(np.uint8), (eb << 3).astype(np.uint8)
                pa, pb = 2.0 ** (ea - 7), 2.0 ** (eb - 7)
            else:
                ea = rng.integers(125, 131, (n, 16, nblk))
                eb = rng.integers(125, 131, (n, 8, nblk))
                sfa, sfb = ea.astype(np.uint8), eb.astype(np.uint8)
                pa, pb = 2.0 ** (ea - 127), 2.0 ** (eb - 127)
            prod = np.einsum("nigk,njgk->nijg", a.reshape(n, 16, nblk, blk), b.reshape(n, 8, nblk, blk)).astype(np.float64)
            want = f32_bits(c + np.einsum("nijg,nig,njg->nij", prod, pa, pb))
            got = probe.tiles(instr, e2m1_encode_ints(a), e2m1_encode_ints(b), f32_bits(c), sfa, sfb)
        else:
            want = f32_bits(c + np.einsum("nik,njk->nij", a, b))
            got = probe.tiles(instr, e2m1_encode_ints(a), e2m1_encode_ints(b), f32_bits(c))
    if not np.array_equal(want, got):
        bad = int(np.count_nonzero(want != got))
        raise RuntimeError(f"fragment layout check failed for {instr}: {bad}/{want.size} elements differ from C + A.Bt on exact integer tiles")
    return "passed"


# --- families ------------------------------------------------------------------------------------------------------------------------


def families_nvf4(rng, n_random: int, scale_kind: str):
    """Yield (name, A (n,16,64) codes, Bt (n,8,64), SFA (n,16,nblk), SFB (n,8,nblk), C (n,16,8))."""
    k = 64
    nblk = 4 if scale_kind == "ue4m3" else 2

    def codes(shape):
        return rng.integers(0, 16, shape).astype(np.uint8)

    def scales(shape, lo=None, hi=None):
        if scale_kind == "ue4m3":
            return UE4M3_VALID[rng.integers(0, len(UE4M3_VALID), shape)]
        lo, hi = (107, 148) if lo is None else (lo, hi)
        return rng.integers(lo, hi, shape).astype(np.uint8)  # moderate E8M0 exponents (autoproof's random_scale), 0xFF NaN excluded

    def acc_random(n, lo=-30, hi=40):
        return f32_bits((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(lo, hi, (n, 16, 8)))).astype(np.float32))

    zeros = lambda n: np.zeros((n, 16, 8), np.uint32)

    n = n_random
    yield "random_random_scales", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk)), scales((n, 8, nblk)), acc_random(n)

    n = n_random // 2
    yield "random_zero_acc", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk)), scales((n, 8, nblk)), zeros(n)

    n = n_random // 4
    yield ("uniform_encodings", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk)), scales((n, 8, nblk)),
           random_f32_bits(rng, (n, 16, 8), 1, 250))

    if scale_kind == "ue4m3":
        # every valid UE4M3 code on every block: A sweeps codes row-wise, B sweeps col-wise (both operands), then one operand only
        n = n_random // 4
        idx = (np.arange(n)[:, None, None] * 64 + np.arange(16)[None, :, None] * 4 + np.arange(nblk)[None, None, :]) % 127
        sfa = UE4M3_VALID[idx]
        idxb = (np.arange(n)[:, None, None] * 32 + np.arange(8)[None, :, None] * 4 + np.arange(nblk)[None, None, :] + 61) % 127
        sfb = UE4M3_VALID[idxb]
        half = zeros(n)
        half[n // 2:] = acc_random(n - n // 2)
        yield "scale_sweep_both", codes((n, 16, k)), codes((n, 8, k)), sfa, sfb, half
        n = n_random // 8
        yield ("scale_sweep_a_only", codes((n, 16, k)), codes((n, 8, k)), UE4M3_VALID[idx[:n]],
               np.full((n, 8, nblk), 0x38, np.uint8), acc_random(n, -10, 10))  # B scale = 1.0

        # hot-group clip regime: block 0 of both operands hot (0x78..0x7E), blocks 1-3 on subnormal scales (codes 1..7, bits at
        # 2^-20); A block 0 sparse (few live products, small gd) so E1 stays below 22 and the clip decides the lsb.
        n = n_random // 4
        A = codes((n, 16, k))
        Bt = codes((n, 8, k))
        keep = rng.random((n, 16, 16)) < 0.15
        A[:, :, :16] = np.where(keep, A[:, :, :16], 0)
        sfa = rng.integers(1, 8, (n, 16, nblk)).astype(np.uint8)
        sfb = rng.integers(1, 8, (n, 8, nblk)).astype(np.uint8)
        sfa[:, :, 0] = rng.integers(0x78, 0x7F, (n, 16))
        sfb[:, :, 0] = rng.integers(0x78, 0x7F, (n, 8))
        acc = zeros(n)
        acc[n // 2:] = random_f32_bits(rng, (n - n // 2, 16, 8), 100, 120)  # tiny: 2^-27 .. 2^-7
        yield "hot_clip_regime", A, Bt, sfa, sfb, acc

        # hot on a random block, with the other blocks spanning the whole code range (mixed regime)
        n = n_random // 8
        sfa, sfb = scales((n, 16, nblk)), scales((n, 8, nblk))
        blk = rng.integers(0, nblk, n)
        sfa[np.arange(n), :, blk] = rng.integers(0x78, 0x7F, (n, 16))
        sfb[np.arange(n), :, blk] = rng.integers(0x78, 0x7F, (n, 8))
        yield "hot_one_block_mixed", codes((n, 16, k)), codes((n, 8, k)), sfa, sfb, acc_random(n, -30, 10)

        n = n_random // 8
        yield ("hot_all_blocks", codes((n, 16, k)), codes((n, 8, k)), rng.integers(0x78, 0x7F, (n, 16, nblk)).astype(np.uint8),
               rng.integers(0x78, 0x7F, (n, 8, nblk)).astype(np.uint8), acc_random(n, -10, 30))

        n = n_random // 8
        sfa = rng.integers(0x78, 0x7F, (n, 16, nblk)).astype(np.uint8)  # A hot, B not: control (no clip)
        yield "hot_a_only", codes((n, 16, k)), codes((n, 8, k)), sfa, rng.integers(1, 0x78, (n, 8, nblk)).astype(np.uint8), acc_random(n, -30, 10)

    # large cancellation (window-width probe): groups 0 and 1 cancel exactly at the top scale, groups 2-3 carry small terms
    n = n_random // 4
    A = codes((n, 16, k))
    Bt = codes((n, 8, k))
    A[:, :, 16:32] = A[:, :, :16]
    Bt[:, :, 16:32] = Bt[:, :, :16] ^ 8  # negate: every product of group 1 is minus the product of group 0
    Bt[:, :, 16:32][Bt[:, :, :16] == 0] = 0
    if scale_kind == "ue4m3":
        sfa, sfb = scales((n, 16, nblk)), scales((n, 8, nblk))
        top = rng.integers(0x60, 0x7F, (n, 16))
        sfa[:, :, 0] = top
        sfa[:, :, 1] = top
        topb = rng.integers(0x60, 0x7F, (n, 8))
        sfb[:, :, 0] = topb
        sfb[:, :, 1] = topb
        sfa[:, :, 2:] = rng.integers(0, 0x40, (n, 16, 2))
        sfb[:, :, 2:] = rng.integers(0, 0x40, (n, 8, 2))
    else:
        sfa, sfb = scales((n, 16, nblk), 137, 148), scales((n, 8, nblk), 137, 148)
        sfa[:, :, 1] = rng.integers(100, 117, (n, 16))
        sfb[:, :, 1] = rng.integers(100, 117, (n, 8))
    acc = zeros(n)
    acc[n // 2:] = acc_random(n - n // 2, -40, -10)
    yield "large_cancellation", A, Bt, sfa, sfb, acc

    # accumulator window (W2): large accumulators over small terms
    n = n_random // 8
    yield ("acc_window", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk), 100, 120), scales((n, 8, nblk), 100, 120),
           acc_random(n, 20, 60) if scale_kind == "ue4m3" else acc_random(n, 0, 40))

    n = n_random // 8
    yield ("tiny_acc", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk), 100, 120), scales((n, 8, nblk), 100, 120),
           random_f32_bits(rng, (n, 16, 8), 0, 30))

    n = 256
    A = np.where(rng.random((n, 16, k)) < 0.5, 0, 8).astype(np.uint8)  # +-0 operands
    Bt = codes((n, 8, k))
    A[n // 2:, :, :16] = 2  # +1.0
    Bt[n // 2:, :, :16] = np.where(np.arange(16) % 2 == 0, 5, 13).astype(np.uint8)  # +3, -3 alternating: exact cancellation
    acc = np.where(rng.random((n, 16, 8)) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)
    yield "signed_zero", A, Bt, scales((n, 16, nblk)), scales((n, 8, nblk)), acc

    yield from families_fp4_re(rng, n_random, scale_kind)


# --- lane fp4-re: the accumulator families (D16 sign of zero, D17 alignment anchor, D18 ue8m0 scales, domain edges) ---------------


def _scale_value(codes: np.ndarray, scale_kind: str) -> np.ndarray:
    """Exact float64 value of every scale code (UE4M3: (8+m) 2^(e-10) / m 2^-9; UE8M0: 2^(code-127))."""
    c = codes.astype(np.int64)
    if scale_kind == "ue4m3":
        e, m = (c >> 3) & 0xF, c & 7
        return np.where(e == 0, m * 2.0**-9, (8 + m) * np.exp2((e - 10).astype(np.float64)))
    return np.exp2((c - 127).astype(np.float64))


def _signed_pow2_acc(rng, shape, e_lo: int, e_hi: int, exact_frac: float = 0.5) -> np.ndarray:
    """FP32 words +-(1.f) 2^e, e uniform in [e_lo, e_hi], f = 0 with probability ``exact_frac`` else uniform 23 bits."""
    n = int(np.prod(shape))
    e = rng.integers(e_lo, e_hi + 1, n) + 127
    e = np.clip(e, 1, 254).astype(np.uint32)
    f = np.where(rng.random(n) < exact_frac, 0, rng.integers(0, 1 << 23, n)).astype(np.uint32)
    s = rng.integers(0, 2, n).astype(np.uint32)
    return ((s << 31) | (e << 23) | f).reshape(shape)


def families_fp4_re(rng, n_random: int, scale_kind: str):
    """The fp4-re families, both scale kinds: yield (name, A, Bt, SFA, SFB, C) like ``families_nvf4``."""
    k, nblk = 64, (4 if scale_kind == "ue4m3" else 2)
    gpb = 4 // nblk  # groups (of 16) per scale block
    ue4 = scale_kind == "ue4m3"

    def codes(shape):
        return rng.integers(0, 16, shape).astype(np.uint8)

    def nz_codes(shape):
        c = rng.integers(1, 16, shape)
        return np.where(c == 8, 1, c).astype(np.uint8)  # never a zero (0x0 / 0x8)

    def scales(shape, lo=None, hi=None):
        if ue4:
            if lo is None:
                return UE4M3_VALID[rng.integers(0, len(UE4M3_VALID), shape)]
            return UE4M3_VALID[rng.integers(lo, hi, shape)]
        lo, hi = (100, 150) if lo is None else (lo, hi)
        return rng.integers(lo, hi, shape).astype(np.uint8)

    def pm_zero(shape):
        return np.where(rng.random(shape) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)

    zeros = lambda n: np.zeros((n, 16, 8), np.uint32)

    # --- D16: the sign-of-zero matrix.  Accumulator +-0 x (a) all-zero A against random B, (b) random A against all-zero B,
    # (c) +-0 against +-0 operands, (d) every product -0 x -0, (e) pairs cancelling exactly inside every group, (f) group 0
    # cancelled exactly by group 1 under equal scales.  Scales random, including the zero UE4M3 code.
    n = 6 * 96
    A, Bt = codes((n, 16, k)), codes((n, 8, k))
    m = np.arange(n) // 96
    A[m == 0] = np.where(rng.random((int((m == 0).sum()), 16, k)) < 0.5, 0, 8)
    Bt[m == 1] = np.where(rng.random((int((m == 1).sum()), 8, k)) < 0.5, 0, 8)
    A[m == 2] = np.where(rng.random((int((m == 2).sum()), 16, k)) < 0.5, 0, 8)
    Bt[m == 2] = np.where(rng.random((int((m == 2).sum()), 8, k)) < 0.5, 0, 8)
    A[m == 3], Bt[m == 3] = 8, 8
    sel = m == 4
    A[sel, :, 1::2] = A[sel, :, 0::2]
    Bt[sel, :, 1::2] = np.where(Bt[sel, :, 0::2] == 0, 0, Bt[sel, :, 0::2] ^ 8)
    sel = m == 5
    A[sel, :, 16:32] = A[sel, :, :16]
    Bt[sel, :, 16:32] = np.where(Bt[sel, :, :16] == 0, 0, Bt[sel, :, :16] ^ 8)
    A[sel, :, 32:] = 0
    sfa, sfb = scales((n, 16, nblk)), scales((n, 8, nblk))
    if ue4:  # group 0 and group 1 share a block only in the ue8m0 layout; equalise the scales for the cross-group cancellation
        sfa[sel, :, 1] = sfa[sel, :, 0]
        sfb[sel, :, 1] = sfb[sel, :, 0]
    yield "signed_zero_matrix", A, Bt, sfa, sfb, pm_zero((n, 16, 8))

    # --- D17: accumulator offset walk.  One nonzero product per row/col in group g at slot s (both random per tile), the
    # accumulator +-(1.f) 2^(E1 - off) with off uniform in 20..60 binades below the term's msb E1, f exact (0) or random;
    # half the tiles ("many") add random small-scale products in the other groups.  Scales random (nonzero) on the leading block.
    n = n_random // 2
    A, Bt = np.zeros((n, 16, k), np.uint8), np.zeros((n, 8, k), np.uint8)
    g = rng.integers(0, 4, n)
    s = rng.integers(0, 16, n)
    kidx = g * 16 + s
    t = np.arange(n)
    A[t[:, None], np.arange(16)[None, :], kidx[:, None]] = nz_codes((n, 16))
    Bt[t[:, None], np.arange(8)[None, :], kidx[:, None]] = nz_codes((n, 8))
    many = rng.random(n) < 0.5
    other = np.ones((n, 16, k), bool)
    other[t[:, None], :, (g[:, None] * 16 + np.arange(16)[None, :])] = False  # keep the leading group clean
    fill_a, fill_b = codes((n, 16, k)), codes((n, 8, k))
    A = np.where(many[:, None, None] & other, fill_a, A)
    Bt = np.where(many[:, None, None] & other[:, :8, :], fill_b, Bt)
    sfa = scales((n, 16, nblk), 1, len(UE4M3_VALID)) if ue4 else scales((n, 16, nblk), 100, 150)
    sfb = scales((n, 8, nblk), 1, len(UE4M3_VALID)) if ue4 else scales((n, 8, nblk), 100, 150)
    small_a = scales((n, 16, nblk), 1, 0x30) if ue4 else scales((n, 16, nblk), 70, 110)
    small_b = scales((n, 8, nblk), 1, 0x30) if ue4 else scales((n, 8, nblk), 70, 110)
    blk = g // gpb
    lead_a = np.zeros((n, 16, nblk), bool)
    lead_a[t, :, blk] = True
    sfa = np.where(lead_a, sfa, np.where(many[:, None, None], small_a, sfa))
    sfb = np.where(lead_a[:, :8], sfb, np.where(many[:, None, None], small_b, sfb))
    va = model5.E2M1_NUM[A[t[:, None], np.arange(16)[None, :], kidx[:, None]].astype(np.int64)] / 2.0  # (n,16)
    vb = model5.E2M1_NUM[Bt[t[:, None], np.arange(8)[None, :], kidx[:, None]].astype(np.int64)] / 2.0  # (n,8)
    sa = _scale_value(sfa[t, :, blk], scale_kind)  # (n,16)
    sb = _scale_value(sfb[t, :, blk], scale_kind)  # (n,8)
    term = (va * sa)[:, :, None] * (vb * sb)[:, None, :]  # (n,16,8) exact in float64
    e1 = np.floor(np.log2(np.abs(term))).astype(np.int64)
    off = rng.integers(20, 61, (n, 16, 8))
    acc_e = np.clip(e1 - off + 127, 1, 254).astype(np.uint32)
    frac = np.where(rng.random((n, 16, 8)) < 0.5, 0, rng.integers(0, 1 << 23, (n, 16, 8))).astype(np.uint32)
    sign = rng.integers(0, 2, (n, 16, 8)).astype(np.uint32)
    yield "acc_offset_walk", A, Bt, sfa, sfb, ((sign << 31) | (acc_e << 23) | frac).astype(np.uint32)

    # --- D17: the widest group mantissas under the largest scales (nvf4: 6*6 x 16 under 448 x 448 = 19-bit group terms at
    # X = 8; model5's msb-anchored window and the scale-exponent anchor differ by 4 binades here) with accumulators around
    # the two grids (2^-24 .. 2^-8, both signs) and random large accumulators.
    n = n_random // 8
    A = np.where(rng.random((n, 16, k)) < 0.8, 0x7, 0xF).astype(np.uint8)
    Bt = np.where(rng.random((n, 8, k)) < 0.8, 0x7, 0xF).astype(np.uint8)
    A = np.where(rng.random((n, 16, k)) < 0.1, codes((n, 16, k)), A)
    if ue4:
        sfa, sfb = rng.integers(0x78, 0x7F, (n, 16, nblk)).astype(np.uint8), rng.integers(0x78, 0x7F, (n, 8, nblk)).astype(np.uint8)
    else:
        sfa, sfb = scales((n, 16, nblk), 128, 141), scales((n, 8, nblk), 128, 141)
    acc = _signed_pow2_acc(rng, (n, 16, 8), -24, -8)
    acc[n // 2:] = _signed_pow2_acc(rng, (n - n // 2, 16, 8), 20, 40)
    yield "wide_mantissa_top_scales", A, Bt, sfa, sfb, acc

    # --- anchor participation.  A tiny accumulator (+-2^e, e in -60..-30, exact or not) against (a) a "dead" group 0: nonzero
    # A nibbles, all-zero B nibbles, the largest scales -- no elementwise product, the accumulator must come back exact;
    # (b) a live group 0 whose dot cancels to zero under the largest scales -- the accumulator is truncated away (+0);
    # (c) nvf4 only: a live nonzero group 0 under a zero scale code on A, then on B -- no participation, exact accumulator.
    n = 3 * 128 if ue4 else 2 * 128
    A, Bt = np.zeros((n, 16, k), np.uint8), np.zeros((n, 8, k), np.uint8)
    m = np.arange(n) // 128
    A[:, :, :16] = nz_codes((n, 16, 16))
    Bt[m >= 1, :, :16] = nz_codes((int((m >= 1).sum()), 8, 16))
    sel = m == 1
    A[sel, :, 1:16:2] = A[sel, :, 0:16:2]
    Bt[sel, :, 1:16:2] = Bt[sel, :, 0:16:2] ^ 8
    if ue4:
        sfa, sfb = rng.integers(0x60, 0x7F, (n, 16, nblk)).astype(np.uint8), rng.integers(0x60, 0x7F, (n, 8, nblk)).astype(np.uint8)
        sel = m == 2
        half = sel & (np.arange(n) % 2 == 0)
        sfa[half, :, 0] = 0
        sfb[sel & ~half, :, 0] = 0
    else:
        sfa, sfb = scales((n, 16, nblk), 140, 170), scales((n, 8, nblk), 140, 170)
    yield "anchor_participation", A, Bt, sfa, sfb, _signed_pow2_acc(rng, (n, 16, 8), -60, -30)

    # --- FP32 subnormal accumulators (exponent field 0, random mantissa incl. 0) under small terms
    n = n_random // 8
    acc = random_f32_bits(rng, (n, 16, 8), 0, 0)
    if ue4:
        yield "acc_subnormal", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk), 0, 0x20), scales((n, 8, nblk), 0, 0x20), acc
    else:
        yield "acc_subnormal", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk), 40, 80), scales((n, 8, nblk), 40, 80), acc

    if not ue4:
        # --- D18 / D13: UE8M0 scale extremes.  Every code 0..254 on one operand against 0..115 on the other (ca + cb <= 369 keeps
        # |sum| < 2^127), then mirrored; random accumulators over 2^-126 .. 2^100 on half the tiles.  Sums below the FP32
        # subnormal grid decide the sign of the underflowed zero (BLACKWELL_SM120_MXF4 vs ..._UNDERFLOW_SIGNED).
        n = n_random // 2
        sfa = rng.integers(0, 255, (n, 16, nblk)).astype(np.uint8)
        sfb = rng.integers(0, 116, (n, 8, nblk)).astype(np.uint8)
        swap = np.arange(n) % 2 == 1
        sfa[swap] = rng.integers(0, 116, (int(swap.sum()), 16, nblk))
        sfb[swap] = rng.integers(0, 255, (int(swap.sum()), 8, nblk))
        acc = zeros(n)
        acc[n // 2:] = _signed_pow2_acc(rng, (n - n // 2, 16, 8), -126, 100, exact_frac=0.2)
        yield "ue8m0_scale_extremes", codes((n, 16, k)), codes((n, 8, k)), sfa, sfb, acc

        # --- D13: nonzero sums entirely below the subnormal grid, both signs, zero and tiny accumulators
        n = n_random // 8
        sfa = rng.integers(0, 60, (n, 16, nblk)).astype(np.uint8)
        sfb = rng.integers(0, 60, (n, 8, nblk)).astype(np.uint8)
        acc = zeros(n)
        acc[n // 2:] = random_f32_bits(rng, (n - n // 2, 16, 8), 0, 2)
        yield "ue8m0_underflow", codes((n, 16, k)), codes((n, 8, k)), sfa, sfb, acc

    # --- domain edges (record only): NaN / +-inf accumulators over random products; +-FLT_MAX accumulators with the largest
    # same-sign products (nvf4 cannot overflow on its own, mxf4 can: 6*6 x 64 under 2^127 x 2^127 scales)
    n = 128
    A, Bt = codes((n, 16, k)), codes((n, 8, k))
    special = np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FBFFFFF, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0xFF7FFFFF], np.uint32)
    acc = special[rng.integers(0, len(special), (n, 16, 8))]
    sel = np.arange(n) >= n // 2
    A[sel] = 0x7
    Bt[sel] = 0x7
    acc[sel] = np.where(rng.random((int(sel.sum()), 16, 8)) < 0.5, np.uint32(0x7F7FFFFF), np.uint32(0x7F000000))
    if ue4:
        sfa, sfb = scales((n, 16, nblk)), scales((n, 8, nblk))
        sfa[sel], sfb[sel] = 0x7E, 0x7E
    else:
        sfa, sfb = scales((n, 16, nblk)), scales((n, 8, nblk))
        sfa[sel] = rng.integers(200, 255, (int(sel.sum()), 16, nblk))
        sfb[sel] = rng.integers(200, 255, (int(sel.sum()), 8, nblk))
    yield "specials", A, Bt, sfa, sfb, acc


def families_bf16(rng, n_random: int):
    k = 16
    cast = lambda x: to_bf16_bits(x.astype(np.float32))
    n = n_random
    yield "randn_random_acc", cast(rng.standard_normal((n, 16, k))), cast(rng.standard_normal((n, 8, k))), None, None, \
        f32_bits((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(-20, 21, (n, 16, 8)))).astype(np.float32))
    n = n_random // 2
    yield "uniform_bits", random_bf16_bits(rng, (n, 16, k)), random_bf16_bits(rng, (n, 8, k)), None, None, random_f32_bits(rng, (n, 16, 8))
    n = n_random // 4
    a = rng.standard_normal((n, 16, k)) * 2.0**8
    b = rng.standard_normal((n, 8, k)) * 2.0**8
    a[:, :, 1::2] = a[:, :, 0::2]
    b[:, :, 1::2] = -b[:, :, 0::2]
    a = np.where(rng.random((n, 16, k)) < 0.15, rng.standard_normal((n, 16, k)) * 2.0**-6, a)
    yield "cancellation", cast(a), cast(b), None, None, f32_bits((rng.standard_normal((n, 16, 8)) * 2.0**-4).astype(np.float32))
    n = n_random // 4
    yield "subnormal", random_bf16_bits(rng, (n, 16, k), 0, 3), random_bf16_bits(rng, (n, 8, k), 0, 3), None, None, random_f32_bits(rng, (n, 16, 8), 0, 3)


# --- the model, per tile, in processes ---------------------------------------------------------------------------------------------

_CTX: dict = {}


PRIMARY_MODEL = {"nvf4": "verity", "mxf4": "verity", "f8f6f4": "verity", "bf16": None}


def verity_models(instr: str) -> dict:
    """The verity ``BlockScaledAlignAdd`` instances evaluated for ``instr``: ``verity`` is the registry's model, the others are
    named alternatives (mxf4: the sign of an underflowed zero, log entry D13)."""
    from verity.ml.tc import models as M
    if instr == "nvf4":
        return {"verity": M.BLACKWELL_SM120_NVF4}
    if instr == "mxf4":
        return {"verity": M.BLACKWELL_SM120_MXF4, "verity_underflow_signed": M.BLACKWELL_SM120_MXF4_UNDERFLOW_SIGNED}
    if instr == "f8f6f4":
        return {"verity": M.BlockScaledAlignAdd(
            name="probe_unscaled_fp4_m16n8k32", arch="sm_120", dtype="e2m1", operand_bits=4, scale=None, k_per_instruction=32,
            group=16, frac_bits=M.BLACKWELL_SM120_NVF4.frac_bits, acc_window=M.BLACKWELL_SM120_NVF4.acc_window,
            underflow_zero="positive", validated="probe instance: the nvf4 adder on unscaled e2m1 (unit weight 2^-2)")}
    return {}


def _init(instr: str, group: int):
    _CTX["instr"] = instr
    _CTX["group"] = group
    try:
        if instr == "bf16":
            from verity.ml.tc import silicon as S
            _CTX["pipes"] = {"ampere_bf16_m16n8k16": S.AMPERE_BF16_M16N8K16, "hopper_bf16_m16n8k16": S.HOPPER_BF16_M16N8K16}
            _CTX["tc_dot"] = S.tc_dot
        else:
            _CTX["verity"] = verity_models(instr)
    except Exception as e:  # noqa: BLE001 - a failing initializer would make the pool respawn workers forever; fail the chunk instead
        _CTX["init_error"] = repr(e)


def _eval_chunk(args):
    """Every model's word per element of the chunk: ({model: D (n,16,8)}, {model: domain-error mask})."""
    if "init_error" in _CTX:
        raise RuntimeError(f"model worker failed to initialize: {_CTX['init_error']}")
    A, Bt, SFA, SFB, C = args
    n = A.shape[0]
    instr = _CTX["instr"]
    if instr == "bf16":
        outs = {name: np.zeros((n, 16, 8), np.uint32) for name in _CTX["pipes"]}
        errs = {name: np.zeros((n, 16, 8), np.bool_) for name in _CTX["pipes"]}
        for t in range(n):
            a_rows = [A[t, i].tolist() for i in range(16)]
            b_rows = [Bt[t, j].tolist() for j in range(8)]
            for i in range(16):
                for j in range(8):
                    for name, pipe in _CTX["pipes"].items():
                        try:
                            outs[name][t, i, j] = _CTX["tc_dot"](pipe, int(C[t, i, j]), a_rows[i], b_rows[j])
                        except Exception:  # noqa: BLE001
                            errs[name][t, i, j] = True
        return outs, errs
    out5 = np.zeros((n, 16, 8), np.uint32)
    err5 = np.zeros((n, 16, 8), np.bool_)
    empty = np.zeros((16, 0), np.uint8)
    for t in range(n):
        out5[t], err5[t] = model5.tile_eval(A[t], Bt[t], SFA[t] if SFA is not None else empty, SFB[t] if SFB is not None else empty[:8],
                                            C[t], SCALE_KIND[instr], _CTX["group"])
    outs = {"model5": out5}
    errs = {"model5": err5}
    vm = _CTX["verity"]
    for name in vm:
        outs[name] = np.zeros((n, 16, 8), np.uint32)
        errs[name] = np.zeros((n, 16, 8), np.bool_)
    Al, Bl, Cl = A.tolist(), Bt.tolist(), C.tolist()
    SAl = SFA.tolist() if SFA is not None else None
    SBl = SFB.tolist() if SFB is not None else None
    for t in range(n):
        for i in range(16):
            sa = SAl[t][i] if SAl is not None else ()
            for j in range(8):
                sb = SBl[t][j] if SBl is not None else ()
                for name, model in vm.items():
                    try:
                        outs[name][t, i, j] = model.step_scaled(Cl[t][i][j], Al[t][i], Bl[t][j], sa, sb)
                    except Exception:  # noqa: BLE001 - outside the model's domain (non-finite accumulator, sum leaving binary32)
                        errs[name][t, i, j] = True
    return outs, errs


def model_eval(pool, A, Bt, SFA, SFB, C, procs):
    n = A.shape[0]
    chunks = max(1, min(n // 16 + 1, procs * 8))
    idx = np.array_split(np.arange(n), chunks)
    parts = pool.map(_eval_chunk, [(A[i], Bt[i], SFA[i] if SFA is not None else None, SFB[i] if SFB is not None else None, C[i]) for i in idx])
    outs = {name: np.concatenate([p[0][name] for p in parts]) for name in parts[0][0]}
    errs = {name: np.concatenate([p[1][name] for p in parts]) for name in parts[0][1]}
    return outs, errs


def observed_summary(D: np.ndarray, C: np.ndarray, err: np.ndarray) -> dict:
    """What the hardware returned where the model has no answer: word classes and the most frequent words."""
    d = D[err].astype(np.uint32)
    if d.size == 0:
        return {"elements": 0}
    exp = (d >> 23) & 0xFF
    frac = d & 0x7FFFFF
    cls = {"nan": int(((exp == 0xFF) & (frac != 0)).sum()), "pos_inf": int((d == 0x7F800000).sum()), "neg_inf": int((d == 0xFF800000).sum()),
           "finite": int((exp != 0xFF).sum())}
    words, counts = np.unique(d, return_counts=True)
    top = sorted(zip(counts.tolist(), words.tolist()), reverse=True)[:12]
    acc_in = {}
    for a in np.unique(C[err]).tolist()[:16]:
        sel = err & (C == a)
        w, c = np.unique(D[sel], return_counts=True)
        acc_in[f"{a:08x}"] = {f"{int(x):08x}": int(y) for x, y in sorted(zip(w.tolist(), c.tolist()), key=lambda p: -p[1])[:6]}
    return {"elements": int(d.size), "classes": cls, "top_words": {f"{w:08x}": c for c, w in top}, "by_accumulator": acc_in}


# --- records -----------------------------------------------------------------------------------------------------------------------


def hexrec(acc, a, b, d, sfa=None, sfb=None, bf16=False) -> dict:
    w = 4 if bf16 else 1
    rec = {"acc": f"{int(acc):08x}", "a": "".join(f"{int(x):0{w}x}" for x in a), "b": "".join(f"{int(x):0{w}x}" for x in b), "d": f"{int(d):08x}"}
    if sfa is not None:
        rec["sfa"] = "".join(f"{int(x):02x}" for x in sfa)
        rec["sfb"] = "".join(f"{int(x):02x}" for x in sfb)
    return rec


# --- autoproof replay ----------------------------------------------------------------------------------------------------------------


def replay_autoproof(probe: Probe, path: Path) -> dict:
    """Every ``sample_cases`` record at its own tile position, zero accumulator, dependent launches in ascending K (D -> C)."""
    data = json.loads(path.read_text())
    probe_no = int(data["probe"])
    instr = "nvf4" if probe_no == 3 else "f8f6f4"
    k_mma = K_OF[instr]
    cases = data["sample_cases"]
    by_k: dict[int, list[int]] = {}
    for r, c in enumerate(cases):
        by_k.setdefault(len(c["a_codes"]), []).append(r)
    mism = []
    other_nonzero = 0
    matched = 0
    t0 = time.time()
    for k_total, rows in sorted(by_k.items()):
        n = len(rows)
        steps = k_total // k_mma
        A = np.zeros((n, 16, k_total), np.uint8)
        Bt = np.zeros((n, 8, k_total), np.uint8)
        nblk_total = k_total // 16 if instr == "nvf4" else 0
        SFA = np.zeros((n, 16, max(nblk_total, 1)), np.uint8)
        SFB = np.zeros((n, 8, max(nblk_total, 1)), np.uint8)
        pos = [(r % 16, (r // 16) % 8) for r in rows]
        for t, r in enumerate(rows):
            i, j = pos[t]
            A[t, i] = cases[r]["a_codes"]
            Bt[t, j] = cases[r]["b_codes"]
            if instr == "nvf4":
                SFA[t, i] = cases[r]["a_scales"]
                SFB[t, j] = cases[r]["b_scales"]
        C = np.zeros((n, 16, 8), np.uint32)
        for s in range(steps):
            ks = slice(s * k_mma, (s + 1) * k_mma)
            if instr == "nvf4":
                bs = slice(s * 4, (s + 1) * 4)
                D = probe.tiles(instr, A[:, :, ks], Bt[:, :, ks], C, SFA[:, :, bs], SFB[:, :, bs])
            else:
                D = probe.tiles(instr, A[:, :, ks], Bt[:, :, ks], C)
            C = D
        for t, r in enumerate(rows):
            i, j = pos[t]
            want = int(cases[r]["silicon_c_bits"]) & 0xFFFFFFFF
            got = int(D[t, i, j])
            if got == want:
                matched += 1
            elif len(mism) < 20:
                mism.append({"index": r, "k": k_total, "autoproof_silicon": f"{want:08x}", "hardware": f"{got:08x}",
                             "a": "".join(f"{x:x}" for x in cases[r]["a_codes"]), "b": "".join(f"{x:x}" for x in cases[r]["b_codes"]),
                             "sfa": cases[r]["a_scales"], "sfb": cases[r]["b_scales"]})
            rest = D[t].copy()
            rest[i, j] = 0
            other_nonzero += int(np.count_nonzero(rest))
    return {"file": path.name, "probe": probe_no, "instruction": PTX[instr], "autoproof_target": data.get("target"),
            "autoproof_kernel_sha256": data.get("kernel_sha256"), "autoproof_gpu_uuid": data.get("gpu_uuid"),
            "cases": len(cases), "by_k": {str(k): len(v) for k, v in by_k.items()}, "matched": matched,
            "mismatches": len(cases) - matched, "first_mismatches": mism, "other_positions_nonzero": other_nonzero,
            "gpu_seconds": round(time.time() - t0, 2)}


# --- main --------------------------------------------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="nvf4", choices=sorted(PTX))
    ap.add_argument("--arch", default="sm_120a")
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay-autoproof", action="append", default=[], metavar="JSON")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-random", type=int, default=4096, help="tiles of the largest family (128 elements each)")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--group", type=int, default=16, help="model5 group width hypothesis (16 = the ue4m3 4X structure)")
    ap.add_argument("--procs", type=int, default=max(1, min(32, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--golden-per-family", type=int, default=60)
    ap.add_argument("--families", default=None, help="comma list (default all)")
    ap.add_argument("--sass-check", action="store_true")
    args = ap.parse_args()
    # the remote runner passes argv without a shell: `$RESEARCH_RUN_DIR` arrives literally and is expanded here
    args.replay_autoproof = [os.path.expandvars(p) for p in args.replay_autoproof]
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    instr = args.instruction

    so, nvcc = build(args.arch, out)
    probe = Probe(so)
    identity = probe.identity()
    identity["nvcc"] = nvcc
    identity["arch"] = args.arch
    if instr == "bf16":
        model_desc, model_src = "verity.ml.tc.silicon.tc_dot", "verity.ml.tc.silicon"
    else:
        model_desc = "verity.ml.tc.models." + {"nvf4": "BLACKWELL_SM120_NVF4", "mxf4": "BLACKWELL_SM120_MXF4",
                                               "f8f6f4": "BlockScaledAlignAdd(scale=None, k=32)"}[instr] + " (control: autoproof model5)"
        model_src = "verity.ml.tc.models + " + str(HERE / "model5.py")
    results: dict = {"schema": "fp4-re/tc-probe-fp4/v1", "instruction": PTX[instr], "instruction_id": INSTRUCTION_ID[instr],
                     "identity": identity, "model": model_desc, "model_source": model_src, "primary_model": PRIMARY_MODEL[instr]}
    print("device", json.dumps(identity), flush=True)
    failures = []

    if args.sass_check:
        sass = sass_check(so)
        results["sass"] = sass
        for kern, rec in sass.items():
            if rec["mma_count"] != 1:
                failures.append(f"sass: {kern} has {rec['mma_count']} MMA instructions")
        print("sass", json.dumps(sass), flush=True)

    if instr == "mxf4" and not identity["has_mxf4"]:
        failures.append("mxf4 kernel not compiled")
    else:
        results["layout_check"] = layout_check(probe, instr)
        print("layout_check", instr, results["layout_check"], flush=True)

    replays = []
    for p in args.replay_autoproof:
        rep = replay_autoproof(probe, Path(p))
        replays.append(rep)
        if rep["mismatches"] or rep["other_positions_nonzero"]:
            failures.append(f"autoproof replay {rep['file']}: mismatches={rep['mismatches']} other_positions_nonzero={rep['other_positions_nonzero']}")
        print("replay", json.dumps({k: v for k, v in rep.items() if k != "first_mismatches"}), flush=True)
    if replays:
        results["autoproof_replay"] = replays

    fams: dict = {}
    if args.sweep and not (instr == "mxf4" and not identity["has_mxf4"]):
        only = set(args.families.split(",")) if args.families else None
        rng = np.random.default_rng(args.seed)
        gen = families_bf16(rng, args.n_random) if instr == "bf16" else families_nvf4(rng, args.n_random, SCALE_KIND[instr])
        capture = {"instruction": PTX[instr], "instruction_id": INSTRUCTION_ID[instr], "k": K_OF[instr], "scale_blocks": NBLK[instr],
                   "note": "acc/d FP32 words (hex); a/b operand codes in K order (e2m1 nibbles as hex digits; bf16 as 4-hex words); "
                           "sfa/sfb scale codes per K-block (hex bytes), as measured on the device in identity",
                   "identity": identity, "records": []}
        with mp.Pool(args.procs, initializer=_init, initargs=(instr, args.group)) as pool:
            for name, A, Bt, SFA, SFB, C in gen:
                if only is not None and name not in only:
                    continue
                t0 = time.time()
                D = probe.tiles(instr, A, Bt, C, SFA, SFB)
                t_gpu = time.time() - t0
                t0 = time.time()
                models, errs = model_eval(pool, A, Bt, SFA, SFB, C, args.procs)
                t_model = time.time() - t0
                fam = {"tiles": int(A.shape[0]), "elements": int(A.shape[0]) * 128, "record_only": name in RECORD_ONLY_FAMILIES,
                       "nonzero_accumulators": int(np.count_nonzero(C & 0x7FFFFFFF)),
                       "model_domain_errors": {m: int(e.sum()) for m, e in errs.items()},
                       "gpu_seconds": round(t_gpu, 2), "model_seconds": round(t_model, 1), "mismatches_vs_model": {}, "first_mismatches": {}}
                if SFA is not None and instr == "nvf4":
                    live = np.zeros((A.shape[0], 16, 8, 4), bool)
                    a_nz = (model5.E2M1_NUM[A.astype(np.int64)] != 0).reshape(-1, 16, 4, 16)
                    b_nz = (model5.E2M1_NUM[Bt.astype(np.int64)] != 0).reshape(-1, 8, 4, 16)
                    live = np.einsum("nigk,njgk->nijg", a_nz.astype(np.int64), b_nz.astype(np.int64)) > 0
                    hot = live & ((SFA.astype(np.int64) & 0x78) == 0x78)[:, :, None, :] & ((SFB.astype(np.int64) & 0x78) == 0x78)[:, None, :, :]
                    fam["hot_firings"] = int(np.count_nonzero(hot.any(-1)))
                acc_nz = (C & 0x7FFFFFFF) != 0
                fam["mismatch_split"] = {}
                for mname, model in models.items():
                    err = errs[mname]
                    mism = (model != D) | err
                    fam["mismatches_vs_model"][mname] = int(mism.sum())
                    fam["mismatch_split"][mname] = {"acc_zero": int((mism & ~acc_nz).sum()), "acc_nonzero": int((mism & acc_nz).sum()),
                                                    "acc_negative_zero": int((mism & (C == 0x80000000)).sum()),
                                                    "domain_errors": int(err.sum()), "wrong_word": int((mism & ~err).sum())}
                    first = []
                    for t, i, j in list(zip(*np.nonzero(mism)))[:20]:
                        rec = hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j], SFA[t, i] if SFA is not None else None,
                                     SFB[t, j] if SFB is not None else None, bf16=(instr == "bf16"))
                        rec["model"] = "error" if err[t, i, j] else f"{int(model[t, i, j]):08x}"
                        first.append(rec)
                    fam["first_mismatches"][mname] = first
                primary = PRIMARY_MODEL[instr]
                if primary in errs and errs[primary].any():
                    fam["observed_outside_model_domain"] = observed_summary(D, C, errs[primary])
                kw = {"A": A, "Bt": Bt, "C": C, "D": D, **{f"model_{m}": v for m, v in models.items()}}
                if SFA is not None:
                    kw["SFA"], kw["SFB"] = SFA, SFB
                np.savez_compressed(out / f"tiles_{name}.npz", **kw)
                flat = rng.choice(A.shape[0] * 128, size=min(args.golden_per_family, A.shape[0] * 128), replace=False)
                for f in sorted(flat.tolist()):
                    t, i, j = f // 128, (f % 128) // 8, f % 8
                    capture["records"].append({**hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j], SFA[t, i] if SFA is not None else None,
                                                        SFB[t, j] if SFB is not None else None, bf16=(instr == "bf16")), "family": name})
                fams[name] = fam
                print(name, json.dumps({k: v for k, v in fam.items() if k != "first_mismatches"}), flush=True)
        model_names = sorted({m for f in fams.values() for m in f["mismatches_vs_model"]})
        gated = {k: f for k, f in fams.items() if not f["record_only"]}
        results["sweep"] = {"seed": args.seed, "n_random": args.n_random, "group_hypothesis": args.group, "families": fams,
                            "total_elements": sum(f["elements"] for f in fams.values()),
                            "gated_elements": sum(f["elements"] for f in gated.values()),
                            "record_only_families": sorted(k for k, f in fams.items() if f["record_only"]),
                            "total_nonzero_accumulators": sum(f["nonzero_accumulators"] for f in fams.values()),
                            "total_hot_firings": sum(f.get("hot_firings", 0) for f in fams.values()),
                            "total_mismatches_vs_model": {m: sum(f["mismatches_vs_model"][m] for f in gated.values()) for m in model_names},
                            "record_only_mismatches_vs_model": {m: sum(f["mismatches_vs_model"][m] for f in fams.values() if f["record_only"])
                                                                for m in model_names}}
        (out / "capture_sample.json").write_text(json.dumps(capture, separators=(",", ":")))
        if instr == "bf16":
            agreeing = [m for m, c in results["sweep"]["total_mismatches_vs_model"].items() if c == 0]
            results["bf16_pipeline_finding"] = {"agreeing_pipelines": agreeing, "mismatches": results["sweep"]["total_mismatches_vs_model"]}
            if not agreeing:
                failures.append("bf16: neither AMPERE nor HOPPER pipeline reproduces every word")
        else:
            primary = PRIMARY_MODEL[instr]
            total = results["sweep"]["total_mismatches_vs_model"].get(primary, 0)
            if total:
                failures.append(f"sweep mismatches vs {primary}={total} (record-only families excluded)")
        print("sweep TOTAL", results["sweep"]["total_elements"], "elements,", results["sweep"]["total_mismatches_vs_model"],
              "record-only:", results["sweep"]["record_only_mismatches_vs_model"], flush=True)

    results["seconds"] = round(time.time() - t_all, 1)
    results["failures"] = failures
    (out / "probe_results.json").write_text(json.dumps(results, indent=1))

    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id:
        agree = {}
        first_mism: list = []
        if "sweep" in results:
            for m, c in results["sweep"]["total_mismatches_vs_model"].items():
                agree[m] = c == 0
            for fam in fams.values():
                for m, lst in fam["first_mismatches"].items():
                    for rec in lst:
                        if len(first_mism) < 20:
                            first_mism.append({"model": m, **rec})
        for rep in replays:
            agree[f"autoproof_replay:{rep['file']}"] = rep["mismatches"] == 0 and rep["other_positions_nonzero"] == 0
            for rec in rep["first_mismatches"]:
                if len(first_mism) < 20:
                    first_mism.append({"model": f"autoproof_replay:{rep['file']}", **rec})
        primary = PRIMARY_MODEL[instr]
        # the verdict: the primary model over the gated families and every autoproof replay; controls/alternatives are recorded
        verdict_keys = [k for k in agree if k == primary or k.startswith("autoproof_replay:")]
        model_agrees = all(agree[k] for k in verdict_keys) if verdict_keys else None
        result = {
            "schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": run_id,
            "instruction": INSTRUCTION_ID[instr], "ptx": PTX[instr], "arch": args.arch, "sku": identity["device"],
            "compute_capability": identity["compute_capability"], "driver": identity["driver"], "gpu_uuid": identity["gpu_uuid"],
            "nvcc": nvcc, "sass_mma_counts": {k: v["mma_count"] for k, v in results.get("sass", {}).items()},
            "families": {name: {"elements": f["elements"], "mismatches": f["mismatches_vs_model"], "hot_firings": f.get("hot_firings"),
                                "record_only": f["record_only"], "domain_errors": f["model_domain_errors"],
                                "observed_outside_model_domain": f.get("observed_outside_model_domain")} for name, f in fams.items()},
            "counts": {"elements": results.get("sweep", {}).get("total_elements", 0),
                       "gated_elements": results.get("sweep", {}).get("gated_elements", 0),
                       "replay_cases": sum(r["cases"] for r in replays), "replay_matched": sum(r["matched"] for r in replays),
                       "mismatches": results.get("sweep", {}).get("total_mismatches_vs_model", {}),
                       "record_only_mismatches": results.get("sweep", {}).get("record_only_mismatches_vs_model", {})},
            "model": results["model"], "primary_model": primary, "model_agrees": model_agrees, "agreement": agree,
            "first_mismatches": first_mism, "seed": args.seed, "n": args.n_random, "group_hypothesis": args.group,
            "bf16_pipeline_finding": results.get("bf16_pipeline_finding"),
            "workload_fingerprint": {"probe": "tools/tc_probe_fp4/probe.py", "instruction": INSTRUCTION_ID[instr], "arch": args.arch,
                                     "device": identity["device"], "seed": args.seed, "n_random": args.n_random,
                                     "replays": [Path(p).name for p in args.replay_autoproof]},
            "validation": {"status": "passed" if not failures else "failed",
                           "checker": "tc_probe_fp4 vs verity.ml.tc.models (control: model5) / tc_dot / autoproof vectors",
                           "detail": "; ".join(failures) or "every hardware word reproduced", "evidence": ["probe_results.json"]},
            "measurements": [{"name": "wall", "value": results["seconds"], "unit": "s"}],
            "measurement_files": ["probe_results.json"],
            "artifacts": (["capture_sample.json"] if "sweep" in results else []) + sorted(p.name for p in out.glob("tiles_*.npz")),
        }
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1))
        os.replace(tmp, out / "result.json")
    print("FAILURES" if failures else "ALL HARDWARE WORDS REPRODUCED", failures, f"{results['seconds']}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
