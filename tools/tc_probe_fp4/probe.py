"""Blackwell FP4 tensor-core probe (lane fp4-5090): one ``mma.sync`` per tile on an RTX 5090, raw bits captured and compared
bit for bit against ``model5.py`` (autoproof's five-way align-add model, a HYPOTHESIS until this probe reproduces it).

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
import model5  # noqa: E402

MAX_TILES_PER_CALL = 1 << 14

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
            # distinct power-of-two scales per (row, block) and (col, block): exact, and any lane/byte misrouting changes the result
            if instr == "nvf4":
                ea = rng.integers(4, 11, (n, 16, nblk))
                eb = rng.integers(4, 11, (n, 8, nblk))
                sfa, sfb = (ea << 3).astype(np.uint8), (eb << 3).astype(np.uint8)
                pa, pb = 2.0 ** (ea - 7), 2.0 ** (eb - 7)
            else:
                ea = rng.integers(122, 133, (n, 16, nblk))
                eb = rng.integers(122, 133, (n, 8, nblk))
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


def _init(instr: str, group: int):
    _CTX["instr"] = instr
    _CTX["group"] = group
    if instr == "bf16":
        from verity.ml.tc import silicon as S
        _CTX["pipes"] = {"ampere_bf16_m16n8k16": S.AMPERE_BF16_M16N8K16, "hopper_bf16_m16n8k16": S.HOPPER_BF16_M16N8K16}
        _CTX["tc_dot"] = S.tc_dot


def _eval_chunk(args):
    A, Bt, SFA, SFB, C = args
    n = A.shape[0]
    instr = _CTX["instr"]
    if instr == "bf16":
        outs = {name: np.zeros((n, 16, 8), np.uint32) for name in _CTX["pipes"]}
        err = np.zeros((n, 16, 8), np.bool_)
        for t in range(n):
            a_rows = [A[t, i].tolist() for i in range(16)]
            b_rows = [Bt[t, j].tolist() for j in range(8)]
            for i in range(16):
                for j in range(8):
                    for name, pipe in _CTX["pipes"].items():
                        try:
                            outs[name][t, i, j] = _CTX["tc_dot"](pipe, int(C[t, i, j]), a_rows[i], b_rows[j])
                        except Exception:  # noqa: BLE001
                            err[t, i, j] = True
        return outs, err
    out = np.zeros((n, 16, 8), np.uint32)
    err = np.zeros((n, 16, 8), np.bool_)
    empty = np.zeros((16, 0), np.uint8)
    for t in range(n):
        out[t], err[t] = model5.tile_eval(A[t], Bt[t], SFA[t] if SFA is not None else empty, SFB[t] if SFB is not None else empty[:8],
                                          C[t], SCALE_KIND[instr], _CTX["group"])
    return {"model5": out}, err


def model_eval(pool, A, Bt, SFA, SFB, C, procs):
    n = A.shape[0]
    chunks = max(1, min(n // 16 + 1, procs * 8))
    idx = np.array_split(np.arange(n), chunks)
    parts = pool.map(_eval_chunk, [(A[i], Bt[i], SFA[i] if SFA is not None else None, SFB[i] if SFB is not None else None, C[i]) for i in idx])
    outs = {name: np.concatenate([p[0][name] for p in parts]) for name in parts[0][0]}
    return outs, np.concatenate([p[1] for p in parts])


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
    results: dict = {"schema": "fp4-5090/tc-probe-fp4/v0", "instruction": PTX[instr], "instruction_id": INSTRUCTION_ID[instr],
                     "identity": identity, "model": "autoproof-model5" if instr != "bf16" else "verity.ml.tc.silicon.tc_dot",
                     "model_source": str(HERE / "model5.py") if instr != "bf16" else "verity.ml.tc.silicon"}
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
                models, err = model_eval(pool, A, Bt, SFA, SFB, C, args.procs)
                t_model = time.time() - t0
                fam = {"tiles": int(A.shape[0]), "elements": int(A.shape[0]) * 128,
                       "nonzero_accumulators": int(np.count_nonzero(C & 0x7FFFFFFF)), "model_domain_errors": int(err.sum()),
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
                    mism = (model != D) | err
                    fam["mismatches_vs_model"][mname] = int(mism.sum())
                    fam["mismatch_split"][mname] = {"acc_zero": int((mism & ~acc_nz).sum()), "acc_nonzero": int((mism & acc_nz).sum()),
                                                    "acc_negative_zero": int((mism & (C == 0x80000000)).sum())}
                    first = []
                    for t, i, j in list(zip(*np.nonzero(mism)))[:20]:
                        rec = hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j], SFA[t, i] if SFA is not None else None,
                                     SFB[t, j] if SFB is not None else None, bf16=(instr == "bf16"))
                        rec["model"] = "error" if err[t, i, j] else f"{int(model[t, i, j]):08x}"
                        first.append(rec)
                    fam["first_mismatches"][mname] = first
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
        results["sweep"] = {"seed": args.seed, "n_random": args.n_random, "group_hypothesis": args.group, "families": fams,
                            "total_elements": sum(f["elements"] for f in fams.values()),
                            "total_nonzero_accumulators": sum(f["nonzero_accumulators"] for f in fams.values()),
                            "total_hot_firings": sum(f.get("hot_firings", 0) for f in fams.values()),
                            "total_mismatches_vs_model": {m: sum(f["mismatches_vs_model"][m] for f in fams.values()) for m in model_names}}
        (out / "capture_sample.json").write_text(json.dumps(capture, separators=(",", ":")))
        if instr == "bf16":
            agreeing = [m for m, c in results["sweep"]["total_mismatches_vs_model"].items() if c == 0]
            results["bf16_pipeline_finding"] = {"agreeing_pipelines": agreeing, "mismatches": results["sweep"]["total_mismatches_vs_model"]}
            if not agreeing:
                failures.append("bf16: neither AMPERE nor HOPPER pipeline reproduces every word")
        else:
            total = results["sweep"]["total_mismatches_vs_model"].get("model5", 0)
            if total:
                failures.append(f"sweep mismatches vs model5={total}")
        print("sweep TOTAL", results["sweep"]["total_elements"], "elements,", results["sweep"]["total_mismatches_vs_model"], flush=True)

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
        primary = "model5" if instr != "bf16" else None
        model_agrees = all(agree.values()) if agree else None
        result = {
            "schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": run_id,
            "instruction": INSTRUCTION_ID[instr], "ptx": PTX[instr], "arch": args.arch, "sku": identity["device"],
            "compute_capability": identity["compute_capability"], "driver": identity["driver"], "gpu_uuid": identity["gpu_uuid"],
            "nvcc": nvcc, "sass_mma_counts": {k: v["mma_count"] for k, v in results.get("sass", {}).items()},
            "families": {name: {"elements": f["elements"], "mismatches": f["mismatches_vs_model"], "hot_firings": f.get("hot_firings")} for name, f in fams.items()},
            "counts": {"elements": results.get("sweep", {}).get("total_elements", 0),
                       "replay_cases": sum(r["cases"] for r in replays), "replay_matched": sum(r["matched"] for r in replays),
                       "mismatches": results.get("sweep", {}).get("total_mismatches_vs_model", {})},
            "model": results["model"], "primary_model": primary, "model_agrees": model_agrees, "agreement": agree,
            "first_mismatches": first_mism, "seed": args.seed, "n": args.n_random, "group_hypothesis": args.group,
            "bf16_pipeline_finding": results.get("bf16_pipeline_finding"),
            "workload_fingerprint": {"probe": "tools/tc_probe_fp4/probe.py", "instruction": INSTRUCTION_ID[instr], "arch": args.arch,
                                     "device": identity["device"], "seed": args.seed, "n_random": args.n_random,
                                     "replays": [Path(p).name for p in args.replay_autoproof]},
            "validation": {"status": "passed" if not failures else "failed", "checker": "tc_probe_fp4 vs model5 / tc_dot / autoproof vectors",
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
