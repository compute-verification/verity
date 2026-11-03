"""F2: one- and two-level Strassen over one NVFP4 group on the TF32 tensor core, against dense NVFP4 on
sm_120a, word for word.

A cheaper-computation search: Strassen's 2 x 2 recursion multiplies block matrices in 7 products instead of 8, so L levels over
the TF32 tensor core issue (7/8)^L of its MMAs, plus additions.  Within one NVFP4 group (16 consecutive k) a row of A carries one
scale and a column of B one scale, so the group's product of scaled values is diag(sa) (CA CB) diag(sb).  The route runs
Strassen on the E2M1 codes as integer halves (exact in TF32; every product's sum an integer below 2^14, exact in the FP32
accumulator), then feeds the group dots with the scales to the atom's align-add, as the atom does
(``verity.ml.circuits.silicon.models.BLACKWELL_SM120_NVF4``).  Strassen on the scaled values themselves (``scaled``, recorded, never gated) adds
rows and columns of different scales, and TF32 rounds those sums.

    python benchmarks/tc_probe_fp4/f2_strassen.py build --out DIR                        # the library and its SASS gate
    python benchmarks/tc_probe_fp4/f2_strassen.py run --build DIR --level l1 --out DIR2 --expect-uuid GPU-... --locked-mhz 2100
    python benchmarks/tc_probe_fp4/f2_strassen.py collect --build DIR --runs D1,D2 --out DIR3  # the price table over both levels

Kernels (f2_strassen.cu): ``dots`` gives each warp one tile per iteration, one group (K = 16) per tile:
- level 1: A 32 x 16 by B 16 x 16, in 7 m16n8k8 TF32 MMAs (of 8);
- level 2: A 64 x 16 by B 16 x 32, in 49 m16n8k4 (of 64).
The pre-added operands are register resident, the recombination is in FP32, and every group dot is consumed.  ``native`` is
the OMMA tile (m16n8k64, four groups) on the same tiles.  ``step`` is the atom's align-add on the four groups' Strassen dots: the
route's words.

Gates, before any timing (a failed gate writes no measurements):
- the device UUID and compute capability 12.0;
- the build's SASS gate: each Strassen loop holds its 7^L TF32 MMAs and no other MMA or memory access, the native loop NT
  OMMA, no spills, no FP32 ``.FTZ``;
- on the mixed, uniform and gaussian families, every group's Strassen dots equal the integer group dots on every element, and
  the step's words at chains of 1 and 3 equal the native atom's on every word and the pinned model's on a sample;
- after timing, the last iteration's dots and the checksum of every iteration's at the timed length.
``--corrupt-word`` flips one word: the run must then fail.

In the route's favour (``FAVOURS_STRASSEN``, in every record): the pre-additions are outside the loop (A's are amortized over
the N / TN column tiles, B's are fixed weights), and so is the codes' conversion to TF32; the operands are register resident,
as the native arm's; the align-add is not timed, so the dots price is a lower bound on the exact route's.  Labels: rates and
prices Measured; the MMA-only floors (7/8 and 49/64 of F1's TF32 prices) Derived.
"""
from __future__ import annotations

import argparse
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
import f1_rates as F1  # noqa: E402
import f3_lut as F3  # noqa: E402
import probe as P  # noqa: E402
import vec as V  # noqa: E402
from verity_config.settings import SHARED  # noqa: E402

ARCH = "sm_120a"
SCHEMA = "pouw-rtxpro/f2-strassen-sm120/v1"
SRC = HERE / "f2_strassen.cu"
HEADER = HERE / "nvf4_step.cuh"
NVCC_FLAGS = ["-O2", "-std=c++17"]  # no -ftz / fast-math; build refuses them
NATIVE_TPB = 128
LEVELS = {"l1": 1, "l2": 2}
FAMILIES, TIMED, GATE_LENGTHS = F3.FAMILIES, F3.TIMED, F3.GATE_LENGTHS
NATIVE_OPCODE, NATIVE_PRODUCTS = F3.NATIVE_OPCODE, F3.NATIVE_PRODUCTS
TF32_OPCODE = {1: "HMMA.1688.F32.TF32", 2: "HMMA.1684.F32.TF32"}
#: distinct Strassen tiles per launch (warps take tile w mod NTILES), and the column tiles they cycle through
NTILES, NCT = 256, 4
#: F1's TF32 prices in FP4 slots per product, m16n8k8 and m16n8k4 (F1's collect, Measured)
F1_TF32_SLOTS = {1: 8.0, 2: 16.0}
FAVOURS_STRASSEN = ("the pre-additions outside the timed loop: A's (U) are amortized over the N/TN column tiles, B's (V) are "
                    "preprocessing of fixed weights; so is the E2M1 codes' conversion to TF32",
                    "every product's operands resident in registers and re-used every iteration, as the native arm's",
                    "the align-add (the atom's scales and rounding) is not timed: the dots price is a lower bound on the exact route's")
MEM_OPS = {"LD", "LDG", "LDS", "LDL", "LDSM", "ST", "STG", "STS", "STL", "ATOM", "ATOMS", "RED"}

#: Strassen's bilinear algorithm on 2 x 2 blocks, quadrant q = 2 i + j (11, 12, 21, 22): M_p = (sum_q U[p, q] A_q)(sum_q V[p, q] B_q),
#: C_q = sum_p W[q, p] M_p (f2_strassen.cu's W1)
U1 = np.array([[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]], np.int64)
V1 = np.array([[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], np.int64)
W1 = np.array([[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]], np.int64)
LANE_G, LANE_T = np.arange(32) >> 2, np.arange(32) & 3


@dataclasses.dataclass(frozen=True)
class Shape:
    level: int

    @property
    def nb(self) -> int:
        return 1 << self.level

    @property
    def np_(self) -> int:
        return 7 ** self.level

    @property
    def kb(self) -> int:
        return 16 >> self.level

    @property
    def tm(self) -> int:
        return 16 * self.nb

    @property
    def tn(self) -> int:
        return 8 * self.nb

    @property
    def native_per_tile(self) -> int:
        return (self.tm // 16) * (self.tn // 8)


def coeffs(level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(U, V, W) of ``level`` levels: U (NP, NB, NB) over A's (row block, k block), V over B's (k block, column block), W (NB, NB, NP)
    over C's (row block, column block).  Level 2 is level 1 on level 1's blocks: p = 7 outer + inner, block index 2 outer + inner."""
    u, v, w = U1.reshape(7, 2, 2), V1.reshape(7, 2, 2), W1.reshape(2, 2, 7)
    if level == 1:
        return u, v, w
    def kron(x):
        return np.einsum("pij,qkl->pqikjl", x, x).reshape(49, 4, 4)
    return kron(u), kron(v), np.einsum("ijp,klq->ikjlpq", w, w).reshape(4, 4, 49)


def cu_w1(src: str | None = None) -> np.ndarray:
    """f2_strassen.cu's W1 table."""
    text = src if src is not None else SRC.read_text()
    body = re.search(r"constexpr int w\[4\]\[7\] = \{(.*?)\};", text).group(1)
    return np.array([[int(x) for x in row.split(",")] for row in re.findall(r"\{([-\d, ]+)\}", body)], np.int64)


# ---------------------------------------------------------------------------------------------------------------------- operands

def tile_rows(sh: Shape) -> np.ndarray:
    return np.arange(NTILES)[:, None] * sh.tm + np.arange(sh.tm)[None, :]


def tile_cols(sh: Shape) -> np.ndarray:
    return (np.arange(NTILES) % NCT)[:, None] * sh.tn + np.arange(sh.tn)[None, :]


def make_operands(family: str, sh: Shape, seed: int) -> F3.Operands:
    """F3's operand families on NTILES x TM rows of A and NCT x TN columns of B (one column group)."""
    return F3.make_operands(family, NTILES * sh.tm, 1, NCT * sh.tn, seed)


def group_halves(ops: F3.Operands, sh: Shape, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Group b of every tile as integer halves: A (T, TM, 16), B (T, 16, TN)."""
    a = F3.E2M1_HALVES[ops.acodes[:, 16 * b:16 * b + 16]].reshape(NTILES, sh.tm, 16)
    w = F3.E2M1_HALVES[ops.wcodes[0][:, 16 * b:16 * b + 16]][tile_cols(sh)]
    return a, w.transpose(0, 2, 1)


def scale_values(scales: np.ndarray) -> np.ndarray:
    m, e = V.ue4m3_dyadic(scales.astype(np.int64))
    return m * np.exp2(e.astype(np.float64))


def group_scaled(ops: F3.Operands, sh: Shape, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Group b's scaled values (E2M1 value x UE4M3 value, exact in binary64 and in TF32): A (T, TM, 16), B (T, 16, TN)."""
    a, w = group_halves(ops, sh, b)
    sa = scale_values(ops.ascale[:, b]).reshape(NTILES, sh.tm)
    sb = scale_values(ops.wscale[0][:, b])[tile_cols(sh)]
    return a / 2 * sa[:, :, None], w / 2 * sb[:, None, :]


def combos(a: np.ndarray, b: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Each product's pre-added operands: A (T, NP, 16, KB) = sum U[p] A blocks, B (T, NP, KB, 8) = sum V[p] B blocks."""
    sh, (u, v, _) = Shape(level), coeffs(level)
    t = a.shape[0]
    ab = a.reshape(t, sh.nb, 16, sh.nb, sh.kb)
    bb = b.reshape(t, sh.nb, sh.kb, sh.nb, 8)
    return np.einsum("pij,tiajb->tpab", u, ab), np.einsum("pij,tiajb->tpab", v, bb)


def a_positions(kb: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(row, k) of each A register per lane: m16n8k8 TF32 a0..a3 = (g, t), (g+8, t), (g, t+4), (g+8, t+4); m16n8k4 the first two."""
    g, t = LANE_G, LANE_T
    return [(g, t), (g + 8, t), (g, t + 4), (g + 8, t + 4)][:kb // 2]


def b_positions(kb: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(k, n) of each B register per lane: b0 = (t, g), b1 = (t+4, g) (m16n8k4: b0 only)."""
    g, t = LANE_G, LANE_T
    return [(t, g), (t + 4, g)][:kb // 4]


def a_frag(x: np.ndarray, kb: int) -> np.ndarray:
    return np.stack([x[:, :, r, c] for r, c in a_positions(kb)], -1)


def b_frag(x: np.ndarray, kb: int) -> np.ndarray:
    return np.stack([x[:, :, r, c] for r, c in b_positions(kb)], -1)


def tf32_rna(x: np.ndarray) -> np.ndarray:
    """binary64 to the nearest TF32 (10 fraction bits), ties away from zero (cvt.rna.tf32.f32; normal range)."""
    m, e = np.frexp(np.asarray(x, np.float64))
    return np.ldexp(np.sign(m) * np.floor(np.abs(m) * 2.0 ** 11 + 0.5), e - 11)


def tf32_words(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(tf32_rna(x).astype(np.float32).view(np.uint32))


def fragments(a: np.ndarray, b: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray, int]:
    """The kernel's afr / bfr words for one group, and how many pre-added operands TF32 had to round."""
    kb = Shape(level).kb
    ac, bc = combos(a, b, level)
    rounded = int(np.count_nonzero(tf32_rna(ac) != ac) + np.count_nonzero(tf32_rna(bc) != bc))
    return tf32_words(a_frag(ac, kb)), tf32_words(b_frag(bc, kb)), rounded


def tile_terms(ops: F3.Operands, sh: Shape) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per element (tile, row, column) and group, each (T, TM, TN, 4): the group dot (quarters), the scale mantissas' product,
    X = -2 + ea + eb, and participation, as the pinned model's ``group_terms``."""
    a = F3.E2M1_HALVES[ops.acodes].reshape(NTILES, sh.tm, 4, 16)
    w = F3.E2M1_HALVES[ops.wcodes[0]].reshape(-1, 4, 16)[tile_cols(sh)]
    gd = np.einsum("tibk,tjbk->tijb", a, w)
    anyp = np.einsum("tibk,tjbk->tijb", (a != 0).astype(np.int64), (w != 0).astype(np.int64)) > 0
    ma, ea = V.ue4m3_dyadic(ops.ascale.astype(np.int64))
    mb, eb = V.ue4m3_dyadic(ops.wscale[0].astype(np.int64))
    ma, ea = ma.reshape(NTILES, sh.tm, 4), ea.reshape(NTILES, sh.tm, 4)
    mb, eb = mb[tile_cols(sh)], eb[tile_cols(sh)]
    mm = ma[:, :, None, :] * mb[:, None, :, :]
    return gd, mm, ea[:, :, None, :] + eb[:, None, :, :] - 2, anyp & (mm != 0)


def tile_c0(ops: F3.Operands, sh: Shape) -> np.ndarray:
    return np.ascontiguousarray(ops.c0[tile_rows(sh)[:, :, None], tile_cols(sh)[:, None, :]], np.uint32)


def lane_sums(out: np.ndarray, sh: Shape) -> np.ndarray:
    """(T, 32, 4) u32: per tile, lane and C register, the sum of the bits of the lane's group dots over the tile's C blocks (the
    kernel's per-iteration checksum)."""
    blk, o = np.arange(sh.nb), np.arange(4)
    rows = 16 * blk[:, None, None, None] + LANE_G[None, None, :, None] + 8 * (o >> 1)[None, None, None, :]   # (I, 1, lane, o)
    cols = 8 * blk[None, :, None, None] + 2 * LANE_T[None, None, :, None] + (o & 1)[None, None, None, :]     # (1, J, lane, o)
    rows, cols = np.broadcast_arrays(rows, cols)
    x = out.reshape(-1, sh.tm, sh.tn).astype(np.uint64)[:, rows, cols]                                          # (T, I, J, lane, o)
    return (x.sum(axis=(1, 2)) & 0xFFFFFFFF).astype(np.uint32)


# ------------------------------------------------------------------------------------------------------------------ build + SASS

def kernel_name(fn: str) -> str | None:
    m = re.search(r"strassen_kernelILi(\d)E", fn)
    if m:
        return f"dots_l{m.group(1)}"
    return "native" if "native_kernel" in fn else "step" if "step_kernel" in fn else None


def sass_gate(loops: dict, ptxas: dict, nt: int) -> dict:
    """Per kernel: each Strassen loop's 7^L TF32 MMAs and nothing else from the tensor core or memory, the native loop's NT OMMA,
    no spills anywhere."""
    by_name = {kernel_name(fn): (fn, hist) for fn, hist in loops.items() if kernel_name(fn)}
    regs = {kernel_name(fn): v for fn, v in ptxas.items() if kernel_name(fn)}
    gates: dict = {}
    for name in ("dots_l1", "dots_l2", "native", "step"):
        fn, hist = by_name.get(name, (None, {}))
        mma = {k: v for k, v in hist.items() if F1.MMA_RE.search(k)}
        mem = {k: v for k, v in hist.items() if k.split(".")[0] in MEM_OPS}
        pr = regs.get(name, {})
        spill = pr.get("spill_stores", 1) + pr.get("spill_loads", 1)
        if name.startswith("dots"):
            level = int(name[-1])
            ok = mma == {TF32_OPCODE[level]: 7 ** level} and not mem
        elif name == "native":
            ok = mma == {NATIVE_OPCODE: nt}
        else:
            ok = not mma
        gates[name] = {"ok": bool(fn) and ok and spill == 0, "function": fn, "mma_opcodes": mma, "memory_ops": mem,
                       "fadd": hist.get("FADD", 0), "loop_instructions": sum(hist.values()), "ptxas": pr, "loop_opcodes": hist}
    return gates


def build(out: Path) -> dict:
    """The library, its ptxas report, its SASS gate and FTZ scan; writes build.json."""
    if not F1._flags_ok(NVCC_FLAGS):
        raise SystemExit("refused: an -ftz / fast-math flag in NVCC_FLAGS")
    out.mkdir(parents=True, exist_ok=True)
    src = SRC.read_text()
    so = out / f"libf2_strassen_{ARCH}.so"
    nvcc = P.nvcc_bin()
    cmd = [nvcc, *NVCC_FLAGS, *P.arch_flags(ARCH), "-Xptxas", "-v", "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(SRC)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    nt = int(re.search(r"#define NT (\d+)", src).group(1))
    rec: dict = {"schema": SCHEMA + "/build", "arch": ARCH, "nvcc_flags": NVCC_FLAGS, "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
                 "header_sha256": hashlib.sha256(HEADER.read_bytes()).hexdigest(), "nt": nt, "so": so.name}
    if r.returncode:
        rec.update(ok=False, error=(r.stderr + r.stdout)[-4000:])
        (out / "build.json").write_text(json.dumps(rec, indent=1) + "\n")
        return rec
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-2:]
    sass = subprocess.run([str(Path(nvcc).parent / "cuobjdump"), "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    gates = sass_gate(F1.loops_of(sass), F3.ptxas_report(r.stderr + r.stdout), nt)
    ftz = sorted(set(m.group(0) for m in F1.FTZ_RE.finditer(sass)))
    rec.update(nvcc=" / ".join(x.strip() for x in ver), so_sha256=hashlib.sha256(so.read_bytes()).hexdigest(), gate=gates, ftz_fp32=ftz,
               ok=(not ftz) and all(g["ok"] for g in gates.values()))
    (out / "build.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


# ------------------------------------------------------------------------------------------------------------------------ device

_p = F3._p


class F2Device:
    def __init__(self, so: Path):
        self.lib = lib = ctypes.CDLL(str(so))
        I, L, V = ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p
        pI, pF = ctypes.POINTER(I), ctypes.POINTER(ctypes.c_float)
        lib.f2_shape.argtypes = [I, pI, pI, pI, pI]
        lib.f2_device.argtypes = [pI] * 4
        lib.f2_occupancy.argtypes = [I, I, pI]
        lib.f2_run_strassen.argtypes = [I, V, V, I, I, L, V, V, V, pF]
        lib.f2_run_native.argtypes = [V, L, V, L, V, I, I, I, I, I, I, I, L, V, V, pF]
        lib.f2_run_step.argtypes = [V, V, V, V, V, L, I, V]
        lib.device_uuid.restype = I
        lib.device_uuid.argtypes = [ctypes.c_char_p, I]
        lib.has_mxf4.restype = I
        lib.device_identity.restype = I
        lib.device_identity.argtypes = [pI, pI, ctypes.c_char_p, I]
        self.tcgen05 = False

    identity = P.Probe.identity
    clocks = P.Probe.clocks

    def shape(self, level: int) -> tuple[int, int, int, int]:
        v = [ctypes.c_int() for _ in range(4)]
        if self.lib.f2_shape(level, *v):
            raise RuntimeError(f"f2_shape({level})")
        return tuple(x.value for x in v)

    def nt(self) -> int:
        return self.lib.f2_nt()

    def tpb(self) -> int:
        return self.lib.f2_tpb()

    def device(self) -> dict:
        v = [ctypes.c_int() for _ in range(4)]
        rc = self.lib.f2_device(*v)
        if rc:
            raise RuntimeError(f"f2_device rc={rc}")
        return {"sms": v[0].value, "max_threads_per_sm": v[1].value, "cc": f"{v[2].value}.{v[3].value}"}

    def occupancy(self, kind: int, tpb: int) -> int:
        n = ctypes.c_int()
        rc = self.lib.f2_occupancy(kind, tpb, ctypes.byref(n))
        if rc:
            raise RuntimeError(f"f2_occupancy({kind}) rc={rc}")
        return n.value

    def run_strassen(self, level, afr, bfr, ntiles, nblocks, iters):
        sh, ms = Shape(level), ctypes.c_float()
        out = np.zeros(ntiles * sh.tm * sh.tn, np.uint32)
        sums = np.zeros(nblocks * self.tpb() // 32 * 32 * 4, np.uint32)
        cyc = np.zeros(nblocks, np.int64)
        rc = self.lib.f2_run_strassen(level, _p(afr), _p(bfr), ntiles, nblocks, iters, _p(out), _p(sums), _p(cyc), ctypes.byref(ms))
        if rc:
            raise RuntimeError(f"f2_run_strassen({level}) rc={rc}")
        return ms.value, out, sums, cyc

    def run_native(self, arows, brows, c0, sh: Shape, ntiles, nblocks, iters, want=True):
        ms = ctypes.c_float()
        out = np.zeros(c0.size, np.uint32) if want else None
        cyc = np.zeros(nblocks, np.int64)
        rc = self.lib.f2_run_native(_p(arows), arows.shape[0], _p(brows), brows.shape[0], _p(c0), sh.tm, sh.tn, NCT, NTILES, ntiles, nblocks,
                                    NATIVE_TPB, iters, _p(out), _p(cyc), ctypes.byref(ms))
        if rc:
            raise RuntimeError(f"f2_run_native rc={rc}")
        return ms.value, out, cyc

    def run_step(self, gdf, mm, xg, part, c0, length):
        out = np.zeros(c0.size, np.uint32)
        rc = self.lib.f2_run_step(_p(gdf), _p(mm), _p(xg), _p(part), _p(c0), c0.size, length, _p(out))
        if rc:
            raise RuntimeError(f"f2_run_step rc={rc}")
        return out


class FakeF2:
    """CPU stand-in for the dry run: the Strassen kernel is emulated through the packed fragments (unpacked, multiplied per product
    and recombined by W), the native arm is the vectorised pinned model on the packed codes, the step the same model on the given
    group dots.  Its rates are invented (TF32 m16n8k8 at 256 and m16n8k4 at 128 products per SM per clock, NVFP4 at 2,048);
    ``--fake-silicon`` never runs with ``RESEARCH_RUN_ID`` set."""

    SMS, MHZ = 64, 2000.0   # 64 SMs x one 4-warp block: a warp per tile
    TPB = 128

    def __init__(self, nt: int):
        self._nt = nt

    def identity(self) -> dict:
        return {"device": "FAKE SILICON", "compute_capability": "12.0", "driver": None, "gpu_uuid": "GPU-fake", "backend": "fake",
                "cuda_visible_devices": SHARED.CUDA_VISIBLE_DEVICES}

    def clocks(self, uuid):
        return None

    def shape(self, level):
        sh = Shape(level)
        return sh.np_, sh.kb, sh.tm, sh.tn

    def nt(self):
        return self._nt

    def tpb(self):
        return self.TPB

    def device(self) -> dict:
        return {"sms": self.SMS, "max_threads_per_sm": 1536, "cc": "12.0"}

    def occupancy(self, kind, tpb) -> int:
        return 1

    def _ms(self, cycles: float) -> float:
        return cycles / (self.MHZ * 1e3)

    def run_strassen(self, level, afr, bfr, ntiles, nblocks, iters):
        _p(afr), _p(bfr)
        sh, (_, _, w) = Shape(level), coeffs(level)
        a = np.zeros((ntiles, sh.np_, 16, sh.kb))
        b = np.zeros((ntiles, sh.np_, sh.kb, 8))
        fa = afr.view(np.float32).reshape(ntiles, sh.np_, 32, -1).astype(np.float64)
        fb = bfr.view(np.float32).reshape(ntiles, sh.np_, 32, -1).astype(np.float64)
        for i, (r, c) in enumerate(a_positions(sh.kb)):
            a[:, :, r, c] = fa[..., i]
        for i, (r, c) in enumerate(b_positions(sh.kb)):
            b[:, :, r, c] = fb[..., i]
        m = np.einsum("tpik,tpkj->tpij", a, b)
        c = np.einsum("IJp,tpij->tIiJj", w, m).reshape(ntiles, sh.tm, sh.tn)
        out = np.ascontiguousarray(c.astype(np.float32).view(np.uint32)).reshape(-1)
        warps = nblocks * self.TPB // 32
        tile = np.arange(warps) % ntiles
        sums = ((lane_sums(out, sh)[tile].astype(np.uint64) * np.uint64(iters)) & 0xFFFFFFFF).astype(np.uint32).reshape(-1)
        rate = 256 if level == 1 else 128
        cycles = iters * warps * sh.np_ * 16 * 8 * sh.kb / (self.SMS * rate)
        return self._ms(cycles), out, sums, np.full(nblocks, int(cycles), np.int64)

    def run_native(self, arows, brows, c0, sh: Shape, ntiles, nblocks, iters, want=True):
        for x in (arows, brows, c0):
            _p(x)
        warps = nblocks * NATIVE_TPB // 32
        cycles = iters * warps * self._nt * NATIVE_PRODUCTS / (self.SMS * 2048)
        cyc = np.full(nblocks, int(cycles), np.int64)
        if not want:
            return self._ms(cycles), None, cyc
        ac, asc = F3.unpack_rows(arows)
        wc, wsc = F3.unpack_rows(brows)
        ops = F3.Operands(ac, asc, wc[None], wsc[None], np.zeros((arows.shape[0], brows.shape[0]), np.uint32))
        gd, mm, x, part = tile_terms(ops, sh)
        return self._ms(cycles), F3.chain(c0.reshape(gd.shape[:3]), gd * mm, x, part, iters).reshape(-1), cyc

    def run_step(self, gdf, mm, xg, part, c0, length):
        for x in (gdf, mm, xg, part, c0):
            _p(x)
        gd = gdf.view(np.float32).astype(np.int64)
        pt = ((part[:, None] >> np.arange(4, dtype=np.uint32)) & 1).astype(bool)
        return F3.chain(c0, gd * mm, xg.astype(np.int64), pt, length)


# ------------------------------------------------------------------------------------------------------------------------- run

def strassen_grid(d: dict, occ: int, tpb: int) -> dict:
    bps = min(occ, d["max_threads_per_sm"] // tpb)
    return {"threads_per_block": tpb, "blocks_per_sm": bps, "blocks": d["sms"] * bps, "warps": d["sms"] * bps * tpb // 32,
            "warps_per_sm": bps * tpb // 32}


def native_grid(d: dict, occ: int) -> dict:
    bps = min(occ, 4)
    return {"threads_per_block": NATIVE_TPB, "blocks_per_sm": bps, "blocks": d["sms"] * bps, "warps_per_sm": bps * NATIVE_TPB // 32}


@dataclasses.dataclass
class Family:
    ops: F3.Operands
    arows: np.ndarray
    brows: np.ndarray
    c0: np.ndarray         # (T, TM, TN)
    gd: np.ndarray         # (T, TM, TN, 4) and the scale terms beside it
    mm: np.ndarray
    x: np.ndarray
    part: np.ndarray
    frags: list            # per group: (afr, bfr, rounded) of the codes' halves


def family(name: str, sh: Shape, seed: int) -> Family:
    ops = make_operands(name, sh, seed)
    gd, mm, x, part = tile_terms(ops, sh)
    c = np.ascontiguousarray
    return Family(ops, c(F3.pack_rows(ops.acodes, ops.ascale)), c(F3.pack_rows(ops.wcodes[0], ops.wscale[0])), tile_c0(ops, sh), gd, mm, x,
                  part, [fragments(*group_halves(ops, sh, b), sh.level) for b in range(4)])


def dots_check(sh: Shape, out: np.ndarray, sums: np.ndarray, want: np.ndarray, iters: int, warps: int) -> dict:
    """The last iteration's group dots against the integers on every element, and each warp's checksum against iters x its tile's."""
    got = out.view(np.float32).reshape(want.shape)
    tile = np.arange(warps) % NTILES
    want_sums = ((lane_sums(out, sh)[tile].astype(np.uint64) * np.uint64(iters)) & 0xFFFFFFFF).astype(np.uint32)
    return {"dots_mismatches": int(np.count_nonzero(got != want)), "dots": int(want.size),
            "checksum_mismatches": int(np.count_nonzero(sums.reshape(warps, 32, 4) != want_sums))}


def native_words(dev, fam: Family, sh: Shape, nt: int, iters: int) -> np.ndarray:
    ntiles = NTILES * sh.native_per_tile
    warps = -(-ntiles // nt)
    nblocks = -(-warps // (NATIVE_TPB // 32))
    _, out, _ = dev.run_native(fam.arows, fam.brows, fam.c0, sh, ntiles, nblocks, iters, want=True)
    return out


def gate_family(dev, fam: Family, sh: Shape, geo: dict, nt: int, rng, sample: int, corrupt: bool) -> dict:
    """Every group's Strassen dots at chains of 1 and 3, Strassen on the scaled values (recorded), and the route's words against
    the native atom and the pinned model."""
    blocks, warps = geo["dots"]["blocks"], geo["dots"]["warps"]
    rec: dict = {"groups": {}, "words": {}}
    dots = np.zeros(fam.gd.shape, np.uint32)
    for b in range(4):
        afr, bfr, rounded = fam.frags[b]
        g: dict = {"rounded_operands": rounded}
        for length in GATE_LENGTHS:
            _, out, sums, _ = dev.run_strassen(sh.level, afr, bfr, NTILES, blocks, length)
            g[f"L{length}"] = dots_check(sh, out, sums, fam.gd[..., b].astype(np.float32), length, warps)
            dots[..., b] = out.reshape(fam.gd.shape[:3])
        sa, sb = group_scaled(fam.ops, sh, b)
        safr, sbfr, srounded = fragments(sa, sb, sh.level)
        _, sout, _, _ = dev.run_strassen(sh.level, safr, sbfr, NTILES, blocks, 1)
        exact = (fam.gd[..., b] * fam.mm[..., b] * np.exp2(fam.x[..., b].astype(np.float64))).astype(np.float32)
        g["scaled"] = {"rounded_operands": srounded, "operands": int(safr.size + sbfr.size),
                       "inexact_group_sums": int(np.count_nonzero(sout.view(np.float32).reshape(exact.shape) != exact)),
                       "group_sums": int(exact.size)}
        g["ok"] = rounded == 0 and all(g[f"L{n}"]["dots_mismatches"] == 0 and g[f"L{n}"]["checksum_mismatches"] == 0 for n in GATE_LENGTHS)
        rec["groups"][str(b)] = g
    gdf = np.ascontiguousarray(dots.reshape(-1, 4))
    mm = np.ascontiguousarray(fam.mm.reshape(-1, 4), np.int32)
    xg = np.ascontiguousarray(fam.x.reshape(-1, 4), np.int32)
    part = np.ascontiguousarray((fam.part.reshape(-1, 4) << np.arange(4)).sum(-1), np.uint32)
    c0 = fam.c0.reshape(-1)
    rows, cols = np.broadcast_arrays(tile_rows(sh)[:, :, None], tile_cols(sh)[:, None, :])
    for length in GATE_LENGTHS:
        route = dev.run_step(gdf, mm, xg, part, c0, length).copy()
        if corrupt:
            route[0] ^= 1
        native = native_words(dev, fam, sh, nt, length)
        e = rng.integers(0, route.size, sample)
        want = F3.model_sample(fam.ops, rows.reshape(-1)[e], cols.reshape(-1)[e], length)
        w = {"route_native_mismatches": int(np.count_nonzero(route != native)), "words": int(route.size), "model_sample": sample,
             "native_model_mismatches": int(np.count_nonzero(native[e] != want)),
             "route_model_mismatches": int(np.count_nonzero(route[e] != want))}
        w["ok"] = w["route_native_mismatches"] == 0 and w["native_model_mismatches"] == 0 and w["route_model_mismatches"] == 0
        rec["words"][f"L{length}"] = w
    rec["ok"] = all(g["ok"] for g in rec["groups"].values()) and all(w["ok"] for w in rec["words"].values())
    return rec


def run_level(args) -> int:
    t_all = time.time()
    bdir, out = Path(os.path.expandvars(args.build)), Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    b = json.loads((bdir / "build.json").read_text())
    level = LEVELS[args.level]
    sh = Shape(level)
    rec: dict = {"schema": SCHEMA, "level": args.level, "levels": level, "tile": [sh.tm, sh.tn, 16], "mmas": sh.np_,
                 "mmas_naive": 8 ** level, "mma_shape": f"m16n8k{sh.kb}", "favours_strassen": FAVOURS_STRASSEN,
                 "build": {k: b.get(k) for k in ("so_sha256", "source_sha256", "header_sha256", "nvcc", "nvcc_flags", "ok", "ftz_fp32")},
                 "seed": args.seed, "sample": args.sample, "fake_silicon": args.fake_silicon, "corrupt_word": args.corrupt_word}
    failures: list[str] = []
    if not b.get("ok"):
        failures.append("build gate failed: " + ", ".join([n for n, v in (b.get("gate") or {}).items() if not v["ok"]]
                                                          + (b.get("ftz_fp32") or []) + ([b["error"][:200]] if b.get("error") else [])))
        return finish(out, rec, failures, args, None, t_all)
    dev = FakeF2(b["nt"]) if args.fake_silicon else F2Device(bdir / b["so"])
    identity = dev.identity()
    rec["identity"] = identity
    if args.expect_uuid and not args.fake_silicon and not P.uuid_allowed(identity.get("gpu_uuid"), args.expect_uuid):
        failures.append(f"device UUID {identity.get('gpu_uuid')} != expected {args.expect_uuid}: stopped before any measurement")
    nt = dev.nt()
    if nt != b["nt"] or dev.shape(level) != (sh.np_, sh.kb, sh.tm, sh.tn):
        failures.append(f"the library's NT {nt} / shape {dev.shape(level)} are not build.json's / {(sh.np_, sh.kb, sh.tm, sh.tn)}")
    if failures:
        return finish(out, rec, failures, args, None, t_all)
    d = dev.device()
    rec["device"] = d
    if d["cc"] != "12.0":
        return finish(out, rec, [f"compute capability {d['cc']} != 12.0"], args, None, t_all)
    geo = {"dots": strassen_grid(d, dev.occupancy(level, dev.tpb()), dev.tpb()), "native": native_grid(d, dev.occupancy(0, NATIVE_TPB))}
    rec["grid"] = geo
    if min(v["blocks_per_sm"] for v in geo.values()) < 1 or geo["dots"]["warps"] < NTILES:
        return finish(out, rec, [f"a kernel does not fit, or fewer warps than tiles: {geo}"], args, None, t_all)
    rng = np.random.default_rng([args.seed, level, 2])
    fams = {f: family(f, sh, args.seed) for f in FAMILIES}
    gate = {f: gate_family(dev, fams[f], sh, geo, nt, rng, args.sample, args.corrupt_word and f == FAMILIES[0]) for f in FAMILIES}
    rec["gate"] = gate
    failures += [f"gate {f}: {json.dumps(v)[:600]}" for f, v in gate.items() if not v["ok"]]
    if failures:
        return finish(out, rec, failures, args, None, t_all)

    ntn = NTILES * sh.native_per_tile

    def launch(f, arm, iters):
        fam = fams[f]
        if arm == "native":
            return dev.run_native(fam.arows, fam.brows, fam.c0, sh, ntn, geo["native"]["blocks"], iters, want=False)
        afr, bfr, _ = fam.frags[0]
        return dev.run_strassen(level, afr, bfr, NTILES, geo["dots"]["blocks"], iters)

    keys = [(f, arm) for f in TIMED for arm in ("native", "dots")]
    iters = {}
    for k in keys:
        ms = launch(*k, 1000)[0]
        iters[k] = max(1000, int(1000 * args.target_ms / max(ms, 1e-3)))
    t0 = time.time()
    while time.time() - t0 < args.sustain_s:
        launch(TIMED[0], "native", iters[(TIMED[0], "native")])
    launches: dict = {k: [] for k in keys}
    samples = []
    post: dict = {}
    for rep in range(args.reps):
        for k in keys[rep % len(keys):] + keys[:rep % len(keys)]:
            res = launch(*k, iters[k])
            ms, cyc = res[0], res[-1]
            launches[k].append({"rep": rep, "ms": ms, "cycles_max": int(cyc.max()), "sm_mhz": float(cyc.max()) / (ms * 1e3)})
            if k[1] == "dots" and k[0] not in post:
                post[k[0]] = {"length": iters[k], **dots_check(sh, res[1], res[2], fams[k[0]].gd[..., 0].astype(np.float32), iters[k],
                                                               geo["dots"]["warps"])}
        samples.append(dev.clocks(identity.get("gpu_uuid")))
    for v in post.values():
        v["ok"] = v["dots_mismatches"] == 0 and v["checksum_mismatches"] == 0
    rec["timed_length_gate"] = post
    failures += [f"timed-length gate {f}: {v}" for f, v in post.items() if not v["ok"]]
    if failures:
        return finish(out, rec, failures, args, None, t_all)
    per: dict = {}
    for (f, arm), ls in launches.items():
        ms = statistics.median(x["ms"] for x in ls)
        mhz = statistics.median(x["sm_mhz"] for x in ls)
        denom = d["sms"] * mhz * 1e6 * ms / 1e3
        if arm == "native":
            prod = float(iters[(f, arm)]) * geo["native"]["blocks"] * NATIVE_TPB / 32 * nt * NATIVE_PRODUCTS
            mma = prod
        else:
            prod = float(iters[(f, arm)]) * geo["dots"]["warps"] * sh.tm * sh.tn * 16
            mma = float(iters[(f, arm)]) * geo["dots"]["warps"] * sh.np_ * 16 * 8 * sh.kb
        per[f"{f}/{arm}"] = {"family": f, "arm": arm, "iterations": iters[(f, arm)], "ms_median": ms, "ms_all": [x["ms"] for x in ls],
                             "sm_mhz_median": mhz, "sm_mhz_all": [x["sm_mhz"] for x in ls], "warps_per_sm": geo[arm]["warps_per_sm"],
                             "products_per_sm_per_clock": prod / denom, "mma_products_per_sm_per_clock": mma / denom}
    for v in per.values():
        v["fp4_slots_per_product"] = per[f"{v['family']}/native"]["products_per_sm_per_clock"] / v["products_per_sm_per_clock"]
    rec.update(per_arm=per, launches={f"{f}/{a}": v for (f, a), v in launches.items()}, clock_samples=samples,
               clocks=F1.clock_label(samples, [x["sm_mhz"] for ls in launches.values() for x in ls], args.locked_mhz),
               iterations={f"{f}/{a}": v for (f, a), v in iters.items()},
               mma_floor_fp4_slots={"value": F1_TF32_SLOTS[level] * (7 / 8) ** level,
                                    "label": "Derived: (7/8)^L of F1's Measured TF32 price, the MMAs alone"})
    return finish(out, rec, failures, args, per, t_all)


def finish(out: Path, rec: dict, failures: list[str], args, per: dict | None, t_all: float) -> int:
    rec["failures"] = failures
    rec["elapsed_s"] = time.time() - t_all
    name = f"{args.level}.json" if not args.fake_silicon else f"{args.level}.fake.json"
    tmp = out / (name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    tmp.replace(out / name)
    if failures:
        print("FAILURES", failures, flush=True)
        return 2 if any("UUID" in f for f in failures) else 1
    print(f"F2 {args.level} RECORDED ({rec.get('clocks')})", flush=True)
    for k, v in (per or {}).items():
        print(f"  {k:16s} {v['products_per_sm_per_clock']:9.1f} products/SM/clk  {v['fp4_slots_per_product']:8.3f} FP4 slots/product  "
              f"{v['sm_mhz_median']:7.1f} MHz  {v['warps_per_sm']} warps/SM", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------------------- collect

def collect(args) -> int:
    """The price table over both levels (each against the NVFP4 of its own run) and the claim's check."""
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    bj = Path(os.path.expandvars(args.build)) / "build.json"
    b = json.loads(bj.read_text())
    kept = out / "inputs"
    kept.mkdir(exist_ok=True)
    (kept / "build.json").write_bytes(bj.read_bytes())
    rows, levels, failures = {}, {}, []
    for d in [x for x in args.runs.split(",") if x]:
        for p in sorted(Path(os.path.expandvars(d)).glob("*.json")):
            r = json.loads(p.read_text())
            if r.get("schema") != SCHEMA:
                continue
            lv = r["level"]
            if lv in levels:
                failures.append(f"{lv}: recorded twice ({levels[lv]['file']}, {p})")
                continue
            (kept / p.name).write_bytes(p.read_bytes())
            levels[lv] = {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "gpu_uuid": (r.get("identity") or {}).get("gpu_uuid"),
                          "clocks": r.get("clocks"), "failures": r.get("failures"), "build_so_sha256": r["build"].get("so_sha256"),
                          "fake_silicon": r.get("fake_silicon"), "grid": r.get("grid"), "mma_floor_fp4_slots": r.get("mma_floor_fp4_slots"),
                          "scaled": {f: {g: v["scaled"] for g, v in fam["groups"].items()} for f, fam in (r.get("gate") or {}).items()}}
            if r.get("fake_silicon"):
                failures.append(f"{lv}: fake silicon (a dry run's invented numbers)")
            if r.get("failures"):
                failures.append(f"{lv}: {r['failures']}")
                continue
            if r["build"].get("so_sha256") != b.get("so_sha256"):
                failures.append(f"{lv}: ran a different build ({str(r['build'].get('so_sha256'))[:12]})")
            for k, v in r["per_arm"].items():
                rows[f"{lv}/{k}"] = {**v, "level": lv, "gpu_uuid": levels[lv]["gpu_uuid"]}
    missing = [lv for lv in LEVELS if lv not in levels]
    dots = {k: v for k, v in rows.items() if v["arm"] == "dots"}
    cheaper = {k: v["fp4_slots_per_product"] for k, v in dots.items() if v["fp4_slots_per_product"] < 1 - args.tolerance}
    doc = {"schema": SCHEMA + "/table", "unit": "one dense NVFP4 product (mma.sync kind::mxf4nvf4 4X ue4m3 m16n8k64) per SM per clock, "
                                              "measured in each level's own run",
           "build_so_sha256": b.get("so_sha256"), "levels": levels, "missing_levels": missing, "rows": rows,
           "sorted_fp4_slots_per_product": sorted(((k, v["fp4_slots_per_product"]) for k, v in dots.items()), key=lambda x: x[1]),
           "claim": {"statement": "fp4-tile-only/sm120: no Strassen level over one NVFP4 group on the TF32 tensor core computes the atom's "
                                  "group dots (exact, so the atom's words) for fewer FP4 slots per product than dense NVFP4, with the "
                                  "route's advantages of FAVOURS_STRASSEN",
                     "tolerance": args.tolerance, "cheaper": cheaper, "holds_over_measured": not cheaper and not missing},
           "favours_strassen": FAVOURS_STRASSEN, "label": "Measured"}
    (out / "f2_strassen.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    ok = not failures and not missing
    meas = [{"name": f"{k}_fp4_slots_per_product", "value": v["fp4_slots_per_product"], "unit": "NVFP4 products"} for k, v in rows.items()]
    meas += [{"name": f"{k}_products_per_sm_per_clock", "value": v["products_per_sm_per_clock"], "unit": "products/(SM*cycle)"}
             for k, v in rows.items()]
    clocks = sorted({str(v["clocks"]) for v in levels.values()})
    res = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": SHARED.RESEARCH_RUN_ID,
           "clocks": clocks[0] if len(clocks) == 1 else "mixed", "label": "Measured" if clocks == ["locked-2100"] else "Measured, " + ",".join(clocks),
           "method": "F2 Strassen kernels from the fill queue (one level per chunk, each against dense NVFP4 in the same process), "
                     "collected; see f2_strassen.json",
           "workload_fingerprint": {"tool": "tc_strassen_fp4", "version": "1", "build_so_sha256": b.get("so_sha256"),
                                    "levels": {k: v["sha256"] for k, v in levels.items()}, "commit": SHARED.RESEARCH_SOURCE_COMMIT},
           "validation": {"status": "passed" if ok else "failed", "checker": "every level's gates (UUID, cc 12.0, SASS, every group's Strassen "
                          "dots = the integer group dots on every element, the route's words = the native atom's on every word and the "
                          "pinned model's on a sample at chains of 1 and 3, the dots and their checksum at the timed length) passed in its "
                          "chunk, on this build",
                          "detail": "; ".join(failures) or f"{len(levels)} levels, {len(rows)} rows", "evidence": ["f2_strassen.json"]},
           "claim": doc["claim"], "measurements": meas, "measurement_files": ["f2_strassen.json"], "artifacts": []}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("F2 TABLE", "ok" if ok else "FAILED", failures, missing, flush=True)
    for k, p in doc["sorted_fp4_slots_per_product"]:
        print(f"  {k:20s} {p:9.3f} FP4 slots per product", flush=True)
    return 0 if ok else 1


def build_cmd(args) -> int:
    rec = build(Path(os.path.expandvars(args.out)))
    print(f"F2 BUILD {'ok' if rec['ok'] else 'FAILED'}: FTZ {rec.get('ftz_fp32') or 'none'}", flush=True)
    for n, g in (rec.get("gate") or {}).items():
        print(f"  {n:8s} {'ok ' if g['ok'] else 'BAD'} {g['ptxas'].get('registers')} regs  loop {g['loop_instructions']} instr  "
              f"{g['mma_opcodes']}  FADD {g['fadd']}  memory {g['memory_ops'] or 'none'}", flush=True)
    if rec.get("error"):
        print(rec["error"], flush=True)
    if SHARED.RESEARCH_RUN_ID:
        res = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": SHARED.RESEARCH_RUN_ID,
               "label": "Measured (ptxas and SASS of the F2 kernels; no device)",
               "method": "f2_strassen.cu compiled for sm_120a with -Xptxas -v; each kernel's timed loop read from cuobjdump",
               "workload_fingerprint": {"tool": "tc_strassen_fp4", "version": "1", "step": "build", "so_sha256": rec.get("so_sha256"),
                                        "source_sha256": rec["source_sha256"], "header_sha256": rec["header_sha256"],
                                        "commit": SHARED.RESEARCH_SOURCE_COMMIT},
               "validation": {"status": "passed" if rec["ok"] else "failed",
                              "checker": "7^L TF32 MMAs and no memory access per Strassen loop, NT OMMA in the native loop, no spills, "
                                         "no FADD/FFMA/FMUL .FTZ",
                              "detail": ", ".join(n for n, g in (rec.get("gate") or {}).items() if not g["ok"]) or "every kernel",
                              "evidence": ["build.json"]},
               "measurements": [{"name": f"{n}_loop_instructions", "value": g["loop_instructions"], "unit": "instructions"}
                                for n, g in (rec.get("gate") or {}).items()],
               "measurement_files": ["build.json"], "artifacts": []}
        (Path(os.path.expandvars(args.out)) / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0 if rec["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--build", required=True)
    r.add_argument("--level", required=True, choices=sorted(LEVELS))
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--target-ms", type=float, default=100.0)
    r.add_argument("--sustain-s", type=float, default=2.0)
    r.add_argument("--seed", type=int, default=20260930)
    r.add_argument("--sample", type=int, default=2048, help="words per family and chain length checked against the pinned model")
    r.add_argument("--expect-uuid", default=None)
    r.add_argument("--locked-mhz", type=float, default=None)
    r.add_argument("--corrupt-word", action="store_true")
    r.add_argument("--fake-silicon", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--build", required=True)
    c.add_argument("--runs", required=True, help="comma list of directories holding <level>.json")
    c.add_argument("--out", required=True)
    c.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args(argv)
    if getattr(args, "fake_silicon", False) and SHARED.RESEARCH_RUN_ID:
        ap.error("--fake-silicon is a dry run and never publishes: unset RESEARCH_RUN_ID")
    return {"build": build_cmd, "run": run_level, "collect": collect}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
