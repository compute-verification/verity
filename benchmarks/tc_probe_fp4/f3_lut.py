"""F3: LUT-GEMM over the 16-code E2M1 alphabet, tables from the weights, against dense NVFP4 on sm_120a.

A cheaper-computation search: with the weights fixed, the partial dots of G consecutive weights of C columns against every one
of the 16^G activation-code tuples can be tabulated in advance, so one table lookup replaces G x C E2M1 products.  This tool
prices that route against the tensor core's dense NVFP4 tile (``mma.sync kind::mxf4nvf4 4X ue4m3 m16n8k64``, F1's unit) on the
words the atom produces (``verity.ml.circuits.silicon.models.BLACKWELL_SM120_NVF4``), bit for bit:

    python benchmarks/tc_probe_fp4/f3_lut.py build --out DIR                          # the library and its SASS gate
    python benchmarks/tc_probe_fp4/f3_lut.py run --build DIR --width g1c2 --out DIR2 --expect-uuid GPU-... --locked-mhz 2100
    python benchmarks/tc_probe_fp4/f3_lut.py collect --build DIR --runs D1,D2 --out DIR3   # the price table over every width

Kernels (f3_lut.cu): per thread one activation row and one column group of C columns, chained over ``iters`` repeats of one
K = 64 atom.  ``dots`` is the lookups and their unpacking into the four blocks' group dots (the table's own price); ``exact``
adds the atom's align-add on the CUDA cores, so its words are the atom's; ``native`` is the OMMA tile on the same rows and
columns.  Widths (G, C) are ``WIDTHS``: G <= 2 tables in shared memory, G = 4 in global memory (L2).  Operand families:
``mixed`` (gate only: zero blocks, zero and subnormal scales, every kind of finite accumulator), ``uniform`` (uniform codes and
UE4M3 scales) and ``gaussian`` (``gaussian_operand``, after Pearl-C4's quantizer), both from +0.

Gates, before any timing (a failed gate writes no measurements): the device UUID and compute capability 12.0; the build's SASS
gate (each LUT loop issues its lookups and no MMA, the native loop NT OMMA, no spills, no FP32 ``.FTZ``); for chains of 1 and
3 atoms on every family, the native words equal the pinned model on a sample, the exact words equal the native ones on every
word (and the model on the sample), and the dots equal the integer group dots times the chain length; after timing, the same
at each arm's timed length (exact against native, dots against the integers mod 2^32).  ``--corrupt-word`` flips one exact
word: the run must then fail.

What the timed loop leaves out, all in the table's favour (``FAVOURS_THE_TABLE``, in every record): the tables are one atom's,
resident in shared memory (G <= 2) or L2 (G = 4) and re-read every iteration, where a GEMM's tables (K/64 x N/C of them per
weight matrix) would stream; their construction (preprocessing of fixed weights); and the activation side's addresses, scale
decode and masks (amortized over the N / C column groups).  A measured price is therefore a lower bound on the route's.
Labels: rates and prices Measured; the comparison with the FP4-tile model's lookup price (16.2-66 FP4 slots) Derived.
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
import probe as P  # noqa: E402
import vec as V  # noqa: E402
from verity_config.settings import SHARED  # noqa: E402

from verity.ml.circuits.silicon import models as M  # noqa: E402

ARCH = "sm_120a"
SCHEMA = "pouw-rtxpro/f3-lut-sm120/v1"
SRC = HERE / "f3_lut.cu"
HEADER = HERE / "nvf4_step.cuh"
NVCC_FLAGS = ["-O2", "-std=c++17"]  # no -ftz / fast-math; build refuses them
LUT_TPB, NATIVE_TPB = 256, 128
AROW = 9
WIDTHS = {"g1c2": (1, 2), "g1c4": (1, 4), "g1c8": (1, 8), "g2c2": (2, 2), "g2c4": (2, 4), "g4c2": (4, 2)}
ARMS = ("dots", "exact")
FAMILIES = ("mixed", "uniform", "gaussian")
TIMED = ("uniform", "gaussian")
GATE_LENGTHS = (1, 3)
NATIVE_OPCODE = "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"
NATIVE_PRODUCTS = 16 * 8 * 64
#: E2M1 codes as signed integer halves (bit 3 the sign)
E2M1_HALVES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int64)
#: the FP4-tile model's price of one lookup (docs/pouw/fp4-tile-model.md, a2/fp4-tile)
MODEL_LOOKUP_SLOTS = (16.2, 66.0)
FAVOURS_THE_TABLE = ("one atom's tables, resident (shared memory for G <= 2, L2 for G = 4) and re-read every iteration: a GEMM's "
                     "K/64 x N/C tables per weight matrix would stream",
                     "table construction (preprocessing of fixed weights) outside the timed loop",
                     "the activation side's table addresses, scale decode and nonzero masks outside the timed loop (amortized over "
                     "the N/C column groups)")
LUT_RE = re.compile(r"lut_kernelILi(\d+)ELi(\d+)ELi(\d+)E")
ARM_OF = {"A_DOTS": "dots", "A_EXACT": "exact", 0: "dots", 1: "exact"}


def ng_of(g: int) -> int:
    """Column groups (distinct tables) per launch: 16 for shared-memory tables, 1 for G = 4 (4 MiB per column pair)."""
    return 16 if g <= 2 else 1


def variants(src: str | None = None) -> list[tuple[str, int, int]]:
    """The library's variant table (arm, G, C), in its index order, read from f3_lut.cu."""
    text = src if src is not None else SRC.read_text()
    block = text[text.index("const Variant VARIANTS[]"):]
    block = block[:block.index("};")]
    return [(ARM_OF[a], int(g), int(c)) for a, g, c in re.findall(r"V\((A_DOTS|A_EXACT), (\d+), (\d+)\)", block)]


def nt_of(src: str | None = None) -> int:
    return int(re.search(r"#define NT (\d+)", src if src is not None else SRC.read_text()).group(1))


# ---------------------------------------------------------------------------------------------------------------------- operands

@dataclasses.dataclass
class Operands:
    acodes: np.ndarray   # (R, 64) E2M1 codes
    ascale: np.ndarray   # (R, 4) UE4M3
    wcodes: np.ndarray   # (ng, C, 64)
    wscale: np.ndarray   # (ng, C, 4)
    c0: np.ndarray       # (R, C) binary32 words

    @property
    def groups(self) -> np.ndarray:
        """Each row's column group: the LUT kernel's blockIdx.x % ng."""
        return (np.arange(self.acodes.shape[0]) // LUT_TPB) % self.wcodes.shape[0]


def mixed_acc(rng, shape) -> np.ndarray:
    """+0, -0, subnormals, every finite exponent, and exponents near the products' binades, in equal parts."""
    kind = rng.integers(0, 5, shape)
    frac = rng.integers(0, 1 << 23, shape).astype(np.uint64)
    sign = rng.integers(0, 2, shape).astype(np.uint64) << np.uint64(31)
    ef = rng.integers(1, 255, shape).astype(np.uint64)
    near = np.clip(127 + rng.integers(-40, 60, shape), 1, 254).astype(np.uint64)
    words = np.select([kind == 0, kind == 1, kind == 2, kind == 3],
                      [np.uint64(0), np.uint64(1 << 31), sign | np.maximum(frac, 1), sign | (ef << np.uint64(23)) | frac],
                      sign | (near << np.uint64(23)) | frac)
    return words.astype(np.uint32)


#: RN-even thresholds between consecutive E2M1 magnitudes: (midpoint, a tie rounds up)
_E2M1_MIDS = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False))


def _e2m1_rn(v: np.ndarray) -> np.ndarray:
    mag = np.abs(v)
    code = np.zeros(v.shape, np.int64)
    for mid, up in _E2M1_MIDS:
        code += (mag >= mid) if up else (mag > mid)
    return np.where((v < 0) & (code > 0), code | 8, code)


def _ue4m3_rn(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    out = np.zeros(v.shape, np.int64)
    sub = v < 2.0 ** -6
    m_sub = np.rint(v / 2.0 ** -9).astype(np.int64)
    out = np.where(sub, np.where(m_sub >= 8, 1 << 3, m_sub), out)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.floor(np.log2(np.where(sub, 1.0, v))).astype(np.int64)
        m = np.rint((np.where(sub, 1.0, v) / np.exp2(e) - 1.0) * 8).astype(np.int64)
    e = np.where(m == 8, e + 1, e)
    m = np.where(m == 8, 0, m)
    ef = e + 7
    sat = (ef > 15) | ((ef == 15) & (m > 6))
    ef, m = np.where(sat, 15, ef), np.where(sat, 6, m)
    return np.where(sub, out, (ef << 3) | m)


def gaussian_operand(x: np.ndarray, delta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The ``gaussian`` family's input generator, not a model: an NVFP4 operand drawn the way Pearl-C4's clean path forms one
    (alpha0 from the row max and RMS, row-relative noise, noisy UE4M3 block scales, RN E2M1 codes), in float64 with its own
    draws.  Its draws and roundings fix the family's operands, so it stays as written.  Returns E2M1 codes (n, k) and UE4M3
    scales (n, k / 16)."""

    s = np.abs(x).max(axis=1, keepdims=True)
    rho = np.sqrt((x[:, ::8] ** 2).mean(axis=1, keepdims=True))
    sigma = delta * rho
    alpha0 = 2688.0 / np.maximum(s + 4 * sigma, 2.0 ** -32)
    v = alpha0 * (x + sigma * rng.standard_normal(x.shape))
    n, k = v.shape
    amax = np.abs(v).reshape(n, k // 16, 16).max(axis=2)
    sc = np.maximum(_ue4m3_rn(amax / 6.0), 1)                          # floor at 2^-9 (code 1)
    ma, ea = V.ue4m3_dyadic(sc)
    scale_val = ma * np.exp2(ea)
    codes = _e2m1_rn(v / np.repeat(scale_val, 16, axis=1))
    return codes, sc


def make_operands(family: str, rows: int, ng: int, c: int, seed: int) -> Operands:
    rng = np.random.default_rng([seed, FAMILIES.index(family), rows, ng, c])
    if family == "gaussian":
        ac, asc = gaussian_operand(rng.standard_normal((rows, 64)), 0.05, rng)
        wc, wsc = gaussian_operand(rng.standard_normal((ng * c, 64)), 0.05, rng)
        ops = Operands(ac.astype(np.uint8), asc.astype(np.uint8), wc.astype(np.uint8).reshape(ng, c, 64),
                       wsc.astype(np.uint8).reshape(ng, c, 4), np.zeros((rows, c), np.uint32))
        return ops
    ac = rng.integers(0, 16, (rows, 64)).astype(np.uint8)
    wc = rng.integers(0, 16, (ng * c, 64)).astype(np.uint8)
    asc = P.UE4M3_VALID[rng.integers(0, len(P.UE4M3_VALID), (rows, 4))]
    wsc = P.UE4M3_VALID[rng.integers(0, len(P.UE4M3_VALID), (ng * c, 4))]
    c0 = np.zeros((rows, c), np.uint32)
    if family == "mixed":
        for codes, scales in ((ac, asc), (wc, wsc)):
            n = scales.shape[0]
            zb = rng.random((n, 4)) < 0.125
            blocks = codes.reshape(n, 4, 16)
            blocks[zb] = (rng.integers(0, 2, (int(zb.sum()), 16)) * 8).astype(np.uint8)   # +0 and -0 codes
            u = rng.random((n, 4))
            scales[u < 0.125] = 0
            sub = (u >= 0.125) & (u < 0.25)
            scales[sub] = rng.integers(1, 8, int(sub.sum())).astype(np.uint8)
        c0 = mixed_acc(rng, (rows, c))
    return Operands(ac, asc, wc.reshape(ng, c, 64), wsc.reshape(ng, c, 4), c0)


def pack_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """(..., 64) codes and (..., 4) scale bytes as (..., 9) u32: nibble j of word w is k = 8w + j, word 8 the scales."""
    c = codes.astype(np.uint32).reshape(*codes.shape[:-1], 8, 8)
    words = np.bitwise_or.reduce(c << (4 * np.arange(8, dtype=np.uint32)), axis=-1)
    s = np.bitwise_or.reduce(scales.astype(np.uint32) << (8 * np.arange(4, dtype=np.uint32)), axis=-1)
    return np.concatenate([words, s[..., None]], axis=-1).astype(np.uint32)


def unpack_rows(words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = words.astype(np.uint32)
    codes = ((w[..., :8, None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(*w.shape[:-1], 64)
    scales = (w[..., 8, None] >> (8 * np.arange(4, dtype=np.uint32))) & 0xFF
    return codes.astype(np.uint8), scales.astype(np.uint8)


def lut_tables(wcodes: np.ndarray, g: int) -> np.ndarray:
    """(ng, 64/G, 16^G, C/2) u32: entry idx of slice kg holds, for column pair p, the exact partial dots (quarters) of the G
    activation codes idx names (nibble i: k = G kg + i) against columns 2p and 2p + 1, as lo + 65536 hi mod 2^32."""
    ng, c, _ = wcodes.shape
    nkg, nidx = 64 // g, 16 ** g
    digits = (np.arange(nidx)[:, None] >> (4 * np.arange(g))) & 15
    part = np.einsum("xg,nckg->nkxc", E2M1_HALVES[digits], E2M1_HALVES[wcodes].reshape(ng, c, nkg, g))
    return ((part[..., 0::2] + 65536 * part[..., 1::2]) & 0xFFFFFFFF).astype(np.uint32)


def weight_scale_words(wcodes: np.ndarray, wscale: np.ndarray) -> np.ndarray:
    """(ng, C, 4) u32 per column and block: mb | (eb + 10) << 4 | nonzero-code mask << 8 (UE4M3: e = 0 is (m, -9))."""
    ng, c, _ = wcodes.shape
    s = wscale.astype(np.int64)
    e, m = (s >> 3) & 15, s & 7
    mb, field = np.where(e == 0, m, 8 + m), np.where(e == 0, 1, e)
    nz = ((wcodes.reshape(ng, c, 4, 16).astype(np.int64) & 7) != 0).astype(np.int64)
    mask = (nz << np.arange(16)).sum(-1)
    return (mb | field << 4 | mask << 8).astype(np.uint32)


@dataclasses.dataclass
class Packed:
    arows: np.ndarray    # (R, 9)
    wrows: np.ndarray    # (ng, C, 9)
    tables: np.ndarray   # (ng, 64/G, 16^G, C/2)
    wsc: np.ndarray      # (ng, C, 4)
    c0: np.ndarray       # (R, C)


def pack(ops: Operands, g: int) -> Packed:
    c = np.ascontiguousarray
    return Packed(c(pack_rows(ops.acodes, ops.ascale)), c(pack_rows(ops.wcodes, ops.wscale)), c(lut_tables(ops.wcodes, g)),
                  c(weight_scale_words(ops.wcodes, ops.wscale)), c(ops.c0, np.uint32))


def terms(acodes, ascale, wcodes, wscale, grp) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per (row, column, block), each (R, C, 4): the group dot (quarters), its scaled mantissa, X = -2 + ea + eb, participation
    (a nonzero product and both scale mantissas nonzero), as the pinned model's ``group_terms``."""
    r, c = acodes.shape[0], wcodes.shape[1]
    ma, ea = V.ue4m3_dyadic(ascale.astype(np.int64))
    mb, eb = V.ue4m3_dyadic(wscale.astype(np.int64))
    a = E2M1_HALVES[acodes].reshape(r, 4, 16)
    gd = np.zeros((r, c, 4), np.int64)
    anyp = np.zeros((r, c, 4), np.bool_)
    for gi in np.unique(grp):
        s = grp == gi
        w = E2M1_HALVES[wcodes[gi]].reshape(c, 4, 16)
        gd[s] = np.einsum("rbj,cbj->rcb", a[s], w)
        anyp[s] = np.einsum("rbj,cbj->rcb", (a[s] != 0).astype(np.int64), (w != 0).astype(np.int64)) > 0
    mm = ma[:, None, :] * mb[grp]
    return gd, gd * mm, ea[:, None, :] + eb[grp] - 2, anyp & (mm != 0)


def chain(c0: np.ndarray, mant: np.ndarray, x: np.ndarray, part: np.ndarray, length: int) -> np.ndarray:
    """The atom chained ``length`` times on fixed operands (``vec.nvf4_step``, the pinned model vectorised)."""
    c = c0.copy()
    ms, xs, ps = [mant[..., b] for b in range(4)], [x[..., b] for b in range(4)], [part[..., b] for b in range(4)]
    for _ in range(length):
        c = V.nvf4_step(c, ms, xs, ps)
    return c


def model_sample(ops: Operands, rs: np.ndarray, cs: np.ndarray, length: int) -> np.ndarray:
    """``BLACKWELL_SM120_NVF4.step_scaled`` chained ``length`` times on each sampled (row, column)."""
    m, g = M.BLACKWELL_SM120_NVF4, ops.groups
    out = np.zeros(len(rs), np.uint32)
    for i, (r, c) in enumerate(zip(rs.tolist(), cs.tolist())):
        w = int(ops.c0[r, c])
        a, sa = ops.acodes[r].tolist(), ops.ascale[r].tolist()
        b, sb = ops.wcodes[g[r], c].tolist(), ops.wscale[g[r], c].tolist()
        for _ in range(length):
            w = m.step_scaled(w, a, b, sa, sb)
        out[i] = w
    return out


# ------------------------------------------------------------------------------------------------------------------ build + SASS

def ptxas_report(text: str) -> dict:
    """{mangled kernel: {"registers", "spill_stores", "spill_loads"}} from ``-Xptxas -v``."""
    out: dict = {}
    fn = None
    for line in text.splitlines():
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            fn = m.group(1)
            out[fn] = {}
            continue
        m = re.search(r"Function properties for (\S+)", line)
        if m:
            fn = m.group(1)
            out.setdefault(fn, {})
            continue
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and fn:
            out[fn].update(spill_stores=int(m.group(1)), spill_loads=int(m.group(2)))
        m = re.search(r"Used (\d+) registers", line)
        if m and fn:
            out[fn]["registers"] = int(m.group(1))
    return out


def _load_words(op: str) -> int:
    m = re.search(r"\.(64|128)\b", op)
    return {"64": 2, "128": 4}[m.group(1)] if m else 1


def kernel_name(fn: str) -> str | None:
    m = LUT_RE.search(fn)
    if m:
        return f"{ARM_OF[int(m.group(1))]}_g{m.group(2)}c{m.group(3)}"
    return "native" if "native_kernel" in fn else None


def sass_gate(loops: dict, ptxas: dict, table: list[tuple[str, int, int]], nt: int) -> dict:
    """Per kernel: its timed loop's lookups (a LUT loop: one shared-memory load per slice for G <= 2, 64/G x C/2 words of global
    loads for G = 4, the other space untouched, no MMA), the native loop's NT OMMA, and no spills."""
    by_name = {kernel_name(fn): (fn, hist) for fn, hist in loops.items() if kernel_name(fn)}
    regs = {kernel_name(fn): v for fn, v in ptxas.items() if kernel_name(fn)}
    gates: dict = {}
    for arm, g, c in table:
        name = f"{arm}_g{g}c{c}"
        fn, hist = by_name.get(name, (None, {}))
        nkg, w = 64 // g, c // 2
        lds = {k: v for k, v in hist.items() if k.startswith("LDS")}
        ldg = {k: v for k, v in hist.items() if k.startswith("LDG")}
        look, other = (lds, ldg) if g <= 2 else (ldg, lds)
        n, words = sum(look.values()), sum(v * _load_words(k) for k, v in look.items())
        mma = {k: v for k, v in hist.items() if F1.MMA_RE.search(k)}
        pr = regs.get(name, {})
        spill = pr.get("spill_stores", 1) + pr.get("spill_loads", 1)
        ok = bool(fn) and words == nkg * w and nkg <= n <= nkg * w and not other and not mma and spill == 0
        gates[name] = {"ok": ok, "function": fn, "lookup_loads": look, "lookups": nkg, "entry_words": w, "other_space_loads": other,
                       "mma_opcodes": mma, "loop_instructions": sum(hist.values()), "ptxas": pr, "loop_opcodes": hist}
    fn, hist = by_name.get("native", (None, {}))
    mma = {k: v for k, v in hist.items() if F1.MMA_RE.search(k)}
    pr = regs.get("native", {})
    gates["native"] = {"ok": bool(fn) and mma == {NATIVE_OPCODE: nt} and pr.get("spill_stores", 1) + pr.get("spill_loads", 1) == 0,
                       "function": fn, "mma_opcodes": mma, "loop_instructions": sum(hist.values()), "ptxas": pr}
    return gates


def build(out: Path) -> dict:
    """The library, its ptxas report, its SASS gate and FTZ scan; writes build.json."""
    if not F1._flags_ok(NVCC_FLAGS):
        raise SystemExit("refused: an -ftz / fast-math flag in NVCC_FLAGS")
    out.mkdir(parents=True, exist_ok=True)
    src = SRC.read_text()
    so = out / f"libf3_lut_{ARCH}.so"
    nvcc = P.nvcc_bin()
    cmd = [nvcc, *NVCC_FLAGS, *P.arch_flags(ARCH), "-Xptxas", "-v", "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(SRC)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    rec: dict = {"schema": SCHEMA + "/build", "arch": ARCH, "nvcc_flags": NVCC_FLAGS, "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
                 "header_sha256": hashlib.sha256(HEADER.read_bytes()).hexdigest(), "variants": variants(src), "nt": nt_of(src), "so": so.name}
    if r.returncode:
        rec.update(ok=False, error=(r.stderr + r.stdout)[-4000:])
        (out / "build.json").write_text(json.dumps(rec, indent=1) + "\n")
        return rec
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-2:]
    sass = subprocess.run([str(Path(nvcc).parent / "cuobjdump"), "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    gates = sass_gate(F1.loops_of(sass), ptxas_report(r.stderr + r.stdout), rec["variants"], rec["nt"])
    ftz = sorted(set(m.group(0) for m in F1.FTZ_RE.finditer(sass)))
    rec.update(nvcc=" / ".join(x.strip() for x in ver), so_sha256=hashlib.sha256(so.read_bytes()).hexdigest(), gate=gates, ftz_fp32=ftz,
               ok=(not ftz) and all(g["ok"] for g in gates.values()))
    (out / "build.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


# ------------------------------------------------------------------------------------------------------------------------ device

def _p(a: np.ndarray | None):
    """A kernel argument's address.  C-contiguous only: a kernel reads memory order, which for a strided array (einsum's output,
    say) is not its index order."""
    if a is None:
        return None
    if not a.flags["C_CONTIGUOUS"]:
        raise ValueError(f"kernel argument of shape {a.shape} is not C-contiguous")
    return a.ctypes.data


class F3Device:
    def __init__(self, so: Path):
        self.lib = lib = ctypes.CDLL(str(so))
        I, L, V = ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p
        pI, pL, pF = ctypes.POINTER(I), ctypes.POINTER(L), ctypes.POINTER(ctypes.c_float)
        lib.f3_variant.argtypes = [I, pI, pI, pI, pL, pL]
        lib.f3_device.argtypes = [pI] * 4
        lib.f3_occupancy.argtypes = [I, I, pI]
        lib.f3_run_lut.argtypes = [I, V, V, V, V, I, I, I, L, V, V, pF]
        lib.f3_run_native.argtypes = [V, L, V, V, I, I, I, I, I, I, L, V, V, pF]
        lib.device_uuid.restype = I
        lib.device_uuid.argtypes = [ctypes.c_char_p, I]
        lib.has_mxf4.restype = I
        lib.device_identity.restype = I
        lib.device_identity.argtypes = [pI, pI, ctypes.c_char_p, I]
        self.tcgen05 = False

    identity = P.Probe.identity
    clocks = P.Probe.clocks

    def table(self) -> tuple[list[tuple[str, int, int]], int]:
        out = []
        for v in range(self.lib.f3_nvariants()):
            a, g, c, tw, sm = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_longlong(), ctypes.c_longlong()
            self.lib.f3_variant(v, a, g, c, tw, sm)
            out.append((ARM_OF[a.value], g.value, c.value))
        return out, self.lib.f3_nt()

    def device(self) -> dict:
        v = [ctypes.c_int() for _ in range(4)]
        rc = self.lib.f3_device(*v)
        if rc:
            raise RuntimeError(f"f3_device rc={rc}")
        return {"sms": v[0].value, "max_threads_per_sm": v[1].value, "cc": f"{v[2].value}.{v[3].value}"}

    def occupancy(self, v: int, tpb: int) -> int:
        n = ctypes.c_int()
        rc = self.lib.f3_occupancy(v, tpb, ctypes.byref(n))
        if rc:
            raise RuntimeError(f"f3_occupancy({v}) rc={rc}")
        return n.value

    def run_lut(self, v, arm, c, pk: Packed, rows, ng, nblocks, iters, want=True):
        ms = ctypes.c_float()
        out = np.zeros(rows * c * (4 if arm == "dots" else 1), np.uint32)
        cyc = np.zeros(nblocks, np.int64)
        ar, c0 = np.ascontiguousarray(pk.arows[:rows]), np.ascontiguousarray(pk.c0[:rows])
        rc = self.lib.f3_run_lut(v, _p(ar), _p(pk.tables), _p(pk.wsc), _p(c0), ng, nblocks, LUT_TPB, iters, _p(out), _p(cyc), ctypes.byref(ms))
        if rc:
            raise RuntimeError(f"f3_run_lut({v}) rc={rc}")
        return ms.value, out, cyc

    def run_native(self, pk: Packed, rows, c, ng, ntiles, nblocks, iters, want=True):
        ms = ctypes.c_float()
        out = np.zeros(ntiles * 128, np.uint32) if want else None
        cyc = np.zeros(nblocks, np.int64)
        ar, c0 = np.ascontiguousarray(pk.arows[:rows]), np.ascontiguousarray(pk.c0[:rows])
        rc = self.lib.f3_run_native(_p(ar), rows, _p(pk.wrows), _p(c0), c, LUT_TPB, ng, ntiles, nblocks, NATIVE_TPB, iters, _p(out), _p(cyc),
                                    ctypes.byref(ms))
        if rc:
            raise RuntimeError(f"f3_run_native rc={rc}")
        return ms.value, out, cyc


class FakeF3:
    """CPU stand-in for the dry run: the LUT arms emulate f3_lut.cu through the packed tables and scale words, the native arm is
    the vectorised pinned model on the packed codes.  Its rates are invented (lookups at 32 per SM per clock, exact at a third of
    that, NVFP4 at 2,048 products per SM per clock); ``--fake-silicon`` never runs with ``RESEARCH_RUN_ID`` set."""

    SMS, MHZ = 2, 2000.0

    def __init__(self, table, nt):
        self._table, self._nt = table, nt

    def identity(self) -> dict:
        return {"device": "FAKE SILICON", "compute_capability": "12.0", "driver": None, "gpu_uuid": "GPU-fake", "backend": "fake",
                "cuda_visible_devices": SHARED.CUDA_VISIBLE_DEVICES}

    def clocks(self, uuid):
        return None

    def table(self):
        return self._table, self._nt

    def device(self) -> dict:
        return {"sms": self.SMS, "max_threads_per_sm": 1536, "cc": "12.0"}

    def occupancy(self, v, tpb) -> int:
        return 1

    def _ms(self, cycles: float) -> float:
        return cycles / (self.MHZ * 1e3)

    def run_lut(self, v, arm, c, pk: Packed, rows, ng, nblocks, iters, want=True):
        for a in (pk.arows, pk.tables, pk.wsc, pk.c0):
            _p(a)
        _, g, _ = self._table[v]
        nkg, nidx, w = 64 // g, 16 ** g, c // 2
        cycles = iters * (rows / self.SMS) * nkg / (32 / (3 if arm == "exact" else 1))
        cyc = np.full(nblocks, int(cycles), np.int64)
        if not want:
            return self._ms(cycles), None, cyc
        codes, sc = unpack_rows(pk.arows[:rows])
        grp = (np.arange(rows) // LUT_TPB) % ng
        idx = (codes.reshape(rows, nkg, g).astype(np.int64) << (4 * np.arange(g))).sum(-1)
        ent = pk.tables.reshape(ng, nkg, nidx, w)[grp[:, None], np.arange(nkg)[None, :], idx]
        s = ent.reshape(rows, 4, nkg // 4, w).sum(2, dtype=np.uint32)
        lo = (s & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.int64)
        hi = (s.view(np.int32).astype(np.int64) - lo) >> 16
        gd = np.zeros((rows, c, 4), np.int64)
        gd[:, 0::2, :], gd[:, 1::2, :] = lo.transpose(0, 2, 1), hi.transpose(0, 2, 1)
        if arm == "dots":
            return self._ms(cycles), ((gd * iters) & 0xFFFFFFFF).astype(np.uint32).reshape(-1), cyc
        ma, ea = V.ue4m3_dyadic(sc.astype(np.int64))
        wv = pk.wsc.astype(np.int64)[grp]
        mm = ma[:, None, :] * (wv & 15)
        x = ea[:, None, :] + ((wv >> 4) & 15) - 12
        amask = ((codes.reshape(rows, 4, 16).astype(np.int64) & 7) != 0).astype(np.int64)
        amask = (amask << np.arange(16)).sum(-1)
        part = ((amask[:, None, :] & (wv >> 8)) != 0) & (mm != 0)
        return self._ms(cycles), chain(pk.c0[:rows], gd * mm, x, part, iters).reshape(-1), cyc

    def run_native(self, pk: Packed, rows, c, ng, ntiles, nblocks, iters, want=True):
        for a in (pk.arows, pk.wrows, pk.c0):
            _p(a)
        warps = nblocks * NATIVE_TPB // 32
        cycles = iters * warps * self._nt * NATIVE_PRODUCTS / (self.SMS * 2048)
        cyc = np.full(nblocks, int(cycles), np.int64)
        if not want:
            return self._ms(cycles), None, cyc
        n = ntiles * 16
        codes, sc = unpack_rows(pk.arows[:n])
        wcodes, wscale = unpack_rows(pk.wrows)
        grp = (np.arange(n) // LUT_TPB) % ng
        _, mant, x, part = terms(codes, sc, wcodes, wscale, grp)
        d = chain(pk.c0[:n], mant, x, part, iters)
        return self._ms(cycles), d[:, np.arange(8) % c].reshape(-1), cyc


# ------------------------------------------------------------------------------------------------------------------------- run

def lut_grid(d: dict, occ: int) -> dict:
    bps = min(occ, d["max_threads_per_sm"] // LUT_TPB)
    return {"threads_per_block": LUT_TPB, "blocks_per_sm": bps, "blocks": d["sms"] * bps, "rows": d["sms"] * bps * LUT_TPB,
            "warps_per_sm": bps * LUT_TPB // 32}


def native_grid(d: dict, occ: int) -> dict:
    bps = min(occ, 4)
    return {"threads_per_block": NATIVE_TPB, "blocks_per_sm": bps, "blocks": d["sms"] * bps, "warps_per_sm": bps * NATIVE_TPB // 32}


def native_words(dev, pk: Packed, rows: int, c: int, ng: int, nt: int, iters: int) -> np.ndarray:
    """The native atom's words on the first ``rows`` rows, (rows, 8): column j is group column j mod C."""
    ntiles = rows // 16
    warps = -(-ntiles // nt)
    nblocks = -(-warps // (NATIVE_TPB // 32))
    _, out, _ = dev.run_native(pk, rows, c, ng, ntiles, nblocks, iters, want=True)
    return out.reshape(rows, 8)


def check(dev, ops: Operands, pk: Packed, gd: np.ndarray, idx: dict, geo: dict, c: int, ng: int, nt: int, length: int, rng,
          sample: int, corrupt: bool, model: bool = True) -> dict:
    """The words of both LUT arms and the native atom at one chain length, compared."""
    rd, re_ = geo["dots"]["rows"], geo["exact"]["rows"]
    _, dots, _ = dev.run_lut(idx["dots"], "dots", c, pk, rd, ng, geo["dots"]["blocks"], length)
    _, exact, _ = dev.run_lut(idx["exact"], "exact", c, pk, re_, ng, geo["exact"]["blocks"], length)
    exact = exact.reshape(re_, c).copy()
    if corrupt:
        exact[0, 0] ^= 1
    native = native_words(dev, pk, re_, c, ng, nt, length)
    want_dots = ((gd[:rd] * length) & 0xFFFFFFFF).astype(np.uint32)
    rec = {"length": length, "rows_dots": rd, "rows_exact": re_,
           "dots_mismatches": int(np.count_nonzero(dots.reshape(rd, c, 4) != want_dots)), "dots_words": int(want_dots.size),
           "exact_native_mismatches": int(np.count_nonzero(exact[:, np.arange(8) % c] != native)), "exact_words": int(exact.size),
           "native_words": int(native.size)}
    if model:
        rs, cs = rng.integers(0, re_, sample), rng.integers(0, c, sample)
        want = model_sample(ops, rs, cs, length)
        rec.update(model_sample=sample, native_model_mismatches=int(np.count_nonzero(native[rs, cs] != want)),
                   exact_model_mismatches=int(np.count_nonzero(exact[rs, cs] != want)))
    rec["ok"] = (rec["dots_mismatches"] == 0 and rec["exact_native_mismatches"] == 0 and rec.get("native_model_mismatches", 0) == 0
                 and rec.get("exact_model_mismatches", 0) == 0)
    return rec


def run_width(args) -> int:
    t_all = time.time()
    bdir, out = Path(os.path.expandvars(args.build)), Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    b = json.loads((bdir / "build.json").read_text())
    g, c = WIDTHS[args.width]
    ng, nkg = ng_of(g), 64 // g
    rec: dict = {"schema": SCHEMA, "width": args.width, "g": g, "c": c, "column_groups": ng, "lookups_per_row_atom": nkg,
                 "products_per_lookup": g * c, "favours_the_table": FAVOURS_THE_TABLE,
                 "build": {k: b.get(k) for k in ("so_sha256", "source_sha256", "nvcc", "nvcc_flags", "ok", "ftz_fp32")},
                 "seed": args.seed, "sample": args.sample, "fake_silicon": args.fake_silicon, "corrupt_word": args.corrupt_word}
    failures: list[str] = []
    if not b.get("ok"):
        failures.append("build gate failed: " + ", ".join([n for n, v in (b.get("gate") or {}).items() if not v["ok"]]
                                                          + (b.get("ftz_fp32") or []) + ([b["error"][:200]] if b.get("error") else [])))
        return finish(out, rec, failures, args, None, t_all)
    table = [tuple(x) for x in b["variants"]]
    dev = FakeF3(table, b["nt"]) if args.fake_silicon else F3Device(bdir / b["so"])
    identity = dev.identity()
    rec["identity"] = identity
    if args.expect_uuid and not args.fake_silicon and not P.uuid_allowed(identity.get("gpu_uuid"), args.expect_uuid):
        failures.append(f"device UUID {identity.get('gpu_uuid')} != expected {args.expect_uuid}: stopped before any measurement")
    lib_table, nt = dev.table()
    if [tuple(x) for x in lib_table] != table or nt != b["nt"]:
        failures.append(f"the library's variants {lib_table} / NT {nt} are not build.json's")
    if failures:
        return finish(out, rec, failures, args, None, t_all)
    d = dev.device()
    rec["device"] = d
    if d["cc"] != "12.0":
        return finish(out, rec, [f"compute capability {d['cc']} != 12.0"], args, None, t_all)
    idx = {arm: table.index((arm, g, c)) for arm in ARMS}
    geo = {arm: lut_grid(d, dev.occupancy(idx[arm], LUT_TPB)) for arm in ARMS}
    geo["native"] = native_grid(d, dev.occupancy(-1, NATIVE_TPB))
    rec["grid"] = geo
    if min(v["blocks_per_sm"] for v in geo.values()) < 1:
        return finish(out, rec, [f"a kernel does not fit on an SM: {geo}"], args, None, t_all)
    rows = max(geo[a]["rows"] for a in ARMS)
    rng = np.random.default_rng([args.seed, g, c, 1])
    fam: dict = {}
    for f in FAMILIES:
        ops = make_operands(f, rows, ng, c, args.seed)
        fam[f] = (ops, pack(ops, g), terms(ops.acodes, ops.ascale, ops.wcodes, ops.wscale, ops.groups)[0])
    gate = {}
    for f in FAMILIES:
        ops, pk, gd = fam[f]
        for length in GATE_LENGTHS:
            gate[f"{f}/L{length}"] = check(dev, ops, pk, gd, idx, geo, c, ng, nt, length, rng, args.sample,
                                           args.corrupt_word and not gate)
    rec["gate"] = gate
    failures += [f"gate {k}: {v}" for k, v in gate.items() if not v["ok"]]
    if failures:
        return finish(out, rec, failures, args, None, t_all)

    def launch(f, arm, iters, want=False):
        _, pk, _ = fam[f]
        if arm == "native":
            return dev.run_native(pk, rows, c, ng, rows // 16, geo["native"]["blocks"], iters, want)
        return dev.run_lut(idx[arm], arm, c, pk, geo[arm]["rows"], ng, geo[arm]["blocks"], iters, want)

    keys = [(f, arm) for f in TIMED for arm in ("native",) + ARMS]
    iters = {}
    for k in keys:
        ms, _, _ = launch(*k, 1000)
        iters[k] = max(1000, int(1000 * args.target_ms / max(ms, 1e-3)))
    t0 = time.time()
    while time.time() - t0 < args.sustain_s:
        launch(TIMED[0], "native", iters[(TIMED[0], "native")])
    launches: dict = {k: [] for k in keys}
    samples = []
    for rep in range(args.reps):
        for k in keys[rep % len(keys):] + keys[:rep % len(keys)]:
            ms, _, cyc = launch(*k, iters[k])
            launches[k].append({"rep": rep, "ms": ms, "cycles_max": int(cyc.max()), "sm_mhz": float(cyc.max()) / (ms * 1e3)})
        samples.append(dev.clocks(identity.get("gpu_uuid")))
    post = {}
    for f in TIMED:
        ops, pk, gd = fam[f]
        le, ld = iters[(f, "exact")], iters[(f, "dots")]
        _, ex, _ = launch(f, "exact", le, want=True)
        ex = ex.reshape(-1, c)
        nat = native_words(dev, pk, ex.shape[0], c, ng, nt, le)
        _, dots, _ = launch(f, "dots", ld, want=True)
        rd = geo["dots"]["rows"]
        want_dots = ((gd[:rd] * ld) & 0xFFFFFFFF).astype(np.uint32)
        post[f] = {"exact_length": le, "exact_native_mismatches": int(np.count_nonzero(ex[:, np.arange(8) % c] != nat)),
                   "dots_length": ld, "dots_mismatches": int(np.count_nonzero(dots.reshape(rd, c, 4) != want_dots))}
        post[f]["ok"] = post[f]["exact_native_mismatches"] == 0 and post[f]["dots_mismatches"] == 0
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
            prod, look = float(iters[(f, arm)]) * geo["native"]["blocks"] * NATIVE_TPB / 32 * nt * NATIVE_PRODUCTS, 0.0
        else:
            prod = float(iters[(f, arm)]) * geo[arm]["rows"] * 64 * c
            look = float(iters[(f, arm)]) * geo[arm]["rows"] * nkg
        per[f"{f}/{arm}"] = {"family": f, "arm": arm, "iterations": iters[(f, arm)], "ms_median": ms, "ms_all": [x["ms"] for x in ls],
                             "sm_mhz_median": mhz, "sm_mhz_all": [x["sm_mhz"] for x in ls], "warps_per_sm": geo[arm]["warps_per_sm"],
                             "products_per_sm_per_clock": prod / denom, "lookups_per_sm_per_clock": look / denom}
    for k, v in per.items():
        unit = per[f"{v['family']}/native"]["products_per_sm_per_clock"]
        v["fp4_slots_per_product"] = unit / v["products_per_sm_per_clock"]
        if v["arm"] != "native":
            v["fp4_slots_per_lookup"] = unit / v["lookups_per_sm_per_clock"]
    rec.update(per_arm=per, launches={f"{f}/{a}": v for (f, a), v in launches.items()}, clock_samples=samples,
               clocks=F1.clock_label(samples, [x["sm_mhz"] for ls in launches.values() for x in ls], args.locked_mhz),
               iterations={f"{f}/{a}": v for (f, a), v in iters.items()})
    return finish(out, rec, failures, args, per, t_all)


def finish(out: Path, rec: dict, failures: list[str], args, per: dict | None, t_all: float) -> int:
    rec["failures"] = failures
    rec["elapsed_s"] = time.time() - t_all
    name = f"{args.width}.json" if not args.fake_silicon else f"{args.width}.fake.json"
    tmp = out / (name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    tmp.replace(out / name)
    if failures:
        print("FAILURES", failures, flush=True)
        return 2 if any("UUID" in f for f in failures) else 1
    print(f"F3 {args.width} RECORDED ({rec.get('clocks')})", flush=True)
    for k, v in (per or {}).items():
        extra = f"  {v['fp4_slots_per_lookup']:7.2f} FP4 slots/lookup" if "fp4_slots_per_lookup" in v else ""
        print(f"  {k:18s} {v['products_per_sm_per_clock']:9.1f} products/SM/clk  {v['fp4_slots_per_product']:8.3f} FP4 slots/product"
              f"{extra}  {v['sm_mhz_median']:7.1f} MHz  {v['warps_per_sm']} warps/SM", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------------------- collect

def collect(args) -> int:
    """The price table over every recorded width (each priced against the NVFP4 of its own run) and the claim's check."""
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    bj = Path(os.path.expandvars(args.build)) / "build.json"
    b = json.loads(bj.read_text())
    kept = out / "inputs"
    kept.mkdir(exist_ok=True)
    (kept / "build.json").write_bytes(bj.read_bytes())
    rows, widths, failures = {}, {}, []
    for d in [x for x in args.runs.split(",") if x]:
        for p in sorted(Path(os.path.expandvars(d)).glob("*.json")):
            r = json.loads(p.read_text())
            if r.get("schema") != SCHEMA:
                continue
            w = r["width"]
            if w in widths:
                failures.append(f"{w}: recorded twice ({widths[w]['file']}, {p})")
                continue
            (kept / p.name).write_bytes(p.read_bytes())
            widths[w] = {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "gpu_uuid": (r.get("identity") or {}).get("gpu_uuid"),
                         "clocks": r.get("clocks"), "failures": r.get("failures"), "build_so_sha256": r["build"].get("so_sha256"),
                         "fake_silicon": r.get("fake_silicon"), "grid": r.get("grid")}
            if r.get("fake_silicon"):
                failures.append(f"{w}: fake silicon (a dry run's invented numbers)")
            if r.get("failures"):
                failures.append(f"{w}: {r['failures']}")
                continue
            if r["build"].get("so_sha256") != b.get("so_sha256"):
                failures.append(f"{w}: ran a different build ({str(r['build'].get('so_sha256'))[:12]})")
            for k, v in r["per_arm"].items():
                rows[f"{w}/{k}"] = {**v, "width": w, "g": r["g"], "c": r["c"], "products_per_lookup": r["products_per_lookup"],
                                    "gpu_uuid": widths[w]["gpu_uuid"]}
    missing = [w for w in WIDTHS if w not in widths]
    lut = {k: v for k, v in rows.items() if v["arm"] != "native"}
    cheaper = {k: v["fp4_slots_per_product"] for k, v in lut.items() if v["arm"] == "exact" and v["fp4_slots_per_product"] < 1 - args.tolerance}
    lookups = [v["fp4_slots_per_lookup"] for v in lut.values() if v["arm"] == "dots"]
    doc = {"schema": SCHEMA + "/table", "unit": "one dense NVFP4 product (mma.sync kind::mxf4nvf4 4X ue4m3 m16n8k64) per SM per clock, "
                                              "measured in each width's own run",
           "build_so_sha256": b.get("so_sha256"), "widths": widths, "missing_widths": missing, "rows": rows,
           "sorted_fp4_slots_per_product": sorted(((k, v["fp4_slots_per_product"]) for k, v in lut.items()), key=lambda x: x[1]),
           "claim": {"statement": "fp4-tile-only/sm120: no LUT-GEMM width computes the NVFP4 atom's words for fewer FP4 slots per product "
                                  "than dense NVFP4, with the table's advantages of FAVOURS_THE_TABLE",
                     "tolerance": args.tolerance, "cheaper_exact": cheaper, "holds_over_measured": not cheaper and not missing},
           "lookup_price": {"fp4_slots_per_lookup_dots": [min(lookups), max(lookups)] if lookups else None,
                            "model": list(MODEL_LOOKUP_SLOTS), "label": "Measured (dots arm); the model's range Derived (a2/fp4-tile)"},
           "favours_the_table": FAVOURS_THE_TABLE, "label": "Measured"}
    (out / "f3_lut.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    ok = not failures and not missing
    meas = [{"name": f"{k}_fp4_slots_per_product", "value": v["fp4_slots_per_product"], "unit": "NVFP4 products"} for k, v in rows.items()]
    meas += [{"name": f"{k}_fp4_slots_per_lookup", "value": v["fp4_slots_per_lookup"], "unit": "NVFP4 products"} for k, v in lut.items()]
    meas += [{"name": f"{k}_products_per_sm_per_clock", "value": v["products_per_sm_per_clock"], "unit": "products/(SM*cycle)"}
             for k, v in rows.items()]
    clocks = sorted({str(w["clocks"]) for w in widths.values()})
    res = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": SHARED.RESEARCH_RUN_ID,
           "clocks": clocks[0] if len(clocks) == 1 else "mixed", "label": "Measured" if clocks == ["locked-2100"] else "Measured, " + ",".join(clocks),
           "method": "F3 LUT-GEMM kernels from the fill queue (one width per chunk, each against dense NVFP4 in the same process), "
                     "collected; see f3_lut.json",
           "workload_fingerprint": {"tool": "tc_lut_fp4", "version": "1", "build_so_sha256": b.get("so_sha256"),
                                    "widths": {k: v["sha256"] for k, v in widths.items()}, "commit": SHARED.RESEARCH_SOURCE_COMMIT},
           "validation": {"status": "passed" if ok else "failed", "checker": "every width's gates (UUID, cc 12.0, SASS, exact words = the "
                          "native atom's on every word and the pinned model's on a sample, dots = the integer group dots, at chains of 1 and 3 "
                          "and at the timed length) passed in its chunk, on this build",
                          "detail": "; ".join(failures) or f"{len(widths)} widths, {len(rows)} rows", "evidence": ["f3_lut.json"]},
           "claim": doc["claim"], "measurements": meas, "measurement_files": ["f3_lut.json"], "artifacts": []}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("F3 TABLE", "ok" if ok else "FAILED", failures, missing, flush=True)
    for k, p in doc["sorted_fp4_slots_per_product"]:
        print(f"  {k:24s} {p:9.3f} FP4 slots per product  {rows[k]['fp4_slots_per_lookup']:8.2f} per lookup", flush=True)
    return 0 if ok else 1


def build_cmd(args) -> int:
    rec = build(Path(os.path.expandvars(args.out)))
    print(f"F3 BUILD {'ok' if rec['ok'] else 'FAILED'}: FTZ {rec.get('ftz_fp32') or 'none'}", flush=True)
    for n, g in (rec.get("gate") or {}).items():
        print(f"  {n:12s} {'ok ' if g['ok'] else 'BAD'} {g['ptxas'].get('registers')} regs  loop {g['loop_instructions']} instr  "
              f"{g.get('lookup_loads') or g.get('mma_opcodes')}", flush=True)
    if rec.get("error"):
        print(rec["error"], flush=True)
    if SHARED.RESEARCH_RUN_ID:
        res = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": SHARED.RESEARCH_RUN_ID,
               "label": "Measured (ptxas and SASS of the F3 kernels; no device)",
               "method": "f3_lut.cu compiled for sm_120a with -Xptxas -v; each kernel's timed loop read from cuobjdump",
               "workload_fingerprint": {"tool": "tc_lut_fp4", "version": "1", "step": "build", "so_sha256": rec.get("so_sha256"),
                                        "source_sha256": rec["source_sha256"], "commit": SHARED.RESEARCH_SOURCE_COMMIT},
               "validation": {"status": "passed" if rec["ok"] else "failed",
                              "checker": "lookups per LUT loop, NT OMMA in the native loop, no spills, no FADD/FFMA/FMUL .FTZ",
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
    r.add_argument("--width", required=True, choices=sorted(WIDTHS))
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
    c.add_argument("--runs", required=True, help="comma list of directories holding <width>.json")
    c.add_argument("--out", required=True)
    c.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args(argv)
    if getattr(args, "fake_silicon", False) and SHARED.RESEARCH_RUN_ID:
        ap.error("--fake-silicon is a dry run and never publishes: unset RESEARCH_RUN_ID")
    return {"build": build_cmd, "run": run_width, "collect": collect}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
