"""W1 price microbenchmarks for the FP4 path on sm_120a: what one MMA of each row costs on this card.

    python benchmarks/tc_probe_fp4/w1_price.py --out $RESEARCH_RUN_DIR --expect-uuid GPU-... [--rows nvf4,mxf4,...] [--reps 10]
    python benchmarks/tc_probe_fp4/w1_price.py --out /tmp/w1 --sass-only          # build + SASS gate, no device
    python benchmarks/tc_probe_fp4/w1_price.py --out /tmp/w1 --fake-silicon       # CPU dry run (refused with RESEARCH_RUN_ID set)

Each row is one ``mma.sync`` of ``rows_fp4.cuh`` (the mnemonic the probe's capture gates), timed by ``w1_price.cu``:

  throughput  NSETS x NACC register-resident MMAs per loop iteration from every warp of every SM (``warps_per_sm`` warps each);
              for the scaled rows a second variant rotates every scale register a byte per iteration (``scale_update``), so
              the scale values change as a GEMM's do
  latency     one warp, one accumulator, a dependent chain (cycles per MMA, from clock64)

Every launch's SM clock is its longest thread-0 clock64 span over its event time, so each rate is also stated per SM per clock,
which does not depend on the clock.  The rows:

  nvf4                 kind::mxf4nvf4 4X ue4m3 m16n8k64      OMMA  the honest dense NVFP4 rate
  mxf4                 kind::mxf4nvf4 2X ue8m0 m16n8k64      OMMA
  f8f6f4               kind::f8f6f4 e2m1 x e2m1 m16n8k32     QMMA  unscaled E2M1 in 8-bit containers
  f8f6f4_e2m1_e4m3     kind::f8f6f4 e2m1 x e4m3 m16n8k32     QMMA
  mxf8f6f4_e2m1        kind::mxf8f6f4 1X ue8m0 e2m1 x e2m1   QMMA
  mxf8f6f4_e2m1_e4m3   kind::mxf8f6f4 1X ue8m0 e2m1 x e4m3   QMMA
  f8f6f4_e4m3          kind::f8f6f4 e4m3 x e4m3 m16n8k32     QMMA  the FP8 reference: the W1 unit is one of its MACs
  sp_nvf4              mma.sp m16n8k128 mxf4nvf4 4X ue4m3    OMMA.SP  2:4 sparse NVF4 (logical K 128, 64 stored), valid metadata
  sp_mxf4              mma.sp m16n8k128 mxf4nvf4 2X ue8m0    OMMA.SP

and the conversion and scale ops of an FP4 quantizer (``--ops``; ``w1_price.cu``'s header lists each loop and its glue):
``cvt_e2m1x2_f32`` (the E2M1 cast), ``cvt_f16x2_e2m1x2`` (its decode), ``cvt_e4m3x2_f32_ue4m3`` / ``cvt_f16x2_e4m3x2_ue4m3``
(the UE4M3 scale's encode / decode), ``cvt_ue8m0x2_f32_rp`` (the UE8M0 scale's encode), ``fmax_abs`` (a block amax),
``fmul`` (the per-element scale multiply), ``rcp_approx`` (a scale's reciprocal) and ``lop3`` (the glue alone), each at full
occupancy, priced per element (or per scale) in dense FP8 MACs.

Gates, all before any timing (a failed gate writes no measurements): the device UUID (``--expect-uuid``) and compute capability
12.0; cuobjdump counts exactly NSETS x NACC of the row's opcode in each throughput kernel and LAT_UNROLL in each latency kernel,
and exactly the op's count in each conversion loop's body (the loop's full opcode histogram is recorded); every kernel's outputs
are finite and bit-identical across two launches; every launch is one wave (all its blocks resident, so thread 0's clock64
span is the launch's).  ``--corrupt-word`` flips one output word of the second
gate launch: the run must then fail.  Nothing here checks the arithmetic; ``probe.py`` does.

Clocks: each rep samples ``nvidia-smi -i <uuid>`` (SM and memory clock, power, temperature, throttle reasons).  The run is
labelled ``locked-F`` only with ``--locked-mhz F`` and every sample and every launch's SM clock within 1% of F (a lock at
2,100 MHz reads 2,092), else ``unlocked``; a container can't lock clocks, so RunPod runs are ``unlocked``.

Honest notes.  Operands are uniform random codes (every E2M1 code, finite E4M3 bytes, UE4M3 scales 0x30..0x47, UE8M0 124..131),
a GEMM's data rather than zeros; an attacker's zeros could draw less power, which is not measured here.  The K=32 rows take
E2M1 in 8-bit containers, so a GEMM that stores packed nibbles pays an unpack this instruction-level rate leaves out.  The
throughput is an upper bound on any kernel built from the instruction, not a GEMM rate.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402
import sparse as S  # noqa: E402
from verity_config.settings import SHARED  # noqa: E402

SOURCE = "w1_price.cu"
ARCH = "sm_120a"
SCHEMA = "pouw-rtxpro/w1-price-fp4/v2"
NSETS, NACC, WORDS, LAT_UNROLL = 4, 8, 12, 8
BODY = NSETS * NACC
CV_CHAINS, CV_UNROLL = 8, 4
FP8_REF = ("f8f6f4_e4m3", 0)

#: row -> (RowKind in rows_fp4.cuh, logical K, A format, B format, scale format or None, SASS opcode)
ROWS = {
    "nvf4": (0, 64, "e2m1", "e2m1", "ue4m3", "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"),
    "mxf4": (1, 64, "e2m1", "e2m1", "ue8m0", "OMMA.SF.16864.F32.E2M1.E2M1.E8"),
    "f8f6f4": (2, 32, "e2m1", "e2m1", None, "QMMA.16832.F32.E2M1.E2M1"),
    "f8f6f4_e2m1_e4m3": (3, 32, "e2m1", "e4m3", None, "QMMA.16832.F32.E2M1.E4M3"),
    "mxf8f6f4_e2m1": (5, 32, "e2m1", "e2m1", "ue8m0", "QMMA.SF.16832.F32.E2M1.E2M1.E8"),
    "mxf8f6f4_e2m1_e4m3": (6, 32, "e2m1", "e4m3", "ue8m0", "QMMA.SF.16832.F32.E2M1.E4M3.E8"),
    "f8f6f4_e4m3": (8, 32, "e4m3", "e4m3", None, "QMMA.16832.F32.E4M3.E4M3"),
    "sp_nvf4": (9, 128, "e2m1", "e2m1", "ue4m3", S.SASS_OPCODE["sp_nvf4"]),
    "sp_mxf4": (10, 128, "e2m1", "e2m1", "ue8m0", S.SASS_OPCODE["sp_mxf4"]),
}
SPARSE_ROWS = ("sp_nvf4", "sp_mxf4")
#: op -> (CvtOp in w1_price.cu, SASS opcode of the op, ops per chain step, elements (or scales) per op, per what, role)
OPS = {
    "cvt_e2m1x2_f32": (0, "F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C", 1, 2, "element", "the E2M1 cast (cvt.rn.satfinite.e2m1x2.f32)"),
    "cvt_f16x2_e2m1x2": (1, "F2FP.F16.E2M1.UNPACK_B", 1, 2, "element", "the E2M1 decode (cvt.rn.f16x2.e2m1x2)"),
    "cvt_e4m3x2_f32_ue4m3": (2, "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", 1, 2, "scale",
                             "the UE4M3 scale encode (cvt.rn.satfinite.e4m3x2.f32, positive inputs)"),
    "cvt_f16x2_e4m3x2_ue4m3": (3, "F2FP.F16.E4M3.UNPACK_B", 1, 2, "scale", "the UE4M3 scale decode (cvt.rn.f16x2.e4m3x2)"),
    "cvt_ue8m0x2_f32_rp": (4, "F2FP.SATFINITE.E8.F32.PACK_AB_MERGE_C.RP", 1, 2, "scale", "the UE8M0 scale encode (cvt.rp.satfinite.ue8m0x2.f32)"),
    "fmax_abs": (5, "FMNMX", 2, 1, "element", "a block amax step (max.f32 with |x|)"),
    "fmul": (6, "FMUL", 1, 1, "element", "the per-element scale multiply (mul.rn.f32)"),
    "rcp_approx": (7, "MUFU.RCP", 1, 1, "scale", "a scale's reciprocal (rcp.approx.ftz.f32)"),
    "lop3": (8, "LOP3.LUT", 1, 1, "op", "the LOP3 the encode and rcp chains carry (lop3.b32), alone"),
}
#: an NVFP4 quantizer's work per element as op counts (Estimated: the op list is this recipe's, TE's adds a global-scale
#: multiply): amax and scale multiply per element, the E2M1 cast per element, and per 16-element block the scale's amax/6
#: multiply, its UE4M3 encode and decode and one reciprocal
NVF4_QUANTIZE_RECIPE = {"fmax_abs": 1.0, "fmul": 1.0 + 1 / 16, "cvt_e2m1x2_f32": 1.0, "cvt_e4m3x2_f32_ue4m3": 1 / 16,
                        "cvt_f16x2_e4m3x2_ue4m3": 1 / 16, "rcp_approx": 1 / 16}
PTX = {
    "nvf4": "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
    "mxf4": "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
    "f8f6f4": "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32",
    "f8f6f4_e2m1_e4m3": "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32",
    "mxf8f6f4_e2m1": "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0",
    "mxf8f6f4_e2m1_e4m3": "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0",
    "f8f6f4_e4m3": "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32",
    **S.PTX,
}
MODES = {0: "const_scales", 1: "scale_update"}
#: the named comparisons (numerator row/mode, denominator row/mode), each a ratio of logical flop rates
COMPARISONS = {
    "mxf4_over_nvf4": (("mxf4", 0), ("nvf4", 0)),
    "f8f6f4_e2m1_over_nvf4": (("f8f6f4", 0), ("nvf4", 0)),
    "f8f6f4_e2m1_e4m3_over_nvf4": (("f8f6f4_e2m1_e4m3", 0), ("nvf4", 0)),
    "mxf8f6f4_e2m1_over_nvf4": (("mxf8f6f4_e2m1", 0), ("nvf4", 0)),
    "f8f6f4_e2m1_over_f8f6f4_e4m3": (("f8f6f4", 0), ("f8f6f4_e4m3", 0)),
    "mxf8f6f4_e2m1_over_f8f6f4_e2m1": (("mxf8f6f4_e2m1", 0), ("f8f6f4", 0)),
    "nvf4_scale_update_over_const": (("nvf4", 1), ("nvf4", 0)),
    "mxf4_scale_update_over_const": (("mxf4", 1), ("mxf4", 0)),
    "mxf8f6f4_e2m1_scale_update_over_const": (("mxf8f6f4_e2m1", 1), ("mxf8f6f4_e2m1", 0)),
    "mxf8f6f4_e2m1_e4m3_scale_update_over_const": (("mxf8f6f4_e2m1_e4m3", 1), ("mxf8f6f4_e2m1_e4m3", 0)),
    "sp_nvf4_over_nvf4": (("sp_nvf4", 0), ("nvf4", 0)),
    "sp_mxf4_over_mxf4": (("sp_mxf4", 0), ("mxf4", 0)),
    "sp_nvf4_scale_update_over_const": (("sp_nvf4", 1), ("sp_nvf4", 0)),
}
_E4M3_FINITE = np.array([c for c in range(256) if c & 0x7F != 0x7F], np.uint32)


def flop_per_mma(row: str) -> int:
    return 2 * 16 * 8 * ROWS[row][1]


def _bytes(rng, fmt: str, shape) -> np.ndarray:
    if fmt == "e2m1":  # a code at bits [2,6) of its 8-bit container
        return rng.integers(0, 16, size=shape, dtype=np.uint32) << 2
    if fmt == "e4m3":
        return _E4M3_FINITE[rng.integers(0, len(_E4M3_FINITE), size=shape)]
    if fmt == "ue4m3":  # 2^-3 .. 2^4 x mantissa: accumulators stay finite over any run length used here
        return rng.integers(0x30, 0x48, size=shape, dtype=np.uint32)
    if fmt == "ue8m0":
        return rng.integers(124, 132, size=shape, dtype=np.uint32)
    raise ValueError(fmt)


def _pack(b: np.ndarray) -> np.ndarray:
    return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)


def fragments(row: str, nwarps: int, seed: int) -> np.ndarray:
    """u32[nwarps*32][NSETS][WORDS]: a[4] b[4] sfa sfb e pad per operand set per lane, uniform random codes of the row's formats
    (b[2..3] only for the sparse rows, whose e is eight valid metadata nibbles; zero elsewhere)."""
    kind, k, fa, fb, sf, _ = ROWS[row]
    rng = np.random.default_rng([seed, kind])
    lanes = nwarps * 32
    f = np.zeros((lanes, NSETS, WORDS), np.uint32)
    if k in (64, 128):  # E2M1 nibbles, every code of both signs
        nb = 4 if row in SPARSE_ROWS else 2
        f[:, :, :4 + nb] = rng.integers(0, 2**32, size=(lanes, NSETS, 4 + nb), dtype=np.uint64).astype(np.uint32)
    else:
        f[:, :, 0:4] = _pack(_bytes(rng, fa, (lanes, NSETS, 4, 4)))
        f[:, :, 4:6] = _pack(_bytes(rng, fb, (lanes, NSETS, 2, 4)))
    if sf:
        f[:, :, 8:10] = _pack(_bytes(rng, sf, (lanes, NSETS, 2, 4)))
    if row in SPARSE_ROWS:
        nib = S.META_VALID[rng.integers(0, len(S.META_VALID), size=(lanes, NSETS, 8))].astype(np.uint32)
        f[:, :, 10] = (nib << (4 * np.arange(8, dtype=np.uint32))).sum(axis=-1, dtype=np.uint32)
    return f


def _f32_words(rng, shape, lo: int, hi: int, signed: bool) -> np.ndarray:
    v = rng.uniform(1.0, 2.0, size=shape) * np.exp2(rng.integers(lo, hi, size=shape))
    if signed:
        v *= rng.choice([-1.0, 1.0], size=shape)
    return v.astype(np.float32).view(np.uint32)


def op_inputs(op: str, nthreads: int, seed: int) -> np.ndarray:
    """u32[nthreads][CV_CHAINS][2]: each chain's (x, y), in the domain the op sees in a quantizer."""
    rng = np.random.default_rng([seed, 100 + OPS[op][0]])
    shape = (nthreads, CV_CHAINS)
    x = y = None
    if op == "cvt_e2m1x2_f32":  # scaled activations: |x| in [2^-3, 2^4), both signs, past E2M1's 6 too
        x, y = _f32_words(rng, shape, -3, 3, True), _f32_words(rng, shape, -3, 3, True)
    elif op in ("cvt_e4m3x2_f32_ue4m3", "cvt_ue8m0x2_f32_rp"):  # block scales: positive, 2^-6 .. 2^8
        x, y = _f32_words(rng, shape, -6, 8, False), _f32_words(rng, shape, -6, 8, False)
    elif op == "cvt_f16x2_e4m3x2_ue4m3":  # two finite E4M3 bytes in the low half
        x = _E4M3_FINITE[rng.integers(0, len(_E4M3_FINITE), size=shape)] | (_E4M3_FINITE[rng.integers(0, len(_E4M3_FINITE), size=shape)] << 8)
        y = np.zeros(shape, np.uint32)
    elif op in ("cvt_f16x2_e2m1x2", "lop3"):
        x = rng.integers(0, 2**32, size=shape, dtype=np.uint64).astype(np.uint32)
        y = rng.integers(0, 2**32, size=shape, dtype=np.uint64).astype(np.uint32) if op == "lop3" else np.zeros(shape, np.uint32)
    elif op == "fmax_abs":
        x, y = _f32_words(rng, shape, -8, 8, True), _f32_words(rng, shape, -8, 8, True)
    elif op == "fmul":  # y = 1.0f: the chain stays finite for any length
        x, y = _f32_words(rng, shape, -8, 8, True), np.full(shape, 0x3F800000, np.uint32)
    elif op == "rcp_approx":  # y = 0: the chain's LOP3 leaves x alone, and x <-> 1/x stays in range
        x, y = _f32_words(rng, shape, -4, 4, False), np.zeros(shape, np.uint32)
    return np.stack([x, y], axis=-1).astype(np.uint32)


def sass_loops(so: Path) -> dict:
    """{kernel: opcode histogram of its innermost timed loop} from ``cuobjdump -sass``: the instructions from a backward branch's
    target to the branch (the last such loop in the function: the timed one; the sink reduction after it has none)."""
    cuobjdump = str(Path(P.nvcc_bin()).parent / "cuobjdump")
    text = P.subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    return loops_of(text)


def loops_of(text: str) -> dict:
    out: dict = {}
    fn, ins = None, []

    def flush():
        if fn is None:
            return
        at = {a: i for i, (a, _) in enumerate(ins)}
        span = None
        for i, (a, toks) in enumerate(ins):
            if _opcode(toks).startswith("BRA"):
                m = re.search(r"0x([0-9a-f]+)", " ".join(toks))
                if m and int(m.group(1), 16) < a and int(m.group(1), 16) in at:
                    span = (at[int(m.group(1), 16)], i)
        hist: dict = {}
        for _, toks in (ins[span[0]:span[1] + 1] if span else []):
            hist[_opcode(toks)] = hist.get(_opcode(toks), 0) + 1
        out[fn] = hist

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


def _opcode(toks: list[str]) -> str:
    return toks[1] if toks[0].startswith("@") and len(toks) > 1 else toks[0]


def op_kernel(op: str) -> str:
    return f"cvt_tput_kernelILi{OPS[op][0]}EE"


def ops_sass_gate(so: Path, ops: list[str]) -> dict:
    per_fn = sass_loops(so)
    out, ok = {}, True
    for op in ops:
        recs = [h for fn, h in per_fn.items() if op_kernel(op) in fn]
        want = CV_CHAINS * CV_UNROLL * OPS[op][2]
        hist = recs[0] if len(recs) == 1 else {}
        good = len(recs) == 1 and hist.get(OPS[op][1], 0) == want
        ok &= good
        out[op_kernel(op)] = {"loop_opcodes": hist, "expected_opcode": OPS[op][1], "expected": want, "ok": good}
    return {"ok": ok, "kernels": out}


def variants(rows: list[str], scale_update: bool) -> list[tuple[str, int]]:
    return [(r, m) for r in rows for m in ((0, 1) if (scale_update and ROWS[r][4]) else (0,))]


def kernel_name(row: str, mode: int | None) -> str:
    kind = ROWS[row][0]
    return f"price_latency_kernelILi{kind}EE" if mode is None else f"price_tput_kernelILi{kind}ELi{mode}EE"


def sass_gate(so: Path, rows: list[str], scale_update: bool) -> dict:
    per_fn = P.sass_check(so)
    out, ok = {}, True
    for row, mode in variants(rows, scale_update) + [(r, None) for r in rows]:
        name = kernel_name(row, mode)
        recs = [v for fn, v in per_fn.items() if name in fn]
        want = BODY if mode is not None else LAT_UNROLL
        got = recs[0] if len(recs) == 1 else {"mma_count": 0, "opcodes": []}
        good = len(recs) == 1 and got["mma_count"] == want and set(got["opcodes"]) == {ROWS[row][5]}
        ok &= good
        out[name] = {"mma_count": got["mma_count"], "expected": want, "opcodes": sorted(set(got["opcodes"])),
                     "expected_opcode": ROWS[row][5], "ok": good}
    return {"ok": ok, "kernels": out}


class W1Device:
    def __init__(self, so: Path):
        self.lib = lib = ctypes.CDLL(str(so))
        V, I, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        lib.w1_device.argtypes = [ctypes.POINTER(I)] * 4
        lib.w1_setup.argtypes = [V, I, I, I]
        lib.w1_run.argtypes = [I, I, I, I, I, L, ctypes.POINTER(ctypes.c_float), V, V]
        lib.w1_latency.argtypes = [I, I, L, V, V]
        lib.w1_occupancy.argtypes = [I, I, I, ctypes.POINTER(I)]
        lib.device_uuid.restype = I
        lib.device_uuid.argtypes = [ctypes.c_char_p, I]
        lib.has_mxf4.restype = I
        lib.device_identity.restype = I
        lib.device_identity.argtypes = [ctypes.POINTER(I), ctypes.POINTER(I), ctypes.c_char_p, I]
        lib.w1_cvt_setup.argtypes = [V, I, I, I]
        lib.w1_cvt_occupancy.argtypes = [I, I, ctypes.POINTER(I)]
        lib.w1_cvt_run.argtypes = [I, I, I, I, L, ctypes.POINTER(ctypes.c_float), V, V]
        if (lib.w1_cvt_chains(), lib.w1_cvt_unroll(), lib.w1_words_per_lane()) != (CV_CHAINS, CV_UNROLL, NSETS * WORDS):
            raise RuntimeError("w1_price.cu's CV_CHAINS / CV_UNROLL / WORDS differ from w1_price.py's")
        self.tcgen05 = False

    identity = P.Probe.identity
    clocks = P.Probe.clocks

    def device(self) -> dict:
        v = [ctypes.c_int() for _ in range(4)]
        rc = self.lib.w1_device(*v)
        if rc:
            raise RuntimeError(f"w1_device rc={rc}")
        return {"sms": v[0].value, "max_threads_per_sm": v[1].value, "cc": f"{v[2].value}.{v[3].value}"}

    def occupancy(self, kind: int, mode: int, wpb: int) -> int:
        n = ctypes.c_int()
        rc = self.lib.w1_occupancy(kind, mode, wpb, ctypes.byref(n))
        if rc:
            raise RuntimeError(f"w1_occupancy(kind={kind}, mode={mode}) rc={rc}")
        return n.value

    def setup(self, frags: np.ndarray, nwarps: int, nslots: int, nblocks: int) -> None:
        self._frags = np.ascontiguousarray(frags, np.uint32)
        rc = self.lib.w1_setup(self._frags.ctypes.data, nwarps, nslots, nblocks)
        if rc:
            raise RuntimeError(f"w1_setup rc={rc}")

    def run(self, kind: int, mode: int, slot: int, nblocks: int, wpb: int, iters: int, want: bool = False):
        ms = ctypes.c_float()
        sink = np.zeros(nblocks * wpb * 32 * 4, np.float32) if want else None
        cyc = np.zeros(nblocks, np.int64)
        rc = self.lib.w1_run(kind, mode, slot, nblocks, wpb, iters, ctypes.byref(ms), sink.ctypes.data if want else None, cyc.ctypes.data)
        if rc:
            raise RuntimeError(f"w1_run(kind={kind}, mode={mode}) rc={rc}")
        return ms.value, sink, cyc

    def latency(self, kind: int, slot: int, iters: int):
        sink = np.zeros(32 * 4, np.float32)
        cyc = np.zeros(1, np.int64)
        rc = self.lib.w1_latency(kind, slot, iters, sink.ctypes.data, cyc.ctypes.data)
        if rc:
            raise RuntimeError(f"w1_latency(kind={kind}) rc={rc}")
        return sink, int(cyc[0])

    def cvt_occupancy(self, op: int, tpb: int) -> int:
        n = ctypes.c_int()
        rc = self.lib.w1_cvt_occupancy(op, tpb, ctypes.byref(n))
        if rc:
            raise RuntimeError(f"w1_cvt_occupancy(op={op}) rc={rc}")
        return n.value

    def cvt_setup(self, init: np.ndarray, nthreads: int, nslots: int, nblocks: int) -> None:
        self._init = np.ascontiguousarray(init, np.uint32)
        rc = self.lib.w1_cvt_setup(self._init.ctypes.data, nthreads, nslots, nblocks)
        if rc:
            raise RuntimeError(f"w1_cvt_setup rc={rc}")

    def cvt_run(self, op: int, slot: int, nblocks: int, tpb: int, iters: int, want: bool = False):
        ms = ctypes.c_float()
        sink = np.zeros(nblocks * tpb, np.uint32) if want else None
        cyc = np.zeros(nblocks, np.int64)
        rc = self.lib.w1_cvt_run(op, slot, nblocks, tpb, iters, ctypes.byref(ms), sink.ctypes.data if want else None, cyc.ctypes.data)
        if rc:
            raise RuntimeError(f"w1_cvt_run(op={op}) rc={rc}")
        return ms.value, sink, cyc

    def teardown(self) -> None:
        self.lib.w1_teardown()


class FakeW1:
    """CPU stand-in for the driver's dry run: a 188-SM part at 2,000 MHz issuing one MMA per SM per 8 clocks (K=64) or 4 (K=32).
    Its numbers are invented; ``--fake-silicon`` never runs with ``RESEARCH_RUN_ID`` set."""

    tcgen05 = False
    SMS, MHZ = 188, 2000.0

    def identity(self) -> dict:
        return {"device": "FAKE SILICON", "compute_capability": "12.0", "driver": None, "gpu_uuid": "GPU-fake", "backend": "fake",
                "cuda_visible_devices": SHARED.CUDA_VISIBLE_DEVICES}

    def clocks(self, uuid):
        return None

    def device(self) -> dict:
        return {"sms": self.SMS, "max_threads_per_sm": 1536, "cc": "12.0"}

    def occupancy(self, kind, mode, wpb) -> int:
        return 4

    def setup(self, frags, nwarps, nslots, nblocks) -> None:
        self._frags = frags

    def _cpm(self, kind: int) -> int:
        return 16 if kind in (9, 10) else 8 if kind in (0, 1) else 4

    def run(self, kind, mode, slot, nblocks, wpb, iters, want=False):
        warps_per_sm = nblocks * wpb / self.SMS
        cycles = iters * BODY * warps_per_sm * self._cpm(kind) * (1.01 if mode else 1.0)
        ms = cycles / (self.MHZ * 1e3)
        sink = np.full(nblocks * wpb * 32 * 4, float(kind + slot), np.float32) if want else None
        return ms, sink, self._spans(nblocks, cycles)

    @staticmethod
    def _spans(nblocks: int, cycles: float) -> np.ndarray:
        """Blocks sharing an SM finish at different times, as on silicon: only the longest span is the launch's."""
        return (np.linspace(0.6, 1.0, nblocks) * cycles).astype(np.int64)

    def latency(self, kind, slot, iters):
        return np.zeros(128, np.float32), iters * LAT_UNROLL * (20 if kind in (0, 1) else 16)

    def cvt_occupancy(self, op, tpb) -> int:
        return 6

    def cvt_setup(self, init, nthreads, nslots, nblocks) -> None:
        self._init = init

    def cvt_run(self, op, slot, nblocks, tpb, iters, want=False):
        per_clk = 16.0 if op in (0, 1, 2, 3, 4, 7) else 64.0  # thread-ops per SM per clock (invented)
        steps = iters * CV_CHAINS * CV_UNROLL * nblocks * tpb / self.SMS
        cycles = steps / per_clk
        ms = cycles / (self.MHZ * 1e3)
        sink = np.full(nblocks * tpb, op + slot, np.uint32) if want else None
        return ms, sink, self._spans(nblocks, cycles)

    def teardown(self) -> None:
        pass


def _median(xs):
    return statistics.median(xs) if xs else None


def clock_label(samples: list[dict | None], launch_mhz: list[float], locked_mhz: float | None) -> str:
    if locked_mhz is None or not samples or any(s is None for s in samples):
        return "unlocked"
    try:
        at = all(abs(float(s["clocks.sm"]) - locked_mhz) <= 0.01 * locked_mhz for s in samples)
    except (KeyError, TypeError, ValueError):
        return "unlocked"
    near = all(abs(m - locked_mhz) <= 0.01 * locked_mhz for m in launch_mhz)
    return f"locked-{locked_mhz:g}" if (at and near) else "unlocked"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(ROWS), help="comma list of " + ", ".join(ROWS))
    ap.add_argument("--reps", type=int, default=10, help="timed launches per variant, interleaved round-robin")
    ap.add_argument("--target-ms", type=float, default=100.0, help="calibrated length of one timed launch")
    ap.add_argument("--sustain-s", type=float, default=3.0, help="back-to-back nvf4 launches before the first rep")
    ap.add_argument("--warps-per-sm", type=int, default=16)
    ap.add_argument("--ops", default=",".join(OPS), help="comma list of " + ", ".join(OPS) + " (empty: none)")
    ap.add_argument("--op-warps-per-sm", type=int, default=None, help="the conversion loops' warps per SM (default: the SM's maximum)")
    ap.add_argument("--latency-iters", type=int, default=20000, help="dependent chain length / LAT_UNROLL")
    ap.add_argument("--no-scale-update", action="store_true")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-seconds", type=float, default=900.0, help="refuse a plan longer than this (quiet windows are 20 min)")
    ap.add_argument("--expect-uuid", default=None)
    ap.add_argument("--locked-mhz", type=float, default=None, help="the clock the GPU was locked at outside this run, if it was")
    ap.add_argument("--corrupt-word", action="store_true", help="gate self-test: one flipped output word must fail the run")
    ap.add_argument("--fake-silicon", action="store_true")
    ap.add_argument("--sass-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in args.rows.split(",") if r]
    ops = [o for o in args.ops.split(",") if o]
    bad = [r for r in rows if r not in ROWS] + [o for o in ops if o not in OPS]
    if bad:
        ap.error(f"unknown rows or ops {bad}")
    if args.fake_silicon and SHARED.RESEARCH_RUN_ID:
        ap.error("--fake-silicon is a dry run and never publishes: unset RESEARCH_RUN_ID")
    scale_update = not args.no_scale_update
    plan = variants(rows, scale_update)
    items = [("mma", r, m) for r, m in plan] + [("op", o, None) for o in ops]
    planned_s = args.sustain_s + len(items) * (args.reps + 3) * args.target_ms / 1e3 + args.reps * 0.2 * 2
    if planned_s > args.max_seconds:
        ap.error(f"the plan takes about {planned_s:.0f} s > --max-seconds {args.max_seconds:g}")
    t_all = time.time()

    have_nvcc = Path(P.nvcc_bin()).is_file()
    so, nvcc = (P.build(ARCH, out, SOURCE) if (have_nvcc or not args.fake_silicon) else (None, "not built (fake silicon)"))
    sass = sass_gate(so, rows, scale_update) if so is not None else {"ok": None, "kernels": {}, "note": "no nvcc"}
    ops_sass = ops_sass_gate(so, ops) if so is not None else {"ok": None, "kernels": {}, "note": "no nvcc"}
    dev = FakeW1() if args.fake_silicon else (None if args.sass_only else W1Device(so))
    identity = dev.identity() if dev is not None else {"device": None, "gpu_uuid": None, "backend": "none (--sass-only)"}
    identity["nvcc"] = nvcc
    record: dict = {"schema": SCHEMA, "identity": identity, "rows": rows, "ops": ops, "plan": [item_key(i) for i in items],
                    "sass": sass, "ops_sass": ops_sass, "fake_silicon": args.fake_silicon, "corrupt_word": args.corrupt_word,
                    "seed": args.seed}
    failures: list[str] = []
    if sass["ok"] is False:
        failures.append("SASS gate: " + ", ".join(k for k, v in sass["kernels"].items() if not v["ok"]))
    if ops_sass["ok"] is False:
        failures.append("SASS gate (op loops): " + ", ".join(k for k, v in ops_sass["kernels"].items() if not v["ok"]))
    if args.expect_uuid and dev is not None and not args.fake_silicon and not P.uuid_allowed(identity.get("gpu_uuid"), args.expect_uuid):
        failures.append(f"device UUID {identity.get('gpu_uuid')} != expected {args.expect_uuid}: stopped before any measurement")
    if dev is None or failures:
        return finish(out, record, failures, args, None, t_all)

    d = dev.device()
    record["device"] = d
    if d["cc"] != "12.0":
        failures.append(f"compute capability {d['cc']} != 12.0")
        return finish(out, record, failures, args, None, t_all)
    wpb = 4
    nblocks = d["sms"] * max(1, args.warps_per_sm // wpb)
    nwarps = nblocks * wpb
    grid = {"sms": d["sms"], "blocks": nblocks, "warps_per_block": wpb, "warps": nwarps, "warps_per_sm": nblocks * wpb // d["sms"],
            "operand_sets": NSETS, "accumulator_chains": NACC, "mmas_per_iteration": BODY, "latency_unroll": LAT_UNROLL}
    tpb = 256
    op_wps = args.op_warps_per_sm or d["max_threads_per_sm"] // 32
    op_blocks = d["sms"] * max(1, op_wps * 32 // tpb)
    grid["ops"] = {"threads_per_block": tpb, "blocks": op_blocks, "warps_per_sm": op_blocks * tpb // 32 // d["sms"],
                   "chains_per_thread": CV_CHAINS, "steps_per_chain_per_iteration": CV_UNROLL}
    record["grid"] = grid
    slot = {r: i for i, r in enumerate(rows)}
    frags = np.stack([fragments(r, nwarps, args.seed) for r in rows]) if rows else np.zeros((1, nwarps * 32, NSETS, WORDS), np.uint32)
    record["operands_sha256"] = hashlib.sha256(frags.tobytes()).hexdigest()
    dev.setup(frags, nwarps, max(1, len(rows)), nblocks)
    op_slot = {o: i for i, o in enumerate(ops)}
    if ops:
        init = np.stack([op_inputs(o, op_blocks * tpb, args.seed) for o in ops])
        record["op_inputs_sha256"] = hashlib.sha256(init.tobytes()).hexdigest()
        dev.cvt_setup(init, op_blocks * tpb, len(ops), op_blocks)

    def launch(item, iters, want=False):
        kind, name, mode = item
        if kind == "mma":
            return dev.run(ROWS[name][0], mode, slot[name], nblocks, wpb, iters, want=want)
        return dev.cvt_run(OPS[name][0], op_slot[name], op_blocks, tpb, iters, want=want)

    gate: dict = {}
    for item in items + [("lat", r, None) for r in rows]:
        if item[0] == "lat":
            s1, _ = dev.latency(ROWS[item[1]][0], slot[item[1]], 64)
            s2, _ = dev.latency(ROWS[item[1]][0], slot[item[1]], 64)
            name = kernel_name(item[1], None)
        else:
            _, s1, _ = launch(item, 64, want=True)
            _, s2, _ = launch(item, 64, want=True)
            name = kernel_name(item[1], item[2]) if item[0] == "mma" else op_kernel(item[1])
        if args.corrupt_word and not gate:
            s2 = s2.copy()
            s2.view(np.uint32)[0] ^= 1
        finite = bool(s1.dtype != np.float32 or (np.isfinite(s1).all() and np.isfinite(s2).all()))
        same = bool(np.array_equal(s1.view(np.uint32), s2.view(np.uint32)))
        gate[name] = {"finite": finite, "bit_identical": same, "ok": finite and same}
    record["output_gate"] = gate
    occ = {f"{r}/{MODES[m]}": (dev.occupancy(ROWS[r][0], m, wpb), nblocks // d["sms"]) for r, m in plan}
    occ.update({f"op/{o}": (dev.cvt_occupancy(OPS[o][0], tpb), op_blocks // d["sms"]) for o in ops})
    record["occupancy_blocks_per_sm"] = {k: v[0] for k, v in occ.items()}
    failures += [f"occupancy {k}: {v} blocks/SM < {need} (not one wave)" for k, (v, need) in occ.items() if v < need]
    failures += [f"output gate {k}: finite={v['finite']} bit_identical={v['bit_identical']}" for k, v in gate.items() if not v["ok"]]
    if failures:
        dev.teardown()
        return finish(out, record, failures, args, None, t_all)

    iters = {}
    for item in items:
        ms, _, _ = launch(item, 2000)
        iters[item] = max(2000, int(2000 * args.target_ms / max(ms, 1e-3)))
    t0 = time.time()
    while time.time() - t0 < args.sustain_s:
        launch(items[0], iters[items[0]])
    launches: dict = {item_key(i): [] for i in items}
    samples = []
    for rep in range(args.reps):
        for item in items[rep % len(items):] + items[:rep % len(items)]:
            ms, _, cyc = launch(item, iters[item])
            # the longest block's span is the launch's: blocks sharing an SM finish at different times, so a median span
            # undercounts the clock (by 1.3x to 1.7x on node 2)
            launches[item_key(item)].append({"rep": rep, "ms": ms, "cycles_median": float(np.median(cyc)), "cycles_max": int(cyc.max()),
                                             "sm_mhz": float(cyc.max()) / (ms * 1e3)})
        samples.append(dev.clocks(identity.get("gpu_uuid")))
    latency = {}
    for row in rows:
        per = [dev.latency(ROWS[row][0], slot[row], args.latency_iters)[1] / (args.latency_iters * LAT_UNROLL) for _ in range(5)]
        latency[row] = {"cycles_per_mma": per, "cycles_per_mma_median": _median(per)}
    dev.teardown()

    per_variant, per_op = {}, {}
    for item in items:
        key = item_key(item)
        ls = launches[key]
        ms_med = _median([x["ms"] for x in ls])
        mhz_med = _median([x["sm_mhz"] for x in ls])
        timing = {"iterations": iters[item], "ms_median": ms_med, "ms_all": [x["ms"] for x in ls], "sm_mhz_median": mhz_med,
                  "sm_mhz_all": [x["sm_mhz"] for x in ls]}
        if item[0] == "mma":
            row, mode = item[1], item[2]
            mma_s = float(iters[item]) * BODY * nwarps / (ms_med / 1e3)
            per_variant[key] = {"row": row, "mode": MODES[mode], "instruction": PTX[row], "sass_opcode": ROWS[row][5], "k": ROWS[row][1],
                                **timing, "mma_per_second": mma_s, "flop_per_second": mma_s * flop_per_mma(row),
                                "mma_per_sm_per_clock": mma_s / (d["sms"] * mhz_med * 1e6),
                                "flop_per_sm_per_clock": mma_s * flop_per_mma(row) / (d["sms"] * mhz_med * 1e6)}
        else:
            op = item[1]
            op_s = float(iters[item]) * CV_CHAINS * CV_UNROLL * OPS[op][2] * op_blocks * tpb / (ms_med / 1e3)
            per_op[op] = {"op": op, "role": OPS[op][5], "sass_opcode": OPS[op][1], "per": OPS[op][4], "per_op": OPS[op][3], **timing,
                          "op_per_second": op_s, "op_per_sm_per_clock": op_s / (d["sms"] * mhz_med * 1e6),
                          f"{OPS[op][4]}_per_sm_per_clock": op_s * OPS[op][3] / (d["sms"] * mhz_med * 1e6),
                          "loop_opcodes": ops_sass["kernels"].get(op_kernel(op), {}).get("loop_opcodes")}
    comparisons = {}
    for name, (num, den) in COMPARISONS.items():
        a, b = (per_variant.get(f"{r}/{MODES[m]}") for r, m in (num, den))
        if a and b:
            comparisons[name] = {"flop_rate_ratio": a["flop_per_second"] / b["flop_per_second"],
                                 "per_clock_ratio": a["flop_per_sm_per_clock"] / b["flop_per_sm_per_clock"]}
    nv = per_variant.get("nvf4/const_scales")
    prices = ({key: nv["flop_per_sm_per_clock"] / v["flop_per_sm_per_clock"] for key, v in per_variant.items()} if nv else {})
    record.update(per_variant=per_variant, per_op=per_op, latency=latency, comparisons=comparisons, clock_samples=samples,
                  launches=launches, price_per_mac_in_dense_nvf4_units=prices, fp8_prices=fp8_prices(per_variant, per_op),
                  clocks=clock_label(samples, [x["sm_mhz"] for ls in launches.values() for x in ls], args.locked_mhz))
    return finish(out, record, failures, args, per_variant, t_all)


def item_key(item) -> str:
    return f"{item[1]}/{MODES[item[2]]}" if item[0] == "mma" else f"op/{item[1]}"


def fp8_prices(per_variant: dict, per_op: dict) -> dict:
    """Every price in the W1 unit, one dense FP8 E4M3 ``mma.sync`` MAC (``f8f6f4_e4m3``, constant operands) per SM per clock:
    a row's price is FP8 MACs per logical MAC (and per stored MAC for the sparse rows, half the logical K), an op's is FP8 MACs
    per element or per scale it handles, the op's chain glue included (``per_op[op]['loop_opcodes']``)."""
    ref = per_variant.get(f"{FP8_REF[0]}/{MODES[FP8_REF[1]]}")
    if not ref:
        return {}
    fp8_mac = ref["flop_per_sm_per_clock"] / 2
    rows = {}
    for key, v in per_variant.items():
        mac = v["flop_per_sm_per_clock"] / 2
        rows[key] = {"per_logical_mac": fp8_mac / mac}
        if v["row"] in SPARSE_ROWS:
            rows[key]["per_stored_mac"] = fp8_mac / (mac / 2)
    ops = {o: {f"per_{v['per']}": fp8_mac / v[f"{v['per']}_per_sm_per_clock"], "per_instruction": fp8_mac / v["op_per_sm_per_clock"]}
           for o, v in per_op.items()}
    out = {"unit": "one dense FP8 E4M3 mma.sync MAC (kind::f8f6f4 e4m3 x e4m3 m16n8k32, f32 accumulate)",
           "fp8_mac_per_sm_per_clock": fp8_mac, "rows": rows, "ops": ops}
    if all(o in per_op for o in NVF4_QUANTIZE_RECIPE):
        out["nvf4_quantize_per_element"] = {
            "value": sum(n * ops[o][f"per_{per_op[o]['per']}"] for o, n in NVF4_QUANTIZE_RECIPE.items()),
            "label": "Derived from the Measured op prices; the recipe (op counts per element) is Estimated",
            "recipe": NVF4_QUANTIZE_RECIPE}
    return out


def finish(out: Path, record: dict, failures: list[str], args, per_variant: dict | None, t_all: float) -> int:
    record["failures"] = failures
    record["elapsed_s"] = time.time() - t_all
    (out / "w1_price.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    ok = not failures and (per_variant is not None or args.sass_only)
    detail = ("; ".join(failures) if failures else
              ("SASS gate only (--sass-only)" if per_variant is None else
               f"{len(record['output_gate'])} kernels finite and bit-identical across two launches; SASS {len(record['sass']['kernels'])} "
               f"kernels each exactly the row's opcode x the body"))
    meas = []
    if per_variant and not failures:
        for key, v in per_variant.items():
            stem = key.replace("/", "_")
            meas += [{"name": f"{stem}_flop_per_second", "value": v["flop_per_second"], "unit": "flop/s"},
                     {"name": f"{stem}_flop_per_sm_per_clock", "value": v["flop_per_sm_per_clock"], "unit": "flop/(SM*cycle)"},
                     {"name": f"{stem}_sm_mhz_median", "value": v["sm_mhz_median"], "unit": "MHz"}]
        for row, v in record["latency"].items():
            meas.append({"name": f"{row}_latency_cycles_per_mma", "value": v["cycles_per_mma_median"], "unit": "cycles"})
        for name, v in record["comparisons"].items():
            meas.append({"name": name, "value": v["per_clock_ratio"], "unit": "ratio"})
        for op, v in record["per_op"].items():
            meas += [{"name": f"op_{op}_per_sm_per_clock", "value": v["op_per_sm_per_clock"], "unit": "op/(SM*cycle)"},
                     {"name": f"op_{op}_sm_mhz_median", "value": v["sm_mhz_median"], "unit": "MHz"}]
        fp8 = record["fp8_prices"]
        for key, v in fp8.get("rows", {}).items():
            meas += [{"name": f"{key.replace('/', '_')}_fp8_mac_{k}", "value": x, "unit": "fp8 MAC"} for k, x in v.items()]
        for op, v in fp8.get("ops", {}).items():
            meas += [{"name": f"op_{op}_fp8_mac_{k}", "value": x, "unit": "fp8 MAC"} for k, x in v.items()]
        if "nvf4_quantize_per_element" in fp8:
            meas.append({"name": "nvf4_quantize_fp8_mac_per_element", "value": fp8["nvf4_quantize_per_element"]["value"],
                         "unit": "fp8 MAC", "label": fp8["nvf4_quantize_per_element"]["label"]})
        meas.append({"name": "elapsed", "value": record["elapsed_s"], "unit": "s"})
    ident = record["identity"]
    doc = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": SHARED.RESEARCH_RUN_ID,
           "device": ident.get("device"), "gpu_uuid": ident.get("gpu_uuid"), "driver": ident.get("driver"), "nvcc": ident.get("nvcc"),
           "cuda_visible_devices": ident.get("cuda_visible_devices"), "clocks": record.get("clocks", "unlocked"),
           "label": "Measured" + ("" if record.get("clocks", "unlocked") != "unlocked" else ", unlocked clocks"),
           "method": (f"instruction microbenchmark: each row's mma.sync issued back to back from {record.get('grid', {}).get('warps')} "
                      f"warps, {NACC} accumulator chains x {NSETS} register-resident operand sets, median of {args.reps} interleaved "
                      f"event-timed launches after {args.sustain_s:g} s sustained; SM clock per launch from clock64; latency from one "
                      f"warp's dependent chain; the conversion and scale ops as {CV_CHAINS} chains per thread at "
                      f"{record.get('grid', {}).get('ops', {}).get('warps_per_sm')} warps per SM, interleaved with the rows"),
           "grid": record.get("grid"), "comparisons": record.get("comparisons"),
           "price_per_mac_in_dense_nvf4_units": record.get("price_per_mac_in_dense_nvf4_units"),
           "fp8_prices": record.get("fp8_prices"),
           "workload_fingerprint": {"tool": "tc_price_fp4", "version": "2", "rows": record["rows"], "ops": record.get("ops"),
                                    "plan": record["plan"], "reps": args.reps, "target_ms": args.target_ms, "sustain_s": args.sustain_s,
                                    "warps_per_sm": args.warps_per_sm, "op_warps_per_sm": args.op_warps_per_sm,
                                    "latency_iters": args.latency_iters, "seed": args.seed,
                                    "operands_sha256": record.get("operands_sha256"), "op_inputs_sha256": record.get("op_inputs_sha256"),
                                    "commit": SHARED.RESEARCH_SOURCE_COMMIT},
           "validation": {"status": "passed" if ok else "failed",
                          "checker": "device UUID + cc 12.0; cuobjdump opcode counts per kernel and per op loop; outputs finite and "
                                     "bit-identical across two launches (no arithmetic check: probe.py's)", "detail": detail,
                          "evidence": ["w1_price.json"]},
           "measurements": meas, "measurement_files": ["w1_price.json"], "artifacts": []}
    if not args.fake_silicon:
        (out / "result.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    else:
        (out / "result.fake.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    if failures:
        print("FAILURES", failures, flush=True)
        return 2 if any("UUID" in f for f in failures) else 1
    print("W1 PRICE", "SASS ONLY" if per_variant is None else "RECORDED", flush=True)
    fp8 = record.get("fp8_prices", {})
    for k, v in (per_variant or {}).items():
        p = fp8.get("rows", {}).get(k, {}).get("per_logical_mac")
        print(f"  {k:40s} {v['flop_per_second'] / 1e12:9.1f} TFLOP/s  {v['flop_per_sm_per_clock']:8.1f} flop/SM/clk  {v['sm_mhz_median']:7.1f} MHz"
              + (f"  {p:7.4f} fp8 MAC/MAC" if p is not None else ""))
    for op, v in record.get("per_op", {}).items():
        p = fp8.get("ops", {}).get(op, {}).get(f"per_{v['per']}")
        print(f"  op/{op:37s} {v['op_per_sm_per_clock']:8.2f} op/SM/clk  {v['sm_mhz_median']:7.1f} MHz"
              + (f"  {p:7.3f} fp8 MAC/{v['per']}" if p is not None else ""))
    if "nvf4_quantize_per_element" in fp8:
        print(f"  nvf4 quantize (Derived)                  {fp8['nvf4_quantize_per_element']['value']:7.3f} fp8 MAC/element")
    return 0


if __name__ == "__main__":
    sys.exit(main())
