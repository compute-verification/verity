"""verity/ml/definitions/numerics/twins.py — vectorized twins of the registered numerical bodies, bit-exact against the gate evaluator.

A *twin* re-implements one registered composite (`verity/ml/definitions/registry/b1.py`) for speed — numpy over whole operands,
or the numerics lane's C++ models through ctypes — and is admissible only where its output words are IDENTICAL to
`verity.ml.circuits.evaluate.evaluate_call` on the registered body.  A twin never redefines a body: where it cannot be exact it
declines the input (`Declined`, the gate path is the fallback) or does not claim the specialisation (`covers`).  Twins
are core `verity.ml.circuits.evaluation` kernels (`KERNEL`) of their Definitions, checked against the reference by core `self_check`:

    covers(spec, KERNEL)                   # None when a twin claims spec (e.g. `Gemm_v1{K=576,N=960}`), else why not
    words = instance(spec, KERNEL)(*args, **spec.statics)
                                           # args: one unsigned-integer array per signature argument, in signature
                                           # order, holding the leaf encodings exactly as `Replayer.operand_words`
                                           # lists them (uint64; uint16/uint32 straight from a snapshot/checkpoint
                                           # carry the same values); returns the output encodings as uint64.
    uv run verity-vllm twins --check --log LOG --snapshots DIR --profile P --seed S --out twins.json   (`program.twins_check`)

Twins (twin_id -> registered body it reproduces; see `out/capture/reports/h4_twins.md` for the reduction orders):
    gemm.bf16.tc_chain.v1        Gemm_v1              tc_model.cpp `gemm_model` (one ascending k16 chain per coordinate,
                                                      zero accumulator, RNE to bf16); coordinates with a non-finite
                                                      operand or result re-evaluated with the registered primitives
    bias_add.bf16.v1             BiasAdd_v1           numpy f32 add, RNE (NaN 0x7FC0)
    embedding.gather.v1          Embedding_v1         numpy row gather (out-of-range token -> 0x7FFF row)
    token_select.argmax.v1       TokenSelect_v1       numpy strict-greater scan semantics (first index wins ties, NaN never wins)
    silu_mul.bf16.table.v1       SiluMul_v1           65,536-entry table of the registered `bf16(silu_f32(g))`, numpy f32 mul, RNE
    rope.neox.bf16.v1            RoPE_v1              numpy: t = RN32(y*s) in f32, fma(x, c, -t) via an exact float64 intermediate
    rmsnorm.triton.v1            RMSNormTriton_v1     rms_triton_model.cpp `rms_triton_rows` (variant 0)
    rmsnorm.fused_cuda.v2.v1     RMSNormFusedCuda_v2  numpy: residual add, per-thread fma chain (float64 intermediate), cub
                                                      BlockReduce tree, IEEE div/add, MUFU.RSQ (C++), two f32 products, RNE
    attention.fa2.bf16.v3        Attention_v3         fa2_model.cpp `fa2_rows` (online softmax, blocks last -> first); declines
                                                      non-finite operands and rows with a non-finite intermediate (flags)

Equivalence evidence is machine-checked by `--check` against the gate evaluator and written as `verity-gen/twins/v1` JSON.
"""
from __future__ import annotations

import ctypes
import sys
import time
import zlib
from collections import Counter
from functools import partial
from typing import Any, Callable

import numpy as np

from verity.ml.circuits.evaluation import self_check
from verity.ml.circuits.defs import SpecializedDefinition, bind
from verity.ml.definitions.numerics.kernel_registry import Declined, covers, instance, register
from verity.ml.definitions.registry import b1, prims as P
from verity.ml.definitions.numerics import cuda_rmsnorm_rows as CNR, prims_twins

SCHEMA = "verity-gen/twins/v1"
U16, U32, U64, F32, F64 = np.uint16, np.uint32, np.uint64, np.float32, np.float64

# word helpers ------------------------------------------------------------------------------------------------------------
def _words16(a) -> np.ndarray:
    """bf16 leaf encodings as a contiguous uint16 array (the Replayer hands uint16/uint32/uint64 of the same values)."""
    a = np.asarray(a)
    if a.dtype != U16:
        a = a.astype(U16)
    return np.ascontiguousarray(a)


def _f32(w16: np.ndarray) -> np.ndarray:
    """`Bf16ToF32`: the bf16 word in the high half of an f32."""
    return (np.asarray(w16, dtype=U32) << 16).view(F32)


def _bf16_rn(x: np.ndarray, nan: int) -> np.ndarray:
    """`f32_to_bf16_rn_bits` over an f32 array: RNE to the high half, every NaN -> `nan` (0x7FFF F2FP / 0x7FC0 torch)."""
    u = np.ascontiguousarray(x, dtype=F32).view(U32).astype(U64)
    lsb = (u >> 16) & 1
    r = (((u + 0x7FFF + lsb) >> 16) & 0xFFFF).astype(U16)
    r[((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0)] = nan
    return r


def _nonfinite16(w16: np.ndarray) -> np.ndarray:
    return (w16 & 0x7F80) == 0x7F80


def _any_nonfinite16(w16: np.ndarray, chunk: int = 1 << 20) -> bool:
    """`_nonfinite16(w16).any()` in bounded slices (a 28M-word lm_head operand would otherwise cost ~85 MB of temporaries
    per instance; the laptop budget is 1.5 GB)."""
    w = w16.reshape(-1)
    for i in range(0, w.size, chunk):
        if _nonfinite16(w[i:i + chunk]).any():
            return True
    return False


def _i32(word: int) -> int:
    word &= 0xFFFFFFFF
    return word - (1 << 32) if word & 0x80000000 else word


# ---------------------------------------------------------------------------------------------------------
# the numerics lane's C++ models (built on demand, OpenMP when the compiler has it, exact without it)
# ---------------------------------------------------------------------------------------------------------

_TC_LIB: ctypes.CDLL | None = None
LIBRARIES: dict[str, Any] = {}


def _tc_lib() -> ctypes.CDLL:
    """`tc_model.cpp` (`gemm_model`), built by `cpu_model.build` (`_jit.cxx_compile`: OpenMP when the compiler has it; the model
    is integer arithmetic on exact terms, so exact either way, only slower without)."""
    global _TC_LIB
    if _TC_LIB is None:
        from verity.ml.definitions.numerics import cpu_model
        _TC_LIB = cpu_model.lib()
        LIBRARIES["tc_model"] = str(cpu_model.LIB)
    return _TC_LIB


def fa2_twin():
    fa2, tabs = prims_twins.fa2_twin()
    LIBRARIES["fa2_model"] = str(fa2.LIB)
    return fa2, tabs


def rms_twin():
    rm, tabs = prims_twins.rms_twin()
    LIBRARIES["rms_triton_model"] = str(rm.LIB)
    return rm, tabs


def preload() -> dict[str, Any]:
    """Build/load the three C++ models and the MUFU tables now.  The tables are reconstructed from int8 deltas with
    int64 cumsum temporaries (~0.25 GB transient over the 0.22 GB they occupy, shared with the gate primitives); doing
    this before the fold's Program and the checkpoint fields are resident keeps the process high-water mark lower and
    lets the freed transient space be reused (laptop: 1.4 GB -> ~1.1 GB on M1 tier b)."""
    _tc_lib()
    fa2_twin()
    rms_twin()
    silu_table()
    return dict(LIBRARIES)


# ---------------------------------------------------------------------------------------------------------
# random operands (for the checks)
# ---------------------------------------------------------------------------------------------------------

# ±0, ±inf, NaNs (canonical F2FP 0x7FFF, torch 0x7FC0, payloads), min/max subnormals, min normal, max finite, ±1
SPECIAL_WORDS = np.array([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FFF, 0x7FC0, 0xFFC1, 0x7F81, 0x0001, 0x8001, 0x007F, 0x807F,
                          0x0080, 0x8080, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80, 0x0100, 0x4000], dtype=U16)
RANDOM_KINDS = ("gauss", "wide", "gauss", "mixed")
EDGE_KINDS = ("special", "extreme", "subnormal", "boundary")


def rand_bf16(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    """Random bf16 words of `kind`: gauss (N(0, scale), scale in {0.02, 1, 40}); wide (exponents 2^-24..2^24, random
    mantissa); mixed (gauss with a sprinkle of wide); special (gauss with 8% of SPECIAL_WORDS); extreme (every exponent
    incl. 0 and 255: zeros, subnormals, infs, NaNs); subnormal (subnormals and the smallest normals); boundary (wide)."""
    if kind == "gauss":
        scale = float(rng.choice([0.02, 1.0, 40.0]))
        return _bf16_rn((rng.standard_normal(n) * scale).astype(F32), P.BF16_NAN_TORCH)
    if kind == "spread":   # gauss with per-element binades 2^-6 .. 2^8 (moderate dynamic range, no attention overflow)
        return _bf16_rn((rng.standard_normal(n) * np.exp2(rng.integers(-6, 9, n))).astype(F32), P.BF16_NAN_TORCH)
    if kind in ("wide", "boundary"):
        e = rng.integers(127 - 24, 127 + 25, n, dtype=np.int64)
    elif kind == "extreme":
        e = rng.integers(0, 256, n, dtype=np.int64)
    elif kind == "subnormal":
        e = rng.integers(0, 3, n, dtype=np.int64)
    elif kind == "mixed":
        w = rand_bf16(rng, n, "gauss")
        m = rng.random(n) < 0.15
        w[m] = rand_bf16(rng, int(m.sum()), "wide")
        return w
    elif kind == "special":
        w = rand_bf16(rng, n, "gauss")
        m = rng.random(n) < 0.08
        w[m] = rng.choice(SPECIAL_WORDS, int(m.sum()))
        return w
    else:
        raise ValueError(kind)
    m = rng.integers(0, 128, n, dtype=np.int64)
    s = rng.integers(0, 2, n, dtype=np.int64)
    return ((s << 15) | (e << 7) | m).astype(U16)


def tie_pairs_add(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(a, b) with b = ±2^(e_a - 8): the f32 sum a + b has exactly bit 15 set below the bf16 half -> an RNE tie."""
    a = rand_bf16(rng, n, "gauss")
    ea = ((a >> 7) & 0xFF).astype(np.int64)
    small = ea < 9
    a[small] = (a[small] & 0x807F) | (np.int64(0x7F) << 7)   # lift tiny exponents so e_a - 8 stays normal
    ea = ((a >> 7) & 0xFF).astype(np.int64)
    b = ((rng.integers(0, 2, n, dtype=np.int64) << 15) | ((ea - 8) << 7)).astype(U16)
    return a, b


def tie_pairs_mul(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(a, b) whose exact f32 product has the bf16 tie pattern (low 16 bits 0x8000), by rejection over `wide` words."""
    a = np.empty(n, U16)
    b = np.empty(n, U16)
    filled = 0
    for _ in range(400):
        ca, cb = rand_bf16(rng, 4096, "wide"), rand_bf16(rng, 4096, "wide")
        prod = (_f32(ca) * _f32(cb)).view(U32)
        ok = np.nonzero((prod & 0xFFFF) == 0x8000)[0]
        take = ok[:n - filled]
        a[filled:filled + take.size], b[filled:filled + take.size] = ca[take], cb[take]
        filled += take.size
        if filled >= n:
            break
    if filled < n:   # fall back to plain words for the remainder (should not happen)
        a[filled:], b[filled:] = rand_bf16(rng, n - filled, "wide"), rand_bf16(rng, n - filled, "wide")
    return a, b


# ---------------------------------------------------------------------------------------------------------
# twins
# ---------------------------------------------------------------------------------------------------------


class Twin:
    """One specialization's vectorized evaluator.  `__call__(args)` takes one unsigned-integer array per signature
    argument (leaf encodings in signature order) and returns the output encodings (uint64), or raises `Declined`."""

    twin_id: str = ""
    base_id: str = ""
    covers: str = ""

    def __init__(self, spec: SpecializedDefinition) -> None:
        if spec.base_id != self.base_id:
            raise TypeError(f"{self.twin_id} covers {self.base_id}, not {spec.id}")
        why = self.accepts(spec.statics)
        if why:
            raise Declined(f"{self.twin_id} does not cover {spec.id}: {why}")
        self.spec = spec
        self.s = spec.S()

    @classmethod
    def accepts(cls, statics: dict) -> str | None:
        """None when the twin covers these statics, else the reason it does not."""
        return None

    def __call__(self, args: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    # -- check support ----------------------------------------------------------------------------------------

    @classmethod
    def check_statics(cls, statics: dict, rng: np.random.Generator) -> dict:
        """Statics for a gate-vs-twin random instance of a spec with `statics`: the spec's own when the gate evaluation
        is cheap, else shrunk along the free size while keeping the structure under test."""
        return dict(statics)

    @classmethod
    def small_statics(cls, rng: np.random.Generator) -> dict:
        """Independent small random statics (structural coverage of the body: block boundaries, partial warps, ...)."""
        raise NotImplementedError

    @classmethod
    def random_args(cls, statics: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
        raise NotImplementedError


class GemmTwin(Twin):
    twin_id = "gemm.bf16.tc_chain.v1"
    base_id = "Gemm_v1"
    covers = ("Gemm_v1{K,N}: out[j] = bf16_rne(chain_j), chain_j = K/16 `AmpereBF16TcDot16` steps in ascending k from a zero f32 "
              "accumulator (mma.sync m16n8k16: groups (8,8), 25-bit adder, exponent floor -132, RNE to f32, sticky inf) -- "
              "tc_model.cpp `gemm_model` (M=1, nsplit=1, no bias, NaN word 0x7FFF).  A coordinate with a non-finite operand "
              "word or a non-finite result is re-evaluated with the registered primitives (the model has no inf/NaN extension).")

    @classmethod
    def accepts(cls, st: dict) -> str | None:
        K, N = int(st["K"]), int(st["N"])
        if K < 16 or K % 16:
            return f"K={K} is not a positive multiple of 16"
        if N < 1:
            return f"N={N}"
        return None

    def __call__(self, args: list[np.ndarray]) -> np.ndarray:
        K, N = int(self.s.K), int(self.s.N)
        x, w = _words16(args[0]), _words16(args[1])
        if x.size != K or w.size != N * K:
            raise ValueError(f"{self.spec.id}: operands {x.size}, {w.size} words")
        out = np.empty(N, dtype=U16)
        groups = (ctypes.c_int * 2)(8, 8)
        _tc_lib().gemm_model(0, x.ctypes.data, w.ctypes.data, None, None, 1, N, K, groups, 2, 25, -132,
                             1, None, 0, None, 0, None, out.ctypes.data, P.BF16_NAN_F2FP)
        w2 = w.reshape(N, K)
        if _nonfinite16(x).any():
            bad = np.arange(N)
        elif _any_nonfinite16(w2):
            bad = np.nonzero(_nonfinite16(w2).any(axis=1) | _nonfinite16(out))[0]
        else:
            bad = np.nonzero(_nonfinite16(out))[0]
        for j in bad:
            out[j] = _gemm_coordinate_gates(x, w2[j])
        return out.astype(U64)

    @classmethod
    def check_statics(cls, st: dict, rng: np.random.Generator) -> dict:
        return {"K": int(st["K"]), "N": int(rng.integers(1, 4))}   # the real chain length, a few coordinates

    @classmethod
    def small_statics(cls, rng: np.random.Generator) -> dict:
        return {"K": 16 * int(rng.integers(1, 7)), "N": int(rng.integers(1, 9))}

    @classmethod
    def random_args(cls, st: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
        K, N = int(st["K"]), int(st["N"])
        return [rand_bf16(rng, K, kind).astype(U64), rand_bf16(rng, N * K, kind).astype(U64)]


def _gemm_coordinate_gates(x16: np.ndarray, w16: np.ndarray) -> int:
    """`GemmCoordinate_v1` with the registered primitives themselves (the inf/NaN extension path)."""
    xl, wl = x16.tolist(), w16.tolist()
    acc = 0
    for s in range(len(xl) // 16):
        acc = P.AmpereBF16TcDot16.evaluate(acc, *xl[16 * s:16 * s + 16], *wl[16 * s:16 * s + 16])
    return P.F2fpBf16.evaluate(acc)


class BiasAddTwin(Twin):
    twin_id = "bias_add.bf16.v1"
    base_id = "BiasAdd_v1"
    covers = "BiasAdd_v1{N}: out[i] = bf16_rne(f32(y[i]) + f32(b[i])) (aten.add, one IEEE f32 add, NaN -> 0x7FC0), numpy f32."

    def __call__(self, args: list[np.ndarray]) -> np.ndarray:
        N = int(self.s.N)
        y, b = _words16(args[0]), _words16(args[1])
        if y.size != N or b.size != N:
            raise ValueError(f"{self.spec.id}: operands {y.size}, {b.size} words")
        return _bf16_rn(_f32(y) + _f32(b), P.BF16_NAN_TORCH).astype(U64)

    @classmethod
    def small_statics(cls, rng: np.random.Generator) -> dict:
        return {"N": int(rng.integers(1, 300))}

    @classmethod
    def random_args(cls, st: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
        N = int(st["N"])
        if kind == "boundary":
            a, b = tie_pairs_add(rng, N)
            return [a.astype(U64), b.astype(U64)]
        return [rand_bf16(rng, N, kind).astype(U64), rand_bf16(rng, N, kind).astype(U64)]


class EmbeddingTwin(Twin):
    twin_id = "embedding.gather.v1"
    base_id = "Embedding_v1"
    covers = "Embedding_v1{V,H}: out = table[tok] (row gather); signed i32 token outside [0, V) -> a row of 0x7FFF (GatherBf16 extension)."

    def __call__(self, args: list[np.ndarray]) -> np.ndarray:
        V, H = int(self.s.V), int(self.s.H)
        tok = np.asarray(args[0]).ravel()
        table = np.asarray(args[1]).ravel()
        if tok.size != 1 or table.size != V * H:
            raise ValueError(f"{self.spec.id}: operands {tok.size}, {table.size} words")
        t = _i32(int(tok[0]))
        if 0 <= t < V:
            return table[t * H:(t + 1) * H].astype(U64)
        return np.full(H, P.BF16_NAN_F2FP, dtype=U64)

    @classmethod
    def check_statics(cls, st: dict, rng: np.random.Generator) -> dict:
        return cls.small_statics(rng)   # the real table (V*H leaves) is too wide for a standalone gate program

    @classmethod
    def small_statics(cls, rng: np.random.Generator) -> dict:
        return {"V": int(rng.integers(1, 40)), "H": int(rng.integers(1, 24))}

    @classmethod
    def random_args(cls, st: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
        V, H = int(st["V"]), int(st["H"])
        if kind in EDGE_KINDS:
            tok = int(rng.choice([-1, V, V + 7, 0x7FFFFFFF, -(1 << 31), 0, V - 1]))
        else:
            tok = int(rng.integers(0, V))
        return [np.array([tok & 0xFFFFFFFF], dtype=U64), rand_bf16(rng, V * H, kind).astype(U64)]


class TokenSelectTwin(Twin):
    twin_id = "token_select.argmax.v1"
    base_id = "TokenSelect_v1"
    covers = ("TokenSelect_v1{V}: the ArgmaxStep scan from (logits[0], 0): index of the first strictly-greatest bf16 logit "
              "(x > best on f32 values: ties keep the earlier index, NaN never wins, a NaN logits[0] is never beaten -> 0).")

    def __call__(self, args: list[np.ndarray]) -> np.ndarray:
        V = int(self.s.V)
        logits = _words16(args[0])
        if logits.size != V:
            raise ValueError(f"{self.spec.id}: {logits.size} logits")
        v = _f32(logits)
        if np.isnan(v[0]):
            return np.zeros(1, dtype=U64)
        finite = ~np.isnan(v)
        best = np.max(v[finite])
        idx = int(np.nonzero(finite & (v == best))[0][0])   # -0.0 == +0.0: the earlier index wins, as `>` never replaces it
        return np.array([idx & 0xFFFFFFFF], dtype=U64)

    @classmethod
    def check_statics(cls, st: dict, rng: np.random.Generator) -> dict:
        return {"V": int(rng.integers(2, 2049))}

    @classmethod
    def small_statics(cls, rng: np.random.Generator) -> dict:
        return {"V": int(rng.integers(2, 200))}

    @classmethod
    def random_args(cls, st: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
        V = int(st["V"])
        w = rand_bf16(rng, V, kind)
        if kind in ("gauss", "mixed") and V > 1:   # force ties: duplicate the maximum somewhere later
            i = int(np.argmax(_f32(w)))
            j = int(rng.integers(0, V))
            w[max(i, j)] = w[min(i, j)]
        if kind == "special" and rng.random() < 0.3:
            w[0] = P.BF16_NAN_F2FP   # a NaN first logit is never beaten
        return [w.astype(U64)]


_SILU_TABLE: np.ndarray | None = None


def silu_table() -> np.ndarray:
    """bf16 word -> `bf16_rn(silu_f32(g))` for every bf16 word, computed once with the registered primitive's own code."""
    global _SILU_TABLE
    if _SILU_TABLE is None:
        t = np.empty(65536, dtype=U16)
        for g in range(65536):
            t[g] = P.f32_to_bf16_rn_bits(P._silu_f32_bits(g << 16), P.BF16_NAN_TORCH)
        _SILU_TABLE = t
    return _SILU_TABLE


class SiluMulTwin(Twin):
    twin_id = "silu_mul.bf16.table.v1"
    base_id = "SiluMul_v1"
    covers = ("SiluMul_v1{I}: out[i] = bf16_rne(f32(s) * f32(u[i])), s = bf16_rne(silu_f32(g[i])) looked up in a 65,536-entry "
              "table built from the registered `_silu_f32_bits` + RNE (NaN 0x7FC0); the product is one numpy IEEE f32 multiply.")

    def __call__(self, args: list[np.ndarray]) -> np.ndarray:
        I = int(self.s.I)
        gu = _words16(args[0])
        if gu.size != 2 * I:
            raise ValueError(f"{self.spec.id}: {gu.size} words")
        s = silu_table()[gu[:I]]
        return _bf16_rn(_f32(s) * _f32(gu[I:]), P.BF16_NAN_TORCH).astype(U64)

    @classmethod
    def small_statics(cls, rng: np.random.Generator) -> dict:
        return {"I": int(rng.integers(1, 200))}

    @classmethod
    def random_args(cls, st: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
        I = int(st["I"])
        if kind == "boundary":
            g = rand_bf16(rng, I, "gauss")
            s = silu_table()[g]
            u = np.empty(I, U16)
            for i in range(I):   # a u whose product with s ties at the bf16 half (rejection), else a wide word
                sf = _f32(np.array([s[i]], U16))
                cand = rand_bf16(rng, 2048, "wide")
                ok = np.nonzero(((sf * _f32(cand)).view(U32) & 0xFFFF) == 0x8000)[0]
                u[i] = cand[ok[0]] if ok.size else cand[0]
            return [np.concatenate([g, u]).astype(U64)]
        return [rand_bf16(rng, 2 * I, kind).astype(U64)]


class RoPETwin(Twin):
    twin_id = "rope.neox.bf16.v1"
    base_id = "RoPE_v1"
    covers = ("RoPE_v1{NHEADS,D}: per head, R = D/2: out[i] = bf16_rne(fma(x[i], cos[i], -RN32(x[i+R]*sin[i]))), "
              "out[i+R] = bf16_rne(fma(x[i+R], cos[i], RN32(x[i]*sin[i]))) (NaN 0x7FC0).  The f32 product t is a numpy f32 "
              "multiply; the fma is x*c (exact in float64: 8x8-bit significands) ± t rounded once to f32 -- exact because a "
              "16-bit and a 24-bit addend either fit float64 or the smaller one lies below the f32 rounding boundary.")

    @classmethod
    def accepts(cls, st: dict) -> str | None:
        D, NH = int(st["D"]), int(st["NHEADS"])
        if D < 2 or D % 2:
            return f"D={D} is not a positive even number"
        if NH < 1:
            return f"NHEADS={NH}"
        return None

    def __call__(self, args: list[np.ndarray]) -> np.ndarray:
        NH, D = int(self.s.NHEADS), int(self.s.D)
        R = D // 2
        x, cs = _words16(args[0]), _words16(args[1])
        if x.size != NH * D or cs.size != D:
            raise ValueError(f"{self.spec.id}: operands {x.size}, {cs.size} words")
        x = _f32(x).reshape(NH, D)
        x1, x2 = x[:, :R], x[:, R:]
        c, s = _f32(cs[:R]), _f32(cs[R:])
        t_lo = (x2 * s).astype(F32)                       # RN32(y * s)
        t_hi = (x1 * s).astype(F32)                       # RN32(x * s)
        o_lo = (x1.astype(F64) * c.astype(F64) - t_lo.astype(F64)).astype(F32)   # fma(x, c, -t)
        o_hi = (x2.astype(F64) * c.astype(F64) + t_hi.astype(F64)).astype(F32)   # fma(y, c, t)
        out = np.concatenate([_bf16_rn(o_lo, P.BF16_NAN_TORCH), _bf16_rn(o_hi, P.BF16_NAN_TORCH)], axis=1)
        return out.reshape(-1).astype(U64)

    @classmethod
    def small_statics(cls, rng: np.random.Generator) -> dict:
        return {"NHEADS": int(rng.integers(1, 5)), "D": 2 * int(rng.integers(1, 20))}

    @classmethod
    def random_args(cls, st: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
        NH, D = int(st["NHEADS"]), int(st["D"])
        if kind == "boundary":
            a, b = tie_pairs_mul(rng, NH * D)
            cs = tie_pairs_mul(rng, D)[1]
            return [a.astype(U64), cs.astype(U64)]
        if kind in ("gauss", "mixed", "special"):   # a cos/sin-like table in [-1, 1]
            cs = _bf16_rn(rng.uniform(-1, 1, D).astype(F32), P.BF16_NAN_TORCH)
            if kind == "special":
                m = rng.random(D) < 0.08
                cs[m] = rng.choice(SPECIAL_WORDS, int(m.sum()))
        else:
            cs = rand_bf16(rng, D, kind)
        return [rand_bf16(rng, NH * D, kind).astype(U64), cs.astype(U64)]


class RMSNormTritonTwin(Twin):
    twin_id = "rmsnorm.triton.v1"
    base_id = "RMSNormTriton_v1"
    covers = ("RMSNormTriton_v1{N,EPS}: rms_triton_model.cpp `rms_triton_rows` variant 0 -- chunks of 1024 columns, 128 threads x 8 "
              "columns (`acc = x1*x1; fma(x0,x0,acc); fma(x2..x7)`), warp butterfly xor 16,8,4,2,1, CTA `(w0+w2)+(w1+w3)`, chunks "
              "added from +0; `mean_eps = fma(rcp(N), sum, eps)` (div.full range scaling), MUFU.SQRT, `inv = div.full(1, rms)`, "
              "`out = bf16_rne((inv*x)*w)` -- the same MUFU tables the primitives use.")

    @classmethod
    def accepts(cls, st: dict) -> str | None:
        N = int(st["N"])
        if N < 8 or N % 8:
            return f"N={N} is not a positive multiple of 8 (the body reads 8 columns per thread)"
        return None

    def __call__(self, args: list[np.ndarray]) -> np.ndarray:
        N = int(self.s.N)
        x, w = _words16(args[0]), _words16(args[1])
        if x.size != N or w.size != N:
            raise ValueError(f"{self.spec.id}: operands {x.size}, {w.size} words")
        rm, tabs = rms_twin()
        out = rm.rms_norm_rows(x.reshape(1, N), w, float(self.s.EPS), tabs, variant=0, kernel="triton")
        return np.asarray(out).reshape(-1).astype(U64)

    @classmethod
    def small_statics(cls, rng: np.random.Generator) -> dict:
        N = int(rng.choice([8, 16, 24, 64, 200, 1024, 1032, 1040, 2056]))
        return {"N": N if N % 8 == 0 else N - N % 8, "EPS": float(rng.choice([1e-5, 1e-6]))}

    @classmethod
    def random_args(cls, st: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
        N = int(st["N"])
        if kind == "boundary":
            a, b = tie_pairs_mul(rng, N)
            return [a.astype(U64), b.astype(U64)]
        return [rand_bf16(rng, N, kind).astype(U64), rand_bf16(rng, N, "gauss" if kind in RANDOM_KINDS else kind).astype(U64)]


class RMSNormFusedTwin(Twin):
    twin_id = "rmsnorm.fused_cuda.v2.v1"
    base_id = "RMSNormFusedCuda_v2"
    covers = ("RMSNormFusedCuda_v2{N,EPS}: r = bf16_rne(f32(x)+f32(res)) (NaN 0x7FFF); nthreads = min(N, 1024); thread t: "
              "acc = RN32(r[t]^2), then acc = fma(r[j], r[j], acc) for j = t+nthreads, t+2*nthreads, ... (float64 intermediate: "
              "the 16-bit square and the 24-bit accumulator either fit float64 exactly or the smaller lies below the f32 rounding "
              "boundary; no cancellation, all terms >= 0); cub BlockReduce: per warp shfl_down offsets 1,2,4,8,16 with the valid-lane "
              "predicate, warp aggregates added sequentially from warp 0; mean = var / N (IEEE), + eps, MUFU.RSQ via "
              "`rsqrt.approx.f32` range scaling (rms_triton_model.rsqrt_f32); out = bf16_rne((r*s)*w); returns (out, r).")

    @classmethod
    def accepts(cls, st: dict) -> str | None:
        return None if int(st["N"]) >= 1 else f"N={st['N']}"

    def __call__(self, args: list[np.ndarray]) -> np.ndarray:
        N = int(self.s.N)
        x, res, w = _words16(args[0]), _words16(args[1]), _words16(args[2])
        if x.size != N or res.size != N or w.size != N:
            raise ValueError(f"{self.spec.id}: operands {x.size}, {res.size}, {w.size} words")
        r = _bf16_rn(_f32(x) + _f32(res), P.BF16_NAN_F2FP)
        rf = _f32(r)
        var = _fused_variance(rf, N)
        rm, tabs = rms_twin()
        mean = (var / np.float32(N)).astype(F32)
        arg = (mean + np.float32(float(self.s.EPS))).astype(F32)
        sc = np.asarray(rm.rsqrt_f32(arg.reshape(1), tabs), dtype=F32)[0]
        y = ((rf * sc).astype(F32) * _f32(w)).astype(F32)
        return np.concatenate([_bf16_rn(y, P.BF16_NAN_F2FP), r]).astype(U64)

    @classmethod
    def small_statics(cls, rng: np.random.Generator) -> dict:
        N = int(rng.choice([1, 2, 3, 7, 8, 31, 32, 33, 63, 64, 65, 100, 1023, 1024, 1025, 1030, 1100, 2049]))
        return {"N": N, "EPS": float(rng.choice([1e-5, 1e-6]))}

    @classmethod
    def random_args(cls, st: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
        N = int(st["N"])
        if kind == "boundary":
            x, res = tie_pairs_add(rng, N)
            return [x.astype(U64), res.astype(U64), tie_pairs_mul(rng, N)[1].astype(U64)]
        wk = "gauss" if kind in RANDOM_KINDS else kind
        return [rand_bf16(rng, N, kind).astype(U64), rand_bf16(rng, N, kind).astype(U64), rand_bf16(rng, N, wk).astype(U64)]


def _fused_variance(rf: np.ndarray, N: int) -> np.float32:
    """Thread partials (`x*x`, then `fma(x, x, acc)` per further element of the thread) and the cub BlockReduce tree."""
    nthreads = min(N, b1.FUSED_RMS_BLOCK)
    acc = (rf[:nthreads] * rf[:nthreads]).astype(F32)          # `variance += x*x` from 0: RN32 of the exact square
    for j0 in range(nthreads, N, nthreads):
        z = rf[j0:j0 + nthreads].astype(F64)
        n = z.size
        acc[:n] = (z * z + acc[:n].astype(F64)).astype(F32)   # fma(z, z, acc), one rounding
    warps = []
    for w0 in range(0, nthreads, 32):
        lanes = {i: acc[w0 + i:w0 + i + 1] for i in range(min(32, nthreads - w0))}
        for off in (1, 2, 4, 8, 16):
            lanes = {i: ((v + lanes[i + off]).astype(F32) if i + off in lanes else v) for i, v in lanes.items() if i % (2 * off) == 0}
        warps.append(lanes[0])
    var = warps[0]
    for wv in warps[1:]:
        var = (var + wv).astype(F32)
    return np.float32(var[0])


class AttentionTwin(Twin):
    twin_id = "attention.fa2.bf16.v3"
    base_id = "Attention_v3"
    covers = ("Attention_v3{T,NH,KVH,D,BN}: fa2_model.cpp `fa2_rows` (variant 0, AMPERE_BF16 tensor-core step, inv_degenerate 1) per "
              "head with KV head h // (NH/KVH), col_limit T, ceil(T/BN) blocks visited last -> first: S = QK^T as D/16 ascending "
              "k16 chain steps from 0; row max; rescale ex2((m_prev - m)*scale_log2); P = ex2(fma(S, scale_log2, -m*scale_log2)); "
              "per-lane sums (lane (c%8)//2, ascending, the block's first add fused with the rescale as one FFMA.FTZ); P -> bf16 "
              "(0x7FFF); O = O*scale + P V in BN/16 ascending k16 steps; sum = (l0+l2)+(l1+l3); inv = (sum==0||NaN) ? 1 : MUFU.RCP; "
              "out = bf16_rne(O*inv).  Declines non-finite q/K/V words and rows whose S/P/rescale/O/sum is non-finite (flags): "
              "there the model saturates where the registered body follows its inf/NaN extension.")

    @classmethod
    def accepts(cls, st: dict) -> str | None:
        T, NH, KVH, D, BN = (int(st[k]) for k in ("T", "NH", "KVH", "D", "BN"))
        if D < 16 or D % 16:
            return f"D={D} is not a positive multiple of 16"
        if BN < 16 or BN % 16:
            return f"BN={BN} is not a positive multiple of 16"
        if NH < 1 or KVH < 1 or NH % KVH:
            return f"NH={NH}, KVH={KVH}"
        if T < 1:
            return f"T={T}"
        return None

    def __call__(self, args: list[np.ndarray]) -> np.ndarray:
        T, NH, KVH, D, BN = (int(getattr(self.s, k)) for k in ("T", "NH", "KVH", "D", "BN"))
        q, kc, vc = _words16(args[0]), _words16(args[1]), _words16(args[2])
        if q.size != NH * D or kc.size != T * KVH * D or vc.size != T * KVH * D:
            raise ValueError(f"{self.spec.id}: operands {q.size}, {kc.size}, {vc.size} words")
        if _nonfinite16(q).any() or _nonfinite16(kc).any() or _nonfinite16(vc).any():
            raise Declined("non-finite q/K/V words: the FA2 model saturates where the registered body follows its inf/NaN extension")
        fa2, tabs = fa2_twin()
        G = NH // KVH
        kvh = (np.arange(NH, dtype=np.int32) // G).astype(np.int32)
        col = np.full(NH, T, dtype=np.int32)
        nbm = np.full(NH, -(-T // BN), dtype=np.int32)
        out = np.empty(NH * D, dtype=U16)
        flags = np.zeros(NH, dtype=np.uint8)
        scale = float(np.array([b1.scale_log2_bits(D)], dtype=U32).view(F32)[0])
        fa2.lib().fa2_rows(q.ctypes.data, kc.ctypes.data, vc.ctypes.data, KVH, D, kvh.ctypes.data, col.ctypes.data, nbm.ctypes.data,
                           NH, BN, ctypes.c_float(scale), tabs.ex2.ctypes.data, tabs.rcp.ctypes.data, out.ctypes.data,
                           None, None, None, None, None, None, 0, 0, flags.ctypes.data)
        if flags.any():
            heads = np.nonzero(flags)[0].tolist()
            raise Declined(f"non-finite intermediate (S/P/rescale/O/sum) in head(s) {heads[:8]}: outside the FA2 model's measured domain")
        return out.astype(U64)

    @classmethod
    def check_statics(cls, st: dict, rng: np.random.Generator) -> dict:
        # the real geometry (T, D, BN, i.e. the block structure and chain lengths) with few heads: the gate cost is per head
        return {"T": int(st["T"]), "NH": int(rng.integers(1, 3)), "KVH": 1, "D": int(st["D"]), "BN": int(st["BN"])}

    @classmethod
    def small_statics(cls, rng: np.random.Generator) -> dict:
        BN = 16 * int(rng.integers(1, 4))
        KVH = int(rng.integers(1, 3))
        return {"T": int(rng.integers(1, 3 * BN + 3)), "NH": KVH * int(rng.integers(1, 3)), "KVH": KVH,
                "D": 16 * int(rng.integers(1, 3)), "BN": BN}

    @classmethod
    def random_args(cls, st: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
        T, NH, KVH, D = (int(st[k]) for k in ("T", "NH", "KVH", "D"))
        # scores beyond ~2^20 make `fma(S, scale, -m*scale)` (the rounding error of m*scale, exact in the fma) large enough
        # for ex2 to overflow: the kernel's -- and the body's -- own behaviour, which the model flags and the twin declines;
        # the random kinds stay in a moderate range so that the check exercises the twin, the edge kinds document the domain
        k = {"wide": "spread", "mixed": "spread", "boundary": "spread"}.get(kind, kind)
        return [rand_bf16(rng, NH * D, k).astype(U64), rand_bf16(rng, T * KVH * D, k).astype(U64),
                rand_bf16(rng, T * KVH * D, k).astype(U64)]
FAMILIES: dict[str, type[Twin]] = {cls.base_id: cls for cls in (GemmTwin, BiasAddTwin, EmbeddingTwin, TokenSelectTwin, SiluMulTwin,
                                                                RoPETwin, RMSNormTritonTwin, RMSNormFusedTwin, AttentionTwin)}
FAMILIES.update(CNR.twin_families(Twin, rand_bf16, RANDOM_KINDS))
RMSNormCudaVector8Twin, RMSNormFusedCudaVector8Twin = (
    FAMILIES[k] for k in CNR.TWIN_DEFINITIONS)
DEFINITIONS = {"Gemm_v1": b1.Gemm, "BiasAdd_v1": b1.BiasAdd, "Embedding_v1": b1.Embedding, "TokenSelect_v1": b1.TokenSelect,
               "SiluMul_v1": b1.SiluMul, "RoPE_v1": b1.RoPE, "RMSNormTriton_v1": b1.RMSNormTriton,
               "RMSNormFusedCuda_v2": b1.RMSNormFusedCuda, "Attention_v3": b1.AttentionV3}
DEFINITIONS.update(CNR.TWIN_DEFINITIONS)
# ---------------------------------------------------------------------------------------------------------
# the twins as core `verity.ml.circuits.evaluation` kernels, checked against the reference by core `self_check`
# ---------------------------------------------------------------------------------------------------------

KERNEL = "twin"


def _instance(cls: type[Twin]) -> Callable[..., np.ndarray]:
    def fn(*args: np.ndarray, **statics: Any) -> np.ndarray:
        return cls(bind(DEFINITIONS[cls.base_id], **statics))([np.asarray(a).reshape(-1) for a in args])
    return fn


def _sampler(cls: type[Twin]) -> Callable[..., list[np.ndarray]]:
    """Core's sampler form of `random_args`: one (n, leaves) array per parameter, the random kinds in turn."""
    def sample(rng: np.random.Generator, n: int, params: Any, **statics: Any) -> list[np.ndarray]:
        rows = [cls.random_args(statics, rng, RANDOM_KINDS[i % len(RANDOM_KINDS)]) for i in range(n)]
        return [np.stack([np.asarray(r[j]).reshape(-1) for r in rows]) for j in range(len(params))]
    return sample


def check_targets(cls: type[Twin], rng: np.random.Generator, k: int = 3) -> list[SpecializedDefinition]:
    """`k` small specialisations of the family (`small_statics`: block boundaries, partial warps, ...)."""
    return [bind(DEFINITIONS[cls.base_id], **cls.small_statics(rng)) for _ in range(k)]


for _cls in FAMILIES.values():
    register(DEFINITIONS[_cls.base_id], _instance(_cls), kernel=KERNEL, domain=_cls.covers, sample=_sampler(_cls),
             check_targets=partial(check_targets, _cls), covers=_cls.accepts)


def _first_diff(got: np.ndarray, expect: np.ndarray) -> dict:
    if got.size != expect.size:
        return {"n_diff": -1, "reason": f"{got.size} words vs {expect.size}"}
    neq = np.nonzero(got != expect)[0]
    return {"n_diff": int(neq.size), "first": [(int(i), int(got[i]), int(expect[i])) for i in neq[:4]]}


def _mismatch(m: tuple) -> dict:
    """One core `SelfCheck` mismatch: the twin's words against the reference's, or the reference's error."""
    got, ref = np.asarray(m[2], dtype=U64).reshape(-1), m[3]
    return {"n_diff": -1, "reason": ref} if isinstance(ref, str) else _first_diff(got, np.asarray(ref, dtype=U64).reshape(-1))


def check_instances(cls: type[Twin], statics_of: Callable[[np.random.Generator], dict], rng: np.random.Generator, n: int,
                    kinds: tuple[str, ...]) -> dict:
    """`n` random instances of twin family `cls` (statics drawn per instance, `kinds` in turn), each compared with the
    reference by core `self_check`; an instance the twin declines is counted under its reason."""
    rep: dict[str, Any] = {"n": 0, "n_declined": 0, "statics": [], "kinds": Counter(), "mismatches": [], "words": 0}
    for i in range(n):
        kind = kinds[i % len(kinds)]
        st = statics_of(rng)
        why = cls.accepts(st)
        if why:
            raise ValueError(f"{cls.twin_id}: check statics {st} not accepted: {why}")
        spec = bind(DEFINITIONS[cls.base_id], **st)
        args = cls.random_args(st, rng, kind)
        try:
            instance(spec, KERNEL)(*args, **spec.statics)
        except Declined as e:
            rep["n_declined"] += 1
            rep.setdefault("declined", Counter())[str(e).split(":")[0]] += 1
            continue
        sc = self_check(spec, kernel=KERNEL, inputs=[np.asarray(a).reshape(1, -1) for a in args])
        rep["kinds"][kind] += 1
        if spec.id not in rep["statics"]:
            rep["statics"].append(spec.id)
        rep["words"] += int(spec.signature()[1].leaves)
        if sc.mismatches:
            rep["mismatches"].append({"spec": spec.id, "kind": kind, **_mismatch(sc.mismatches[0])})
        else:
            rep["n"] += 1
    rep["kinds"] = dict(rep["kinds"])
    if "declined" in rep:
        rep["declined"] = dict(rep["declined"])
    rep["status"] = "fail" if rep["mismatches"] else ("pass" if rep["n"] else "declined")
    return rep


def admit(spec: SpecializedDefinition, n: int = 3, seed: int = 0) -> dict:
    """The short in-process admission check the Replayer runs before a twin's first use on `spec`: `n` random instances
    of the spec (its own statics, or the family's cheap variant of them) against the reference (`check_instances`)."""
    t0 = time.time()
    cls = FAMILIES[spec.base_id]
    rng = np.random.default_rng([seed, zlib.crc32(spec.id.encode())])
    rep = check_instances(cls, lambda r: cls.check_statics(spec.statics, r), rng, n, RANDOM_KINDS)
    return {"twin_id": cls.twin_id, "status": rep["status"], "n_random": rep["n"], "n_declined": rep["n_declined"],
            "statics": rep["statics"], "mismatches": rep["mismatches"], "seconds": round(time.time() - t0, 3)}


# ---------------------------------------------------------------------------------------------------------
# --check: the evidence file
# ---------------------------------------------------------------------------------------------------------


def check_spec(spec: SpecializedDefinition, rng: np.random.Generator, *, n_random: int, n_edge: int, captured: list | None,
               k: int) -> dict:
    """One spec: random instances (its statics or the cheap variant), captured instances (twin vs the output snapshot),
    edge instances; `captured` is a list of (args, expected) providers (callables) or None when unavailable."""
    t0 = time.time()
    entry: dict[str, Any] = {"base_id": spec.base_id, "twin_id": None, "n_random": 0, "n_captured": 0, "n_edge": 0,
                             "n_declined": 0, "status": "no_twin"}
    why = covers(spec, KERNEL)
    if why:
        entry["reason"] = why
        entry["seconds"] = round(time.time() - t0, 3)
        return entry
    tw, cls = instance(spec, KERNEL), FAMILIES[spec.base_id]
    entry["twin_id"] = cls.twin_id
    statics_of = lambda r: cls.check_statics(spec.statics, r)  # noqa: E731
    rnd = check_instances(cls, statics_of, rng, n_random, RANDOM_KINDS)
    edge = check_instances(cls, statics_of, rng, n_edge, EDGE_KINDS)
    entry.update(n_random=rnd["n"], n_edge=edge["n"], n_declined=rnd["n_declined"] + edge["n_declined"],
                 statics_checked=sorted(set(rnd["statics"]) | set(edge["statics"])), words_random=rnd["words"] + edge["words"],
                 mismatches=rnd["mismatches"] + edge["mismatches"])
    if edge.get("declined") or rnd.get("declined"):
        entry["declined"] = {**rnd.get("declined", {}), **edge.get("declined", {})}
    cap: dict[str, Any] = {"n": 0, "mismatches": [], "skipped": Counter(), "n_declined": 0, "nodes": []}
    for provider in (captured or [])[:k * 4]:
        if cap["n"] + len(cap["mismatches"]) >= k:
            break
        try:
            node, args, expect = provider()
        except Exception as e:  # noqa: BLE001 - missing evidence is counted, not fatal
            cap["skipped"][_skip_class(e)] += 1
            continue
        try:
            got = tw(*args, **spec.statics)
        except Declined as e:
            cap["n_declined"] += 1
            cap["skipped"][f"declined: {str(e)[:60]}"] += 1
            continue
        cap["nodes"].append(node)
        if got.shape != expect.shape or not np.array_equal(got, expect):
            cap["mismatches"].append({"node": node, **_first_diff(got, expect)})
        else:
            cap["n"] += 1
    entry["n_captured"] = cap["n"]
    entry["captured_nodes"] = cap["nodes"]
    entry["captured_skipped"] = dict(cap["skipped"])
    entry["mismatches"] += cap["mismatches"]
    entry["n_declined"] += cap["n_declined"]
    if entry["mismatches"]:
        entry["status"] = "fail"
    elif entry["n_random"] == 0:
        entry["status"] = "declined"
    else:
        entry["status"] = "pass"
    entry["seconds"] = round(time.time() - t0, 3)
    return entry


def _skip_class(e: Exception) -> str:
    """Short class of a missing-evidence reason, without the node-specific text."""
    msg = str(e)
    if "no snapshot" in msg:
        return "operand/output not snapshotted"
    if "weight source" in msg:
        return "no weight source"
    if "view unknown" in msg:
        return "output view unknown"
    return f"{type(e).__name__}: {msg[:60]}"


def check_families(rng: np.random.Generator, *, n_random: int, n_edge: int) -> dict:
    """Small random statics per twin family (block boundaries, partial warps, multi-chunk norms, ...) against the gates."""
    out = {}
    for base, cls in FAMILIES.items():
        t0 = time.time()
        rnd = check_instances(cls, cls.small_statics, rng, n_random, RANDOM_KINDS)
        edge = check_instances(cls, cls.small_statics, rng, n_edge, EDGE_KINDS)
        e = {"base_id": base, "n_random": rnd["n"], "n_edge": edge["n"], "n_declined": rnd["n_declined"] + edge["n_declined"],
             "statics": sorted(set(rnd["statics"]) | set(edge["statics"])), "words": rnd["words"] + edge["words"],
             "mismatches": rnd["mismatches"] + edge["mismatches"],
             "status": "fail" if rnd["mismatches"] or edge["mismatches"] else ("pass" if rnd["n"] else "declined"),
             "seconds": round(time.time() - t0, 3)}
        if rnd.get("declined") or edge.get("declined"):
            e["declined"] = {**rnd.get("declined", {}), **edge.get("declined", {})}
        out[cls.twin_id] = e
        print(f"  family {cls.twin_id:28s} {e['status']:8s} random {e['n_random']:3d} edge {e['n_edge']:3d} declined {e['n_declined']:2d} "
              f"{e['seconds']}s", file=sys.stderr, flush=True)
    return out
