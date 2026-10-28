"""The property red team's refusal suite: Programs a stacking property (property/v1, a closed free set with its family and
limb rules) must refuse or fully budget, and the honest steps.py shapes it must pass.

A check is any function from a `verity.ml.circuits.program.Program` to a verdict, a mapping ``{"refused": bool, "charged":
{class: units}}`` over `CLASSES`: ``roundings`` (casts, special functions, FP arithmetic, integer floors, mods and wraps, carries
and limb splits), ``compare_select`` (comparisons, clamps, selects), ``gathers`` (in fresh runtime index bits), ``products``
(products of two runtime values) and ``other``.
`CASES` are the cases, each built at full size by `Case.build()`; `run.py` grades a check against them, and `cases.json` is
their metadata as data (`run.py --json`).

An attack case's ``verdict`` is property/v1's, ``refuse`` or ``budget``, and its ``charge`` the least charge under which
accepting it is still sound (``{class or "any": units}``, "any" the sum over the classes), or None when no charge is (a type
pun). Refusing an attack is always sound. An honest case's ``allow`` is the most a check may charge it and still pass it.

Primitives are main's registry, with two steps registered under their ids only where absent, each with the signature and
silicon model of the branch that defines it: `BlackwellS8ImmaDot32_v1` (the census branch's exact s8 IMMA step) and
`BlackwellU8S8ImmaDot32_v1` (the s8 family's u8·s8 step without .satfinite). A case that binds the s8 family
(`verity.ml.circuits.ops.s8`: `S8Dot_v1`, `S8Matmul_v1`, `LimbSplit_v1`) names it in ``needs`` and is skipped where the
family is absent. Every honest case does, since property/v1 refuses a bare IMMA. The receipt fold (`PouwDuoAbsorb_v1`) is
registered nowhere yet, so ``fold-reused`` uses its LOP3+IMAD step (`benchmarks/pouw/edited_gemm/receipt_folds_r9.py`) as a
Definition of its own.
"""
from __future__ import annotations

import importlib
import struct
from dataclasses import dataclass, field
from typing import Any, Callable

from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition
from verity.ml.circuits.ops import prims
from verity.ml.circuits.ops import scalar  # noqa: F401  (I32Add, SelectI32, F32Fmaxf, F32Fminf register on import)
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, array_of, concat, tuple_of
from verity.ml.circuits.silicon.integer import (BLACKWELL_SM120_S8_M16N8K32, IntegerMma, S32_MAX, S32_MIN, s32_value,
                                                 s32_word)
from verity.ml.circuits.types import Array, Tuple as TupleT, Value

import verity.core.protocols.pouw.circuit.words  # noqa: F401  (PouwNoiseByte)
from verity.ml.definitions.registry import dsa, dsv4_qnorm_insert, flashmla, marlin_moe, topp_word_gates  # noqa: F401
from verity.ml.definitions.registry.ref_prims import gather_f32

try:
    FAMILY: Any = importlib.import_module("verity.ml.circuits.ops.s8")
except ModuleNotFoundError as e:
    if e.name != "verity.ml.circuits.ops.s8":
        raise
    FAMILY = None

CLASSES = ("roundings", "compare_select", "gathers", "products", "other")
NEEDS_FAMILY = "verity.ml.circuits.ops.s8"
M32 = (1 << 32) - 1
S8, S32, F32 = Value(8), Value(32), Value(32)
A32 = Array(32, S8)
S8_RANGE, U8_RANGE = (-128, 127), (0, 255)


def _p(fid: str) -> PrimitiveDefinition:
    return REGISTRY[fid]  # type: ignore[return-value]


def _step(fid: str, model: IntegerMma, doc: str) -> PrimitiveDefinition:
    if fid in REGISTRY:
        return _p(fid)
    name = fid.rsplit("_v", 1)[0]
    return PrimitiveDefinition(name, 1, (("acc", S32), ("a", A32), ("b", A32)), S32,
                               lambda acc, *ab, _m=model: _m.step(acc, ab[:32], ab[32:64]), doc=doc,
                               conformance="exact-model-tested")


U8S8_WRAP = IntegerMma(name="blackwell_sm120_u8s8_m16n8k32", arch="sm_120", dtype="u8", b_dtype="s8", operand_bits=8,
                       k_per_instruction=32, satfinite=False,
                       validated="the u8.s8 form without .satfinite on the device: run r20261015-084000-06d8")
IMMA = _step("BlackwellS8ImmaDot32_v1", BLACKWELL_SM120_S8_M16N8K32,
             "acc (s32) plus 32 exact s8 x s8 products, clamped once to s32 (IMMA.16832.S8.S8.SAT)")
IMMA_U8S8 = _step("BlackwellU8S8ImmaDot32_v1", U8S8_WRAP,
                  "acc (s32) plus 32 exact u8 x s8 products, modulo 2^32 (IMMA.16832.U8.S8)")


def _fn(name: str, params: list[tuple[str, Any]], ret: Any, body: Callable[..., Coll]) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda _S, _p=tuple(params), _r=ret: (_p, _r), body, register=False)


class _Consts:
    """One constant node per (width, bits) in a body."""

    def __init__(self, B: Any) -> None:
        self.B, self.memo = B, {}

    def word(self, w: int, v: int) -> Coll:
        key = (w, v & ((1 << w) - 1))
        if key not in self.memo:
            self.memo[key] = self.B.call(prims.const(*key))
        return self.memo[key]

    def lanes(self, values: list[int]) -> Coll:
        return array_of([self.word(8, v) for v in values])


def shift_lanes(total: int, a_max: int = 127, b_max: int = 127) -> tuple[list[int], list[int]]:
    """32 operand pairs 0 <= a <= a_max, 0 <= b <= b_max whose products sum to `total`: whole a_max * b_max lanes, then
    a_max * y and z * 1 for the rest."""
    q, r = divmod(total, a_max * b_max)
    y, z = divmod(r, a_max)
    a, b = [a_max] * q, [b_max] * q
    for x, w in ((a_max, y), (z, 1)):
        if x and w:
            a.append(x)
            b.append(w)
    if total < 0 or len(a) > 32:
        raise ValueError(f"{total} is not a sum of 32 products within [0, {a_max}] x [0, {b_max}]")
    return a + [0] * (32 - len(a)), b + [0] * (32 - len(b))


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _sext8(v: int) -> int:
    return ((v & 0xFF) ^ 0x80) - 0x80


@dataclass(frozen=True)
class Case:
    """``build(**size)`` returns the Program; ``samples(size)`` small instances ``(size, inputs, outputs)`` whose outputs
    are the claimed function's (``computes``), computed in Python apart from the IR."""

    id: str
    kind: str
    verdict: str
    computes: str
    rule: str
    build: Callable[..., Program] = field(repr=False)
    charge: dict | None = None
    allow: dict | None = None
    needs: tuple[str, ...] = ()
    size: dict = field(default_factory=dict)
    samples: Callable[[], list[tuple[dict, tuple, tuple]]] | None = field(default=None, repr=False)

    @property
    def available(self) -> bool:
        return all(m != NEEDS_FAMILY or FAMILY is not None for m in self.needs)

    def program(self) -> Program:
        return self.build(**self.size)

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "verdict": self.verdict, "charge": self.charge, "allow": self.allow,
                "computes": self.computes, "rule": self.rule, "needs": list(self.needs), "size": self.size}


# ---- carries from registered integer primitives ---------------------------------------------------------------------------


def carry_idiv() -> Program:
    def body(B, _S, t):
        k = _Consts(B)
        return tuple_of(B.call(_p("I32Rem_v1"), t, k.word(32, 256)), B.call(_p("I32Div_v1"), t, k.word(32, 256)))
    return Program(_fn("RedteamCarryIdiv", [("t", S32)], TupleT(S32, S32), body))


CMP_ROUNDS = 4


def carry_cmp() -> Program:
    """t mod 256 and t div 256 for 0 <= t < 256 * 2^CMP_ROUNDS: per round one U32Ge against a constant and two SelectI32."""
    def body(B, _S, t):
        k = _Consts(B)
        r, q = t, k.word(32, 0)
        for b in reversed(range(CMP_ROUNDS)):
            ge = B.call(_p("U32Ge_v1"), r, k.word(32, 256 << b))
            r = B.call(_p("SelectI32_v1"), ge, B.call(_p("I32Sub_v1"), r, k.word(32, 256 << b)), r)
            q = B.call(_p("SelectI32_v1"), ge, B.call(_p("I32Add_v1"), q, k.word(32, 1 << b)), q)
        return tuple_of(r, q)
    return Program(_fn("RedteamCarryCmp", [("t", S32)], TupleT(S32, S32), body))


# ---- an IMMA step with a runtime incoming accumulator ---------------------------------------------------------------------

WINDOW, THRESHOLD = 1 << 18, 1000  # r in [0, WINDOW) sits just under the s32 edge; the comparisons are against THRESHOLD


def _clamp_body(dot: Any) -> Callable[..., Coll]:
    """[r < THRESHOLD] = D1 - D2, D1 = sat(C + W - T + 1), D2 = sat(C + W - T), C = r + S32_MAX - W."""
    def body(B, _S, r):
        k = _Consts(B)
        c = B.call(_p("I32Add_v1"), r, k.word(32, S32_MAX - WINDOW))
        d = []
        for shift in (WINDOW - THRESHOLD + 1, WINDOW - THRESHOLD):
            a, b = shift_lanes(shift)
            d.append(B.call(dot, c, k.lanes(a), k.lanes(b)))
        return B.call(_p("I32Sub_v1"), d[0], d[1])
    return body


def _wrap_body(dot: Any) -> Callable[..., Coll]:
    """D = C + W - T modulo 2^32, C = r + 2^31 - W: D's s32 sign bit is [r >= THRESHOLD]."""
    def body(B, _S, r):
        k = _Consts(B)
        c = B.call(_p("I32Add_v1"), r, k.word(32, (1 << 31) - WINDOW))
        a, b = shift_lanes(WINDOW - THRESHOLD, 255, 127)
        return B.call(dot, c, k.lanes(a), k.lanes(b))
    return body


def imma_clamp_bare() -> Program:
    return Program(_fn("RedteamImmaClamp", [("r", S32)], S32, _clamp_body(IMMA)))


def imma_wrap_bare() -> Program:
    return Program(_fn("RedteamImmaWrap", [("r", S32)], S32, _wrap_body(IMMA_U8S8)))


def _clamp_samples() -> list[tuple[dict, tuple, tuple]]:
    return [({}, (r,), (int(r < THRESHOLD),)) for r in (0, 1, THRESHOLD - 1, THRESHOLD, THRESHOLD + 1, WINDOW - 1)]


def _wrap_samples() -> list[tuple[dict, tuple, tuple]]:
    return [({}, (r,), ((r + (1 << 31) - THRESHOLD) & M32,)) for r in (0, THRESHOLD - 1, THRESHOLD, WINDOW - 1)]


# ---- a declared operand bound the producer exceeds ------------------------------------------------------------------------

DITHER_K = 132_096  # within s32 for operands in [-127, 127] (K <= 133,144), past it for [-128, 127] (K <= 131,071)


def _chain_step(dot: PrimitiveDefinition) -> CompositeDefinition:
    def body(B, _S, acc, a, b):
        d = B.call(dot, acc, a, b)
        return tuple_of(d, d)
    return _fn("RedteamChainStep", [("acc", S32), ("a", A32), ("b", A32)], TupleT(S32, S32), body)


CHAIN_STEP = _chain_step(IMMA)


def imma_bound_bare(K: int = DITHER_K) -> Program:
    """A bare chain of K/32 IMMA steps from 0 over two run inputs, annotated with a declared operand bound of +-127."""
    def body(B, _S, a, b):
        blocks = Array(K // 32, A32)
        return B.scan(CHAIN_STEP, B.call(prims.ZERO32), xs=[a.reshape(blocks), b.reshape(blocks)])[0]
    return Program(_fn("RedteamImmaBound", [("a", Array(K, S8)), ("b", Array(K, S8))], S32, body),
                   annotations={"declared_operand_bound": [-127, 127]})


def _dither_samples() -> list[tuple[dict, tuple, tuple]]:
    """-128 by -128 reaches S32_MAX at step 4,096 and clamps there; -128 by 127 stays exact."""
    K = DITHER_K
    return [({}, ([0x80] * K, [0x80] * K), (S32_MAX,)),
            ({}, ([0x80] * K, [0x7F] * K), (s32_word(-128 * 127 * K),))]


# ---- type puns, noise, the receipt fold, a gather -------------------------------------------------------------------------


def pun_s32_as_e4m3() -> Program:
    """A computed s32 word's leaf read where an 8-bit E4M3 code is typed (E4m3ToF32, a free cast in census v1)."""
    def body(B, _S, t):
        k = _Consts(B)
        y = B.call(_p("I32Add_v1"), t, k.word(32, 0))
        return B.call(_p("E4m3ToF32_v1"), Coll(S8, y.refs))
    return Program(_fn("RedteamPunE4m3", [("t", S32)], F32, body))


def noise_runtime() -> Program:
    def body(B, _S, w, c):
        return B.call(_p("PouwNoiseByte_v1"), w, c)
    return Program(_fn("RedteamNoiseRuntime", [("w", S32), ("c", S32)], S8, body))


FOLD_A0, FOLD_K0, FOLD_SEED = 0x9E3779B1, 0x7F4A7C15, 0x01234567


def _fold_body(B, _S, y, d0, d1):
    k = _Consts(B)
    x = B.call(_p("U32Xor_v1"), B.call(_p("U32Xor_v1"), y, d0), k.word(32, FOLD_K0))
    return B.call(_p("I32Add_v1"), B.call(_p("I32Mul_v1"), x, k.word(32, FOLD_A0)), d1)


#: receipt_folds_r9's LOP3+IMAD step, y' = (y ^ d0 ^ k0) * a0 + d1 (mod 2^32), standing in for PouwDuoAbsorb_v1
RECEIPT_FOLD = _fn("RedteamReceiptFold", [("y", S32), ("d0", S32), ("d1", S32)], S32, _fold_body)


def fold(y: int, d0: int, d1: int) -> int:
    return (((y ^ d0 ^ FOLD_K0) * FOLD_A0) + d1) & M32


def fold_reused() -> Program:
    """The fold's output is the receipt and is reused: added into another output."""
    def body(B, _S, d0, d1, x):
        k = _Consts(B)
        y = B.call(RECEIPT_FOLD, k.word(32, FOLD_SEED), d0, d1)
        return tuple_of(y, B.call(_p("I32Add_v1"), y, x))
    return Program(_fn("RedteamFoldReused", [("d0", S32), ("d1", S32), ("x", S32)], TupleT(S32, S32), body))


def table(i: int) -> int:
    return _f32(float((i * i + 7 * i) % 251))


def gather_table(V: int = 128) -> Program:
    """A run index into a constant column of V f32 words: column[i] = (i^2 + 7i) mod 251, any function of the index."""
    def body(B, _S, i):
        k = _Consts(B)
        return B.call(gather_f32(V), i, array_of([k.word(32, table(j)) for j in range(V)]))
    return Program(_fn("RedteamGatherTable", [("i", S32)], F32, body))


# ---- the s8 family (verity.ml.circuits.ops.s8) ----------------------------------------------------------------------------


def _dot(**kw: Any) -> Any:
    statics = {"L": kw["K"], "T0": "zero", "C": None, "DOT": IMMA, "A": S8_RANGE, "B": S8_RANGE, "TAIL": (), "TDOT": None,
               "NOISE": "none", "LIMBS": "none", "COMMIT": "final"} | kw
    return FAMILY.S8Dot.bind(**statics)


def _matmul(**kw: Any) -> Any:
    statics = {"L": kw["K"], "T0": "zero", "C": None, "DOT": IMMA, "A": S8_RANGE, "B": S8_RANGE, "TAIL": (), "TDOT": None,
               "NOISE": "none", "LIMBS": "none", "COMMIT": "final"} | kw
    return FAMILY.S8Matmul.bind(**statics)


def fam_bound_dither(K: int = DITHER_K) -> Program:
    """S8Dot bound with A = B = (-127, 127), whose no_wrap then holds at K, over run inputs that reach -128."""
    dot = _dot(K=K, A=(-127, 127), B=(-127, 127))
    return Program(_fn("RedteamFamBound", [("a", Array(K, S8)), ("b", Array(K, S8))], S32,
                       lambda B, _S, a, b: B.call(dot, a, b)))


def fam_acc_clamp() -> Program:
    """S8Dot{T0 = input} with C declared (0, W), its accumulator in fact r + S32_MAX - W."""
    dot = _dot(K=32, T0="input", C=(0, WINDOW), A=(0, 127), B=(0, 127))
    return Program(_fn("RedteamFamAccClamp", [("r", S32)], S32, _clamp_body(dot)))


def fam_acc_wrap() -> Program:
    dot = _dot(K=32, T0="input", C=(0, WINDOW), DOT=IMMA_U8S8, A=U8_RANGE, B=(0, 127))
    return Program(_fn("RedteamFamAccWrap", [("r", S32)], S32, _wrap_body(dot)))


SPLIT_HALF = 131_040  # the most K within s32 for operands in [-128, 127] that is a multiple of 32


def fam_splitk(H: int = SPLIT_HALF) -> Program:
    """Two S8Dot halves, each proven within s32, combined by I32Add, which may wrap."""
    dot = _dot(K=H)

    def body(B, _S, a, b):
        return B.call(_p("I32Add_v1"), B.call(dot, a[:H], b[:H]), B.call(dot, a[H:], b[H:]))
    return Program(_fn("RedteamFamSplitK", [("a", Array(2 * H, S8)), ("b", Array(2 * H, S8))], S32, body))


def _splitk_samples() -> list[tuple[dict, tuple, tuple]]:
    H = SPLIT_HALF
    half = -128 * -128 * H
    return [({}, ([0x80] * 2 * H, [0x80] * 2 * H), ((2 * half) & M32,)),
            ({}, ([0x80] * H + [0] * H, [0x80] * 2 * H), (half,))]


Y_DOT32 = (32 * -128 * 127, 32 * 128 * 128)  # an S8Dot{K = 32} word's interval over s8 operands


def _y(B: Any, x: Coll, w: Coll) -> Coll:
    return B.call(_dot(K=32), x, w)


def limb_split(N: int = 3, Y: tuple[int, int] = Y_DOT32, name: str = "RedteamLimbSplit") -> Program:
    """LimbSplit_v1{bytes} of a runtime S8Dot word, which N - 1 carries cover when Y is its true interval."""
    split = FAMILY.LimbSplit.bind(FORM="bytes", N=N, Y=Y)
    return Program(_fn(name, [("x", A32), ("w", A32)], Array(N, S8), lambda B, _S, x, w: B.call(split, _y(B, x, w))))


def limb_split_bare(N: int = 3) -> Program:
    """The same split from LimbSplit_v1's own primitives, called outside it."""
    def body(B, _S, x, w):
        r, limbs = _y(B, x, w), []
        for _ in range(N - 1):
            limbs.append(B.call(FAMILY.S32LowByte, r))
            r = B.call(FAMILY.S32ByteCarry, r)
        limbs.append(B.call(FAMILY.S32LowByte, r))
        return array_of(limbs)
    return Program(_fn("RedteamLimbSplitBare", [("x", A32), ("w", A32)], Array(N, S8), body))


def fam_limb_y() -> Program:
    """LimbSplit_v1{bytes, N = 2} with Y declared (-32640, 32639) on a word in Y_DOT32: its last limb is a truncation."""
    return limb_split(N=2, Y=(-32640, 32639), name="RedteamFamLimbY")


def limbs(y: int, N: int) -> tuple[int, ...]:
    """LimbSplit_v1{bytes}'s limbs of y as s8 codes: sext8 of each remainder, the last remainder's low byte."""
    out, r = [], y
    for _ in range(N - 1):
        out.append(_sext8(r) & 0xFF)
        r = (r - _sext8(r)) >> 8
    out.append(r & 0xFF)
    return tuple(out)


def _dot_value(x: list[int], w: list[int]) -> int:
    return sum(_sext8(a) * _sext8(b) for a, b in zip(x, w))


def _limb_samples(N: int) -> Callable[[], list[tuple[dict, tuple, tuple]]]:
    def samples() -> list[tuple[dict, tuple, tuple]]:
        out = []
        for x, w in (([0x80] * 32, [0x80] * 32), ([0x7F] * 32, [0x80] * 32), ([3] * 32, [5] * 32), ([0x81, 7] * 16, [9, 0x90] * 16)):
            out.append(({}, (x, w), limbs(_dot_value(x, w), N)))
        return out
    return samples


def pun_s32_as_s8() -> Program:
    """A runtime S8Dot word read back as 32 s8 operand codes of a second S8Dot, without a split."""
    def body(B, _S, x, w, v):
        y = _y(B, x, w)
        return B.call(_dot(K=32), Coll(A32, concat([y.refs] * 32)), v)
    return Program(_fn("RedteamPunS8", [("x", A32), ("w", A32), ("v", A32)], S32, body))


# ---- the honest steps.py shapes -------------------------------------------------------------------------------------------


def linear(K: int = 28_672) -> Program:
    """One output coordinate of an int8 linear (Llama-3.1-70B's MLP down projection, K = 28,672): activations per run,
    weights registered, both s8 codes."""
    dot = _dot(K=K)
    return Program(_fn("RedteamLinear", [("x", Array(K, S8)), ("w", Array(K, S8))], S32,
                       lambda B, _S, x, w: B.call(dot, x, w)), registered=("w",))


def qk(N: int = 131_072, K: int = 128) -> Program:
    """One query row against N keys at head dimension K."""
    mm = _matmul(N=N, K=K)
    return Program(_fn("RedteamQK", [("q", Array(K, S8)), ("keys", Array(N, Array(K, S8)))], Array(N, S32),
                       lambda B, _S, q, keys: B.call(mm, q, keys)))


def _pquant_body(B, _S, p):
    k = _Consts(B)
    x = B.call(_p("F32Fminf_v1"), B.call(_p("F32Fmaxf_v1"), p, k.word(32, _f32(0.0))), k.word(32, _f32(127.0)))
    return B.call(_p("F32ToU32RzByte_v1"), x)


#: P's producer: fminf(fmaxf(p, 0), 127) truncated to a byte, so P in [0, 127] (steps.py's round(.).clamp(0, 127) with its
#: half folded into the scale)
PQUANT = _fn("RedteamPQuant", [("p", F32)], S8, _pquant_body)


def _quant(name: str, lo: float | None, hi: float | None) -> CompositeDefinition:
    """P's producer with either clamp left out or moved: RzByte(fminf(fmaxf(p, lo), hi))."""
    def body(B, _S, p):
        k, x = _Consts(B), p
        if lo is not None:
            x = B.call(_p("F32Fmaxf_v1"), x, k.word(32, _f32(lo)))
        if hi is not None:
            x = B.call(_p("F32Fminf_v1"), x, k.word(32, _f32(hi)))
        return B.call(_p("F32ToU32RzByte_v1"), x)
    return _fn(name, [("p", F32)], S8, body)


#: F32ToU32RzByte takes every negative, -0 and NaN to 0, so P is in [0, 127] without the fmaxf
PQUANT_NO_FMAXF = _quant("RedteamPQuantNoFmaxf", None, 127.0)
#: without the fminf P is the truncation's low byte, the whole byte (0xFF from 2^32 and +inf)
PQUANT_NO_FMINF = _quant("RedteamPQuantNoFminf", 0.0, None)
#: a clamp at 128, whose byte the s8 step reads as -128
PQUANT_128 = _quant("RedteamPQuant128", 0.0, 128.0)

#: the most keys a family P·V holds within s32 with P in [0, 127] against V in [-128, 127]: 127·128·K <= 2^31 gives
#: K <= 132,104, and the family's K is a multiple of 32
PV_LIMIT = 132_096


def _pv(K: int, N: int, quant: CompositeDefinition, A: tuple[int, int], name: str) -> Program:
    mm = _matmul(N=N, K=K, A=A)

    def body(B, _S, p, v):
        return B.call(mm, B.batch(quant, p, axes=[0]), v)
    return Program(_fn(name, [("p", Array(K, F32)), ("v", Array(N, Array(K, S8)))], Array(N, S32), body))


def pv(K: int = 131_072, N: int = 128) -> Program:
    """P (one query row's probabilities over K keys, quantized by PQUANT) against the N value channels."""
    return _pv(K, N, PQUANT, (0, 127), "RedteamPV")


def pv_past(K: int = PV_LIMIT + 32, N: int = 128) -> Program:
    """P·V one family step past the limit, bound with A = (0, 126) so that its no_wrap holds for the declaration."""
    return _pv(K, N, PQUANT, (0, 126), "RedteamPVPast")


def pv_no_fmaxf(K: int = PV_LIMIT, N: int = 128) -> Program:
    return _pv(K, N, PQUANT_NO_FMAXF, (0, 127), "RedteamPVNoFmaxf")


def pv_no_fminf(K: int = 131_072, N: int = 128) -> Program:
    return _pv(K, N, PQUANT_NO_FMINF, (0, 127), "RedteamPVNoFminf")


def pv_clamp128(K: int = PV_LIMIT, N: int = 128) -> Program:
    return _pv(K, N, PQUANT_128, (0, 127), "RedteamPVClamp128")


def pv_splitk(K: int = 131_072, N: int = 128) -> Program:
    """P·V split at K/2, the two halves' words added by I32Add (its interval within s32)."""
    H = K // 2
    mm = _matmul(N=N, K=H, A=(0, 127))

    def body(B, _S, p, v0, v1):
        P = B.batch(PQUANT, p, axes=[0])
        return B.batch(_p("I32Add_v1"), B.call(mm, P[:H], v0), B.call(mm, P[H:], v1), axes=[0, 0])
    return Program(_fn("RedteamPVSplitK", [("p", Array(K, F32)), ("v0", Array(N, Array(H, S8))),
                                           ("v1", Array(N, Array(H, S8)))], Array(N, S32), body))


def pquant(p: int) -> int:
    """PQUANT on an f32 encoding: NaN and every value below 0 to 0, above 127 to 127, else truncated."""
    x = struct.unpack("<f", struct.pack("<I", p))[0]
    return 0 if x != x else int(min(max(x, 0.0), 127.0))


def rz_byte(x: float) -> int:
    """F32ToU32RzByte on a value: the low byte of cvt.rzi.u32.f32, which takes a NaN and every negative to 0 and every
    x >= 2^32 to 0xFFFFFFFF."""
    if x != x or x < 0:
        return 0
    return 0xFF if x >= 2.0 ** 32 else int(x) & 0xFF


def quant(p: int, lo: float | None, hi: float | None) -> int:
    """`_quant(lo, hi)` on an f32 encoding: fmaxf and fminf return the operand that is not NaN."""
    x = struct.unpack("<f", struct.pack("<I", p))[0]
    if lo is not None:
        x = lo if x != x else max(x, lo)
    if hi is not None:
        x = hi if x != x else min(x, hi)
    return rz_byte(x)


def _rows(N: int, K: int, seed: int) -> list[list[int]]:
    return [[(seed * 7919 + 131 * n + 17 * k) % 256 for k in range(K)] for n in range(N)]


def _ps(K: int) -> list[int]:
    vals = [-3.5, 0.0, 0.4, 1.0, 126.6, 127.0, 300.0, float("inf"), float("-inf"), -0.0]
    return [_f32(vals[k % len(vals)]) if k % 3 else 0x7FC00000 + k for k in range(K)]


def _qk_samples() -> list[tuple[dict, tuple, tuple]]:
    N, K = 3, 128
    q, keys = _rows(1, K, 1)[0], _rows(N, K, 2)
    return [({"N": N, "K": K}, (q, [c for row in keys for c in row]),
             tuple(s32_word(_dot_value(q, row)) for row in keys))]


def _linear_samples() -> list[tuple[dict, tuple, tuple]]:
    x, w = _rows(2, 64, 3)
    return [({"K": 64}, (x, w), (s32_word(_dot_value(x, w)),)),
            ({"K": 64}, ([0x80] * 64, [0x80] * 64), (64 * 128 * 128,))]


def _pv_reference(p: list[int], v: list[list[int]], q: Callable[[int], int] = pquant) -> tuple[int, ...]:
    """P's bytes against V's, both read as s8 by the step, so a P byte of 128 or more is negative."""
    P = [q(x) for x in p]
    return tuple(s32_word(sum(_sext8(a) * _sext8(b) for a, b in zip(P, row))) for row in v)


def _pv_samples() -> list[tuple[dict, tuple, tuple]]:
    N, K = 2, 64
    p, v = _ps(K), _rows(N, K, 4)
    return [({"N": N, "K": K}, (p, [c for row in v for c in row]), _pv_reference(p, v))]


def _ps_edges(K: int) -> list[int]:
    """P's inputs at the variants' edges: past 127 and 128, past a byte, at 2^32, the infinities, -0 and NaNs."""
    vals = [-3.5, 0.0, 0.4, 126.6, 127.9, 128.0, 128.5, 200.0, 255.9, 256.0, 300.0, 2.0 ** 32, float("inf"),
            float("-inf"), -0.0]
    return [_f32(vals[k % len(vals)]) if k % 4 else 0x7FC00000 + k for k in range(K)]


def _pv_variant_samples(lo: float | None, hi: float | None, seed: int) -> Callable[[], list[tuple[dict, tuple, tuple]]]:
    def samples() -> list[tuple[dict, tuple, tuple]]:
        N, K = 2, 64
        p, v = _ps_edges(K), _rows(N, K, seed)
        return [({"N": N, "K": K}, (p, [c for row in v for c in row]),
                 _pv_reference(p, v, lambda x: quant(x, lo, hi)))]
    return samples


def _pv_splitk_samples() -> list[tuple[dict, tuple, tuple]]:
    N, K = 2, 64
    p, v = _ps(K), _rows(N, K, 5)
    H = K // 2
    flat = [c for row in v for c in row[:H]], [c for row in v for c in row[H:]]
    return [({"N": N, "K": K}, (p, *flat), _pv_reference(p, v))]


# ---- the cases ------------------------------------------------------------------------------------------------------------

_FAM = (NEEDS_FAMILY,)
_BARE = "a bare IMMA outside the s8 family is refused"
_RULES = "the interval rules for P's producers (F32Fmaxf, F32Fminf, F32ToU32RzByte)"

CASES: tuple[Case, ...] = (
    Case("carry-idiv", "attack", "budget", "a carry: (t mod 256, t div 256) of a run word, by I32Rem and I32Div",
         "integer floors and mods are budgeted, one unit an output", carry_idiv, charge={"roundings": 2},
         samples=lambda: [({}, (t,), (t % 256, t // 256)) for t in (0, 255, 256, 1000, 65_535, S32_MAX)]),
    Case("carry-cmp", "attack", "budget",
         f"a carry from comparisons: t mod 256 and t div 256 for t < {256 << CMP_ROUNDS}, by {CMP_ROUNDS} rounds of U32Ge "
         "and two SelectI32", "compares and selects are budgeted, one unit an output", carry_cmp,
         charge={"compare_select": 3 * CMP_ROUNDS},
         samples=lambda: [({}, (t,), (t % 256, t // 256)) for t in (0, 255, 256, 1000, (256 << CMP_ROUNDS) - 1)]),
    Case("imma-clamp-bare", "attack", "refuse",
         f"[r < {THRESHOLD}] for r < 2^18 from two .satfinite IMMA steps whose incoming accumulator is r + S32_MAX - 2^18",
         _BARE + "; a runtime T_0 with no derived interval makes the clamp a comparison", imma_clamp_bare,
         charge={"compare_select": 2}, samples=_clamp_samples),
    Case("imma-wrap-bare", "attack", "refuse",
         f"a word whose s32 sign is [r >= {THRESHOLD}], from one u8 x s8 IMMA step without .satfinite at T_0 = r + 2^31 - 2^18",
         _BARE + "; a runtime T_0 with no derived interval makes the wrap a comparison", imma_wrap_bare,
         charge={"roundings": 1}, samples=_wrap_samples),
    Case("imma-bound-bare", "attack", "refuse",
         f"a K = {DITHER_K:,} chain over run s8 inputs that clamps at S32_MAX from step 4,096 on (33 steps), under a "
         "declared operand bound of +-127 the inputs exceed at -128",
         _BARE + "; operand ranges come from the producer, never from a declaration", imma_bound_bare,
         charge={"compare_select": 33}, samples=_dither_samples),
    Case("pun-s32-as-e4m3", "attack", "refuse", "a computed s32 word's leaf read where an 8-bit E4M3 code is typed",
         "an implicit reinterpretation of a runtime word is refused; one type per committed word", pun_s32_as_e4m3,
         samples=lambda: [({}, (0x38,), (_f32(1.0),)), ({}, (0x40,), (_f32(2.0),))]),
    Case("noise-runtime", "attack", "refuse",
         "six bits of a run word at a run-chosen byte offset, by PouwNoiseByte over a value that is not the coin",
         "the noise derivation's inputs are the coin and statics only; else PouwNoiseByte is budgeted",
         noise_runtime, charge={"roundings": 1},
         samples=lambda: [({}, (w, c), ((((w >> (8 * (c & 3))) & 63) - 31) & 0xFF,))
                          for w, c in ((0xDEADBEEF, 0), (0xDEADBEEF, 3), (0x01020304, 6))]),
    Case("fold-reused", "attack", "budget",
         "the receipt fold's mixer, y' = (y ^ d0 ^ k0) * a0 + d1 mod 2^32, its output added into a second output",
         "the receipt fold's output is a sink; the fold is unregistered, so its four gates are budgeted",
         fold_reused, charge={"any": 4},
         samples=lambda: [({}, (d0, d1, x), (fold(FOLD_SEED, d0, d1), (fold(FOLD_SEED, d0, d1) + x) & M32))
                          for d0, d1, x in ((1, 2, 3), (0xFFFFFFFF, 0x80000000, 0x7FFFFFFF))]),
    Case("gather-table", "attack", "budget", "any function of a 7-bit run index, (i^2 + 7i) mod 251, by GatherF32x128",
         "gathers are budgeted by fresh runtime index bits", gather_table, charge={"gathers": 7},
         samples=lambda: [({}, (i,), (table(i),)) for i in (0, 1, 100, 127)] + [({}, (128,), (0x7FC00000,))]),
    Case("fam-bound-dither", "attack", "refuse",
         f"S8Dot{{K = {DITHER_K:,}, A = B = (-127, 127)}} binds (its no_wrap holds for the declared bounds) and clamps "
         "at S32_MAX on run inputs at -128", "operand ranges come from the producing Definition, never declared",
         fam_bound_dither, charge={"compare_select": 33}, needs=_FAM, samples=_dither_samples),
    Case("fam-acc-clamp", "attack", "refuse",
         f"[r < {THRESHOLD}] from two S8Dot{{T0 = input, C = (0, 2^18)}} instances whose accumulator is r + S32_MAX - 2^18",
         "T_0's interval is derived from its producer, not declared", fam_acc_clamp, charge={"compare_select": 2},
         needs=_FAM, samples=_clamp_samples),
    Case("fam-acc-wrap", "attack", "refuse",
         f"a word whose s32 sign is [r >= {THRESHOLD}], from S8Dot{{T0 = input, C = (0, 2^18), DOT = u8 x s8}} at "
         "T_0 = r + 2^31 - 2^18", "T_0's interval is derived from its producer, not declared", fam_acc_wrap,
         charge={"roundings": 1}, needs=_FAM, samples=_wrap_samples),
    Case("fam-splitk", "attack", "budget",
         f"two S8Dot{{K = {SPLIT_HALF:,}}} halves over run s8 inputs, each within s32, whose I32Add combine wraps",
         "a split-K combine is inside the interval analysis; one that may wrap is not free-affine", fam_splitk,
         charge={"roundings": 1}, needs=_FAM, samples=_splitk_samples),
    Case("limb-split", "attack", "budget", "a runtime S8Dot word as three s8 limbs by LimbSplit_v1{bytes, N = 3}",
         "a LimbSplit of a runtime word is budgeted, a carry a limb beyond the first", limb_split,
         charge={"roundings": 2}, needs=_FAM, samples=_limb_samples(3)),
    Case("limb-split-bare", "attack", "budget",
         "the same three limbs from S32LowByte and S32ByteCarry called outside LimbSplit_v1",
         "every registered primitive outside the free set is budgeted, one unit an output", limb_split_bare,
         charge={"roundings": 2}, needs=_FAM, samples=_limb_samples(3)),
    Case("fam-limb-y", "attack", "refuse",
         "LimbSplit_v1{bytes, N = 2, Y = (-32640, 32639)} on a word in [-520192, 524288]: its last limb truncates a second "
         "carry", "a split's Y is its producer's interval, not declared", fam_limb_y, charge={"roundings": 2},
         needs=_FAM, samples=_limb_samples(2)),
    Case("pun-s32-as-s8", "attack", "refuse",
         "a runtime S8Dot word read back as the s8 operand codes of a second S8Dot, without a split",
         "an implicit reinterpretation (an s32 word read back as s8 limbs) is refused", pun_s32_as_s8,
         needs=_FAM, samples=lambda: [({}, ([1] * 32, [2] * 32, [1] * 32), (64 * 32,))]),
    Case("linear-28672", "honest", "pass", "one int8 linear coordinate at K = 28,672 (s8 by s8, from T_0 = 0)",
         "free-family: the window is proven from the operands' type range", linear, allow={}, needs=_FAM,
         samples=_linear_samples),
    Case("qk-128", "honest", "pass", "one query against 131,072 keys at head dimension 128",
         "free-family", qk, allow={}, needs=_FAM, samples=_qk_samples),
    Case("pv-131072", "honest", "pass",
         "P·V at 131,072 keys: P in [0, 127] from its producer against V in [-128, 127], 128 channels",
         "free-family only with P's interval derived from PQUANT (with P in [-128, 127] the window ends at 131,071)",
         pv, allow={"any": 3 * 131_072}, needs=_FAM, samples=_pv_samples),
    Case("pv-131072-splitk", "honest", "pass", "P·V at 131,072 keys split at 65,536, the halves added by I32Add",
         "free-family halves; the combine is free-affine, its interval within s32", pv_splitk,
         allow={"any": 3 * 131_072}, needs=_FAM, samples=_pv_splitk_samples),
    Case("pv-132096", "honest", "pass",
         f"P·V at {PV_LIMIT:,} keys, the most a family K holds within s32 for P in [0, 127] against V in [-128, 127]",
         _RULES + "; without them P reads as s8 and the window ends at 131,071", pv, allow={"any": 3 * PV_LIMIT},
         needs=_FAM, size={"K": PV_LIMIT}, samples=_pv_samples),
    Case("pv-132128", "attack", "refuse",
         f"P·V at {PV_LIMIT + 32:,} keys over PQUANT's P in [0, 127], bound with A = (0, 126): from runs at -128 the "
         "last step clamps at S32_MIN, one comparison a channel",
         "operand ranges come from the producer, never declared; " + _RULES + " prove the window to 132,104 "
         "keys only", pv_past, charge={"any": 3 * (PV_LIMIT + 32) + 128}, needs=_FAM,
         samples=_pv_variant_samples(0.0, 127.0, 6)),
    Case("pv-no-fmaxf-132096", "honest", "pass",
         f"P·V at {PV_LIMIT:,} keys with P = RzByte(fminf(p, 127)), no fmaxf: RzByte takes every negative and NaN to 0",
         "F32ToU32RzByte's rule is cvt.rzi.u32.f32's, which saturates below at 0; a rule that reads a negative input as "
         "a wrapped byte refuses this, over-strict", pv_no_fmaxf, allow={"any": 2 * PV_LIMIT}, needs=_FAM,
         samples=_pv_variant_samples(None, 127.0, 7)),
    Case("pv-no-fminf-131072", "attack", "refuse",
         "P·V at 131,072 keys with P = RzByte(fmaxf(p, 0)), no fminf: P is the truncation's whole low byte, read as s8, "
         "so the last step clamps at S32_MAX, one comparison a channel",
         "F32ToU32RzByte's output is [0, 255] unless its input is below 256: " + _RULES, pv_no_fminf,
         charge={"any": 2 * 131_072 + 128}, needs=_FAM, samples=_pv_variant_samples(0.0, None, 8)),
    Case("pv-clamp128-132096", "attack", "refuse",
         f"P·V at {PV_LIMIT:,} keys with P = RzByte(fminf(fmaxf(p, 0), 128)): the byte 128 is s8 -128, so the last 33 "
         "steps may clamp at S32_MAX, 33 comparisons a channel",
         "P's interval is its producer's, [0, 128], and its s8 reading [-128, 127]: " + _RULES, pv_clamp128,
         charge={"any": 3 * PV_LIMIT + 33 * 128}, needs=_FAM, samples=_pv_variant_samples(0.0, 128.0, 9)),
)

BY_ID = {c.id: c for c in CASES}
assert len(BY_ID) == len(CASES)

__all__ = ["BY_ID", "CASES", "CLASSES", "Case", "FAMILY", "IMMA", "IMMA_U8S8", "NEEDS_FAMILY", "PQUANT", "RECEIPT_FOLD",
           "fold", "limbs", "pquant", "shift_lanes", "table"]
