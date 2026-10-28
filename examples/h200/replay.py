"""The CPU replay of the H200 demo's drawn units: given a step's drawn units, each with its opened input rows and its
committed output rows, recompute every unit's outputs on the CPU and compare them bit for bit with the GPU's.

**Templates.** One replay function per template of the frozen shape (MQA decoder, width 16,384, 52 layers, 256 query heads of
64, one KV head, MLP 65,536, byte vocabulary, context 32), in circuits' forms (their plan's row semantics), each at any binding
of its statics, so a tiny instance of the architecture replays through the same functions. Each `Template` is one of:

* ``definition``: main's Definition, evaluated by the circuits evaluator (`evaluation.evaluate_batch`, which takes its
  numpy kernel where one is registered): ``ScaledMmS8Coordinate_v1{K, DOT}`` (the s8 step, ``GEMM_SM90_S8``),
  ``GemmCoordinate_v2{K, DOT}`` (``GEMM_HOPPER``), ``Bf16Add_v2``, and any circuits Definition below once it is importable
  (`Template.definition`), which the reference then must equal (the suite checks it).
* ``pending-definition``: a Definition circuits states but main doesn't have yet. Its function here is a numpy reference
  of circuits' stated semantics, every f32 operation main's primitive in the stated order: ``add``, ``mul``, ``div`` are
  `fp32.F32AddV2` / ``F32MulV2`` / ``F32DivV2`` (IEEE RN, every NaN 0x7FFFFFFF), ``sqrt`` is ``PouwF32SqrtRn``, ``ex2`` is
  `mufu.MufuEx2Ftz`'s kernel, every bf16 rounding `prims.F2fpBf16`, every reduction sequential in ascending index from +0.
  The suite holds each reference to a scalar composition of those primitives' own evaluators.

| template | form | status |
|---|---|---|
| ``RmsSumSqPart_v1{C, HALF}`` | x: bf16[C] -> half of sum_i RN(x_i^2) | pending |
| ``F32SumPart_v1{G, HALF}`` | s: f32[G] -> half of sum_g s_g | pending |
| ``RmsNormRecip_v1{N, G, EPS, HALF}`` | s: f32[G] -> half of rs = RN(1/RN(sqrt(RN(RN(sum/N) + eps)))) | pending |
| ``RmsNormOut_v1`` | (x: bf16, rs: f32, w: bf16) -> bf16(RN(w * bf16(RN(x * rs)))) | pending |
| ``RowAbsMax_v1{N, RELU}`` | x: bf16[N] -> max abs (relu'd first under RELU) | pending |
| ``QuantS8Pair_v1{RELU}`` | (x: bf16[2], amax: bf16) -> s8[2] | pending |
| ``QuantS8Scale_v1{HALF}`` | amax: bf16 -> half of RN(amax/127) (1 when amax is not > 0) | pending |
| ``S8AccumulatorPart_v1{K, HALF}`` | (x, wrow: s8[K]) -> half of the exact s32 dot | pending |
| ``S32SumPart_v1{P, HALF}`` | parts: s32[P] -> half of their sum mod 2^32 | pending |
| ``ScaledMmS8Epilogue_v1{P}`` | (parts: s32[P], sx, sw: f32) -> bf16(RN(RN(f32(sum) * sx) * sw)) | pending |
| ``ScaledMmS8Coordinate_v1{K, DOT}`` | the whole W8A8 coordinate | definition |
| ``GemmCoordinate_v2{K, DOT}`` | the BF16 coordinate (QK^T) | definition |
| ``GemmCoordinatePad_v1{T, K, DOT}`` | (x, wrow: bf16[T]) -> ``GemmCoordinate_v2{K}`` on both padded with +0 (PV) | pending |
| ``SoftmaxMax_v1{N}`` | x: bf16[N] -> the first greatest (+0 = -0), 0x7FFF if any is NaN | pending |
| ``SoftmaxRecip_v1{N, HALF}`` | (x: bf16[N], m: bf16) -> half of RN(1 / sum_j e_j) | pending |
| ``SoftmaxOut_v1`` | (x: bf16, m: bf16, r: f32) -> bf16(RN(e * r)) | pending |
| ``Bf16Add_v2`` | the residual add | definition |
| ``EmbeddingCoordinate_v1{V}`` | (tok: u16, col: bf16[V], pos: bf16) -> bf16_rn(f32(col[tok]) + f32(pos)) | pending |
| ``ArgmaxChunkBest_v1{C}``, ``ArgmaxChunkIdx_v1{C}`` | x: bf16[C] -> `TokenSelectChunk_v1`'s (best, index) | pending |
| ``ArgmaxCombine_v1{V, C}`` | (x0: bf16, best: bf16[n], idx: u16[n]) -> `TokenSelectCombine_v1`'s index | pending |

where e_j = ``MufuEx2Ftz``(RN(RN(x_j - m) * RN32(log2 e))). A 32-bit value that one unit outputs as two 16-bit halves (a part's
s32, a scale, rs, a softmax reciprocal) is one 32-bit input word of its readers here: a bundle joins the halves (`join16`).

**Bundles.** `replay` takes a step's drawn units as ``{"units": [{"unit", "template", "inputs": {param: words},
"outputs": words}, ...]}`` (words as unsigned integers of each parameter's width, in its row-major order), groups them by
template, evaluates each group in one batch and reports every unit whose outputs differ. The drawn-unit bundle maps
onto it in `units_of`.

**The draw** (`draw`): the units of a step to replay, from live coins (a key holder's bytes, never a commitment) through
`verity.ml.randomness`, as core's draw object (`one_stage.draw`): ``uniform:K`` (a K-subset of all units), ``stratified:K``
(core's per-template sizes, `draw.stratum_k`) or an explicit plan ``{template: k}`` (proofs' stratified plan).
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage.draw import stratum_k
from verity.ml import randomness
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import mufu as MU
from verity.ml.kernels.tc import f2fp_bf16_batch

np.seterr(all="ignore")

DEFINITION, PENDING = "definition", "pending-definition"
#: the GPU's word for every NaN result of f32 arithmetic (`scalar.F32_NAN_PTX`)
F32_NAN = 0x7FFFFFFF
#: RN32(log2 e), softmax's exponent scale (`ex2.approx` of x * log2 e)
LOG2E = 0x3FB8AA3B
F32_ONE, F32_127 = 0x3F800000, 0x42FE0000
BF16_NEG_INF, BF16_NAN = 0xFF80, 0x7FFF
#: RMSNorm's epsilon, 1e-5 as the nearest f32
EPS_1E5 = 0x3727C5AC
DRAW_DOMAIN = "verity/e2e-h200/replay-draw/v1"


# ---- words -------------------------------------------------------------------------------------------------------------

def _u32(a: Any) -> np.ndarray:
    return np.asarray(a).astype(np.uint32)


def _f(a: np.ndarray) -> np.ndarray:
    return _u32(a).view(np.float32)


def _canon(r: np.ndarray) -> np.ndarray:
    """f32 results as words, every NaN the GPU's 0x7FFFFFFF (`fp32.f32_gpu_nan`)."""
    return np.where(np.isnan(r), np.uint32(F32_NAN), np.asarray(r, np.float32).view(np.uint32))


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`fp32.F32AddV2` on words."""
    return _canon(_f(a) + _f(b))


def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """RN(a - b), every NaN 0x7FFFFFFF."""
    return _canon(_f(a) - _f(b))


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`fp32.F32MulV2` on words."""
    return _canon(_f(a) * _f(b))


def div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`fp32.F32DivV2` on words."""
    return _canon(_f(a) / _f(b))


def sqrt(a: np.ndarray) -> np.ndarray:
    """``PouwF32SqrtRn`` (sqrt.rn.f32) on words: sqrt(-0) = -0, a NaN or a negative nonzero operand 0x7FFFFFFF."""
    return _canon(np.sqrt(_f(a)))


def widen(b: np.ndarray) -> np.ndarray:
    """bf16 words to their f32 words (`prims.Bf16ToF32`)."""
    return np.asarray(b).astype(np.uint32) << np.uint32(16)


def bf16(a: np.ndarray) -> np.ndarray:
    """f32 words to bf16 words by `prims.F2fpBf16` (RNE, every NaN 0x7FFF)."""
    shape = np.shape(a)
    return f2fp_bf16_batch(_u32(a).reshape(-1))[0].reshape(shape)


def ex2(a: np.ndarray) -> np.ndarray:
    """`mufu.MufuEx2Ftz` (ex2.approx.ftz.f32) on words, its registered kernel."""
    a = _u32(a)
    if a.size == 0:
        return a
    return np.asarray(evaluate_batch(MU.MufuEx2Ftz, a.reshape(-1)), dtype=np.uint32).reshape(a.shape)


def seq_sum(words: np.ndarray) -> np.ndarray:
    """(n, m) f32 words -> (n,): RN sums in ascending index from +0 (`F32AddV2` m times)."""
    acc = np.zeros(words.shape[0], np.uint32)
    for j in range(words.shape[1]):
        acc = add(acc, words[:, j])
    return acc


def half(w: np.ndarray, h: int) -> np.ndarray:
    """A 32-bit word's low (h 0) or high (h 1) 16 bits."""
    return ((_u32(w) >> np.uint32(16 * h)) & np.uint32(0xFFFF)).astype(np.uint16)


def join16(lo: Any, hi: Any) -> np.ndarray:
    """The 32-bit word whose halves two units output."""
    return (np.asarray(hi).astype(np.uint32) << np.uint32(16)) | np.asarray(lo).astype(np.uint32)


def _order16(x: np.ndarray) -> np.ndarray:
    """An integer whose order is the bf16 values' (+0 and -0 equal; NaNs not ordered: callers mask them)."""
    x = np.asarray(x).astype(np.int64)
    m = x & 0x7FFF
    return np.where(x & 0x8000, -m, m)


def _is_nan16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(np.int64)
    return ((x & 0x7F80) == 0x7F80) & ((x & 0x7F) != 0)


def _s32(w: np.ndarray) -> np.ndarray:
    return _u32(w).view(np.int32).astype(np.int64)


def _s8(w: np.ndarray) -> np.ndarray:
    return np.asarray(w).astype(np.uint8).view(np.int8).astype(np.int64)


# ---- templates ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Template:
    """One template at one binding: its circuits id ``name`` and statics, its parameters ``(name, width, count)`` (count 0: one
    word, passed to ``fn`` as ``(n,)``; else ``(n, count)``) and output
    ``(width, count)`` in words, ``fn`` its batched replay (one array per parameter, a leading batch axis -> ``(n, count)``
    output words), ``status`` (module docstring), ``source`` where its semantics are stated, and ``definition`` a callable
    returning main's Definition at this binding, or None while it isn't importable."""
    name: str
    statics: tuple[tuple[str, Any], ...]
    params: tuple[tuple[str, int, int], ...]
    out: tuple[int, int]
    fn: Callable[..., np.ndarray] = field(repr=False, compare=False)
    source: str = field(default="", compare=False)
    definition: Callable[[], Any] | None = field(default=None, repr=False, compare=False)

    @property
    def key(self) -> str:
        return key_of(self.name, dict(self.statics))

    @property
    def status(self) -> str:
        return DEFINITION if self.target() is not None else PENDING

    def target(self) -> Any:
        """Main's Definition at this binding, or None."""
        if self.definition is None:
            return None
        try:
            return self.definition()
        except (ImportError, AttributeError):
            return None

    def reference(self, *arrays: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(*arrays)).reshape(len(arrays[0]) if arrays else 0, self.out[1])

    def evaluate(self, *arrays: np.ndarray) -> np.ndarray:
        """The unit outputs: main's Definition when it exists, else the reference; ``(n, count)`` words."""
        arrays = tuple(_as_param(a, p) for a, p in zip(arrays, self.params))
        t = self.target()
        if t is None:
            return self.reference(*arrays)
        n = len(arrays[0])
        return np.asarray(evaluate_batch(t, *arrays)).reshape(n, self.out[1])


def _as_param(a: Any, p: tuple[str, int, int]) -> np.ndarray:
    _, w, c = p
    a = np.asarray(a)
    dt = {8: np.uint8, 16: np.uint16, 32: np.uint32}[w]
    if a.dtype.kind == "i":
        a = a.view(f"u{a.dtype.itemsize}")
    a = a.astype(dt)
    return a.reshape(len(a)) if c == 0 else a.reshape(len(a), c)


def key_of(name: str, statics: Mapping[str, Any]) -> str:
    """``Name{A=1,B=true}``, the statics in their stated order (circuits' descriptor form, DOT by its step's id)."""
    if not statics:
        return name
    parts = []
    for k, v in statics.items():
        v = getattr(v, "id", v)
        parts.append(f"{k}={str(v).lower() if isinstance(v, bool) else v}")
    return f"{name}{{{','.join(parts)}}}"


_REGISTRY: dict[str, Callable[..., Template]] = {}


def _template(name: str) -> Callable[[Callable[..., Template]], Callable[..., Template]]:
    def reg(f: Callable[..., Template]) -> Callable[..., Template]:
        _REGISTRY[name] = f
        return f
    return reg


def _quant_def(attr: str, **statics: Any) -> Callable[[], Any]:
    def get() -> Any:
        from verity.ml.circuits.ops import quant  # circuits' quantizer Definitions, once on main
        return bind(getattr(quant, attr), **statics)
    return get


def _circuits_def(attr: str, **statics: Any) -> Callable[[], Any]:
    """A circuits Definition in `verity.ml.circuits.ops` once some module there exports ``attr``."""
    def get() -> Any:
        import importlib
        for mod in ("verity.ml.circuits.ops.rmsnorm", "verity.ml.circuits.ops.softmax", "verity.ml.circuits.ops.argmax",
                    "verity.ml.circuits.ops.embedding", "verity.ml.circuits.ops.gemm", "verity.ml.circuits.ops.quant"):
            try:
                m = importlib.import_module(mod)
            except ImportError:
                continue
            if hasattr(m, attr):
                return bind(getattr(m, attr), **statics)
        return None
    return get


PLAN = "circuits' plan (af6de1f1), section 1b"


# -- RMSNorm

@_template("RmsSumSqPart_v1")
def rms_sum_sq_part(C: int, HALF: int) -> Template:
    def fn(x):
        sq = mul(widen(x), widen(x))
        return half(seq_sum(sq), HALF)
    return Template("RmsSumSqPart_v1", (("C", C), ("HALF", HALF)), (("x", 16, C),), (16, 1), fn,
                    f"{PLAN}: sum of RN(x_i^2) in ascending i from +0", _circuits_def("RmsSumSqPart", C=C, HALF=HALF))


@_template("F32SumPart_v1")
def f32_sum_part(G: int, HALF: int) -> Template:
    return Template("F32SumPart_v1", (("G", G), ("HALF", HALF)), (("s", 32, G),), (16, 1),
                    lambda s: half(seq_sum(s), HALF), f"{PLAN}: the G chunk sums in ascending order from +0",
                    _circuits_def("F32SumPart", G=G, HALF=HALF))


def rms_recip(s: np.ndarray, N: int, EPS: int) -> np.ndarray:
    """rs from the row's G partial sums: RN(1 / RN(sqrt(RN(RN(sum / N) + eps))))."""
    mean = div(seq_sum(s), np.uint32(np.float32(N).view(np.uint32)))
    return div(np.uint32(F32_ONE), sqrt(add(mean, np.uint32(EPS))))


@_template("RmsNormRecip_v1")
def rms_norm_recip(N: int, G: int, EPS: int, HALF: int) -> Template:
    return Template("RmsNormRecip_v1", (("N", N), ("G", G), ("EPS", EPS), ("HALF", HALF)), (("s", 32, G),), (16, 1),
                    lambda s: half(rms_recip(s, N, EPS), HALF),
                    f"{PLAN}: RMSNormRef_v1's rs, the G group sums from +0", _circuits_def("RmsNormRecip", N=N, G=G, EPS=EPS, HALF=HALF))


def rms_out(x: np.ndarray, rs: np.ndarray, w: np.ndarray) -> np.ndarray:
    return bf16(mul(widen(w), widen(bf16(mul(widen(x), rs)))))


@_template("RmsNormOut_v1")
def rms_norm_out() -> Template:
    return Template("RmsNormOut_v1", (), (("x", 16, 0), ("rs", 32, 0), ("w", 16, 0)), (16, 1), rms_out,
                    f"{PLAN}: h = bf16(RN(x rs)), out = bf16(RN(w h))", _circuits_def("RmsNormOut"))


# -- the quantizer (`ops.quant`)

def relu16(x: np.ndarray, on: bool) -> np.ndarray:
    """``x < 0 ? +0 : x`` on bf16 words (-0 and NaNs are themselves)."""
    x = np.asarray(x).astype(np.uint16)
    if not on:
        return x
    return np.where(~_is_nan16(x) & (_order16(x) < 0), np.uint16(0), x)


def row_abs_max(x: np.ndarray, relu: bool) -> np.ndarray:
    """max |x_k| (relu'd first) as a bf16 word: the max of the words with their sign bits cleared."""
    return (relu16(x, relu) & np.uint16(0x7FFF)).max(axis=1).astype(np.uint16)


@_template("RowAbsMax_v1")
def row_abs_max_t(N: int, RELU: bool) -> Template:
    return Template("RowAbsMax_v1", (("N", N), ("RELU", RELU)), (("x", 16, N),), (16, 1), lambda x: row_abs_max(x, RELU),
                    "ops.quant.RowAbsMax_v1", _quant_def("RowAbsMax", N=N, RELU=RELU))


def quant_scale(amax: np.ndarray) -> np.ndarray:
    """RN(amax / 127) when 1 <= f32(amax) <= +inf as signed words, else 1.0 (`ops.quant`'s ``_scale``)."""
    a = widen(amax)
    pos = (a.view(np.int32) >= 1) & (a.view(np.int32) <= 0x7F800000)
    return np.where(pos, div(a, np.uint32(F32_127)), np.uint32(F32_ONE)).astype(np.uint32)


def to_s8_rni_sat(w: np.ndarray) -> np.ndarray:
    """`scalar.F32ToS8RniSat_v1` (cvt.rni.sat.s8.f32): ties to even, saturated, NaN 0; s8 words."""
    v = _f(w)
    r = np.where(np.isnan(v), np.float32(0), np.clip(np.rint(v), -128, 127))
    return r.astype(np.int8).view(np.uint8)


@_template("QuantS8Pair_v1")
def quant_s8_pair(RELU: bool) -> Template:
    def fn(x, amax):
        return to_s8_rni_sat(div(widen(relu16(x, RELU)), quant_scale(amax)[:, None]))
    return Template("QuantS8Pair_v1", (("RELU", RELU),), (("x", 16, 2), ("amax", 16, 0)), (8, 2), fn,
                    "ops.quant.QuantS8Pair_v1", _quant_def("QuantS8Pair", RELU=RELU))


@_template("QuantS8Scale_v1")
def quant_s8_scale(HALF: int) -> Template:
    return Template("QuantS8Scale_v1", (("HALF", HALF),), (("amax", 16, 0),), (16, 1), lambda a: half(quant_scale(a), HALF),
                    "ops.quant.QuantS8Scale_v1", _quant_def("QuantS8Scale", HALF=HALF))


# -- the static matmuls, K-split

@_template("S8AccumulatorPart_v1")
def s8_accumulator_part(K: int, HALF: int) -> Template:
    def fn(x, w):
        return half((_s8(x) * _s8(w)).sum(axis=1) & 0xFFFFFFFF, HALF)
    return Template("S8AccumulatorPart_v1", (("K", K), ("HALF", HALF)), (("x", 8, K), ("wrow", 8, K)), (16, 1), fn,
                    f"{PLAN}: GemmAccumulatorS8_v1{{K}}'s s32 (exact below K = 2^17), one 16-bit half",
                    _circuits_def("S8AccumulatorPart", K=K, HALF=HALF))


def s32_sum(parts: np.ndarray) -> np.ndarray:
    return (_s32(parts).sum(axis=1) & 0xFFFFFFFF).astype(np.uint32)


@_template("S32SumPart_v1")
def s32_sum_part(P: int, HALF: int) -> Template:
    return Template("S32SumPart_v1", (("P", P), ("HALF", HALF)), (("parts", 32, P),), (16, 1),
                    lambda p: half(s32_sum(p), HALF), f"{PLAN}: an exact s32 adder tree (|sum| < 2^31 at the frozen shape)",
                    _circuits_def("S32SumPart", P=P, HALF=HALF))


def epilogue(parts: np.ndarray, sx: np.ndarray, sw: np.ndarray) -> np.ndarray:
    acc = _s32(s32_sum(parts)).astype(np.int32).astype(np.float32).view(np.uint32)
    return bf16(mul(mul(acc, sx), sw))


@_template("ScaledMmS8Epilogue_v1")
def scaled_mm_s8_epilogue(P: int) -> Template:
    return Template("ScaledMmS8Epilogue_v1", (("P", P),), (("parts", 32, P), ("sx", 32, 0), ("sw", 32, 0)), (16, 1), epilogue,
                    f"{PLAN}: cvt.rn.bf16.f32((cvt.rn.f32.s32(sum) * sx) * sw), ScaledMmS8Coordinate_v1's epilogue",
                    _circuits_def("ScaledMmS8Epilogue", P=P))


@_template("ScaledMmS8Coordinate_v1")
def scaled_mm_s8_coordinate(K: int, DOT: Any = G.GEMM_SM90_S8["DOT"]) -> Template:
    def fn(x, sx, w, sw):
        acc = (_s8(x) * _s8(w)).sum(axis=1).astype(np.int32).astype(np.float32).view(np.uint32)
        return bf16(mul(mul(acc, sx), sw))
    return Template("ScaledMmS8Coordinate_v1", (("K", K), ("DOT", DOT)), (("x", 8, K), ("sx", 32, 0), ("wrow", 8, K), ("sw", 32, 0)),
                    (16, 1), fn, "verity.ml.circuits.ops.gemm.ScaledMmS8Coordinate_v1",
                    lambda: bind(G.ScaledMmS8Coordinate, K=K, DOT=DOT))


# -- attention

@_template("GemmCoordinate_v2")
def gemm_coordinate(K: int, DOT: Any = G.GEMM_HOPPER["DOT"]) -> Template:
    t = bind(G.GemmCoordinate, K=K, DOT=DOT)
    return Template("GemmCoordinate_v2", (("K", K), ("DOT", DOT)), (("x", 16, K), ("wrow", 16, K)), (16, 1),
                    lambda x, w: evaluate_batch(t, x, w), "verity.ml.circuits.ops.gemm.GemmCoordinate_v2", lambda: t)


@_template("GemmCoordinatePad_v1")
def gemm_coordinate_pad(T: int, K: int, DOT: Any = G.GEMM_HOPPER["DOT"]) -> Template:
    if not 1 <= T <= K:
        raise ValueError(f"GemmCoordinatePad_v1 pads T = {T} operands to K = {K} >= T")
    inner = bind(G.GemmCoordinate, K=K, DOT=DOT)

    def fn(x, w):
        pad = np.zeros((len(x), K - T), np.uint16)
        return evaluate_batch(inner, np.concatenate([x, pad], 1), np.concatenate([w, pad], 1))
    return Template("GemmCoordinatePad_v1", (("T", T), ("K", K), ("DOT", DOT)), (("x", 16, T), ("wrow", 16, T)), (16, 1), fn,
                    f"{PLAN}: T operands padded to K with +0, then GemmCoordinate_v2{{K}}",
                    _circuits_def("GemmCoordinatePad", T=T, K=K, DOT=DOT))


def softmax_max(x: np.ndarray) -> np.ndarray:
    """The first greatest bf16 word of each row (+0 = -0), 0x7FFF where any is NaN."""
    j = np.argmax(_order16(x), axis=1)
    m = np.asarray(x)[np.arange(len(x)), j].astype(np.uint16)
    return np.where(_is_nan16(x).any(axis=1), np.uint16(BF16_NAN), m)


def softmax_exp(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """e = MufuEx2Ftz(RN(RN(x - m) * RN32(log2 e))), x (n, N) or (n,) bf16 and m (n,) bf16."""
    mm = widen(m) if np.ndim(x) == 1 else widen(m)[:, None]
    return ex2(mul(sub(widen(x), mm), np.uint32(LOG2E)))


def softmax_recip(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    return div(np.uint32(F32_ONE), seq_sum(softmax_exp(x, m)))


@_template("SoftmaxMax_v1")
def softmax_max_t(N: int) -> Template:
    return Template("SoftmaxMax_v1", (("N", N),), (("x", 16, N),), (16, 1), softmax_max,
                    f"{PLAN} section 1: SoftmaxRef_v1's m", _circuits_def("SoftmaxMax", N=N))


@_template("SoftmaxRecip_v1")
def softmax_recip_t(N: int, HALF: int) -> Template:
    return Template("SoftmaxRecip_v1", (("N", N), ("HALF", HALF)), (("x", 16, N), ("m", 16, 0)), (16, 1),
                    lambda x, m: half(softmax_recip(x, m), HALF), f"{PLAN} section 1: r = RN(1 / sum_j e_j) from +0",
                    _circuits_def("SoftmaxRecip", N=N, HALF=HALF))


@_template("SoftmaxOut_v1")
def softmax_out_t() -> Template:
    return Template("SoftmaxOut_v1", (), (("x", 16, 0), ("m", 16, 0), ("r", 32, 0)), (16, 1),
                    lambda x, m, r: bf16(mul(softmax_exp(x, m), r)), f"{PLAN} section 1: p_j = bf16(RN(e_j r))",
                    _circuits_def("SoftmaxOut"))


# -- residual, embedding, argmax

def bf16_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`fp32.Bf16Add_v1`'s function (``Bf16Add_v2``'s): bf16_rn(f32(a) + f32(b)), NaN 0x7FC0."""
    s = add(widen(a), widen(b))
    return np.where((s & 0x7FFFFFFF) > 0x7F800000, np.uint16(0x7FC0), bf16(s)).astype(np.uint16)


@_template("Bf16Add_v2")
def bf16_add_t() -> Template:
    def get() -> Any:
        from verity.ml.circuits.ops import fp32
        return fp32.Bf16Add
    return Template("Bf16Add_v2", (), (("a", 16, 0), ("b", 16, 0)), (16, 1), bf16_add,
                    "verity.ml.circuits.ops.boolean.elementwise.Bf16Add_v2, evaluated by its word view fp32.Bf16Add_v1", get)


@_template("EmbeddingCoordinate_v1")
def embedding_coordinate(V: int) -> Template:
    def fn(tok, col, pos):
        t = np.asarray(tok).astype(np.int64)
        return bf16_add(np.asarray(col)[np.arange(len(col)), np.clip(t, 0, V - 1)], pos)
    return Template("EmbeddingCoordinate_v1", (("V", V),), (("tok", 16, 0), ("col", 16, V), ("pos", 16, 0)), (16, 1), fn,
                    f"{PLAN} section 1: col[tok] + pos, Bf16Add's rounding (tok < V)", _circuits_def("EmbeddingCoordinate", V=V))


def chunk_best(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`TokenSelectChunk_v1`: from (-inf, 0), x_i is taken iff it is strictly above the best (NaN never): (best, index)."""
    o = np.where(_is_nan16(x), np.iinfo(np.int64).min, _order16(x))
    j = np.argmax(o, axis=1)
    top = o[np.arange(len(x)), j]
    some = top > _order16(np.uint16(BF16_NEG_INF))
    best = np.where(some, np.asarray(x)[np.arange(len(x)), j], np.uint16(BF16_NEG_INF)).astype(np.uint16)
    return best, np.where(some, j, 0).astype(np.uint16)


@_template("ArgmaxChunkBest_v1")
def argmax_chunk_best(C: int) -> Template:
    return Template("ArgmaxChunkBest_v1", (("C", C),), (("x", 16, C),), (16, 1), lambda x: chunk_best(x)[0],
                    f"{PLAN} section 1: TokenSelectChunk_v1's best", _circuits_def("ArgmaxChunkBest", C=C))


@_template("ArgmaxChunkIdx_v1")
def argmax_chunk_idx(C: int) -> Template:
    return Template("ArgmaxChunkIdx_v1", (("C", C),), (("x", 16, C),), (16, 1), lambda x: chunk_best(x)[1],
                    f"{PLAN} section 1: TokenSelectChunk_v1's index in the chunk", _circuits_def("ArgmaxChunkIdx", C=C))


def argmax_combine(x0: np.ndarray, best: np.ndarray, idx: np.ndarray, C: int) -> np.ndarray:
    """`TokenSelectCombine_v1`: from (x0, 0), chunk c's (best[c], cC + idx[c]) is taken iff best[c] > the running best."""
    run, run_i = np.asarray(x0).astype(np.uint16), np.zeros(len(x0), np.int64)
    for c in range(best.shape[1]):
        b = best[:, c].astype(np.uint16)
        gt = ~_is_nan16(b) & ~_is_nan16(run) & (_order16(b) > _order16(run))
        run_i = np.where(gt, c * C + idx[:, c].astype(np.int64), run_i)
        run = np.where(gt, b, run)
    return (run_i & 0xFFFF).astype(np.uint16)


@_template("ArgmaxCombine_v1")
def argmax_combine_t(V: int, C: int) -> Template:
    n = -(-V // C)
    return Template("ArgmaxCombine_v1", (("V", V), ("C", C)), (("x0", 16, 0), ("best", 16, n), ("idx", 16, n)), (16, 1),
                    lambda x0, b, i: argmax_combine(x0, b, i, C), f"{PLAN} section 1: TokenSelectCombine_v1",
                    _circuits_def("ArgmaxCombine", V=V, C=C))


# ---- keys --------------------------------------------------------------------------------------------------------------

_DOTS = {getattr(d["DOT"], "id"): d["DOT"] for d in (G.GEMM_HOPPER, G.GEMM_SM90_S8, G.GEMM_AMPERE)}
_KEY = re.compile(r"([A-Za-z0-9]+_v[0-9]+)(?:\{(.*)\})?")


def _static(v: str) -> Any:
    if v in ("true", "false"):
        return v == "true"
    if v in _DOTS:
        return _DOTS[v]
    return int(v)


@lru_cache(maxsize=None)
def template(key: str) -> Template:
    """The template at a key ``Name{A=1,...}`` (`key_of`); a DOT by its step's id."""
    m = _KEY.fullmatch(key)
    if not m or m.group(1) not in _REGISTRY:
        raise KeyError(f"no replay for template {key!r}: one of {sorted(_REGISTRY)}")
    statics = dict(kv.split("=", 1) for kv in m.group(2).split(",")) if m.group(2) else {}
    t = _REGISTRY[m.group(1)](**{k: _static(v) for k, v in statics.items()})
    if t.key != key:
        raise KeyError(f"{key!r} is not in its canonical form {t.key!r}")
    return t


def templates() -> list[str]:
    return sorted(_REGISTRY)


# ---- replay ------------------------------------------------------------------------------------------------------------

def _row(words: Any, p: tuple[str, int, int]) -> np.ndarray:
    a = np.asarray(words, dtype=np.uint64).reshape(-1)
    if a.size != max(p[2], 1) or (a.size and int(a.max()) >> p[1]):
        raise ValueError(f"parameter {p[0]}: {a.size} words of {p[1]} bits, not {max(p[2], 1)}")
    return a


def replay(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Replay a step's drawn units (module docstring): ``{"units", "matched", "mismatched": [{"unit", "template", "got",
    "want"}], "malformed": [{"unit", "template", "error"}], "templates": {key: {"status", "units", "mismatched"}}}``. A unit
    is matched only when its committed outputs are its template's outputs on its opened inputs, word for word."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    report: dict[str, Any] = {"units": len(units), "matched": 0, "mismatched": [], "malformed": [], "templates": {}}
    for u in units:
        groups.setdefault(u["template"], []).append(u)
    for key, us in groups.items():
        entry = report["templates"].setdefault(key, {"status": None, "units": len(us), "mismatched": 0})
        try:
            t = template(key)
        except (KeyError, ValueError) as e:
            report["malformed"] += [{"unit": u["unit"], "template": key, "error": str(e)} for u in us]
            continue
        entry["status"] = t.status
        good, cols, outs = [], [[] for _ in t.params], []
        for u in us:
            try:
                ins = u["inputs"]
                rows = [_row(ins[p[0]], p) for p in t.params]
                out = _row(u["outputs"], ("outputs", t.out[0], t.out[1]))
            except (KeyError, ValueError, TypeError) as e:
                report["malformed"].append({"unit": u["unit"], "template": key, "error": str(e)})
                continue
            good.append(u)
            for c, r in zip(cols, rows):
                c.append(r)
            outs.append(out)
        if not good:
            continue
        want = t.evaluate(*(np.stack(c) for c in cols)).astype(np.uint64).reshape(len(good), t.out[1])
        got = np.stack(outs).astype(np.uint64)
        for u, g, w in zip(good, got, want):
            if np.array_equal(g, w):
                report["matched"] += 1
            else:
                entry["mismatched"] += 1
                report["mismatched"].append({"unit": u["unit"], "template": key, "got": g.tolist(), "want": w.tolist()})
    return report


def ok(report: Mapping[str, Any]) -> bool:
    return report["matched"] == report["units"] and not report["mismatched"] and not report["malformed"]


IFACE_B = "verity/drawn-unit-bundle/v1"


def key_at(definition: str, binding: Mapping[str, Any]) -> str:
    """The key of ``definition`` at a drawn-unit bundle's binding (statics as JSON, a DOT as ``{"fn": id}``)."""
    if definition not in _REGISTRY:
        raise KeyError(f"no replay for template {definition!r}: one of {sorted(_REGISTRY)}")
    statics = {k: _DOTS[v["fn"]] if isinstance(v, Mapping) else v for k, v in binding.items()}
    return _REGISTRY[definition](**statics).key


def _unpack(row: bytes, width: int) -> list[int]:
    return np.frombuffer(row, dtype=f"<u{width // 8}").astype(np.uint64).tolist()


def units_of(bundle: Mapping[str, Any], payload: bytes | None = None) -> list[dict[str, Any]]:
    """A bundle's units in `replay`'s form: ``{"units": [...]}`` already in it, or a drawn-unit bundle
    (``verity/drawn-unit-bundle/v1``) with its ``payload``: each port's row, u16 words little-endian, read as words of its
    parameter's width, the output rows joined in port order. Openings are `replay_bundle.check`'s, not this."""
    if bundle.get("format") != IFACE_B:
        if "units" not in bundle:
            raise ValueError("not a replay bundle: no 'units'")
        return list(bundle["units"])
    if payload is None:
        raise ValueError(f"a {IFACE_B} bundle's rows are in its payload")
    out = []
    for u in bundle["units"]:
        tpl = u["template"]
        try:
            t = template(key_at(tpl["definition"], tpl.get("binding", {})))
        except (KeyError, ValueError, TypeError):
            out.append({"unit": u["unit"], "template": tpl["id"], "coords": u.get("coords"), "inputs": {}, "outputs": []})
            continue
        rows = {r["port"]: (r["kind"], payload[r["row"][0]:r["row"][0] + r["row"][1]]) for r in u["rows"]}
        width = dict((p[0], p[1]) for p in t.params)
        ins = {p: _unpack(row, width.get(p, 16)) for p, (kind, row) in rows.items() if kind == "in"}
        outs = b"".join(row for kind, row in rows.values() if kind == "out")
        out.append({"unit": u["unit"], "template": t.key, "coords": u.get("coords"), "inputs": ins,
                    "outputs": _unpack(outs, t.out[0])})
    return out


# ---- the draw ----------------------------------------------------------------------------------------------------------

def draw(population: Sequence[tuple[str, int]], plan: str | Mapping[str, int], coins: bytes,
         context: Mapping[str, randomness.Part]) -> dict[str, Any]:
    """The units of one step to replay. ``population``: each template's unit count in the step's unit order (template-major:
    global unit = the template's base + its index); ``plan``: ``uniform:K``, ``stratified:K`` or ``{template: k}``;
    ``coins``: the challenger's live bytes; ``context``: what the draw must follow (the step's committed root, at least).
    -> core's draw object: ``{"law", "population", "k", ["strata"], "units"}``, units ascending; an explicit plan's law is
    ``plan`` (its strata's k are the plan's, not core's `stratum_k`)."""
    n = sum(c for _, c in population)
    key = randomness.derive(coins, DRAW_DOMAIN, context)
    if isinstance(plan, str):
        kind, _, arg = plan.partition(":")
        if kind not in ("uniform", "stratified") or not arg.isdigit() or not 1 <= int(arg) <= n:
            raise ValueError(f"a plan is uniform:K or stratified:K with 1 <= K <= {n}, or {{template: k}}, not {plan!r}")
        k = int(arg)
        if kind == "uniform":
            return {"law": "subset", "population": n, "k": k, "units": list(key.subset(k, n, "uniform"))}
        sizes = {t: stratum_k(k, c, n) for t, c in population}
    else:
        unknown = set(plan) - {t for t, _ in population}
        if unknown:
            raise ValueError(f"the plan names templates the step has no units of: {sorted(unknown)}")
        sizes, k = {t: int(plan.get(t, 0)) for t, _ in population}, sum(int(v) for v in plan.values())
    strata, units, base = [], [], 0
    for s, (t, c) in enumerate(population):
        ks = sizes[t]
        if not 0 <= ks <= c:
            raise ValueError(f"{t}: k = {ks} of its {c} units")
        picked = key.subset(ks, c, ("stratum", s))
        strata.append({"template": t, "units": [[base, base + c]], "k": ks})
        units += [base + i for i in picked]
        base += c
    return {"law": "stratified" if isinstance(plan, str) else "plan", "population": n, "k": k, "strata": strata,
            "units": sorted(units)}
