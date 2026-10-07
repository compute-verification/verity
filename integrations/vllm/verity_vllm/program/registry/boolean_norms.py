"""The served RMSNorms on bits: `RMSNormFusedCuda_v2` and `RMSNormTriton_v1` as Boolean Definitions, cut into Calls a root can hold.

    RMSNormFusedCuda_v3{N, EPS, RSQRT}   (x, res, w: Array<N, Array<16, Value<1>>>) -> (normed, res')    word view: RMSNormFusedCuda_v2{N, EPS}
    RMSNormTriton_v2{N, EPS, SQRT, SCALEA, RCP}
                                         (x, w: Array<N, Array<16, Value<1>>>) -> Array<N, Array<16, ...>>  word view: RMSNormTriton_v1{N, EPS}

Each is the word body's dataflow, in the word body's operand order, over these parts (`rmsnorm_cuda_calls`, `rmsnorm_triton_calls`):

    RMSNormLanesCuda_v1{N, BLOCK}  per thread t < T = min(N, BLOCK): r = Bf16AddF2fp_v2(x, res) for its elements t, t + T, ...
                                   (`RMSNormLaneCuda_v1{J}`), partial = square of the first, then FFMA of each next square
    RMSNormWarpsCuda_v1{T}         per warp, cub's shfl_down tree 1, 2, 4, 8, 16 over the valid lanes (`RMSNormWarpCuda_v1{L}`)
    RMSNormArgCuda_v1{N, EPS, NW}  the warp sums added from warp 0, then `RMSNormVarEpsCuda_v2{N, EPS}`: `/ N` (div.rn), `+ EPS`
    RSQRT                          `rsqrtf` (`RsqrtApprox_v1`'s function)
    RMSNormScaleCuda_v1            s's own gates of the output word, once
    RMSNormOutRowCuda_v1{N}        bf16_rn((f32(r_j) * s) * f32(w_j))   (`RMSNormOutCuda_v1`)

    RMSNormThreadsTriton_v1{N}     per thread of 8 columns: x1 * x1, then FFMA x0, x2, ..., x7  (`RMSNormThreadTriton_v1`)
    RMSNormWarpsTriton_v1{N}       per warp, the xor butterfly 16, 8, 4, 2, 1, lane 0's cone (`RMSNormWarpTriton_v1{L}`)
    RMSNormMeanEpsTriton_v1{N, EPS}  per chunk of 1024 columns (w0 + w2) + (w1 + w3), chunks added in order, then
                                   `RMSNormVarEpsTriton_v2{N, EPS}`: FFMA(rcp(N), sum, EPS)
    SQRT, SCALEA(1.0, .), RCP      `sqrt.approx.ftz` and `div.full`'s two MUFU pieces; `F32Mul_v3` of the last two is `inv`
    RMSNormInvTriton_v1            inv's own gates of the output word, once
    RMSNormOutRowTriton_v1{N}      bf16_rn((inv * f32(x_j)) * f32(w_j))   (`RMSNormOutTriton_v1`)

The leaf pieces are traced from `fp`'s builders, so the bf16 operands' 16 constant-zero bits and the constants N, 1 / N and EPS
specialize them. The f32 adds, divide, multiply and FMA between them are `verity.ml.circuits.ops.boolean.elementwise`'s Calls. The lanes,
warps and output rows are separable (each node reads only parameters and every leaf it makes is returned), so `Q_word` cuts them
member by member; the norm itself is not, so a served Program calls the parts at its root (the two emitters above), never the
whole norm, whose flat circuit is millions of gates at a served width.

No value is computed twice. The output word `bf16_rn((a * b) * f32(w))` (`_out_cuda`, `_out_triton`) is traced once and its gates
split by what they read (`_split`): those reading only the shared scale (s, inv) are `RMSNormScaleCuda_v1` / `RMSNormInvTriton_v1`,
one Call per row; those reading only the element (r_j, x_j) are its decode, which the square that already reads that element
returns beside its sum (`RMSNormSquare_v1`, `RMSNormSquareFma_v1`, `RMSNormThreadTriton_v1`: the decode's ANDs are the square's
own, shared through `forms`' CSE); the rest are the output piece, which reads both decodes. So the row's members share no gate,
and Triton's output pass recomputes nothing of the thread pass that reads the same x.

MUFU: ``RSQRT``, ``SQRT``, ``SCALEA`` and ``RCP`` are statics: Definitions on bits whose word Definitions are `RsqrtApprox_v1`,
`MufuSqrtFtz_v1`, `DivFullScaleA_v1` (or `_v2`, equal on the numerator 1.0) and `DivFullRcp_v1`, or, until proofs' Boolean MUFU
Definitions exist, those word ids themselves (`MUFU_WORD`). A word id is called on words: the norm packs a value into its word
(`PackF32_v1`) and unpacks a word into bits (`UnpackF32_v1`) once each, when a Call first needs that form (`_F32`), so Triton's
rms goes from `MufuSqrtFtz_v1` to both division pieces as one word. Those Calls and the word ids are the only word gates left
in a norm. `rcp(N)` is a constant of the specialization, folded when it is built (`DivFullRcp_v1` of the word N), as is every
other constant.

Triton's idle lanes and warps (columns past N) hold +0, and the word body adds them. A lane's partial is never -0 and never a
signalling NaN (a square's sign is clear, an FFMA's NaN is 0x7FC00000, an FADD quiets its NaN), so adding such a +0 returns the
other operand unchanged under `F32Add_v1`: those adds are not emitted.
"""
from __future__ import annotations

from functools import cache
from typing import Callable

import numpy as np

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind, composite
from verity.ml.circuits.refs import Coll, Explicit, array_of, coll_concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import NOT, ONE, ZERO
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.ops.prims import const
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.sampling import bit_at

__all__ = ["CUDA_WORD", "MUFU_WORD", "PackF32", "RMSNormFusedCuda", "RMSNormTriton", "TRITON_WORD", "UnpackF32",
           "rmsnorm_cuda_calls", "rmsnorm_triton_calls"]

BIT = BL.BIT
F32 = Array(32, BIT)
BF16 = Array(16, BIT)
F32W, BF16W = Value(32), Value(16)

#: lanes per warp, columns per Triton thread, Triton threads per chunk
WARP, TRITON_COLS, TRITON_THREADS = 32, 8, 128


def _f32(x: float) -> int:
    return P.bits_of(np.float32(x))


def _bits(n: int) -> Array:
    return Array(n, BIT)


class _Consts:
    """Constant words in one body, from the two constant bits (each emitted once, when first read)."""

    def __init__(self, B) -> None:
        self.B, self.bit = B, {}

    def __call__(self, value: int, width: int = 32) -> Coll:
        refs = []
        for j in range(width):
            b = (value >> j) & 1
            if b not in self.bit:
                self.bit[b] = self.B.call(BL.constant(b)).refs[0]
            refs.append(self.bit[b])
        return Coll(Array(width, BIT), Explicit(tuple(refs)))


# --- gate circuits: traced once, split by what they read, laid out as `trace.definition` lays out its own -----------------------

def _sub(T: TR.Traced, keep: list[int], inputs: list[int], outputs: list[int]) -> TR.Traced:
    """The gates ``keep`` of ``T`` (indices, ascending) as a circuit of their own: its inputs the wires ``inputs`` of ``T`` in order,
    its outputs the wires ``outputs``. Every operand of a kept gate is a constant, one of ``inputs`` or a kept gate."""
    new = {0: 0, 1: 1, **{w: 2 + k for k, w in enumerate(inputs)}}
    gates: list[tuple[str, tuple[int, ...]]] = []
    for i in keep:
        kind, ops = T.gates[i]
        gates.append((kind, tuple(new[o] for o in ops)))
        new[T.first_gate + i] = 2 + len(inputs) + len(gates) - 1
    counts = {k: sum(1 for g, _ in gates if g == k) for k in ("and", "xor", "not")}
    return TR.Traced(len(inputs), gates, [new[o] for o in outputs], counts)


def _split(T: TR.Traced, widths: list[int], shared: int, own: int | None) -> tuple[TR.Traced, TR.Traced, TR.Traced]:
    """``T``'s gates by the parameters they read (``widths``: bits per parameter): those reading only parameter ``shared``, those
    reading only parameter ``own`` (none when ``own`` is None), the rest. Returns the first two as circuits from their
    parameter's bits to their frontier (their gates the rest read), and the rest as a circuit from the parameters, each of the
    two followed by its frontier, to ``T``'s outputs."""
    starts = [2 + sum(widths[:p]) for p in range(len(widths))]
    wires = [list(range(s, s + n)) for s, n in zip(starts, widths)]
    sup = [0, 0] + [1 << p for p, n in enumerate(widths) for _ in range(n)]
    for _, ops in T.gates:
        m = 0
        for o in ops:
            m |= sup[o]
        sup.append(m)
    first, n = T.first_gate, len(T.gates)

    def part(p: int) -> tuple[list[int], list[int]]:
        keep = [i for i in range(n) if sup[first + i] == 1 << p]
        mine = {first + i for i in keep}
        read = {o for i, (_, ops) in enumerate(T.gates) if sup[first + i] != 1 << p for o in ops if o in mine}
        return keep, sorted(read | (mine & set(T.outputs)))

    (ks, fs), (ko, fo) = part(shared), part(own) if own is not None else ([], [])
    inputs = [w for p in range(len(widths)) for w in wires[p] + (fs if p == shared else fo if p == own else [])]
    drop = set(ks) | set(ko)
    rest = _sub(T, [i for i in range(n) if i not in drop], inputs, T.outputs)
    return _sub(T, ks, wires[shared], fs), _sub(T, ko, wires[own] if own is not None else [], fo), rest


def _replay(T: TR.Traced, C, bits: list) -> list:
    """``T``'s outputs over the forms ``bits`` (its inputs, in order), its ANDs made by ``C`` (so CSE shares those ``C`` has)."""
    w = [ZERO, ONE] + list(bits)
    for kind, ops in T.gates:
        a = w[ops[0]]
        w.append(C.AND(a, w[ops[1]]) if kind == "and" else a ^ w[ops[1]] if kind == "xor" else NOT(a))
    return [w[o] for o in T.outputs]


def _traced(name: str, version: int, statics: tuple[str, ...], sig: Callable, make: Callable[..., TR.Traced], doc: str,
            word=None) -> CompositeDefinition:
    """The Boolean Definition whose body, for the bindings ``S``, is the gate circuit ``make(S)`` (made once per binding)."""
    made: dict[tuple, TR.Traced] = {}

    def traced(S) -> TR.Traced:
        key = tuple(getattr(S, k) for k in statics)
        if key not in made:
            made[key] = make(S)
        return made[key]

    def body(B, S, *colls):
        params, ret = sig(S)
        return TR.emit(traced(S), [t for _, t in params], ret, B, *colls)

    d = CompositeDefinition(name, version, statics, sig, body, doc=doc, conformance="exact-model-tested")
    d.traced = traced  # type: ignore[attr-defined]
    d.word = word  # type: ignore[attr-defined]
    return d


# --- the MUFU seam ------------------------------------------------------------------------------------------------------------

@composite("PackF32", 1, [], lambda S: ((("a", F32),), F32W), conformance="exact-model-tested")
def PackF32(B, S, a):
    """The word of the 32 bits ``a`` (bit 0 first): `SelectI32_v1` of each bit's place value, summed (`I32Add_v1`)."""
    zero, acc = B.call(const(32, 0)), None
    for j in range(32):
        v = B.call(P.SelectI32, a[j], B.call(const(32, 1 << j)), zero)
        acc = v if acc is None else B.call(P.I32Add, acc, v)
    return acc


@composite("UnpackF32", 1, [], lambda S: ((("v", F32W),), F32), conformance="exact-model-tested")
def UnpackF32(B, S, v):
    """The 32 bits of the word ``v``, bit 0 first (`BitAtx32_v1`)."""
    return array_of([B.call(bit_at(32), v, B.call(const(32, j))) for j in range(32)])


#: the norms' MUFU statics at their word ids, until proofs' Boolean MUFU Definitions replace them
MUFU_WORD: dict = {"RSQRT": P.RsqrtApprox, "SQRT": P.MufuSqrtFtz, "SCALEA": P.DivFullScaleA, "RCP": P.DivFullRcp}
CUDA_WORD: dict = {"RSQRT": MUFU_WORD["RSQRT"]}
TRITON_WORD: dict = {k: MUFU_WORD[k] for k in ("SQRT", "SCALEA", "RCP")}

#: the word Definition each MUFU static must compute (DivFullScaleA's _v1 and _v2 differ only on a NaN numerator; here it is 1.0)
_MUFU_WORDS = {"RSQRT": {P.RsqrtApprox.id}, "SQRT": {P.MufuSqrtFtz.id}, "RCP": {P.DivFullRcp.id},
               "SCALEA": {P.DivFullScaleA.id, P.DivFullScaleAV2.id}}


def _on_word(d) -> bool:
    """Whether the MUFU static ``d`` is still a word id (it returns a 32-bit word, not 32 bits)."""
    return (bind(d) if isinstance(d, CompositeDefinition) else d).signature()[1] == F32W


def _mufu_word(d):
    return d if _on_word(d) else word_of(d)


def _check_mufu(b: dict, keys) -> None:
    for k in keys:
        got = _mufu_word(b[k]).id
        if got not in _MUFU_WORDS[k]:
            raise ValueError(f"{k} computes {got}, not {' or '.join(sorted(_MUFU_WORDS[k]))}")


def _word_view(word, statics: tuple[str, ...], mufu: tuple[str, ...] = ()) -> Callable[[dict], object]:
    """The word view of a binding ``b``: ``word`` bound at ``b``'s ``statics``, once each MUFU static in ``mufu`` is checked."""
    def view(b: dict):
        _check_mufu(b, mufu)
        return bind(word, **{k: b[k] for k in statics})
    return view


def _on_bits(name: str, version: int, statics: list[str], signature: Callable, word: Callable[[dict], object],
             word_statics: dict | None = None):
    """Decorator: the Boolean composite ``name_v{version}``, documented by its body's docstring, whose word view
    (`trace.word_of`) is ``word`` of its bindings. ``word_statics`` names, for each MUFU static, the word Definition it
    computes; the switch (`verity_vllm.program.boolean`) binds that static to the Definition's Boolean version."""
    def deco(body: Callable) -> CompositeDefinition:
        d = CompositeDefinition(name, version, tuple(statics), signature, body, doc=body.__doc__ or "",
                                conformance="exact-model-tested")
        d.word = word  # type: ignore[attr-defined]
        if word_statics is not None:
            d.word_statics = word_statics  # type: ignore[attr-defined]
        return d
    return deco


class _F32:
    """An f32 of a norm's body as bits, as a word, or both; each form made at most once (`UnpackF32_v1`, `PackF32_v1`, or a
    constant's own Calls), so a value two MUFU word Calls read is packed once."""

    def __init__(self, B, bits: Coll | None = None, word: Coll | None = None, value: int | None = None) -> None:
        self.B, self._bits, self._word, self.value = B, bits, word, value

    def bits(self) -> Coll:
        if self._bits is None:
            self._bits = _Consts(self.B)(self.value) if self.value is not None else self.B.call(UnpackF32, self._word)
        return self._bits

    def word(self) -> Coll:
        if self._word is None:
            self._word = self.B.call(const(32, self.value)) if self.value is not None else self.B.call(PackF32, self._bits)
        return self._word


def _mufu(B, d, *args: _F32) -> _F32:
    """One Call of the MUFU static ``d`` on ``args``: on their words while ``d`` is a word id, on their bits once it is Boolean."""
    if _on_word(d):
        return _F32(B, word=B.call(d, *(a.word() for a in args)))
    return _F32(B, bits=B.call(d, *(a.bits() for a in args)))


# --- the output word, split: the scale's own gates, the element's own gates (its decode), the rest --------------------------------

def _out_cuda(C, z, s, w):
    y = fp.f32_mul(C, fp.f32_mul(C, fp.bf16_to_f32(z), s, ir_nan=True), fp.bf16_to_f32(w), ir_nan=True)
    return fp.f32_to_bf16(C, y, W.BF16_NAN_F2FP)


def _out_triton(C, inv, x, w):
    y = fp.f32_mul(C, fp.f32_mul(C, inv, fp.bf16_to_f32(x), ir_nan=True), fp.bf16_to_f32(w), ir_nan=True)
    return fp.f32_to_bf16(C, y, W.BF16_NAN_F2FP)


@cache
def _out_parts(kind: str) -> tuple[TR.Traced, TR.Traced, TR.Traced]:
    """(scale part, element decode, rest) of the output word: CUDA's over (z, s, w), Triton's over (inv, x, w)."""
    if kind == "cuda":
        return _split(TR.trace(_out_cuda, [BF16, F32, BF16], BF16), [16, 32, 16], shared=1, own=0)
    return _split(TR.trace(_out_triton, [F32, BF16, BF16], BF16), [32, 16, 16], shared=0, own=1)


def _scale_bits(kind: str) -> int:
    return len(_out_parts(kind)[0].outputs)


def _dec_bits(kind: str) -> int:
    return len(_out_parts(kind)[1].outputs)


def _scale_sig(kind: str, name: str, t):
    return lambda S: (((name, t),), _bits(_scale_bits(kind)))


RMSNormScaleCuda = _traced("RMSNormScaleCuda", 1, (), _scale_sig("cuda", "s", F32), lambda S: _out_parts("cuda")[0],
                           "the gates of bf16_rn((f32(z) * s) * f32(w)) that read only s (`_out_cuda`, `_split`): once per row")
RMSNormInvTriton = _traced("RMSNormInvTriton", 1, (), _scale_sig("triton", "inv", F32), lambda S: _out_parts("triton")[0],
                           "the gates of bf16_rn((inv * f32(x)) * f32(w)) that read only inv (`_out_triton`, `_split`): once per row")
RMSNormOutCuda = _traced(
    "RMSNormOutCuda", 1, (),
    lambda S: ((("z", BF16), ("zd", _bits(_dec_bits("cuda"))), ("s", F32), ("sd", _bits(_scale_bits("cuda"))), ("w", BF16)), BF16),
    lambda S: _out_parts("cuda")[2],
    "bf16_rn((f32(z) * s) * f32(w)) (F32Mul_v1 twice, F2fpBf16_v1) from z's decode zd (`RMSNormSquare_v1`) and s's "
    "(`RMSNormScaleCuda_v1`): the gates that read z or w")
RMSNormOutTriton = _traced(
    "RMSNormOutTriton", 1, (),
    lambda S: ((("inv", F32), ("invd", _bits(_scale_bits("triton"))), ("x", BF16), ("xd", _bits(_dec_bits("triton"))), ("w", BF16)),
               BF16),
    lambda S: _out_parts("triton")[2],
    "bf16_rn((inv * f32(x)) * f32(w)) (F32Mul_v1 twice, F2fpBf16_v1) from inv's decode (`RMSNormInvTriton_v1`) and x's "
    "(`RMSNormThreadTriton_v1`): the gates that read x or w")


# --- the squares, each returning its element's decode ------------------------------------------------------------------------

def _square(C, z):
    zf = fp.bf16_to_f32(z)
    return [fp.f32_mul(C, zf, zf, ir_nan=True), _replay(_out_parts("cuda")[1], C, z)]


def _square_fma(C, z, acc):
    zf = fp.bf16_to_f32(z)
    return [fp.f32_fma(C, zf, zf, acc), _replay(_out_parts("cuda")[1], C, z)]


def _thread(C, *x):
    xf = [fp.bf16_to_f32(v) for v in x]
    acc = fp.f32_mul(C, xf[1], xf[1], ir_nan=True)
    acc = fp.f32_fma(C, xf[0], xf[0], acc)
    for j in range(2, TRITON_COLS):
        acc = fp.f32_fma(C, xf[j], xf[j], acc)
    return [acc] + [_replay(_out_parts("triton")[1], C, v) for v in x]


def _sq_sig(*params):
    return lambda S: (params, Tuple(F32, _bits(_dec_bits("cuda"))))


def _thread_sig(S):
    return (("x", Array(TRITON_COLS, BF16)),), Tuple(F32, Array(TRITON_COLS, _bits(_dec_bits("triton"))))


RMSNormSquare = _traced("RMSNormSquare", 1, (), _sq_sig(("z", BF16)),
                        lambda S: TR.trace(_square, [BF16], Tuple(F32, _bits(_dec_bits("cuda")))),
                        "(F32Mul_v1(f32(z), f32(z)), z's decode for `RMSNormOutCuda_v1`): the fused norm's first square")
RMSNormSquareFma = _traced("RMSNormSquareFma", 1, (), _sq_sig(("z", BF16), ("acc", F32)),
                           lambda S: TR.trace(_square_fma, [BF16, F32], Tuple(F32, _bits(_dec_bits("cuda")))),
                           "(F32Fma_v1(f32(z), f32(z), acc), z's decode for `RMSNormOutCuda_v1`): the fused norm's next square")
RMSNormThreadTriton = _traced("RMSNormThreadTriton", 1, (), _thread_sig,
                              lambda S: TR.trace(_thread, [Array(TRITON_COLS, BF16)], _thread_sig(S)[1]),
                              "one Triton thread's 8 columns: acc = x1 * x1, then FFMA x0, x2, ..., x7 (F32Mul_v1, F32Fma_v1), "
                              "and each column's decode for `RMSNormOutTriton_v1`")


# --- the constant steps: variance / N + EPS, fma(rcp(N), sum, EPS), the constants folded --------------------------------------

def _var_eps_cuda_word(B, S, var):
    return B.call(P.F32Add, B.call(P.F32Div, var, B.call(const(32, _f32(float(S.N))))), B.call(const(32, _f32(S.EPS))))


def _rcp_n(N: int) -> int:
    (v,) = evaluate(P.DivFullRcp, _f32(float(N)))
    return int(v)


def _var_eps_triton_word(B, S, total):
    return B.call(P.F32Fma, B.call(const(32, _rcp_n(S.N))), total, B.call(const(32, _f32(S.EPS))))


RMSNormVarEpsCudaWord = CompositeDefinition("RMSNormVarEpsCuda", 1, ("N", "EPS"), lambda S: ((("var", F32W),), F32W),
                                            _var_eps_cuda_word, doc="F32Add_v1(F32Div_v1(var, f32(N)), f32(EPS))",
                                            conformance="exact-model-tested")
RMSNormVarEpsTritonWord = CompositeDefinition("RMSNormVarEpsTriton", 1, ("N", "EPS"), lambda S: ((("total", F32W),), F32W),
                                              _var_eps_triton_word,
                                              doc="F32Fma_v1(DivFullRcp_v1(f32(N)), total, f32(EPS)), the reciprocal a constant",
                                              conformance="exact-model-tested")


def _var_eps_cuda(S):
    n, eps = fp.f32_const(_f32(float(S.N))), fp.f32_const(_f32(S.EPS))
    return TR.trace(lambda C, v: fp.f32_add(C, fp.f32_div(C, v, n, ir_nan=True), eps, ir_nan=True), [F32], F32)


def _var_eps_triton(S):
    r, eps = fp.f32_const(_rcp_n(S.N)), fp.f32_const(_f32(S.EPS))
    return TR.trace(lambda C, t: fp.f32_fma(C, r, t, eps), [F32], F32)


RMSNormVarEpsCuda = _traced("RMSNormVarEpsCuda", 2, ("N", "EPS"), lambda S: ((("var", F32),), F32), _var_eps_cuda,
                            "var / N + EPS on bits (div.rn, add.rn; N and EPS folded): RMSNormVarEpsCuda_v1",
                            word=lambda b: bind(RMSNormVarEpsCudaWord, N=b["N"], EPS=b["EPS"]))
RMSNormVarEpsTriton = _traced("RMSNormVarEpsTriton", 2, ("N", "EPS"), lambda S: ((("total", F32),), F32), _var_eps_triton,
                              "fma(rcp(N), total, EPS) on bits (rcp(N) and EPS folded): RMSNormVarEpsTriton_v1",
                              word=lambda b: bind(RMSNormVarEpsTritonWord, N=b["N"], EPS=b["EPS"]))


# --- the fused CUDA norm (vLLM's generic fused_add_rms_norm_kernel) -------------------------------------------------------------

def _threads_cuda(N: int, BLOCK: int) -> list[tuple[int, int, int]]:
    """(first thread, end, elements per thread) of each run of threads holding equally many elements."""
    T = min(N, BLOCK)
    runs: list[tuple[int, int, int]] = []
    for t in range(T):
        J = -(-(N - t) // T)
        if runs and runs[-1][2] == J:
            runs[-1] = (runs[-1][0], t + 1, J)
        else:
            runs.append((t, t + 1, J))
    return runs


@composite("RMSNormLaneCuda", 1, ["J"],
           lambda S: ((("x", Array(S.J, BF16)), ("res", Array(S.J, BF16))),
                      Tuple(Array(S.J, BF16), Array(S.J, _bits(_dec_bits("cuda"))), F32)),
           conformance="exact-model-tested")
def RMSNormLaneCuda(B, S, x, res):
    """One thread's J elements: r = Bf16AddF2fp_v2(x, res) each, r's decodes, and its partial: the first r squared, then FFMA
    of each next. Returns (r, decodes, partial)."""
    r = B.batch(EW.Bf16AddF2fp, x, res, axes=(0, 0))
    sq = B.call(RMSNormSquare, r[0])
    acc, dec = sq[0], [sq[1]]
    for k in range(1, S.J):
        sq = B.call(RMSNormSquareFma, r[k], acc)
        acc = sq[0]
        dec.append(sq[1])
    return tuple_of(r, array_of(dec), acc)


@composite("RMSNormLanesCuda", 1, ["N", "BLOCK"],
           lambda S: ((("x", Array(S.N, BF16)), ("res", Array(S.N, BF16))),
                      Tuple(Array(S.N, BF16), Array(S.N, _bits(_dec_bits("cuda"))), Array(min(S.N, S.BLOCK), F32))),
           conformance="exact-model-tested")
def RMSNormLanesCuda(B, S, x, res):
    """The residual sum r, its decodes and every thread's partial of a block of min(N, BLOCK) threads, thread t owning
    t, t + BLOCK, ...: one batch of `RMSNormLaneCuda_v1{J}` per run of threads with J elements (separable)."""
    N, T = S.N, min(S.N, S.BLOCK)
    r: dict[int, Coll] = {}
    dec: dict[int, Coll] = {}
    partials: list[Coll] = []
    for t0, t1, J in _threads_cuda(N, S.BLOCK):
        M = t1 - t0
        els = [t0 + m + k * T for m in range(M) for k in range(J)]
        out = B.batch(bind(RMSNormLaneCuda, J=J), coll_concat(Array(M, Array(J, BF16)), [x[e] for e in els]),
                      coll_concat(Array(M, Array(J, BF16)), [res[e] for e in els]), axes=(0, 0))
        for m in range(M):
            for k in range(J):
                r[t0 + m + k * T] = out[m][0][k]
                dec[t0 + m + k * T] = out[m][1][k]
            partials.append(out[m][2])
    return tuple_of(array_of([r[e] for e in range(N)]), array_of([dec[e] for e in range(N)]), array_of(partials))


@composite("RMSNormWarpCuda", 1, ["L"], lambda S: ((("p", Array(S.L, F32)),), F32), conformance="exact-model-tested")
def RMSNormWarpCuda(B, S, p):
    """cub `WarpReduceShfl` over L valid lanes: offsets 1, 2, 4, 8, 16, lane i adding lane i + offset when it exists; lane 0."""
    if not 1 <= S.L <= WARP:
        raise ValueError(f"a warp has 1 to {WARP} lanes, not {S.L}")
    lanes = {i: p[i] for i in range(S.L)}
    for off in (1, 2, 4, 8, 16):
        lanes = {i: (B.call(EW.F32Add, v, lanes[i + off]) if i + off in lanes else v)
                 for i, v in lanes.items() if i % (2 * off) == 0}
    return lanes[0]


def _warps(B, p: Coll, n: int, warp) -> Coll:
    """Each 32-lane warp's sum of the n partials ``p``, the last warp possibly partial: ``warp(L)`` is the warp Definition."""
    full, rest = divmod(n, WARP)
    sums: list[Coll] = []
    if full:
        out = B.batch(warp(WARP), p[:WARP * full].reshape(Array(full, Array(WARP, F32))), axes=(0,))
        sums += [out[i] for i in range(full)]
    if rest:
        sums.append(B.call(warp(rest), p[WARP * full:]))
    return array_of(sums)


@composite("RMSNormWarpsCuda", 1, ["T"], lambda S: ((("p", Array(S.T, F32)),), Array(-(-S.T // WARP), F32)),
           conformance="exact-model-tested")
def RMSNormWarpsCuda(B, S, p):
    """Every warp's aggregate of the T threads' partials (`RMSNormWarpCuda_v1`; separable)."""
    return _warps(B, p, S.T, lambda L: bind(RMSNormWarpCuda, L=L))


@composite("RMSNormArgCuda", 1, ["N", "EPS", "NW"], lambda S: ((("s", Array(S.NW, F32)),), F32), conformance="exact-model-tested")
def RMSNormArgCuda(B, S, s):
    """`variance / N + EPS`: cub's `ApplyWarpAggregates` (warp sums added in order from warp 0, F32Add_v3), then
    `RMSNormVarEpsCuda_v2{N, EPS}`."""
    var = s[0]
    for i in range(1, S.NW):
        var = B.call(EW.F32Add, var, s[i])
    return B.call(bind(RMSNormVarEpsCuda, N=S.N, EPS=S.EPS), var)


@composite("RMSNormOutRowCuda", 1, ["N"],
           lambda S: ((("r", Array(S.N, BF16)), ("rd", Array(S.N, _bits(_dec_bits("cuda")))), ("s", F32),
                       ("sd", _bits(_scale_bits("cuda"))), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def RMSNormOutRowCuda(B, S, r, rd, s, sd, w):
    """bf16_rn((f32(r_j) * s) * f32(w_j)) for every j (`RMSNormOutCuda_v1`; separable)."""
    return B.batch(RMSNormOutCuda, r, rd, s, sd, w, axes=(0, 0, None, None, 0))


def rmsnorm_cuda_calls(B, N: int, EPS: float, RSQRT, x: Coll, res: Coll, w: Coll) -> Coll:
    """`RMSNormFusedCuda_v3`'s Calls, written into the body ``B`` (a composite's, or a Program root's): (normed, res')."""
    T = min(N, b1.FUSED_RMS_BLOCK)
    lanes = B.call(bind(RMSNormLanesCuda, N=N, BLOCK=b1.FUSED_RMS_BLOCK), x, res)
    sums = B.call(bind(RMSNormWarpsCuda, T=T), lanes[2])
    arg = B.call(bind(RMSNormArgCuda, N=N, EPS=EPS, NW=-(-T // WARP)), sums)
    s = _mufu(B, RSQRT, _F32(B, bits=arg)).bits()
    out = B.call(bind(RMSNormOutRowCuda, N=N), lanes[0], lanes[1], s, B.call(RMSNormScaleCuda, s), w)
    return tuple_of(out, lanes[0])


@_on_bits("RMSNormFusedCuda", 3, ["N", "EPS", "RSQRT"],
          lambda S: ((("x", Array(S.N, BF16)), ("res", Array(S.N, BF16)), ("w", Array(S.N, BF16))),
                     Tuple(Array(S.N, BF16), Array(S.N, BF16))),
          _word_view(b1.RMSNormFusedCuda, ("N", "EPS"), ("RSQRT",)), word_statics=CUDA_WORD)
def RMSNormFusedCuda(B, S, x, res, w):
    """`RMSNormFusedCuda_v2` on bits (`rmsnorm_cuda_calls`): vLLM's generic `fused_add_rms_norm_kernel<bf16, 0, true>`,
    `s = rsqrtf(variance / N + eps)` with ``RSQRT`` computing `RsqrtApprox_v1`. Returns (normed, res')."""
    return rmsnorm_cuda_calls(B, S.N, S.EPS, S.RSQRT, x, res, w)


# --- the Triton norm (vLLM's batch-invariant _rms_norm_kernel) ------------------------------------------------------------------

def _add0(B, a: Coll | None, b: Coll | None) -> Coll | None:
    """``a + b`` (F32Add_v3), an idle +0 (None) dropped: neither operand is -0 or a signalling NaN (the module docstring)."""
    if a is None or b is None:
        return b if a is None else a
    return B.call(EW.F32Add, a, b)


@composite("RMSNormThreadsTriton", 1, ["N"],
           lambda S: ((("x", Array(S.N, BF16)),), Tuple(Array(S.N // TRITON_COLS, F32), Array(S.N, _bits(_dec_bits("triton"))))),
           conformance="exact-model-tested")
def RMSNormThreadsTriton(B, S, x):
    """Each thread's sum of squares of its 8 columns, threads in column order, and every column's decode
    (`RMSNormThreadTriton_v1`; separable)."""
    if S.N % TRITON_COLS:
        raise ValueError(f"the Triton norm reads {TRITON_COLS} columns per thread; N = {S.N} is not a multiple")
    n = S.N // TRITON_COLS
    out = B.batch(RMSNormThreadTriton, x.reshape(Array(n, Array(TRITON_COLS, BF16))), axes=(0,))
    return tuple_of(array_of([out[t][0] for t in range(n)]),
                    array_of([out[t][1][j] for t in range(n) for j in range(TRITON_COLS)]))


@composite("RMSNormWarpTriton", 1, ["L"], lambda S: ((("p", Array(S.L, F32)),), F32), conformance="exact-model-tested")
def RMSNormWarpTriton(B, S, p):
    """Lane 0 after the xor butterfly 16, 8, 4, 2, 1 (lane i adds lane i ^ m, its own value first) over a warp whose lanes from L
    on are idle (+0): only lane 0's cone, without the idle adds."""
    if not 1 <= S.L <= WARP:
        raise ValueError(f"a warp has 1 to {WARP} lanes, not {S.L}")
    masks = (16, 8, 4, 2, 1)
    memo: dict[tuple[int, int], Coll | None] = {}

    def lane(k: int, i: int) -> Coll | None:
        if k == 0:
            return p[i] if i < S.L else None
        if (k, i) not in memo:
            memo[(k, i)] = _add0(B, lane(k - 1, i), lane(k - 1, i ^ masks[k - 1]))
        return memo[(k, i)]

    return lane(len(masks), 0)


@composite("RMSNormWarpsTriton", 1, ["N"], lambda S: ((("p", Array(S.N // TRITON_COLS, F32)),),
                                                        Array(-(-S.N // (TRITON_COLS * WARP)), F32)),
           conformance="exact-model-tested")
def RMSNormWarpsTriton(B, S, p):
    """Each warp's butterfly sum, for the warps holding at least one column (chunk-major, 4 per chunk; separable)."""
    return _warps(B, p, S.N // TRITON_COLS, lambda L: bind(RMSNormWarpTriton, L=L))


@composite("RMSNormMeanEpsTriton", 1, ["N", "EPS"],
           lambda S: ((("s", Array(-(-S.N // (TRITON_COLS * WARP)), F32)),), F32), conformance="exact-model-tested")
def RMSNormMeanEpsTriton(B, S, s):
    """Per chunk of 1024 columns `(w0 + w2) + (w1 + w3)` (idle warps +0), the chunks added in order from +0 (F32Add_v3), then
    `RMSNormVarEpsTriton_v2{N, EPS}`."""
    nw = len(s)
    warps = TRITON_THREADS // WARP
    total = None
    for c0 in range(0, nw, warps):
        w = [s[c0 + i] if c0 + i < nw else None for i in range(warps)]
        total = _add0(B, total, _add0(B, _add0(B, w[0], w[2]), _add0(B, w[1], w[3])))
    return B.call(bind(RMSNormVarEpsTriton, N=S.N, EPS=S.EPS), total)


@composite("RMSNormOutRowTriton", 1, ["N"],
           lambda S: ((("inv", F32), ("invd", _bits(_scale_bits("triton"))), ("x", Array(S.N, BF16)),
                       ("xd", Array(S.N, _bits(_dec_bits("triton")))), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def RMSNormOutRowTriton(B, S, inv, invd, x, xd, w):
    """bf16_rn((inv * f32(x_j)) * f32(w_j)) for every j (`RMSNormOutTriton_v1`; separable)."""
    return B.batch(RMSNormOutTriton, inv, invd, x, xd, w, axes=(None, None, 0, 0, 0))


def rmsnorm_triton_calls(B, N: int, EPS: float, SQRT, SCALEA, RCP, x: Coll, w: Coll) -> Coll:
    """`RMSNormTriton_v2`'s Calls, written into the body ``B`` (a composite's, or a Program root's)."""
    th = B.call(bind(RMSNormThreadsTriton, N=N), x)
    s = B.call(bind(RMSNormWarpsTriton, N=N), th[0])
    rms = _mufu(B, SQRT, _F32(B, bits=B.call(bind(RMSNormMeanEpsTriton, N=N, EPS=EPS), s)))
    inv = B.call(EW.F32Mul, _mufu(B, SCALEA, _F32(B, value=_f32(1.0)), rms).bits(), _mufu(B, RCP, rms).bits())
    return B.call(bind(RMSNormOutRowTriton, N=N), inv, B.call(RMSNormInvTriton, inv), x, th[1], w)


@_on_bits("RMSNormTriton", 2, ["N", "EPS", "SQRT", "SCALEA", "RCP"],
          lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
          _word_view(b1.RMSNormTriton, ("N", "EPS"), ("SQRT", "SCALEA", "RCP")), word_statics=TRITON_WORD)
def RMSNormTriton(B, S, x, w):
    """`RMSNormTriton_v1` on bits (`rmsnorm_triton_calls`): vLLM's batch-invariant Triton `_rms_norm_kernel`,
    `inv = div.full(1, sqrt.approx.ftz(fma(rcp(N), sum, eps)))` with ``SQRT``, ``SCALEA`` and ``RCP`` computing `MufuSqrtFtz_v1`,
    `DivFullScaleA_v1` and `DivFullRcp_v1`."""
    return rmsnorm_triton_calls(B, S.N, S.EPS, S.SQRT, S.SCALEA, S.RCP, x, w)
