"""Element-wise FP32, BF16 and I32 arithmetic as Boolean Definitions: the next version of each word primitive, the same function
on bits.

    F32Add_v3, F32Mul_v3, F32Div_v3   (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>        word view: their _v1
    F32Fma_v3                         (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>     word view: F32Fma_v1
    F32Add_v4, F32Mul_v4, F32Div_v4   (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>        word view: their _v2
    F32Fma_v4                         (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>     word view: F32Fma_v2
    Bf16Add_v2, Bf16AddF2fp_v2        (a, b: Array<16, Value<1>>) -> Array<16, Value<1>>        word view: their _v1
    Bf16MulBf16_v2                    (a, b: Array<16, Value<1>>) -> Array<16, Value<1>>        word view: Bf16MulBf16_v1
    Bf16MulF32_v2                     (a: Array<16, Value<1>>, s: Array<32, Value<1>>) -> Array<16, Value<1>>
                                                                                                word view: Bf16MulF32_v1
    I32Add_v2                         (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>        word view: I32Add_v1

Each is traced from the builder C-Flock's lowering runs for its word primitive (`trace.definition`), so its And gates are that
builder's live ANDs: `fp.f32_add` / `f32_mul` / `f32_div` with the IR's NaN words (numpy float32 on x86-64), `fp.f32_fma`, the
same four with every NaN the GPU's 0x7FFFFFFF for the `_v4`s (the `_v2` words, as C-Flock's `tail_pieces` lowers them), the
f32 add or multiply of the widened operands rounded by `fp.f32_to_bf16` to the word's NaN (0x7FC0 for `Bf16Add_v1`, F2FP's
0x7FFF for the others), and `forms.add` at 32 bits. None is flushed: a
subnormal operand or result is kept, as the `_v1` words keep it. `F32Div_v3` is the IEEE divider (`div.rn.f32`, which
`F32Div_v1` is); the MUFU pieces of `div.full.f32` (`DivFullRcp_v1`, `DivFullScaleA_v1`) are their own Calls, not inside it.

`bf16_band` is the piece a function of one bf16 word reads where it is neither exact arithmetic nor a constant: a band of
exponent fields per sign, held as a constant table of the function's own values and read with plain gates (one-hot minterms
and fixed XORs of them).

A row of an element-wise operation computes no value twice (`Q_word`'s no-recompute rule) when an operand every member shares
is decoded once: `shared_operand` cuts a builder's circuit into the gates that read only that operand (one Call per row) and the
rest (one per member). A constant operand is not an operand at all: `specialized` traces the builder for each binding of its
statics, so the constant is folded into the gates; `shared_specialized` cuts such a builder per binding, at an operand or at
each element of one. `on_bits` is a composite of such Calls whose word view is a word composite at the same statics.
"""
from __future__ import annotations

import functools
import operator
from typing import Any, Callable, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array, Type

from .. import fp32 as W
from .. import scalar as S
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import NOT, ZERO, add, and_reduce
from verity.ml.circuits.boolean.trace import Traced, definition, emit, trace, word_of

__all__ = ["Bf16Add", "Bf16AddF2fp", "Bf16MulBf16", "Bf16MulF32", "F32Add", "F32AddV4", "F32Div", "F32DivV4", "F32Fma", "F32FmaV4",
           "F32Mul", "F32MulV4", "I32Add", "bf16_band", "on_bits", "shared_operand", "shared_specialized", "specialized"]

F32 = I32 = Array(32, BL.BIT)
BF16 = Array(16, BL.BIT)
_AB = (("a", F32), ("b", F32))


def _blocks(ks) -> dict[int, int]:
    out: dict[int, int] = {}
    for k in ks:
        out[k >> 2] = out.get(k >> 2, 0) | 1 << (k & 3)
    return out


def bf16_band(C, g, value: Callable[[int], int], lo: int, hi: tuple[int, int]):
    """The 15 magnitude bits of ``value(g)`` for a bf16 word g (LSB first) whose exponent field lies in ``[lo, hi[sign]]``, and
    0 for every other g.

    g's mantissa and lowest exponent bit decode to 256 one-hot minterms, and each row of the band (a sign and an exponent pair
    e, e + 1) to one minterm; each output bit of a row is one AND of the row's minterm with a fixed XOR of low minterms, or its
    complement, whichever has fewer terms. A column's XOR is a chain over blocks of four in ascending order, each block's part
    a chain over its minterms, so equal parts and equal prefixes are one wire."""
    if not all(0 <= lo <= h <= 255 for h in hi):
        raise ValueError(f"not a band of exponent fields: [{lo}, {hi}]")
    m, e, s = list(g[:7]), list(g[7:15]), g[15]
    low = fp.decode(C, m + [e[0]])
    high = fp.decode(C, e[1:4] + [s])
    nibbles: dict[int, Any] = {}
    parts: dict[tuple[int, int], Any] = {}

    def part(b: int, mask: int):
        if (b, mask) not in parts:
            top = mask.bit_length() - 1
            rest = mask ^ 1 << top
            parts[(b, mask)] = part(b, rest) ^ low[4 * b + top] if rest else low[4 * b + top]
        return parts[(b, mask)]

    out = [ZERO] * 15
    for sign in (0, 1):
        for e0 in range(lo & ~1, hi[sign] + 1, 2):
            n = e0 >> 4
            if n not in nibbles:
                nibbles[n] = and_reduce(C, [e[4 + i] if n >> i & 1 else NOT(e[4 + i]) for i in range(4)])
            row = C.AND(high[(e0 & 0xF) >> 1 | sign << 3], nibbles[n])
            words = [value(sign << 15 | (e0 | k >> 7) << 7 | k & 0x7F) & 0x7FFF if lo <= e0 | k >> 7 <= hi[sign] else 0
                     for k in range(256)]
            for j in range(15):
                ones = [k for k in range(256) if words[k] >> j & 1]
                if len(ones) == 256:
                    out[j] = out[j] ^ row
                elif ones:
                    blocks, zeros = _blocks(ones), _blocks(sorted(set(range(256)) - set(ones)))
                    pick, flip = (blocks, False) if len(blocks) <= len(zeros) else (zeros, True)
                    col = None
                    for b in sorted(pick):
                        col = part(b, pick[b]) if col is None else col ^ part(b, pick[b])
                    out[j] = out[j] ^ C.AND(row, NOT(col) if flip else col)
    return out


def _sub(T: Traced, keep: list[int], inputs: list[int], outputs: list[int]) -> Traced:
    """The gates ``keep`` of ``T`` (ascending) as a circuit from the wires ``inputs`` of ``T`` to the wires ``outputs``."""
    new = {0: 0, 1: 1, **{w: 2 + k for k, w in enumerate(inputs)}}
    gates: list[tuple[str, tuple[int, ...]]] = []
    for i in keep:
        kind, ops = T.gates[i]
        gates.append((kind, tuple(new[o] for o in ops)))
        new[T.first_gate + i] = 1 + len(inputs) + len(gates)
    return Traced(len(inputs), gates, [new[o] for o in outputs], {k: sum(g == k for g, _ in gates) for k in ("and", "xor", "not")})


def _split(T: Traced, types: Sequence[Type], shared: int, by_element: bool = False) -> list[Traced]:
    """``T`` over parameters of ``types`` cut in two by what its gates read: the gates that read only parameter ``shared``, from
    its bits to their frontier (those the other gates read, or outputs), and the rest, from every parameter's bits and then the
    frontier to the outputs.  With ``by_element``, ``shared`` an Array each of whose gates reads one element, every element's
    gates one circuit (else `ValueError`): the first part is that circuit, from one element's bits to its frontier, and the
    rest reads the frontier element by element."""
    starts = [2 + sum(t.leaves for t in types[:p]) for p in range(len(types))]
    wires = [list(range(s, s + t.leaves)) for s, t in zip(starts, types)]
    sup = [0, 0] + [1 << p for p, t in enumerate(types) for _ in range(t.leaves)]
    for _, ops in T.gates:
        sup.append(functools.reduce(operator.or_, (sup[o] for o in ops)))
    keep = [i for i in range(len(T.gates)) if sup[T.first_gate + i] == 1 << shared]
    mine = {T.first_gate + i for i in keep}
    frontier = sorted({o for i, (_, ops) in enumerate(T.gates) if T.first_gate + i not in mine for o in ops if o in mine}
                      | (mine & set(T.outputs)))
    rest = [i for i in range(len(T.gates)) if T.first_gate + i not in mine]
    own = _sub(T, keep, wires[shared], frontier)
    if by_element:
        t = types[shared]
        m = t.elem.leaves
        el = {w: (w - starts[shared]) // m for w in wires[shared]}
        for i in keep:
            reads = {el[o] for o in T.gates[i][1] if o in el}
            if len(reads) != 1:
                raise ValueError(f"gate {i} reads elements {sorted(reads)} of parameter {shared}")
            el[T.first_gate + i] = reads.pop()
        by = [[w for w in frontier if el[w] == j] for j in range(t.n)]
        parts = [_sub(T, [i for i in keep if el[T.first_gate + i] == j], wires[shared][j * m:(j + 1) * m], by[j]) for j in range(t.n)]
        if any(p.gates != parts[0].gates or p.outputs != parts[0].outputs for p in parts):
            raise ValueError(f"the elements of parameter {shared} are not one circuit")
        own, frontier = parts[0], [w for ws in by for w in ws]
    return [own, _sub(T, rest, [w for ws in wires for w in ws] + frontier, T.outputs)]


def shared_operand(names: tuple[str, str], version: int, build: Callable, params: Sequence[tuple[str, Type]], ret: Type,
                   shared: int, docs: tuple[str, str], register: bool = True) -> tuple[CompositeDefinition, CompositeDefinition]:
    """``build``'s circuit (`trace.trace`) for a row whose members share parameter ``shared``, cut in two by what its gates read,
    so that the row computes no value twice: ``names[0]`` is the gates that read only that parameter, from its bits to their
    frontier (those the other gates read, or outputs), one Call per row; ``names[1]`` is the rest, from every parameter's bits
    and then the frontier ``<shared>d`` to the outputs, one Call per member. Traced on first use; each has ``traced()``."""
    types = [t for _, t in params]
    parts: list[Traced] = []

    def split() -> list[Traced]:
        if not parts:
            parts.extend(_split(trace(build, types, ret), types, shared))
        return parts

    name, t = params[shared]
    dt = lambda: Array(len(split()[0].outputs), BL.BIT)  # noqa: E731
    own = CompositeDefinition(names[0], version, (), lambda S: (((name, t),), dt()),
                              lambda B, S, *c: emit(split()[0], [t], dt(), B, *c), doc=docs[0], conformance="exact-model-tested",
                              register=register)
    rest = CompositeDefinition(names[1], version, (), lambda S: (tuple(params) + ((name + "d", dt()),), ret),
                               lambda B, S, *c: emit(split()[1], types + [dt()], ret, B, *c), doc=docs[1],
                               conformance="exact-model-tested", register=register)
    own.traced = lambda: split()[0]  # type: ignore[attr-defined]
    rest.traced = lambda: split()[1]  # type: ignore[attr-defined]
    return own, rest


def specialized(name: str, version: int, statics: Sequence[str], build: Callable,
                signature: Callable[[Any], tuple[Sequence[tuple[str, Type]], Type]], doc: str = "", word: Any = None,
                register: bool = True) -> CompositeDefinition:
    """The Boolean Definition ``name_v{version}{statics}`` whose circuit for bindings S is the builder ``build(S)`` traced over
    ``signature(S)`` (`trace.trace`), so that a constant or a length the statics name is folded into its gates, never an operand.
    Traced once per binding, on first use; ``traced(S)``. ``word``, a function of the bindings, names the word Definition it
    computes (`trace.word_of`)."""
    made: dict[tuple, Traced] = {}

    def types(S) -> tuple[list[Type], Type]:
        params, ret = signature(S)
        return [t for _, t in params], ret

    def traced(S) -> Traced:
        key = tuple(getattr(S, k) for k in statics)
        if key not in made:
            made[key] = trace(build(S), *types(S))
        return made[key]

    d = CompositeDefinition(name, version, tuple(statics), signature,
                            lambda B, S, *c: emit(traced(S), *types(S), B, *c), doc=doc, conformance="exact-model-tested",
                            register=register)
    d.traced = traced  # type: ignore[attr-defined]
    d.word = word  # type: ignore[attr-defined]
    return d


def shared_specialized(names: tuple[str, str], version: int, statics: Sequence[str], build: Callable,
                       signature: Callable[[Any], tuple[Sequence[tuple[str, Type]], Type]], shared: int, docs: tuple[str, str],
                       register: bool = True, element: Type = BL.BIT,
                       by_element: bool = False) -> tuple[CompositeDefinition, CompositeDefinition]:
    """`shared_operand` of a `specialized` builder: for bindings S, ``build(S)`` traced over ``signature(S)`` and cut at parameter
    ``shared``, ``names[0]{statics}`` the gates that read only it and ``names[1]{statics}`` the rest, from every parameter's
    bits and then the frontier ``<shared>d``, an ``Array<n, element>`` (one n-bit word; n 1-bit words with ``element`` an
    ``Array<1, Value<1>>``). With ``by_element`` the cut is at one element of ``shared``, an Array whose elements' gates are
    each the same circuit (`_split`): ``names[0]`` reads one element, one Call per element, and ``<shared>d`` is the Array of
    their frontiers. Traced once per binding, on first use; each has ``traced(S)``."""
    made: dict[tuple, list[Traced]] = {}

    def split(S) -> list[Traced]:
        key = tuple(getattr(S, k) for k in statics)
        if key not in made:
            params, ret = signature(S)
            types = [t for _, t in params]
            made[key] = _split(trace(build(S), types, ret), types, shared, by_element)
        return made[key]

    def own_sig(S):
        name, t = signature(S)[0][shared]
        return ((name, t.elem if by_element else t),), Array(len(split(S)[0].outputs), element)

    def rest_sig(S):
        (params, ret), (_, dt) = signature(S), own_sig(S)
        name, t = params[shared]
        return tuple(params) + ((name + "d", Array(t.n, dt) if by_element else dt),), ret

    def body(part: int, sig: Callable):
        def emit_part(B, S, *c):
            params, ret = sig(S)
            return emit(split(S)[part], [t for _, t in params], ret, B, *c)
        return emit_part

    own = CompositeDefinition(names[0], version, tuple(statics), own_sig, body(0, own_sig), doc=docs[0],
                              conformance="exact-model-tested", register=register)
    rest = CompositeDefinition(names[1], version, tuple(statics), rest_sig, body(1, rest_sig), doc=docs[1],
                               conformance="exact-model-tested", register=register)
    own.traced = lambda S: split(S)[0]  # type: ignore[attr-defined]
    rest.traced = lambda S: split(S)[1]  # type: ignore[attr-defined]
    return own, rest


def on_bits(name: str, version: int, statics: Sequence[str], signature: Callable, body: Callable, word: Any,
            register: bool = True, lifted: Sequence[str] = ()) -> CompositeDefinition:
    """The Boolean composite ``name_v{version}{statics}`` with body ``body`` (whose Calls are Boolean Definitions), documented by
    ``body``'s docstring, whose word view is the word Definition ``word`` bound at the same statics (`trace.word_of`), each static
    in ``lifted`` (a Boolean Definition) read as its own word view. A static-free one names ``word`` itself: the switch
    (`verity_vllm.program.boolean`) finds a word Definition without bindings by its id alone."""
    d = CompositeDefinition(name, version, tuple(statics), signature, body, doc=body.__doc__ or "",
                            conformance="exact-model-tested", register=register)
    if statics:
        d.word = lambda b: bind(word, **{k: (word_of(b[k]) if k in lifted else b[k]) for k in statics})  # type: ignore[attr-defined]
    else:
        d.word = bind(word) if isinstance(word, CompositeDefinition) else word  # type: ignore[attr-defined]
    return d


def bf16_add(C, a, b, nan: int = W.BF16_NAN_TORCH):
    """``bf16_rn(f32(a) + f32(b))``, every NaN ``nan``: 0x7FC0 is ``Bf16Add_v1``, F2FP's 0x7FFF ``Bf16AddF2fp_v1``."""
    return fp.f32_to_bf16(C, fp.f32_add(C, fp.bf16_to_f32(a), fp.bf16_to_f32(b)), nan)


def bf16_add_f2fp(C, a, b):
    """``Bf16AddF2fp_v1``: ``bf16_rn(f32(a) + f32(b))``, every NaN 0x7FFF."""
    return bf16_add(C, a, b, W.BF16_NAN_F2FP)


def bf16_mul_f32(C, a, s):
    """``Bf16MulF32_v1``: ``bf16_rn(f32(a) * s)`` (one f32 rounding, then F2FP's), every NaN 0x7FFF."""
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(a), s), W.BF16_NAN_F2FP)


F32Add = definition("F32Add", 3, lambda C, a, b: fp.f32_add(C, a, b, ir_nan=True), _AB, F32, word=W.F32Add,
                    doc="FADD on bits: F32Add_v1, RN(a + b), subnormals kept, numpy float32's NaN words")
F32Mul = definition("F32Mul", 3, lambda C, a, b: fp.f32_mul(C, a, b, ir_nan=True), _AB, F32, word=W.F32Mul,
                    doc="FMUL on bits: F32Mul_v1, RN(a * b), subnormals kept, numpy float32's NaN words")
F32Fma = definition("F32Fma", 3, fp.f32_fma, (("a", F32), ("b", F32), ("c", F32)), F32, word=W.F32Fma,
                    doc="FFMA on bits: F32Fma_v1, RN(a * b + c) with one rounding, subnormals kept, every NaN 0x7FC00000")
F32Div = definition("F32Div", 3, lambda C, a, b: fp.f32_div(C, a, b, ir_nan=True), _AB, F32, word=W.F32Div,
                    doc="div.rn.f32 on bits: F32Div_v1, RN(a / b), subnormals kept, numpy float32's NaN words")
F32AddV4 = definition("F32Add", 4, lambda C, a, b: fp.f32_add(C, a, b, nan_word=fp.F32_NAN_PTX), _AB, F32, word=W.F32AddV2,
                      doc="FADD on bits: F32Add_v2, RN(a + b), subnormals kept, every NaN 0x7FFFFFFF")
F32MulV4 = definition("F32Mul", 4, lambda C, a, b: fp.f32_mul(C, a, b, nan_word=fp.F32_NAN_PTX), _AB, F32, word=W.F32MulV2,
                      doc="FMUL on bits: F32Mul_v2, RN(a * b), subnormals kept, every NaN 0x7FFFFFFF")
F32FmaV4 = definition("F32Fma", 4, lambda C, a, b, c: fp.f32_fma(C, a, b, c, nan_word=fp.F32_NAN_PTX),
                      (("a", F32), ("b", F32), ("c", F32)), F32, word=W.F32FmaV2,
                      doc="FFMA on bits: F32Fma_v2, RN(a * b + c) with one rounding, subnormals kept, every NaN 0x7FFFFFFF")
F32DivV4 = definition("F32Div", 4, lambda C, a, b: fp.f32_div(C, a, b, nan_word=fp.F32_NAN_PTX), _AB, F32, word=W.F32DivV2,
                      doc="div.rn.f32 on bits: F32Div_v2, RN(a / b), subnormals kept, every NaN 0x7FFFFFFF")
Bf16AddF2fp = definition("Bf16AddF2fp", 2, bf16_add_f2fp, (("a", BF16), ("b", BF16)), BF16, word=W.Bf16AddF2fp,
                         doc="the fused_add_rms_norm residual add on bits: Bf16AddF2fp_v1, bf16_rn(f32(a) + f32(b)), NaN 0x7FFF")
Bf16Add = definition("Bf16Add", 2, bf16_add, (("a", BF16), ("b", BF16)), BF16, word=W.Bf16Add,
                     doc="aten.add(bf16, bf16) on bits: Bf16Add_v1, bf16_rn(f32(a) + f32(b)), NaN 0x7FC0")
Bf16MulF32 = definition("Bf16MulF32", 2, bf16_mul_f32, (("a", BF16), ("s", F32)), BF16, word=W.Bf16MulF32,
                        conformance=W.Bf16MulF32.conformance,
                        doc="a bf16 word times an f32 word on bits: Bf16MulF32_v1, bf16_rn(RN32(f32(a) * s)), NaN 0x7FFF")
Bf16MulBf16 = definition("Bf16MulBf16", 2, lambda C, a, b: bf16_mul_f32(C, a, fp.bf16_to_f32(b)), (("a", BF16), ("b", BF16)),
                         BF16, word=W.Bf16MulBf16, conformance=W.Bf16MulBf16.conformance,
                         doc="bf16 times bf16 on bits: Bf16MulBf16_v1, bf16_rn(RN32(f32(a) * f32(b))), NaN 0x7FFF")
I32Add = definition("I32Add", 2, lambda C, a, b: add(C, a, b, width=32), (("a", I32), ("b", I32)), I32, word=S.I32Add,
                    doc="wrapping 32-bit add on bits: I32Add_v1, a ripple-carry adder")
