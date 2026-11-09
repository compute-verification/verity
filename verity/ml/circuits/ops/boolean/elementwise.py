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

Each is traced from the builder Flock's lowering runs for its word primitive (`trace.definition`), so its And gates are that
builder's live ANDs: `fp.f32_add` / `f32_mul` / `f32_div` with the word primitives' NaN words (numpy float32 on x86-64), `fp.f32_fma`, the
same four with every NaN the GPU's 0x7FFFFFFF for the `_v4`s (the `_v2` words, as Flock's `tail_pieces` lowers them), the
f32 add or multiply of the widened operands rounded by `fp.f32_to_bf16` to the word's NaN (0x7FC0 for `Bf16Add_v1`, F2FP's
0x7FFF for the others), and `forms.add` at 32 bits. None is flushed: a
subnormal operand or result is kept, as the `_v1` words keep it. `F32Div_v3` is the IEEE divider (`div.rn.f32`, which
`F32Div_v1` is); the MUFU pieces of `div.full.f32` (`DivFullRcp_v1`, `DivFullScaleA_v1`) are their own Calls, not inside it.

A row of an element-wise operation computes no value twice (`Q_word`'s no-recompute rule) when an operand every member shares
is decoded once: `shared_operand` cuts a builder's circuit into the gates that read only that operand (one Call per row) and the
rest (one per member). A constant operand is not an operand at all: `specialized` traces the builder for each binding of its
statics, so the constant is folded into the gates; `shared_specialized` cuts such a builder per binding, at an operand or at
each element of one. `on_bits` is a composite of such Calls whose word view is a word composite at the same statics. `rolled`
is `specialized` over an Array whose elements' words are computed alike, written as one Call per element of each per-element
circuit and one of the gates they share (`roll`), so its description does not grow with the elements' count.
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
from verity.ml.circuits.boolean.forms import ONE, ZERO, add
from verity.ml.circuits.boolean.trace import Traced, _refs, definition, emit, trace, word_of
from verity.ml.circuits.refs import Coll

__all__ = ["Bf16Add", "Bf16AddF2fp", "Bf16MulBf16", "Bf16MulF32", "F32Add", "F32AddV4", "F32Div", "F32DivV4", "F32Fma", "F32FmaV4",
           "F32Mul", "F32MulV4", "I32Add", "on_bits", "shared_operand", "shared_specialized", "specialized"]

F32 = I32 = Array(32, BL.BIT)
BF16 = Array(16, BL.BIT)
_AB = (("a", F32), ("b", F32))


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
            parts.extend(_split(trace(build, types, ret, key=("/".join(names), version, {})), types, shared))
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
                register: bool = True, refuse: Callable[[dict], str | None] | None = None, ports: Any = None,
                conformance: str = "exact-model-tested") -> CompositeDefinition:
    """The Boolean Definition ``name_v{version}{statics}`` whose circuit for bindings S is the builder ``build(S)`` traced over
    ``signature(S)`` (`trace.trace`), so that a constant or a length the statics name is folded into its gates, never an operand.
    Traced once per binding, on first use; ``traced(S)``. ``word``, a function of the bindings, names the word Definition it
    computes (`trace.word_of`). ``refuse``, ``ports`` and ``conformance`` are `CompositeDefinition`'s."""
    made: dict[tuple, Traced] = {}

    def types(S) -> tuple[list[Type], Type]:
        params, ret = signature(S)
        return [t for _, t in params], ret

    def traced(S) -> Traced:
        key = tuple(getattr(S, k) for k in statics)
        if key not in made:
            made[key] = trace(build(S), *types(S), key=(name, version, dict(zip(statics, key))))
        return made[key]

    d = CompositeDefinition(name, version, tuple(statics), signature,
                            lambda B, S, *c: emit(traced(S), *types(S), B, *c), doc=doc, conformance=conformance,
                            register=register, refuse=refuse, ports=ports)
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
            made[key] = _split(trace(build(S), types, ret, key=("/".join(names), version, dict(zip(statics, key)))), types,
                               shared, by_element)
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


MANY = -1


def _element_of(T: Traced, n: int, width: int) -> list[int]:
    """Each wire's element of T's one parameter, ``n`` elements of ``width`` bits: the element every operand reads, else MANY
    (the constants read as none)."""
    el = [MANY, MANY] + [k // width for k in range(n * width)]
    for _, ops in T.gates:
        reads = {el[o] for o in ops if o >= 2}
        el.append(reads.pop() if len(reads) == 1 else MANY)
    return el


def _owner_of(T: Traced, n: int, out_width: int) -> list[int]:
    """Each wire's output element: the one element of T's first ``n * out_width`` outputs every reader of it reaches, else MANY."""
    own: list[int | None] = [None] * (T.first_gate + len(T.gates))
    for k, w in enumerate(T.outputs):
        j = k // out_width if k < n * out_width else MANY
        own[w] = j if own[w] in (None, j) else MANY
    for i in reversed(range(len(T.gates))):
        o = own[T.first_gate + i]
        if o is None:
            raise ValueError(f"gate {i} is dead")
        for op in T.gates[i][1]:
            own[op] = o if own[op] in (None, o) else MANY
    return own


def _ordered(ws) -> list[int]:
    return list(dict.fromkeys(ws))


def roll(T: Traced, n: int, width: int, out_width: int) -> dict:
    """``T``, over one parameter of ``n`` elements of ``width`` bits whose first ``n * out_width`` outputs are its elements'
    words, cut into two circuits per element and the gates the elements share. An element's ``in`` is the gates that read
    only that element (`_element_of`) and that more than its word reads, from its bits to the frontier the rest reads; its
    ``out`` the gates only its word reads (`_owner_of`), from its bits, its ``in`` frontier and the ``core`` wires it reads to
    its word; ``core`` the rest, from every bit and each cut element's ``in`` frontier to the wires the ``out``s read, then the
    outputs no ``out`` gives. The elements whose two circuits at least one other element's are too are cut, by ``classes``
    (each its ``elements``, ``in``, ``out``, ``front`` and ``other`` widths, and ``common``: every element's ``out`` reads
    the same ``core`` wires), the largest first; each other element's gates are ``core``'s."""
    fg, gates = T.first_gate, T.gates
    if T.n_inputs != n * width:
        raise ValueError(f"T has {T.n_inputs} input bits, not {n} elements of {width}")
    el, own = _element_of(T, n, width), _owner_of(T, n, out_width)
    ins: list[list[int]] = [[] for _ in range(n)]
    outs: list[list[int]] = [[] for _ in range(n)]
    for i in range(len(gates)):
        w = fg + i
        if own[w] != MANY:
            outs[own[w]].append(i)
        elif el[w] != MANY:
            ins[el[w]].append(i)
    xs = [list(range(2 + j * width, 2 + (j + 1) * width)) for j in range(n)]
    read_by: dict[int, set] = {}
    for i, (_, ops) in enumerate(gates):
        for o in ops:
            read_by.setdefault(o, set()).add(i)
    out_set = set(T.outputs)
    keys, cut = [], []
    for j in range(n):
        mine = set(ins[j])
        front = [fg + i for i in ins[j] if fg + i in out_set or read_by.get(fg + i, set()) - mine]
        word = T.outputs[j * out_width:(j + 1) * out_width]
        have = set(xs[j]) | set(front) | {fg + i for i in outs[j]} | {0, 1}
        other = _ordered([o for i in outs[j] for o in gates[i][1] if o not in have] + [w for w in word if w not in have])
        a, b = _sub(T, ins[j], xs[j], front), _sub(T, outs[j], xs[j] + front + other, word)
        cut.append((front, other, a, b))
        keys.append((tuple(a.gates), tuple(a.outputs), tuple(b.gates), tuple(b.outputs), len(other)))
    by: dict[tuple, list[int]] = {}
    for j, k in enumerate(keys):
        by.setdefault(k, []).append(j)
    classes, fronts, wanted, taken = [], [], [], set()
    for elements in sorted((e for e in by.values() if len(e) > 1), key=lambda e: (-len(e), e[0])):
        first = cut[elements[0]]
        common = all(cut[j][1] == first[1] for j in elements)
        classes.append({"elements": elements, "in": first[2], "out": first[3], "front": len(first[0]), "other": len(first[1]),
                        "common": common})
        taken |= {fg + i for j in elements for i in ins[j] + outs[j]}
        fronts += [w for j in elements for w in cut[j][0]]
        wanted += first[1] if common else [w for j in elements for w in cut[j][1]]
    done = {j for c in classes for j in c["elements"]}
    rest = [k for k in range(len(T.outputs)) if not (k < n * out_width and k // out_width in done)]
    core = _sub(T, [i for i in range(len(gates)) if fg + i not in taken], [w for x in xs for w in x] + fronts,
                wanted + [T.outputs[k] for k in rest])
    return {"classes": classes, "core": core, "rest": rest}


def emit_order(T: Traced) -> list[int]:
    """T's gates in the order `emit` writes them: by depth, then kind (`trace._KINDS`), then index."""
    fg, gates = T.first_gate, T.gates
    rank = {"and": 0, "xor": 1, "not": 2}
    depth = [0] * (fg + len(gates))
    for i, (_, ops) in enumerate(gates):
        depth[fg + i] = 1 + max(depth[o] for o in ops)
    return sorted(range(len(gates)), key=lambda i: (depth[fg + i], rank[gates[i][0]], i))


def rolled(name: str, version: int, statics: Sequence[str], build: Callable, signature: Callable, doc: str = "",
           word: Any = None, register: bool = True) -> CompositeDefinition:
    """`specialized`'s Definition with its circuit for bindings S, ``build(S)`` traced over ``signature(S)`` (one parameter, an
    Array whose elements' words are the first field of its return), written as calls (`roll`): for each class V of elements
    a batch of ``name``In{V}, then one Call of ``name``Core, then for each class a batch of ``name``Out{V}, each traced once
    per binding, so its description is a circuit per class of elements and the gates the elements share, never each
    element's. Its ``traced(S)`` is the whole circuit. It has no ``build``: a lowering that runs builders
    (`verity_flock.ir_lower`) runs its ``fold_build(S, C, *x)``, which writes the whole circuit's gates in `emit`'s order, so
    it lowers to what `specialized`'s Definition lowers to gate by gate."""
    if "V" in statics:
        raise ValueError(f"{name}: V names a class of elements")
    made: dict[tuple, Traced] = {}
    cuts: dict[tuple, dict] = {}

    def key(S) -> tuple:
        return tuple(getattr(S, k) for k in statics)

    def shape(S):
        params, ret = signature(S)
        (_, x), = params
        return x, ret, ret.fields[0].elem

    def traced(S) -> Traced:
        k = key(S)
        if k not in made:
            params, ret = signature(S)
            made[k] = trace(build(S), [t for _, t in params], ret, key=(name, version, dict(zip(statics, k))))
        return made[k]

    def cut(S) -> dict:
        k = key(S)
        if k not in cuts:
            x, _, q = shape(S)
            cuts[k] = roll(traced(S), x.n, x.elem.leaves, q.leaves)
        return cuts[k]

    def bits(n: int) -> Array:
        return Array(n, BL.BIT)

    def cls(S) -> dict:
        classes = cut(S)["classes"]
        if not 0 <= S.V < len(classes):
            raise ValueError(f"{name}: no class V={S.V} of {len(classes)}")
        return classes[S.V]

    def in_sig(S):
        return (("x", shape(S)[0].elem),), bits(cls(S)["front"])

    def core_sig(S):
        x, _, _ = shape(S)
        R = cut(S)
        ds = tuple((f"d{v}", Array(len(c["elements"]), bits(c["front"]))) for v, c in enumerate(R["classes"]))
        return (("x", x),) + ds, bits(len(R["core"].outputs))

    def out_sig(S):
        x, _, q = shape(S)
        c = cls(S)
        return (("x", x.elem), ("d", bits(c["front"])), ("c", bits(c["other"]))), q

    def part(suffix: str, at: tuple, sig: Callable, of: Callable, what: str) -> CompositeDefinition:
        def body(B, S, *c):
            params, ret = sig(S)
            return emit(of(S)[what], [t for _, t in params], ret, B, *c)
        return CompositeDefinition(name + suffix, version, tuple(statics) + at, sig, body, register=register,
                                   doc=f"{name}_v{version}'s {suffix}: `elementwise.roll`'s ``{what}``",
                                   conformance="exact-model-tested")

    In = part("In", ("V",), in_sig, cls, "in")
    Core = part("Core", (), core_sig, cut, "core")
    Out = part("Out", ("V",), out_sig, cls, "out")

    def body(B, S, x):
        _, ret = signature(S)
        R = cut(S)
        bk = {k: getattr(S, k) for k in statics}
        q = shape(S)[2].leaves

        def xs(c):
            el = c["elements"]
            return x if len(el) == x.type.n else Coll(Array(len(el), x.type.elem), _refs([r for j in el for r in x[j].refs]))

        ds = [B.batch(bind(In, V=v, **bk), xs(c), axes=(0,)) for v, c in enumerate(R["classes"])]
        core = B.call(bind(Core, **bk), x, *ds)
        refs: dict[int, Any] = {}
        at = 0
        for v, (c, d) in enumerate(zip(R["classes"], ds)):
            m, o = len(c["elements"]), c["other"]
            if c["common"]:
                cs, ax, at = Coll(bits(o), core.refs.slice(at, at + o)), None, at + o
            else:
                cs, ax, at = Coll(Array(m, bits(o)), core.refs.slice(at, at + m * o)), 0, at + m * o
            e = B.batch(bind(Out, V=v, **bk), xs(c), d, cs, axes=(0, 0, ax))
            for p, j in enumerate(c["elements"]):
                for b in range(q):
                    refs[j * q + b] = e.refs[p * q + b]
        for p, k in enumerate(R["rest"]):
            refs[k] = core.refs[at + p]
        return Coll(ret, _refs([refs[k] for k in range(ret.leaves)]))

    def fold_build(S, C, *elements):
        """The whole circuit's gates on ``C`` over the forms ``elements``, in `emit`'s order."""
        T = traced(S)
        W = [ZERO, ONE] + [f for e in elements for f in e] + [None] * len(T.gates)
        for i in emit_order(T):
            kind, ops = T.gates[i]
            W[T.first_gate + i] = C.AND(W[ops[0]], W[ops[1]]) if kind == "and" else W[ops[0]] ^ W[ops[1]] if kind == "xor" \
                else W[ops[0]].inv()
        out = [W[o] for o in T.outputs]
        widths = BL.elements(signature(S)[1])
        ends = [sum(widths[:i + 1]) for i in range(len(widths))]
        return [out[e - w:e] for w, e in zip(widths, ends)]

    d = CompositeDefinition(name, version, tuple(statics), signature, body, doc=doc, conformance="exact-model-tested",
                            register=register)
    d.traced = traced  # type: ignore[attr-defined]
    d.word = word  # type: ignore[attr-defined]
    d.fold_build = fold_build  # type: ignore[attr-defined]
    d.parts = (In, Core, Out)  # type: ignore[attr-defined]
    d.cut = cut  # type: ignore[attr-defined]
    return d


def on_bits(name: str, version: int, statics: Sequence[str], signature: Callable, body: Callable, word: Any,
            register: bool = True, lifted: Sequence[str] = ()) -> CompositeDefinition:
    """The Boolean composite ``name_v{version}{statics}`` with body ``body`` (whose Calls are Boolean Definitions), documented by
    ``body``'s docstring, whose word view is the word Definition ``word`` bound at the same statics (`trace.word_of`), each static
    in ``lifted`` (a Boolean Definition) read as its own word view. A static-free one names ``word`` itself: the switch
    (`verity.ml.definitions.boolean`) finds a word Definition without bindings by its id alone."""
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
