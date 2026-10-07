"""Boolean Definitions from circuit builders: run a builder over `forms`, then write what it computed as And/Xor/Not gates.

A builder is ``build(C, *elements) -> elements``: one argument per element of the Definition's parameters (`boolean.elements`,
in order), each a list of forms LSB first, as C-Flock's pieces take one argument per leaf of the word Definition; it returns a
list of forms (one element) or a list of elements, those of the return type in order.

`trace` runs it once on a `Circuit` whose forms remember how they were made (`TL`: the XOR of two forms, or the NOT of one),
and keeps the ANDs live in the outputs. Each AND becomes one `And2_v1`, except a committed copy (an AND with the constant 1,
``A = form, B = 1``, as `sha512.commit` writes one), which becomes no gate: its bit is its form's wire. Each AND operand and each
output is written as the XORs and NOTs the builder made it by (identical forms share one wire), or, for a form the builder did
not make from tracked forms, as a balanced XOR tree over its bits. So the gate circuit has exactly the builder's live ANDs (its
copies none), and its XOR and NOT gates are the builder's own linear work.

`emit` writes a traced circuit into a body, laid out level by level: each node is a batch of one gate kind over gates of equal
depth, its operands named explicitly. `definition` wraps a builder as a `CompositeDefinition` whose body is its traced circuit,
emitted; the body is traced on first use, not on import. A Boolean Definition names the word Definition it computes as its ``word`` (a Definition, or for one with statics a
function of its bindings); `word_of` reads it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, FunctionDefinition, SpecializedDefinition
from verity.ml.circuits.refs import Affine, Coll, Explicit, Ref, concat
from verity.ml.circuits.types import Array, Type

from .forms import ONE, Circuit, L

__all__ = ["TL", "Traced", "TracingCircuit", "definition", "emit", "trace", "word_of"]


class TL(L):
    """A form that remembers its making: ``src`` is ``None`` (a committed bit, or made without tracking), ``("x", a, b)`` or
    ``("n", a)``."""

    __slots__ = ("src",)

    def __init__(self, s=frozenset(), c=0, src=None):
        super().__init__(s, c)
        self.src = src

    def __xor__(self, o):
        return TL(self.s ^ o.s, self.c ^ o.c, ("x", self, o))

    def __rxor__(self, o):
        return TL(o.s ^ self.s, o.c ^ self.c, ("x", o, self))

    def inv(self):
        return TL(self.s, self.c ^ 1, ("n", self))


class TracingCircuit(Circuit):
    """A `Circuit` whose committed bits are `TL` forms; it refuses hints and table reads (not Boolean gates)."""

    def _new(self, kind, data):
        if kind not in ("in", "and"):
            raise TypeError(f"a {kind!r} bit has no Boolean gate")
        w = super()._new(kind, data)
        return TL(w.s, 0, None)

    def commit(self, word):
        """The word itself: a commit (`forms.Circuit.commit`) is a row of the lowering, not a gate."""
        return list(word)


_KINDS = ("and", "xor", "not")


@dataclass
class Traced:
    """The gate circuit: wires 0 and 1 are the constants, then the input bits, then one wire per gate. ``gates[i]`` is
    (kind, operand wires) and makes wire ``first_gate + i``; ``outputs`` is each output bit's wire."""

    n_inputs: int
    gates: list[tuple[str, tuple[int, ...]]]
    outputs: list[int]
    counts: dict[str, int]

    @property
    def first_gate(self) -> int:
        return 2 + self.n_inputs


def _elements_in(C: Circuit, params: Sequence[Type]) -> list[list[L]]:
    out = []
    for i, t in enumerate(params):
        for j, w in enumerate(BL.elements(t)):
            out.append(C.input(f"p{i}e{j}", w))
    return out


def _flat_out(res: Any, ret: Type) -> list[L]:
    widths = BL.elements(ret)
    if res and isinstance(res[0], L):
        res = [res]
    if [len(e) for e in res] != widths:
        raise ValueError(f"the builder returned elements of widths {[len(e) for e in res]}, the type needs {widths}")
    return [f for e in res for f in e]


def trace(build: Callable, params: Sequence[Type], ret: Type, *, cse: bool = True) -> Traced:
    C = TracingCircuit(cse=cse)
    ins = _elements_in(C, params)
    outs = _flat_out(build(C, *ins), ret)
    C.output("out", outs)
    live = C.live()

    gates: list[tuple[str, tuple[int, ...]]] = []
    wire_of_bit: dict[int, int] = {}
    n_in = 0
    for e in ins:
        for f in e:
            (v,) = f.s
            wire_of_bit[v] = 2 + n_in
            n_in += 1
    first = 2 + n_in
    by_key: dict[tuple, int] = {}
    clean: dict[int, bool] = {}

    def gate(kind: str, *ops: int) -> int:
        gates.append((kind, ops))
        return first + len(gates) - 1

    def tracked(f: L) -> bool:
        """Whether the form's making reaches only live bits through tracked steps."""
        stack = [f]
        while stack:
            g = stack[-1]
            if id(g) in clean:
                stack.pop()
                continue
            if len(g.s) <= 1:
                clean[id(g)] = g.is_const() or next(iter(g.s)) in wire_of_bit
                stack.pop()
                continue
            src = getattr(g, "src", None)
            if src is None:
                clean[id(g)] = False
                stack.pop()
                continue
            todo = [h for h in src[1:] if id(h) not in clean]
            if todo:
                stack.extend(todo)
                continue
            clean[id(g)] = all(clean[id(h)] for h in src[1:])
            stack.pop()
        return clean[id(f)]

    def tree(f: L) -> int:
        ws = sorted(wire_of_bit[v] for v in f.s)
        while len(ws) > 1:
            nxt = [gate("xor", ws[i], ws[i + 1]) for i in range(0, len(ws) - 1, 2)]
            ws = nxt + ws[len(ws) - len(ws) % 2:]
        return gate("not", ws[0]) if f.c else ws[0]

    def wire(f: L) -> int:
        if f.is_const():
            return f.c
        key = (f.s, f.c)
        hit = by_key.get(key)
        if hit is not None:
            return hit
        if len(f.s) == 1:
            w = wire_of_bit[next(iter(f.s))]
            if f.c:
                w = by_key[key] = gate("not", w)
            return w
        if not tracked(f):
            w = by_key[key] = tree(f)
            return w
        stack = [f]
        while stack:
            g = stack[-1]
            k = (g.s, g.c)
            if g.is_const() or k in by_key or len(g.s) == 1:
                stack.pop()
                continue
            src = g.src
            todo = [h for h in src[1:] if not (h.is_const() or (h.s, h.c) in by_key or len(h.s) == 1)]
            if todo:
                stack.extend(todo)
                continue
            stack.pop()
            if src[0] == "n":
                by_key[k] = gate("not", _w(src[1]))
            else:
                a, b = src[1], src[2]
                if a.is_const() or b.is_const():
                    x, cst = (b, a) if a.is_const() else (a, b)
                    by_key[k] = gate("not", _w(x)) if cst.c else _w(x)
                else:
                    by_key[k] = gate("xor", _w(a), _w(b))
        return by_key[key]

    def _w(h: L) -> int:
        if h.is_const():
            raise AssertionError("a constant operand reached a gate")
        k = (h.s, h.c)
        hit = by_key.get(k)
        if hit is not None:
            return hit
        w = wire_of_bit[next(iter(h.s))]
        return by_key.setdefault(k, gate("not", w) if h.c else w)

    for v in sorted(v for v in live if C.kind[v] == "and"):
        a, b = C.data[v]
        if ONE.key() in (a.key(), b.key()):
            wire_of_bit[v] = wire(b if a.key() == ONE.key() else a)
            continue
        wire_of_bit[v] = gate("and", wire(a), wire(b))
    outputs = [wire(f) for f in outs]
    gates, outputs = _prune(first, gates, outputs)
    counts = {k: sum(1 for g, _ in gates if g == k) for k in _KINDS}
    return Traced(n_in, gates, outputs, counts)


def _prune(first: int, gates: list[tuple[str, tuple[int, ...]]], outputs: list[int]):
    """The gates some output depends on, renumbered: a form's sources can be written before the same form turns up by another
    making, and are then read by nothing. Every live AND is read (its bit is in a consumer's form), so none goes."""
    need = [False] * len(gates)
    stack = [o - first for o in outputs if o >= first]
    while stack:
        i = stack.pop()
        if need[i]:
            continue
        need[i] = True
        stack.extend(o - first for o in gates[i][1] if o >= first)
    new, kept = {}, []
    for i, (kind, ops) in enumerate(gates):
        if need[i]:
            new[first + i] = first + len(kept)
            kept.append((kind, tuple(new.get(o, o) for o in ops)))
    return kept, [new.get(o, o) for o in outputs]


def _refs(refs: list[Ref]) -> Any:
    """A reference sequence: contiguous runs as `Affine` parts, the rest explicit."""
    parts, loose = [], []
    i = 0
    while i < len(refs):
        s, k, j = refs[i]
        n = 1
        while i + n < len(refs) and refs[i + n] == (s, k, j + n):
            n += 1
        if n >= 4:
            if loose:
                parts.append(Explicit(tuple(loose)))
                loose = []
            parts.append(Affine(s, k, j, n))
        else:
            loose.extend(refs[i:i + n])
        i += n
    if loose:
        parts.append(Explicit(tuple(loose)))
    return concat(parts)


def emit(T: Traced, params: Sequence[Type], ret: Type, B, *colls) -> Coll:
    """The circuit ``T`` written into the body ``B`` over operands ``colls`` of types ``params``: a batch per gate kind and depth,
    its outputs as one ``ret``."""
    ref: dict[int, Ref] = {}
    w = 2
    for i, t in enumerate(params):
        for j in range(t.leaves):
            ref[w] = ("p", i, j)
            w += 1
    if any(o in (0, 1) for o in T.outputs):
        for b in (0, 1):
            if b in T.outputs:
                c = B.call(BL.constant(b))
                ref[b] = c.refs[0]
    depth = [0] * (T.first_gate + len(T.gates))
    for i, (_, ops) in enumerate(T.gates):
        depth[T.first_gate + i] = 1 + max(depth[o] for o in ops)
    order = sorted(range(len(T.gates)), key=lambda i: (depth[T.first_gate + i], _KINDS.index(T.gates[i][0]), i))
    g = 0
    while g < len(order):
        i0 = order[g]
        kind, d = T.gates[i0][0], depth[T.first_gate + i0]
        h = g
        while h < len(order) and T.gates[order[h]][0] == kind and depth[T.first_gate + order[h]] == d:
            h += 1
        group = order[g:h]
        if kind == "not":
            out = B.batch(BL.NOT, Coll(Array(len(group), BL.BIT), _refs([ref[T.gates[i][1][0]] for i in group])), axes=(0,))
        else:
            # operand-major (`Array<2, Array<m, Value<1>>>` over axis 1), so each operand's column keeps its runs
            ops = _refs([ref[T.gates[i][1][a]] for a in range(2) for i in group])
            out = B.batch(BL.AND if kind == "and" else BL.XOR, Coll(Array(2, Array(len(group), BL.BIT)), ops), axes=(1,))
        for m, i in enumerate(group):
            ref[T.first_gate + i] = out.refs[m]
        g = h
    return Coll(ret, _refs([ref[o] for o in T.outputs]))


def definition(name: str, version: int, build: Callable, params: Sequence[tuple[str, Type]], ret: Type, *, word: Any = None,
               doc: str = "", conformance: str = "exact-model-tested", register: bool = True) -> CompositeDefinition:
    """The Boolean Definition ``name_v{version}`` that ``build`` computes (traced once, on first use of its body), computing the
    word Definition ``word``."""
    types = [t for _, t in params]
    traced: list[Traced] = []

    def body(B, S, *colls):
        if not traced:
            traced.append(trace(build, types, ret))
        return emit(traced[0], types, ret, B, *colls)

    d = CompositeDefinition(name, version, (), lambda S: (tuple(params), ret), body, doc=doc, conformance=conformance,
                            register=register)
    d.build = build  # type: ignore[attr-defined]
    d.traced = lambda: traced[0] if traced else trace(build, types, ret)  # type: ignore[attr-defined]
    d.word = word  # type: ignore[attr-defined]
    return d


def word_of(target: Any) -> FunctionDefinition:
    """The word Definition a Boolean Definition computes (through `boolean.word_type`)."""
    d, bindings = (target.defn, target.bindings) if isinstance(target, SpecializedDefinition) else (target, {})
    w = getattr(d, "word", None)
    if w is None:
        raise KeyError(f"{getattr(target, 'id', target)!r} names no word Definition")
    return w if isinstance(w, FunctionDefinition) else w(bindings)
