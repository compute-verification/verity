"""A Boolean Definition over an Array whose elements' words are computed alike, written as calls: the next version of a
`specialized` one whose description grows with its elements.

    roll(T, n, width, out_width)    T cut into its classes of elements (each a circuit that reads one element, `In`, and one
                                    that only its word reads, `Out`) and the gates they share (`Core`)
    rolled(name, version, ...)      `elementwise.specialized`'s Definition written as batches of each class's `In` and `Out`
                                    and one Call of `Core`, lowered (``fold_build``) as the whole circuit's gates in `emit`'s
                                    order, so its lowered circuit's description is `specialized`'s

A group quantiser decodes each of its words for its scale and for its quantised word, so most of its circuit is per word and
alike; its description, and `verity_flock.struct_lower`'s template of it, are then a class's circuit once and the shared gates,
not each word's.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import ONE, ZERO
from verity.circuits.boolean.trace import Traced, _refs, emit, trace
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import Coll
from verity.circuits.types import Array

from .elementwise import _sub

__all__ = ["roll", "rolled"]


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
    """Each wire's output element: the one element of T's first ``n * out_width`` outputs every reader of it reaches, else
    MANY."""
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
    """`elementwise.specialized`'s Definition with its circuit for bindings S, ``build(S)`` traced over ``signature(S)`` (one
    parameter, an Array whose elements' words are the first field of its return), written as calls (`roll`): for each class V
    of elements a batch of ``name``In{V}, then one Call of ``name``Core, then for each class a batch of ``name``Out{V}, each
    traced once per binding, so its description is a circuit per class of elements and the gates the elements share, never
    each element's. Its ``traced(S)`` is the whole circuit. It has no ``build``: a lowering that runs builders
    (`verity_flock.ir_lower`) runs its ``fold_build(S, C, *x)``, which writes the whole circuit's gates in `emit`'s order, so
    it lowers to what `elementwise.specialized`'s Definition lowers to gate by gate."""
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
                                   doc=f"{name}_v{version}'s {suffix}: `roll`'s ``{what}``",
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
