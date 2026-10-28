"""Function definitions, static bindings, and composite bodies (IR §3, §4).

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

* `PrimitiveDefinition`: registered id `Name_vN`, a signature whose return type is one `Value<w>`, and a *total*
  Python evaluator on encodings (Python ints holding the bit pattern).
* `CompositeDefinition`: static parameters (dimensions and function slots), a signature that may depend on the
  statics, and a body builder producing explicit dataflow over calls. The composite definition *is* the
  decomposition.
* `bind(F, **statics)` -> `SpecializedDefinition`: an explicit binding record. Its body is built once per
  specialization and cached with gate counts and offsets.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .ports import bind_ports
from .refs import Affine, Coll, Concat, Explicit, Refs, Strided, concat
from .types import Array, Record, Tuple as TupleT, Type, Value, field_at, field_starts, field_types, range_widths, remove_axis

StaticValue = Any  # int | str | tuple[int, ...] | FunctionDefinition


class Registry:
    """Registered function definitions by id `Name_vN`."""

    def __init__(self) -> None:
        self.defs: dict[str, "FunctionDefinition"] = {}

    def add(self, d: "FunctionDefinition") -> "FunctionDefinition":
        prev = self.defs.get(d.id)
        if prev is not None and prev is not d:
            raise ValueError(f"registry already has a different definition for {d.id}")
        self.defs[d.id] = d
        return d

    def __getitem__(self, fid: str) -> "FunctionDefinition":
        return self.defs[fid]

    def __contains__(self, fid: str) -> bool:
        return fid in self.defs


REGISTRY = Registry()


class FunctionDefinition:
    name: str
    version: int

    @property
    def id(self) -> str:
        return f"{self.name}_v{self.version}"


#: what a table a primitive reads by index is: a ``library`` object (an operation's own definition, `verity.ml.circuits.library`) or a
#: ``registered`` input of the program, which the primitive still holds while it declares it
TABLE_KINDS = ("library", "registered")


@dataclass(eq=False)
class PrimitiveDefinition(FunctionDefinition):
    """`tables`: every table the primitive's definition reads by index (its evaluator, or the circuit that proves it), as
    ``(name, kind)`` with a kind of `TABLE_KINDS`. Such a table is no gate of the program, so only its declaration shows it
    (`constants.program_data`). `ports`: its port types (`ports.Ports`, PROTOCOL.md §12), None when it declares none."""

    name: str
    version: int
    params: tuple[tuple[str, Type], ...]
    ret: Value
    evaluate: Callable[..., int] = field(repr=False)
    doc: str = ""
    conformance: str = "not-yet-established"  # exact-model-tested | model-untested | not-yet-established
    register: bool = True
    tables: tuple[tuple[str, str], ...] = ()
    ports: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.ret, Value):
            raise TypeError(f"a primitive returns exactly one Value<w>; {self.id} declares {self.ret!r}")
        self.params = tuple((n, t) for n, t in self.params)
        self.tables = tuple((n, k) for n, k in self.tables)
        bad = [k for _n, k in self.tables if k not in TABLE_KINDS]
        if bad:
            raise ValueError(f"{self.id}: a table is a {' or '.join(TABLE_KINDS)} table, not {bad}")
        self.port_types = bind_ports(self.id, self.params, self.ret, self.ports, primitive=True)
        if self.register:
            REGISTRY.add(self)

    # A primitive is a trivially specialized function: no statics, one gate.
    @property
    def statics(self) -> dict:
        return {}

    @property
    def gates(self) -> int:
        return 1

    @property
    def out_leaves(self) -> int:
        return 1

    def signature(self) -> tuple[tuple[tuple[str, Type], ...], Type]:
        return self.params, self.ret

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"<prim {self.id}>"


@dataclass(eq=False)
class CompositeDefinition(FunctionDefinition):
    """`signature(S)` -> (params, ret); `body(B, S, *args)` -> Coll of type ret. `S` is a namespace over the
    statics (attribute access). Statics with default `None` must be bound before use. `refuse(bindings)`, given the
    statics bound so far, says why `bind` refuses them (a binding the body does not cover), or returns None. `ports`: its
    port types (`ports.Ports`, PROTOCOL.md §12), or a function of `S` giving them; None when it declares none."""

    name: str
    version: int
    statics: tuple[str, ...]
    signature: Callable[[Any], tuple[Sequence[tuple[str, Type]], Type]] = field(repr=False)
    body: Callable[..., Coll] = field(repr=False)
    doc: str = ""
    conformance: str = "not-yet-established"
    register: bool = True
    refuse: Callable[[dict], str | None] | None = field(default=None, repr=False)
    ports: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.statics = tuple(self.statics)
        self._cache: dict[tuple, "SpecializedDefinition"] = {}
        if self.register:
            REGISTRY.add(self)

    def bind(self, **kw: StaticValue) -> "SpecializedDefinition":
        return bind(self, **kw)

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"<composite {self.name}_v{self.version} statics={self.statics}>"


class _Statics:
    def __init__(self, d: dict) -> None:
        self.__dict__.update(d)

    def __getattr__(self, k: str):
        raise AttributeError(f"static parameter {k!r} is not bound")


def _static_key(v: StaticValue):
    if isinstance(v, SpecializedDefinition):
        return ("spec", v.base_id, v.key)
    if isinstance(v, FunctionDefinition):
        return ("fn", v.id)
    if isinstance(v, (list, tuple)):
        return ("tuple",) + tuple(_static_key(x) for x in v)
    if isinstance(v, bool):  # True == 1 and hash(True) == hash(1): untagged, FLAG=1 would hit FLAG=True's entry
        return ("bool", v)
    if isinstance(v, (int, str)) or v is None:
        return v
    if isinstance(v, float):
        return ("float", v.hex())
    if isinstance(v, dict):
        return ("dict",) + tuple((k, _static_key(v[k])) for k in sorted(v))
    raise TypeError(f"unsupported static value {v!r}")


class IdentityConflict(ValueError):
    """`E_IDENTITY`: two different definitions claim the same fully qualified id within one program. Never
    resolved first-wins — neither by the binding cache nor by the codec."""


def same_definition(a: "FunctionDefinition", b: "FunctionDefinition") -> bool:
    """Whether two definition objects with equal ids denote the same definition: primitives by signature (the
    evaluators are code; the registry pins them); complete specializations (and static-free composites, through
    their unique specialization) by their encoded definition — signature, binding record and body; composites
    with unbound statics only when they share the signature/body builders."""
    if a is b:
        return True
    if type(a) is not type(b) or a.id != b.id:
        return False
    if isinstance(a, PrimitiveDefinition):
        return a.params == b.params and a.ret == b.ret and a.port_types == b.port_types
    if isinstance(a, CompositeDefinition):
        if a.statics != b.statics:
            return False
        if a.signature is b.signature and a.body is b.body:
            return True
        return not a.statics and same_definition(bind(a), bind(b))
    assert isinstance(a, SpecializedDefinition)
    if a.key != b.key or not _consistent_statics(a.bindings, b.bindings):
        return False
    # (for a static-free definition `bind(defn)` is this very specialization: compare the objects, not recurse)
    if a.defn is b.defn or (a.defn.statics and same_definition(a.defn, b.defn)):
        return True
    if not (a.complete and b.complete):
        return False
    from .codec import _encode_definition  # the descriptor's view of a definition: signature + record + body
    return _encode_definition(a) == _encode_definition(b)


def _consistent_statics(a, b) -> bool:
    """Key-equal static values whose function references all denote the same definitions."""
    if isinstance(a, FunctionDefinition):
        return isinstance(b, FunctionDefinition) and same_definition(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_consistent_statics(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return set(a) == set(b) and all(_consistent_statics(a[k], b[k]) for k in a)
    return True  # scalars: the key already compared them


def bind(F: "CompositeDefinition | SpecializedDefinition", **kw: StaticValue) -> "SpecializedDefinition":
    """Explicit binding record. Partial binds compose; all statics must be bound before the body is built. A record
    the composite's `refuse` names a reason for is refused (`ValueError`) before it is cached.

    Specializations are shared through a per-definition cache keyed by the binding record. Function-valued
    statics are keyed by id, so a cache hit whose function references are *different* definitions under the
    same id is a conflict, not a silent reuse of the first body."""
    if isinstance(F, SpecializedDefinition):
        merged = dict(F.bindings)
        merged.update(kw)
        F = F.defn
    else:
        merged = dict(kw)
    for k in merged:
        if k not in F.statics:
            raise TypeError(f"{F.id} has no static parameter {k!r}")
    key = tuple((k, _static_key(merged[k])) for k in F.statics if k in merged)
    sp = F._cache.get(key)
    if sp is None:
        why = F.refuse(merged) if F.refuse is not None else None
        if why:
            raise ValueError(f"{F.id} refuses this binding: {why}")
        sp = SpecializedDefinition(F, merged, key)
        F._cache[key] = sp
    elif not _consistent_statics(sp.bindings, merged):
        raise IdentityConflict(f"E_IDENTITY: {sp.id} is already bound to a different definition under the same "
                               f"function id (two bodies claim one fully qualified specialization)")
    return sp


@dataclass(eq=False)
class Node:
    """One body node: a primitive call, a composite call, a `batch`, or a `scan`. Each node denotes one
    interval of gates in the canonical layout."""

    form: str  # "primitive" | "call" | "batch" | "scan"
    fn: "PrimitiveDefinition | SpecializedDefinition"
    args: tuple[Coll, ...]  # call/prim: parameter collections; batch: batched arguments; scan: (init, *xs, *shared)
    out: Type
    axes: tuple[int | None, ...] = ()  # batch only
    n: int = 1  # batch members / scan iterations
    n_xs: int = 0  # scan only: number of per-iteration sequence args
    name: str | None = None  # optional annotation (not part of the program digest)

    @property
    def gates(self) -> int:
        if self.form == "primitive":
            return 1
        if self.form == "call":
            return self.fn.gates
        return self.n * self.fn.gates

    @property
    def out_leaves(self) -> int:
        return self.out.leaves


@dataclass(eq=False)
class Body:
    params: tuple[tuple[str, Type], ...]
    ret_type: Type
    nodes: list[Node]
    ret: Refs
    offsets: list[int]  # gate offset of node k within this function; len == len(nodes) + 1

    @property
    def gates(self) -> int:
        return self.offsets[-1]

    def node_at(self, g: int) -> int:
        """Index of the node whose interval contains function-relative gate `g` (division/bisection)."""
        k = bisect.bisect_right(self.offsets, g) - 1
        if k < 0 or k >= len(self.nodes):
            raise IndexError(g)
        return k


_UNBOUND = object()


class SpecializedDefinition(FunctionDefinition):
    """A composite definition with (some) statics bound. Fully-bound specializations have a cached body."""

    def __init__(self, defn: CompositeDefinition, bindings: dict, key: tuple) -> None:
        self.defn = defn
        self.bindings = dict(bindings)
        self.key = key
        self._body: Body | None = None
        self._port_types: Any = _UNBOUND
        self._id: str | None = None

    @property
    def name(self) -> str:
        return self.defn.name

    @property
    def version(self) -> int:
        return self.defn.version

    @property
    def base_id(self) -> str:
        """`Name_vN` of the definition being specialised."""
        return f"{self.name}_v{self.version}"

    @property
    def id(self) -> str:
        """Function identity = definition + static bindings: `Name_vN{K=..,..}` (bare `Name_vN` if static-free). A complete
        specialization's is formatted once: its bindings are final (the decoder fills an incomplete one's)."""
        if not self.defn.statics:
            return self.base_id
        if self._id is not None:
            return self._id
        parts = [f"{k}={_fmt_static(self.bindings[k])}" for k in self.defn.statics if k in self.bindings]
        out = self.base_id + "{" + ",".join(parts) + "}"
        if self.complete:
            self._id = out
        return out

    @property
    def statics(self) -> dict:
        return self.bindings

    @property
    def complete(self) -> bool:
        return all(k in self.bindings for k in self.defn.statics)

    def bind(self, **kw: StaticValue) -> "SpecializedDefinition":
        return bind(self, **kw)

    def S(self) -> _Statics:
        return _Statics(self.bindings)

    def signature(self) -> tuple[tuple[tuple[str, Type], ...], Type]:
        if self._body is not None:
            return self._body.params, self._body.ret_type
        if not self.complete:
            missing = [k for k in self.defn.statics if k not in self.bindings]
            raise TypeError(f"{self.id} is missing static bindings {missing}")
        params, ret = self.defn.signature(self.S())
        return tuple((n, t) for n, t in params), ret

    @property
    def body(self) -> Body:
        if self._body is None:
            self._body = _build_body(self)
        return self._body

    @property
    def gates(self) -> int:
        return self.body.gates

    @property
    def out_leaves(self) -> int:
        return self.body.ret_type.leaves

    def set_body(self, body: Body) -> None:  # used by the decoder
        self._body = body

    @property
    def port_types(self) -> Any:
        """Its declared ports bound to its signature (`ports.PortTypes`), None when it declares none."""
        if self._port_types is _UNBOUND:
            decl = self.defn.ports
            if callable(decl):
                decl = decl(self.S())
            self._port_types = None if decl is None else bind_ports(self.id, *self.signature(), decl)
        return self._port_types

    def set_port_types(self, pt: Any) -> None:  # used by the decoder and the lowering
        self._port_types = pt

    def intrinsic(self) -> "IntrinsicSummary":
        """Intrinsic summary (IR §D.1): facts reusable for every instance of this specialization — gate
        count, interface types, node offsets. Contextual facts (consumers, cuts, reachability, protected
        outputs) are never cached here."""
        b = self.body
        return IntrinsicSummary(self.id, b.gates, tuple(t for _, t in b.params), b.ret_type, tuple(b.offsets),
                                len(b.nodes))


@dataclass(frozen=True)
class IntrinsicSummary:
    id: str
    gates: int
    param_types: tuple[Type, ...]
    ret_type: Type
    offsets: tuple[int, ...]
    nodes: int

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        b = ",".join(f"{k}={_fmt_static(v)}" for k, v in self.bindings.items())
        return f"<{self.id}[{b}]>"


def _fmt_static(v):
    if isinstance(v, FunctionDefinition):
        return v.id
    if isinstance(v, (list, tuple)):
        return "(" + ",".join(_fmt_static(x) for x in v) + ")"
    if isinstance(v, dict):
        import hashlib, json
        return "{" + hashlib.sha512(json.dumps(_plain_static(v), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12] + "}"
    return repr(v)


def _plain_static(v):
    """A dict-valued static as plain JSON for the readable id's hash: function references (which may be nested in
    a record/tuple bundle) become `{"fn": id}`; everything else is unchanged, so dicts without functions hash as
    before."""
    if isinstance(v, FunctionDefinition):
        return {"fn": v.id}
    if isinstance(v, (list, tuple)):
        return [_plain_static(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain_static(x) for k, x in v.items()}
    return v


def _build_body(sp: SpecializedDefinition) -> Body:
    params, ret_type = sp.signature()
    B = Builder(params, ret_type)
    args = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
    out = sp.defn.body(B, sp.S(), *args)
    return B.finish(out)


def _first_other(T: Type, base: int, n: int, w: int) -> int | None:
    """The first of leaves [base, base + n) of `T` whose width is not `w` (as an offset from `base`), None when there is none."""
    if range_widths(T, base, base + n) == {w}:
        return None
    if n == 1:
        return 0
    h = n // 2
    k = _first_other(T, base, h, w)
    if k is not None:
        return k
    k = _first_other(T, base + h, n - h, w)
    return None if k is None else h + k


def _span(p: Affine | Strided) -> tuple[int, int]:
    """The least and the greatest leaf a non-empty `Affine` or `Strided` part names."""
    if isinstance(p, Affine):
        return p.base, p.base + p.count - 1
    q, last = divmod(p.count - 1, p.inner)
    vals = [p.base + a * p.outer_stride + b * p.inner_stride
            for a, b in [(0, 0), (0, min(p.inner, p.count) - 1), (q, 0), (q, last)] + ([(q - 1, p.inner - 1)] if q else [])]
    return min(vals), max(vals)


def _cover(t: Type, lo: int, hi: int) -> tuple[Type, int]:
    """The smallest subtree of `t` holding leaves [lo, hi), and its first leaf's offset in `t`."""
    off = 0
    while True:
        if isinstance(t, Array) and t.elem.leaves:
            L = t.elem.leaves
            i = (lo - off) // L
            if (hi - 1 - off) // L != i:
                return t, off
            t, off = t.elem, off + i * L
        elif isinstance(t, (Record, TupleT)) and 0 <= lo - off < t.leaves:
            starts = field_starts(t)
            i = field_at(t, lo - off)
            if hi - off > starts[i + 1]:
                return t, off
            t, off = field_types(t)[i], off + starts[i]
        else:
            return t, off


def _residues(stride: int, count: int, L: int, cap: int = 4096) -> set[int] | None:
    """{k * stride mod L : k < count}, or None when it has more than `cap` members."""
    g = math.gcd(stride, L)
    if count >= L // g:
        return set(range(0, L, g)) if L // g <= cap else None
    return {k * stride % L for k in range(count)} if count <= cap else None


class _Widths:
    """The leaf widths a body's references name, for `Builder.finish`'s reinterpretation check (`E_REINTERPRET`). A reference
    that names no leaf is not this check's: the decoder and the checkers refuse it. Nor is one to node `below` or later, which
    node `below`'s arguments cannot read in dataflow order: such a descriptor decodes, and the checkers name the defect
    (PROTOCOL.md §4.4)."""

    def __init__(self, params: tuple[tuple[str, Type], ...], nodes: list["Node"]) -> None:
        self.params, self.nodes = params, nodes
        self.below = len(nodes)
        self.forward = False  # a reference to node `below` or later was passed over: the `PartLog` memo is off from then on
        self.token = object()  # this check's entry in a `PartLog`'s memo
        self._src: dict[tuple[str, int], tuple[Type | None, int | None, int]] = {}
        self._same: dict[tuple[int, int], bool] = {}

    def src(self, s: str, i: int) -> tuple[Type | None, int | None, int]:
        """(type, homogeneous width, leaves) of parameter or node `i`; (None, None, 0) when there is none, or when `i` is node
        `below` or later."""
        if s == "n" and i >= self.below:
            self.forward = True
            return (None, None, 0)
        hit = self._src.get((s, i))
        if hit is None:
            T = (self.params[i][1] if s == "p" and 0 <= i < len(self.params) else
                 self.nodes[i].out if s == "n" and 0 <= i < len(self.nodes) else None)
            hit = self._src[(s, i)] = (None, None, 0) if T is None else (T, T.homogeneous_width(), T.leaves)
        return hit

    def leaf(self, s: str, i: int, j: int) -> int | None:
        T, h, n = self.src(s, i)
        if T is None or not 0 <= j < n:
            return None
        if h is not None:
            return h
        (w,) = range_widths(T, j, j + 1)
        return w

    def other(self, p: Refs, w: int) -> int | None:
        """The first leaf of part `p` (no `Concat`) whose width is not `w`, as an offset into `p`; None when there is none."""
        if isinstance(p, Explicit):
            return next((k for k, (s, i, j) in enumerate(p.items) if self.leaf(s, i, j) not in (None, w)), None)
        if not p.count:
            return None
        T, h, n = self.src(p.space, p.idx)  # type: ignore[attr-defined]
        if T is None or h == w:
            return None
        lo, hi = _span(p)  # type: ignore[arg-type]
        if lo < 0 or hi >= n:
            return None
        if h is not None:
            return 0
        if isinstance(p, Affine):
            return _first_other(T, p.base, p.count, w)
        if range_widths(T, lo, hi + 1) == {w} or self._periodic(T, p, lo, hi, w):  # type: ignore[arg-type]
            return None
        return next((k for k, (s, i, j) in enumerate(p) if self.leaf(s, i, j) != w), None)

    @staticmethod
    def _periodic(T: Type, p: Strided, lo: int, hi: int, w: int) -> bool:
        """Every leaf `p` names is of width `w`, by the offsets it reads within the elements of the array that holds them all."""
        node, off = _cover(T, lo, hi + 1)
        if not isinstance(node, Array):
            return False
        L = node.elem.leaves
        A = _residues(p.outer_stride, -(-p.count // p.inner), L)
        B = _residues(p.inner_stride, p.inner, L)
        if A is None or B is None or len(A) * len(B) > 4096:
            return False
        r0 = (p.base - off) % L
        return all(range_widths(node.elem, r, r + 1) == {w} for r in {(r0 + a + b) % L for a in A for b in B})

    def misread(self, t: Type, pos: int, p: Refs) -> int | None:
        """The first leaf of part `p` (no `Concat`), read at leaves pos.. of a `t`, whose width is not the one `t` gives that leaf,
        as an offset into `p`; None when there is none."""
        n = p.count
        if not n:
            return None
        ws = range_widths(t, pos, pos + n)
        if len(ws) == 1:
            return self.other(p, next(iter(ws)))
        if isinstance(p, Affine) and self._same_layout(t, pos, p):
            return None
        node, off = _cover(t, pos, pos + n)
        if isinstance(node, Array):
            L = node.elem.leaves
            first = (pos - off) // L
            bounds = [off + k * L for k in range(first, (pos + n - 1 - off) // L + 2)]
        else:
            starts = field_starts(node)  # type: ignore[arg-type]
            first, last = field_at(node, pos - off), field_at(node, pos + n - 1 - off)  # type: ignore[arg-type]
            bounds = [off + s for s in starts[first:last + 2]]
        for a, b in zip(bounds, bounds[1:]):
            a, b = max(a, pos), min(b, pos + n)
            if a < b:
                k = self.misread(t, a, p.slice(a - pos, b - pos))
                if k is not None:
                    return a - pos + k
        return None

    def _same_layout(self, t: Type, pos: int, p: Affine) -> bool:
        """Leaves pos.. of `t` have the widths of the leaves `p` names (True also when `p` names no leaf), by structure."""
        T, _h, nT = self.src(p.space, p.idx)
        n = p.count
        if T is None or p.base < 0 or p.base + n > nT:
            return True
        if n <= 64:
            return all(range_widths(t, pos + k, pos + k + 1) == range_widths(T, p.base + k, p.base + k + 1) for k in range(n))
        a, ao = _cover(t, pos, pos + n)
        b, bo = _cover(T, p.base, p.base + n)
        if pos - ao == p.base - bo and (a is b or a == b):
            return True
        if isinstance(a, Array) and isinstance(b, Array) and a.elem.leaves == b.elem.leaves:
            L = a.elem.leaves
            return (pos - ao) % L == (p.base - bo) % L and self._same_widths(a.elem, b.elem)
        return False

    def _same_widths(self, E: Type, F: Type) -> bool:
        key = (id(E), id(F))
        if key not in self._same:
            self._same[key] = E is F or E == F or (E.leaves <= 4096 and E.leaf_widths() == F.leaf_widths())
        return self._same[key]

    def first(self, c: Coll) -> tuple[int, tuple[str, int, int]] | None:
        """(position, reference) of the first leaf of `c` that names a leaf of another width than `c.type` gives it, None when
        there is none. A `Concat` cut from a `PartLog` costs only its parts past the longest cut checked at that width."""
        t = c.type

        def walk(r: Refs, pos: int) -> tuple[int, tuple[str, int, int]] | None:
            if not isinstance(r, Concat):
                k = self.misread(t, pos, r)
                return None if k is None else (pos + k, r[k])
            first = 0
            if r.prefix is not None and not self.forward:
                log, j = r.prefix
                ws = range_widths(t, pos, pos + log.starts[j])
                if len(ws) == 1:
                    (w,) = ws
                    owner, done = log.memo.get("defs.widths", (None, None))
                    if owner is not self.token:
                        done = {}
                        log.memo["defs.widths"] = (self.token, done)
                    for k in range(done.get(w, 0), j):
                        hit = walk(log.parts[k], pos + log.starts[k])
                        if hit is not None:
                            return hit
                    if not self.forward:
                        done[w] = max(done.get(w, 0), j)
                    first = j
            for k in range(first, len(r.parts)):
                hit = walk(r.parts[k], pos + r.starts[k])
                if hit is not None:
                    return hit
            return None

        return walk(c.refs, 0) if c.refs.count else None

    def refuse(self, what: str, t: Type, hit: tuple[int, tuple[str, int, int]]) -> TypeError:
        pos, (s, i, j) = hit
        (want,) = range_widths(t, pos, pos + 1)
        src = f"parameter {self.params[i][0]!r}" if s == "p" else f"node {i}"
        return TypeError(f"E_REINTERPRET: {what}: leaf {pos} is typed Value<{want}> but names leaf {j} of {src}, a "
                         f"Value<{self.leaf(s, i, j)}>: a silent reinterpretation. A word is read only at the width it was "
                         f"committed with; a runtime word is split into limbs only by a registered LimbSplit Definition, which "
                         f"computes the limbs and their carries")


class Builder:
    """Collects the explicit dataflow of one composite body, in function-body order.

    `finish` refuses a body in which a node's argument or the return names a leaf at another width than its type gives that
    position (`E_REINTERPRET`, PROTOCOL.md §4.4): an s32 word typed as s8 lanes is a silent reinterpretation, a limb split no
    gate computes. A runtime word is split into limbs only by a Definition that computes them (a registered `LimbSplit`, with
    its carries). The check runs at `finish`, when the parameters' types are final (a frontend may declare one before it knows
    its fields) and every node a reference names exists."""

    def __init__(self, params: tuple[tuple[str, Type], ...], ret_type: Type) -> None:
        self.params = params
        self.ret_type = ret_type
        self.nodes: list[Node] = []
        self.offsets: list[int] = [0]

    # -- helpers --------------------------------------------------------------------
    def _emit(self, node: Node) -> Coll:
        k = len(self.nodes)
        self.nodes.append(node)
        self.offsets.append(self.offsets[-1] + node.gates)
        return Coll(node.out, Affine("n", k, 0, node.out.leaves))

    @staticmethod
    def _check_args(fn, args: Sequence[Coll], what: str) -> tuple[tuple[tuple[str, Type], ...], Type]:
        params, ret = fn.signature()
        if len(args) != len(params):
            raise TypeError(f"{what} {fn.id}: expected {len(params)} arguments, got {len(args)}")
        for (pname, pt), a in zip(params, args):
            if not isinstance(a, Coll):
                raise TypeError(f"{what} {fn.id}: argument {pname!r} must be a Coll, got {type(a).__name__}")
            if a.type != pt:
                raise TypeError(f"{what} {fn.id}: argument {pname!r} has type {a.type!r}, expected {pt!r}")
        return params, ret

    # -- constructs -----------------------------------------------------------------
    def call(self, fn: "PrimitiveDefinition | SpecializedDefinition | CompositeDefinition", *args: Coll, name: str | None = None) -> Coll:
        if isinstance(fn, CompositeDefinition):
            fn = bind(fn)
        _, ret = self._check_args(fn, args, "call")
        form = "primitive" if isinstance(fn, PrimitiveDefinition) else "call"
        return self._emit(Node(form, fn, tuple(args), ret, name=name))

    def batch(self, fn, *args: Coll, axes: Sequence[int | None], n: int | None = None, name: str | None = None) -> Coll:
        """`batch(F, args, axes)`: `n` independent calls of `F` (dataflow-independent). `axes[i]` is the
        array axis of `args[i]` mapped over members, or `None` for a shared argument. Members are laid out in
        structural order; the output is `Array<n, ret(F)>`."""
        if isinstance(fn, CompositeDefinition):
            fn = bind(fn)
        axes = tuple(axes)
        if len(axes) != len(args):
            raise TypeError("batch: one axis entry per argument")
        params, ret = fn.signature()
        if len(params) != len(args):
            raise TypeError(f"batch {fn.id}: expected {len(params)} arguments, got {len(args)}")
        lengths = set()
        for (pname, pt), a, ax in zip(params, args, axes):
            if ax is None:
                mt = a.type
            else:
                ln, mt = remove_axis(a.type, ax)
                lengths.add(ln)
            if mt != pt:
                raise TypeError(f"batch {fn.id}: member type of {pname!r} is {mt!r}, expected {pt!r}")
        if len(lengths) > 1:
            raise TypeError(f"batch {fn.id}: mapped axes have different lengths {sorted(lengths)}")
        if lengths:
            if n is not None and n != next(iter(lengths)):
                raise TypeError("batch: explicit n disagrees with mapped axis length")
            n = next(iter(lengths))
        if n is None:
            raise TypeError("batch: n is required when no argument is mapped")
        return self._emit(Node("batch", fn, tuple(args), Array(n, ret), axes=axes, n=n, name=name))

    def scan(self, step, init: Coll, xs: Sequence[Coll] = (), shared: Sequence[Coll] = (), n: int | None = None,
             name: str | None = None) -> Coll:
        """`scan(Step, init, xs, shared)`: `Step(carry, x_1..x_k, s_1..s_m) -> (carry', y)`; iteration `i`
        receives `xs[j][i]` and the previous carry. Output `(carry_n, Array<n, Y>)`."""
        if isinstance(step, CompositeDefinition):
            step = bind(step)
        params, ret = step.signature()
        xs, shared = tuple(xs), tuple(shared)
        if len(params) != 1 + len(xs) + len(shared):
            raise TypeError(f"scan {step.id}: step arity {len(params)} != 1 + {len(xs)} xs + {len(shared)} shared")
        if not (isinstance(ret, TupleT) and len(ret.fields) == 2):
            raise TypeError(f"scan {step.id}: step must return (carry, y), got {ret!r}")
        carry_t, y_t = ret.fields
        if params[0][1] != carry_t:
            raise TypeError(f"scan {step.id}: carry parameter {params[0][1]!r} != carried type {carry_t!r}")
        if init.type != carry_t:
            raise TypeError(f"scan {step.id}: init has type {init.type!r}, expected {carry_t!r}")
        lengths = set()
        for (pname, pt), x in zip(params[1:1 + len(xs)], xs):
            if not isinstance(x.type, Array):
                raise TypeError(f"scan {step.id}: xs argument {pname!r} must be an Array")
            lengths.add(x.type.n)
            if x.type.elem != pt:
                raise TypeError(f"scan {step.id}: xs element type {x.type.elem!r} != {pt!r} for {pname!r}")
        for (pname, pt), s in zip(params[1 + len(xs):], shared):
            if s.type != pt:
                raise TypeError(f"scan {step.id}: shared argument {pname!r} has type {s.type!r}, expected {pt!r}")
        if len(lengths) > 1:
            raise TypeError(f"scan {step.id}: xs lengths differ {sorted(lengths)}")
        if lengths:
            if n is not None and n != next(iter(lengths)):
                raise TypeError("scan: explicit n disagrees with xs length")
            n = next(iter(lengths))
        if n is None:
            raise TypeError("scan: n is required when there are no xs")
        out_t = TupleT(carry_t, Array(n, y_t))
        return self._emit(Node("scan", step, (init,) + xs + shared, out_t, n=n, n_xs=len(xs), name=name))

    def finish(self, out: Coll) -> Body:
        if not isinstance(out, Coll):
            raise TypeError("composite body must return a Coll")
        if out.type != self.ret_type:
            raise TypeError(f"body returns {out.type!r}, declared {self.ret_type!r}")
        W = _Widths(self.params, self.nodes)
        for k, node in enumerate(self.nodes):
            W.below = k
            for i, a in enumerate(node.args):
                hit = W.first(a)
                if hit is not None:
                    raise W.refuse(f"node {k} ({node.form} {node.fn.id}) argument {node.fn.signature()[0][i][0]!r}", a.type, hit)
        W.below = len(self.nodes)
        hit = W.first(out)
        if hit is not None:
            raise W.refuse("the body's return", out.type, hit)
        return Body(self.params, self.ret_type, self.nodes, out.refs, self.offsets)


def primitive(name: str, version: int, params: Sequence[tuple[str, Type]], ret: Value, doc: str = "",
              conformance: str = "not-yet-established", tables: Sequence[tuple[str, str]] = (), ports: Any = None):
    """Decorator: register a one-output primitive with a total evaluator on encodings."""

    def deco(fn: Callable[..., int]) -> PrimitiveDefinition:
        return PrimitiveDefinition(name, version, tuple(params), ret, fn, doc=doc or (fn.__doc__ or ""), conformance=conformance,
                                   tables=tuple(tables), ports=ports)

    return deco


def composite(name: str, version: int, statics: Sequence[str], signature, doc: str = "",
              conformance: str = "not-yet-established", refuse: Callable[[dict], str | None] | None = None,
              ports: Any = None):
    """Decorator: register a composite; the decorated function is the body builder `(B, S, *args) -> Coll`."""

    def deco(fn: Callable[..., Coll]) -> CompositeDefinition:
        return CompositeDefinition(name, version, tuple(statics), signature, fn, doc=doc or (fn.__doc__ or ""),
                            conformance=conformance, refuse=refuse, ports=ports)

    return deco


__all__ = ["Registry", "REGISTRY", "FunctionDefinition", "PrimitiveDefinition", "CompositeDefinition", "SpecializedDefinition", "bind", "Node",
           "Body", "Builder", "TABLE_KINDS", "primitive", "composite", "Explicit", "concat"]
