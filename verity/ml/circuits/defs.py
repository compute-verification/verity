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
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .refs import Affine, Coll, Explicit, Refs, concat
from .types import Array, Tuple as TupleT, Type, Value, remove_axis

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
    (`constants.program_data`)."""

    name: str
    version: int
    params: tuple[tuple[str, Type], ...]
    ret: Value
    evaluate: Callable[..., int] = field(repr=False)
    doc: str = ""
    conformance: str = "not-yet-established"  # exact-model-tested | model-untested | not-yet-established
    register: bool = True
    tables: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ret, Value):
            raise TypeError(f"a primitive returns exactly one Value<w>; {self.id} declares {self.ret!r}")
        self.params = tuple((n, t) for n, t in self.params)
        self.tables = tuple((n, k) for n, k in self.tables)
        bad = [k for _n, k in self.tables if k not in TABLE_KINDS]
        if bad:
            raise ValueError(f"{self.id}: a table is a {' or '.join(TABLE_KINDS)} table, not {bad}")
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
    statics (attribute access). Statics with default `None` must be bound before use."""

    name: str
    version: int
    statics: tuple[str, ...]
    signature: Callable[[Any], tuple[Sequence[tuple[str, Type]], Type]] = field(repr=False)
    body: Callable[..., Coll] = field(repr=False)
    doc: str = ""
    conformance: str = "not-yet-established"
    register: bool = True

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
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    if isinstance(v, float):
        return ("float", v.hex())
    if isinstance(v, dict):
        return ("dict",) + tuple((k, _static_key(v[k])) for k in sorted(v))
    raise TypeError(f"unsupported static value {v!r}")


class IdentityConflict(ValueError):
    """`E_IDENTITY`: two different definitions claim the same fully qualified id within one program (frontend
    rulings §8.1/§8.3). Never resolved first-wins — neither by the binding cache nor by the codec."""


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
        return a.params == b.params and a.ret == b.ret
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
    """Explicit binding record. Partial binds compose; all statics must be bound before the body is built.

    Specializations are shared through a per-definition cache keyed by the binding record. Function-valued
    statics are keyed by id, so a cache hit whose function references are *different* definitions under the
    same id is a conflict (ruling §8.1), not a silent reuse of the first body."""
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


class SpecializedDefinition(FunctionDefinition):
    """A composite definition with (some) statics bound. Fully-bound specializations have a cached body."""

    def __init__(self, defn: CompositeDefinition, bindings: dict, key: tuple) -> None:
        self.defn = defn
        self.bindings = dict(bindings)
        self.key = key
        self._body: Body | None = None

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
        """Function identity = definition + static bindings: `Name_vN{K=..,..}` (bare `Name_vN` if static-free)."""
        if not self.defn.statics:
            return self.base_id
        parts = [f"{k}={_fmt_static(self.bindings[k])}" for k in self.defn.statics if k in self.bindings]
        return self.base_id + "{" + ",".join(parts) + "}"

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


class Builder:
    """Collects the explicit dataflow of one composite body, in function-body order."""

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
        return Body(self.params, self.ret_type, self.nodes, out.refs, self.offsets)


def primitive(name: str, version: int, params: Sequence[tuple[str, Type]], ret: Value, doc: str = "",
              conformance: str = "not-yet-established", tables: Sequence[tuple[str, str]] = ()):
    """Decorator: register a one-output primitive with a total evaluator on encodings."""

    def deco(fn: Callable[..., int]) -> PrimitiveDefinition:
        return PrimitiveDefinition(name, version, tuple(params), ret, fn, doc=doc or (fn.__doc__ or ""), conformance=conformance,
                                   tables=tuple(tables))

    return deco


def composite(name: str, version: int, statics: Sequence[str], signature, doc: str = "",
              conformance: str = "not-yet-established"):
    """Decorator: register a composite; the decorated function is the body builder `(B, S, *args) -> Coll`."""

    def deco(fn: Callable[..., Coll]) -> CompositeDefinition:
        return CompositeDefinition(name, version, tuple(statics), signature, fn, doc=doc or (fn.__doc__ or ""),
                            conformance=conformance)

    return deco


__all__ = ["Registry", "REGISTRY", "FunctionDefinition", "PrimitiveDefinition", "CompositeDefinition", "SpecializedDefinition", "bind", "Node",
           "Body", "Builder", "TABLE_KINDS", "primitive", "composite", "Explicit", "concat"]
