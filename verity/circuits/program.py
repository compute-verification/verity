"""Programs: a fully-bound composite lowered to a root circuit.

Lowering (IR §5 convention): one input gate per distinct prescribed scalar leaf of each root parameter,
emitted first and in parameter order as `batch(Input<w>, n)` nodes, then the program body inlined with those
input values as its arguments. The input gates are the canonical references of the prescribed inputs.

A parameter is bound per run (`Input<w>_v1`: a request's data and parameters, such as tokens, temperature, top-p and
Philox positions), at registration (`RegisteredInput<w>_v1`: what the program brings, such as weights, a norm's epsilon,
the attention scale, a softcap and the program's tables), or supplied (`SuppliedInput<w>_v1`: a value the verifier computes
by a public function, PROTOCOL.md §5.4). `Program(fn, registered=names, supplied=marks)` names the parameters bound at
registration and the supplied ones, each with its mark. All three are inputs: the program holds none's value.

`Program(fn, theta=names)` names the parameters that are the model's weights, whatever their binding time. Their input nodes
are listed in `annotations["theta"]`, outside every digest, and their leaves have origin ``theta`` in the port facts
(`ports.facts`), so the census and the stacking property tell the weights, which the audited store holds, from a request's
inputs. The list binds nothing: a consumer that treats a leaf differently for being a weight checks it against the weights'
commitment.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Callable

from . import ports
from .supplied import SUPPLIED, SUPPLIED_ID
from .defs import Body, Builder, CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from .layout import Circuit
from .refs import Coll, coll_concat, runs as ref_runs
from .types import Array, RecordType, Tuple as TupleT, Type, Value

_INPUT_PRIMS: dict[tuple[str, int], PrimitiveDefinition] = {}

#: the ids that denote `Input<w>` (w in canonical decimal): a node is a run-input node exactly when its callee's id matches
INPUT_ID = re.compile(r"^Input([1-9][0-9]*)_v1$")
#: the ids that denote `RegisteredInput<w>`: a node is a registered-input node exactly when its callee's id matches
REGISTERED_ID = re.compile(r"^RegisteredInput([1-9][0-9]*)_v1$")

RUN = "run"
REGISTRATION = "registration"
BINDINGS = (RUN, REGISTRATION, SUPPLIED)

#: the catalog's lookup of a word primitive's one Boolean Definition (`register_boolean_bodies`), which `Program` calls at
#: build; a decoder never calls it, as the descriptor holds every body
_BODY_LOOKUP: list[Callable[[PrimitiveDefinition], Any]] = []


class NoBooleanBody(ValueError):
    """``E_NO_BODY``: a Program reaches word primitives with no one Boolean Definition; ``missing`` names each with why."""

    def __init__(self, missing: dict[str, str]) -> None:
        self.missing = dict(missing)
        super().__init__("E_NO_BODY: no one Boolean Definition for " + "; ".join(f"{k}: {v}" for k, v in sorted(self.missing.items())))


def register_boolean_bodies(lookup: Callable[[PrimitiveDefinition], Any]) -> None:
    """Make ``lookup`` (a word primitive -> its one Boolean Definition, `LookupError` where it has none) the one `Program`
    builds its Boolean map with."""
    _BODY_LOOKUP[:] = [lookup]


def is_word_primitive(fn: Any) -> bool:
    """A primitive that is no Boolean gate and no input: one the Boolean map names a body for."""
    from .boolean import is_boolean
    return isinstance(fn, PrimitiveDefinition) and not is_input(fn) and not is_boolean(fn)


def boolean_body(p: PrimitiveDefinition) -> Any:
    """``p``'s one Boolean Definition by the registered lookup, bound, of ``p``'s bit signature (`boolean.bit_type`);
    `LookupError` saying why where it has none."""
    from .boolean import bit_type, is_boolean
    if not _BODY_LOOKUP:
        raise LookupError("no Boolean lookup is registered (`register_boolean_bodies`; the catalog's registers as it loads)")
    body = _BODY_LOOKUP[0](p)
    body = bind(body) if isinstance(body, CompositeDefinition) else body
    params, ret = p.signature()
    bparams, bret = body.signature()
    if not is_boolean(body) or ([t for _, t in bparams], bret) != ([bit_type(t) for _, t in params], bit_type(ret)):
        raise LookupError(f"{body.id} is not a Boolean Definition of its bit signature")
    return body


def boolean_map(program: Any) -> dict[str, Any]:
    """Each word primitive among ``program``'s definitions (`codec.definitions`), by descriptor id, to its one Boolean
    Definition by the registered lookup (`boolean_body`), sorted by id. `NoBooleanBody` names each that has none, or whose
    body is not Boolean."""
    from .codec import definitions
    out: dict[str, Any] = {}
    missing: dict[str, str] = {}
    for fid, p in definitions(program).items():
        if not is_word_primitive(p):
            continue
        try:
            out[fid] = boolean_body(p)
        except LookupError as e:
            missing[fid] = str(e)
    if missing:
        raise NoBooleanBody(missing)
    return dict(sorted(out.items()))


def is_registered(fn: Any) -> bool:
    """A registered input: `RegisteredInput<w>_v1`, bound at registration."""
    return isinstance(fn, PrimitiveDefinition) and REGISTERED_ID.fullmatch(fn.id) is not None


def is_input(fn: Any) -> bool:
    """An input gate's primitive: `Input<w>_v1` (bound per run), `RegisteredInput<w>_v1` (bound at registration) or
    `SuppliedInput<w>_v1` (supplied)."""
    return isinstance(fn, PrimitiveDefinition) and input_id(fn.id) is not None


def input_id(fid: str) -> tuple[str, int] | None:
    """(binding time, width) of an input id, or None for any other id."""
    for when, rx in ((RUN, INPUT_ID), (REGISTRATION, REGISTERED_ID), (SUPPLIED, SUPPLIED_ID)):
        m = rx.fullmatch(fid)
        if m is not None:
            return when, int(m[1])
    return None


def input_prim(w: int, binding: str = RUN) -> PrimitiveDefinition:
    """`Input<w>` (``run``), `RegisteredInput<w>` (``registration``) or `SuppliedInput<w>` (``supplied``): a zero-operand
    gate whose value is prescribed (asserted equal to the claimed input), never held by the program."""
    key = (binding, w)
    p = _INPUT_PRIMS.get(key)
    if p is None:
        if binding not in BINDINGS:
            raise ValueError(f"an input is bound per {RUN!r}, at {REGISTRATION!r} or {SUPPLIED!r}, not {binding!r}")

        def _no_eval(*_a: int) -> int:
            raise ValueError("Input gates take their value from the trace, they are not evaluated")

        name, doc = {RUN: ("Input", f"prescribed input leaf of width {w}, bound per run"),
                     REGISTRATION: ("RegisteredInput", f"prescribed input leaf of width {w}, bound at registration"),
                     SUPPLIED: ("SuppliedInput", f"prescribed input leaf of width {w}, supplied")}[binding]
        p = PrimitiveDefinition(f"{name}{w}", 1, (), Value(w), _no_eval, doc=doc, conformance="exact-model-tested")
        p.family = "Input"  # type: ignore[attr-defined]
        p.binding = binding  # type: ignore[attr-defined]
        _INPUT_PRIMS[key] = p
    return p


def input_nodes(t: Type) -> int:
    """The input nodes `_lower_input` makes for a parameter of type ``t``."""
    if t.homogeneous_width() is not None:
        return 1 if t.leaves else 0
    if isinstance(t, (RecordType, TupleT)):
        return sum(input_nodes(ft) for _k, ft in t.fields)
    if isinstance(t, Array):
        return t.n * input_nodes(t.elem)
    raise TypeError(t)


def _lower_input(B: Builder, t: Type, name: str, binding: str = RUN) -> Coll:
    w = t.homogeneous_width()
    if w is not None:
        if t.leaves == 0:
            from .refs import Explicit
            return Coll(t, Explicit(()))
        c = B.batch(input_prim(w, binding), axes=(), n=t.leaves, name=name)
        return c.reshape(t)
    if isinstance(t, RecordType):
        parts = [_lower_input(B, ft, f"{name}.{k}", binding) for k, ft in t.fields]
        return coll_concat(t, parts)
    if isinstance(t, TupleT):
        parts = [_lower_input(B, ft, f"{name}.{i}", binding) for i, ft in enumerate(t.fields)]
        return coll_concat(t, parts)
    if isinstance(t, Array):
        parts = [_lower_input(B, t.elem, f"{name}[{i}]", binding) for i in range(t.n)]
        return coll_concat(t, parts)
    raise TypeError(t)



def recut_gates(recut: Any) -> str:
    """The gate layout's refusal of a root reading the input nodes ``recut`` recut (`Circuits.Program.recutGates`)."""
    return (f"recut-gates: input nodes {sorted(recut)} are read in leaves of part of their width, which the gate layout "
            "doesn't lay out")

class Program:
    """`Program(fn)` where `fn` is a fully-bound `SpecializedDefinition` (or a static-free `CompositeDefinition`).
    `registered`: the names of `fn`'s parameters bound at registration; `supplied`: the supplied ones, each name to its mark
    (`supplied`, PROTOCOL.md §5.4; None where only the kind is known, as in a block `restrict` cuts); every other parameter
    is bound per run. `theta`: the
    names of `fn`'s parameters that are the model's weights. `annotations["theta"]` is the Program's own (their input nodes):
    one passed in, which may name another layout's nodes, is dropped. `boolean`: each word primitive's one Boolean Definition
    (`boolean_map`), computed once here; a Program with a word primitive that has none is refused (`NoBooleanBody`).
    `strict`: a guard (`verity.circuits.operations.program_data`, root -> the data it holds); the Program is refused when it holds
    any. Off by default until programs move their data to inputs."""

    def __init__(self, fn: SpecializedDefinition | CompositeDefinition, annotations: dict[str, Any] | None = None,
                 registered: tuple[str, ...] | list[str] | frozenset[str] = (),
                 strict: Callable[[Any], dict] | None = None,
                 theta: tuple[str, ...] | list[str] | frozenset[str] = (),
                 supplied: Mapping[str, Mapping[str, Any] | None] | tuple[str, ...] | list[str] = ()) -> None:
        if isinstance(fn, CompositeDefinition):
            fn = bind(fn)
        if not fn.complete:
            raise TypeError(f"program function {fn.id} is not fully bound")
        names = [n for n, _t in fn.signature()[0]]
        supplied = dict(supplied) if isinstance(supplied, Mapping) else dict.fromkeys(supplied)
        for what, given in (("register", registered), ("mark theta", theta), ("supply", supplied)):
            unknown = sorted(set(given) - set(names))
            if unknown:
                raise ValueError(f"program function {fn.id} has no parameter {unknown} to {what} (it has {names})")
        self.fn = fn
        self.annotations: dict[str, Any] = {k: v for k, v in (annotations or {}).items() if k != "theta"}
        if both := sorted(set(registered) & set(supplied)):
            raise ValueError(f"program function {fn.id}: {both} both registered and supplied")
        self.bindings: tuple[str, ...] = tuple(REGISTRATION if n in set(registered) else SUPPLIED if n in supplied else RUN
                                               for n in names)
        self.supplied: dict[str, Mapping[str, Any] | None] = {n: supplied[n] for n in names if n in supplied}
        self.theta: tuple[str, ...] = tuple(n for n in names if n in set(theta))
        self.root = _lower_root(fn, self.bindings, frozenset(theta))
        if self.root.theta:
            self.annotations["theta"] = sorted(self.root.theta)
        _refuse_inner_inputs(self.root)
        ports.check(self.root)
        self.boolean: dict[str, Any] = {}
        self.boolean = boolean_map(self)
        self._circuit: Circuit | None = None
        if strict is not None and (held := strict(self.root)):
            raise ValueError(f"program function {fn.id} holds data, which a strict Program refuses: "
                             + "; ".join(f"{d}: {len(v)} ({', '.join(sorted({x for _g, x in v})[:3])})" for d, v in held.items()))

    @property
    def circuit(self) -> Circuit:
        """The gate layout. Refused (ValueError) while the root reads an input node recut (`refs.recut`): a gate is one
        leaf of its width, and the layout has no operand of part of a gate."""
        if self._circuit is None:
            if self.root.body.recut:
                raise ValueError(recut_gates(self.root.body.recut))
            self._circuit = Circuit(self.root)
        return self._circuit

    @property
    def gates(self) -> int:
        return self.root.gates

    @property
    def input_gates(self) -> int:
        return sum(n.gates for n in self.root.body.nodes if is_input(n.fn))

    def input_coll(self, pi: int) -> Coll:
        """Root-scoped gate indices of prescribed parameter `pi`, as a typed collection over gate indices."""
        return self.root.input_colls[pi]  # type: ignore[attr-defined]

    def inputs_entry(self) -> list[list[Any]] | None:
        """The descriptor's top-level `inputs` (PROTOCOL.md §5.4): every parameter in order, `[name, n]` with ``n`` its
        input nodes, a supplied one's `[name, n, mark]`; none where nothing is supplied or a supplied parameter has no mark
        (a block `restrict` cuts), so such a Program's descriptor is as before."""
        if not self.supplied or any(m is None for m in self.supplied.values()):
            return None
        return [[n, input_nodes(t), *([dict(self.supplied[n])] if n in self.supplied else [])] for n, t in self.fn.signature()[0]]


def _lower_root(fn: SpecializedDefinition, bindings: tuple[str, ...] | None = None,
                theta: frozenset[str] = frozenset()) -> SpecializedDefinition:
    params, ret = fn.signature()
    B = Builder((), ret)
    bindings = bindings or (RUN,) * len(params)
    input_colls = [_lower_input(B, t, name, when) for (name, t), when in zip(params, bindings)]
    theta_nodes = frozenset(k for (name, _t), c in zip(params, input_colls) if name in theta for _s, k, _a, _b in ref_runs(c.refs))
    out = fn.defn.body(B, fn.S(), *input_colls)
    body = B.finish(out)
    root_def = CompositeDefinition(f"Root[{fn.id}]", 1, (), lambda S: ((), ret), lambda *_: None, register=False)
    root = SpecializedDefinition(root_def, {}, ())
    root.set_body(body)
    root.theta = theta_nodes  # type: ignore[attr-defined]
    root.set_port_types(ports.root_ports(fn, input_colls, [b == REGISTRATION for b in bindings], [n.out for n in body.nodes]))
    root.program_fn = fn  # type: ignore[attr-defined]
    # Gate indices of the Input gates per parameter, as collections over refs ("g", 0, gate_index). Kept
    # affine (an Input batch is one contiguous interval) so weight tensors stay O(1).
    root.input_colls = [_gate_coll(c, body) for c in input_colls]  # type: ignore[attr-defined]
    return root


def _refuse_inner_inputs(root: SpecializedDefinition) -> None:
    """An input node appears only in the root's body (PROTOCOL.md §5.2): one inside a Definition the root instantiates would be
    a gate no parameter binds, so it is refused, as `codec.decode_program` refuses it."""
    seen: set[str] = set()
    todo = [n.fn for n in root.body.nodes]
    while todo:
        fn = todo.pop()
        if not isinstance(fn, SpecializedDefinition) or fn.id in seen:
            continue
        seen.add(fn.id)
        for k, node in enumerate(fn.body.nodes):
            if is_input(node.fn):
                raise ValueError(f"definition {fn.id!r} node {k}: an input node ({node.fn.id}) appears only in the root's body")
            todo.append(node.fn)


def _gate_refs(r, body: Body):
    from .refs import Affine, Concat, Explicit, Grid, Strided, concat
    if isinstance(r, Affine):
        assert r.space == "n"
        return Affine("g", 0, body.offsets[r.idx] + r.base, r.count)
    if isinstance(r, Strided):
        assert r.space == "n"
        return Strided("g", 0, body.offsets[r.idx] + r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
    if isinstance(r, Concat):
        return Concat([_gate_refs(p, body) for p in r.parts])
    if isinstance(r, Grid):
        assert all(s == "n" for s, *_x in r.lanes)
        return Grid(r.run, tuple(("g", 0, body.offsets[i] + b, e, lv) for _s, i, b, e, lv in r.lanes))
    parts = []
    for space, k, leaf in r:
        assert space == "n"
        parts.append(Affine("g", 0, body.offsets[k] + leaf, 1))
    return concat(parts) if parts else Explicit(())


def _gate_coll(c: Coll, body: Body) -> Coll:
    return Coll(c.type, _gate_refs(c.refs, body))
