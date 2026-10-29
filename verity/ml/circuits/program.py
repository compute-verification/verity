"""Programs: a fully-bound composite lowered to a root circuit.

Lowering (IR §5 convention): one input gate per distinct prescribed scalar leaf of each root parameter,
emitted first and in parameter order as `batch(Input<w>, n)` nodes, then the program body inlined with those
input values as its arguments. The input gates are the canonical references of the prescribed inputs.

A parameter is bound per run (`Input<w>_v1`: a request's data and parameters, such as tokens, temperature, top-p and
Philox positions) or at registration (`RegisteredInput<w>_v1`: what the program brings, such as weights, a norm's epsilon,
the attention scale, a softcap and the program's tables). `Program(fn, registered=names)` names the parameters bound at
registration. Both are inputs: the program holds neither's value.

`Program(fn, theta=names)` names the parameters that are the model's weights, whatever their binding time. Their input nodes
are listed in `annotations["theta"]`, outside every digest, and their leaves have origin ``theta`` in the port facts
(`ports.facts`), so the census and the stacking property tell the weights, which the audited store holds, from a request's
inputs. The list binds nothing: a consumer that treats a leaf differently for being a weight checks it against the weights'
commitment.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from . import ports
from .defs import Body, Builder, CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from .layout import Circuit
from .refs import Coll, coll_concat, runs as ref_runs
from .types import Array, Record, Tuple as TupleT, Type, Value

_INPUT_PRIMS: dict[tuple[str, int], PrimitiveDefinition] = {}

#: the ids that denote `Input<w>` (w in canonical decimal): a node is a run-input node exactly when its callee's id matches
INPUT_ID = re.compile(r"^Input([1-9][0-9]*)_v1$")
#: the ids that denote `RegisteredInput<w>`: a node is a registered-input node exactly when its callee's id matches
REGISTERED_ID = re.compile(r"^RegisteredInput([1-9][0-9]*)_v1$")

RUN = "run"
REGISTRATION = "registration"


def is_registered(fn: Any) -> bool:
    """A registered input: `RegisteredInput<w>_v1`, bound at registration."""
    return isinstance(fn, PrimitiveDefinition) and REGISTERED_ID.fullmatch(fn.id) is not None


def is_input(fn: Any) -> bool:
    """An input gate's primitive: `Input<w>_v1` (bound per run) or `RegisteredInput<w>_v1` (bound at registration)."""
    return isinstance(fn, PrimitiveDefinition) and (INPUT_ID.fullmatch(fn.id) is not None or REGISTERED_ID.fullmatch(fn.id) is not None)


def input_id(fid: str) -> tuple[str, int] | None:
    """(binding time, width) of an input id, or None for any other id."""
    for when, rx in ((RUN, INPUT_ID), (REGISTRATION, REGISTERED_ID)):
        m = rx.fullmatch(fid)
        if m is not None:
            return when, int(m[1])
    return None


def input_prim(w: int, binding: str = RUN) -> PrimitiveDefinition:
    """`Input<w>` (``run``) or `RegisteredInput<w>` (``registration``): a zero-operand gate whose value is prescribed (asserted
    equal to the claimed input), never held by the program."""
    key = (binding, w)
    p = _INPUT_PRIMS.get(key)
    if p is None:
        if binding not in (RUN, REGISTRATION):
            raise ValueError(f"an input is bound per {RUN!r} or at {REGISTRATION!r}, not {binding!r}")

        def _no_eval(*_a: int) -> int:
            raise ValueError("Input gates take their value from the transcript, they are not evaluated")

        name, doc = (("Input", f"prescribed input leaf of width {w}, bound per run") if binding == RUN else
                     ("RegisteredInput", f"prescribed input leaf of width {w}, bound at registration"))
        p = PrimitiveDefinition(f"{name}{w}", 1, (), Value(w), _no_eval, doc=doc, conformance="exact-model-tested")
        p.family = "Input"  # type: ignore[attr-defined]
        p.binding = binding  # type: ignore[attr-defined]
        _INPUT_PRIMS[key] = p
    return p


def _lower_input(B: Builder, t: Type, name: str, binding: str = RUN) -> Coll:
    w = t.homogeneous_width()
    if w is not None:
        if t.leaves == 0:
            from .refs import Explicit
            return Coll(t, Explicit(()))
        c = B.batch(input_prim(w, binding), axes=(), n=t.leaves, name=name)
        return c.reshape(t)
    if isinstance(t, Record):
        parts = [_lower_input(B, ft, f"{name}.{k}", binding) for k, ft in t.fields]
        return coll_concat(t, parts)
    if isinstance(t, TupleT):
        parts = [_lower_input(B, ft, f"{name}.{i}", binding) for i, ft in enumerate(t.fields)]
        return coll_concat(t, parts)
    if isinstance(t, Array):
        parts = [_lower_input(B, t.elem, f"{name}[{i}]", binding) for i in range(t.n)]
        return coll_concat(t, parts)
    raise TypeError(t)


class Program:
    """`Program(fn)` where `fn` is a fully-bound `SpecializedDefinition` (or a static-free `CompositeDefinition`).
    `registered`: the names of `fn`'s parameters bound at registration; every other parameter is bound per run. `theta`: the
    names of `fn`'s parameters that are the model's weights. `annotations["theta"]` is the Program's own (their input nodes):
    one passed in, which may name another layout's nodes, is dropped.
    `strict`: a guard (`verity.ml.circuits.operations.program_data`, root -> the data it holds); the Program is refused when it holds
    any. Off by default until programs move their data to inputs."""

    def __init__(self, fn: SpecializedDefinition | CompositeDefinition, annotations: dict[str, Any] | None = None,
                 registered: tuple[str, ...] | list[str] | frozenset[str] = (),
                 strict: Callable[[Any], dict] | None = None,
                 theta: tuple[str, ...] | list[str] | frozenset[str] = ()) -> None:
        if isinstance(fn, CompositeDefinition):
            fn = bind(fn)
        if not fn.complete:
            raise TypeError(f"program function {fn.id} is not fully bound")
        names = [n for n, _t in fn.signature()[0]]
        for what, given in (("register", registered), ("mark theta", theta)):
            unknown = sorted(set(given) - set(names))
            if unknown:
                raise ValueError(f"program function {fn.id} has no parameter {unknown} to {what} (it has {names})")
        self.fn = fn
        self.annotations: dict[str, Any] = {k: v for k, v in (annotations or {}).items() if k != "theta"}
        self.bindings: tuple[str, ...] = tuple(REGISTRATION if n in set(registered) else RUN for n in names)
        self.theta: tuple[str, ...] = tuple(n for n in names if n in set(theta))
        self.root = _lower_root(fn, self.bindings, frozenset(theta))
        if self.root.theta:
            self.annotations["theta"] = sorted(self.root.theta)
        _refuse_inner_inputs(self.root)
        ports.check(self.root)
        self._circuit: Circuit | None = None
        if strict is not None and (held := strict(self.root)):
            raise ValueError(f"program function {fn.id} holds data, which a strict Program refuses: "
                             + "; ".join(f"{d}: {len(v)} ({', '.join(sorted({x for _g, x in v})[:3])})" for d, v in held.items()))

    @property
    def circuit(self) -> Circuit:
        if self._circuit is None:
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
    from .refs import Affine, Concat, Explicit, Strided, concat
    if isinstance(r, Affine):
        assert r.space == "n"
        return Affine("g", 0, body.offsets[r.idx] + r.base, r.count)
    if isinstance(r, Strided):
        assert r.space == "n"
        return Strided("g", 0, body.offsets[r.idx] + r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
    if isinstance(r, Concat):
        return Concat([_gate_refs(p, body) for p in r.parts])
    parts = []
    for space, k, leaf in r:
        assert space == "n"
        parts.append(Affine("g", 0, body.offsets[k] + leaf, 1))
    return concat(parts) if parts else Explicit(())


def _gate_coll(c: Coll, body: Body) -> Coll:
    return Coll(c.type, _gate_refs(c.refs, body))
