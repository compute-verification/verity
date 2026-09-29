"""Programs: a fully-bound composite lowered to a root circuit.

Lowering (IR §5 convention): one `Input` gate per distinct prescribed scalar leaf of each root parameter,
emitted first and in parameter order as `batch(Input<w>, n)` nodes, then the program body inlined with those
`Input` values as its arguments. The `Input` gates are the canonical references of the prescribed inputs.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from .defs import Body, Builder, CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from .layout import Circuit
from .refs import Coll, coll_concat
from .types import Array, Record, Tuple as TupleT, Type, Value

_INPUT_PRIMS: dict[int, PrimitiveDefinition] = {}

#: the ids that denote `Input<w>` (w in canonical decimal): a node is an Input node exactly when its callee's id matches
INPUT_ID = re.compile(r"^Input([1-9][0-9]*)_v1$")


def is_input(fn: Any) -> bool:
    return isinstance(fn, PrimitiveDefinition) and INPUT_ID.match(fn.id) is not None


def input_prim(w: int) -> PrimitiveDefinition:
    """`Input<w>`: a zero-operand gate whose value is prescribed (asserted equal to the claimed input)."""
    p = _INPUT_PRIMS.get(w)
    if p is None:
        def _no_eval(*_a: int) -> int:
            raise ValueError("Input gates take their value from the transcript, they are not evaluated")

        p = PrimitiveDefinition(f"Input{w}", 1, (), Value(w), _no_eval,
                         doc=f"prescribed input leaf of width {w}", conformance="exact-model-tested")
        p.family = "Input"  # type: ignore[attr-defined]
        _INPUT_PRIMS[w] = p
    return p


def _lower_input(B: Builder, t: Type, name: str) -> Coll:
    w = t.homogeneous_width()
    if w is not None:
        if t.leaves == 0:
            from .refs import Explicit
            return Coll(t, Explicit(()))
        c = B.batch(input_prim(w), axes=(), n=t.leaves, name=name)
        return c.reshape(t)
    if isinstance(t, Record):
        parts = [_lower_input(B, ft, f"{name}.{k}") for k, ft in t.fields]
        return coll_concat(t, parts)
    if isinstance(t, TupleT):
        parts = [_lower_input(B, ft, f"{name}.{i}") for i, ft in enumerate(t.fields)]
        return coll_concat(t, parts)
    if isinstance(t, Array):
        parts = [_lower_input(B, t.elem, f"{name}[{i}]") for i in range(t.n)]
        return coll_concat(t, parts)
    raise TypeError(t)


class Program:
    """`Program(fn)` where `fn` is a fully-bound `SpecializedDefinition` (or a static-free `CompositeDefinition`)."""

    def __init__(self, fn: SpecializedDefinition | CompositeDefinition, annotations: dict[str, Any] | None = None) -> None:
        if isinstance(fn, CompositeDefinition):
            fn = bind(fn)
        if not fn.complete:
            raise TypeError(f"program function {fn.id} is not fully bound")
        self.fn = fn
        self.annotations: dict[str, Any] = dict(annotations or {})
        self.root = _lower_root(fn)
        self._circuit: Circuit | None = None

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


def _lower_root(fn: SpecializedDefinition) -> SpecializedDefinition:
    params, ret = fn.signature()
    B = Builder((), ret)
    input_colls = [_lower_input(B, t, name) for name, t in params]
    out = fn.defn.body(B, fn.S(), *input_colls)
    body = B.finish(out)
    root_def = CompositeDefinition(f"Root[{fn.id}]", 1, (), lambda S: ((), ret), lambda *_: None, register=False)
    root = SpecializedDefinition(root_def, {}, ())
    root.set_body(body)
    root.program_fn = fn  # type: ignore[attr-defined]
    # Gate indices of the Input gates per parameter, as collections over refs ("g", 0, gate_index). Kept
    # affine (an Input batch is one contiguous interval) so weight tensors stay O(1).
    root.input_colls = [_gate_coll(c, body) for c in input_colls]  # type: ignore[attr-defined]
    return root


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
