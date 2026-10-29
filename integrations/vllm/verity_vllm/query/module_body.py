"""`Q_module_body_v1`: units = module BODIES.

A unit is the maximal set of Calls whose recorded owning module path (the correspondence's `module`) is identical.  A non-leaf
module's own Calls -- the ones its forward issues itself, not through a child -- form their own body (a decoder's own `embed *
scale` under `model`, beside the bodies of `model.layers.0.input_layernorm`, ...); Calls with no module form the root body
(unit id "").  Nothing about a module's class, name or family enters: two modules with identical paths are one body because the
correspondence says the same module issued their Calls.

Under this query the boundary Values are exactly the Values that leave a module body for another module's body (or for the
Program's output) and the prescribed inputs a module body reads.  v1's "promotion" of interior rows whose output another module
reads is not a pass here: such a Value crosses the body boundary and so is a boundary Value by definition.

Literals.  A Program has no literal operand (a reference is `("p", i, j)` -- a parameter leaf -- or `("n", k, j)` -- a body node's
output leaf): a literal is a Call that reads no Value (`Const<w>[bits]`, a zero-arity gate), whose output is a function of its binding
record alone and so is written in the Program descriptor.  Both ProgramView readers read the Program function's body, whose prescribed
inputs are its parameters (never `Input<w>` gates, which only the lowered root circuit has), so there a Call with no reads is always
such a literal.  A literal is part of the Program, not a runtime Value to acquire: the Values of `literal_calls(P)` are never boundary
Values, wherever the correspondence places the Call and whichever body reads it (`module_body_boundary`).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from verity_vllm.query.units import Boundary, Partition, ValueRef, boundary, partition_by

from verity_vllm.correspondence.reader_for_query import Correspondence
from .program_view import ProgramView

__all__ = ["Q_MODULE_BODY_ID", "Q_MODULE_BODY_TEXT", "Q_module_body_v1", "literal_calls", "without_literals", "boundary_without_literals",
           "module_body_boundary"]

Q_MODULE_BODY_ID = "Q_module_body_v1"
Q_MODULE_BODY_TEXT = ("Q_module_body_v1: unit = module body = the maximal set of Calls whose recorded owning module path (correspondence: "
                      "nn_module_stack-derived) is identical; a non-leaf module's own Calls are their own body; Calls with no module are the "
                      "root body; boundary values = U_V (in(V) u out(V)) over these units (verity_vllm.query.units.boundary_values) minus "
                      "the Program's literals (the Values of Calls that read no Value: written in the descriptor, never acquired)")


def Q_module_body_v1(P: ProgramView, correspondence: Correspondence) -> Partition:
    """The partition of `P`'s Calls by owning module path (unit id = the Program module path; "" for the root body)."""
    if correspondence.P is not P:
        raise ValueError("Q_module_body_v1: the correspondence was read off another Program")
    return partition_by(lambda c: correspondence.module_of(c.id))


def literal_calls(P: ProgramView) -> frozenset[int]:
    """The Calls of `P` that read no Value: the Program's literals (module docstring).  O(1) per Call on a columnar body."""
    cols = P.columns
    if cols is not None:
        off = cols.read_off
        return frozenset(k for k in range(cols.n) if off[k + 1] == off[k])
    return frozenset(c.id for c in P.calls if not len(c.reads))


def without_literals(B: Boundary, literals: frozenset[int]) -> Boundary:
    """`B` without the Values a literal Call produces (the units and the Call -> unit map unchanged)."""
    drop = frozenset(v for v in B.values if v.call in literals)
    if not drop:
        return B
    return Boundary(B.values - drop, {v: u for v, u in B.produced_in.items() if v not in drop},
                    {v: r for v, r in B.read_by.items() if v not in drop}, B.units, B.owner)


#: the reads `boundary_without_literals` takes per numpy step (bounds its scratch arrays, about 30 bytes a read)
_READS_PER_STEP = 1 << 24


class _ReadBy(Mapping):
    """`Boundary.read_by` of `without_literals(boundary(P, Q), literals)`, computed by `boundary` when first read.  No manifest reads
    it, and under `Q_word` every Call is a unit, so a key that every later token's attention reads has a reader per token."""

    def __init__(self, P: ProgramView, Q: Partition, literals: frozenset[int]) -> None:
        self._args, self._d = (P, Q, literals), None

    def _get(self) -> Mapping:
        if self._d is None:
            P, Q, literals = self._args
            self._d = without_literals(boundary(P, Q), literals).read_by
        return self._d

    def __getitem__(self, v):
        return self._get()[v]

    def __iter__(self):
        return iter(self._get())

    def __len__(self) -> int:
        return len(self._get())


def boundary_without_literals(P: ProgramView, Q: Partition, literals: frozenset[int]) -> Boundary:
    """`without_literals(boundary(P, Q), literals)`.  On a columnar body its Values and `produced_in` (in `boundary`'s order: a
    Value's first read from another unit, in body order, then the Program's outputs) come from numpy passes over the read columns,
    and `read_by` is `boundary`'s, computed when first read; a request Program's attention reads every earlier token's keys and
    values, so the reads grow with the context squared (B1 4096/512: 3.4e8).  Any other body, or units that are not a partition,
    go to `boundary` itself."""
    cols = P.columns
    units = tuple(Q(P))
    owner = {cid: u.id for u in units for cid in u.calls}
    n = 0 if cols is None else cols.n
    if (cols is None or len(owner) != n or len(owner) != sum(len(u.calls) for u in units)
            or (n and (min(owner) != 0 or max(owner) != n - 1))):
        return without_literals(boundary(P, Q), literals)
    index: dict[str, int] = {}
    unit = np.fromiter((index.setdefault(owner[k], len(index)) for k in range(n)), np.int32, n)
    literal = np.zeros(n, bool)
    literal[np.fromiter(literals, np.int64, len(literals))] = True
    off = np.frombuffer(cols.read_off, cols.read_off.typecode)
    call = np.frombuffer(cols.read_call, cols.read_call.typecode)
    port = np.frombuffer(cols.read_port, cols.read_port.typecode)
    keys, firsts = [], []
    k0 = 0
    while k0 < n:
        k1 = min(n, max(k0 + 1, int(np.searchsorted(off, off[k0] + _READS_PER_STEP, "right")) - 1))
        lo, hi = int(off[k0]), int(off[k1])
        c = call[lo:hi]
        made = c >= 0                                    # a Call's Value; the others are prescribed inputs, read from outside every unit
        at = np.where(made, c, 0)
        reader = np.repeat(unit[k0:k1], np.diff(off[k0:k1 + 1]))
        crossing = np.flatnonzero(np.where(made, unit[at] != reader, True) & ~(made & literal[at]))
        key, first = np.unique((c[crossing].astype(np.int64) << 16) | port[lo:hi][crossing], return_index=True)
        keys.append(key)
        firsts.append(crossing[first] + lo)
        k0 = k1
    produced_in: dict[ValueRef, str | None] = {}
    if keys:
        key, at = np.concatenate(keys), np.concatenate(firsts)
        key = key[np.argsort(at)]
        _, first = np.unique(key, return_index=True)
        key = key[np.sort(first)]
        names, get = cols.port_names, owner.get
        for cid, p in zip((key >> 16).tolist(), (key & 0xFFFF).tolist()):
            produced_in[ValueRef(cid, names[p])] = get(cid)
    for v in P.outputs:
        if v not in produced_in and v.call not in literals:
            produced_in[v] = owner.get(v.call)
    return Boundary(frozenset(produced_in), produced_in, _ReadBy(P, Q, literals), units, owner)


def module_body_boundary(P: ProgramView, correspondence: Correspondence, literals: frozenset[int] | None = None) -> Boundary:
    """The boundary of `Q_module_body_v1(P, correspondence)`: the crossing Values of the module-body partition minus the literals."""
    return boundary_without_literals(P, Q_module_body_v1(P, correspondence), literal_calls(P) if literals is None else literals)
