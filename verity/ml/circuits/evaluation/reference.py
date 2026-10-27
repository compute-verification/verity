"""The plain-integer reference: what every kernel is checked against.

`evaluate_region` / `evaluate_call` are the gate-by-gate oracle of `verity.ml.circuits.evaluate`, re-exported unchanged (the code
moves here once no importer of the old path is left).  `evaluate(target, *args)` runs a target on one input of plain
ints: a primitive through its evaluator, a composite specialization through the oracle on a one-call Program.

A target is a Definition (`PrimitiveDefinition`, a static-free `CompositeDefinition`, a complete
`SpecializedDefinition`) or a `ScalarReference`: a silicon function that no core Definition wraps yet (the FP8 and NVFP4
steps), with a signature in the Program's types so that batching, sampling and checking treat both alike.

A table held in a registered input (`tanh_rn`: 2^27 entries of two words) can be passed to `evaluate` as a `RegisteredTable`
instead of its flat word list: the same words, read one at a time when a gate asks for them.  The result is the flattened
form's, bit for bit; `evaluate_batch` and the bit-sliced evaluators refuse it by name (`refuse_registered_tables`).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ml.circuits.evaluate import ExecutionTranscript, MissingValue, evaluate_call, evaluate_region
from verity.ml.circuits.layout import FunctionCall
from verity.ml.circuits.ports import forward, harness
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine
from verity.ml.circuits.types import Type, Value

__all__ = [
    "ExecutionTranscript",
    "MissingValue",
    "RegisteredTable",
    "ScalarReference",
    "Target",
    "evaluate",
    "evaluate_call",
    "evaluate_region",
    "key_of",
    "refuse_registered_tables",
    "resolve",
    "signature_of",
    "standalone_call",
    "statics_of",
]


@dataclass(frozen=True, eq=False)
class ScalarReference:
    """A scalar function on encodings with a signature in the Program's types, for silicon semantics no core Definition wraps yet.

    ``fn(*args, **bindings)`` takes one int per ``Value`` parameter and a flat int sequence per aggregate parameter (leaf
    order), and returns an int (a ``Value`` result) or a flat int sequence.  ``signature(**bindings)`` returns
    ``(param_types, ret_type)``.  ``id`` must not look like a Definition id (``Name_vN``); use a dotted name."""

    id: str
    fn: Callable[..., Any] = field(repr=False)
    signature: Callable[..., tuple[Sequence[Type], Type]] = field(repr=False)
    statics: tuple[str, ...] = ()
    bindings: dict = field(default_factory=dict)
    doc: str = ""

    def bind(self, **kw: Any) -> "ScalarReference":
        unknown = set(kw) - set(self.statics)
        if unknown:
            raise TypeError(f"{self.id} has no static parameter(s) {sorted(unknown)}")
        return replace(self, bindings={**self.bindings, **kw})

    @property
    def complete(self) -> bool:
        return all(k in self.bindings for k in self.statics)


@dataclass(frozen=True, eq=False)
class RegisteredTable:
    """A table held in a registered input, as an operand of `evaluate`: ``word(j)`` is the word of its leaf ``j`` (``0 <= j <
    leaves``, in the parameter's leaf order), called only for the leaves a gate reads.

    ``declared_by`` is a primitive that declares ``name`` as a ``registered`` table (`PrimitiveDefinition.tables`); no other
    table is taken this way: a ``library`` table is an operation's own definition, gates and never an operand.  It binds an
    aggregate parameter of ``leaves`` leaves, of one width (one `Input` batch of the target's standalone Program)."""

    name: str
    word: Callable[[int], int] = field(repr=False)
    leaves: int
    declared_by: PrimitiveDefinition

    def __post_init__(self) -> None:
        kinds = {k for n, k in getattr(self.declared_by, "tables", ()) if n == self.name}
        if "registered" not in kinds:
            what = "it as a library table, an operation's own definition" if "library" in kinds else "no table of that name"
            raise ValueError(f"{self.name!r} is not a registered table of {self.declared_by.id}: it declares {what}")
        if self.leaves < 1:
            raise ValueError(f"registered table {self.name!r}: {self.leaves} leaves")

    def read(self, j: int) -> int:
        """Leaf ``j``'s word, indexed as the flattened form's list is (a negative ``j`` counts from the end)."""
        if j < 0:
            j += self.leaves
        if not 0 <= j < self.leaves:
            raise IndexError(f"registered table {self.name!r} has {self.leaves} leaves, not {j}")
        return int(self.word(j))


def refuse_registered_tables(target_id: str, inputs: Sequence[Any], who: str) -> None:
    """Raise, by name, for a `RegisteredTable` among ``inputs``: ``who`` takes arrays, one row per input."""
    for j, a in enumerate(inputs):
        if isinstance(a, RegisteredTable):
            raise TypeError(f"{target_id} argument {j}: {who} takes arrays, not the registered table {a.name!r} "
                            f"({a.leaves} leaves); `verity.ml.circuits.evaluation.evaluate` reads it lazily, one instance at a time")


Target = PrimitiveDefinition | CompositeDefinition | SpecializedDefinition | ScalarReference


def resolve(target: Target) -> PrimitiveDefinition | SpecializedDefinition | ScalarReference:
    if isinstance(target, CompositeDefinition):
        target = bind(target)
    if isinstance(target, (SpecializedDefinition, ScalarReference)) and not target.complete:
        raise TypeError(f"{target.id} is not fully bound")
    return target


def key_of(target: Target | str) -> str:
    """The registry key of a target: the Definition id without statics (``GemmCoordinate_v2``), or the reference's id."""
    if isinstance(target, str):
        return target
    if isinstance(target, SpecializedDefinition):
        return target.base_id
    return target.id


def statics_of(target: Target) -> dict:
    t = resolve(target)
    if isinstance(t, SpecializedDefinition):
        return dict(t.bindings)
    if isinstance(t, ScalarReference):
        return dict(t.bindings)
    return {}


def signature_of(target: Target) -> tuple[tuple[Type, ...], Type]:
    t = resolve(target)
    if isinstance(t, ScalarReference):
        params, ret = t.signature(**t.bindings)
        return tuple(params), ret
    params, ret = t.signature()
    return tuple(pt for _, pt in params), ret


_STANDALONE = harness(CompositeDefinition("EvaluationStandalone", 1, ("FN",), lambda S: S.FN.signature(),
                                          lambda B, S, *args: B.call(S.FN, *args),
                                          doc="root wrapper: one call of FN over Input gates", register=False,
                                          ports=lambda S: forward(S.FN)))
_CALLS: dict[SpecializedDefinition, tuple[Program, FunctionCall]] = {}


def standalone_call(spec: SpecializedDefinition) -> tuple[Program, FunctionCall]:
    """The one-call Program of ``spec`` and that call: the gate list :func:`evaluate` walks for a composite."""
    hit = _CALLS.get(spec)
    if hit is None:
        prog = Program(bind(_STANDALONE, FN=spec))
        calls = [c for c in prog.circuit.root_calls() if not c.is_primitive]
        if len(calls) != 1:
            raise AssertionError(f"standalone program of {spec.id} has {len(calls)} calls")
        hit = _CALLS[spec] = (prog, calls[0])
    return hit


def _flat(args: Sequence[Any]) -> list[int]:
    out: list[int] = []
    for a in args:
        if isinstance(a, int):
            out.append(a)
        else:
            out.extend(int(v) for v in a)
    return out


def _leaves(a: Any) -> list[int]:
    return [a] if isinstance(a, int) else [int(v) for v in a]


class _Operands:
    """The flattened operand list of ``args`` with each `RegisteredTable`'s words read on demand: ``ops[j]`` is ``flat[j]``."""

    def __init__(self, args: Sequence[Any]) -> None:
        self.starts: list[int] = []
        self.parts: list[Any] = []
        n = 0
        for a in args:
            part = a if isinstance(a, RegisteredTable) else _leaves(a)
            self.starts.append(n)
            self.parts.append(part)
            n += part.leaves if isinstance(part, RegisteredTable) else len(part)
        self.n = n

    def __getitem__(self, j: int) -> int:
        if j < 0:
            j += self.n
        if not 0 <= j < self.n:
            raise IndexError("list index out of range")
        k = bisect.bisect_right(self.starts, j) - 1
        part, i = self.parts[k], j - self.starts[k]
        return part.read(i) if isinstance(part, RegisteredTable) else part[i]


class _Transcript(dict):
    """An execution trace whose registered tables' Input gates are read through the table, never stored.

    ``in`` doesn't see those gates, only ``[]`` does. `evaluate_region` asks ``in`` only of Input gates inside its selection,
    and a standalone call's selection holds none."""

    def __init__(self) -> None:
        super().__init__()
        self.los: list[int] = []
        self.tables: list[RegisteredTable] = []

    def add_table(self, lo: int, table: RegisteredTable) -> None:
        k = bisect.bisect_left(self.los, lo)
        self.los.insert(k, lo)
        self.tables.insert(k, table)

    def __missing__(self, g: int) -> int:
        k = bisect.bisect_right(self.los, g) - 1
        if k >= 0 and g - self.los[k] < self.tables[k].leaves:
            return self.tables[k].read(g - self.los[k])
        raise KeyError(g)


def _evaluate_with_tables(t: Any, args: Sequence[Any]) -> tuple[int, ...]:
    """`evaluate` with a `RegisteredTable` among ``args``: each binds a whole aggregate parameter, and only the words the gates
    read are read."""
    params, _ = signature_of(t)
    if len(args) != len(params):
        raise TypeError(f"{t.id}: expected {len(params)} arguments, got {len(args)}")
    for j, (a, pt) in enumerate(zip(args, params)):
        if not isinstance(a, RegisteredTable):
            continue
        if isinstance(pt, Value) or pt.leaves != a.leaves or pt.homogeneous_width() is None:
            raise TypeError(f"{t.id} argument {j}: the registered table {a.name!r} ({a.leaves} leaves) binds an aggregate "
                            f"parameter of as many leaves of one width, not {pt!r}")
    if isinstance(t, ScalarReference):
        raise TypeError(f"{t.id} is a ScalarReference, whose function takes every leaf: a registered table is read through "
                        "a Definition's gates")
    if isinstance(t, PrimitiveDefinition):
        lazy = getattr(t, "evaluate_lazy", None)
        if lazy is None:
            raise TypeError(f"{t.id} has no evaluate_lazy: its evaluator takes every leaf of the registered table")
        return (lazy(_Operands(args).__getitem__),)
    prog, call = standalone_call(t)
    tr = _Transcript()
    for pi, a in enumerate(args):
        refs = prog.input_coll(pi).refs
        if isinstance(a, RegisteredTable):
            if not (isinstance(refs, Affine) and refs.count == a.leaves):
                raise TypeError(f"{t.id} argument {pi}: the registered table {a.name!r} ({a.leaves} leaves) needs one "
                                f"contiguous run of as many Input gates, not {refs!r}")
            tr.add_table(refs.base, a)
            continue
        vals = _leaves(a)
        if len(vals) != refs.count:
            raise ValueError(f"{t.id} argument {pi}: expected {refs.count} leaves, got {len(vals)}")
        tr.update((refs[k][2], v) for k, v in enumerate(vals))
    evaluate_region(prog.circuit, call, tr)
    return tuple(tr[g] for g in call.returns)


def evaluate(target: Target, *args: Any) -> tuple[int, ...]:
    """``target`` on one input of plain ints (one int per ``Value`` parameter, a flat int sequence per aggregate
    parameter, in leaf order); returns the output leaves.  Raises whatever the reference raises outside its domain.

    An aggregate parameter that holds a registered table may take a `RegisteredTable` instead of its words: a primitive then
    reads it through its ``evaluate_lazy``, a composite's gates through the trace, and the result is the same."""
    t = resolve(target)
    if any(isinstance(a, RegisteredTable) for a in args):
        return _evaluate_with_tables(t, args)
    if isinstance(t, ScalarReference):
        params, _ = signature_of(t)
        r = t.fn(*(a if isinstance(pt, Value) else list(a) for a, pt in zip(args, params)), **t.bindings)
        return (r,) if isinstance(r, int) else tuple(r)
    flat = _flat(args)
    if isinstance(t, PrimitiveDefinition):
        lazy = getattr(t, "evaluate_lazy", None)
        return (lazy(flat.__getitem__),) if lazy is not None else (t.evaluate(*flat),)
    prog, call = standalone_call(t)
    return evaluate_call(prog.circuit, call, flat)
