"""The plain-integer reference: what every kernel is checked against.

`evaluate_region` / `evaluate_call` are the gate-by-gate oracle of `verity.ir.evaluate`, re-exported unchanged (the code
moves here once no importer of the old path is left).  `evaluate(target, *args)` runs one instance of a target on plain
ints: a primitive through its evaluator, a composite specialization through the oracle on a one-call Program.

A target is an IR Definition (`PrimitiveDefinition`, a static-free `CompositeDefinition`, a complete
`SpecializedDefinition`) or a `ScalarReference`: a silicon function that no core Definition wraps yet (the FP8 and NVFP4
steps), with a signature in IR types so that batching, sampling and checking treat both alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from verity.ir.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ir.evaluate import ExecutionTranscript, MissingValue, evaluate_call, evaluate_region
from verity.ir.layout import FunctionCall
from verity.ir.program import Program
from verity.ir.types import Type, Value

__all__ = [
    "ExecutionTranscript",
    "MissingValue",
    "ScalarReference",
    "Target",
    "evaluate",
    "evaluate_call",
    "evaluate_region",
    "key_of",
    "resolve",
    "signature_of",
    "standalone_call",
    "statics_of",
]


@dataclass(frozen=True, eq=False)
class ScalarReference:
    """A scalar function on encodings with an IR-typed signature, for silicon semantics no core Definition wraps yet.

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


_STANDALONE = CompositeDefinition("EvaluationStandalone", 1, ("FN",), lambda S: S.FN.signature(),
                                  lambda B, S, *args: B.call(S.FN, *args),
                                  doc="root wrapper: one call of FN over Input gates", register=False)
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


def evaluate(target: Target, *args: Any) -> tuple[int, ...]:
    """One instance of ``target`` on plain ints (one int per ``Value`` parameter, a flat int sequence per aggregate
    parameter, in leaf order); returns the output leaves.  Raises whatever the reference raises outside its domain."""
    t = resolve(target)
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
