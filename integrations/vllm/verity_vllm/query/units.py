"""Queries over a Program and the values they require (vLLM integration v2, Draft 3.1 §12; Phase 2).

A *query* ``Q`` partitions the Calls of a Program ``P`` into *proof blocks*; the values a verifier must be able to
open are the values that cross a unit boundary, plus what the protocol prescribes beside them::

    Q(P)                      -> a sequence of VerificationUnit (a stable id + a set of Call ids)
    boundary_values(P, Q)     =  U_{V in Q(P)} ( in(V) u out(V) )
    required_values(P, Q, pi) =  boundary_values(P, Q)  u  pi.prescribed_inputs  u  pi.protocol_required

where, for a unit ``V``: ``in(V)`` is every Value produced *outside* ``V`` (by a Call of another unit, or by a
prescribed input of the Program) that a Call inside ``V`` reads, and ``out(V)`` is every Value produced inside ``V``
that a Call outside ``V`` reads or that is a Program output.  A Value is a *logical* identity ``(producer, port)``:
the Call that produces it and the output port it leaves the Call by (``verity.ml.circuits`` invariant: "Composite interfaces ->
ordered collections of value references"; a port is one such collection).  Nothing here is a commitment position, a
tensor, a module or a kernel.

This module is deliberately small and dependency-free: it knows Calls, ports, reads, units.  What a Call *is* (a
Definition instance of some registry), how a Program is stored, which annotation says which module owns a Call, and how
an identity is spelt for a collector are the integration's concern (``verity_vllm.query``).  The one-unit / two-unit
example of the design (``x -- Matmul -- u -- Activation -- y``: one unit -> {x, y}; two units -> {x, u, y}) is the
contract; see ``integrations/vllm/tests/query/test_units.py``.

The Program is read through the :class:`Program` protocol so an integration can present a decoded ``verity.ml.circuits`` descriptor,
a streamed instance sequence, or a synthetic graph without converting it: a Program is a sequence of :class:`Call` records
(id, definition id, output ports, the Values each read), the Values its prescribed inputs produce, and its outputs.
Prescribed inputs are modelled as *Values of a producer that is not a Call* (``ValueRef.call`` is the input's id, which
the Program lists in ``inputs``), exactly as a Program gives ``Input`` gates their own canonical references.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, runtime_checkable

__all__ = ["ValueRef", "Call", "Program", "VerificationUnit", "Query", "Partition", "partition_by", "unit_boundary",
           "boundary_values", "Policy", "RequiredValues", "required_values", "admissibility", "PartitionError"]


class ValueRef(NamedTuple):
    """One logical Value: the ``port`` of producer ``call``.  ``call`` is a Call id of the Program, or the id of one of its
    prescribed inputs (``Program.inputs`` says which); ``port`` names the output collection it leaves by (``"out"`` for a
    Definition returning one collection, ``"0"``/``"1"``/... for the slots of a tuple, a field name for a record input).
    A tuple, so millions of them hash and compare at C speed (a derived request Program has 10^6 Calls)."""

    call: int
    port: str


@runtime_checkable
class Call(Protocol):
    """One Call of the Program's flat body.  ``reads`` are the Values it consumes (any port, any producer, duplicates
    allowed); ``ports`` the output collections it produces (``("out",)`` for a single-collection Definition)."""

    @property
    def id(self) -> int: ...

    @property
    def definition(self) -> str: ...

    @property
    def ports(self) -> Sequence[str]: ...

    @property
    def reads(self) -> Sequence[ValueRef]: ...


@runtime_checkable
class Program(Protocol):
    """What a query reads.  ``calls`` in body order (ids unique; a Call id never equals an input id); ``inputs`` the Values
    the prescribed inputs produce (their ``call`` ids are not Call ids); ``outputs`` the Values the Program returns."""

    @property
    def calls(self) -> Sequence[Call]: ...

    @property
    def inputs(self) -> Sequence[ValueRef]: ...

    @property
    def outputs(self) -> Sequence[ValueRef]: ...


@dataclass(frozen=True)
class VerificationUnit:
    """A set of Call ids under a stable id.  The id is the query's (a module path, a layer index, ...); two queries that
    put the same Calls in a unit under different ids are different queries."""

    id: str
    calls: frozenset[int]

    def __post_init__(self) -> None:
        if not self.calls:
            raise ValueError(f"unit {self.id!r}: a proof block has at least one Call")


#: ``Q(P) -> units``.  A ``Query`` is any callable of this shape; :func:`partition_by` builds one from a Call -> unit-id map.
Query = Callable[[Program], Sequence[VerificationUnit]]
#: A query whose units are a partition of the Program's Calls (every Call in exactly one unit); :func:`unit_boundary` and
#: :func:`boundary_values` require this and check it (:class:`PartitionError`).
Partition = Query


class PartitionError(ValueError):
    """The units are not a partition of the Program's Calls (a Call in two units, or in none)."""


def partition_by(key: Callable[[Call], str], *, order: Callable[[str], object] | None = None) -> Partition:
    """The query whose units are the classes of ``key`` over the Program's Calls: unit id = the key, unit = every Call with that
    key.  Units are returned in first-seen body order unless ``order`` sorts the ids.  The canonical way to write a
    membership-derived query (``key`` = the owning module path, the layer, the Definition family, ...)."""

    def Q(P: Program) -> Sequence[VerificationUnit]:
        groups: dict[str, set[int]] = {}
        for c in P.calls:
            groups.setdefault(key(c), set()).add(c.id)
        ids: Iterable[str] = groups if order is None else sorted(groups, key=order)
        return tuple(VerificationUnit(k, frozenset(groups[k])) for k in ids)

    return Q


def _index(P: Program) -> tuple[dict[int, Call], frozenset[int]]:
    by_id = {c.id: c for c in P.calls}
    if len(by_id) != len(P.calls):
        raise ValueError("Program: Call ids are not unique")
    inputs = frozenset(v.call for v in P.inputs)
    clash = inputs & by_id.keys()
    if clash:
        raise ValueError(f"Program: input ids {sorted(clash)[:4]} are also Call ids")
    return by_id, inputs


def _check_partition(P: Program, units: Sequence[VerificationUnit]) -> dict[int, str]:
    """Call id -> unit id; raises :class:`PartitionError` unless every Call is in exactly one unit."""
    owner: dict[int, str] = {}
    for u in units:
        for cid in u.calls:
            if cid in owner:
                raise PartitionError(f"Call {cid} is in units {owner[cid]!r} and {u.id!r}")
            owner[cid] = u.id
    missing = [c.id for c in P.calls if c.id not in owner]
    if missing:
        raise PartitionError(f"{len(missing)} Call(s) in no unit (first: {missing[:4]})")
    unknown = sorted(set(owner) - {c.id for c in P.calls})
    if unknown:
        raise PartitionError(f"units name {len(unknown)} id(s) that are not Calls (first: {unknown[:4]})")
    return owner


def unit_boundary(P: Program, unit: VerificationUnit, owner: Mapping[int, str] | None = None) -> tuple[frozenset[ValueRef], frozenset[ValueRef]]:
    """``(in(V), out(V))`` of one unit.  ``owner`` (Call id -> unit id, from the partition) lets the outputs be decided; without
    it, every read from outside the unit's own Calls counts as outside (correct for a partition, computed per call)."""
    by_id, _inputs = _index(P)
    inside = unit.calls
    ins: set[ValueRef] = set()
    for cid in inside:
        for v in by_id[cid].reads:
            if v.call not in inside:                     # produced by another unit's Call or by a prescribed input
                ins.add(v)
    outs: set[ValueRef] = set()
    for c in P.calls:
        if c.id in inside:
            continue
        for v in c.reads:
            if v.call in inside:
                outs.add(v)
    for v in P.outputs:
        if v.call in inside:
            outs.add(v)
    return frozenset(ins), frozenset(outs)


@dataclass(frozen=True)
class Boundary:
    """The boundary of a partition: every crossing Value, with the units it leaves (``produced_in``: unit id, or None for a
    prescribed input) and enters (``read_by``: unit ids, in body order of the reading Calls)."""

    values: frozenset[ValueRef]
    produced_in: Mapping[ValueRef, str | None]
    read_by: Mapping[ValueRef, tuple[str, ...]]
    units: tuple[VerificationUnit, ...]
    owner: Mapping[int, str]


#: `boundary` keeps a Value's readers in a list, and a set beside it once there are this many
_FEW_READERS = 16


def boundary(P: Program, Q: Partition) -> Boundary:
    """One pass over the Program: the crossing Values of ``Q(P)`` with their producing and reading units."""
    units = tuple(Q(P))
    owner = _check_partition(P, units)
    _index(P)
    produced_in: dict[ValueRef, str | None] = {}
    read_by: dict[ValueRef, list[str]] = {}
    many: dict[ValueRef, set[str]] = {}                  # the readers of a Value read by many units, as a set beside the list
    owner_get = owner.get
    for c in P.calls:
        me = owner[c.id]
        for v in c.reads:
            producer = owner_get(v[0])                   # None: a prescribed input (never interior)
            if producer == me:
                continue                                 # interior to the unit: not a boundary Value
            rb = read_by.get(v)
            if rb is None:
                produced_in[v] = producer
                read_by[v] = [me]
            elif len(rb) < _FEW_READERS:
                if me not in rb:
                    rb.append(me)
            else:                                        # a key every later token's attention reads: a scan of the list would be O(T)
                seen = many.get(v)
                if seen is None:
                    seen = many[v] = set(rb)
                if me not in seen:
                    seen.add(me)
                    rb.append(me)
    for v in P.outputs:                                  # a Program output leaves its unit whoever reads it
        if v not in read_by:
            produced_in[v] = owner_get(v[0])
            read_by[v] = []
    return Boundary(frozenset(produced_in), produced_in, {v: tuple(r) for v, r in read_by.items()}, units, owner)


def boundary_values(P: Program, Q: Partition) -> frozenset[ValueRef]:
    """``U_{V in Q(P)} (in(V) u out(V))`` as logical Values.  In a partition every ``in(V)`` produced by a Call is some other
    unit's ``out``, so the union is: every Call-produced Value read across a unit boundary or returned by the Program, plus
    every prescribed-input Value some Call reads."""
    return boundary(P, Q).values


@dataclass(frozen=True)
class Policy:
    """What the protocol requires beside the boundary -- stated, never inferred from a query or a Program.

    ``prescribed_inputs``: Values of the Program's inputs required whether or not a Call reads them (weights of record, a
    seed).  ``protocol_required``: Values a protocol names outright (a kernel's exported stream, a sampler's output) -- any
    Value the Program has, referred to by its identity.  Both are plain sets; a policy is data."""

    prescribed_inputs: frozenset[ValueRef] = field(default_factory=frozenset)
    protocol_required: frozenset[ValueRef] = field(default_factory=frozenset)

    @staticmethod
    def none() -> "Policy":
        return Policy()


@dataclass(frozen=True)
class RequiredValues:
    """``required_values(P, Q, policy)``: the three sources kept apart (a reader can tell a boundary Value from a prescribed
    one) and their union.  Every Value is checked to be one the Program has (a policy naming a Value of no producer is an
    error, not a silent requirement)."""

    boundary: frozenset[ValueRef]
    prescribed: frozenset[ValueRef]
    protocol: frozenset[ValueRef]

    @property
    def values(self) -> frozenset[ValueRef]:
        return self.boundary | self.prescribed | self.protocol

    def __len__(self) -> int:
        return len(self.values)

    def __contains__(self, v: object) -> bool:
        return v in self.values


def required_values(P: Program, Q: Partition, policy: Policy) -> RequiredValues:
    """Boundary Values of ``Q(P)`` u the policy's prescribed inputs u the policy's protocol-required Values."""
    by_id, inputs = _index(P)
    known_inputs = frozenset(P.inputs)
    for v in policy.prescribed_inputs:
        if v not in known_inputs:
            raise ValueError(f"policy prescribes {v}, which is not a prescribed input of the Program")
    for v in policy.protocol_required:
        if v.call not in by_id and v not in known_inputs:
            raise ValueError(f"policy requires {v}, which no Call or input of the Program produces")
        if v.call in by_id and v.port not in by_id[v.call].ports:
            raise ValueError(f"policy requires {v}, but Call {v.call} has ports {tuple(by_id[v.call].ports)}")
    return RequiredValues(boundary_values(P, Q), frozenset(policy.prescribed_inputs), frozenset(policy.protocol_required))


def admissibility(required: Iterable[ValueRef], acquisition_support: Mapping[ValueRef, bool] | Callable[[ValueRef], bool] | Iterable[ValueRef]) -> list[ValueRef]:
    """The required Values the acquisition cannot support, sorted.  ``acquisition_support`` is a table (a set of the supported
    Values, a mapping Value -> bool, or a predicate) -- an *input*: this module has no collector.  Empty result = admissible."""
    if isinstance(acquisition_support, Mapping):
        ok = lambda v: bool(acquisition_support.get(v, False))  # noqa: E731
    elif callable(acquisition_support):
        ok = acquisition_support
    else:
        s = frozenset(acquisition_support)
        ok = s.__contains__
    return sorted(v for v in set(required) if not ok(v))
