"""Batch invariance of a template: what a request's outputs read, and whether its batch picks the Definition.

A template is batch-invariant when no request's outputs depend on its batch-mates. Two properties, checked in the clear:

    separable   (a) each owner's outputs read, of every parameter, only the leaves that owner holds and shared ones, never a
                batch leaf: their dependency cone (`liveness.Liveness`, transitive: the leaves a gate with a path to them reads)
    invariant   (b) a role binds one Definition at each request coordinate whatever launch shape it runs in: one id, or
                distinct ids of one structure (`structure`)

A label says who holds a parameter's leaves: `SHARED` (weights, tables, constants: any output may read them), `BATCH` (a value
the batch picks, such as the top-p split count S = SplitsFor_v1(live rows): no output may read it), `Rows(owners)` (equal rows,
row i held by ``owners[i]``) or `Fields(labels)` (a tuple or a record, field by field). An owner is any hashable request label.

`circuit-check --batch` runs both over the templates a compiler declares (`verity_vllm.program.predict.batch.templates`), and the
compiler holds each composition to (a) (`verity_vllm.program.predict.batch.check_compiled`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Sequence

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ml.circuits.intervals import Intervals, full, isect, norm
from verity.ml.circuits.liveness import Liveness
from verity.ml.circuits.types import Record, Tuple, Type, field_starts, field_types


class _Mark:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name.upper()


SHARED = _Mark("shared")
BATCH = _Mark("batch")


@dataclass(frozen=True)
class Rows:
    """Equal rows of leaves, row i held by ``owners[i]``; a tuple or a record splits each field into the same rows."""

    owners: tuple[Hashable, ...]


@dataclass(frozen=True)
class Fields:
    """A tuple's or a record's fields, each by its own label."""

    labels: tuple[Any, ...]


def own(owner: Hashable) -> Rows:
    """Every leaf held by ``owner``."""
    return Rows((owner,))


def _held(label: Any, t: Type, base: int = 0, out: dict | None = None) -> dict[Any, list[tuple[int, int]]]:
    out = {} if out is None else out
    if label is SHARED or label is BATCH:
        out.setdefault(label, []).append((base, base + t.leaves))
    elif isinstance(label, Fields):
        if not isinstance(t, (Tuple, Record)) or len(field_types(t)) != len(label.labels):
            raise ValueError(f"{len(label.labels)} field labels for {t!r}")
        for start, ft, lab in zip(field_starts(t), field_types(t), label.labels):
            _held(lab, ft, base + start, out)
    elif isinstance(label, Rows):
        if isinstance(t, (Tuple, Record)):
            for start, ft in zip(field_starts(t), field_types(t)):
                _held(label, ft, base + start, out)
            return out
        n = len(label.owners)
        if n == 0 or t.leaves % n:
            raise ValueError(f"{t!r} ({t.leaves} leaves) doesn't split into {n} equal rows")
        step = t.leaves // n
        for i, o in enumerate(label.owners):
            if o is SHARED or o is BATCH:
                raise ValueError("a row is held by an owner, not SHARED or BATCH")
            out.setdefault(o, []).append((base + i * step, base + (i + 1) * step))
    else:
        raise TypeError(f"a label is SHARED, BATCH, Rows or Fields, not {label!r}")
    return out


def held(label: Any, t: Type) -> dict[Any, Intervals]:
    """The leaves of a value of type ``t`` by holder (an owner, `SHARED` or `BATCH`), as ``label`` says."""
    return {k: norm(v) for k, v in _held(label, t).items()}


@dataclass(frozen=True)
class Crossing:
    """``owner``'s outputs read leaves of parameter ``param`` that ``holder`` (another owner, or `BATCH`) holds."""

    owner: Hashable
    param: str
    holder: Any
    leaves: Intervals
    approximate: bool = False

    def __str__(self) -> str:
        lo, hi = self.leaves[0]
        whose = "a batch-derived value" if self.holder is BATCH else f"owner {self.holder!r}"
        return (f"the outputs of owner {self.owner!r} read {self.param} leaves {lo}…{hi - 1}"
                f"{f' (+{len(self.leaves) - 1} runs)' if len(self.leaves) > 1 else ''}, held by {whose}"
                f"{' (an over-approximated cone)' if self.approximate else ''}")


def _specialized(fn: Any) -> Any:
    return bind(fn) if isinstance(fn, CompositeDefinition) else fn


def separable(fn: Any, params: Sequence[Any], ret: Any, *, liveness: Liveness | None = None) -> list[Crossing]:
    """(a) Where an owner's outputs read what it doesn't hold: for each owner of ``fn``'s return (``ret``, in order of its first
    leaf), the parameters' leaves its outputs' cone reads that another owner or `BATCH` holds (``params``, one label each). Empty:
    every owner's outputs read only its own leaves and shared ones. A primitive reads all of every parameter."""
    sd = _specialized(fn)
    if isinstance(sd, PrimitiveDefinition):
        ptypes, ret_type = list(sd.params), sd.ret
    else:
        ptypes, ret_type = list(sd.body.params), sd.body.ret_type
    if len(params) != len(ptypes):
        raise ValueError(f"{len(params)} labels for {sd.id}'s {len(ptypes)} parameters")
    outs = held(ret, ret_type)
    if SHARED in outs or BATCH in outs:
        raise ValueError("a template's outputs are held by owners")
    holders = [held(lab, t) for lab, (_, t) in zip(params, ptypes)]
    lv = liveness or Liveness(transitive=True)
    found: list[Crossing] = []
    for owner, live in sorted(outs.items(), key=lambda kv: kv[1][0]):
        if isinstance(sd, PrimitiveDefinition):
            reads, approx = tuple(full(t.leaves) for _, t in ptypes), False
        else:
            res = lv.analyse(sd, live)
            reads, approx = res.reads, res.approximate
        for (name, _), h, r in zip(ptypes, holders, reads):
            for holder, ivs in h.items():
                if holder is SHARED or holder == owner:
                    continue
                if hit := isect(r, ivs):
                    found.append(Crossing(owner, name, holder, hit, approx))
    return found


def structure(fn: Any, _memo: dict[int, tuple[Any, str]] | None = None) -> str:
    """``fn``'s structure digest: a primitive's id; a composite's body (its parameter and return types, each node's form, callee
    by structure, argument types and references, output type, batch axes and counts, and the return's references), leaving out
    its own name and statics. One structure computes one function; distinct structures may still agree."""
    memo = {} if _memo is None else _memo
    hit = memo.get(id(fn))
    if hit is not None and hit[0] is fn:
        return hit[1]
    sd = _specialized(fn)
    if isinstance(sd, PrimitiveDefinition):
        d = hashlib.sha512(b"primitive\0" + sd.id.encode()).hexdigest()
    else:
        b = sd.body
        nodes = []
        for n in b.nodes:
            e = [n.form, structure(n.fn, memo), [[a.type.to_json(), a.refs.to_json()] for a in n.args], n.out.to_json()]
            if n.form == "batch":
                e += [list(n.axes), n.n]
            elif n.form == "scan":
                e += [n.n, n.n_xs]
            nodes.append(e)
        body = {"params": [t.to_json() for _, t in b.params], "ret_type": b.ret_type.to_json(), "nodes": nodes, "ret": b.ret.to_json()}
        d = hashlib.sha512(b"composite\0" + json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    memo[id(fn)] = (fn, d)
    return d


@dataclass(frozen=True)
class Split:
    """At request coordinate ``request``, launch ``launches[0]`` binds ``ids[0]`` and launch ``launches[1]`` binds ``ids[1]``,
    of distinct structures."""

    request: Any
    launches: tuple[Any, Any]
    ids: tuple[str, str]

    def __str__(self) -> str:
        return (f"at {self.request}: launch {self.launches[0]} binds {self.ids[0]}, launch {self.launches[1]} binds {self.ids[1]} "
                f"(distinct structures)")


def invariant(bind_at: Callable[[Any, Any], Any], shapes: Iterable[tuple[Any, Any]], *, limit: int = 1) -> list[Split]:
    """(b) Where the launch picks the Definition: for each (request coordinate, launch shape) of ``shapes``, the Definition
    ``bind_at(request, launch)`` against the first one bound at that request coordinate; one id or one `structure` agrees. At most
    ``limit`` splits. Empty: one Definition at every request coordinate."""
    first: dict[Any, tuple[Any, Any]] = {}
    memo: dict = {}
    found: list[Split] = []
    for req, launch in shapes:
        fn = bind_at(req, launch)
        if req not in first:
            first[req] = (launch, fn)
            continue
        l0, f0 = first[req]
        if fn.id == f0.id or structure(fn, memo) == structure(f0, memo):
            continue
        found.append(Split(req, (l0, launch), (f0.id, fn.id)))
        if len(found) >= limit:
            break
    return found


@dataclass(frozen=True)
class Template:
    """A template as a compiler emits it, with what each property reads and the verdicts declared beside it.

    (a) reads ``fn``, its parameters' labels ``params`` and its return's ``ret``; (b) reads ``bind_at`` over ``shapes`` (request
    coordinate, launch shape). ``expect`` declares (a)'s and (b)'s verdicts (True passes, False fails, None: not checked), and
    ``why`` says what a declared failure is."""

    name: str
    fn: Any = None
    params: tuple[Any, ...] = ()
    ret: Any = None
    bind_at: Callable[[Any, Any], Any] | None = field(default=None, compare=False)
    shapes: tuple[tuple[Any, Any], ...] = ()
    expect: tuple[bool | None, bool | None] = (True, True)
    why: str = ""


@dataclass(frozen=True)
class TemplateInvariance:
    """A template's verdicts: (a)'s crossings and (b)'s splits, each None where it isn't checked."""

    template: Template
    crossings: tuple[Crossing, ...] | None
    splits: tuple[Split, ...] | None
    shapes: int

    @property
    def got(self) -> tuple[bool | None, bool | None]:
        return (None if self.crossings is None else not self.crossings, None if self.splits is None else not self.splits)

    @property
    def ok(self) -> bool:
        """Each verdict is the one declared."""
        return self.got == self.template.expect

    def to_json(self) -> dict:
        t = self.template
        return {"template": t.name, "id": getattr(t.fn, "id", None), "expect": list(t.expect), "got": list(self.got), "ok": self.ok,
                "crossings": None if self.crossings is None else [str(c) for c in self.crossings],
                "splits": None if self.splits is None else [str(s) for s in self.splits], "shapes": self.shapes, "why": t.why}


def check(templates: Iterable[Template], *, liveness: Liveness | None = None) -> list[TemplateInvariance]:
    """Each template's verdicts."""
    lv = liveness or Liveness(transitive=True)
    out = []
    for t in templates:
        crossings = None if t.fn is None else tuple(separable(t.fn, t.params, t.ret, liveness=lv))
        splits = None if t.bind_at is None else tuple(invariant(t.bind_at, t.shapes))
        out.append(TemplateInvariance(t, crossings, splits, len(t.shapes)))
    return out


def report(verdicts: Sequence[TemplateInvariance]) -> str:
    """The verdicts as a Markdown table: each template, (a) and (b) with their first witness, and whether each is the one declared."""
    def cell(got: bool | None, witness: tuple | None) -> str:
        if got is None:
            return "—"
        return "pass" if got else f"**fail**: {witness[0]}"

    rows = ["| template | (a) separable | (b) shape-invariant | shapes | as declared |", "|---|---|---|---|---|"]
    for v in verdicts:
        a, b = v.got
        rows.append(f"| `{v.template.name}` | {cell(a, v.crossings)} | {cell(b, v.splits)} | {v.shapes or '—'} | "
                    f"{'yes' if v.ok else '**NO**'} |")
    return "\n".join(rows)
