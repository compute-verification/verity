"""Partition and width validation of a query result *used* as a proof-unit decomposition.

Partitionhood is not a property of queries: a `Family` (an ordered sequence of `GateSet`s, `query_ast`) may overlap or
omit gates, and nothing in the query model forbids it.  It is a property checked here, when a family is used as the
decomposition of a Program into proof units.  Three checks, one verdict each, composed by `validate_vu_partition`:

* `validate_partition` -- every gate of the Program (Input and constant gates included) in exactly one set.
  Fast path (tier A, design §4): the family's structural description `Family.tiling()` is checked by per-body
  induction -- in every body the tiling refines, every node index is assigned exactly once, `zip` groups have equal
  `n`, member ranges partition `[0, n)`, descents land on the callee's body.  Instances of a specialization are
  layout-identical (offsets are intrinsic), so a per-body tiling is an exact partition of every instance and, by
  induction along the hierarchy from the root body (Input nodes are ordinary nodes), of the whole circuit.  Zero gate
  visits, O(nodes) per refined body.  Fallback (tier B): materialise every set's intervals, sort the endpoints and
  sweep -- overlap iff two intervals intersect, coverage iff the union is `[0, program.gates)` -- up to
  `Limits.max_intervals` intervals; beyond that the answer is `unsupported`, never a verdict.
* `validate_width` -- `w_out(S) <= w_max` for every set.  Per member class (one class per tiling rule; without a
  tiling, one class per set): `boundary.w_out_bound(part) <= w_max` certifies the class; otherwise the exact
  `boundary.w_out(part)` decides per set, within `Limits.max_members_exact` exact evaluations.  A bound above `w_max`
  proves nothing; `Unsupported` is reported separately and is neither a pass nor a fail.
* `readiness_32bit` -- does every primitive gate of the Program produce <= 32 bits?  Violators are listed by
  specialization and primitive (`Const64[..]` in the Gumbel body, `Input64`).  This is a Program property separate from
  the boundary condition; 32 is a protocol parameter (`w_max`), not a language constant.

A decomposition may have several levels, coarse to fine (a `Hierarchy` of named `Level`s, e.g. replay units and the
proof units inside them).  `validate_refinement` checks that each level's sets nest inside the level above and
counts them per parent (`members_per_parent`); `validate_partition(..., within=S)` checks a level that partitions only
the gates S (a level leaving out Inputs checked elsewhere).

Codes (`PartitionVerdict.codes`, `WidthVerdict.detail["codes"]`):
    tiling-unassigned        a refined body has a node no entry covers                     -> gap
    tiling-members-gap       a node's entries leave members of [0, n) uncovered            -> gap
    tiling-members-overlap   a node's entries cover a member twice                         -> overlap
    tiling-malformed         the description could not be interpreted (a zip group of unequal n, a descent on a
                             primitive or into the wrong specialization, an unknown rule): fallback to the sweep
    tiling-count-mismatch    the tiling's structural count != Family.count() (fallback to sweep)
    sweep-overlap / sweep-gap / sweep-outside (with `within`) / limit-intervals / empty-set
    width: limit-members-exact, limit-members-bound, multi-part-exact, boundary-unsupported, limit-sets

The tiling read by the fast path is `query_ast.Tiling`: `Tiling(root: plan key, bodies: Mapping[plan key,
BodyTiling])`; `BodyTiling(spec, entries, n_nodes?, count?)`; `NodeTiling(node, kind, members, k, group, child)` with `kind` "atomic" (k adjacent members of each range -> one set, last block shorter),
"whole" (one set per range), "zip" (member m of every node of the group; equal n, full ranges), "descend" (the members
of the range are tiled by `bodies[child]`).  A node's ranges over all its entries must partition `[0, n)`; bodies are
keyed by plan, not by specialization, so one specialization may be tiled differently in different contexts.  Several
`descend` entries of one node may send the same members into different plans (`concat(A, rest())`): those plans form a
*group* and the induction checks that the group's entries together tile the callee body (`CheckedTiling`).  Read duck-typed
(attributes or mapping keys).

Nothing here touches `program_digest`.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Literal, Mapping

from . import boundary
from .defs import PrimitiveDefinition, SpecializedDefinition
from .intervals import Intervals, complement, isect, norm
from .layout import Scope
from .parts import Limits, Part, Unsupported, n_members
from .query_ast import GateSet, evaluate

if TYPE_CHECKING:  # pragma: no cover
    from .program import Program

PartitionStatus = Literal["exact", "overlap", "outside", "gap", "unsupported"]

_MAX_LISTED = 64  # cap on listed indices / examples inside a verdict (totals are always reported)


# ---------------------------------------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionVerdict:
    """`status`: "exact" (a partition), "overlap", "outside" (only with `within`: a set reaches a gate outside it),
    "gap" (the first applicable wins; every code is listed), "unsupported" (the check exceeded `Limits`; not a
    verdict).  `detail` is JSON-serialisable: the path taken ("tiling" | "sweep"), counts, and bounded examples of the
    offending nodes / gate intervals."""
    status: PartitionStatus
    codes: tuple[str, ...]
    detail: dict

    def to_json(self) -> dict:
        return {"status": self.status, "codes": list(self.codes), "detail": self.detail}


@dataclass(frozen=True)
class WidthVerdict:
    """`status`: "ok" (every set certified `w_out <= w_max`), "violation" (some set has exact `w_out > w_max`;
    listed in `violators` as family indices when they can be resolved), "unsupported" (no violation proven but some
    sets could be neither certified nor refuted; listed in `unsupported`).  `max_seen` is the largest certified or
    exact value seen: exact where the exact path ran, else the certifying bound (so it is an upper bound on the true
    maximum, equal to it when every set went through the exact path), `None` when nothing was evaluated.
    `violators` / `unsupported` list at most 64 indices each; `detail` carries the totals and structural examples."""
    w_max: int
    max_seen: int | None
    violators: tuple[int, ...]
    unsupported: tuple[int, ...]
    status: str
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"w_max": self.w_max, "max_seen": self.max_seen, "violators": list(self.violators),
                "unsupported": list(self.unsupported), "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class ReadinessViolation:
    spec: str        # specialization id whose body holds the node (the root body is `Root[...]`)
    node: int        # body node index
    prim: str        # primitive id, e.g. `Const64[0x...]_v1`, `Input64_v1`
    w: int           # the gate's output width
    form: str        # "primitive" | "batch"
    members: int     # gates per instance of the body (n for a batch, 1 otherwise)
    instances: int   # instances of the body in the Program

    def to_json(self) -> dict:
        return {"spec": self.spec, "node": self.node, "prim": self.prim, "w": self.w, "form": self.form,
                "members": self.members, "instances": self.instances}


@dataclass(frozen=True)
class ReadinessReport:
    """Every primitive gate output `<= w_max` bits?  `max_w` is the widest gate; `specs` the distinct
    specializations visited; `gates_over` the number of root gates wider than `w_max`."""
    w_max: int
    ok: bool
    violators: tuple[ReadinessViolation, ...]
    max_w: int
    specs: int
    gates_over: int

    def to_json(self) -> dict:
        return {"w_max": self.w_max, "ok": self.ok, "violators": [v.to_json() for v in self.violators],
                "max_w": self.max_w, "specs": self.specs, "gates_over": self.gates_over}


@dataclass(frozen=True)
class VuVerdict:
    """`status`: "ok" | "fail" (overlap/gap or a proven width violation) | "not-ready" (a gate wider than `w_max`
    exists; partition and width did not fail) | "unsupported" (nothing failed, something could not be decided)."""
    partition: PartitionVerdict
    width: WidthVerdict
    count: int
    readiness_32: ReadinessReport
    status: str

    def to_json(self) -> dict:
        return {"status": self.status, "count": self.count, "partition": self.partition.to_json(),
                "width": self.width.to_json(), "readiness_32": self.readiness_32.to_json()}


# ---------------------------------------------------------------------------------------------------------
# reading a Family.tiling() (duck-typed: a `query_ast.Tiling`, or anything with its field names)
# ---------------------------------------------------------------------------------------------------------


class _Malformed(Exception):
    """The tiling description cannot be interpreted; the caller falls back to the sweep."""


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
        if isinstance(obj, Mapping) and n in obj:
            return obj[n]
    return default


def _need(obj: Any, *names: str) -> Any:
    v = _attr(obj, *names)
    if v is None:
        raise _Malformed(f"{type(obj).__name__} lacks {names[0]!r}")
    return v


def _body_spec(bt: Any) -> SpecializedDefinition:
    spec = _attr(bt, "spec")
    if not isinstance(spec, SpecializedDefinition):
        raise _Malformed(f"body tiling without a SpecializedDefinition: {bt!r}")
    return spec


def _ranges(entry: Any, n: int) -> tuple[tuple[int, int], ...]:
    """The entry's member ranges, ascending; an absent/empty `members` means the whole node `[0, n)`."""
    raw = _attr(entry, "members")
    if not raw:
        return ((0, n),) if n else ()
    out = tuple((int(a), int(b)) for a, b in raw)
    for a, b in out:
        if not (0 <= a < b <= n):
            raise _Malformed(f"member range [{a}, {b}) outside [0, {n})")
    return tuple(sorted(out))


def _blocks(ranges: tuple[tuple[int, int], ...], k: int) -> Iterator[tuple[int, int]]:
    for a, b in ranges:
        for c in range(a, b, k):
            yield c, min(c + k, b)


Group = tuple[str, ...]
"""Plan keys (sorted) whose entries together tile one body: the root plan alone, or every plan the parent's
`descend` entries on one node send the same members into (`concat(A, rest())` descends twice into one node)."""


@dataclass
class _Class:
    """One member class of a tiling: the sets one entry induces, structurally identical across activations of the
    body.  `rule`: "atomic" (k adjacent members of `ranges` per set), "whole" (one set per range of `ranges`),
    "zip" (member m of every node of `group`, m in [0, n))."""
    key: Group                                  # plan group tiling the body (activation multiplicity is per group)
    plan: str                                   # plan key of the entry
    rule: str
    node: int                                   # node index; first node of a zip group
    n: int = 1                                  # members of the node (zip: the common n)
    k: int = 1                                  # "atomic": members per set
    ranges: tuple[tuple[int, int], ...] = ()    # "atomic" / "whole"
    group: tuple[int, ...] = ()                 # "zip"

    def templates(self) -> Iterator[tuple[frozenset[tuple[int, int]], str]]:
        """(item set, shape) of one activation, one per set; sets of equal `shape` have the same gates up to
        translation (same nodes, same number of members)."""
        if self.rule == "atomic":
            for a, b in _blocks(self.ranges, self.k):
                yield frozenset((self.node, m) for m in range(a, b)), f"len{b - a}"
        elif self.rule == "whole":
            for a, b in self.ranges:
                yield frozenset((self.node, m) for m in range(a, b)), f"len{b - a}"
        else:
            for m in range(self.n):
                yield frozenset((j, m) for j in self.group), "zip"

    def shapes(self) -> dict[str, tuple[int, int]]:
        """shape -> (sets per activation, members per set), without enumerating the templates."""
        out: dict[str, tuple[int, int]] = {}

        def add(length: int, count: int = 1) -> None:
            c, _ = out.get(f"len{length}", (0, length))
            out[f"len{length}"] = (c + count, length)

        if self.rule == "atomic":
            for a, b in self.ranges:
                full, tail = divmod(b - a, self.k)
                if full:
                    add(self.k, full)
                if tail:
                    add(tail)
        elif self.rule == "whole":
            for a, b in self.ranges:
                add(b - a)
        else:
            out["zip"] = (self.n, 1)
        return out

    def sets_per_activation(self) -> int:
        return sum(c for c, _ in self.shapes().values())

    def label(self, spec_id: str) -> dict:
        d: dict = {"spec": spec_id, "plan": self.plan, "rule": self.rule, "node": self.node}
        if len(self.key) > 1:
            d["plans"] = list(self.key)
        if self.rule == "atomic":
            d["k"] = self.k
        if self.rule in ("atomic", "whole"):
            d["ranges"] = [list(r) for r in self.ranges[:8]]
        if self.rule == "zip":
            d["group"] = list(self.group)
            d["n"] = self.n
        return d


@dataclass
class _Plan:
    """One `BodyTiling` as read: its classes, its direct node coverage, its descents, its declared count."""
    key: str
    spec: SpecializedDefinition
    declared_count: int | None
    classes: list[_Class] = field(default_factory=list)
    direct: dict[int, list[tuple[int, int]]] = field(default_factory=dict)         # node -> member ranges
    descends: dict[int, list[tuple[int, int, str]]] = field(default_factory=dict)  # node -> (a, b, child plan)


class CheckedTiling:
    """The interpreted `Family.tiling()` (a `query_ast.Tiling`: `root` plan key, `bodies` plan key ->
    `BodyTiling(spec, entries)`, entries `NodeTiling(node, kind, members, k, group, child)`).

    Induction is per *plan group*: the plans whose entries together tile one body -- the root plan alone at the
    root; below, every plan the parent's `descend` entries send the same members of one node into (a `concat` of
    two structural queries descends twice into the same node, each plan covering part of the callee body).  Per
    group: every node's members `[0, n)` must be covered exactly once, by a direct entry (atomic / whole / zip) or
    by a descent; members sent down are tiled by the child group, recursively.  Activation multiplicity is per
    group.  Coverage problems are collected (`problems`); anything unreadable raises `_Malformed`."""

    def __init__(self, program: "Program", tiling: Any) -> None:
        self.program = program
        bodies = _need(tiling, "bodies")
        root_key = _need(tiling, "root")
        if not isinstance(bodies, Mapping):
            raise _Malformed("Tiling.bodies is not a mapping")
        if root_key not in bodies:
            raise _Malformed(f"Tiling.root {root_key!r} is not in Tiling.bodies")
        self.root_plan = root_key
        self.root: Group = (root_key,)
        self._bodies = bodies
        rspec = _body_spec(bodies[root_key])
        if rspec is not program.root and rspec.body is not program.root.body:
            raise _Malformed(f"tiling root is {rspec.id}, not the program root {program.root.id}")
        self.plans: dict[str, _Plan] = {}                            # every plan read (reachable from the root)
        self.groups: list[Group] = []                                # discovery order
        self.spec: dict[Group, SpecializedDefinition] = {}
        self.children: dict[Group, list[tuple[int, tuple[int, int], Group]]] = {}  # group -> [(j, (a, b), child)]
        self.classes: list[_Class] = []
        self.problems: list[tuple[str, dict]] = []
        self._walk(self.root, 0)
        # activation multiplicity: instances(group) = sum over its parents of instances(parent) x members descended.
        # (Discovery order is not topological for a body shared by two parents, so recurse over parents, memoised.)
        parents: dict[Group, list[tuple[Group, int]]] = {}
        for g in self.groups:
            for _, (a, b), child in self.children[g]:
                parents.setdefault(child, []).append((g, b - a))
        self.instances: dict[Group, int] = {self.root: 1}

        def inst(g: Group) -> int:
            if g not in self.instances:
                self.instances[g] = sum(inst(p) * k for p, k in parents.get(g, ()))
            return self.instances[g]

        for g in self.groups:
            inst(g)

    # -- reading one plan --------------------------------------------------------------------
    def _plan(self, key: str, group: Group) -> _Plan:
        if key in self.plans:
            return self.plans[key]
        bt = self._bodies[key]
        spec = _body_spec(bt)
        body = spec.body
        n_nodes = _attr(bt, "n_nodes")
        if n_nodes is not None and int(n_nodes) != len(body.nodes):
            raise _Malformed(f"{key}: n_nodes {n_nodes} != {len(body.nodes)} nodes of {spec.id}")
        dc = _attr(bt, "count")
        plan = _Plan(key, spec, int(dc) if dc is not None else None)
        zips: dict[Any, list[tuple[int, int, tuple[tuple[int, int], ...]]]] = {}
        for entry in _need(bt, "entries"):
            j = int(_need(entry, "node"))
            if not (0 <= j < len(body.nodes)):
                raise _Malformed(f"{key}: entry for node {j}, body {spec.id} has {len(body.nodes)} nodes")
            node = body.nodes[j]
            n = n_members(node)
            ranges = _ranges(entry, n)
            kind = str(_need(entry, "kind"))
            if kind == "descend":
                child = _need(entry, "child")
                if isinstance(node.fn, PrimitiveDefinition):
                    raise _Malformed(f"{key}: node {j} is a primitive ({node.fn.id}); it cannot be descended")
                if child not in self._bodies:
                    raise _Malformed(f"{key}: node {j} descends into unknown plan key {child!r}")
                cspec = _body_spec(self._bodies[child])
                if cspec is not node.fn and cspec.body is not node.fn.body:
                    raise _Malformed(f"{key}: node {j} calls {node.fn.id} but plan {child!r} tiles {cspec.id}")
                plan.descends.setdefault(j, []).extend((a, b, child) for a, b in ranges)
                continue
            plan.direct.setdefault(j, []).extend(ranges)
            if kind == "atomic":
                k = _attr(entry, "k", default=1)
                if not isinstance(k, int) or k <= 0:
                    raise _Malformed(f"{key}: node {j}: atomic entry needs a positive int k, got {k!r}")
                plan.classes.append(_Class(group, key, "atomic", j, n=n, k=k, ranges=ranges))
            elif kind == "whole":
                plan.classes.append(_Class(group, key, "whole", j, n=n, ranges=ranges))
            elif kind == "zip":
                zips.setdefault(_need(entry, "group"), []).append((j, n, ranges))
            else:
                raise _Malformed(f"{key}: unknown entry kind {kind!r} on node {j}")
        for gkey, members in zips.items():
            ns = sorted({n for _, n, _ in members})
            nodes = tuple(j for j, _, _ in members)
            if len(ns) != 1:
                raise _Malformed(f"{key}: zip group {gkey!r} over nodes {list(nodes)} mixes n = {ns} (tiling-zip-unequal-n)")
            if any(r != ((0, n),) for _, n, r in members if n):
                raise _Malformed(f"{key}: zip group {gkey!r} has partial member ranges")
            plan.classes.append(_Class(group, key, "zip", nodes[0], n=ns[0], group=nodes))
        self.plans[key] = plan
        return plan

    # -- induction over plan groups ----------------------------------------------------------
    def _walk(self, group: Group, depth: int) -> None:
        if group in self.spec:
            return
        if depth > 10_000:
            raise _Malformed("tiling deeper than 10000 bodies (cyclic description?)")
        plans = [self._plan(key, group) for key in group]
        spec = plans[0].spec
        for p in plans[1:]:
            if p.spec is not spec and p.spec.body is not spec.body:
                raise _Malformed(f"plans {list(group)} descend into one node but tile {spec.id} and {p.spec.id}")
        self.groups.append(group)
        self.spec[group] = spec
        self.children[group] = []
        for p in plans:
            self.classes.extend(p.classes)
        where = {"spec": spec.id, "plan": group[0] if len(group) == 1 else list(group)}
        for j, node in enumerate(spec.body.nodes):
            if node.gates == 0:
                continue
            n = n_members(node)
            # events on the member axis: direct ranges count coverage; descents name the child plans
            events: list[tuple[int, int, str | None]] = []
            for p in plans:
                for a, b in p.direct.get(j, ()):
                    events.append((a, +1, None))
                    events.append((b, -1, None))
                for a, b, child in p.descends.get(j, ()):
                    events.append((a, +1, child))
                    events.append((b, -1, child))
            if not events:
                self.problems.append(("tiling-unassigned", {**where, "node": j, "fn": node.fn.id}))
                continue
            events.sort(key=lambda e: e[0])
            positions = sorted({0, n} | {e[0] for e in events})
            direct = 0
            below: dict[str, int] = {}
            i = 0
            for a, b in zip(positions, positions[1:]):
                while i < len(events) and events[i][0] == a:
                    _, delta, child = events[i]
                    i += 1
                    if child is None:
                        direct += delta
                    else:
                        below[child] = below.get(child, 0) + delta
                        if below[child] == 0:
                            del below[child]
                if direct == 0 and not below:
                    self.problems.append(("tiling-members-gap", {**where, "node": j, "members": [a, b]}))
                    continue
                if direct > 1 or (direct == 1 and below) or any(c > 1 for c in below.values()):
                    self.problems.append(("tiling-members-overlap", {**where, "node": j, "at": a}))
                if below:  # descend even under an overlap: the child group is read and its problems reported
                    self._descend(group, j, (a, b), tuple(sorted(below)), depth)

    def _descend(self, group: Group, j: int, members: tuple[int, int], child: Group, depth: int) -> None:
        self.children[group].append((j, members, child))
        self._walk(child, depth + 1)

    # -- structural facts ------------------------------------------------------------------
    def count_per_activation(self, key: str, memo: dict[str, int] | None = None) -> int:
        """Sets one activation of plan `key` contributes: its own classes plus, per descent, members x the child
        plan's count -- the Family's own accounting (`BodyTiling.count`), independent of grouping."""
        memo = {} if memo is None else memo
        if key in memo:
            return memo[key]
        memo[key] = -1  # in progress: a plan counting itself is a cyclic description
        plan = self.plans.get(key) or self._plan(key, (key,))  # a plan reached only over 0 members is still counted
        total = sum(cl.sets_per_activation() for cl in plan.classes)
        for descents in plan.descends.values():
            for a, b, child in descents:
                sub = self.count_per_activation(child, memo)
                if sub < 0:
                    raise _Malformed(f"plan {child!r} descends into itself (cyclic description)")
                total += (b - a) * sub
        memo[key] = total
        return total

    def count(self) -> int:
        """Number of sets the tiling induces over the whole Program (no gate visits)."""
        return self.count_per_activation(self.root_plan)

    def declared_count_mismatch(self) -> dict | None:
        """The first plan whose declared `BodyTiling.count` disagrees with the structural count read here."""
        memo: dict[str, int] = {}
        for key, plan in self.plans.items():
            if plan.declared_count is not None and plan.declared_count != self.count_per_activation(key, memo):
                return {"plan": key, "declared": plan.declared_count, "structural": memo[key]}
        return None

    def scopes_of(self, target: Group) -> Iterator[Scope]:
        """Every activation Scope of the body tiled by `target`, depth-first from the root (lazy)."""
        reach: dict[Group, bool] = {}

        def reaches(g: Group) -> bool:
            if g not in reach:
                reach[g] = g == target or any(reaches(ch) for _, _, ch in self.children[g])
            return reach[g]

        def walk(sc: Scope, g: Group) -> Iterator[Scope]:
            if g == target:
                yield sc
                return
            for j, (a, b), child in self.children[g]:
                if reaches(child):
                    for m in range(a, b):
                        yield from walk(sc.child(j, m), child)

        yield from walk(self.program.circuit.root_scope, self.root)


# ---------------------------------------------------------------------------------------------------------


def family_tiling(fam: Any) -> Any:
    """`fam.tiling()` (or `fam.tiling`), or None when the family has no tiling or cannot produce one."""
    t = getattr(fam, "tiling", None)
    if t is None:
        return None
    try:
        return t() if callable(t) else t
    except Exception:  # noqa: BLE001 -- a family that cannot describe itself is swept
        return None


def _family_count(fam: Any) -> int | None:
    c = getattr(fam, "count", None)
    if c is None:
        return None
    try:
        return int(c() if callable(c) else c)
    except Exception:  # noqa: BLE001
        return None


def validate_partition(program: "Program", fam, limits: Limits = Limits(), *, within: Any = None) -> PartitionVerdict:
    """Is `fam` an exact partition of every gate of `program` (Input and constant gates included)?

    Fast path from `fam.tiling()` by per-body induction; fallback interval sweep over materialised intervals up to
    `limits.max_intervals`; beyond that "unsupported" (never a verdict).  See the module docstring.

    `within` (a `GateSet`, or `(lo, hi)` root gate intervals) asks instead whether `fam` partitions exactly those
    gates: each of them in one set, and no set reaching outside them ("outside", code `sweep-outside`).  It is how a
    level leaves out the Inputs checked outside the draws.  This mode always sweeps."""
    if within is not None:
        return _sweep(program, fam, limits, _gate_intervals(within, program.gates))
    tiling = family_tiling(fam)
    fallback_codes: list[str] = []
    fallback_detail: dict = {}
    if tiling is not None:
        try:
            t = CheckedTiling(program, tiling)
        except _Malformed as e:
            fallback_codes.append("tiling-malformed")
            fallback_detail["tiling_malformed"] = str(e)
        else:
            mismatch = _count_mismatch(t, fam)
            if mismatch is not None:
                fallback_codes.append("tiling-count-mismatch")
                fallback_detail["tiling_count"] = mismatch
            else:
                return _tiling_verdict(program, t, t.count())
    v = _sweep(program, fam, limits)
    if fallback_codes:
        v = PartitionVerdict(v.status, tuple(fallback_codes) + v.codes, {**fallback_detail, **v.detail})
    return v


def _count_mismatch(t: CheckedTiling, fam: Any) -> dict | None:
    """The tiling must describe the family: the structural count read here equals `Family.count()` and every
    `BodyTiling.count` the family declared.  A disagreement means the description is not of these sets."""
    per_body = t.declared_count_mismatch()
    if per_body is not None:
        return per_body
    structural, declared = t.count(), _family_count(fam)
    if declared is not None and declared != structural:
        return {"structural": structural, "family": declared}
    return None


def _tiling_verdict(program: "Program", t: CheckedTiling, count: int) -> PartitionVerdict:
    codes = tuple(sorted({code for code, _ in t.problems}))
    if "tiling-members-overlap" in codes:
        status: PartitionStatus = "overlap"
    elif codes:
        status = "gap"
    else:
        status = "exact"
    detail: dict = {"path": "tiling", "gates": program.gates, "sets": count, "bodies": len(t.groups),
                    "plans": len(t.plans), "classes": len(t.classes)}
    if t.problems:
        detail["problems"] = [{"code": c, **d} for c, d in t.problems[:_MAX_LISTED]]
        detail["problem_count"] = len(t.problems)
    return PartitionVerdict(status, codes, detail)


def _gate_intervals(gates: Any, size: int) -> Intervals:
    raw = gates.intervals() if hasattr(gates, "intervals") else gates
    ivs = norm((int(lo), int(hi)) for lo, hi in raw)
    if ivs and (ivs[0][0] < 0 or ivs[-1][1] > size):
        raise ValueError(f"gate intervals {ivs[0]} .. {ivs[-1]} outside the program's {size} gates")
    return ivs


def _sweep(program: "Program", fam, limits: Limits, within: Intervals | None = None) -> PartitionVerdict:
    size = program.gates
    target = within if within is not None else ((0, size),) if size else ()
    ivs: list[tuple[int, int, int]] = []
    n_sets = 0
    empty: list[int] = []
    for i, s in enumerate(fam):
        n_sets += 1
        sivs = s.intervals()
        if not sivs:
            empty.append(i)
        for lo, hi in sivs:
            if not (0 <= lo <= hi <= size):
                raise ValueError(f"set {i}: interval [{lo}, {hi}) outside the program's {size} gates")
            if hi > lo:
                ivs.append((lo, hi, i))
        if len(ivs) > limits.max_intervals:
            return PartitionVerdict("unsupported", ("limit-intervals",),
                                    {"path": "sweep", "gates": size, "intervals_seen": len(ivs),
                                     "sets_seen": n_sets, "max_intervals": limits.max_intervals})
    ivs.sort()
    expected = 0
    prev_set = -1
    overlaps: list[dict] = []
    covered: list[tuple[int, int]] = []
    n_over = 0
    for lo, hi, i in ivs:
        if lo < expected:
            n_over += 1
            if len(overlaps) < _MAX_LISTED:
                overlaps.append({"gates": [lo, min(hi, expected)], "sets": [prev_set, i]})
        if covered and lo <= covered[-1][1]:
            covered[-1] = (covered[-1][0], max(covered[-1][1], hi))
        else:
            covered.append((lo, hi))
        if hi >= expected:
            expected, prev_set = hi, i
    union = tuple(covered)
    missing = isect(target, tuple(complement(union, size)))
    outside = isect(union, tuple(complement(target, size))) if within is not None else ()
    n_gap, n_out = len(missing), len(outside)
    gaps = [{"gates": [lo, hi]} for lo, hi in missing[:_MAX_LISTED]]
    codes: list[str] = []
    if n_over:
        codes.append("sweep-overlap")
    if n_out:
        codes.append("sweep-outside")
    if n_gap:
        codes.append("sweep-gap")
    if empty:
        codes.append("empty-set")
    status: PartitionStatus = "overlap" if n_over else "outside" if n_out else "gap" if n_gap else "exact"
    detail: dict = {"path": "sweep", "gates": size, "sets": n_sets, "intervals": len(ivs)}
    if within is not None:
        detail["within"] = sum(hi - lo for lo, hi in target)
    if n_over:
        detail["overlaps"] = overlaps
        detail["overlap_count"] = n_over
    if n_out:
        detail["outside"] = [{"gates": [lo, hi]} for lo, hi in outside[:_MAX_LISTED]]
        detail["outside_count"] = n_out
    if n_gap:
        detail["gaps"] = gaps
        detail["gap_count"] = n_gap
    if empty:
        detail["empty_sets"] = empty[:_MAX_LISTED]
        detail["empty_count"] = len(empty)
    return PartitionVerdict(status, tuple(codes), detail)


# ---------------------------------------------------------------------------------------------------------
# width
# ---------------------------------------------------------------------------------------------------------

_BOUNDARY: Any = None  # tests inject a stub here (an object with `w_out_bound(part)` and `w_out(part)`)


def _boundary() -> Any:
    return boundary if _BOUNDARY is None else _BOUNDARY


def _family_index(fam: Any, part: Part) -> int | None:
    """The family index of the set `{part}`, when the family can resolve it."""
    try:
        return fam.index_of(GateSet((part,)))
    except Exception:  # noqa: BLE001
        return None


class _WidthRun:
    """Accumulates one `validate_width` call: certified / violating / unsupported sets, the exact budget."""

    def __init__(self, w_max: int, limits: Limits) -> None:
        self.w_max, self.limits = w_max, limits
        self.exact_budget = limits.max_members_exact
        self.max_seen: int | None = None
        self.violators: list[int] = []
        self.unsupported: list[int] = []
        self.n_violators = self.n_unsupported = 0
        self.bound_certified = self.exact_certified = self.exact_evaluated = 0
        self.codes: set[str] = set()
        self.examples: list[dict] = []
        self.widths: dict[int, tuple[Any, int]] = {}   # `total_width` memo for this run

    def total_width(self, fn: Any) -> int:
        return total_width(fn, self.widths)

    def _seen(self, v: int) -> None:
        self.max_seen = v if self.max_seen is None else max(self.max_seen, v)

    def _record(self, kind: str, index: int | None, label: dict, value: Any) -> None:
        if kind == "violation":
            self.n_violators += 1
            if index is not None and len(self.violators) < _MAX_LISTED and index not in self.violators:
                self.violators.append(index)
        else:
            self.n_unsupported += 1
            if index is not None and len(self.unsupported) < _MAX_LISTED and index not in self.unsupported:
                self.unsupported.append(index)
        if len(self.examples) < _MAX_LISTED:
            self.examples.append({"kind": kind, "index": index, **label, "value": value})

    def bound(self, parts: tuple[Part, ...]) -> int:
        B = _boundary()
        return sum(int(B.w_out_bound(p)) for p in parts)

    def exact(self, parts: tuple[Part, ...]) -> int | Unsupported:
        """Exact `w_out` of one part; for several parts the sum of their exact values, which is an upper bound on
        the union's (`out(A|B) <= out(A) | out(B)`) -- it certifies, it never refutes."""
        B = _boundary()
        total = 0
        for p in parts:
            v = B.w_out(p)
            if isinstance(v, Unsupported) or v is None:
                return v if isinstance(v, Unsupported) else Unsupported("boundary-none", "w_out returned None")
            total += int(v)
        return total

    def decide(self, parts: tuple[Part, ...], index: int | None, label: dict, bound: int | None = None) -> None:
        """Certify / refute / give up on one set: structural `sum w(g)` first, then the boundary bound, then exactly
        (budget permitting)."""
        if bound is None:
            bound = sum(self.total_width(p.spec.body.nodes[j].fn) for p in parts for j, _ in p.items)
            if bound > self.w_max:
                bound = self.bound(parts)
        if bound <= self.w_max:
            self.bound_certified += 1
            self._seen(bound)
            return
        if self.exact_budget <= 0:
            self.codes.add("limit-members-exact")
            self._record("unsupported", index, label, {"bound": bound, "reason": "limit-members-exact"})
            return
        self.exact_budget -= 1
        self.exact_evaluated += 1
        v = self.exact(parts)
        if isinstance(v, Unsupported):
            self.codes.add("boundary-unsupported")
            self._record("unsupported", index, label, {"bound": bound, "reason": v.reason, "detail": v.detail})
            return
        if v <= self.w_max:
            self.exact_certified += 1
            self._seen(v)
            return
        if len(parts) > 1:
            self.codes.add("multi-part-exact")
            self._record("unsupported", index, label, {"bound": bound, "sum_exact": v, "reason": "multi-part-exact"})
            return
        self._seen(v)
        self._record("violation", index, label, {"w_out": v})

    def verdict(self, path: str, extra: dict) -> WidthVerdict:
        status = "violation" if self.n_violators else "unsupported" if self.n_unsupported else "ok"
        detail = {"path": path, "bound_certified": self.bound_certified, "exact_evaluated": self.exact_evaluated,
                  "exact_certified": self.exact_certified, "violation_count": self.n_violators,
                  "unsupported_count": self.n_unsupported, "codes": sorted(self.codes), **extra}
        if self.examples:
            detail["examples"] = self.examples
        return WidthVerdict(self.w_max, self.max_seen, tuple(self.violators), tuple(self.unsupported), status, detail)


def validate_width(program: "Program", fam, w_max: int, limits: Limits = Limits()) -> WidthVerdict:
    """`w_out(S) <= w_max` for every set of `fam`?  Per member class `boundary.w_out_bound <= w_max` certifies;
    otherwise exact `boundary.w_out` per set within `limits.max_members_exact` exact evaluations.  A bound above
    `w_max` never proves a violation; `Unsupported` is reported separately, never counted as pass or fail."""
    run = _WidthRun(w_max, limits)
    tiling = family_tiling(fam)
    if tiling is not None:
        try:
            t = CheckedTiling(program, tiling)
        except _Malformed:
            t = None
        if t is not None and not t.problems and _count_mismatch(t, fam) is None:
            return _width_by_classes(run, t, fam)
    return _width_by_sets(run, fam, limits)


def total_width(fn: Any, memo: dict[int, tuple[Any, int]] | None = None) -> int:
    """Sum of `w(g)` over every gate of one activation of `fn` (a primitive: its `ret.w`).
    `w_out(S) <= sum_{g in S} w(g)` for any set S, so this is the cheapest sound bound on a member's `w_out`.
    `memo` (per validation run) is keyed by identity and holds the definition, so no id can be reused."""
    memo = {} if memo is None else memo
    hit = memo.get(id(fn))
    if hit is not None and hit[0] is fn:
        return hit[1]
    if isinstance(fn, PrimitiveDefinition):
        w = fn.ret.w
    else:
        w = sum(n_members(node) * total_width(node.fn, memo) for node in fn.body.nodes)
    memo[id(fn)] = (fn, w)
    return w


def _width_by_classes(run: _WidthRun, t: CheckedTiling, fam: Any) -> WidthVerdict:
    """One representative instance per class.  Three tiers per class:

    1. structural: `sum w(g)` over one set of each shape (`_Class.shapes`) -- if `<= w_max`, every set of that shape
       in every instance is certified without a boundary call (this is what makes a 10^8-member Input batch cheap);
    2. `boundary.w_out_bound` per template on the representative instance: the bound is intrinsic to (spec, items)
       -- it charges every returned leaf whatever the parent does -- so one instance certifies all;
    3. exact `boundary.w_out` per set over ALL instances of the templates still pending (the exact returned-leaf term
       depends on the parent's consumers), within the exact budget.
    Boundary-bound evaluations per class are capped at `limits.max_intervals`; the rest are unsupported."""
    n_sets = 0
    classes_struct = classes_bound = classes_exact = 0
    struct_certified = 0
    for cl in t.classes:
        insts = t.instances.get(cl.key, 0)
        spec = t.spec[cl.key]
        body = spec.body
        shapes = cl.shapes()
        n_sets += insts * cl.sets_per_activation()
        if insts == 0:
            continue
        # tier 1: structural width per shape
        pending_shapes: set[str] = set()
        for shape, (count, members) in shapes.items():
            if cl.rule == "zip":
                w = sum(run.total_width(body.nodes[j].fn) for j in cl.group)
            else:
                w = members * run.total_width(body.nodes[cl.node].fn)
            if w <= run.w_max:
                struct_certified += count * insts
                run.bound_certified += count * insts
                run._seen(w)
            else:
                pending_shapes.add(shape)
        if not pending_shapes:
            classes_struct += 1
            continue
        # tier 2: boundary bound on the representative
        rep = next(iter(t.scopes_of(cl.key)))
        pending: list[tuple[frozenset, int]] = []
        evaluated = 0
        for items, shape in cl.templates():
            if shape not in pending_shapes:
                continue
            if evaluated >= run.limits.max_intervals:
                run.codes.add("limit-members-bound")
                run.n_unsupported += insts
                continue
            evaluated += 1
            b = run.bound((Part(rep, items),))
            if b <= run.w_max:
                run.bound_certified += insts
                run._seen(b)
            else:
                pending.append((items, b))
        if not pending:
            classes_bound += 1
            continue
        # tier 3: exact, every instance
        classes_exact += 1
        for sc in t.scopes_of(cl.key):
            for items, b in pending:
                part = Part(sc, items)
                label = {**cl.label(spec.id), "instance": sc.path(), "members": sorted(set(m for _, m in items))[:8]}
                if run.exact_budget <= 0:
                    run.codes.add("limit-members-exact")
                    run._record("unsupported", None, label, {"bound": b, "reason": "limit-members-exact"})
                    continue
                run.decide((part,), _family_index(fam, part), label, bound=b)
    return run.verdict("tiling", {"sets": n_sets, "classes": len(t.classes), "struct_certified": struct_certified,
                                  "classes_struct_certified": classes_struct, "classes_bound_certified": classes_bound,
                                  "classes_exact": classes_exact})


def _width_by_sets(run: _WidthRun, fam: Any, limits: Limits) -> WidthVerdict:
    """No usable tiling: every set is its own class.  Iterates at most `limits.max_intervals` sets (each set is at
    least one interval, the same budget the sweep spends); the rest are unsupported."""
    n_sets = 0
    for i, s in enumerate(fam):
        if i >= limits.max_intervals:
            run.codes.add("limit-sets")
            declared = _family_count(fam)
            rest = (declared - i) if declared is not None else 1
            run.n_unsupported += rest
            break
        n_sets += 1
        parts = tuple(getattr(s, "parts", ()))
        if not parts:
            run.codes.add("boundary-unsupported")
            run._record("unsupported", i, {"set": i}, {"reason": "set without Parts (no boundary analysis possible)"})
            continue
        label = {"set": i, "spec": parts[0].spec.id, "parts": len(parts)}
        run.decide(parts, i, label)
    return run.verdict("sets", {"sets": n_sets})


# ---------------------------------------------------------------------------------------------------------
# readiness: every primitive gate <= w_max bits
# ---------------------------------------------------------------------------------------------------------


def gate_width_readiness(program: "Program", w_max: int) -> ReadinessReport:
    """Every primitive gate output of `program` is `<= w_max` bits?  Structural walk over the distinct
    specializations reachable from the root body (memoised by identity), instance counts by multiplicity."""
    specs: list[SpecializedDefinition] = []
    parents: dict[int, list[tuple[int, int]]] = {}  # id(spec) -> [(id(parent), multiplicity)]
    seen: set[int] = set()

    def visit(spec: SpecializedDefinition) -> None:
        if id(spec) in seen:
            return
        seen.add(id(spec))
        specs.append(spec)
        for node in spec.body.nodes:
            if not isinstance(node.fn, PrimitiveDefinition):
                parents.setdefault(id(node.fn), []).append((id(spec), n_members(node)))
                visit(node.fn)

    visit(program.root)
    inst: dict[int, int] = {id(program.root): 1}

    def instances(sid: int) -> int:
        if sid not in inst:
            inst[sid] = sum(instances(p) * k for p, k in parents.get(sid, ()))
        return inst[sid]

    violators: list[ReadinessViolation] = []
    max_w = 0
    gates_over = 0
    for spec in specs:
        for j, node in enumerate(spec.body.nodes):
            if isinstance(node.fn, PrimitiveDefinition):
                w = node.fn.ret.w
                max_w = max(max_w, w)
                if w > w_max:
                    k = n_members(node)
                    n_inst = instances(id(spec))
                    violators.append(ReadinessViolation(spec.id, j, node.fn.id, w, node.form, k, n_inst))
                    gates_over += k * n_inst
    return ReadinessReport(w_max, not violators, tuple(violators), max_w, len(specs), gates_over)


def readiness_32bit(program: "Program") -> ReadinessReport:
    """Every primitive gate output `<= 32` bits?  Violators by specialization / primitive (`Const64` in the Gumbel
    body, `Input64`).  Separate from the boundary condition; 32 is the protocol's `w_max`."""
    return gate_width_readiness(program, 32)


# ---------------------------------------------------------------------------------------------------------
# hierarchies: levels, refinement, members per parent
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Level:
    """One level of a partition hierarchy.  `name` is its owner's label ("replay", "check", "matmul", ...), never
    interpreted here; `query` is the id (`query_ast.query_id`) of the query whose family is this level's members, or
    None for a level not bound to a Program."""
    name: str
    query: str | None = None


Hierarchy = tuple[Level, ...]
"""Levels coarse to fine; each level's members refine the level above (`validate_refinement`)."""


@dataclass(frozen=True)
class RefinementVerdict:
    """Does every set of a finer family lie inside exactly one set of a coarser one?  `status`: "ok", "violation"
    (codes: `refine-crosses`, a fine set meets two coarse sets; `refine-outside`, a fine set reaches gates in no
    coarse set; `refine-empty`, a fine set with no gates; `refine-coarse-overlap`, the coarse sets overlap), or
    "unsupported" (`limit-intervals`; not a verdict).  `counts[i]` is the number of fine sets inside coarse set i,
    in family order, when the status is "ok" (empty otherwise)."""
    status: str
    codes: tuple[str, ...]
    counts: tuple[int, ...]
    detail: dict

    def to_json(self) -> dict:
        return {"status": self.status, "codes": list(self.codes), "counts": list(self.counts), "detail": self.detail}


def validate_refinement(program: "Program", coarse, fine, limits: Limits = Limits()) -> RefinementVerdict:
    """Is `fine` a refinement of `coarse`: every set of `fine` inside exactly one set of `coarse`?  Containment
    only; whether each level covers its gates is `validate_partition`'s question.  Families built with
    `each(F, G)` nest by construction; others need this check.  Interval sweep over both families' materialised
    intervals, up to `limits.max_intervals` in total."""
    size = program.gates
    runs: list[tuple[int, int, int]] = []
    n_coarse = 0
    for c, s in enumerate(coarse):
        n_coarse += 1
        runs.extend((lo, hi, c) for lo, hi in _gate_intervals(s, size))
        if len(runs) > limits.max_intervals:
            return _refinement_unsupported(len(runs), limits)
    runs.sort()
    overlaps = [(runs[t - 1][2], runs[t][2]) for t in range(1, len(runs)) if runs[t][0] < runs[t - 1][1]]
    if overlaps:
        return RefinementVerdict("violation", ("refine-coarse-overlap",), (),
                                 {"coarse_sets": n_coarse, "overlap_count": len(overlaps),
                                  "overlaps": [list(o) for o in overlaps[:_MAX_LISTED]]})
    starts = [lo for lo, _, _ in runs]
    counts = [0] * n_coarse
    problems: dict[str, list[int]] = {}
    n_fine = seen = 0
    for f, s in enumerate(fine):
        n_fine += 1
        ivs = _gate_intervals(s, size)
        seen += len(ivs)
        if seen + len(runs) > limits.max_intervals:
            return _refinement_unsupported(seen + len(runs), limits)
        parents: set[int] = set()
        outside = False
        for lo, hi in ivs:
            t = max(bisect.bisect_right(starts, lo) - 1, 0)
            pos = lo
            while pos < hi and t < len(runs):
                rlo, rhi, c = runs[t]
                if rhi <= pos:
                    t += 1
                    continue
                if rlo > pos:
                    outside = True
                    pos = min(rlo, hi)
                    continue
                parents.add(c)
                pos = min(rhi, hi)
                t += 1
            outside |= pos < hi
        code = ("refine-empty" if not ivs else "refine-outside" if outside else
                "refine-crosses" if len(parents) > 1 else None)
        if code is None:
            counts[parents.pop()] += 1
        else:
            problems.setdefault(code, []).append(f)
    detail: dict = {"coarse_sets": n_coarse, "fine_sets": n_fine}
    if problems:
        for code, sets in problems.items():
            detail[code] = {"count": len(sets), "sets": sets[:_MAX_LISTED]}
        return RefinementVerdict("violation", tuple(sorted(problems)), (), detail)
    return RefinementVerdict("ok", (), tuple(counts), detail)


def _refinement_unsupported(seen: int, limits: Limits) -> RefinementVerdict:
    return RefinementVerdict("unsupported", ("limit-intervals",), (),
                             {"intervals_seen": seen, "max_intervals": limits.max_intervals})


def members_per_parent(program: "Program", coarse, fine, limits: Limits = Limits()) -> tuple[int, ...]:
    """The number of `fine` sets inside each `coarse` set, in `coarse`'s order; `ValueError` unless `fine` refines
    `coarse` (`validate_refinement`)."""
    v = validate_refinement(program, coarse, fine, limits)
    if v.status != "ok":
        raise ValueError(f"not a refinement ({v.status}: {', '.join(v.codes)})")
    return v.counts


# ---------------------------------------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CutVerdict:
    """`validate_unit_cut`'s verdict: `ok`, the failing `codes` and, per code, the count and at most 64 offending gate ids (or
    (producer, consumer) gate pairs for a read)."""
    ok: bool
    codes: tuple[str, ...]
    detail: dict

    def to_json(self) -> dict:
        return {"ok": self.ok, "codes": list(self.codes), "detail": self.detail}


#: the unit holding every input gate (weights, request inputs, randomness): it certifies them by checking each input commitment root
#: against its external anchor (the declared model's weights root, the client's request commitment, the randomness source), always,
#: never sampled
INPUT_UNIT = "input"


#: what `validate_unit_cut` does with a value computed in two units: refuse it (`gate-recomputed`, `Q_word` v1 and every caller's
#: default) or report it (`detail["recomputed_across"]`, `Q_word` v2 only)
RECOMPUTE_RULES = ("refuse", "report")


def validate_unit_cut(total_gates: int, structure_gates: int, owner: Any, reads: Any, outputs: Any, committed: Any,
                      input_gates: Any = (), recomputed: Any = (), recompute: str = "refuse") -> CutVerdict:
    """The partition invariant on one circuit: every gate, input gates included, is certified by exactly one unit.  Input gates form
    the input unit (`INPUT_UNIT`); every computed gate is computed in exactly one computing unit, whose proof being valid implies the
    gate is the correct function of the committed values the unit reads.

    `owner[g]` is the unit of gate `g`: `INPUT_UNIT` for an input gate, a computing unit index >= 0 otherwise.  Structure is an
    operation's own constant bits and wiring: gates any unit may read, bound outside the partition and outside the committed set,
    computed in no unit: `structure_gates` of the circuit's `total_gates` (input gates included) are such.  A program's data is no
    structure: it enters as input gates, registered (bound at registration) or run (bound per request), in the input unit.  `reads` are (producer, consumer) pairs of gates, `outputs` the returned gates,
    `committed` the gates whose values are committed once and reused by every reader (the input gates' values under their roots),
    `recomputed` the (gate, earlier gate) pairs that compute the same function of the same values (a second computation of one value).
    A value computed again in the unit that already computes it crosses no boundary: it is a redundant gate of that unit's statement
    (`detail["redundant_gates"]`), not a recompute.  With `recompute="report"` a value computed in two units is no failure: each copy
    is still its own gate, certified once, and the pairs are reported (`detail["recomputed_across"]`, its count and first pairs).

    * every input gate is in the input unit                                          -> "input-outside-input-unit"  (gate ids)
    * every computed gate is computed in exactly one computing unit                  -> "gate-not-certified-once"   (gate ids)
    * every value is computed in exactly one unit (no recompute; "refuse" only)      -> "gate-recomputed"          ((gate, earlier))
    * the units' gates and the structure sum to the circuit                          -> "gate-count"
    * a cross-unit read goes only through a committed value                           -> "read-uncommitted"          ((producer, consumer))
    * a returned value is committed                                                   -> "output-uncommitted"        (gate ids)
    * a committed value is one a unit certifies and another unit reads, or returned   -> "committed-unread"          (gate ids)
    """
    if recompute not in RECOMPUTE_RULES:
        raise ValueError(f"recompute is one of {RECOMPUTE_RULES}, not {recompute!r}")
    owner = list(owner)
    ins = set(input_gates)
    c = set(committed)
    n = len(owner)
    outside = sorted(g for g in ins if not 0 <= g < n or owner[g] != INPUT_UNIT)
    uncertified = [g for g, u in enumerate(owner) if g not in ins and not (isinstance(u, int) and u >= 0)]
    crossing = sorted((p, q) for p, q in reads if owner[p] != owner[q])
    outs = set(outputs)
    boundary = {p for p, _q in crossing} | outs
    certified = sum(1 for g, u in enumerate(owner) if g not in ins and isinstance(u, int) and u >= 0)
    again = [(int(a), int(b)) for a, b in recomputed]
    across = sorted((a, b) for a, b in again if not (0 <= a < n and 0 <= b < n) or owner[a] != owner[b])
    problems = {"input-outside-input-unit": outside, "gate-not-certified-once": uncertified,
                "gate-recomputed": across if recompute == "refuse" else [],
                "gate-count": [] if certified + len(ins) + int(structure_gates) == int(total_gates) else [certified],
                "read-uncommitted": [(p, q) for p, q in crossing if p not in c], "output-uncommitted": sorted(outs - c),
                "committed-unread": sorted(g for g in c if g not in boundary or not 0 <= g < n)}
    detail: dict = {"input_gates": len(ins), "computed_gates": n - len(ins), "certified": certified, "structure_gates": int(structure_gates),
                    "total_gates": int(total_gates), "committed": len(c), "cross_unit_reads": len(crossing),
                    "computing_units": len({u for u in owner if isinstance(u, int) and u >= 0}),
                    "redundant_gates": len(again) - len(across)}
    if recompute == "report":
        detail["recomputed_across"] = {"n": len(across), "gates": [list(x) for x in across[:_MAX_LISTED]]}
    codes = tuple(k for k, v in problems.items() if v)
    for k in codes:
        detail[k] = {"n": len(problems[k]), "gates": [list(x) if isinstance(x, tuple) else x for x in problems[k][:_MAX_LISTED]]}
    return CutVerdict(not codes, codes, detail)


def vu_status(partition: PartitionVerdict, width: WidthVerdict, readiness: ReadinessReport) -> str:
    if partition.status in ("overlap", "outside", "gap") or width.status == "violation":
        return "fail"
    if not readiness.ok:
        return "not-ready"
    if partition.status == "unsupported" or width.status == "unsupported":
        return "unsupported"
    return "ok"


def validate_vu_family(program: "Program", fam, w_max: int = 32, limits: Limits = Limits()) -> VuVerdict:
    """Both validators and the readiness report over an already evaluated family."""
    partition = validate_partition(program, fam, limits)
    width = validate_width(program, fam, w_max, limits)
    readiness = gate_width_readiness(program, w_max)
    count = _family_count(fam)
    if count is None:
        count = sum(1 for _ in fam)
    return VuVerdict(partition, width, count, readiness, vu_status(partition, width, readiness))


def validate_vu_partition(program: "Program", query, w_max: int = 32) -> VuVerdict:
    """evaluate -> validate_partition + validate_width + readiness -> VuVerdict.  `query` is a `query_ast.Query`
    (or anything `query_ast.evaluate` accepts)."""
    fam = evaluate(program, query)
    return validate_vu_family(program, fam, w_max)


__all__ = ["PartitionVerdict", "WidthVerdict", "ReadinessViolation", "ReadinessReport", "VuVerdict",
           "validate_partition", "validate_width", "gate_width_readiness", "readiness_32bit", "vu_status",
           "validate_vu_family", "validate_vu_partition", "CutVerdict", "INPUT_UNIT", "RECOMPUTE_RULES", "validate_unit_cut",
           "Level", "Hierarchy", "RefinementVerdict", "validate_refinement", "members_per_parent"]
