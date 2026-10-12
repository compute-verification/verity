"""verity_vllm/check/replay/vu_query.py — the local verification units (VUs) of a folded Program as an enumerable population.

Gate G5 (D82) evaluates *local verification units*: a root instance of the flat Program when its argument leaves fit
`max_arg_leaves`, else a child unit reached by descending through the call hierarchy (`Replayer.plan_child` /
`Replayer._descend`: one `GemmCoordinate`, one gather gate, ...).  `replay.tier_a` draws its sample with a random
descent (shuffled node order, one random member per batch node).  This module is the DETERMINISTIC, EXHAUSTIVE view
of the same population, so that the sampler can be conformance-tested on small fully expanded Programs and so that a
production protocol can sample with a distribution the security analysis actually assumes.

    from verity_vllm.query import vu_query as vq
    n = vq.vu_count(result, max_arg_leaves=4_000_000)
    vu = vq.vu_by_index(result, 12345, max_arg_leaves=4_000_000)
    vq.evaluate_vu(result, store, weights, vu)                    # bit-exact against the parent's snapshot
    vq.production_sample(result, seed=7, n=100, max_arg_leaves=4_000_000)   # uniform over VUs, with replacement

Admissibility (identical to `Replayer._descend`, member by member).  Starting from a root instance `k` whose argument
leaves exceed the budget, a node `j` of the enclosing body, member `m` (one member for `primitive`/`call` nodes,
`node.n` for `batch`/`scan`), is examined only if ALL its return gates are among the root instance's return gates
(`rix`).  If its declared argument leaves fit the budget it is a VU iff every one of its argument gates lies outside
the root instance's gate range (it reads only the instance's own operands); a fitting node that reads an interior
value yields nothing and is not descended into.  If it does not fit, it is descended into when it is a composite and
`depth > 0` (`depth` is decremented per level; `depth=3` allows paths of length <= 4).  A primitive that does not
fit, or a node whose returns are interior, yields nothing.

Canonical order.  Root instances in flat-body node order (`result.body.nodes`, i.e. log order); within an oversize
instance, depth-first: body nodes in body order, batch/scan members ascending, and for a member that is descended
into, its own sub-population at that position.  `enumerate_vus` yields exactly this order; `vu_by_index(i)` is the
i-th element of it and `vu_count` its length, computed from per-specialization plans (the plan of a body is
offset-invariant, so instances of one specialization share it) and per-instance prefix sums — never by expansion.

Structural evaluation.  The two gate-level tests of `_descend` are decided from the reference structure: an argument
gate is outside the instance iff its reference chases through `p`-space references up to the instance's own
parameters (an `n`-space reference anywhere on the way is an interior value); a member's returns are among the
enclosing scope's returns iff the enclosing body's `ret` references every output leaf of that member (leaf-level
containment, sufficient since every scope on a descent path has all of its returns admitted).  When leaf-level
containment fails and the specialization's return structure could alias (a `p`-space or repeated reference in some
descendant body's `ret`, or a `scan` node), the module falls back to the literal gate-level test on a
representative activation, so the result equals `_descend`'s in every case.  Hulls of strided references are used
to classify whole arguments without walking their leaves (a 28M-leaf Embedding table is never enumerated).
Torch-free; the only heavy objects it touches are the ones `Replayer` already builds for tier A.
"""
from __future__ import annotations

import bisect
import time
import weakref
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.evaluate import evaluate_call
from verity.ml.circuits.layout import FunctionCall, Scope, resolve
from verity.ml.circuits.refs import Affine, Concat, Refs, Strided, runs

from verity_vllm.check.fold_compare import SnapshotStore, WeightSource
from verity_vllm.observe.fold.fold import FoldResult
from verity_vllm.check.replay.replay import Replayer, Skipped, _diff
from verity_vllm.commit import challenge as CH

DEFAULT_DEPTH = 3
Intervals = tuple[tuple[int, int], ...]        # sorted, disjoint, non-adjacent [lo, hi) pairs


# ---------------------------------------------------------------------------------------------------------
# the unit
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VU:
    """One local verification unit.

    `node`: root instance (index into `result.body.nodes`).  `path=()` and `member=0`: the whole instance.  For a
    child unit `path` is the descent `((node index in the enclosing body, member count), ...)` exactly as
    `Replayer.plan_child` returns it, `member` the member index at the LAST level and `members` the member indices
    at the levels above it (`len(members) == len(path) - 1`; `()` for a one-level descent).  `plan_child` omits the
    members because tier A draws them at random (`tier_a_pool`); an exhaustive unit must name them."""

    node: int
    path: tuple[tuple[int, int], ...] = ()
    member: int = 0
    members: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.path and len(self.members) != len(self.path) - 1:
            raise ValueError(f"VU with a path of {len(self.path)} levels needs {len(self.path) - 1} upper members, got {len(self.members)}")
        if not self.path and (self.member or self.members):
            raise ValueError("an instance-granularity VU has member=0 and members=()")

    @property
    def is_instance(self) -> bool:
        return not self.path

    @property
    def all_members(self) -> tuple[int, ...]:
        """Member index at every level of `path` (top to bottom)."""
        return (self.members + (self.member,)) if self.path else ()

    def to_json(self) -> dict:
        return {"node": self.node, "path": [list(p) for p in self.path], "member": self.member, "members": list(self.members)}

    @classmethod
    def from_json(cls, d: dict) -> "VU":
        return cls(int(d["node"]), tuple((int(a), int(b)) for a, b in d.get("path", [])), int(d.get("member", 0)),
                   tuple(int(x) for x in d.get("members", [])))


# ---------------------------------------------------------------------------------------------------------
# intervals and reference hulls
# ---------------------------------------------------------------------------------------------------------


def _merge(iv: list[tuple[int, int]]) -> Intervals:
    iv = sorted((lo, hi) for lo, hi in iv if hi > lo)
    out: list[tuple[int, int]] = []
    for lo, hi in iv:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return tuple(out)


def _covers(iv: Intervals, lo: int, hi: int) -> bool:
    """Is [lo, hi) inside the union of `iv`?  (empty ranges are covered)"""
    if hi <= lo:
        return True
    i = bisect.bisect_right(iv, (lo, float("inf"))) - 1
    return i >= 0 and iv[i][1] >= hi


def _disjoint(iv: Intervals, lo: int, hi: int) -> bool:
    if hi <= lo:
        return True
    i = bisect.bisect_right(iv, (lo, float("inf"))) - 1
    if i >= 0 and iv[i][1] > lo:
        return False
    return not (i + 1 < len(iv) and iv[i + 1][0] < hi)


def _count(iv: Intervals) -> int:
    return sum(hi - lo for lo, hi in iv)


def _intersect(iv: Intervals, lo: int, hi: int) -> list[tuple[int, int]]:
    out = []
    for a, b in iv:
        if b <= lo:
            continue
        if a >= hi:
            break
        out.append((max(a, lo), min(b, hi)))
    return out


def _hulls(r: Refs) -> list[tuple[str, int, int, int]]:
    """`(space, idx, lo, hi)` hulls of the structural parts of `r`: exact for Affine/Explicit, the bounding range
    for Strided (a superset of its leaves).  O(parts), never O(leaves)."""
    out: list[tuple[str, int, int, int]] = []

    def walk(x: Refs) -> None:
        if isinstance(x, Affine):
            if x.count:
                out.append((x.space, x.idx, x.base, x.base + x.count))
        elif isinstance(x, Strided):
            if x.count:
                n = x.count
                probes = {0, n - 1, min(x.inner - 1, n - 1), ((n - 1) // x.inner) * x.inner}
                if n > x.inner:
                    probes.add(((n - 1) // x.inner) * x.inner - 1)
                leaves = [x[j][2] for j in probes]
                out.append((x.space, x.idx, min(leaves), max(leaves) + 1))
        elif isinstance(x, Concat):
            for p in x.parts:
                walk(p)
        else:
            for space, idx, leaf in x:
                out.append((space, idx, leaf, leaf + 1))

    walk(r)
    return out


def _status(r: Refs, outside: tuple[Intervals, ...]) -> str:
    """Where do the leaves of `r` resolve, given the enclosing scope's `outside` parameter leaves?
    "out": every leaf resolves outside the root instance; "in": every leaf resolves to an interior value;
    "mixed": undecided at the hull level (decide per member with `_member_out`)."""
    any_out = any_in = False
    for space, idx, lo, hi in _hulls(r):
        if space == "n":
            any_in = True
        else:
            iv = outside[idx]
            if _covers(iv, lo, hi):
                any_out = True
            elif _disjoint(iv, lo, hi):
                any_in = True
            else:
                return "mixed"
        if any_in and any_out:
            return "mixed"
    return "in" if any_in else "out"


def _member_out(r: Refs, outside: tuple[Intervals, ...]) -> bool:
    """Exact: does every leaf of `r` resolve outside the root instance?"""
    st = _status(r, outside)
    if st != "mixed":
        return st == "out"
    return all(space == "p" and _covers(outside[idx], lo, hi) for space, idx, lo, hi in runs(r))


def _child_outside(r: Refs, outside: tuple[Intervals, ...]) -> Intervals:
    """The leaves of a child parameter bound to `r` that resolve outside the root instance, in the child's leaf space."""
    st = _status(r, outside)
    if st == "out":
        return ((0, r.count),) if r.count else ()
    if st == "in":
        return ()
    out: list[tuple[int, int]] = []
    pos = 0
    for space, idx, lo, hi in runs(r):
        if space == "p":
            for a, b in _intersect(outside[idx], lo, hi):
                out.append((pos + a - lo, pos + b - lo))
        pos += hi - lo
    return _merge(out)


def _member_refs(node, a, ax: int | None, m: int) -> Refs:
    """The references member `m` of a batch node binds to argument `a` (axis `ax`); the whole coll for shared /
    call / primitive arguments."""
    if ax is None:
        return a.refs
    if ax == 0:
        L = a.type.elem.leaves
        return a.refs.slice(m * L, (m + 1) * L)
    return a.take(ax, m).refs


def _param_leaves(fn) -> int:
    return sum(t.leaves for _, t in fn.signature()[0])


# ---------------------------------------------------------------------------------------------------------
# per-body plans (memoized by specialization, outside signature and depth)
# ---------------------------------------------------------------------------------------------------------


class _Node:
    """The admissible members of one body node and what each yields (a VU, or a sub-plan)."""

    __slots__ = ("j", "nm", "members", "kind", "uniform", "sub", "subs", "counts", "cum", "total")

    def __init__(self, j: int, nm: int) -> None:
        self.j, self.nm = j, nm
        self.members: Intervals = ()          # admissible members
        self.kind = "none"                    # "none" | "leaf" | "recurse"
        self.uniform = True                   # every admissible member has the same count / sub-plan
        self.sub: "_Plan | None" = None       # uniform recurse: the shared sub-plan
        self.subs: list["_Plan"] = []         # non-uniform recurse: one per admissible member (enumeration order)
        self.counts: list[int] = []           # non-uniform: count per admissible member
        self.cum: list[int] = [0]
        self.total = 0

    def finish(self) -> None:
        n = _count(self.members)
        if self.kind == "none" or n == 0:
            self.kind, self.total = "none", 0
            return
        if self.kind == "leaf":
            self.total = n
        elif self.uniform:
            self.total = n * (self.sub.total if self.sub is not None else 0)
        else:
            self.cum = [0]
            for c in self.counts:
                self.cum.append(self.cum[-1] + c)
            self.total = self.cum[-1]

    def member_at(self, pos: int) -> int:
        """The `pos`-th admissible member (ascending)."""
        for lo, hi in self.members:
            if pos < hi - lo:
                return lo + pos
            pos -= hi - lo
        raise IndexError(pos)

    def iter_members(self) -> Iterator[int]:
        for lo, hi in self.members:
            yield from range(lo, hi)


class _Plan:
    __slots__ = ("spec_id", "nodes", "cum", "total")

    def __init__(self, spec_id: str, nodes: list[_Node]) -> None:
        self.spec_id = spec_id
        self.nodes = [n for n in nodes if n.total]
        self.cum = [0]
        for n in self.nodes:
            self.cum.append(self.cum[-1] + n.total)
        self.total = self.cum[-1]


class _Planner:
    """Builds `_Plan`s for the activations of a Program.  Structural where possible, gate-level (on a representative
    activation) where the structure could alias."""

    def __init__(self, result: FoldResult, budget: int, depth: int) -> None:
        self.r = result
        self.budget, self.depth = budget, depth
        self.program = result.program
        self.circuit = self.program.circuit
        self.n_in = len(self.program.root.body.nodes) - len(result.body.nodes)
        self._plans: dict[tuple, _Plan] = {}
        self._clean: dict[str, bool] = {}
        self._ret_cover: dict[str, dict[int, Intervals]] = {}
        self.gate_checks = 0                  # how often the gate-level fallback was used (diagnostics)

    # -- structure of a specialization -------------------------------------------------------------------

    def ret_cover(self, spec) -> dict[int, Intervals]:
        """node index -> output leaves of that node referenced by the body's `ret`."""
        c = self._ret_cover.get(spec.id)
        if c is None:
            acc: dict[int, list[tuple[int, int]]] = {}
            for space, idx, lo, hi in runs(spec.body.ret):
                if space == "n":
                    acc.setdefault(idx, []).append((lo, hi))
            c = {j: _merge(v) for j, v in acc.items()}
            self._ret_cover[spec.id] = c
        return c

    def clean(self, fn) -> bool:
        """No return aliasing anywhere below `fn`: every `ret` reference of every descendant body is a distinct
        `n`-space reference.  Then "leaf-level containment fails" implies "gate-level containment fails"."""
        if isinstance(fn, PrimitiveDefinition):
            return True
        c = self._clean.get(fn.id)
        if c is None:
            self._clean[fn.id] = True             # provisional (recursive specs do not occur, but be safe)
            seen: set = set()
            c = True
            for ref in fn.body.ret:
                if ref[0] != "n" or ref in seen:
                    c = False
                    break
                seen.add(ref)
            if c:
                c = all(self.clean(node.fn) for node in fn.body.nodes)
            self._clean[fn.id] = c
        return c

    # -- plans ---------------------------------------------------------------------------------------------

    def plan(self, scope: Scope, outside: tuple[Intervals, ...], depth: int, root: tuple[int, int]) -> _Plan:
        """The sub-population under activation `scope` (a representative; the plan is offset-invariant).
        `outside[pi]`: leaves of parameter `pi` that resolve outside the root instance.  `root`: the gate range
        of the root instance (for the gate-level fallback only)."""
        spec = scope.spec
        key = (spec.id, outside, depth)
        p = self._plans.get(key)
        if p is not None:
            return p
        body = spec.body
        cover = self.ret_cover(spec)
        # leaf-level containment decides the returns test exactly unless some node of this body (a sibling's
        # pass-through, a repeated return anywhere below) could make two references name one gate
        body_clean = all(self.clean(nd.fn) for nd in body.nodes)
        own_returns: set[int] | None = None       # gate-level fallback: this activation's return gates
        nodes: list[_Node] = []
        for j, node in enumerate(body.nodes):
            fn = node.fn
            nm = 1 if node.form in ("primitive", "call") else node.n
            n = _Node(j, nm)
            is_prim = isinstance(fn, PrimitiveDefinition)
            fits = _param_leaves(fn) <= self.budget
            if not fits and (is_prim or depth <= 0):
                nodes.append(n)
                continue
            # 1. returns: members whose return gates are all among the enclosing activation's returns
            if node.form == "scan":
                ok_leaf: Intervals = ()
                pending = ((0, nm),)
            else:
                L = node.out_leaves // nm
                ok: list[tuple[int, int]] = []
                for lo, hi in cover.get(j, ()):
                    mlo, mhi = -(-lo // L), hi // L
                    if mhi > mlo:
                        ok.append((max(mlo, 0), min(mhi, nm)))
                ok_leaf = _merge(ok)
                pending = _merge(self._complement(ok_leaf, nm)) if not body_clean else ()
            ret_ok = list(ok_leaf)
            if pending:
                if own_returns is None:
                    own_returns = set(resolve(scope, r) for r in body.ret)
                for lo, hi in pending:
                    for m in range(lo, hi):
                        self.gate_checks += 1
                        child = FunctionCall(self.circuit, scope, j, m)
                        if all(g in own_returns for g in child.returns):
                            ret_ok.append((m, m + 1))
            ret_ok_iv = _merge(ret_ok)
            if not ret_ok_iv:
                nodes.append(n)
                continue
            axes = node.axes if node.form == "batch" else (None,) * len(node.args)
            if fits:
                # 2. arguments: every argument gate outside the root instance
                n.kind = "leaf"
                if node.form == "scan":
                    n.members = _merge([(m, m + 1) for lo, hi in ret_ok_iv for m in range(lo, hi)
                                        if self._scan_args_outside(scope, j, m, root)])
                else:
                    sts = [_status(a.refs, outside) for a in node.args]
                    if all(s == "out" for s in sts):
                        n.members = ret_ok_iv
                    elif any(s == "in" for s in sts):
                        n.members = ()
                    else:
                        shared_ok = all(_member_out(a.refs, outside) for a, ax, s in zip(node.args, axes, sts)
                                        if ax is None and s == "mixed")
                        if not shared_ok:
                            n.members = ()
                        else:
                            n.members = _merge([(m, m + 1) for lo, hi in ret_ok_iv for m in range(lo, hi)
                                                if all(s != "mixed" or ax is None or _member_out(_member_refs(node, a, ax, m), outside)
                                                       for a, ax, s in zip(node.args, axes, sts))])
                n.finish()
                nodes.append(n)
                continue
            # 3. oversize composite: descend into every admissible member
            n.kind = "recurse"
            n.members = ret_ok_iv
            if node.form == "scan":
                n.uniform = False
                for m in n.iter_members():
                    child = FunctionCall(self.circuit, scope, j, m)
                    sub = self.plan(child.scope(), self._scan_child_outside(child, root), depth - 1, root)
                    n.subs.append(sub)
                    n.counts.append(sub.total)
            else:
                sts = [_status(a.refs, outside) for a in node.args]
                if all(s != "mixed" or ax is None for s, ax in zip(sts, axes)):
                    child_out = tuple(_child_outside(_member_refs(node, a, ax, 0), outside) if s == "mixed" else
                                      (((0, _member_refs(node, a, ax, 0).count),) if s == "out" else ())
                                      for a, ax, s in zip(node.args, axes, sts))
                    n.sub = self.plan(scope.child(j, ret_ok_iv[0][0]), child_out, depth - 1, root)
                else:
                    n.uniform = False
                    for m in n.iter_members():
                        child_out = tuple(_child_outside(_member_refs(node, a, ax, m), outside) for a, ax in zip(node.args, axes))
                        sub = self.plan(scope.child(j, m), child_out, depth - 1, root)
                        n.subs.append(sub)
                        n.counts.append(sub.total)
            n.finish()
            nodes.append(n)
        p = _Plan(spec.id, nodes)
        self._plans[key] = p
        return p

    @staticmethod
    def _complement(iv: Intervals, n: int) -> list[tuple[int, int]]:
        out, pos = [], 0
        for lo, hi in iv:
            if lo > pos:
                out.append((pos, lo))
            pos = hi
        if pos < n:
            out.append((pos, n))
        return out

    def _scan_args_outside(self, scope: Scope, j: int, m: int, root: tuple[int, int]) -> bool:
        self.gate_checks += 1
        child = FunctionCall(self.circuit, scope, j, m)
        rlo, rhi = root
        return all(not (rlo <= g < rhi) for g in child.args)

    def _scan_child_outside(self, child: FunctionCall, root: tuple[int, int]) -> tuple[Intervals, ...]:
        self.gate_checks += 1
        rlo, rhi = root
        sc = child.scope()
        out = []
        pos = 0
        gates = child.args
        for _, t in sc.spec.body.params:
            iv = [(pos + i, pos + i + 1) for i in range(t.leaves) if not (rlo <= gates[pos + i] < rhi)]
            out.append(tuple((a - pos, b - pos) for a, b in _merge(iv)))
            pos += t.leaves
        return tuple(out)


# ---------------------------------------------------------------------------------------------------------
# the population index
# ---------------------------------------------------------------------------------------------------------


class VUIndex:
    """The VU population of `result` under (`max_arg_leaves`, `depth`): `count`, `by_index(i)`, iteration in
    canonical order, per-instance counts.  Built in O(instances + distinct specializations)."""

    def __init__(self, result: FoldResult, *, max_arg_leaves: int, depth: int = DEFAULT_DEPTH) -> None:
        if result.program is None:
            raise ValueError("fold ran without Program(); the VU query needs the circuit")
        t0 = time.time()
        self.r = result
        self.max_arg_leaves, self.depth = int(max_arg_leaves), int(depth)
        self.program = result.program
        self.circuit = self.program.circuit
        self.root_scope = self.circuit.root_scope
        self.n_in = len(self.program.root.body.nodes) - len(result.body.nodes)
        self.planner = _Planner(result, self.max_arg_leaves, self.depth)
        self.by_node: dict[int, dict] = {m["node"]: m for m in result.instances}
        nodes = result.body.nodes
        self.fits: list[bool] = []
        self.plans: list[_Plan | None] = []
        counts = np.zeros(len(nodes), dtype=np.int64)
        for k, node in enumerate(nodes):
            leaves = sum(a.refs.count for a in node.args)
            if leaves <= self.max_arg_leaves:
                self.fits.append(True)
                self.plans.append(None)
                counts[k] = 1
                continue
            self.fits.append(False)
            call = self.call(k)
            if call.is_primitive:
                self.plans.append(None)                   # an oversize primitive root instance has no child units
                continue
            sc = call.scope()
            outside = tuple(((0, t.leaves),) if t.leaves else () for _, t in sc.spec.body.params)
            plan = self.planner.plan(sc, outside, self.depth, (call.lo, call.hi))
            self.plans.append(plan)
            counts[k] = plan.total
        self.per_instance = counts
        self.prefix = np.concatenate([[0], np.cumsum(counts)])
        self.count = int(self.prefix[-1])
        self.build_seconds = round(time.time() - t0, 4)

    # -- calls -----------------------------------------------------------------------------------------------

    def call(self, k: int) -> FunctionCall:
        return FunctionCall(self.circuit, self.root_scope, self.n_in + k)

    def vu_call(self, vu: VU) -> FunctionCall:
        call = self.call(vu.node)
        for (j, nm), m in zip(vu.path, vu.all_members):
            node = call.scope().spec.body.nodes[j]
            expect = 1 if node.form in ("primitive", "call") else node.n
            if nm != expect or not (0 <= m < nm):
                raise ValueError(f"{vu}: node {j} has {expect} members, path says {nm} / member {m}")
            call = FunctionCall(self.circuit, call.scope(), j, m)
        return call

    def granularity(self, vu: VU) -> str:
        return "instance" if vu.is_instance else f"child:{self.vu_call(vu).id}"

    def labels(self, vu: VU) -> list[str]:
        """`"<spec id>#<member>"` per level, as `Replayer.evaluate_child` reports them."""
        out, call = [], self.call(vu.node)
        for (j, _), m in zip(vu.path, vu.all_members):
            call = FunctionCall(self.circuit, call.scope(), j, m)
            out.append(f"{call.id}#{m}")
        return out

    # -- enumeration -----------------------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[VU]:
        for k in range(len(self.fits)):
            yield from self.of_instance(k)

    def of_instance(self, k: int) -> Iterator[VU]:
        """The VUs of root instance `k`, in canonical order."""
        if self.fits[k]:
            yield VU(k)
            return
        plan = self.plans[k]
        if plan is not None:
            yield from self._walk(plan, k, (), ())

    def _walk(self, plan: _Plan, k: int, path: tuple, members: tuple) -> Iterator[VU]:
        for n in plan.nodes:
            step = path + ((n.j, n.nm),)
            if n.kind == "leaf":
                for m in n.iter_members():
                    yield VU(k, step, m, members)
            elif n.uniform:
                for m in n.iter_members():
                    yield from self._walk(n.sub, k, step, members + (m,))
            else:
                for m, sub in zip(n.iter_members(), n.subs):
                    yield from self._walk(sub, k, step, members + (m,))

    def by_index(self, i: int) -> VU:
        if not (0 <= i < self.count):
            raise IndexError(f"VU index {i} outside the population of {self.count}")
        k = int(np.searchsorted(self.prefix, i, side="right")) - 1
        rem = i - int(self.prefix[k])
        if self.fits[k]:
            return VU(k)
        return self._locate(self.plans[k], k, rem, (), ())

    def _locate(self, plan: _Plan, k: int, rem: int, path: tuple, members: tuple) -> VU:
        t = bisect.bisect_right(plan.cum, rem) - 1
        n = plan.nodes[t]
        rem -= plan.cum[t]
        step = path + ((n.j, n.nm),)
        if n.kind == "leaf":
            return VU(k, step, n.member_at(rem), members)
        if n.uniform:
            per = n.sub.total
            q, rem2 = divmod(rem, per)
            return self._locate(n.sub, k, rem2, step, members + (n.member_at(q),))
        q = bisect.bisect_right(n.cum, rem) - 1
        return self._locate(n.subs[q], k, rem - n.cum[q], step, members + (n.member_at(q),))

    def index_of(self, vu: VU) -> int:
        """Position of `vu` in the canonical order (inverse of `by_index`); ValueError if not in the population."""
        k = vu.node
        base = int(self.prefix[k])
        if self.fits[k]:
            if not vu.is_instance:
                raise ValueError(f"{vu}: instance {k} fits the budget; it has no child units")
            return base
        plan = self.plans[k]
        if plan is None or vu.is_instance:
            raise ValueError(f"{vu} is not in the population")
        pos = 0
        mems = vu.all_members
        for lvl, (j, nm) in enumerate(vu.path):
            n = next((x for x in plan.nodes if x.j == j), None)
            if n is None or n.nm != nm:
                raise ValueError(f"{vu} is not in the population (node {j} at level {lvl})")
            t = plan.nodes.index(n)
            pos += plan.cum[t]
            m = mems[lvl]
            q = self._member_pos(n, m)
            last = lvl == len(vu.path) - 1
            if n.kind == "leaf":
                if not last:
                    raise ValueError(f"{vu}: the descent continues below a leaf unit")
                return base + pos + q
            if last:
                raise ValueError(f"{vu}: the descent stops at a node that is descended into")
            if n.uniform:
                pos += q * n.sub.total
                plan = n.sub
            else:
                pos += n.cum[q]
                plan = n.subs[q]
        raise ValueError(f"{vu} is not in the population")

    @staticmethod
    def _member_pos(n: _Node, m: int) -> int:
        pos = 0
        for lo, hi in n.members:
            if lo <= m < hi:
                return pos + m - lo
            pos += hi - lo
        raise ValueError(f"member {m} of node {n.j} is not admissible")

    def __contains__(self, vu: VU) -> bool:
        try:
            self.index_of(vu)
            return True
        except (ValueError, IndexError):
            return False

    # -- summaries -------------------------------------------------------------------------------------------

    def summary(self) -> dict:
        """Per specialization: instances, VUs, granularity label(s) of the VUs."""
        out: dict[str, dict] = {}
        for k, m in self.by_node.items():
            s = out.setdefault(m["spec"], {"instances": 0, "vus": 0, "granularity": set()})
            s["instances"] += 1
            s["vus"] += int(self.per_instance[k])
            if self.fits[k]:
                s["granularity"].add("instance")
            elif self.plans[k] is not None and self.plans[k].total:
                first = next(self.of_instance(k))
                s["granularity"].add(self.granularity(first))
            else:
                s["granularity"].add("none")
        for s in out.values():
            s["granularity"] = sorted(s["granularity"])
        return {"max_arg_leaves": self.max_arg_leaves, "depth": self.depth, "count": self.count, "instances": len(self.fits),
                "oversize_instances": int(sum(1 for f in self.fits if not f)),
                "instances_without_units": int(sum(1 for k, f in enumerate(self.fits) if not f and not self.per_instance[k])),
                "gate_level_checks": self.planner.gate_checks, "build_seconds": self.build_seconds, "specs": out}


# ---------------------------------------------------------------------------------------------------------
# module API (indexes cached per result object)
# ---------------------------------------------------------------------------------------------------------


_CACHE: dict[tuple[int, int, int], tuple[Any, VUIndex]] = {}


def index(result: FoldResult, *, max_arg_leaves: int, depth: int = DEFAULT_DEPTH) -> VUIndex:
    """The (cached) `VUIndex` of `result`."""
    key = (id(result), int(max_arg_leaves), int(depth))
    hit = _CACHE.get(key)
    if hit is not None and hit[0]() is result:
        return hit[1]
    ix = VUIndex(result, max_arg_leaves=max_arg_leaves, depth=depth)
    _CACHE[key] = (weakref.ref(result), ix)
    return ix


def enumerate_vus(result: FoldResult, *, max_arg_leaves: int, depth: int = DEFAULT_DEPTH) -> Iterator[VU]:
    """Every VU in canonical order (see the module docstring)."""
    yield from index(result, max_arg_leaves=max_arg_leaves, depth=depth)


def vu_count(result: FoldResult, *, max_arg_leaves: int, depth: int = DEFAULT_DEPTH) -> int:
    return index(result, max_arg_leaves=max_arg_leaves, depth=depth).count


def vu_by_index(result: FoldResult, i: int, *, max_arg_leaves: int, depth: int = DEFAULT_DEPTH) -> VU:
    """The i-th VU of the canonical order, without enumeration."""
    return index(result, max_arg_leaves=max_arg_leaves, depth=depth).by_index(i)


def vu_call(result: FoldResult, vu: VU) -> FunctionCall:
    return _plain(result).vu_call(vu)


def operands_of(result: FoldResult, vu: VU) -> tuple[int, ...]:
    """Root-circuit gate indices the VU reads (signature order, repeats kept); the instance's arguments for
    instance granularity."""
    return tuple(int(g) for g in vu_call(result, vu).args)


def returns_of(result: FoldResult, vu: VU) -> tuple[int, ...]:
    """Root-circuit gate indices the VU returns; the instance's return gates for instance granularity."""
    return tuple(int(g) for g in vu_call(result, vu).returns)


def _plain(result: FoldResult) -> VUIndex:
    """An index for call arithmetic only (every instance fits: nothing is planned)."""
    key = (id(result), -1, -1)
    hit = _CACHE.get(key)
    if hit is not None and hit[0]() is result:
        return hit[1]
    ix = VUIndex(result, max_arg_leaves=1 << 62, depth=0)
    _CACHE[key] = (weakref.ref(result), ix)
    return ix


def evaluate_vu(result: FoldResult, store: SnapshotStore, weights: WeightSource | None, vu: VU, *,
                rep: Replayer | None = None) -> dict:
    """Evaluate `vu` on its observed operands and compare bit-exactly with the parent instance's output snapshot
    at the VU's return gates.  Instance granularity: `Replayer.evaluate_instance` (operands by provenance).  Child
    units: operand words by gate (`Replayer.gate_words`: parameters from prompt / checkpoint, produced values from
    the producer's snapshot), expected words from the parent's snapshot by return-gate position.  Raises
    `replay.Skipped` when evidence is missing.  Pass `rep` (a `Replayer(result, store, weights, chain=False)`) to
    evaluate many units with one set of caches."""
    if rep is None:
        rep = Replayer(result, store, weights, chain=False, snapshots=True)
    m = rep.by_node.get(vu.node)
    if m is None:
        raise Skipped(f"node {vu.node} is not an instance with metadata")
    if vu.is_instance:
        e = rep.evaluate_instance(m)
        e["vu"] = vu.to_json()
        return e
    t0 = time.time()
    call = rep.call(vu.node)
    labels = []
    for (j, nm), member in zip(vu.path, vu.all_members):
        call = FunctionCall(rep.circuit, call.scope(), j, member)
        labels.append(f"{call.id}#{member}")
    entry: dict[str, Any] = {"node": vu.node, "name": m["name"], "spec": m["spec"], "step": m["step"],
                             "granularity": f"child:{call.id}", "child": {"path": labels, "gates": [call.lo, call.hi]},
                             "vu": vu.to_json()}
    rep._out_cache.clear()
    args = rep.gate_words(call.args)
    expect_all = rep.expected_words(m)
    if expect_all is None:
        raise Skipped(f"no snapshot of event {m['event']} covers the outputs")
    rix = rep.ret_index(vu.node)
    try:
        pos = [rix[g] for g in call.returns]
    except KeyError as e:
        raise Skipped(f"return gate {e.args[0]} of {call!r} is interior to instance {vu.node}") from None
    expect = expect_all[np.array(pos, dtype=np.int64)]
    got = np.array(evaluate_call(rep.circuit, call, args), dtype=np.uint64)
    n_diff, first = _diff(got, expect)
    entry.update(status="pass" if not n_diff else "fail", n_args=len(args), n_returns=int(got.size), n_diff=n_diff,
                 first_diff=first, seconds=round(time.time() - t0, 3))
    return entry


def production_sample(result: FoldResult, *, seed: int, n: int, max_arg_leaves: int, depth: int = DEFAULT_DEPTH) -> list[VU]:
    """`n` VUs drawn UNIFORMLY WITH REPLACEMENT from the VU population (`random.Random(seed)`, one `randrange`
    per draw; deterministic per seed).  The measure is uniform over VUs — NOT over root instances (an oversize
    Gemm with N coordinates carries N times the mass of one RMSNorm instance) and NOT over specializations.  If
    the security analysis assumes another measure (per instance, per specialization × regime, per gate, ...),
    weight `vu_by_index` accordingly; this function is the uniform-over-VUs baseline."""
    ix = index(result, max_arg_leaves=max_arg_leaves, depth=depth)
    if ix.count == 0:
        raise ValueError("the VU population is empty")
    rng = CH.stream(seed)
    return [ix.by_index(rng.randrange(ix.count)) for _ in range(int(n))]
