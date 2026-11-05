"""Dead gates of a `Program`: hierarchical liveness over the typed Program.

A gate is *dead* when no other gate consumes its value and it is not protected (not a root output). Dead
gates are legal in every mode; they are counted toward the described computation, they are **not** boundary
values of any gate selection by existence (`S.out` is the graph boundary — a value nobody reads is not on it), and
frontends report them (count + ids).

The analysis never expands the circuit. It runs once per (specialised definition, set of live return leaves):
walking a body backwards, the live output leaves of every node are accumulated as *interval sets* over the
leaf order (weight tables, KV caches and `batch` outputs stay O(#reference runs)); a call node passes its live
return leaves into the callee, which answers with the parameter leaves it actually reads (a callee that ignores
a parameter does not consume the argument — `AttnBlock{FIRST=True}` ignores its state arguments, so the constant
`AttentionHead` feeds it is dead) and with its dead-gate count; `batch` members are grouped by identical live
sets (one analysis for 151,936 `GemmCoordinate` members), `scan` iterations are walked backwards through the
carried state. `Input` gates nobody reads (the `cos_sin` rows beyond the served positions) are counted
separately from dead *computed* gates because the accounting treats prescribed inputs differently.

`dead_gates(program)` returns the exact counts; ids are enumerated in canonical order up to a limit by
descending only into activations whose subtree has dead gates. A reference sequence with no compact interval
form that is too large to enumerate raises `intervals.IntervalLimit`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .defs import Body, PrimitiveDefinition, SpecializedDefinition
from .intervals import Intervals, complement, full, norm, ref_intervals, restrict, total
from .refs import Concat, Refs
from .types import Tuple as TupleT

_MEMBER_LIMIT = 1 << 22  # members × partial-read intervals above which a batch argument is over-approximated


def _members(lv: Intervals, n: int, R: int, include_dead: bool = False) -> tuple[list[tuple[int, int, Intervals]], int]:
    """Group the `n` members of a batch (R output leaves each) by their live output subsets. Returns
    `[(m_lo, m_hi, live_subset)]` (ranges of members sharing one subset, in member order) and the number of
    members with no live leaf at all."""
    groups: list[tuple[int, int, Intervals]] = []
    partial: dict[int, list[tuple[int, int]]] = {}
    whole = ((0, R),)
    covered = 0
    for lo, hi in lv:
        m_lo, m_hi = lo // R, (hi - 1) // R
        if m_lo == m_hi:
            partial.setdefault(m_lo, []).append((lo - m_lo * R, hi - m_lo * R))
            continue
        partial.setdefault(m_lo, []).append((lo - m_lo * R, R))
        if m_hi > m_lo + 1:
            groups.append((m_lo + 1, m_hi, whole))
            covered += m_hi - m_lo - 1
        partial.setdefault(m_hi, []).append((0, hi - m_hi * R))
    for m, ivs in partial.items():
        groups.append((m, m + 1, norm(ivs)))
    covered += len(partial)
    groups.sort(key=lambda g: g[0])
    if include_dead:  # the members with no live leaf, as groups with an empty live set
        filled: list[tuple[int, int, Intervals]] = []
        pos = 0
        for g in groups:
            if g[0] > pos:
                filled.append((pos, g[0], ()))
            filled.append(g)
            pos = g[1]
        if pos < n:
            filled.append((pos, n, ()))
        groups = filled
    return groups, n - covered


@dataclass
class BodyLiveness:
    """One activation class: a specialisation with a given set of live return leaves."""
    reads: tuple[Intervals, ...]  # per parameter: the leaves read by live gates
    dead_computed: int  # dead gates in this activation and everything it calls
    dead_input: int  # of which `Input` gates (root only)
    node_live: list[Intervals]  # per body node: live output leaves
    scan_runs: dict[int, list[tuple[Intervals, int]]] = field(default_factory=dict)  # scan node -> run-length live rets per iteration
    approximate: bool = False

    @property
    def dead(self) -> int:
        return self.dead_computed + self.dead_input


class Liveness:
    """Memoised analysis over (specialisation, live return leaves).

    `transitive=False` (the default) computes **dead gates** in the sense of §8.6: no consumer and not
    protected — every consumer counts, even one that is itself dead. `transitive=True` computes the gates with
    **no dependency path to a protected output** (dead gates plus everything that only feeds dead gates):
    the "can its value affect the protected result?" question of §8.6, a superset of the first."""

    def __init__(self, transitive: bool = False) -> None:
        self.transitive = transitive
        self._cache: dict[tuple[int, Intervals], BodyLiveness] = {}
        self._keep: list = []  # keep the specialisations alive (ids are the cache key)

    def analyse(self, sd: SpecializedDefinition, live_ret: Intervals) -> BodyLiveness:
        key = (id(sd), live_ret)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        self._keep.append(sd)
        res = self._analyse(sd.body, live_ret)
        self._cache[key] = res
        return res

    # -- one body -----------------------------------------------------------------------------------
    def _analyse(self, body: Body, live_ret: Intervals) -> BodyLiveness:
        nodes = body.nodes
        live: list[list[tuple[int, int]]] = [[] for _ in nodes]
        pread: list[list[tuple[int, int]]] = [[] for _ in body.params]
        approx = False
        dead_c = dead_i = 0
        node_live: list[Intervals] = [()] * len(nodes)
        scan_runs: dict[int, list[tuple[Intervals, int]]] = {}

        marked: dict[int, list] = {}  # id(PartLog) -> [log, how many of its parts are marked]
        seen: set = set()  # the strided reads marked so far (ref_intervals)

        def mark(refs: Refs) -> None:
            t: dict[tuple[str, int], list] = {}
            if isinstance(refs, Concat) and refs.prefix is not None:
                log, j = refs.prefix
                m = marked.get(id(log))
                if m is None or m[0] is not log:
                    m = marked[id(log)] = [log, 0]
                for p in log.parts[m[1]:j]:
                    ref_intervals(p, t, seen)
                m[1] = max(m[1], j)
                for p in refs.parts[j:]:
                    ref_intervals(p, t, seen)
            else:
                ref_intervals(refs, t, seen)
            for (space, idx), ivs in t.items():
                (live if space == "n" else pread)[idx].extend(ivs)

        def mark_subset(refs: Refs, sub: Intervals) -> None:
            if not sub:
                return
            if total(sub) == refs.count:
                mark(refs)
                return
            for lo, hi in sub:
                mark(refs.slice(lo, hi))

        for lo, hi in live_ret:
            mark(body.ret.slice(lo, hi))

        transitive = self.transitive
        for k in range(len(nodes) - 1, -1, -1):
            node = nodes[k]
            fn = node.fn
            lv = norm(live[k])
            node_live[k] = lv
            tot = total(lv)
            # what the node's arguments are marked with: its live outputs (transitive) or all of them (direct:
            # a consumer counts even when it is dead itself)
            use = lv if transitive else (((0, node.out.leaves),) if node.out.leaves else ())
            if node.form == "primitive":
                if tot == 0:
                    if getattr(fn, "family", None) == "Input":
                        dead_i += 1
                    else:
                        dead_c += 1
                    if transitive:
                        continue
                for a in node.args:
                    mark(a.refs)
            elif node.form == "call":
                sub = self.analyse(fn, lv)
                dead_c += sub.dead_computed
                dead_i += sub.dead_input
                approx |= sub.approximate
                for pi, a in enumerate(node.args):
                    mark_subset(a.refs, sub.reads[pi])
            elif node.form == "batch":
                if isinstance(fn, PrimitiveDefinition):
                    n_dead = node.n - tot
                    if getattr(fn, "family", None) == "Input":
                        dead_i += n_dead
                    else:
                        dead_c += n_dead
                    if total(use) == 0:
                        continue
                    if total(use) == node.n:
                        for a in node.args:
                            mark(a.refs)
                        continue
                    for a, ax in zip(node.args, node.axes):
                        if ax is None:
                            mark(a.refs)
                        elif ax == 0:
                            Lp = a.type.elem.leaves
                            for lo, hi in use:
                                mark(a.refs.slice(lo * Lp, hi * Lp))
                        else:
                            for lo, hi in use:
                                for m in range(lo, hi):
                                    mark(a.take(ax, m).refs)
                else:
                    R = fn.out_leaves
                    # transitive: a member with no live output is entirely dead and reads nothing;
                    # direct: it still reads its arguments, and only its unconsumed producers are dead
                    groups, n_dead = _members(lv, node.n, R, include_dead=not transitive)
                    if transitive:
                        dead_c += n_dead * fn.gates
                    for m_lo, m_hi, lr in groups:
                        sub = self.analyse(fn, lr)
                        cnt = m_hi - m_lo
                        dead_c += cnt * sub.dead_computed
                        dead_i += cnt * sub.dead_input
                        approx |= sub.approximate
                        for pi, (a, ax) in enumerate(zip(node.args, node.axes)):
                            rd = sub.reads[pi]
                            if not rd:
                                continue
                            if ax is None:
                                mark_subset(a.refs, rd)
                            elif ax == 0:
                                Lp = a.type.elem.leaves
                                if total(rd) == Lp:
                                    mark(a.refs.slice(m_lo * Lp, m_hi * Lp))
                                elif cnt * len(rd) <= _MEMBER_LIMIT:
                                    for m in range(m_lo, m_hi):
                                        for lo, hi in rd:
                                            mark(a.refs.slice(m * Lp + lo, m * Lp + hi))
                                else:  # over-approximate: the whole member slice is taken as read
                                    mark(a.refs.slice(m_lo * Lp, m_hi * Lp))
                                    approx = True
                            else:
                                for m in range(m_lo, m_hi):
                                    mark_subset(a.take(ax, m).refs, rd)
            else:  # scan
                assert isinstance(node.out, TupleT)
                c = node.out.fields[0].leaves
                y = node.out.fields[1].elem.leaves
                carry_live = restrict(lv, 0, c)  # live leaves of the final carry
                ys_groups, _ = _members(restrict(lv, c, c + node.n * y), node.n, y) if y else ([], node.n)
                ys_live: dict[int, Intervals] = {}
                for m_lo, m_hi, lr in ys_groups:
                    for m in range(m_lo, m_hi):
                        ys_live[m] = lr
                shared_reads: list[list[tuple[int, int]]] = [[] for _ in range(len(node.args) - 1 - node.n_xs)]
                runs: list[tuple[Intervals, int]] = []
                for i in range(node.n - 1, -1, -1):
                    lr = norm(list(carry_live) + [(c + lo, c + hi) for lo, hi in ys_live.get(i, ())])
                    if runs and runs[-1][0] == lr:
                        runs[-1] = (lr, runs[-1][1] + 1)
                    else:
                        runs.append((lr, 1))
                    sub = self.analyse(fn, lr)
                    dead_c += sub.dead_computed
                    dead_i += sub.dead_input
                    approx |= sub.approximate
                    reads = sub.reads
                    for j in range(node.n_xs):
                        a = node.args[1 + j]
                        Lx = a.type.elem.leaves
                        for lo, hi in reads[1 + j]:
                            mark(a.refs.slice(i * Lx + lo, i * Lx + hi))
                    for j, rd in enumerate(reads[1 + node.n_xs:]):
                        shared_reads[j].extend(rd)
                    carry_live = reads[0]
                mark_subset(node.args[0].refs, carry_live)  # the init feeds iteration 0
                for j, rd in enumerate(shared_reads):
                    mark_subset(node.args[1 + node.n_xs + j].refs, norm(rd))
                scan_runs[k] = runs[::-1]  # iteration order
        reads = tuple(norm(p) for p in pread)
        return BodyLiveness(reads, dead_c, dead_i, node_live, scan_runs, approx)


@dataclass
class DeadGateReport:
    computed: int  # dead computed gates (no consumer, not a root output)
    input: int  # dead `Input` gates (prescribed inputs the program never reads)
    ids: list[int]  # the first `limit` dead gates in canonical order (root-scoped indices)
    ids_complete: bool  # True iff `ids` lists every dead gate
    on_s_out: int  # dead gates that are root outputs: 0 by construction (a root output is live)
    approximate: bool  # a batch argument was over-approximated as fully read (counts are then lower bounds)

    @property
    def total(self) -> int:
        return self.computed + self.input

    def to_json(self) -> dict:
        return {"dead_computed_gates": self.computed, "dead_input_gates": self.input, "dead_gates": self.total,
                "ids": list(self.ids), "ids_complete": self.ids_complete, "on_S_out": self.on_s_out,
                "approximate": self.approximate}


def _live_all(sd: SpecializedDefinition) -> Intervals:
    return full(sd.body.ret_type.leaves)


def dead_gates(program, limit: int = 64, transitive: bool = False) -> DeadGateReport:
    """Exact dead-gate counts of `program` (a `Program` or `DecodedProgram`) and up to `limit` ids.

    `transitive=False`: dead = no consumer and not protected (§8.6). `transitive=True`: no dependency path to
    a protected output (a superset)."""
    root = program.root
    lv = Liveness(transitive=transitive)
    top = lv.analyse(root, _live_all(root))
    ids: list[int] = []
    complete = True
    if limit > 0 and top.dead:
        for g in _enumerate(lv, root, 0, _live_all(root)):
            ids.append(g)
            if len(ids) >= limit:
                complete = False
                break
    complete = complete and len(ids) == top.dead
    roots = set(program.circuit.root_outputs()) if ids else set()
    return DeadGateReport(top.dead_computed, top.dead_input, ids, complete, sum(1 for g in ids if g in roots), top.approximate)


def _enumerate(lv: Liveness, sd: SpecializedDefinition, offset: int, live_ret: Intervals) -> Iterator[int]:
    res = lv.analyse(sd, live_ret)
    if not res.dead:
        return
    body = sd.body
    for k, node in enumerate(body.nodes):
        fn = node.fn
        node_off = offset + body.offsets[k]
        lvk = res.node_live[k]
        if node.form == "primitive":
            if not lvk:
                yield node_off
        elif node.form == "call":
            yield from _enumerate(lv, fn, node_off, lvk)
        elif node.form == "batch":
            if isinstance(fn, PrimitiveDefinition):
                for lo, hi in complement(lvk, node.n):
                    yield from range(node_off + lo, node_off + hi)
            else:
                R, g = fn.out_leaves, fn.gates
                for m in range(node.n):
                    lr = restrict(lvk, m * R, (m + 1) * R)
                    if not lr and lv.transitive:
                        yield from range(node_off + m * g, node_off + (m + 1) * g)
                    else:
                        yield from _enumerate(lv, fn, node_off + m * g, lr)
        else:  # scan
            g = fn.gates
            i = 0
            for lr, cnt in res.scan_runs[k]:
                for _ in range(cnt):
                    yield from _enumerate(lv, fn, node_off + i * g, lr)
                    i += 1


__all__ = ["Liveness", "BodyLiveness", "DeadGateReport", "dead_gates"]
