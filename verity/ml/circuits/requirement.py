"""A declared circuit's requirement of memory, from its wiring and its schedule (the IR, never a kernel): the values its
execution keeps live when each step begins and the values each step reads, in bits, in all and by block of the store.
The Python counterpart of `Security/Definitions/Accounting/Requirement.lean`, by explicit expansion of the canonical
circuit (small Programs, `layout.Circuit`'s explicit limit), which the Lean definitions and the memory map (I3) are held to.

    live(t)  = the values produced before step t (or registered) that a gate run at step t or later has as an operand
    reads(t) = the values produced before step t (or registered) that a gate run at step t uses

A gate's operands are `layout.Circuit.gate(i).operands`. It uses all of them, except that a primitive with an
`evaluate_lazy` (a gather: its index and the word it selects) uses the operands its evaluator fetches, which takes the
run's values (`inputs`, as `evaluate.py`'s transcript holds them). Without them, its array operands are candidates
(`upper`): `reads(t)` refuses a step whose candidates from before `t` aren't otherwise read, since an upper bound would
over-credit the audit, and `reads_upper(t)` counts them. `live(t)` is `boundary.py`'s `out(S_<t) ∩ in(S_≥t)` for `S_<t`
the gates of the steps before `t`, and `reads_upper(t)` is `in(S_t)` less the run inputs that arrive at `t`;
`tests/test_requirement.py` holds both to `boundary.out_gates` and `boundary.in_gates`.

`from_census` is the same requirement at any scale, from a census (`census.census`) in place of the gates: a value is a
census value (a run of a root node's output leaves read alike), and a schedule's steps are whole root Calls. A census
holds no run values, so its candidates stay unresolved. `tests/test_requirement.py` holds it to `requirement` gate by gate.

A schedule is the root gate ranges each step runs: a Program's root Calls split at their first Calls (`Schedule.by_calls`,
the census's steps), or a WorkloadProgram's request calls, request r's step s (its call's first gate plus
`component.step_gates[s]`) at engine step `arrive_step + s`. A computed gate in no range is not run: it is past its
request's executed prefix. A registered input is there before step 0, and a run input arrives at the first step whose
gates read it.

`pack` places the values that are ever live in blocks of `block_bits`: the registered ones in canonical order from store
index 0 (the registered store), then each step's newly live values in fresh blocks after them (the memory map's appends),
each block written at its values' step and freed at their last read. That is the honest layout: `store_matches` and
`plan_matches` hold at every step.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from itertools import accumulate, chain
from typing import Mapping, Sequence

from .census import SCHEMA as CENSUS, step_of as census_step
from .layout import resolve
from .program import REGISTRATION, Program, is_input
from .types import Value

#: the deployed store's block: 4,096 bytes
BLOCK_BITS = 4096 * 8


@dataclass(frozen=True)
class Schedule:
    """The root gate ranges `[lo, hi)` each step runs, as `(lo, hi, step)`, disjoint."""

    ranges: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        ranges = tuple(sorted(self.ranges))
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError("a schedule's gate ranges are disjoint")
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "_los", [lo for lo, _, _ in ranges])

    @classmethod
    def by_calls(cls, program: Program, starts: Sequence[int]) -> "Schedule":
        """The root's Calls (its non-input nodes) split into steps at the Call indices `starts`, ascending from 0."""
        body = program.root.body
        calls = [k for k, nd in enumerate(body.nodes) if not is_input(nd.fn)]
        starts = list(starts)
        if not starts or starts[0] != 0 or any(b <= a for a, b in zip(starts, starts[1:])) or starts[-1] >= len(calls):
            raise ValueError(f"steps start at Call 0 and ascend strictly below {len(calls)} Calls, not {starts}")
        out = []
        for s, (a, b) in enumerate(zip(starts, starts[1:] + [len(calls)])):
            out.extend((body.offsets[k], body.offsets[k + 1], s) for k in calls[a:b])
        return cls(tuple(out))

    def step_of(self, g: int) -> int | None:
        """The step that runs root gate `g`, or None if no step does."""
        i = bisect.bisect_right(self._los, g) - 1  # type: ignore[attr-defined]
        return self.ranges[i][2] if i >= 0 and g < self.ranges[i][1] else None

    def call_step(self, lo: int, hi: int) -> int | None:
        """The step that runs all of root gates `[lo, hi)` (a root Call), or None if no step runs any; refused if they're split."""
        s = self.step_of(lo)
        los = self._los  # type: ignore[attr-defined]
        if s != self.step_of(hi - 1) or bisect.bisect_right(los, hi - 1) != bisect.bisect_right(los, lo):
            raise ValueError(f"the schedule splits root gates [{lo}, {hi}), a Call the census path takes whole")
        return s


@dataclass(frozen=True)
class Requirement:
    """A scheduled execution's requirement: each value's width, when it exists (`born`: -1 for a registered value, else
    the step that produces it or a run input's arrival) and its last structural reader's step (`last`), the values each
    step uses, and the run's values where `inputs` gave them."""

    width: Mapping[int, int]
    registered: frozenset[int]
    born: Mapping[int, int]
    last: Mapping[int, int]
    uses: Mapping[int, frozenset[int]]
    steps: int
    values: Mapping[int, int]
    #: per step, the candidates of its lazy primitives whose selection the run's values didn't resolve
    upper: Mapping[int, frozenset[int]] = field(default_factory=dict)

    def live(self, t: int) -> frozenset[int]:
        return frozenset(v for v, b in self.born.items() if b < t <= self.last.get(v, -1))

    def resolved(self, t: int) -> bool:
        """Every candidate from before step `t` that it may select is read anyway, so its reads are known."""
        used = self.uses.get(t, frozenset())
        return all(v in used or self.born[v] >= t for v in self.upper.get(t, ()))

    def reads(self, t: int) -> frozenset[int]:
        if not self.resolved(t):
            raise ValueError(f"step {t} selects among values from before it that the run's values don't resolve: "
                             "reads_upper bounds its reads, for planning only")
        return frozenset(v for v in self.uses.get(t, ()) if self.born[v] < t)

    def reads_upper(self, t: int) -> frozenset[int]:
        """`reads(t)` with every unresolved candidate: an upper bound, which would over-credit the audit."""
        return frozenset(v for v in chain(self.uses.get(t, ()), self.upper.get(t, ())) if self.born[v] < t)

    def bits(self, values) -> int:
        return sum(self.width[v] for v in values)

    def live_bits(self, t: int) -> int:
        return self.bits(self.live(t))

    def read_bits(self, t: int) -> int:
        return self.bits(self.reads(t))

    def read_bits_upper(self, t: int) -> int:
        return self.bits(self.reads_upper(t))

    def held(self, start: int, length: int) -> int:
        """Bit-steps held over the window's steps, as `Requirement.held` counts them."""
        return sum(self.live_bits(t) for t in range(start, start + length))

    def read(self, start: int, length: int) -> int:
        return sum(self.read_bits(t) for t in range(start, start + length))

    def to_json(self) -> dict:
        """`read_bits` where every step's reads are known, else `read_bits_upper`."""
        exact = all(self.resolved(t) for t in range(self.steps))
        key, read_bits = ("read_bits", self.read_bits) if exact else ("read_bits_upper", self.read_bits_upper)
        return {"steps": self.steps, "live_bits": [self.live_bits(t) for t in range(self.steps)],
                key: [read_bits(t) for t in range(self.steps)],
                "registered_bits": self.bits(v for v in self.registered if v in self.last)}


def requirement(program: Program, schedule: Schedule, inputs: Mapping[int, int] | None = None) -> Requirement:
    """The requirement of `program` run on `schedule`. `inputs`: each input gate's value, for the operands a lazy primitive
    uses; without them a lazy primitive uses its scalar operands and leaves its array operands as candidates."""
    circuit = program.circuit
    width: dict[int, int] = {}
    registered: set[int] = set()
    born: dict[int, int] = {}
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    uses: dict[int, set[int]] = {}
    upper: dict[int, set[int]] = {}
    values: dict[int, int] = dict(inputs or {})
    for i in range(circuit.size):
        g = circuit.gate(i)
        if getattr(g.prim, "family", None) == "Input":
            width[i] = g.prim.ret.w
            if g.prim.binding == REGISTRATION:
                registered.add(i)
                born[i] = -1
            continue
        s = schedule.step_of(i)
        if s is None:
            continue
        width[i], born[i] = g.prim.ret.w, s
        ops = g.operands
        for v in ops:
            first[v] = min(first.get(v, s), s)
            last[v] = max(last.get(v, s), s)
        lazy = getattr(g.prim, "evaluate_lazy", None)
        if inputs is None and lazy is not None:
            used, at = set(), 0
            for _, t in g.prim.params:
                (used if isinstance(t, Value) else upper.setdefault(s, set())).update(ops[at:at + t.leaves])
                at += t.leaves
        elif inputs is None:
            used = set(ops)
        elif lazy is not None:
            used = set()
            values[i] = lazy(lambda j, _o=ops: (used.add(_o[j]), values[_o[j]])[1])
        else:
            used = set(ops)
            values[i] = g.prim.evaluate(*(values[o] for o in ops))
        uses.setdefault(s, set()).update(used)
    for v in width:
        if v not in born and v in first:
            born[v] = first[v]
    steps = 1 + max((s for _, _, s in schedule.ranges), default=-1)
    return Requirement(width, frozenset(registered), born, last, {s: frozenset(u) for s, u in uses.items()}, steps,
                       values if inputs is not None else {}, {s: frozenset(u) for s, u in upper.items()})


def census_offsets(c: Mapping) -> list[int]:
    """Each root node's first gate in a census (and the gate count after the last), as `program.root.body.offsets`."""
    classes, callees = c["classes"], c["callees"]
    return [0, *accumulate(n * classes[callees[k]]["gates"] for k, n in zip(c["nodes"]["callee"], c["nodes"]["n"]))]


def census_gates(program: Program, c: Mapping, values) -> frozenset[int]:
    """The gates of census values (their producers' output leaves): `from_census`'s keys as `requirement`'s."""
    scope, v = program.circuit.root_scope, c["values"]
    return frozenset(resolve(scope, ("n", v["producer"][x], t)) for x in values for t in range(v["lo"][x], v["hi"][x]))


def census_schedule(c: Mapping) -> Schedule:
    """A census's own steps (its `ends`) as a schedule of root gate ranges."""
    off, step = census_offsets(c), census_step(c)
    return Schedule(tuple((off[k], off[k + 1], step(k)) for k in range(c["inputs"], len(off) - 1) if off[k] < off[k + 1]))


def from_census(c: Mapping, schedule: Schedule | None = None) -> Requirement:
    """The requirement of a census's Program run on `schedule` (by default the census's steps), keyed by census value: a
    value's producer and readers are root nodes, so each step must take a root Call whole. With no run values in a census,
    a selection's candidates (the census's `selects`) are unresolved, and `reads` refuses the steps they leave open."""
    if c.get("schema") != CENSUS:
        raise ValueError(f"from_census reads a {CENSUS}, not {c.get('schema')!r}")
    if c["approximate"]:
        raise ValueError("an approximate census's reads are a superset of its Program's: their requirement would over-credit")
    schedule = census_schedule(c) if schedule is None else schedule
    I, off = c["inputs"], census_offsets(c)
    step = [None] * I + [schedule.call_step(off[k], off[k + 1]) if off[k] < off[k + 1] else None for k in range(I, len(off) - 1)]
    binding = [c["classes"][c["callees"][k]]["binding"] for k in c["nodes"]["callee"][:I]]
    pools: dict[int, set[int]] = {}
    upper: dict[int, set[int]] = {}
    for k, vs in zip(c["selects"]["call"], c["selects"]["values"]):
        if step[k] is not None:
            upper.setdefault(step[k], set()).update(vs)
            for v in vs:
                pools.setdefault(v, set()).add(step[k])
    vals = c["values"]
    width = dict(enumerate(vals["bits"]))
    registered: set[int] = set()
    born: dict[int, int] = {}
    last: dict[int, int] = {}
    uses: dict[int, set[int]] = {}
    for v, (j, rs) in enumerate(zip(vals["producer"], vals["reads"])):
        read = {step[k] for k in rs} - {None}
        for s in read:
            uses.setdefault(s, set()).add(v)
        touched = read | pools.get(v, set())
        if touched:
            last[v] = max(touched)
        if j >= I:
            if step[j] is not None:
                born[v] = step[j]
        elif binding[j] == REGISTRATION:
            registered.add(v)
            born[v] = -1
        elif touched:
            born[v] = min(touched)
    steps = 1 + max((s for _, _, s in schedule.ranges), default=-1)
    return Requirement(width, frozenset(registered), born, last, {s: frozenset(u) for s, u in uses.items()}, steps, {},
                       {s: frozenset(u) for s, u in upper.items()})


@dataclass(frozen=True)
class Store:
    """The audit's store: `registered` blocks (store indices below it) and the memory map beside it, the fresh indices each
    step wrote and the indices it freed, and each block's payload as bits (where the values are known)."""

    registered: int
    written: Mapping[int, frozenset[int]]
    freed: Mapping[int, frozenset[int]]
    payload: Mapping[int, int]

    def resident(self, t: int) -> frozenset[int]:
        """The blocks resident when step `t` begins, as `Store.resident` defines them."""
        written = {j for s, js in self.written.items() if s < t for j in js}
        freed = {j for s, js in self.freed.items() if s < t for j in js}
        return frozenset((set(range(self.registered)) | written) - freed)


@dataclass(frozen=True)
class Layout:
    """Each value's block (a store index) and its bit offset there."""

    block: Mapping[int, int]
    offset: Mapping[int, int]

    def blocks(self, values) -> frozenset[int]:
        return frozenset(self.block[v] for v in values)

    def bits_in(self, req: Requirement, values, j: int) -> int:
        return req.bits(v for v in values if self.block[v] == j)

    def bits_at(self, req: Requirement, values, held) -> int:
        return req.bits(v for v in values if self.block[v] in held)

    def packs(self, req: Requirement, block_bits: int, values) -> bool:
        """Each value lies within its block, and two in one block don't overlap."""
        by_block: dict[int, list[tuple[int, int]]] = {}
        for v in values:
            if self.offset[v] + req.width[v] > block_bits:
                return False
            by_block.setdefault(self.block[v], []).append((self.offset[v], self.offset[v] + req.width[v]))
        return all(a[1] <= b[0] for ivs in by_block.values() for a, b in zip(sorted(ivs), sorted(ivs)[1:]))

    def carries(self, req: Requirement, store: Store, values) -> bool:
        return all((store.payload[self.block[v]] >> self.offset[v]) % (1 << req.width[v]) == req.values[v] for v in values)


def pack(req: Requirement, block_bits: int = BLOCK_BITS) -> tuple[Layout, Store]:
    """The honest layout and the store it implies: the registered values that are ever live packed from store index 0, then
    each step's newly live values in fresh blocks, each block written at its values' step and freed at their last read."""
    block: dict[int, int] = {}
    offset: dict[int, int] = {}
    groups = [sorted(v for v in req.registered if v in req.last)]
    groups += [sorted(v for v, b in req.born.items() if b == s and req.last.get(v, -1) > s) for s in range(req.steps)]
    nxt, written, freed, payload, n_registered = 0, {}, {}, {}, 0
    for s, group in enumerate(groups, start=-1):
        first, off = nxt, block_bits
        for v in group:
            if req.width[v] > block_bits:
                raise ValueError(f"value {v} has {req.width[v]} bits, more than a block's {block_bits}")
            if off + req.width[v] > block_bits:
                nxt, off = nxt + 1, 0
            block[v], offset[v] = nxt - 1, off
            off += req.width[v]
        if s < 0:
            n_registered = nxt
        else:
            written[s] = frozenset(range(first, nxt))
    for v, j in block.items():
        if req.values:
            payload[j] = payload.get(j, 0) | req.values[v] << offset[v]
    for j in set(block.values()):
        f = max(req.last[v] for v, k in block.items() if k == j)
        freed.setdefault(f, set()).add(j)
    return Layout(block, offset), Store(n_registered, written, {s: frozenset(js) for s, js in freed.items()}, payload)


def live_blocks(req: Requirement, layout: Layout, t: int) -> frozenset[int]:
    return layout.blocks(req.live(t))


def read_blocks(req: Requirement, layout: Layout, t: int) -> frozenset[int]:
    return layout.blocks(req.reads(t))


def store_matches(req: Requirement, layout: Layout, store: Store, block_bits: int, t: int) -> bool:
    """`StoreMatches`: the resident blocks are W's live blocks, the live values are packed, and each is in its block's
    payload (checked where the values are known)."""
    live = req.live(t)
    return (store.resident(t) == layout.blocks(live) and layout.packs(req, block_bits, live)
            and (not req.values or layout.carries(req, store, live)))


def plan_matches(req: Requirement, layout: Layout, plan, block_bits: int, t: int) -> bool:
    """`PlanMatches`: the plan is the blocks step `t` reads, and the values read are packed."""
    return frozenset(plan) == read_blocks(req, layout, t) and layout.packs(req, block_bits, req.reads(t))
