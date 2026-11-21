"""Partition and width validation of a proof-block decomposition.

* `readiness_32bit` -- does every primitive gate of the Program produce <= 32 bits?  Violators are listed by
  specialization and primitive (`Const64[..]` in the Gumbel body, `Input64`).  This is a Program property separate from
  the boundary condition; 32 is a protocol parameter (`w_max`), not a language constant.

A decomposition may have several levels, coarse to fine (a `Hierarchy` of named `Level`s): a finer level's family is
`each(F, G)` over the coarser level's `F`, so it refines it by construction and no check sweeps its sets.

Nothing here touches `program_digest`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .defs import PrimitiveDefinition, SpecializedDefinition
from .parts import n_members

if TYPE_CHECKING:  # pragma: no cover
    from .program import Program

MAX_LISTED = 64  # cap on listed indices / examples inside a verdict (totals are always reported)


# ---------------------------------------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------------------------------------


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
# hierarchies: levels
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Level:
    """One level of a partition hierarchy.  `name` is its owner's label ("replay", "check", "matmul", ...), never
    interpreted here; `query` is the id (`query_ast.query_id`) of the query whose family is this level's members, or
    None for a level not bound to a Program."""
    name: str
    query: str | None = None


Hierarchy = tuple[Level, ...]
"""Levels coarse to fine; each level's family is `each(F, G)` over the level above's `F`, so its members nest by
construction."""


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
#: against its external anchor (the declared model's weights root, the client's request commitment, the challenger's
#: live-coin part), always, never sampled
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
        detail["recomputed_across"] = {"n": len(across), "gates": [list(x) for x in across[:MAX_LISTED]]}
    codes = tuple(k for k, v in problems.items() if v)
    for k in codes:
        detail[k] = {"n": len(problems[k]), "gates": [list(x) if isinstance(x, tuple) else x for x in problems[k][:MAX_LISTED]]}
    return CutVerdict(not codes, codes, detail)


__all__ = ["ReadinessViolation", "ReadinessReport", "gate_width_readiness", "readiness_32bit",
           "CutVerdict", "INPUT_UNIT", "MAX_LISTED", "RECOMPUTE_RULES", "validate_unit_cut",
           "Level", "Hierarchy"]
