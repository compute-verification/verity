"""Annotations, verification plans and obligations over IR programs (IR §1, §D, §E), plus the one lowering adapter.

Moved from ``verity.ir.plan``: a plan selects and rates gates of a Program for checking; it is verification
policy, not computation (README "Verification is separate from computation semantics").

Prescribed inputs are sampled gates (IR §E): `Input[k] : () -> Value<w_k>` is an ordinary one-output gate of
the fault model. It belongs to a verification unit under the selected plan and is sampled at that plan's
rate like any other gate. WB's "model binding" (every committed weight leaf rebuilt from the registered
checkpoint) is *a plan that gives the weight `Input` gates rate 1*; a plan may instead sample them. `Bound`
charges an unsampled `Input` gate exactly as it charges any other missed gate. There are no trusted sources.

Vocabulary: everything public here uses the IR §D names. Backend-internal names (`KindProgram`, `port`,
`local`, `slot`, `copy`, `kind`, tree table) appear in exactly one place in this package: the body of
`lower()`, which emits the legacy `veritor.protocol.proofs` / `typed-obligation/v0` lowering section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Sequence

from verity.ir.layout import Circuit, FunctionCall, Gate, Subcircuit
from verity.ir.program import Program


@dataclass
class Annotations:
    """Planner hints attached to a program. Never part of the program digest, never a semantic input."""

    entries: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return dict(self.entries)


@dataclass(frozen=True)
class GateClass:
    """A named class of gates selected by a predicate on the canonical circuit (e.g. `Input` gates of the
    weight parameters, `TcDot16` gates, everything else)."""

    name: str
    selector: Callable[[Gate], bool] = field(repr=False, compare=False)
    description: str = ""


@dataclass(frozen=True)
class UnconditionalObligation:
    """A proof obligation required on EVERY audit, outside both sampling draws (GUIDANCE §31.3).

    The prover supplies a proof that every gate the class selects satisfies its relation on the committed
    boundary values, and the verifier checks that proof on every audit, before any sampling -- exactly as it
    checks the public inputs.  A transcript violating a covered gate is rejected with probability 1 up to
    the proof system's soundness error, so covered fault classes have survival 0 in the acceptance law.
    This is NOT a sampling rate of 1: under the two-stage draw a faulty replay unit with h > 0 faulty units
    survives with probability 1 - q + q(1-s)^h, which s = 1 leaves at 1 - q.  `statement` names the relation
    the proof establishes (e.g. the authenticated boundary relation of an inactive region)."""

    gate_class: GateClass
    statement: str = ""

    def to_json(self) -> dict:
        return {"gate_class": self.gate_class.name, "statement": self.statement,
                "description": self.gate_class.description}


@dataclass
class VerificationPlan:
    """Per-gate-class sampling rates over the canonical circuit, plus unconditional proof obligations.

    Rates are exact rationals in [0, 1]; rate 1
    means every gate of that class is checked (the WB model-binding choice for weight `Input` gates), rate
    r < 1 means each gate of the class is sampled with probability r. The first matching class wins; the
    `default_rate` applies to gates in no class. `Input` gates are not special: they get a rate through the
    same mechanism.

    `unconditional` (schema v1, additive) lists proof obligations checked on every audit outside both
    draws; see :class:`UnconditionalObligation`.  A covered gate keeps its sampling rate (sampling a proved
    gate is harmless); the obligation is what removes its fault class from the admissible error sets.  A
    plan without obligations serializes byte-identically to the pre-extension (v0) plan."""

    classes: list[tuple[GateClass, Fraction]]
    default_rate: Fraction
    name: str = ""
    unconditional: tuple[UnconditionalObligation, ...] = ()

    def rate_of(self, gate: Gate) -> Fraction:
        for cls, r in self.classes:
            if cls.selector(gate):
                return r
        return self.default_rate

    def covered_by_proof(self, gate: Gate) -> bool:
        """Whether some unconditional obligation covers `gate` (checked on every audit)."""
        return any(ob.gate_class.selector(gate) for ob in self.unconditional)

    def to_json(self) -> dict:
        d = {
            "name": self.name,
            "classes": [{"gate_class": c.name, "rate": str(r), "description": c.description} for c, r in self.classes],
            "default_rate": str(self.default_rate),
        }
        if self.unconditional:
            # Version bump is explicit and additive: a plan without obligations is byte-identical to v0.
            d["schema"] = "verity-ir/verification-plan/v1"
            d["unconditional"] = [ob.to_json() for ob in self.unconditional]
        return d

    def expected_checked(self, circuit: Circuit, sub: Subcircuit) -> Fraction:
        """Expected number of checked gates in `sub` under this plan (explicit enumeration)."""
        return sum((self.rate_of(g) for g in sub), Fraction(0))


def input_gate_class(program: Program, param_index: int, name: str, description: str = "") -> GateClass:
    """The class of `Input` gates that carry prescribed parameter `param_index` of `program`."""
    coll = program.input_coll(param_index)
    r = coll.refs
    lo = r[0][2] if r.count else 0
    hi = r[r.count - 1][2] + 1 if r.count else 0

    def sel(g: Gate, lo=lo, hi=hi) -> bool:
        return getattr(g.prim, "family", None) == "Input" and lo <= g.index < hi

    return GateClass(name, sel, description)


def weight_binding_plan(program: Program, weight_param_indices: Sequence[int], other_rate: Fraction,
                        weight_rate: Fraction = Fraction(1)) -> VerificationPlan:
    """WB's model binding expressed as a plan: weight `Input` gates at `weight_rate` (1 = every committed
    weight leaf word is checked against the registered checkpoint), all other gates at `other_rate`. Pass a
    `weight_rate` below 1 to sample the weights instead."""
    classes = [(input_gate_class(program, i, f"Input[param {i}]", "weight Input gates"), weight_rate)
               for i in weight_param_indices]
    return VerificationPlan(classes, other_rate, name="weight-binding" if weight_rate == 1 else "weight-sampled")


@dataclass
class AuthenticationLocation:
    """Where a value's encoding is physically authenticated: commitment/domain + leaf/range/offset. This is
    the second namespace (IR §C); it is never a component of a gate output."""

    commitment: str
    domain: str
    leaf: str
    offset: int
    width: int

    def to_json(self) -> dict:
        return {"commitment": self.commitment, "domain": self.domain, "leaf": self.leaf, "offset": self.offset,
                "width": self.width}


@dataclass
class Obligation:
    """One sampled verification unit: a subcircuit of the canonical circuit (here always a `FunctionCall` or
    `Region`), its boundary values with their authentication locations, and the plan that selected it."""

    subcircuit: Subcircuit
    plan: VerificationPlan
    boundary_in: dict[int, AuthenticationLocation | None]  # gate index -> location (None: opened directly)
    boundary_out: dict[int, AuthenticationLocation | None]
    statement: str = ""

    def to_json(self) -> dict:
        d: dict[str, Any] = {
            "schema": "verity-ir/obligation/v0",
            "subcircuit": {"gates": list(self.subcircuit.gate_indices())} if not hasattr(self.subcircuit, "lo")
            else {"region": [self.subcircuit.lo, self.subcircuit.hi]},  # type: ignore[attr-defined]
            "in": [{"gate": g, "location": None if loc is None else loc.to_json()} for g, loc in sorted(self.boundary_in.items())],
            "out": [{"gate": g, "location": None if loc is None else loc.to_json()} for g, loc in sorted(self.boundary_out.items())],
            "plan": self.plan.name,
            "statement": self.statement,
        }
        if isinstance(self.subcircuit, FunctionCall):
            d["function_call"] = {"id": self.subcircuit.id, "args": list(self.subcircuit.args),
                                  "returns": list(self.subcircuit.returns)}
        return d


def lower(sub: Subcircuit) -> dict:
    """Lowering adapter: a supported subcircuit -> the legacy backend's local straight-line program (the
    `typed-obligation/v0` lowering section). This is the ONLY place in `verity.ir` where backend-internal
    names appear. Semantics: gates read `port k` (k-th distinct boundary-in value, ascending producing-gate
    index) or `local i` (i-th gate of the subcircuit in canonical order); `outputs` are the local indices of
    the subcircuit's `out` boundary. An `Input` gate inside the subcircuit lowers to a `port` whose expected
    value is the prescribed input (the sampled `Input[k]` obligation: authenticated supplied value == x[k])."""
    gates = list(sub)
    index_of = {g.index: i for i, g in enumerate(gates)}
    ports = list(sub.in_)
    port_of = {g: k for k, g in enumerate(ports)}
    lowered_gates = []
    input_ports = []
    for i, g in enumerate(gates):
        if getattr(g.prim, "family", None) == "Input":
            # prescribed-input equation: the backend authenticates a supplied value and pins it to x[k]
            input_ports.append({"local": i, "gate": g.index, "expected": "prescribed x[k]", "width": g.prim.ret.w})
            lowered_gates.append({"kind": g.prim.id, "operands": [], "width": g.prim.ret.w, "input_equation": True})
            continue
        ops = []
        for o in g.operands:
            if o in index_of:
                ops.append({"local": index_of[o]})
            else:
                ops.append({"port": port_of[o]})
        lowered_gates.append({"kind": g.prim.id, "operands": ops, "width": g.prim.ret.w})
    return {
        "schema": "typed-obligation/v0/lowering",
        "KindProgram": {
            "ports": [{"port": k, "gate": g, "width": None} for k, g in enumerate(ports)],
            "gates": lowered_gates,
            "outputs": [index_of[o] for o in sub.out],
            "input_equations": input_ports,
        },
    }
