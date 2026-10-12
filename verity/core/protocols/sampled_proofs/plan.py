"""Verification plans and obligations over IR programs (IR §1, §D, §E), under the two-stage law.

Moved from ``verity.verification.plan``, now ``verity_sp1.proofs`` (decision 41).  A plan selects and samples gates of a
Program for checking; it is verification policy, not computation (README "Protocols").  Its
per-class rates are now `Stage`s of the two-stage law (decision 43, `verity.core.protocols.sampled_proofs.two_stage`):
a gate-level plan makes every gate its own replay unit, which is the law's Bernoulli case, and `units` hands those
replay units to `two_stage.select_replay_units`.

Prescribed inputs are sampled gates (IR §E): `Input[k] : () -> Value<w_k>` is an ordinary one-output gate of the
fault model.  It belongs to a proof unit under the selected plan and is sampled like any other gate.  WB's
"model binding" (every committed weight leaf rebuilt from the registered checkpoint) is *a plan whose weight `Input`
gates have p = 1*; a plan may instead sample them.  There are no trusted sources.

Boundaries come from the caller or from a call's declared interface (`args`, `returns`), never from
`FunctionCall.out`, which rescans the circuit on every call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from verity.ml.circuits.layout import FunctionCall, Gate, GateSelection
from verity.ml.circuits.program import Program
from verity.core.protocols.sampled_proofs.two_stage import ReplayUnit, Stage, TwoStageLaw

DEFAULT_CLASS = "default"


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
    This is NOT p = 1: under the two-stage draw a faulty replay unit with h > 0 faulty units survives with
    probability 1 - q + q(1-s)^h, which s = 1 leaves at 1 - q.  `statement` names the relation the proof
    establishes (e.g. the authenticated boundary relation of an inactive region)."""

    gate_class: GateClass
    statement: str = ""

    def to_json(self) -> dict:
        return {"gate_class": self.gate_class.name, "statement": self.statement,
                "description": self.gate_class.description}


@dataclass
class VerificationPlan:
    """A `Stage` per gate class over the canonical circuit, plus unconditional proof obligations.

    The first matching class wins; `default` applies to gates in no class.  `Input` gates are not special: they get
    a stage through the same mechanism.  A covered gate keeps its stage (sampling a proved gate is harmless); the
    obligation is what removes its fault class from the admissible error sets."""

    classes: list[tuple[GateClass, Stage]]
    default: Stage
    name: str = ""
    unconditional: tuple[UnconditionalObligation, ...] = ()

    def __post_init__(self) -> None:
        names = [c.name for c, _ in self.classes]
        if len(set(names)) != len(names) or DEFAULT_CLASS in names:
            raise ValueError(f"gate-class names are distinct and not {DEFAULT_CLASS!r}")

    def class_of(self, gate: Gate) -> str:
        for cls, _ in self.classes:
            if cls.selector(gate):
                return cls.name
        return DEFAULT_CLASS

    @property
    def law(self) -> TwoStageLaw:
        return TwoStageLaw({**{c.name: s for c, s in self.classes}, DEFAULT_CLASS: self.default})

    def stage_of(self, gate: Gate) -> Stage:
        for cls, stage in self.classes:
            if cls.selector(gate):
                return stage
        return self.default

    def check_probability(self, gate: Gate) -> Fraction:
        """The probability `gate`, as its own replay unit, is checked."""
        stage = self.stage_of(gate)
        return Fraction(stage.p) * stage.checked(1)

    def units(self, gates: Iterable[Gate]) -> list[ReplayUnit]:
        """Each gate as its own replay unit, addressed by its gate index."""
        return [ReplayUnit(g.index, self.class_of(g), 1) for g in gates]

    def covered(self) -> frozenset[str]:
        """The gate classes proved on every audit: an `IntegrityProfile`'s covered families."""
        return frozenset(ob.gate_class.name for ob in self.unconditional)

    def covered_by_proof(self, gate: Gate) -> bool:
        """Whether some unconditional obligation covers `gate` (checked on every audit)."""
        return any(ob.gate_class.selector(gate) for ob in self.unconditional)

    def expected_checked(self, sub: GateSelection) -> Fraction:
        """Expected number of checked gates in `sub` under this plan (explicit enumeration)."""
        return sum((self.check_probability(g) for g in sub), Fraction(0))

    def to_json(self) -> dict:
        d: dict[str, Any] = {
            "schema": "verity-sampled-proofs/verification-plan/v2",
            "name": self.name,
            "classes": [{"gate_class": c.name, "stage": s.to_json(), "description": c.description} for c, s in self.classes],
            "default": self.default.to_json(),
        }
        if self.unconditional:
            d["unconditional"] = [ob.to_json() for ob in self.unconditional]
        return d


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
    """WB's model binding expressed as a plan: weight `Input` gates at p = `weight_rate` (1 = every committed weight
    leaf word is checked against the registered checkpoint), all other gates at p = `other_rate`, each gate its own
    replay unit.  Pass a `weight_rate` below 1 to sample the weights instead."""
    classes = [(input_gate_class(program, i, f"Input[param {i}]", "weight Input gates"), Stage(Fraction(weight_rate), 1))
               for i in weight_param_indices]
    return VerificationPlan(classes, Stage(Fraction(other_rate), 1),
                            name="weight-binding" if weight_rate == 1 else "weight-sampled")


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
    """One sampled proof unit: a gate selection of the canonical circuit (here always a `FunctionCall` or
    `Region`), its boundary values with their authentication locations, and the plan that selected it."""

    selection: GateSelection
    plan: VerificationPlan
    boundary_in: dict[int, AuthenticationLocation | None]  # gate index -> location (None: opened directly)
    boundary_out: dict[int, AuthenticationLocation | None]
    statement: str = ""

    @classmethod
    def of_call(cls, call: FunctionCall, plan: VerificationPlan, statement: str = "") -> Obligation:
        """An obligation over a call's declared interface, with every boundary value opened directly."""
        return cls(call, plan, dict.fromkeys(call.args), dict.fromkeys(call.returns), statement)

    def to_json(self) -> dict:
        d: dict[str, Any] = {
            "schema": "verity-ir/obligation/v0",
            "subcircuit": {"gates": list(self.selection.gate_indices())} if not hasattr(self.selection, "lo")
            else {"region": [self.selection.lo, self.selection.hi]},  # type: ignore[attr-defined]
            "in": [{"gate": g, "location": None if loc is None else loc.to_json()} for g, loc in sorted(self.boundary_in.items())],
            "out": [{"gate": g, "location": None if loc is None else loc.to_json()} for g, loc in sorted(self.boundary_out.items())],
            "plan": self.plan.name,
            "statement": self.statement,
        }
        if isinstance(self.selection, FunctionCall):
            d["function_call"] = {"id": self.selection.id, "args": list(self.selection.args),
                                  "returns": list(self.selection.returns)}
        return d


__all__ = ["DEFAULT_CLASS", "AuthenticationLocation", "GateClass", "Obligation", "UnconditionalObligation",
           "VerificationPlan", "input_gate_class", "weight_binding_plan"]
