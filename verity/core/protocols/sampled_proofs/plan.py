"""Verification plans and obligations over IR programs (IR §1, §D, §E).

A plan selects and samples gates of a Program for checking; it is verification policy, not computation (README
"Protocols").  Each gate class has a rate: the
probability a gate of the class is checked, each gate sampled independently.

Prescribed inputs are sampled gates (IR §E): `Input[k] : () -> Value<w_k>` is an ordinary one-output gate of the
fault model.  It belongs to a proof block under the selected plan and is sampled like any other gate.  WB's
"model binding" (every committed weight leaf rebuilt from the registered checkpoint) is *a plan whose weight `Input`
gates have p = 1*; a plan may instead sample them.  There are no trusted sources.

Boundaries come from the caller or from a call's declared interface (`args`, `returns`), never from
`FunctionCall.out`, which rescans the circuit on every call.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from verity.ml.circuits.layout import FunctionCall, Gate, GateSelection
from verity.ml.circuits.program import Program

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
    """A proof obligation required on EVERY audit, outside the sampling draw (GUIDANCE §31.3).

    The prover supplies a proof that every gate the class selects satisfies its relation on the committed
    boundary values, and the verifier checks that proof on every audit, before any sampling -- exactly as it
    checks the public inputs.  A trace violating a covered gate is rejected with probability 1 up to
    the proof system's soundness error, so covered fault classes have acceptance probability 0.
    This is NOT p = 1: a class's rate is a sample, and the obligation holds outside it.  `statement` names the relation
    the proof establishes (e.g. the authenticated boundary relation of an inactive region)."""

    gate_class: GateClass
    statement: str = ""

    def to_json(self) -> dict:
        return {"gate_class": self.gate_class.name, "statement": self.statement,
                "description": self.gate_class.description}


@dataclass
class VerificationPlan:
    """A rate per gate class over the canonical circuit, plus unconditional proof obligations.

    The first matching class wins; `default` applies to gates in no class.  `Input` gates are not special: they get
    a rate through the same mechanism.  A covered gate keeps its rate (sampling a proved gate is harmless); the
    obligation is what removes its fault class from the admissible error sets."""

    classes: list[tuple[GateClass, Fraction]]
    default: Fraction
    name: str = ""
    unconditional: tuple[UnconditionalObligation, ...] = ()

    def __post_init__(self) -> None:
        names = [c.name for c, _ in self.classes]
        if len(set(names)) != len(names) or DEFAULT_CLASS in names:
            raise ValueError(f"gate-class names are distinct and not {DEFAULT_CLASS!r}")
        if not all(0 <= Fraction(r) <= 1 for r in [self.default, *(r for _, r in self.classes)]):
            raise ValueError("a rate is a probability: 0 <= p <= 1")

    def class_of(self, gate: Gate) -> str:
        for cls, _ in self.classes:
            if cls.selector(gate):
                return cls.name
        return DEFAULT_CLASS

    def check_probability(self, gate: Gate) -> Fraction:
        """The probability `gate` is checked: its class's rate."""
        for cls, rate in self.classes:
            if cls.selector(gate):
                return Fraction(rate)
        return Fraction(self.default)

    def covered(self) -> frozenset[str]:
        """The gate classes proved on every audit: an `IntegrityProfile`'s covered families."""
        return frozenset(ob.gate_class.name for ob in self.unconditional)

    def expected_checked(self, sub: GateSelection) -> Fraction:
        """Expected number of checked gates in `sub` under this plan (explicit enumeration)."""
        return sum((self.check_probability(g) for g in sub), Fraction(0))

    def to_json(self) -> dict:
        d: dict[str, Any] = {
            "schema": "verity-sampled-proofs/verification-plan/v3",
            "name": self.name,
            "classes": [{"gate_class": c.name, "rate": str(Fraction(r)), "description": c.description}
                        for c, r in self.classes],
            "default": str(Fraction(self.default)),
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
    leaf word is checked against the registered checkpoint), all other gates at p = `other_rate`.  Pass a
    `weight_rate` below 1 to sample the weights instead."""
    classes = [(input_gate_class(program, i, f"Input[param {i}]", "weight Input gates"), Fraction(weight_rate))
               for i in weight_param_indices]
    return VerificationPlan(classes, Fraction(other_rate),
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
    """One sampled proof block: a gate selection of the canonical circuit (here always a `FunctionCall` or
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
