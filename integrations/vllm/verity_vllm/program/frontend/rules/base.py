"""Derivation rules: versioned frontend artefacts with structural applicability predicates (design §4.1–§4.3,
GUIDANCE §43.3).

A rule is *not* a circuit operation. It is a claim of the form "this source construct, under these structural
preconditions, denotes a call of this REGISTERED definition (or is a pure reference rearrangement)". Every rule:

* is identified by `id` + `version` (recorded in provenance for every application);
* declares the graph targets it may consider (`targets()`), then decides structurally in `applies(node, ctx)`
  — operand dtypes/shapes, sharing, mutation, side inputs — never by op name alone. `applies` returns `True`
  or a *string* explaining why it does not apply (the strings become the `Unsupported` hint);
* emits only through the derivation context (`ctx.call(...)`), i.e. calls of registered / vocabulary
  definitions, so the same guard, codec and digest apply as for authored programs.

Rule kinds (design §4.1): `ModuleRule` (module class / call-graph pattern -> registered composite), `OpRule`
(one functional op -> registered kind), `InPlaceRule` (a mutating op -> pure value + explicit rebinding of the
mutated operand, with alias invalidation), `StateRule` (cache writes + reads -> explicit values with producers),
`ViewRule` (reference rearrangement, no gates), `FamilyRule` (whole-program hierarchy claim verified against a
registered program family's expanded body).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from verity_vllm.program.frontend.derive import Ctx

RULE_KINDS = ("view", "op-binding", "in-place-binding", "state", "module-binding", "module-weights", "program-family")


@dataclass(frozen=True)
class RuleInfo:
    """The versioned record of a rule (what provenance carries)."""
    id: str
    version: int
    kind: str
    applies_to: str
    preconditions: tuple[str, ...]
    evidence_domain: str

    def __post_init__(self) -> None:
        if self.kind not in RULE_KINDS:
            raise ValueError(f"rule kind {self.kind!r} not in {RULE_KINDS}")

    @property
    def rule_id(self) -> str:
        return f"{self.id}_v{self.version}"

    def to_json(self) -> dict:
        return {"rule": self.rule_id, "version": self.version, "kind": self.kind, "applies_to": self.applies_to,
                "preconditions": list(self.preconditions), "evidence_domain": self.evidence_domain}


class Rule:
    """Base class. Subclasses set `info` and implement `targets`, `applies`, `emit`."""

    info: RuleInfo
    modes: frozenset[str] | None = None  # None: any mode

    def targets(self) -> Iterable[Any]:
        """Graph targets (OpOverload / callables) this rule may consider; `ANY` for a catch-all."""
        return ()

    def applies(self, node, ctx: "Ctx") -> bool | str:
        raise NotImplementedError

    def emit(self, node, ctx: "Ctx") -> None:
        raise NotImplementedError

    def declare_inputs(self, ep, module, mode: str) -> list[tuple[str, Any, dict]]:
        """Extra prescribed inputs (fields of the `weights` record) this rule needs for the exported graph `ep`, as
        `(field name, type, derivation)`: a *derived table* — a tensor the source computes in forward from a buffer and
        static structure (HF's RoPE cos/sin from `inv_freq` × positions) that the program takes as an Input with its
        derivation recorded (design §1.3: the same qualification class as vLLM's `cos_sin_cache`). Default: none."""
        return []

    def registered_tables(self, ep, mode: str) -> tuple[str, ...]:
        """The tables a program brings (`verity.ml.definitions.registry.tables`) this rule reads by index from a registered
        parameter under `registered_values`, for the exported graph `ep`: each a parameter after the `values`. Default: none."""
        return ()

    def run_inputs(self, ep, mode: str) -> bool:
        """Whether this rule reads values the served run decides (`Ctx.run_value`) for the exported graph `ep`: values no circuit
        computes, which the Program takes as fields of its `run` parameter and checks in gates (`Ctx.check`), as the top-k order of
        a DSA row past TOPK. Default: none."""
        return False

    @property
    def rule_id(self) -> str:
        return self.info.rule_id

    def __repr__(self) -> str:
        return f"<rule {self.rule_id}>"


class ViewRule(Rule):
    """Reference rearrangement (no gates)."""


class OpRule(Rule):
    """One functional op -> a call of a registered kind."""


class InPlaceRule(Rule):
    """A mutating op -> a pure call plus rebinding of the mutated operands (alias invalidation is the ctx's)."""


class StateRule(Rule):
    """Cache writes + reads -> explicit values with producers."""


class ModuleRule(Rule):
    """A module-class / call-graph pattern -> registered composite."""


class FamilyRule(Rule):
    """A whole-program hierarchy claim (verified by expanded-body isomorphism, `rules/family.py`)."""


ANY = object()


@dataclass
class RuleSet:
    """An ordered, versioned collection of rules. `version` is a content digest over (rule id, version) pairs so
    provenance pins exactly which rule set derived a program."""
    name: str
    rules: list[Rule] = field(default_factory=list)
    families: list[Any] = field(default_factory=list)  # FamilyRule instances (whole-program claims)

    @property
    def version(self) -> str:
        import hashlib
        ids = sorted(r.rule_id for r in self.rules) + sorted(f.rule_id for f in self.families)
        return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]

    def for_target(self, target, mode: str) -> list[Rule]:
        out = []
        for r in self.rules:
            if r.modes is not None and mode not in r.modes:
                continue
            ts = r.targets()
            if ts is ANY or target in ts:
                out.append(r)
        return out

    def extended(self, name: str, rules: Iterable[Rule] = (), families: Iterable[Any] = ()) -> "RuleSet":
        return RuleSet(name, list(self.rules) + list(rules), list(self.families) + list(families))

    def to_json(self) -> dict:
        return {"name": self.name, "version": self.version,
                "rules": [r.info.to_json() for r in self.rules] + [f.info.to_json() for f in self.families]}


__all__ = ["RuleInfo", "Rule", "ViewRule", "OpRule", "InPlaceRule", "StateRule", "ModuleRule", "FamilyRule", "RuleSet", "ANY",
           "RULE_KINDS"]
