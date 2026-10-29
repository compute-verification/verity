"""What a derivation produced (`DeriveReport`) and what it keeps for a relabel (`DerivationTail`); `derive.derive_report` builds both."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verity.ml.circuits.defs import SpecializedDefinition
    from verity.ml.circuits.liveness import DeadGateReport
    from verity.ml.circuits.program import Program
    from verity_vllm.program.frontend.outcomes import Unsupported


@dataclass
class DeriveReport:
    """What a derivation produced: the program (or the refusal), its identity, provenance, and the flat
    leaf program's digest when a family identity was claimed on top of it."""
    program: Program | None
    unsupported: Unsupported | None
    provenance: dict
    digest: str | None = None
    root: str | None = None
    gates: int | None = None
    dead: DeadGateReport | None = None
    flat_digest: str | None = None
    flat_root: str | None = None
    family: dict | None = None
    flat_definition: Any = None  # the flat leaf-level SpecializedDefinition (rules' direct output)
    correspondence_digest: str | None = None  # sha256 of the runtime correspondence document (`annotations["correspondence"]`), digested apart from the Program
    descriptor: dict | None = None  # the flat Program's descriptor its digest was taken over (`program` is the flat one), for the writer

    @property
    def result(self) -> Program | Unsupported:
        return self.program if self.program is not None else self.unsupported  # type: ignore[return-value]

    @property
    def ok(self) -> bool:
        return self.program is not None

    def to_json(self) -> dict:
        return {"ok": self.ok, "digest": self.digest, "root": self.root, "gates": self.gates, "flat_digest": self.flat_digest,
                "flat_root": self.flat_root, "correspondence_digest": self.correspondence_digest,
                "unsupported": self.unsupported.to_json() if self.unsupported else None,
                "dead_gates": self.dead.to_json() if self.dead else None, "family": self.family, "provenance": self.provenance}


@dataclass(eq=False)
class DerivationTail:
    """What `derive_report(..., session=)` keeps of a successful derivation, for `relabel.rank_report` to run the tail again on a
    re-emitted body: the flat definition, the context, the provenance before the tail, and the tail's arguments."""
    flat: SpecializedDefinition
    ctx: Any                   # derive.Ctx (this module imports nothing of the derive's, which imports it)
    head: dict                 # the provenance as it stood before the tail (a shallow copy: the tail only adds top-level keys)
    unresolved: dict           # `ctx.unresolved_module_paths` as the tail read it (the tail's own stack_path calls count on)
    name: str
    mode: str
    ruleset: Any               # rules.base.RuleSet
    check_registry: bool
    rank: tuple[int, int] | None   # the derivation's (world, rank)
    family: dict | None            # the family the derivation claimed (`DeriveReport.family`)
    dead_free: bool = False        # the tail's `dead_free`: a rank written from this derivation is in the same form


__all__ = ["DeriveReport", "DerivationTail"]
