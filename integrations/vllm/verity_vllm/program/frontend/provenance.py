"""Provenance records of an export.

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

* `registry_version()` — the fixed interpretation of the primitive vocabulary a program refers to. The program
  digest records primitives by id and signature and the codec retrieves their evaluators from the registry at
  decode time, so the digest alone does not pin them (§8.1). This record does: the ids, signatures and
  conformance statuses of every registered primitive, the ids and static parameters of every registered
  composite, and the SHA-256 of the registry sources that define the evaluators. It is part of every export
  report's provenance; conformance evidence should be keyed by (program digest, specialisation, this version).
* `DerivationRule` / `RuleApplication` — derivation rules are versioned frontend artefacts recorded in
  provenance, not circuit operations (§8.4): rule id/version, the kind of rule (`hierarchy-around-unchanged-gates`
  adds structure around gates the export already had; `region-replacement` replaces an exported region with a
  separately specified composite), the source and configuration it applied to, the structural preconditions
  it checked, its stated evidence domain (tested coverage, not correctness), and the resulting program identity.
  A rule checks structural preconditions; conformance tests or proofs establish the claimed semantic
  correspondence over a stated domain. Derivation mode is not implemented beyond the `nn.Module` route
  (parameters/buffers -> a trailing `weights` record of `Input` gates), which is the one rule the frontend
  applies today; no derivation scope is added here.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition

RULE_KINDS = ("hierarchy-around-unchanged-gates", "region-replacement")


def registry_version(registry=REGISTRY) -> dict:
    """The primitive-vocabulary version: ids + signatures + conformance of every primitive, ids + statics of
    every composite, and the SHA-256 of the registry sources; `digest` is the SHA-256 of that record."""
    from verity.ml.definitions.registry import b1, prims  # registration
    # The record is computed over the DECLARED vocabulary (module-level definitions of `registry.prims` / `registry.b1`),
    # not over `registry.defs` at call time: after a program is built the registry also holds shape-instantiated
    # primitives (`GatherBf16x49152_v1`, `Const32[…]`, `Input16_v1`) and the digest would depend on process state
    # (R-CAPCORR-1).
    prim_rows = []
    comp_rows = []
    seen: set[str] = set()
    for mod, cls in ((prims, PrimitiveDefinition), (b1, CompositeDefinition)):
        for _n, d in inspect.getmembers(mod):
            if not isinstance(d, cls) or d.id in seen:
                continue
            seen.add(d.id)
            if isinstance(d, PrimitiveDefinition):
                if getattr(d, "family", None) in ("Const", "Input"):
                    continue  # families with an unbounded id space; their meaning is fixed by the constructor
                prim_rows.append([d.id, [[n, t.to_json()] for n, t in d.params], d.ret.to_json(), d.conformance])
            else:
                comp_rows.append([d.id, list(d.statics), d.conformance])
    prim_rows.sort(key=lambda r: r[0])
    comp_rows.sort(key=lambda r: r[0])
    sources = {}
    for mod in (prims, b1):
        src = inspect.getsource(mod).encode()
        sources[mod.__name__] = hashlib.sha256(src).hexdigest()
    rec = {"schema": "verity-ir/registry-version/v0", "primitives": prim_rows, "composites": comp_rows, "sources": sources}
    digest = hashlib.sha256(canonical_json(rec)).hexdigest()
    return {"schema": rec["schema"], "digest": digest, "primitives": len(prim_rows), "composites": len(comp_rows), "sources": sources}


@dataclass(frozen=True)
class DerivationRule:
    """A versioned derivation rule (a frontend artefact, recorded in provenance)."""
    id: str
    version: int
    kind: str  # one of RULE_KINDS
    applies_to: str  # the source construct (module class, op pattern, side state)
    preconditions: tuple[str, ...]  # the structural checks the rule performs
    evidence_domain: str  # over which domain the semantic claim is established, and how (tested coverage)

    def __post_init__(self) -> None:
        if self.kind not in RULE_KINDS:
            raise ValueError(f"rule kind {self.kind!r} not in {RULE_KINDS}")

    @property
    def rule_id(self) -> str:
        return f"{self.id}_v{self.version}"

    def to_json(self) -> dict:
        return {"rule": self.rule_id, "kind": self.kind, "applies_to": self.applies_to,
                "preconditions": list(self.preconditions), "evidence_domain": self.evidence_domain}


@dataclass
class RuleApplication:
    """One application of a rule during an export: what it applied to, what it checked, what it produced."""
    rule: DerivationRule
    source: dict[str, Any]  # source + configuration (class, revision, statics, ...)
    preconditions_checked: dict[str, bool]
    where: str  # the function / module path
    result: dict[str, Any] = field(default_factory=dict)  # resulting program identity (filled after export)

    def to_json(self) -> dict:
        return {**self.rule.to_json(), "source": dict(self.source), "preconditions_checked": dict(self.preconditions_checked),
                "where": self.where, "result": dict(self.result)}


# The one rule the frontend applies today: an `nn.Module`'s parameters and buffers become a trailing `weights`
# record parameter (prescribed `Input` gates); the module's forward must already be authored over `ops`/`batch`/
# `scan`, so no gate is replaced — hierarchy (the weights record) is added around unchanged gates.
MODULE_WEIGHTS_RULE = DerivationRule(
    "ModuleWeights", 1, "hierarchy-around-unchanged-gates",
    applies_to="torch.nn.Module (named_parameters + named_buffers)",
    preconditions=("no lifted parameters/buffers/constants remain in the exported graph (E_HIDDEN_STATE otherwise)",
                   "no buffer or input mutation recorded by export (E_MUTATION otherwise)",
                   "every parameter/buffer dtype carries a Value<w> width"),
    evidence_domain="structural only: the rule moves host state to explicit inputs and asserts no semantics; "
                    "tested on the frontend's module examples (tests/program/test_torch_frontend.py)")

__all__ = ["registry_version", "DerivationRule", "RuleApplication", "MODULE_WEIGHTS_RULE", "RULE_KINDS"]
