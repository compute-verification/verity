"""The Python column of the ``typed-obligation/v0`` conformance suite: re-derive every checked field of a vector.

A vector (``fixtures/typed-obligation-v0/*.json``) records, for one object, what every interpreter must agree
on: the canonical bytes and object digest, the validation verdict (or its code), the lowering's backend data,
the statement bytes and digest, the bound public values, and -- when the record carries values or a witness --
the relation verdict and the reference checker's verdict.  :func:`interpret` recomputes them from the object
alone; ``backends/sp1/common/tests/typed_obligation_v0.rs`` is the Rust column over the same files.

Lifted from veritor ``interface/vectors.py`` (eb865ce2), whose generator (the vectors' *producer*) stays in
veritor for now: this repository consumes the recorded vectors and does not yet regenerate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .binding import bound_public_values, check_binding
from .codes import Reject
from .lowering import check_relations, lowering_json, to_statement
from .transparent import TransparentBackend
from .typed_obligation import (
    InterfaceReject,
    TypedObligationSet,
    canonical_bytes,
    family_gate_set,
    object_digest,
    subcircuit_well_formed,
    validate,
)
from .wire import encode_statement, statement_digest

__all__ = ["FOREIGN_PREFIXES", "Interpretation", "interpret", "subcircuits_well_formed"]

#: Vector files written by other generators (the serving exporter, the identity fixture).
FOREIGN_PREFIXES = ("stage2--", "identity--")


def subcircuits_well_formed(object_json: dict[str, Any]) -> list[bool] | None:
    """Per subcircuit of ``object_json`` (in order), the Lean-scope verdict :func:`subcircuit_well_formed`;
    ``None`` when the object's family does not resolve or its subcircuits do not parse."""

    try:
        obj = TypedObligationSet.from_json(object_json)
    except InterfaceReject:
        return None
    gate_set = family_gate_set(obj.family.id, obj.family.width)
    if gate_set is None or gate_set.digest != obj.family.digest:
        return None
    return [subcircuit_well_formed(s, gate_set) for s in obj.subcircuits]


@dataclass
class Interpretation:
    verdict: str
    code: str | None = None
    lowering: dict[str, Any] | None = None
    statement: str | None = None
    statement_digest: str | None = None
    relations: str | None = None
    witness_verdict: str | None = None
    canonical_ok: bool = True
    digest_ok: bool = True
    subcircuits_well_formed: list[bool] | None = None
    #: The correct prover's 65 public bytes for the object (hex), and the outcome of
    #: :func:`check_binding` on the record's ``binding.public_values``: ``"accept"`` or the binding code.
    bound_public_values: str | None = None
    binding: str | None = None


def interpret(record: dict[str, Any]) -> Interpretation:
    """Re-derive every checked field of a vector from its ``object`` alone."""

    data = record["object"]
    canonical = canonical_bytes(data)
    out = Interpretation("accept")
    out.canonical_ok = canonical.decode("utf-8") == record["canonical"]
    out.digest_ok = object_digest(data) == record["digest"]
    out.subcircuits_well_formed = subcircuits_well_formed(data)
    try:
        obj = TypedObligationSet.from_json(data)
        gate_set = validate(obj)
    except InterfaceReject as error:
        out.verdict, out.code = "reject", str(error.code)
        return out
    out.lowering = lowering_json(obj)
    statement = to_statement(obj)
    raw = encode_statement(statement)
    out.statement = raw.hex()
    out.statement_digest = statement_digest(raw).hex()
    out.bound_public_values = bound_public_values(data).hex()
    if "binding" in record:
        result = check_binding(data, bytes.fromhex(record["binding"]["public_values"]))
        out.binding = "accept" if result.ok and result.verdict == 1 else str(result.code)
    if "values" in record:
        out.relations = "accept" if check_relations(obj, record["values"], gate_set) else "reject"
    if "witness" in record:
        try:
            TransparentBackend(gate_set).verify(statement, bytes.fromhex(record["witness"]))
            out.witness_verdict = "accept"
        except Reject as error:
            out.witness_verdict = f"reject:{error.code.name}"
    return out
