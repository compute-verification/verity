"""The backend boundary: a ``typed-obligation/v0`` object in, an accepted-or-not proof out.

The object (:class:`verity.verification.typed_obligation.TypedObligationSet`) is the backend-independent
statement of what must hold: which exact relations, over which committed words, under which roots.  This module
lowers it to the statement bytes the guest reads (``verity.verification.lowering``), hands the host the witness
(the openings, format 3 for packed leaves) and the object digest ``h_O``, and judges the outcome the way the
protocol requires (README "First vertical slice"): a proof is *accepted* only when

1. it verifies under the **approved** verifying key (:data:`verity_sp1.host.APPROVED`), not merely the one the
   host reports;
2. its public values are exactly ``h_O || h_L || 0x01`` for *this* object and *its own* lowering
   (:func:`verity.verification.binding.check_binding`): the same statement reached through another object, an
   unbound 33-byte proof, or a self-consistent proof of some other statement all fail here;
3. the guest was sound (no diagnostic verdict byte).

Each fact is reported apart in :class:`Acceptance`; ``accepted`` is their conjunction.  Nothing here knows which
frontend or runtime produced the object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.verification.binding import BindingResult, check_binding
from verity.verification.lowering import statement_bytes, statement_bytes_v3
from verity.verification.typed_obligation import TypedObligationSet, object_digest

from .host import APPROVED, ExecutionReport, GuestIdentity, Host, ProofReport, ProveMode, VerifyReport

__all__ = ["Acceptance", "Lowered", "execute_object", "lower_object", "prove_object", "verify_object"]


@dataclass(frozen=True, slots=True)
class Lowered:
    """What the guest is given for an object: the statement bytes in one format and the object digest."""

    statement: bytes
    statement_format: int
    object_digest: str


def lower_object(obj: TypedObligationSet | Mapping[str, Any], *, statement_format: int = 2) -> Lowered:
    """Lower ``obj`` to the statement bytes of ``statement_format`` (2: a position per word; 3: ranges)."""

    typed = obj if isinstance(obj, TypedObligationSet) else TypedObligationSet.from_json(obj)
    if statement_format == 2:
        statement = statement_bytes(typed)
    elif statement_format == 3:
        statement = statement_bytes_v3(typed)
    else:
        raise ValueError(f"statement_format must be 2 or 3, got {statement_format!r}")
    return Lowered(statement, statement_format, object_digest(obj))


def execute_object(
    host: Host, obj: TypedObligationSet | Mapping[str, Any], witness: bytes, *, statement_format: int = 2
) -> tuple[ExecutionReport, BindingResult]:
    """Run the guest on ``obj``'s lowering and say what its public values bind (no proof)."""

    lowered = lower_object(obj, statement_format=statement_format)
    report = host.execute(lowered.statement, witness, typed_object_digest=lowered.object_digest)
    return report, check_binding(obj, report.public_values, statement_format=statement_format)


def prove_object(
    host: Host, obj: TypedObligationSet | Mapping[str, Any], witness: bytes, out: Path, *, statement_format: int = 2,
    mode: ProveMode = "core",
) -> ProofReport:
    """One proof of ``obj``'s lowering with ``witness``, written to ``out``.  The report's execution says what the
    guest concluded; :func:`verify_object` says whether the proof is accepted."""

    lowered = lower_object(obj, statement_format=statement_format)
    return host.prove(lowered.statement, witness, out, typed_object_digest=lowered.object_digest, mode=mode)


@dataclass(frozen=True, slots=True)
class Acceptance:
    """The verifier's facts about a proof of an object, each apart; ``accepted`` is their conjunction."""

    proof_verifies: bool
    vk_hash: str
    approved_key: bool
    binding: BindingResult
    sound: bool
    verify: VerifyReport

    @property
    def accepted(self) -> bool:
        return self.proof_verifies and self.approved_key and self.binding.ok and self.binding.verdict == 1 and self.sound

    def reasons(self) -> list[str]:
        out = []
        if not self.proof_verifies:
            out.append("the proof does not verify under the guest's verifying key")
        if not self.approved_key:
            out.append(f"verifying key {self.vk_hash} is not the approved {APPROVED.vk_hash}")
        if not self.binding.ok:
            out.append(f"public values do not bind this object: {self.binding.code} {self.binding.detail}".rstrip())
        elif self.binding.verdict != 1:
            out.append(f"the guest's verdict byte is {self.binding.verdict}, not 1")
        if not self.sound:
            out.append("the guest was a diagnostic (unsound) build")
        return out


def verify_object(
    host: Host, obj: TypedObligationSet | Mapping[str, Any], proof: Path, *, statement_format: int = 2,
    approved: GuestIdentity = APPROVED,
) -> Acceptance:
    """Judge a saved proof against ``obj``: verifies, approved key, binds this object's own lowering with verdict 1."""

    lowered = lower_object(obj, statement_format=statement_format)
    report = host.verify(Path(proof), lowered.statement)
    binding = check_binding(obj, report.public_values, statement_format=statement_format) if report.ok else BindingResult(
        False, None, "the proof did not verify; its public values bind nothing"
    )
    return Acceptance(
        proof_verifies=report.ok,
        vk_hash=report.vk_hash,
        approved_key=report.ok and report.vk_hash == approved.vk_hash,
        binding=binding,
        sound=not report.unsound,
        verify=report,
    )
