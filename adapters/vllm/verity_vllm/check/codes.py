"""Verdict codes: why a verifier accepted or rejected (promoted from veritor ``protocol/messages.py``).

The code strings are wire constants: the phase-equivalence vectors (``fixtures/proof-format-v3``) name them and
the SP1 kernel's ``CheckError`` maps onto them.  The full veritor enum is kept so those vectors decode.
"""

from __future__ import annotations

from enum import StrEnum

from verity.errors import ProtocolError

__all__ = ["Reject", "VerificationCode"]


class VerificationCode(StrEnum):
    """Why the verifier accepted or rejected."""

    ACCEPTED = "accepted"
    EXPECTATION_MISMATCH = "expectation_mismatch"
    POLICY_REJECTED = "policy_rejected"
    WORK_BUDGET_EXCEEDED = "work_budget_exceeded"
    INVALID_PHASE = "invalid_phase"
    INVALID_COMMITMENT = "invalid_commitment"
    INVALID_OPENING = "invalid_opening"
    INVALID_VALUE = "invalid_value"
    PUBLIC_IO_MISMATCH = "public_io_mismatch"
    CHECK_MISMATCH = "check_mismatch"
    CHALLENGE_MISMATCH = "challenge_mismatch"
    COVERAGE_MISMATCH = "coverage_mismatch"
    RELATION_REJECTED = "relation_rejected"
    INVALID_COMPILED_RESULT = "invalid_compiled_result"
    MALFORMED_TRANSCRIPT = "malformed_transcript"
    NONCANONICAL_TRANSCRIPT = "noncanonical_transcript"
    RESOURCE_LIMIT = "resource_limit"
    TRUSTED_SERVICE_FAILURE = "trusted_service_failure"
    PROOF_REJECTED = "proof_rejected"
    FAULTS_EXCEEDED = "faults_exceeded"
    FAULT_DECLARATION_INVALID = "fault_declaration_invalid"
    EXEMPTIONS_EXCEEDED = "exemptions_exceeded"
    EXEMPTION_INVALID = "exemption_invalid"


class Reject(ProtocolError):
    """The verifier rejects; ``code`` says why."""

    def __init__(self, code: VerificationCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail
