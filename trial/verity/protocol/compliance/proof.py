"""The compliance-proof interface: a subprotocol under `subprotocols/` meets it.

The verifier sees the validated account (a commitment and what accounting established), never the account itself; the
developer proves, in zero knowledge in a construction of record, that the committed account satisfies every rule.
"""

from __future__ import annotations

from typing import Protocol

from verity.protocol.accounting.check import ValidatedAccount
from verity.protocol.compliance.policy import Policy


class ComplianceProof(Protocol):
    subprotocol: str
    name: str

    def verify(self, validated: ValidatedAccount, policy: Policy, developer: object) -> bool:
        """`developer` is the developer's side of this construction."""
