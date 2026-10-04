"""Where coins come from, and the rule every draw follows.

The rule: coins come from the auditor's source (a public beacon round, an auditor secret, the auditor's own OS
randomness), never from a commitment the developer makes; they are drawn only after the commitments they must follow
exist; and each subprotocol draws in its own domain, so no two share coins. A composition's δs add up only under it.
`Challenger` enforces the last two: whoever composes subprotocols hands each one its own, bound to what must already be
fixed. The sources stand in for `verity.randomness.derive` over a drand round and its peers.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


class ChallengeSource(Protocol):
    name: str

    def coins(self, domain: str, context: Mapping[str, str]) -> bytes:
        """32 bytes nobody can predict before every value in `context` is fixed."""


@dataclass(frozen=True)
class Challenger:
    source: ChallengeSource
    subprotocol: str
    bindings: Mapping[str, str]  # what every draw follows: in an epoch, its number, configuration and commitments

    def draw(self, purpose: str, **context: str) -> bytes:
        clash = sorted(set(context) & set(self.bindings))
        if clash:
            raise ValueError(f"{self.subprotocol}: context keys {clash} are bindings")
        return self.source.coins(f"verity/{self.subprotocol}/{purpose}/v1", {**self.bindings, **context})


class AuditorSecret:
    name = "auditor-secret"

    def __init__(self, secret: bytes | None = None):
        self._secret = secret or secrets.token_bytes(32)

    def coins(self, domain: str, context: Mapping[str, str]) -> bytes:
        return hashlib.sha256(self._secret + json.dumps([domain, sorted(context.items())]).encode()).digest()


SOURCES: dict[str, type[ChallengeSource]] = {AuditorSecret.name: AuditorSecret}
