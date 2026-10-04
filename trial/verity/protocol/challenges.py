"""Where the auditor's coins come from, and the rule every draw follows.

The rule: a subprotocol's coins come from the auditor's source (a public beacon round, an auditor secret, or the
auditor's own OS randomness), never from a developer's commitment; they are drawn only after the commitments they must
follow exist; and each subprotocol draws in its own domain, so no two subprotocols share coins. The composition's δs add
up only under this rule.

`Challenger` enforces it: the composition hands each subprotocol one, bound to its interface and to the epoch's
commitments. The sources are constructions (today `verity.randomness.derive` over a drand round, `verity_pouw.beacon`,
or `verity_sampled_proofs.law.OwnRandomness`), named by the configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


class ChallengeSource(Protocol):
    name: str

    def coins(self, domain: str, context: Mapping[str, str]) -> bytes:
        """32 bytes no developer can predict before every value in `context` is fixed."""


@dataclass(frozen=True)
class Challenger:
    source: ChallengeSource
    interface: str
    bindings: Mapping[str, str]  # the epoch, the configuration and the commitments every draw follows

    def draw(self, purpose: str, **context: str) -> bytes:
        clash = sorted(set(context) & set(self.bindings))
        if clash:
            raise ValueError(f"{self.interface}: context keys {clash} are the epoch's bindings")
        return self.source.coins(f"verity/{self.interface}/{purpose}/v1", {**self.bindings, **context})
