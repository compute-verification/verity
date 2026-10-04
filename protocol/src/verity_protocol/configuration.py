"""The configuration: what a deployment enables and with which parameters.

It names, at three levels, the protocol's variant, the subprotocol behind each interface and the construction behind
each subprotocol, with each one's assurance parameters and share of δ; the commitment scheme and challenge source every
party uses; and the policy. A choice is named here only when two valid choices would differ in security or would not
interoperate. Which library computes a circuit query or a trace never is: every correct one gives the same answer.

Integrations build a configuration and pass it in with the constructions it names (`epoch.run`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from verity_protocol.accounting.interfaces import INTERFACES, Construction
from verity_protocol.challenges import ChallengeSource
from verity_protocol.commitments import CommitmentScheme
from verity_protocol.compliance.policy import Policy
from verity_protocol.compliance.proof import ComplianceProof

#: variant -> the accounting interfaces it requires
VARIANTS = {"v0": ("integrity",)}


@dataclass(frozen=True)
class Choice:
    subprotocol: str
    construction: str
    delta: float = 0.0  # its share of the epoch's δ
    parameters: Mapping[str, object] = field(default_factory=dict)  # its assurance parameters


@dataclass(frozen=True)
class Configuration:
    variant: str
    accounting: Mapping[str, Choice]  # interface -> choice; an interface left out establishes nothing
    compliance: Choice
    commitments: str  # the commitment scheme's name
    challenges: str  # the challenge source's name
    policy: Policy
    delta: float  # the epoch's soundness error: the subprotocols' shares add up to at most this

    def digest(self) -> str:
        body = {
            "variant": self.variant,
            "accounting": {k: asdict(v) for k, v in sorted(self.accounting.items())},
            "compliance": asdict(self.compliance),
            "commitments": self.commitments,
            "challenges": self.challenges,
            "policy": [{"rule": r.name, **asdict(r)} for r in self.policy],
            "delta": self.delta,
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def validate(
    config: Configuration,
    constructions: Mapping[str, Construction],
    compliance: ComplianceProof,
    commitments: CommitmentScheme,
    challenges: ChallengeSource,
) -> tuple[str, ...]:
    """Every reason `config` cannot run with these components; empty when it can."""
    why: list[str] = []
    required = VARIANTS.get(config.variant)
    if required is None:
        why.append(f"unknown variant {config.variant!r} (known: {', '.join(VARIANTS)})")
    else:
        why += [f"variant {config.variant} requires {i}" for i in required if i not in config.accounting]
    for interface, choice in config.accounting.items():
        c = constructions.get(interface)
        if interface not in INTERFACES:
            why.append(f"unknown interface {interface!r} (known: {', '.join(INTERFACES)})")
        elif c is None:
            why.append(f"{interface}: no construction given for {choice.subprotocol}:{choice.construction}")
        elif (c.interface, c.subprotocol, c.name) != (interface, choice.subprotocol, choice.construction):
            why.append(f"{interface}: configured {choice.subprotocol}:{choice.construction}, given {c.subprotocol}:{c.name} for {c.interface}")
        else:
            why += [f"{interface}: {r}" for r in c.refusals(config)]
    why += [f"{i}: a construction the configuration does not enable" for i in sorted(set(constructions) - set(config.accounting))]
    if (compliance.subprotocol, compliance.name) != (config.compliance.subprotocol, config.compliance.construction):
        why.append(f"compliance: configured {config.compliance.construction}, given {compliance.name}")
    if commitments.name != config.commitments:
        why.append(f"commitments: configured {config.commitments}, given {commitments.name}")
    if challenges.name != config.challenges:
        why.append(f"challenges: configured {config.challenges}, given {challenges.name}")
    shares = sum(c.delta for c in config.accounting.values())
    if shares > config.delta:
        why.append(f"δ: the subprotocols' shares sum to {shares}, over the configured {config.delta}")
    return tuple(why)
