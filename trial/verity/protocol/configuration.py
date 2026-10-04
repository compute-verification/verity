"""The configuration: what a deployment enables and with which parameters.

It names, at three levels, the protocol's variant, the subprotocol behind each interface and the construction behind
each subprotocol, with each one's assurance parameters and share of δ; the commitment scheme and challenge source every
party uses; and the policy. A choice is named here only when two valid choices would differ in security or would not
interoperate. Which library computes a circuit query or a trace never is: every correct one gives the same answer.

`resolve` loads what it names from the installed constructions (`registry`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from verity.protocol import registry
from verity.protocol.accounting.interfaces import INTERFACES, Construction
from verity.protocol.challenges import ChallengeSource
from verity.protocol.commitments import CommitmentScheme
from verity.protocol.compliance.policy import Policy
from verity.protocol.compliance.proof import ComplianceProof

#: variant -> the accounting interfaces it requires
VARIANTS = {"v0": ("integrity",)}


@dataclass(frozen=True)
class Choice:
    subprotocol: str
    construction: str
    delta: float = 0.0  # its share of the epoch's δ
    parameters: Mapping[str, object] = field(default_factory=dict)  # its assurance parameters

    @property
    def key(self) -> str:
        return f"{self.subprotocol}/{self.construction}"


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


@dataclass(frozen=True)
class Components:
    constructions: Mapping[str, Construction]  # interface -> construction
    compliance: ComplianceProof
    commitments: CommitmentScheme
    challenges: ChallengeSource


def resolve(config: Configuration) -> tuple[Components | None, tuple[str, ...]]:
    """What `config` names, loaded; or every reason it cannot run."""
    why: list[str] = []
    required = VARIANTS.get(config.variant)
    if required is None:
        why.append(f"unknown variant {config.variant!r} (known: {', '.join(VARIANTS)})")
    else:
        why += [f"variant {config.variant} requires {i}" for i in required if i not in config.accounting]

    def load(what: str, key: str) -> object | None:
        found = registry.load(key)
        if found is None:
            why.append(f"{what}: {key} is not installed (installed: {', '.join(registry.installed()) or 'nothing'})")
        return found

    constructions: dict[str, Construction] = {}
    for interface, choice in config.accounting.items():
        if interface not in INTERFACES:
            why.append(f"unknown interface {interface!r} (known: {', '.join(INTERFACES)})")
        elif (c := load(interface, choice.key)) is not None:
            if c.interface != interface:
                why.append(f"{interface}: {choice.key} meets {c.interface}")
            constructions[interface] = c
    compliance = load("compliance", config.compliance.key)
    commitments = load("commitments", f"commitments/{config.commitments}")
    challenges = load("challenges", f"challenges/{config.challenges}")
    shares = sum(c.delta for c in config.accounting.values())
    if shares > config.delta:
        why.append(f"δ: the subprotocols' shares sum to {shares}, over the configured {config.delta}")
    if why:
        return None, tuple(why)
    for interface, c in constructions.items():
        why += [f"{interface}: {r}" for r in c.refusals(config)]
    if why:
        return None, tuple(why)
    return Components(constructions, compliance, commitments, challenges), ()
