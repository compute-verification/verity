"""The configuration: which subprotocols a deployment composes, with which constructions and parameters.

A choice is named here only when two valid choices would differ in security or would not interoperate. Which library
computes a circuit query or a trace never is: every correct one gives the same answer. Which subprotocol fills which role
is not a choice either: it is a field, since composing a different one changes the protocol's guarantee.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from verity.primitives.commitments import SCHEMES
from verity.primitives.randomness import SOURCES
from verity.protocol.compliance import PROOFS
from verity.subprotocols.nci.policy import NciBound


@dataclass(frozen=True)
class Choice:
    construction: str
    delta: float = 0.0  # its share of the epoch's δ
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Configuration:
    sampled_proofs: Choice  # integrity: required
    pouw: Choice | None  # work
    pous: Choice | None  # space
    network_warden: Choice | None  # channel
    nci: Choice  # compliance's proof
    policy: NciBound
    commitments: str
    challenges: str
    delta: float  # the epoch's soundness error: the subprotocols' shares add up to at most this

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def refusals(config: Configuration) -> tuple[str, ...]:
    """Every reason `config` cannot run that shows before any subprotocol is loaded; a construction that is not installed
    is refused when its subprotocol loads it."""
    why = []
    if config.commitments not in SCHEMES:
        why.append(f"commitments: {config.commitments} is not installed (available: {', '.join(SCHEMES)})")
    if config.challenges not in SOURCES:
        why.append(f"challenges: {config.challenges} is not installed (available: {', '.join(SOURCES)})")
    if config.nci.construction not in PROOFS:
        why.append(f"nci: {config.nci.construction} is not installed (available: {', '.join(PROOFS)})")
    shares = sum(c.delta for c in (config.sampled_proofs, config.pouw, config.pous, config.network_warden) if c)
    if shares > config.delta:
        why.append(f"δ: the subprotocols' shares sum to {shares}, over the configured {config.delta}")
    return tuple(why)
