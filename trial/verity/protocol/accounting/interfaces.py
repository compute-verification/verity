"""What accounting needs from each of its subprotocols.

Faithfulness has two halves. Integrity shows the declared work ran as declared. Work, space and channel show nothing
undeclared had room to happen, one resource at a time. Each interface is met by a subprotocol under `subprotocols/` and
realized by one of its constructions, which the configuration names.

A subprotocol runs one session per epoch: `open` when the epoch's commitments exist (the coins it draws are bound to
them), live exchanges during the epoch over its own channels (PoUS's timed challenges, the warden's taps), and `close`
once the developer has committed its account, returning a finding. The composition orders the phases and never carries
a subprotocol's traffic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from verity.protocol.accounting.account import Account, CircuitId, Inventory
from verity.protocol.challenges import Challenger
from verity.protocol.commitments import Commitment

if TYPE_CHECKING:
    from verity.protocol.configuration import Configuration


@dataclass(frozen=True)
class EpochStart:
    epoch: int
    configuration: str  # the configuration's digest
    weights: Commitment  # the weights of record, one commitment every subprotocol binds to
    baseline: Commitment  # θ: the models, code and data committed up front, against which reads are novel
    inventory: Inventory


@dataclass(frozen=True)
class Finding:
    accepted: bool
    delta: float  # the probability of accepting while the finding is false; 0 for an exact check
    record: Mapping[str, object]  # the subprotocol's own audit record; accounting never reads it, the epoch record binds it


@dataclass(frozen=True)
class IntegrityFinding(Finding):
    """Sampled proofs: the audit record and its integrity profile (`IntegrityProfile.bound()` is `wrong_units`)."""

    covered: frozenset[CircuitId]  # circuits whose executions it drew from
    units: int
    wrong_units: int  # at most this many proof units are wrong


@dataclass(frozen=True)
class WorkFinding(Finding):
    """Proofs of useful work: per device, work certified as performed, (1 - γ)(1 - ε_s)·W of PoUW's `WorkProfile`."""

    certified: Mapping[str, int]


@dataclass(frozen=True)
class SpaceFinding(Finding):
    """Proofs of useful space: per device, memory certified as holding the encoded weights (`StorageProfile`'s
    ρ·|C|, or the continuous audit's average)."""

    occupied_bits: Mapping[str, int]


@dataclass(frozen=True)
class ChannelFinding(Finding):
    """The network warden: per isolation unit, the covert capacity charged to it (ingress timing since its last wipe;
    egress is not charged). Exact, so δ = 0; a violation rejects."""

    covert_bits: Mapping[str, int]


INTERFACES: dict[str, type[Finding]] = {
    "integrity": IntegrityFinding,
    "work": WorkFinding,
    "space": SpaceFinding,
    "channel": ChannelFinding,
}


class Session(Protocol):
    """The auditor's side of one subprotocol for one epoch."""

    def open(self, start: EpochStart, challenger: Challenger) -> None: ...

    def close(self, account: Account, developer: object) -> Finding:
        """`developer` is the developer's side of this construction, whose API the subprotocol defines."""


class Construction(Protocol):
    interface: str  # a key of INTERFACES
    subprotocol: str
    name: str

    def refusals(self, configuration: Configuration) -> tuple[str, ...]:
        """Why this construction cannot run under `configuration`, which names what it is composed with; empty when it can."""

    def session(self, parameters: Mapping[str, object]) -> Session: ...
