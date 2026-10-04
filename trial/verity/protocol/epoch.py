"""One epoch of the protocol: the composition.

    validate the configuration          every refusal at once, before anything runs
    developer commits                   the weights of record and the baseline θ, before any coins exist
    sessions open                       each subprotocol binds to those commitments and draws its salts
    developer serves                    the epoch's work, every enabled construction's developer side in place;
                                        subprotocols exchange live (timed challenges, taps) over their own channels
    developer closes                    commits to its account and opens it to the auditor devices
    sessions close                      draws, openings, proofs, verification: one finding each
    accounting decides                  the validated account, or the reasons it was rejected
    compliance decides                  from the validated account alone

The outside auditor learns the two verdicts (`EpochReport`). The auditor devices keep the `EpochRecord`, which binds
every commitment and finding of the epoch to one digest; how much of it a deployment discloses is its configuration's.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Protocol

from verity.protocol.accounting.account import Account, Inventory
from verity.protocol.accounting.check import check
from verity.protocol.accounting.interfaces import EpochStart, Finding
from verity.protocol.challenges import Challenger
from verity.protocol.commitments import Commitment
from verity.protocol.configuration import Configuration, resolve


class Verdict(Enum):
    """Each layer's report every epoch. SILENT is a liveness failure: a party did not answer in time. Never an accept."""

    ACCEPT = "accept"
    REJECT = "reject"
    SILENT = "silent"


class NoAnswer(Exception):
    """Raised by a developer side, or a session waiting on one, that did not answer by its deadline."""


class Refused(ValueError):
    def __init__(self, reasons: tuple[str, ...]):
        self.reasons = reasons
        super().__init__("configuration refused: " + "; ".join(reasons))


class Developer(Protocol):
    def commit(self, epoch: int) -> tuple[Commitment, Commitment]:
        """The weights of record and the baseline θ, before any coins are drawn."""

    def serve(self, start: EpochStart) -> None:
        """Run the epoch's work with every enabled construction's developer side in place."""

    def close(self, epoch: int) -> tuple[Commitment, Account, bytes]:
        """The commitment to the epoch's account, the account, and the nonce it is committed with (leaves: the
        account's encoding, then the nonce)."""

    def side(self, interface: str) -> object:
        """Its side of the construction behind `interface` ("compliance" for the compliance proof)."""


@dataclass(frozen=True)
class EpochReport:
    accounting: Verdict
    compliance: Verdict


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    configuration: str
    weights: Commitment | None
    baseline: Commitment | None
    account: Commitment | None
    findings: Mapping[str, str]  # interface -> digest of its finding
    reasons: tuple[str, ...]  # why accounting rejected, if it did

    def digest(self) -> str:
        return _digest(asdict(self))


def run(
    config: Configuration,
    developer: Developer,
    inventory: Inventory,
    epoch: int,
) -> tuple[EpochReport, EpochRecord]:
    components, refused = resolve(config)
    if refused:
        raise Refused(refused)
    constructions, commitments = components.constructions, components.commitments
    weights = baseline = account_commitment = None
    findings: dict[str, Finding] = {}
    try:
        weights, baseline = developer.commit(epoch)
        start = EpochStart(epoch, config.digest(), weights, baseline, inventory)
        bindings = {"epoch": str(epoch), "configuration": start.configuration, "weights": weights.root.hex(), "baseline": baseline.root.hex()}
        sessions = {i: constructions[i].session(c.parameters) for i, c in config.accounting.items()}
        for interface, session in sessions.items():
            session.open(start, Challenger(components.challenges, interface, bindings))
        developer.serve(start)
        account_commitment, account, nonce = developer.close(epoch)
        for interface, session in sessions.items():
            findings[interface] = session.close(account, developer.side(interface))
    except NoAnswer as e:
        record = EpochRecord(epoch, config.digest(), weights, baseline, account_commitment, _digests(findings), (f"no answer: {e}",))
        return EpochReport(Verdict.SILENT, Verdict.SILENT), record

    opened = () if commitments.commit([account.encode(), nonce]) == account_commitment else ("account: does not open its commitment",)
    validated, why = check(account, account_commitment, start, findings, config.delta)
    reasons = opened + why
    record = EpochRecord(epoch, start.configuration, weights, baseline, account_commitment, _digests(findings), reasons)
    if reasons:
        return EpochReport(Verdict.REJECT, Verdict.REJECT), record
    complies = components.compliance.verify(validated, config.policy, developer.side("compliance"))
    return EpochReport(Verdict.ACCEPT, Verdict.ACCEPT if complies else Verdict.REJECT), record


def _digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default).encode()).hexdigest()


def _default(obj: object) -> object:
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def _digests(findings: Mapping[str, Finding]) -> dict[str, str]:
    return {i: _digest(asdict(f)) for i, f in findings.items()}
