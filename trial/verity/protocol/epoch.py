"""One epoch: the step `run` repeats. Lower-level than `run`, which refuses the configuration once and chains the epochs.

    developer commits       the weights of record and the baseline θ, before any coins exist
    audits open             each subprotocol gets its own challenger, bound to those commitments
    developer serves        the epoch's work, with each subprotocol's prover side in place
    developer closes        commits to its account and opens it to the auditor devices
    audits close            each establishes its profile, from what the account commits to
    accounting decides      the validated account, or the reasons it was rejected
    compliance decides      from the validated account alone

The outside auditor learns the two verdicts (`EpochReport`). The auditor devices keep the `EpochRecord`, which binds
every commitment and profile of the epoch, and the previous epoch's record, to one digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Protocol

from verity.primitives.commitments import Commitment, CommitmentScheme
from verity.primitives.randomness import ChallengeSource, Challenger
from verity.protocol.accounting.account import Account, Inventory
from verity.protocol.accounting.check import Evidence, check
from verity.protocol.compliance import Disclosure, complies
from verity.protocol.configuration import Configuration
from verity.subprotocols.network_warden import audit as network_warden
from verity.subprotocols.pous import audit as pous
from verity.subprotocols.pouw import audit as pouw
from verity.subprotocols.sampled_proofs import audit as sampled_proofs


class Verdict(Enum):
    """Each layer's report every epoch. SILENT is a liveness failure: a party did not answer in time. Never an accept."""

    ACCEPT = "accept"
    REJECT = "reject"
    SILENT = "silent"


class NoAnswer(Exception):
    """Raised by a developer side that did not answer by its deadline."""


@dataclass(frozen=True)
class EpochStart:
    epoch: int
    configuration: str  # its digest
    weights: Commitment  # the weights of record, one commitment every subprotocol binds to
    baseline: Commitment  # θ: what reads are novel against
    inventory: Inventory


class Developer(Protocol):
    def commit(self, epoch: int) -> tuple[Commitment, Commitment]:
        """The weights of record and the baseline θ, before any coins are drawn."""

    def serve(self, start: EpochStart) -> None:
        """Run the epoch's work with every enabled subprotocol's prover side in place."""

    def close(self, epoch: int) -> tuple[Commitment, Account, bytes]:
        """The commitment to the account, the account, and its nonce (leaves: the account's encoding, then the nonce)."""

    def prover(self, subprotocol: str) -> object:
        """Its prover side of `subprotocol`, whose API that subprotocol's audit defines."""

    def disclose(self) -> Disclosure:
        """What compliance's proof needs."""


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
    profiles: Mapping[str, str]  # subprotocol -> digest of what it established
    reasons: tuple[str, ...]
    previous: str | None  # the previous epoch's record digest
    evidence: Evidence | None = field(default=None, compare=False)  # what `profiles` digests, kept by the auditor devices

    def digest(self) -> str:
        return _digest({k: v for k, v in asdict(self).items() if k != "evidence"})


def make_audits(config: Configuration, scheme: CommitmentScheme) -> dict[str, object]:
    """A fresh audit for every enabled subprotocol; raises `NotInstalled` for a construction that is not installed."""
    made: dict[str, object] = {"sampled_proofs": sampled_proofs.Audit(
        config.sampled_proofs.construction, config.sampled_proofs.parameters, config.sampled_proofs.delta, scheme)}
    for name, module, choice in (("pouw", pouw, config.pouw), ("pous", pous, config.pous), ("network_warden", network_warden, config.network_warden)):
        if choice is not None:
            made[name] = module.Audit(choice.construction, choice.parameters)
    return made


def run_epoch(
    config: Configuration, scheme: CommitmentScheme, source: ChallengeSource, developer: Developer, inventory: Inventory,
    epoch: int, previous: str | None,
) -> tuple[EpochReport, EpochRecord]:
    audits = make_audits(config, scheme)
    weights = baseline = account_commitment = None
    evidence = None
    try:
        weights, baseline = developer.commit(epoch)
        start = EpochStart(epoch, config.digest(), weights, baseline, inventory)
        bindings = {"epoch": str(epoch), "previous": previous or "", "configuration": start.configuration, "weights": weights.root.hex(), "baseline": baseline.root.hex()}
        challenger = {name: Challenger(source, name, bindings) for name in audits}
        audits["sampled_proofs"].open(challenger["sampled_proofs"])
        if "pouw" in audits:
            audits["pouw"].open(challenger["pouw"])
        if "pous" in audits:
            audits["pous"].open(weights, challenger["pous"])
        developer.serve(start)
        account_commitment, account, nonce = developer.close(epoch)
        evidence = Evidence(
            integrity=audits["sampled_proofs"].close([e.circuit for e in account.executions], developer.prover("sampled_proofs")),
            work=audits["pouw"].close(developer.prover("pouw")) if "pouw" in audits else None,
            space=audits["pous"].close(developer.prover("pous")) if "pous" in audits else None,
            channel=audits["network_warden"].close([u.id for u in account.units]) if "network_warden" in audits else None,
        )
    except NoAnswer as e:
        record = EpochRecord(epoch, config.digest(), weights, baseline, account_commitment, _profiles(evidence), (f"no answer: {e}",), previous, evidence)
        return EpochReport(Verdict.SILENT, Verdict.SILENT), record

    opened = () if scheme.commit([account.encode(), nonce]) == account_commitment else ("account: does not open its commitment",)
    validated, reasons = check(account, account_commitment, inventory, evidence, config.delta)
    reasons = opened + reasons
    record = EpochRecord(epoch, start.configuration, weights, baseline, account_commitment, _profiles(evidence), reasons, previous, evidence)
    if reasons:
        return EpochReport(Verdict.REJECT, Verdict.REJECT), record
    ok = complies(validated, config.policy, config.nci.construction, scheme, developer.disclose())
    return EpochReport(Verdict.ACCEPT, Verdict.ACCEPT if ok else Verdict.REJECT), record


def _digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default).encode()).hexdigest()


def _default(obj: object) -> object:
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def _profiles(evidence: Evidence | None) -> dict[str, str]:
    return {k: _digest(asdict(v)) for k, v in evidence.present().items()} if evidence else {}
