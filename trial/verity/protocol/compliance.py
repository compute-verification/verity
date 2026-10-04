"""Compliance: whether what happened was allowed, decided from the validated account alone.

The policy is NCI's bound. This module turns the validated account into NCI's charges per isolation unit; the
developer's side discloses the account's per-unit figures (`Disclosure`), which the toy opens against the account's
commitment. NCI's construction of record proves the same relation in zero knowledge instead.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from verity.primitives.commitments import CommitmentScheme
from verity.protocol.accounting.account import Account
from verity.protocol.accounting.check import ValidatedAccount
from verity.subprotocols.nci.disclosed import Disclosed
from verity.subprotocols.nci.policy import Charge, NciBound

PROOFS = {Disclosed.name: Disclosed}


@dataclass(frozen=True)
class Disclosure:
    account: Account
    nonce: bytes
    circuits: Mapping[str, tuple[int, int]]  # circuit id -> (gates, novel bits it reads)


def charges(validated: ValidatedAccount, disclosure: Disclosure) -> dict[str, Charge]:
    unaccounted = {b.device: b.unaccounted_bits for b in validated.balances}
    gates, novel, devices = defaultdict(int), defaultdict(int), defaultdict(set)
    for e in disclosure.account.executions:
        g, n = disclosure.circuits[e.circuit]
        gates[e.unit] += g
        novel[e.unit] += n
        devices[e.unit].add(e.device)
    covert = validated.covert_bits
    return {
        u.id: Charge(gates[u.id], novel[u.id], None if covert is None else covert[u.id], sum(unaccounted[d] for d in devices[u.id]))
        for u in disclosure.account.units
    }


def complies(validated: ValidatedAccount, bound: NciBound, proof: str, scheme: CommitmentScheme, disclosure: Disclosure) -> bool:
    if scheme.commit([disclosure.account.encode(), disclosure.nonce]) != validated.account:
        return False
    return PROOFS[proof]().verify(charges(validated, disclosure), bound)
