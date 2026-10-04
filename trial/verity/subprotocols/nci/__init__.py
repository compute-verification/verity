"""NCI's compliance proof: the committed account satisfies `NciBound`. Meets the compliance-proof interface.

`Disclosed` is the toy: the developer's side discloses the account and each circuit's (gates, novel bits), and the
verifier opens the commitment and evaluates the rule in the clear. The construction of record proves the same in zero
knowledge against `validated.account`.
"""

from __future__ import annotations

from collections import defaultdict

from verity.protocol import registry
from verity.protocol.accounting.check import ValidatedAccount
from verity.protocol.compliance.policy import NciBound, Policy


class Disclosed:
    subprotocol = "nci"
    name = "disclosed"

    def verify(self, validated: ValidatedAccount, policy: Policy, developer: object) -> bool:
        account, nonce, circuits = developer.disclose()
        scheme = registry.load(f"commitments/{validated.account.scheme}")
        if scheme is None or scheme.commit([account.encode(), nonce]) != validated.account:
            return False
        if validated.covert_bits is None:
            return False  # the channel is unbounded without a warden
        unaccounted = {b.device: b.unaccounted_bits for b in validated.balances}
        for rule in policy:
            if not isinstance(rule, NciBound):
                return False
            gates, novel, devices = defaultdict(int), defaultdict(int), defaultdict(set)
            for e in account.executions:
                g, n = circuits[e.circuit]
                gates[e.unit] += g
                novel[e.unit] += n
                devices[e.unit].add(e.device)
            for u in account.units:
                surviving = sum(unaccounted[d] for d in devices[u.id])
                if gates[u.id] > rule.gates or novel[u.id] + validated.covert_bits[u.id] + surviving > rule.budget_bits:
                    return False
        return True
