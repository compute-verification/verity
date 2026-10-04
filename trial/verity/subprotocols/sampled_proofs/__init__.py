"""Sampled proofs: the auditor draws executions from the committed account and has each drawn one verified, by a replay
of its trace or by a proof. Meets `integrity`. What every construction shares is here: the draw, the bound, the session.
A construction supplies only `verify_unit`.

The developer's side answers `executions()`: per execution in the account, in order, its circuit, inputs and trace (a
proof construction's side would answer proofs instead of traces).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping

from verity.circuits import Circuit
from verity.protocol.accounting.account import Account, CircuitId
from verity.protocol.accounting.interfaces import EpochStart, IntegrityFinding
from verity.protocol.challenges import Challenger

VerifyUnit = Callable[[Circuit, tuple[int, ...], object], bool]


def draw(coins: bytes, units: int, k: int) -> list[int]:
    """k distinct unit indices, uniform, from the coins."""
    chosen: list[int] = []
    counter = 0
    while len(chosen) < min(k, units):
        x = int.from_bytes(hashlib.sha256(coins + counter.to_bytes(8, "big")).digest(), "big") % units
        counter += 1
        if x not in chosen:
            chosen.append(x)
    return chosen


class SampledSession:
    def __init__(self, parameters: Mapping[str, object], verify_unit: VerifyUnit):
        self.k = int(parameters.get("k", 8))
        self.fraction = float(parameters.get("fraction", 0.25))  # the wrong fraction the bound is stated at
        self.verify_unit = verify_unit

    def open(self, start: EpochStart, challenger: Challenger) -> None:
        self.challenger = challenger

    def close(self, account: Account, developer: object) -> IntegrityFinding:
        executions = developer.executions()
        n = len(account.executions)
        coins = self.challenger.draw("units", account=hashlib.sha256(account.encode()).hexdigest())
        drawn = draw(coins, n, self.k)
        failed = [
            i for i in drawn
            if executions[i][0].id() != account.executions[i].circuit or not self.verify_unit(*executions[i])
        ]
        covered = frozenset(CircuitId(e.circuit) for e in account.executions)
        delta = (1 - self.fraction) ** len(drawn) if len(drawn) < n else 0.0
        return IntegrityFinding(
            accepted=not failed, delta=delta, record={"drawn": drawn, "failed": failed},
            covered=covered, units=n, wrong_units=math.floor(self.fraction * n) if len(drawn) < n else 0,
        )
