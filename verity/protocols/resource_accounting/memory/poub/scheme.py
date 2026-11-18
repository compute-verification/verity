"""PoUB's declaration as a resource scheme (`SCHEME`), registered when this module is imported.

No verifier loads this module, and it loads none: the verifier states its own id, component name and deadline
(`verifier.p2_parties`), and PoUB's tests hold the two equal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity.commitments import domains
from verity.protocols.resource_accounting import scheme as S

#: the allowance from a window's boundary message to its roots' deadline, which the session's parameters carry
DEADLINE_NS = 10**9
#: what `refresh` waits on: a session with the setting on accepted by the Lean verifier
REFRESH_GOAL = "memory-accounting/refresh-session-accepted"
#: a charge the ledger names and takes off no P2 record: no bound is stated at P2's encode
UNSTATED = ""

SCHEME = S.register(S.Scheme(
    name="bandwidth",
    protocol=domains.POUB_P2,
    resource="read",
    blocks=(S.Block("BlockOpeningCheck_v2", lambda shape: 8 * shape["BYTES"]),),
    assumptions=(
        "ResourceAccounting.Assumptions.SketchHashRandom",
        "ResourceAccounting.Assumptions.SketchRank",
        "ResourceAccounting.Remote.Sketch.Assumptions.KeyBudget",
        "ResourceAccounting.Remote.Sketch.Assumptions.Present",
        "ResourceAccounting.Remote.Sketch.Assumptions.Receipted",
        "ResourceAccounting.Assumptions.WorkHardnessP2",
        "ResourceAccounting.Remote.Held.Assumptions.SquaringCeilingP2",
    ),
    log=(
        S.Order("registration", "challenge"),
        S.Order("roots", "openings", within_ns=DEADLINE_NS,
                assumption="ResourceAccounting.Remote.Held.Assumptions.SquaringCeilingP2"),
    ),
    charges=tuple(S.Charge(name, UNSTATED) for name in ("chip", "counting", "rebuild", "host", "link", "carrier")),
    settings={"refresh": False},
    hooks=("memory", "steps"),
    requires={
        "one-labelling": "its own trees: PoUS's vk root over the store and a receipt root a pass",
        "log-order": "the verifier party's own log of the session, closed under the memory attester's key",
        "log-times": "the verifier party's clock on the auditor's host",
        "coins-at-rate": "a challenger in the verifier's process on its host's entropy; a pass's seed is Fiat-Shamir, "
                         "from its window's challenge and the previous pass's root",
        "one-design": "its own sample of receipts a window and its own pooled bound; the openings go to the verifier "
                      "in the clear and no session proves its block",
    },
))


def settings(given: Mapping[str, Any]) -> dict[str, Any]:
    """`SCHEME`'s settings with ``given`` over their defaults: the one place a run's ``refresh`` is read from."""
    out = S.settings(SCHEME, given)
    if out["refresh"]:
        raise NotImplementedError(REFRESH_GOAL)
    return out


__all__ = ["DEADLINE_NS", "REFRESH_GOAL", "SCHEME", "UNSTATED", "settings"]
