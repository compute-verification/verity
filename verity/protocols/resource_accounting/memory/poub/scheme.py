"""PoUB's two declarations as resource schemes (`verity.protocols.resource_accounting.scheme`), both of read bandwidth,
registered when this module is imported (`verity.protocols.compose.schemes.DECLARATIONS`).

They are two protocols, each with its own verifier and its own store, and no setting turns one into the other:

- `BANDWIDTH` (``bandwidth``, the name onsite's applications list): PoUB-P2, ``verity/remote-bandwidth/p2/v0``, over
  PoUS's registered C*. Its prover reads PoUS's store and the receipts PoUS's decode computed: it attaches through no
  hook of the served run and has no setting.
- `POUB` (``poub``, the adapter's option and the benchmarks' session): the remote session ``verity/remote-bandwidth/v0``
  over the node's own store (`prover.node_store`), which `prover.binder.Binder` keeps current from the served run's
  ``memory`` and ``steps`` events, subscribed under `POUB`'s name. ``refresh`` is its one setting.

No verifier loads this module, and it loads none: each verifier states its own id, component name and deadline, and
PoUB's tests hold them equal to the declarations.

Neither declares a block: PoUB's opening unit (`verifier.commitments.gates.block_check`, ``BlockOpeningCheck_v2``) is
proved in the read session's own window (`prover.window`), not as a template of the served circuit's first level, and
P2's openings go to its verifier in the clear. Neither says its own term (`Scheme.delta_scheme`): in the Lean it is the
sketch bound's error (`DELTA_SCHEME`), a function of the prover's hash budget Q, the session's passes P, the store's
blocks N and the sign blocks' rank tail ε, not one number.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity import hooks as H
from verity.commitments import domains
from verity.protocols.resource_accounting import scheme as S

#: P2's allowance from a window's boundary message to its roots' deadline, which the session's parameters carry
#: (`verifier.p2_parties.SESSION`); the remote session's is a parameter of each session (`verifier.audit.Params`)
DEADLINE_NS = 10**9
#: what `refresh` waits on: a session with the setting on accepted by the Lean verifier
REFRESH_GOAL = "memory-accounting/refresh-session-accepted"
#: each scheme's own term in the Lean, by the guarantee whose conclusion carries it (`SketchBound T δs`): δs at the
#: deployed receipts, a function of Q, P, N and ε
DELTA_SCHEME = {"bandwidth": "ResourceAccounting.Guarantees.Remote.SketchBoundP2",
                "poub": "ResourceAccounting.Guarantees.Remote.SketchBoundDeployed"}
#: the ways to a correct receipt without a read from node memory, as the session's ledger names them
#: (`verifier.audit.GammaTerms`)
CHARGES = ("chip", "counting", "rebuild", "host", "link", "carrier")
_SKETCH = ("ResourceAccounting.Assumptions.SketchHashRandom",
           "ResourceAccounting.Assumptions.SketchRank",
           "ResourceAccounting.Remote.Sketch.Assumptions.KeyBudget",
           "ResourceAccounting.Remote.Sketch.Assumptions.Present",
           "ResourceAccounting.Remote.Sketch.Assumptions.Receipted")
_OUTSIDE = "the opening unit (BlockOpeningCheck_v2) is proved outside the workload's circuit: "
_REQUIRES = {
    "log-order": "the verifier party's own log of the session",
    "log-times": "the verifier party's clock on the auditor's host",
    "coins-at-rate": "a challenger in the verifier's process on its host's entropy; a pass's seed is derived from its "
                     "window's challenge and the previous pass's root",
}

BANDWIDTH = S.register(S.Scheme(
    name="bandwidth",
    protocol=domains.POUB_P2,
    resource="read",
    blocks=(),
    assumptions=(*_SKETCH, "ResourceAccounting.Assumptions.WorkHardnessP2",
                 "ResourceAccounting.Remote.Held.Assumptions.SquaringCeilingP2"),
    log=(
        S.Order("registration", "challenge"),
        S.Order("roots", "boundary", direction="after", within_ns=DEADLINE_NS,
                assumption="ResourceAccounting.Remote.Held.Assumptions.SquaringCeilingP2"),
        S.Order("roots", "openings"),
    ),
    charges=tuple(S.Charge(name, f"no P2 record carries it: verifier.audit.GammaTerms.{name} at P2's units",
                           bounded=False) for name in CHARGES),
    requires={
        **_REQUIRES,
        "one-labelling": "its own trees: PoUS's vk root over the store and a receipt root a pass",
        "one-design": _OUTSIDE + "its own sample of receipts a window and its own pooled bound, the openings sent to "
                                 "the verifier in the clear",
    },
))

POUB = S.register(S.Scheme(
    name="poub",
    protocol=domains.POUB_PROTOCOL,
    resource="read",
    blocks=(),
    assumptions=(*_SKETCH, "ResourceAccounting.Assumptions.WorkHardness",
                 "ResourceAccounting.Remote.Held.Assumptions.SquaringCeiling"),
    log=(S.Order("registration", "challenge"), S.Order("roots", "openings")),
    charges=tuple(S.Charge(name, f"gamma.{name}") for name in CHARGES),
    settings={"refresh": False},
    hooks=("memory", "steps"),
    requires={
        **_REQUIRES,
        "one-labelling": "its own trees: the node store's registration root and a receipt root a pass",
        "one-design": _OUTSIDE + "in the read session's own window (prover.window), whose circuit is not built "
                                 "(memory-accounting/read-session-in-zero-knowledge); today its own sample of "
                                 "receipts a window and its own pooled bound",
    },
))

#: what `POUB` asks of the workload, for the adapter to file under its name: a freed range is reused as late as the
#: workload allows, no range comes back holding another input's values, and a step's output is taken before the next
#: step is scheduled. The binder files the first itself (`Binder(fifo=True)`); the two properties are filed once the
#: adapter declares them.
REUSE_OLDEST_FREED = H.ReuseOldestFreed()
REQUESTS: tuple[H.Request, ...] = (
    REUSE_OLDEST_FREED,
    H.RefuseProperty("prefix-caching", True, "a prefix hit brings a range back with the values another input wrote"),
    H.RequireProperty("async-scheduling", False, "a pass follows each step's schedule and its output in turn"),
)


def settings(given: Mapping[str, Any]) -> dict[str, Any]:
    """`POUB`'s settings with ``given`` over their defaults: the one place a session's ``refresh`` is read from."""
    out = S.settings(POUB, given)
    if out["refresh"]:
        raise NotImplementedError(REFRESH_GOAL)
    return out


__all__ = ["BANDWIDTH", "CHARGES", "DEADLINE_NS", "DELTA_SCHEME", "POUB", "REFRESH_GOAL", "REQUESTS",
           "REUSE_OLDEST_FREED", "settings"]
