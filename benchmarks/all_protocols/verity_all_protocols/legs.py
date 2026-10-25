"""The legs: one function per protocol, each auditing one period of the loop with that protocol's own code.

A leg is ``leg(period: Period) -> Leg``. It reads the period's records (``period.records``: the window the loop wrote,
`verity.ml.windows.interface`'s files: ``circuit.json``, ``registration.json``, the value files, and ``serve.json``) and
whatever its protocol reads on ``period.node``, writes what it keeps under ``period.out``, and returns its protocol's
answer in the core interface's canonical form (`Answer.canonical`, verity/core/protocols/interface/README.md):

    {"commitment": hex | null, "profile": JSON | null, "protocol": ID, "public": JSON | null, "reason": str | null,
     "verdict": "accept" | "reject" | "undecided"}

beside the quantity its acceptance certifies, the case its port's ``serve`` reads (what onsite's evidence carries for it,
once the port replaces onsite's stub) and the ids of the runs it launched. A leg that can't decide (its tools failed, the
records are unreadable to it) raises; the runner reports that as ``error``, never as a verdict. A protocol with no leg yet
is `missing`: it reports ``{"verdict": "missing", "owner": LANE}`` and never a verdict, so nothing composes it as an accept.

`LEGS` is the runner's whole table: a lane plugs its protocol in by replacing its entry's function, and touches nothing
else.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ANSWER_KEYS = ("commitment", "profile", "protocol", "public", "reason", "verdict")
VERDICTS = ("accept", "reject", "undecided")


@dataclass(frozen=True)
class Period:
    """One period of the loop, as every leg sees it.

    ``records``: the window the loop wrote this period. ``circuit``: the SHA-512 of the circuit record the verifier
    registered before the loop ran, from its shape and run id alone. ``run_root``: the run root the loop published.
    ``law``, ``cap``: the proof service's law and member cap the window was committed under. ``tools``: the built
    executables by name (``flock-circuit``, ``flock-verify``, ``flock-firewall``, ``m0-pythonpath``, ...). ``prover_env``:
    the environment of the prover's processes this period (an adversarial period's prover skips its self-check, so it
    sends proofs of a false witness): the verifier never reads it."""

    index: int
    node: str
    records: Path
    out: Path
    circuit: str
    run_root: str
    law: str
    cap: int
    tools: Mapping[str, str] = field(default_factory=dict)
    prover_env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Leg:
    """What a leg returns: its answer (`ANSWER_KEYS`), the quantity acceptance certifies (``None`` unless it accepts),
    its port's case ``{"commitment": hex, "evidence": hex, "params": JSON}`` (``None`` until the leg can build it) and
    the runs it launched."""

    answer: dict[str, Any]
    quantity: dict[str, Any] | None = None
    case: dict[str, Any] | None = None
    runs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Missing:
    """A protocol whose leg hasn't landed: ``owner`` is the lane plugging it in."""

    owner: str


LegFn = Callable[[Period], Leg]


@dataclass(frozen=True)
class Entry:
    """One protocol of the table: its name in the report, its onsite component (``None``: not one of onsite's), the lane
    that owns its leg, and the leg (or `Missing`)."""

    name: str
    component: str | None
    owner: str
    leg: LegFn | Missing


def _integrity(period: Period) -> Leg:
    from . import integrity
    return integrity.leg(period)


LEGS: tuple[Entry, ...] = (
    Entry("integrity", "sampled-proofs", "compute-accounting", _integrity),
    Entry("work", "pouw", "compute-accounting", Missing("compute-accounting")),
    Entry("space", "pous", "memory-accounting", Missing("memory-accounting")),
    Entry("bandwidth", None, "memory-accounting", Missing("memory-accounting")),
    Entry("communication", "certifier", "network-accounting", Missing("network-accounting")),
)


def check_answer(a: Any) -> dict[str, Any]:
    """``a`` when it is an answer in the shapes `Answer.ofJson` reads (`pouw.core_check.answer_of`, the Python mirror of
    the core interface's decoder): an accept has no reason and both its public outputs and its profile, a refusal a
    reason and neither. Anything else is a ValueError, which the runner reports as the leg's error."""
    from verity.core.protocols.pouw import core_check as CC

    try:
        CC.answer_of(a)
    except CC.Refused as e:
        raise ValueError(f"not a core answer: {e}") from None
    return dict(a)


__all__ = ["ANSWER_KEYS", "Entry", "LEGS", "Leg", "LegFn", "Missing", "Period", "VERDICTS", "check_answer"]
