"""Accounting's decision: does the evidence validate the committed account, and what does it establish?

Accounting has no policy. It accepts when every enabled subprotocol accepted and their findings are consistent with the
account; what it establishes (unaccounted work and memory per device, covert capacity per isolation unit, wrong proof
units, the summed δ) goes to compliance in the validated account, whose rules decide what is allowed. An interface the
configuration leaves off establishes nothing: its resource counts as entirely unaccounted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.protocol.accounting.account import Account
from verity.protocol.accounting.interfaces import (
    INTERFACES,
    ChannelFinding,
    EpochStart,
    Finding,
    IntegrityFinding,
    SpaceFinding,
    WorkFinding,
)
from verity.protocol.commitments import Commitment


@dataclass(frozen=True)
class Balance:
    device: str
    declared_work: int
    certified_work: int
    unaccounted_work: int  # capacity the evidence leaves unexplained
    declared_state_bits: int
    occupied_bits: int
    unaccounted_bits: int


@dataclass(frozen=True)
class ValidatedAccount:
    """What compliance reads: the commitment to an account accounting accepted, and what accounting established."""

    epoch: int
    account: Commitment
    balances: tuple[Balance, ...]
    covert_bits: Mapping[str, int] | None  # per isolation unit; None when the channel interface is off (unbounded)
    wrong_units: int
    delta: float


def check(
    account: Account, commitment: Commitment, start: EpochStart, findings: Mapping[str, Finding], delta: float
) -> tuple[ValidatedAccount | None, tuple[str, ...]]:
    """The validated account, or None with every reason it was rejected."""
    why: list[str] = []
    for name, f in findings.items():
        if not isinstance(f, INTERFACES[name]):
            why.append(f"{name}: a {type(f).__name__} is not a {INTERFACES[name].__name__}")
        elif not f.accepted:
            why.append(f"{name}: rejected")
    integrity = findings.get("integrity")
    if not isinstance(integrity, IntegrityFinding):
        why.append("integrity: no finding, and nothing else shows the declared work ran as declared")
    total = sum(f.delta for f in findings.values())
    if total > delta:
        why.append(f"δ: the findings sum to {total}, over the configured {delta}")

    inventory = start.inventory
    devices = {d.id for d in inventory.devices}
    spatial = {s.id: s for s in inventory.spatial_units}
    units = {u.id: u for u in account.units}
    for u in account.units:
        if u.spatial_unit not in spatial:
            why.append(f"isolation unit {u.id}: spatial unit {u.spatial_unit} is not in the inventory")
    for e in account.executions:
        if e.device not in devices:
            why.append(f"{e.circuit}: device {e.device} is not in the inventory")
        elif e.unit not in units:
            why.append(f"{e.circuit}: isolation unit {e.unit} is not in the account")
        elif (s := spatial.get(units[e.unit].spatial_unit)) is not None and e.device not in s.devices:
            why.append(f"{e.circuit}: device {e.device} is outside its isolation unit's spatial unit")
        if isinstance(integrity, IntegrityFinding) and e.circuit not in integrity.covered:
            why.append(f"{e.circuit}: executed, and integrity did not cover it")

    work, space, channel = findings.get("work"), findings.get("space"), findings.get("channel")
    balances = []
    for d in inventory.devices:
        declared_work, declared_bits = account.declared(d.id)
        certified = work.certified.get(d.id, 0) if isinstance(work, WorkFinding) else 0
        occupied = space.occupied_bits.get(d.id, 0) if isinstance(space, SpaceFinding) else 0
        if certified > declared_work:
            why.append(f"{d.id}: {certified} work certified, more than the {declared_work} declared")
        if declared_work > d.work:
            why.append(f"{d.id}: {declared_work} work declared, more than its capacity {d.work}")
        if declared_bits + occupied > d.memory_bits:
            why.append(f"{d.id}: {declared_bits} declared and {occupied} occupied bits exceed its memory {d.memory_bits}")
        balances.append(
            Balance(d.id, declared_work, certified, max(d.work - certified, 0), declared_bits, occupied,
                    max(d.memory_bits - declared_bits - occupied, 0))
        )
    covert = None
    if isinstance(channel, ChannelFinding):
        covert = dict(channel.covert_bits)
        if set(covert) != set(units):
            why.append(f"channel: charged {sorted(covert)}, the account's isolation units are {sorted(units)}")

    if why:
        return None, tuple(why)
    wrong = integrity.wrong_units if isinstance(integrity, IntegrityFinding) else 0
    return ValidatedAccount(account.epoch, commitment, tuple(balances), covert, wrong, total), ()
