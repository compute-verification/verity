"""Accounting's decision: does the evidence validate the committed account, and what does it establish?

Accounting has no policy. It accepts when every enabled subprotocol accepted and what they established is consistent with
the account; what it establishes (unaccounted work and memory per device, covert capacity per isolation unit, wrong proof
units, the summed δ) goes to compliance in the validated account. A subprotocol the configuration leaves off establishes
nothing: its resource counts as entirely unaccounted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.primitives.commitments import Commitment
from verity.protocol.accounting.account import Account, Inventory
from verity.subprotocols.network_warden.audit import ChannelProfile
from verity.subprotocols.pous.audit import StorageProfile
from verity.subprotocols.pouw.audit import WorkProfile
from verity.subprotocols.sampled_proofs.audit import IntegrityProfile


@dataclass(frozen=True)
class Evidence:
    """What the subprotocols established this epoch: integrity is required, the rest are optional."""

    integrity: IntegrityProfile
    work: WorkProfile | None = None
    space: StorageProfile | None = None
    channel: ChannelProfile | None = None

    def present(self) -> dict[str, object]:
        return {k: v for k, v in vars(self).items() if v is not None}


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
    covert_bits: Mapping[str, int] | None  # per isolation unit; None when no warden bounds the channel
    wrong_units: int
    delta: float


def check(
    account: Account, commitment: Commitment, inventory: Inventory, evidence: Evidence, delta: float
) -> tuple[ValidatedAccount | None, tuple[str, ...]]:
    """The validated account, or None with every reason it was rejected."""
    why = [f"{name}: rejected" for name, p in evidence.present().items() if not p.accepted]
    total = sum(p.delta for p in evidence.present().values())
    if total > delta:
        why.append(f"δ: the evidence sums to {total}, over the configured {delta}")

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
        if e.circuit not in evidence.integrity.covered:
            why.append(f"{e.circuit}: executed, and integrity did not cover it")

    balances = []
    for d in inventory.devices:
        declared_work, declared_bits = account.declared(d.id)
        certified = evidence.work.certified.get(d.id, 0) if evidence.work else 0
        occupied = evidence.space.occupied_bits.get(d.id, 0) if evidence.space else 0
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
    if evidence.channel:
        covert = dict(evidence.channel.covert_bits)
        if set(covert) != set(units):
            why.append(f"channel: charged {sorted(covert)}, the account's isolation units are {sorted(units)}")

    if why:
        return None, tuple(why)
    return ValidatedAccount(account.epoch, commitment, tuple(balances), covert, evidence.integrity.wrong_units, total), ()
