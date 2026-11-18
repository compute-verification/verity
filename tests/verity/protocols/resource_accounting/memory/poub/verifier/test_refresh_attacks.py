"""The refresh setting's attack harness (`refresh_attacks`): the honest refresh rota earns no free receipts, each
attack earns its free receipts with its one rule switched off and none with refresh as designed, rejected with the
failure that rule names, and the switches leave the verifier as it was."""
from __future__ import annotations

import dataclasses

import pytest
import refresh_attacks as ra
from test_appends import N

from verity.protocols.resource_accounting.memory.poub.verifier.audit import AuditRun, Params

needs_refresh = pytest.mark.skipif(not ra.refresh_mirrored(), reason="this tree's verifier has no Params.refresh")


@dataclasses.dataclass(frozen=True)
class NoCarrier:
    holds: bool = True

    def carrier_bytes(self, read_bytes: int) -> float:
        return 0.0


def test_a_carrier_answers_for_the_stored_form_it_carries() -> None:
    """Without refresh, B's payload chosen as A's stored form once A's coin is out is accepted: decoding B gives A's
    stored form, so the pass holding both reads one held form for two receipts.  The ledger credits none of them
    without the stacking property's verdict, so nothing is free; at a carrier of 0, what refresh's ledger charges,
    the two passes holding both earn one free receipt each."""
    s = ra.Session(refresh=False, windows=2)
    s.read(ra.REG)
    s.fresh("A", ra.pages(16, 1))
    s.fresh("B", s.stored("A"))
    s.read(ra.REG, "A", "B")
    s.next_window()
    s.read(ra.REG, "A", "B")
    s.finish()
    (a, _), (b, _) = s.runs("A"), s.runs("B")
    assert s.store.payload(b) == s.store.blocks[a]
    assert s.held_reads() == N + 2 * (N + 1)
    assert ra.free_receipts(s) == 0
    assert ra.credited_receipts(s, NoCarrier()) == 3 * N + 4
    assert ra.free_receipts(s, NoCarrier()) == 2


def test_the_rule_switches_leave_the_verifier_as_it_was() -> None:
    """A switch patches the verifier's class for one run and restores it; the session's parameters take none."""
    before = dict(vars(AuditRun))
    with ra.rule_off("key timing"):
        assert vars(AuditRun)["_stale_below"] is ra._stale_below_off
    assert dict(vars(AuditRun)) == before
    assert not {f.name for f in dataclasses.fields(Params)} & set(ra.SWITCHES)
    assert {a.rule for a in ra.ATTACKS} <= set(ra.SWITCHES)


@needs_refresh
def test_the_honest_refresh_rota_is_accepted_and_earns_no_free_receipts() -> None:
    """Registered in windows 0 and 1, then each window's re-keys of the registered store and of F the window after:
    every receipt credited, each of a stored form held."""
    s = ra.Session()
    ra.honest(s)
    s.finish()
    assert ra.credited_receipts(s) == s.held_reads() == 2 * N + 3 * (N + 2)
    assert ra.free_receipts(s) == 0


@needs_refresh
@pytest.mark.parametrize("attack", ra.ATTACKS, ids=[a.name for a in ra.ATTACKS])
def test_each_attack_earns_free_receipts_only_with_its_rule_off(attack: ra.Attack) -> None:
    try:
        off, on = ra.pair(attack)
    except ra.Pending as e:
        pytest.skip(str(e))
    assert off == ra.Outcome(True, None, attack.free_off)
    assert on == ra.Outcome(False, attack.failure, None)
