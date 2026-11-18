"""The refresh setting's attack harness (`refresh_attacks`): the honest refresh rota earns no free receipts, each
attack earns its free receipts with its one rule switched off and none with refresh as designed, rejected with the
failure that rule names, and the switches leave the verifier as it was."""
from __future__ import annotations

import dataclasses

import pytest
import refresh_attacks as ra

from verity.protocols.resource_accounting.memory.poub.verifier.audit import AuditRun, Params

N = ra.N
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
    s.read((0, N))
    s.fresh("A", ra.pages(16, 1))
    s.fresh("B", s.stored("A"))
    s.read((0, N), "A", "B")
    s.next_window()
    s.read((0, N), "A", "B")
    s.finish()
    (a, _), (b, _) = s.runs("A"), s.runs("B")
    assert s.store.payload(b) == s.store.blocks[a]
    assert s.held_reads() == N + 2 * (N + 1)
    assert ra.free_receipts(s) == 0
    assert ra.credited_receipts(s, NoCarrier()) == 3 * N + 4
    assert ra.free_receipts(s, NoCarrier()) == 2


def test_the_rule_switches_leave_the_verifier_as_it_was() -> None:
    """Each switch edits one condition of one method of the verifier's class for one run and restores it; the
    session's parameters take none."""
    before = dict(vars(AuditRun))
    for rule in ra.SWITCHES:
        with ra.rule_off(rule):
            assert {k for k in before if vars(AuditRun)[k] is not before[k]}
        assert dict(vars(AuditRun)) == before
    assert not {f.name for f in dataclasses.fields(Params)} & set(ra.SWITCHES)
    assert {a.rule for a in ra.ATTACKS} <= set(ra.SWITCHES)


@pytest.mark.slow
@needs_refresh
def test_the_honest_refresh_rota_is_accepted_and_earns_no_free_receipts() -> None:
    """The reference prover's rota (`audit_prover.Refresh`): every receipt credited, each of a stored form held, F
    planned once its probation is over and every registered block re-keyed on the rota."""
    s = ra.Session()
    ra.honest(s)
    s.finish()
    reads = sum(sum(c for _, c in ps.plan.runs) for ps in s.passes)
    assert ra.credited_receipts(s) == s.held_reads() == reads
    assert ra.free_receipts(s) == 0
    assert {"R0@2", "R1@1", "F@2"} <= set(s.named)
    assert all(ps.plan.runs[0][0] >= N for ps in s.passes[ra.LIFETIME:])


@pytest.mark.slow
@needs_refresh
@pytest.mark.parametrize("attack", ra.ATTACKS, ids=[a.name for a in ra.ATTACKS])
def test_each_attack_earns_free_receipts_only_with_its_rule_off(attack: ra.Attack) -> None:
    off, on = ra.pair(attack)
    assert off == (ra.Outcome(False, attack.off_failure, None) if attack.free_off is None
                   else ra.Outcome(True, None, attack.free_off))
    assert on == ra.Outcome(False, attack.failure, None)


@pytest.mark.slow
@needs_refresh
def test_laundering_every_chunk_is_caught_and_one_chunk_is_a_rate() -> None:
    """Attack 4 with every rule on at the reference session's 16 openings a window: all of a re-key's 32 chunks
    laundered is caught; one laundered chunk is a rate, whose uncaught runs earn its two reads, no more than the
    sample term charges."""
    every = ra.laundering(ra.CLASS, 1)
    assert every.caught == every.trials and every.bound < 1e-6
    one = ra.laundering(1, 2)
    assert one.caught + len(one.free) == one.trials and set(one.free) <= {2}
    assert all(f <= c for f, c in zip(one.free, one.sample))
    assert 0 < one.bound < 1
