"""The post-seed verdict's binding 4 (`audit.property_reading`, `ledger`): the record's windows and lifetime cap read in
census steps (MEM-38), and the ledger's refusal of a `PostSeed` judged at any other reading.  The verdict itself is the
Lean's (`RemoteAudit.PostSeed.verdict`, through `audit.post_seed`); `lean/tests/test_post_seed_difftest.py` runs it and
holds it to its Python mirror."""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

from test_appends import P512, honest_session
from verity.core.protocols.remote_bandwidth import audit

COMMON = dict(capacity_bytes_per_s=4096 * 10**9 / 10, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))
SPAN = Fraction(1_000_010, 10**9)


@pytest.fixture(scope="module")
def record() -> audit.Record:
    r = honest_session()
    assert all(v.accepted for v in r.finish())
    rec = r.s.record()
    assert [w.steps for w in rec.windows] == [(0, 1, 2), (3, 4)] and rec.lifetime == 60
    return rec


def test_pass_i_covers_the_steps_after_pass_i_minus_1_through_its_own(record) -> None:
    """Passes at steps 0, 1, 2 and 3, 4 over a census of 5 steps: window 0 covers [0, 3) and window 1 [3, 5), each with
    its certified reads U_t·b and its span; L_w = 60 windows covers both, so L_steps is the census's 5 steps."""
    windows, L = audit.property_reading(record, 5)
    assert windows == (audit.StepWindow(0, 3, 30 * 4096, SPAN), audit.StepWindow(3, 5, 24 * 4096, SPAN)) and L == 5
    short = dataclasses.replace(record, lifetime=0)
    assert audit.property_reading(short, 5)[1] == 3


def test_steps_after_an_unpaced_session_s_last_pass_are_uncovered(record) -> None:
    """A census past the last pass's step: refused unpaced; a paced session's closing window takes the rest."""
    with pytest.raises(ValueError, match="steps after the last pass are uncovered"):
        audit.property_reading(record, 7)
    windows, L = audit.property_reading(record, 7, paced=True)
    assert windows[-1] == audit.StepWindow(3, 7, 24 * 4096, SPAN) and L == 7


def test_a_reading_needs_the_passes_steps_inside_the_census(record) -> None:
    """A census no longer than the last pass's step, no census, and a record whose windows name no steps."""
    for steps in (4, 0, True):
        with pytest.raises(ValueError, match="census"):
            audit.property_reading(record, steps)
    stepless = dataclasses.replace(record, windows=tuple(dataclasses.replace(w, steps=()) for w in record.windows))
    with pytest.raises(ValueError, match="name no steps"):
        audit.property_reading(stepless, 5)


def verdict(record: audit.Record, **kw) -> audit.PostSeed:
    """A verdict that holds, judged at the record's own reading (what `audit.post_seed` returns on the honest session)."""
    windows, L = audit.property_reading(record, 5)
    return audit.PostSeed(**{**dict(refused=(), program="w", census_steps=5, lifetime=L, windows=windows,
                                    checked={"ok": True}), **kw})


def test_the_ledger_refuses_a_verdict_judged_at_another_reading(record) -> None:
    """Binding 4: one more block read in window 0, one more step of L, or another census, and a verdict that fails."""
    v = verdict(record)
    more = (dataclasses.replace(v.windows[0], reads=v.windows[0].reads + 4096), *v.windows[1:])
    for other in (verdict(record, windows=more), verdict(record, lifetime=v.lifetime + 1),
                  verdict(record, census_steps=6, windows=audit.property_reading(record, 6, paced=True)[0])):
        with pytest.raises(ValueError, match="binding 4"):
            audit.ledger(record, **COMMON, credit_post_seed=other)
    with pytest.raises(ValueError, match="fails"):
        audit.ledger(record, **COMMON, credit_post_seed=verdict(record, refused=("mapped",)))


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason=audit.OPEN_CARRIER)
def test_the_ledger_credits_a_verdict_at_the_record_s_reading(record) -> None:
    """The verdict at the record's reading passes binding 4; its carrier bytes read L in seconds, the verdict's in steps."""
    audit.ledger(record, **COMMON, credit_post_seed=verdict(record))
