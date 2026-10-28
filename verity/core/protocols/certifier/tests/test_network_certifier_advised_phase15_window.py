"""The advised link's sync check (`advised`): a window whose declared sync differs from the one the certifier ran it
under is refused alone, not with every window of its link, as it would be were each window's floor to read the
declaration of every window that releases into it. Only that backward dependency is real: output λ released at b in
window v needs its request's frames by b − δ − ⌊λ/ρ⌋, which can fall in an earlier window, so a floor read from v's
declaration carries v's word into that window's view. The carry forward reads no sync: every floor bucket is at most
its release bucket, so a later window's carried floor is the same under any δ ≥ 0. So every floor reads the sync
the certifier ran (`LinkWindows.clock_source`), and a declaration is checked in its own window alone, still
fail-closed: the network verdict is the AND over the windows, so it refuses the link with that window.
"""

from fractions import Fraction

import pytest

from verity.core.protocols.certifier import advice as A, advised as AD, service as SVC
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet

import standin
from advised_traffic import SYNCS, StandInClocks, held, serve
from test_network_certifier_advised import Run

T, R, RIN, OWN = 8, 2, 4, 0
INSTANT, ONE = ClockSync(0, Fraction(1)), ClockSync(1, Fraction(1))
LOOSE = SyncSet((0, 1), (1,))
#: a request of n prompt tokens takes ⌈n/2⌉ ingress frames, for prompts of 1 … 8 tokens
FRAMING = tuple(-(-n // 2) for n in range(1, 9))
#: one request of 4 tokens, its 2 frames certified in bucket T, window 1's first, and its one output released there
INGRESS, EGRESS = [0] * T + [2] + [0] * (T - 1), [0] * T + [1] + [0] * (T - 1)
COINCIDENT = AD.LinkAdvice(A.Advice((A.RequestAdvice("a", 4, 1, "eos"),), (A.StepAdvice((A.Row(0, "prefill", 0, 4),)),),
                                    (T,)), (0, 2))


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


def _view(window, declared, ran):
    return AD.view(COINCIDENT, window, T, R, declared, LOOSE, FRAMING, clocks=ran, **held(COINCIDENT))


def _accepts(window, declared, ran):
    try:
        v = _view(window, declared, ran)
    except ValueError:
        return False
    lo = window * T
    return AD.reference(T, RIN, OWN, INGRESS[lo:lo + T], EGRESS[lo:lo + T], v.released, v.floor, v.received)


def test_a_later_windows_declaration_moves_no_earlier_windows_floor():
    """Under δ = 0 certified for window 1 (a served clock) the coincident release is causal and both windows accept
    it; under δ = 1 its floor bucket is T − 1, window 0's last, which the record doesn't meet, so window 0 refuses it.
    Window 0's view and verdict are the same whichever sync window 1 declares, so δ = 0 declared for a window run at
    δ = 1 can't loosen window 0's floor, as it would if the floor read the declaration, and window 1 accepts only its
    certified sync."""
    for ran in (INSTANT, ONE):
        clocks = [ONE, ran]
        for declared in (INSTANT, ONE):
            syncs = [ONE, declared]
            assert _view(0, syncs, clocks) == _view(0, clocks, clocks)
            assert _accepts(0, syncs, clocks) is (ran == INSTANT), (ran, declared)
            assert _accepts(1, syncs, clocks) is (declared == ran), (ran, declared)
    assert _view(0, [ONE, INSTANT], [ONE, ONE]).floor[T - 1] == 2
    assert _view(0, [ONE, INSTANT], [ONE, INSTANT]).floor == (0,) * T
    with pytest.raises(ValueError, match="the certifier ran it under ClockSync"):
        _view(1, [ONE, INSTANT], [ONE, ONE])


def test_a_window_declaring_off_its_certified_sync_is_refused_alone():
    """Window 0 declaring δ = 0 where the certifier ran δ = 1, or a sync outside Σ_sync, is refused, and window 1's
    view is the honest one; a window with no declaration is refused alone too."""
    clocks = [ONE, INSTANT]
    honest = _view(1, clocks, clocks)
    for declared in (INSTANT, ClockSync(2, 1), ClockSync(1, 2)):
        with pytest.raises(ValueError, match="Σ_sync|the certifier ran it under"):
            _view(0, [declared, INSTANT], clocks)
        assert _view(1, [declared, INSTANT], clocks) == honest
    with pytest.raises(ValueError, match="window 1 declares no sync"):
        _view(1, [ONE], clocks)
    assert _view(0, [ONE], clocks) == _view(0, clocks, clocks)


def test_the_verifier_of_record_refuses_only_the_window_run_off_its_declaration(tmp_path):
    """Through the proof system (stand-in proofs) on simulated serving: the clock source says the certifier ran window
    2 at δ = 1, one of Σ_sync, where the prover declared δ = 2. Window 2 is refused and the rest accepted on the floors
    of the syncs the certifier ran, so the network verdict refuses the link (`Run.link` holds it to the AND)."""
    ran = list(SYNCS)
    ran[2] = ClockSync(1, ran[2].rho)
    assert ran[2] != SYNCS[2]
    assert Run(tmp_path, clocks=StandInClocks(tuple(ran))).link(serve(1)) == [True, True, False, True]
