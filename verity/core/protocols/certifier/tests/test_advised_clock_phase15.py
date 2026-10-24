"""Red-team phase 15, target 5 (#3681): δ bound to the certified clock. The lead's two questions, both held shut; the
ported `test_network_certifier_advised_phase11_floor.py` already covers the coincident release, equality-not-a-bound, no
clock source, a too-short clocks list, and the verifier of record's uniform refusals (`slower`, `none`). These add the
per-window angle:

- no path takes the sync from the prover: `advised._floor_buckets` reads the certifier's `clocks` for every release in
  the link, so one window whose declared sync isn't the one the certifier ran it under refuses every window's view, not
  just its own — a prover can't view an honest window around a sibling it declared off the clock;
- without a clock source every advised link is refused: the verifier of record's `_view` is None for every window when a
  single window's declared sync disagrees with its own `ClockSource`, and when the source is absent.

Helpers are the phase-11 floor test's and `advised_traffic`'s stand-ins (the verifier's own tie and clock sources).
"""
from __future__ import annotations

from fractions import Fraction

import pytest

import signers
from verity.core.protocols.certifier import advised as AD
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync
from verity.core.protocols.certifier.service import LinkWindows

from advised_traffic import LINK, StandInClocks, StandInTie
from test_network_certifier_advised_phase11_floor import FRAMING, LOOSE, ONE, R, RIN, T, _advice, held

#: a two-window link: request a's output in window 0 (bucket 3), request b's in window 1 (bucket T + 3), each a
#: two-frame prompt received in its window
TWO = AD.LinkAdvice(_advice((3, T + 3), prompts=(4, 4)), (2, 2))
OFF = ClockSync(0, Fraction(1, 1))


def _view(declared, ran, window):
    return AD.view(TWO, window, T, R, declared, LOOSE, FRAMING, clocks=ran, **held(TWO, windows=2))


def test_phase15_a_clock_mismatch_in_one_window_refuses_every_window_of_the_link():
    """`_floor_buckets` scans all of the link's releases, so a window whose declared sync isn't the certifier's refuses
    every window's view. Window 1 declares a δ the certifier didn't run; both window 0's and window 1's views are
    refused, though window 0's own declaration matches. When every window matches, both are accepted."""
    for window in (0, 1):
        with pytest.raises(ValueError, match="window 1 declares"):
            _view([ONE, OFF], [ONE, ONE], window)
    assert all(_view([ONE, ONE], [ONE, ONE], window) is not None for window in (0, 1))


def _lw(key, clocks):
    return LinkWindows(LINK, GridParams(T, R, 1), LOOSE, key, "r-phase15", signers.SESSION,
                       ingress=GridParams(T, RIN, 1), proof_deadline=60.0,
                       advised=AD.AdvisedProfile(1, FRAMING, 0), ties=StandInTie(windows=2), clocks=clocks)


def test_phase15_the_verifier_of_record_refuses_the_whole_link_when_one_windows_clock_disagrees(tmp_path):
    """Through the verifier of record: its `_view`, which `arguments`, `public_arguments` and `advised_check` read,
    takes the syncs only from its own clock source. When the source says the certifier ran window 1 at a δ the advice
    didn't declare, `_view` is None for both windows (the prover's declaration never substitutes for the source); with a
    matching source both have a view, and with no source at all both are None."""
    key = signers.signer(tmp_path).public_key()
    declared = [ONE, ONE]
    matched = _lw(key, StandInClocks((ONE, ONE)))
    assert all(matched._view(window, declared, TWO) is not None for window in (0, 1))
    disagrees = _lw(key, StandInClocks((ONE, OFF)))
    assert all(disagrees._view(window, declared, TWO) is None for window in (0, 1))
    assert all(_lw(key, None)._view(window, declared, TWO) is None for window in (0, 1))
