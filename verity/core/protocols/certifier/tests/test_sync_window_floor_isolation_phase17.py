"""REDTEAM phase 17, Target 2 (#3870, N2-ADVICE): per-window δ. The fix makes every floor read the sync the certifier
ran (`LinkWindows.clocks`), and checks a window's declared sync in its own window alone. The PR's own tests swap one
window's declaration at a time; these add the angles they don't: swapping several windows' declarations at once, the
most tempting wrong declaration (a looser sync that is still in Σ_sync), and the whole-link refusal end to end through
the network verdict when an output is released past the link's windows.

The guiding claim under test: a window's declared sync never enters another window's `view`, and a prover declaring a
looser sync than the certifier ran gains nothing — the declaration is refused, and the floor is the certified sync's."""

from dataclasses import replace
from fractions import Fraction

import pytest

from verity.core.protocols.certifier import advised as AD, service as SVC
from verity.core.protocols.certifier.schedule import ClockSync

import standin
from advised_traffic import LINK, SYNC_SET, SYNCS, T, WINDOWS, serve
from test_network_certifier_advised import Run, _view


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


def test_swapping_several_windows_declarations_at_once_refuses_exactly_them(tmp_path):
    """Windows 0 and 2 declare wrong syncs at the same time (0 one outside Σ_sync, 2 one inside Σ_sync but not the
    certified one); both are refused and windows 1 and 3 keep the honest view, so even several bad declarations together
    move no honest window's floor."""
    link = serve(1).link
    honest = [_view(link, w, SYNCS, clocks=SYNCS) for w in range(WINDOWS)]
    off = list(SYNCS)
    off[0] = ClockSync(0, Fraction(2))          # outside Σ_sync
    off[2] = SYNCS[1]                            # inside Σ_sync, but the certifier ran SYNCS[2]
    assert off[0] not in SYNC_SET and off[2] in SYNC_SET and off[2] != SYNCS[2]
    for w in (0, 2):
        with pytest.raises(ValueError, match=r"Σ_sync|the certifier ran it under"):
            _view(link, w, off, clocks=SYNCS)
    for w in (1, 3):
        assert _view(link, w, off, clocks=SYNCS) == honest[w]


def test_a_looser_declaration_still_in_the_sync_set_is_refused_and_isolated(tmp_path):
    """The sharp case for a prover: declare a sync that is in Σ_sync but looser (a lower δ) than the one the certifier
    ran, hoping the floor reads the declaration. Every such window is refused ('the certifier ran it under'), and the
    other windows' views are byte-for-byte the honest ones, so the looser declaration neither lowers its own floor nor
    any other window's."""
    link = serve(1).link
    honest = [_view(link, w, SYNCS, clocks=SYNCS) for w in range(WINDOWS)]
    tried = 0
    for w in range(WINDOWS):
        for looser in [s for s in SYNC_SET if s.delta < SYNCS[w].delta]:
            tried += 1
            off = SYNCS[:w] + [looser] + SYNCS[w + 1:]
            with pytest.raises(ValueError, match="the certifier ran it under"):
                _view(link, w, off, clocks=SYNCS)
            for other in range(WINDOWS):
                if other != w:
                    assert _view(link, other, off, clocks=SYNCS) == honest[other], (w, looser, other)
    assert tried >= 2, "Σ_sync should offer at least one looser sibling to try"


def test_an_output_released_past_the_links_windows_refuses_the_whole_link_end_to_end(tmp_path):
    """An advice whose last output is released in bucket WINDOWS*T (past the last window) makes `_floor_buckets` raise
    for every window, so `Run.link` refuses the whole link and the network verdict (the AND over windows) agrees."""
    session = serve(1)
    a = session.link.advice
    bad = AD.LinkAdvice(replace(a, release=a.release[:-1] + (WINDOWS * T,)), session.link.received)
    assert Run(tmp_path).link(session, link=bad) == [False] * WINDOWS
