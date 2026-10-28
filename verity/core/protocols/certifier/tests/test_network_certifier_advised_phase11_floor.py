"""The advised link's floor (`advised`), held as regressions. The floor bucket is the Program's eligibility
b − δ − ⌊λ/ρ⌋ with no `max(1, ·)`, so a release in its request's own bucket is accepted at δ = 0, a served clock's (an
engine's first output in its request's bucket). That is sound only for the δ the window's traffic was certified under:
a floor that read the declared δ, held to Σ_sync alone, would accept an output coincident with its request's last
ingress frame in a window the certifier ran at δ = 1 and the prover declared at δ = 0. So every floor reads the sync
the certifier ran, which the verifier of record takes only from its own clock source (`LinkWindows.clock_source`, here
the stand-in `advised_traffic.StandInClocks`; of record, the certifier's signed clock syncs,
`test_network_certifier_advised_signed_clocks.py`), and `advised.view` refuses a window whose declared sync isn't that
one (only that window: `test_network_certifier_advised_phase15_window.py`).

The rest stays closed: no release strictly before its request's last ingress frame passes under any δ ≥ 0, the
ingress-end forgery (`test_network_certifier_advised_adversarial.py`) is refused by the tie, a sync outside Σ_sync is
refused, bucket 0 holds the certifier's own frame within R_in, and the advice digest moves with every input it reads.
"""

import inspect
from fractions import Fraction

import pytest

from verity.core.protocols.certifier import advice as A, advised as AD, service as SVC
from verity.core.protocols.certifier.certificate import Direction, RecordHead
from verity.core.protocols.certifier.grid import COMPLETE, Certifier, GridParams, occupancy
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.certifier.service import LinkWindows

import signers
import standin
from advised_traffic import LINK, SYNCS, WINDOWS, StandInClocks, StandInTie, held, serve
from test_network_certifier_advised import Run

T, R, RIN, OWN = 8, 2, 4, 0
ONE = ClockSync(1, Fraction(1, 1))
ONLY_ONE = SyncSet((1,), (1,))
LOOSE = SyncSet((0, 1), (1,))
#: a request of n prompt tokens takes ⌈n/2⌉ ingress frames, for prompts of 1 … 8 tokens
FRAMING = tuple(-(-n // 2) for n in range(1, 9))
#: one request, its 2 frames certified in bucket 3, and its output released in bucket 3, the same bucket
COINCIDENT = ([0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], (2,))


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


def _advice(release, prompts=(4,)):
    """One kept output per request, each a whole prefill of `prompts[k]` tokens in its own step; output k is request
    k's, released in bucket `release[k]`."""
    reqs = tuple(A.RequestAdvice(chr(ord("a") + i), prompts[i], 1, "eos") for i in range(len(prompts)))
    steps = tuple(A.StepAdvice((A.Row(i, "prefill", 0, prompts[i]),)) for i in range(len(prompts)))
    return A.Advice(reqs, steps, tuple(release))


def _view(adv, received, sync=ONE, sync_set=ONLY_ONE, ran=None, certified=None, window=0):
    """`advised.view` of `adv`'s `window`, every window declaring `sync` and run by the certifier under `ran` (by
    default `sync`), under the stand-in tie on an ingress that certified the prompts `certified`."""
    link = AD.LinkAdvice(adv, tuple(received))
    n = len(received)
    return AD.view(link, window, T, R, [sync] * n, sync_set, FRAMING, clocks=[ran or sync] * n,
                   **held(link, certified))


def _accepts(adv, received, ingress, egress, sync=ONE, sync_set=ONLY_ONE, own=OWN, ran=None, certified=None,
             window=0):
    try:
        v = _view(adv, received, sync, sync_set, ran, certified, window)
    except ValueError:
        return False
    lo = window * T
    return AD.reference(T, RIN, own, ingress[lo:lo + T], egress[lo:lo + T], v.released, v.floor, v.received)


def test_the_honest_advised_window_is_accepted():
    """One request, its 2 frames released in bucket 2 and its output in bucket 3, the bucket after (δ = 1)."""
    assert _accepts(_advice((3,)), (2,), [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0])


def test_no_release_passes_strictly_before_its_requests_last_ingress_frame():
    """Over δ ∈ {0, 1, 2}, ρ ∈ {1, 2, 1/2, 3} and the request's 2 frames all released in bucket `anchor`, no output
    released in a bucket before `anchor` is accepted, the certifier having run the declared sync: δ ≥ 0 keeps the floor
    bucket b − δ − ⌊λ/ρ⌋ ≤ b, and `reference` needs the ingress prefix sum at the release bucket to reach `ends[j]`."""
    accepted = []
    for delta in (0, 1, 2):
        for rho in (Fraction(1), Fraction(2), Fraction(1, 2), Fraction(3)):
            for anchor in range(1, 6):
                ingress = [0] * T
                ingress[anchor] = FRAMING[4 - 1]
                for b in range(anchor):
                    egress = [0] * T
                    egress[b] = 1
                    if _accepts(_advice((b,)), (FRAMING[3],), ingress, egress, ClockSync(delta, rho),
                                SyncSet((delta,), (rho,))):
                        accepted.append((delta, str(rho), anchor, b))
    assert accepted == [], accepted


def test_a_window_declaring_a_delta_below_the_one_it_ran_under_is_refused():
    """The phase's gap, closed. The coincident release is accepted where the certifier ran the window at δ = 0 (a
    served clock), and refused at δ = 1 by the floor, which needs the frames by bucket 2. Declared δ = 0 for the window
    the certifier ran at δ = 1, it was accepted, since the floor read the declaration; now the view refuses the
    window, whatever its counts, and the δ = 1 floor is the one the record is held to."""
    ingress, egress, received = COINCIDENT
    adv = _advice((3,))
    assert _accepts(adv, received, ingress, egress, ClockSync(0, 1), LOOSE)
    assert not _accepts(adv, received, ingress, egress, ClockSync(1, 1), LOOSE)
    assert not _accepts(adv, received, ingress, egress, ClockSync(0, 1), LOOSE, ran=ClockSync(1, 1))
    with pytest.raises(ValueError, match="the certifier ran it under ClockSync"):
        _view(adv, received, ClockSync(0, 1), LOOSE, ran=ClockSync(1, 1))
    assert _view(adv, received, ClockSync(0, 1), LOOSE).floor[2:4] == (0, 2)
    assert _view(adv, received, ClockSync(1, 1), LOOSE).floor[2:4] == (2, 2)
    before = [0, 0, 1, 0, 0, 0, 0, 0]
    for delta in (0, 1):
        assert not _accepts(_advice((2,)), received, before, before, ClockSync(delta, 1), LOOSE)


def test_a_declaration_other_than_the_certified_one_is_refused_either_way():
    """The binding is equality, not a bound: a declared δ above the certified one, or another ρ, is refused too, so
    the floor reads exactly the sync the record was certified under."""
    ingress, egress = [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]
    sync_set = SyncSet((0, 1), (1, 2))
    assert _accepts(_advice((4,)), (2,), ingress, egress, ONE, sync_set)
    for declared, ran in ((ONE, ClockSync(0, 1)), (ONE, ClockSync(1, 2)), (ClockSync(1, 2), ONE)):
        assert not _accepts(_advice((4,)), (2,), ingress, egress, declared, sync_set, ran=ran), (declared, ran)


def test_the_coincident_release_needs_delta_zero_certified_with_the_own_frame_and_in_a_later_window():
    """With OWN = 1 (the certifier's own frame in bucket 0) and the request's frames and output both in bucket 3, the
    release is accepted only on a window run at δ = 0; in window 1, frames and output both in its second bucket, the
    same."""
    ingress, egress = [1, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]
    adv = _advice((3,))
    assert _accepts(adv, (2,), ingress, egress, ClockSync(0, 1), LOOSE, own=1)
    assert not _accepts(adv, (2,), ingress, egress, ClockSync(1, 1), LOOSE, own=1)
    assert not _accepts(adv, (2,), ingress, egress, ClockSync(0, 1), LOOSE, own=1, ran=ClockSync(1, 1))
    ing, eg = [0] * 2 * T, [0] * 2 * T
    ing[9], eg[9] = 2, 1
    assert _accepts(_advice((9,)), (0, 2), ing, eg, ClockSync(0, 1), LOOSE, window=1)
    assert not _accepts(_advice((9,)), (0, 2), ing, eg, ClockSync(0, 1), LOOSE, ran=ClockSync(1, 1), window=1)


def test_a_view_without_the_certified_syncs_or_past_them_is_refused():
    """The certified syncs are never optional: no clock source refuses the advice, as does a release in a window the
    certifier reports no sync for."""
    link = AD.LinkAdvice(_advice((3,)), (2,))
    with pytest.raises(ValueError, match="clock source"):
        AD.view(link, 0, T, R, [ONE], ONLY_ONE, FRAMING, clocks=None, **held(link))
    later = AD.LinkAdvice(_advice((T + 3,)), (2, 0))
    with pytest.raises(ValueError, match="the certifier ran it under no sync"):
        AD.view(later, 0, T, R, [ONE, ONE], ONLY_ONE, FRAMING, clocks=[ONE], **held(later))


def _link(key, clocks):
    """A one-window link of this case's shape whose verifier holds the stand-in tie and the clock source `clocks`."""
    return LinkWindows(LINK, GridParams(T, R, 1), LOOSE, key, "r-phase11", signers.SESSION,
                       ingress=GridParams(T, RIN, 1), proof_deadline=60.0, advised=AD.AdvisedProfile(1, FRAMING, 0),
                       ties=StandInTie(windows=1), clocks=clocks)


def test_the_verifier_of_record_takes_the_certified_syncs_only_from_its_own_source(tmp_path):
    """`LinkWindows._view`, which `public_arguments`, `arguments` and `advised_check` read, holds the declared sync to
    its clock source's: δ = 0 declared for a window run at δ = 1 has no view, nor does any advised window without a
    source, as in the package; and no entry point of the verifier of record takes the certified syncs beside what the
    prover delivers."""
    key = signers.signer(tmp_path).public_key()
    link, instant = AD.LinkAdvice(_advice((3,)), (2,)), [ClockSync(0, 1)]
    assert _link(key, StandInClocks((ClockSync(0, 1),)))._view(0, instant, link) == \
        _view(_advice((3,)), (2,), ClockSync(0, 1), LOOSE)
    assert _link(key, StandInClocks((ClockSync(1, 1),)))._view(0, instant, link) is None
    assert _link(key, None)._view(0, instant, link) is None
    for entry in (LinkWindows.verify, LinkWindows.clear, LinkWindows.advised_check, LinkWindows.outcome,
                  LinkWindows.arguments, LinkWindows.public_arguments, LinkWindows.unproved_outcome):
        assert "clocks" not in inspect.signature(entry).parameters, entry.__name__


def test_the_verifier_of_record_refuses_a_link_declared_off_its_certified_syncs(tmp_path):
    """Through the proof system (stand-in proofs): an honest session is accepted in every window where the clock
    source holds the declared syncs, and refused in every window where it says the certifier ran each window at a δ
    above the declared one, or where there is no source."""
    session = serve(1)
    for name in ("ran", "slower", "none"):
        (tmp_path / name).mkdir()
    assert Run(tmp_path / "ran", clocks=StandInClocks(tuple(SYNCS))).link(session) == [True] * WINDOWS
    slower = tuple(ClockSync(s.delta + 1, s.rho) for s in SYNCS)
    assert Run(tmp_path / "slower", clocks=StandInClocks(slower)).link(session) == [False] * WINDOWS
    assert Run(tmp_path / "none", clocks=None).link(session) == [False] * WINDOWS


def test_the_ingress_end_forgery_is_closed_by_the_tie():
    """Two requests of 2 frames each, released in buckets 1 and 3, one output in bucket 3 and one in bucket 5. On the
    certified record (prompts 4, 4) both are causal and accepted. A split that would lower the first request's end (a
    1-token and a 7-token prompt) is refused by the tie, before any floor is computed: `ends` derives from v0's prompt
    lengths through the framing, and only the certified split passes."""
    ingress, egress, received = [0, 2, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0], (4,)
    assert _accepts(_advice((3, 5), (4, 4)), received, ingress, egress, certified=(4, 4))
    splits = [(p, q) for p in range(1, 9) for q in range(1, 9) if FRAMING[p - 1] + FRAMING[q - 1] == 4]
    passing = [s for s in splits if _accepts(_advice((3, 5), s), received, ingress, egress, certified=(4, 4))]
    assert passing == [(4, 4)]
    with pytest.raises(ValueError, match="doesn't accept"):
        _view(_advice((3, 5), (1, 7)), received, certified=(4, 4))


def test_a_sync_outside_the_set_is_refused():
    """A declared sync outside Σ_sync is refused in the window that declares it, even one the certifier ran: the floor
    can't be relaxed by a value the Program didn't fix."""
    with pytest.raises(ValueError, match="Σ_sync"):
        _view(_advice((3,)), (2,), ClockSync(0, 1), SyncSet((2,), (1,)))


def test_bucket_zero_is_capped_at_rin_with_the_certifiers_own_frame():
    """R_in already holds the own frame: R_in request frames arriving in bucket 0 certify as R_in there and 1 in bucket
    1, and the honest window is accepted. A count of R_in + 1 isn't a certified count (no head holds it)."""
    T_, params = 4, GridParams(buckets=4, allocation=RIN, frame_bytes=1)
    cert = Certifier("requests", params)
    cert.enqueue(b"s")
    for _ in range(RIN):
        cert.enqueue(b"q")
    for _ in range(T_):
        cert.tick()
    ingress = list(occupancy(cert.close_window().grid))
    assert ingress == [RIN, 1, 0, 0]
    link = AD.LinkAdvice(_advice((2,), (8,)), (RIN,))
    v = AD.view(link, 0, T_, R, [ONE], ONLY_ONE, FRAMING, clocks=[ONE], **held(link))
    assert AD.reference(T_, RIN, 1, ingress, [0, 0, 1, 0], v.released, v.floor, v.received)
    assert not AD.reference(T_, RIN, 1, [RIN + 1, 0, 0, 0], [0, 0, 1, 0], v.released, v.floor, v.received)
    with pytest.raises(ValueError, match="counts of 0"):
        RecordHead("requests", Direction.INGRESS, 0, T_, RIN, 1, 1, COMPLETE, True, bytes(64), (RIN + 1, 0, 0, 0),
                   bytes(64))


def test_the_digest_distinguishes_every_input_and_the_egress_counts_are_pinned():
    """The advice `digest` moves with `release`, the prompt lengths the ends derive from, and `received`, so
    `network_verdict`'s one-advice AND refuses a cross-window swap; and `reference` needs `egress == view.released`,
    so one frame off in any bucket is refused."""
    base = AD.digest(AD.LinkAdvice(_advice((3, 5), (4, 4)), (4,)))
    assert AD.digest(AD.LinkAdvice(_advice((4, 5), (4, 4)), (4,))) != base
    assert AD.digest(AD.LinkAdvice(_advice((3, 5), (2, 6)), (4,))) != base
    assert AD.digest(AD.LinkAdvice(_advice((3, 5), (4, 4)), (3, 1))) != base
    ingress = [0, 2, 0, 2, 0, 0, 0, 0]
    v = _view(_advice((3, 5), (4, 4)), (4,))
    assert AD.reference(T, RIN, OWN, ingress, list(v.released), v.released, v.floor, v.received)
    for t in range(T):
        for d in (-1, 1):
            bad = list(v.released)
            if bad[t] + d >= 0:
                bad[t] += d
                assert not AD.reference(T, RIN, OWN, ingress, bad, v.released, v.floor, v.received), (t, d)
