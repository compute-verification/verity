"""Red-team (REDTEAM-ADVICE, phase 11 target 1) re-running the phase-3 advised-counts cases against N2-ADVICE's floor
change (#3681): the floor bucket is now the Program's eligibility `b − δ − ⌊λ/ρ⌋` with the `max(1, ·)` dropped. That
`max(1, ·)` was what refused the phase-3 δ = 0 forgery (a release coincident with its request's last ingress frame).
Does any release now pass before its request's last ingress frame, at δ = 0 or under a sync the clock check accepts?

Held against the verifier's reference (`advised.view`, `advised.ends`, `advised.reference`, which circuit-check holds to
the circuit), the advice's validity and digest, and the tie stand-in (`advised_traffic.tied`, which stands in for
IoTieRequest's verdict as the verifier of record would take it).

Findings:
- strictly before: NO. `ClockSync`/`SyncSet` enforce δ ≥ 0, so for the first output a = b − δ ≤ b, and `reference`
  requires the ingress prefix sum at the release bucket to already reach `ends[j]`; a sweep over δ, ρ and both window
  sides accepts no release strictly before its request's last ingress frame;
- coincident (same bucket): YES at δ = 0, re-enabled by dropping `max(1, ·)` — the exact zero-serving-latency release
  the fix had refused. δ ≥ 1 still refuses it;
- the declared δ is not bound to the certified record in the unit or its reference (the audit's clock check is Σ_sync
  membership plus one-clock-per-spatial-unit consistency only), so a prover declares δ = 0 wherever δ = 0 ∈ Σ_sync;
- the phase-3 ingress-end forgery is CLOSED by the tie: `ends` is derived from v0's prompt lengths, and on one record
  only the certified split passes.
"""

from fractions import Fraction

import pytest

from verity.core.protocols.certifier import advice as A, advised as AD
from verity.core.protocols.certifier.certificate import Direction, RecordHead
from verity.core.protocols.certifier.grid import COMPLETE, Certifier, GridParams, occupancy
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet

from advised_traffic import tied

T, R, RIN, OWN = 8, 2, 4, 0
ONE = ClockSync(1, Fraction(1, 1))
ONLY_ONE = SyncSet((1,), (1,))
#: a request of n prompt tokens takes ⌈n/2⌉ ingress frames, for prompts of 1 … 8 tokens
FRAMING = tuple(-(-n // 2) for n in range(1, 9))


def _advice(release, prompts=(4,)):
    """One kept output per request, each a whole prefill of `prompts[k]` tokens in its own step; output k is request
    k's, released in bucket `release[k]`."""
    reqs = tuple(A.RequestAdvice(chr(ord("a") + i), prompts[i], 1, "eos") for i in range(len(prompts)))
    steps = tuple(A.StepAdvice((A.Row(i, "prefill", 0, prompts[i]),)) for i in range(len(prompts)))
    return A.Advice(reqs, steps, tuple(release))


def _link(adv, received, certified=None):
    return tied(AD.LinkAdvice(adv, tuple(received)), certified)


def _accepts(adv, received, ingress, egress, sync=ONE, sync_set=ONLY_ONE, own=OWN, certified=None, window=0):
    try:
        v = AD.view(_link(adv, received, certified), window, T, R, [sync] * len(received), sync_set, FRAMING)
    except ValueError:
        return False
    lo = window * T
    return AD.reference(T, RIN, own, ingress[lo:lo + T], egress[lo:lo + T], v.released, v.floor, v.received)


def test_the_honest_advised_window_is_accepted():
    """One request, its 2 frames released in bucket 2 and its output in bucket 3, the bucket after (δ = 1)."""
    assert _accepts(_advice((3,)), (2,), [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0])


def test_no_release_passes_strictly_before_its_requests_last_ingress_frame():
    """The phase-3 δ forgery, re-hunted. Over δ ∈ {0, 1, 2}, ρ ∈ {1, 2, 1/2, 3} and the request's 2 frames all released
    in bucket `anchor`, no output released in a bucket < `anchor` is accepted: δ ≥ 0 keeps the floor bucket
    b − δ − ⌊λ/ρ⌋ ≤ b, and `reference` requires the ingress prefix sum at the release bucket to reach `ends[j]`."""
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


def test_dropping_max_one_reopens_the_coincident_zero_latency_release_at_delta_zero():
    """SOUNDNESS GAP (the drop's residual). One request, its 2 frames certified in bucket 3, its output in bucket 3 —
    the same bucket. Under the honest certified-path latency (δ = 1, the proxy hands a frame read in bucket t at t + 1)
    it is refused: the floor needs the frames by bucket 2. Under δ = 0 it is accepted again — the zero-serving-latency
    release the `max(1, ·)` had refused. Nothing binds the declared δ to the record, so a prover declares δ = 0
    wherever δ = 0 ∈ Σ_sync (this PR adds it for the vLLM served window). A release in bucket 2, before the frames,
    stays refused under every δ."""
    ingress, egress, received = [0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], (2,)
    loose = SyncSet((0, 1), (1,))
    assert not _accepts(_advice((3,)), received, ingress, egress, ClockSync(1, 1), loose)
    assert _accepts(_advice((3,)), received, ingress, egress, ClockSync(0, 1), loose)
    before = [0, 0, 1, 0, 0, 0, 0, 0]
    for delta in (0, 1):
        assert not _accepts(_advice((2,)), received, before, before, ClockSync(delta, 1), loose)
    # the δ = 0 floor is the anchor bucket itself; the δ = 1 floor is one earlier.
    assert AD.view(_link(_advice((3,)), received), 0, T, R, [ClockSync(0, 1)], loose, FRAMING).floor[2:4] == (0, 2)
    assert AD.view(_link(_advice((3,)), received), 0, T, R, [ClockSync(1, 1)], loose, FRAMING).floor[2:4] == (2, 2)


def test_the_coincident_release_passes_with_the_own_frame_and_across_a_window_boundary():
    """The δ = 0 coincident release isn't an artifact of window 0 or OWN = 0. With OWN = 1 (the certifier's own frame
    in bucket 0) and the request's frames and output both in a non-zero bucket, δ = 0 accepts and δ = 1 refuses; and in
    window 1, frames and output both in that window's last bucket, δ = 0 accepts."""
    ingress = [1, 0, 0, 2, 0, 0, 0, 0]  # own frame in bucket 0, 2 request frames in bucket 3
    egress = [0, 0, 0, 1, 0, 0, 0, 0]
    loose = SyncSet((0, 1), (1,))
    assert _accepts(_advice((3,)), (2,), ingress, egress, ClockSync(0, 1), loose, own=1)
    assert not _accepts(_advice((3,)), (2,), ingress, egress, ClockSync(1, 1), loose, own=1)
    ncols = 2 * T
    ing = [0] * ncols
    ing[9] = 2
    eg = [0] * ncols
    eg[9] = 1
    assert _accepts(_advice((9,)), (0, 2), ing, eg, ClockSync(0, 1), loose, window=1)


def test_the_ingress_end_forgery_is_closed_by_the_tie():
    """Phase-3 finding #1, now CLOSED. Two requests of 2 frames each, released in buckets 1 and 3; one output in bucket
    3 and one in bucket 5. On the certified record (prompts 4, 4) the link's first request's output at 3 and its
    second's at 5 are causal and accepted. A prover who swaps the split to lower the first request's end (claiming a
    1-token and a 7-token prompt) can no longer forge the floor: `ends` is derived from v0's prompt lengths through the
    framing, and the tie refuses every split but the certified one, before any floor is computed."""
    ingress, egress, received = [0, 2, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0], (4,)
    assert _accepts(_advice((3, 5), (4, 4)), received, ingress, egress, certified=(4, 4))
    splits = [(p, q) for p in range(1, 9) for q in range(1, 9) if FRAMING[p - 1] + FRAMING[q - 1] == 4]
    passing = [s for s in splits if _accepts(_advice((3, 5), s), received, ingress, egress, certified=(4, 4))]
    assert passing == [(4, 4)]
    # a swapped split that would lower request 0's end is refused by the tie, not the floor.
    with pytest.raises(ValueError, match="doesn't accept"):
        AD.view(_link(_advice((3, 5), (1, 7)), received, certified=(4, 4)), 0, T, R, [ONE], ONLY_ONE, FRAMING)


def test_a_sync_outside_the_set_is_refused():
    """A declared sync outside Σ_sync is refused in whichever window releases under it, so the floor can't be relaxed by
    a value the Program didn't fix."""
    received = (2,)
    with pytest.raises(ValueError, match="Σ_sync"):
        AD.view(_link(_advice((3,)), received), 0, T, R, [ClockSync(0, 1)], SyncSet((2,), (1,)), FRAMING)


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
    v = AD.view(_link(_advice((2,), (8,)), (RIN,)), 0, T_, R, [ONE], ONLY_ONE, FRAMING)
    assert AD.reference(T_, RIN, 1, ingress, [0, 0, 1, 0], v.released, v.floor, v.received)
    assert not AD.reference(T_, RIN, 1, [RIN + 1, 0, 0, 0], [0, 0, 1, 0], v.released, v.floor, v.received)
    with pytest.raises(ValueError, match="counts of 0"):
        RecordHead("requests", Direction.INGRESS, 0, T_, RIN, 1, 1, COMPLETE, True, bytes(64), (RIN + 1, 0, 0, 0),
                   bytes(64))


def test_the_digest_distinguishes_every_input_and_the_egress_counts_are_pinned():
    """CLOSED. The advice `digest` moves with `release`, the prompt lengths the ends derive from, and `received`, so
    `network_verdict`'s one-advice AND refuses a cross-window swap; and `reference` requires `egress == view.released`,
    so one frame off in any bucket is refused."""
    base = AD.digest(AD.LinkAdvice(_advice((3, 5), (4, 4)), (4,)))
    assert AD.digest(AD.LinkAdvice(_advice((4, 5), (4, 4)), (4,))) != base
    assert AD.digest(AD.LinkAdvice(_advice((3, 5), (2, 6)), (4,))) != base
    assert AD.digest(AD.LinkAdvice(_advice((3, 5), (4, 4)), (3, 1))) != base
    ingress, received = [0, 2, 0, 2, 0, 0, 0, 0], (4,)
    v = AD.view(_link(_advice((3, 5), (4, 4)), received), 0, T, R, [ONE], ONLY_ONE, FRAMING)
    assert AD.reference(T, RIN, OWN, ingress, list(v.released), v.released, v.floor, v.received)
    for t in range(T):
        for d in (-1, 1):
            bad = list(v.released)
            if bad[t] + d >= 0:
                bad[t] += d
                assert not AD.reference(T, RIN, OWN, ingress, bad, v.released, v.floor, v.received), (t, d)
