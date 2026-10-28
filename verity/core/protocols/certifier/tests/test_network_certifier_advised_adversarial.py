"""Adversarial cases against N2's advised counts (`advised`), each held as a regression of what the verifier pins: the
verifier's reference (`advised.view`, `advised.ends` and `advised.reference`, which circuit-check holds to the circuit)
and the advice's validity and digest (`advised.LinkAdvice`, `advised.digest`), the CPU-only surfaces a malicious
prover reaches.

- The ingress ends are derived from v0's prompt lengths through the link's public framing, request j the link's j-th,
  the last the certified total, and the tie (IoTieRequest's verdict, `advised.IoTieVerdict`, here its stand-in,
  `advised_traffic.held`) holds those lengths to the certified prompts', so
  on one record one split passes, and an output released before its request's frames is refused whatever lengths
  the advice states.
- A declared sync is one of Σ_sync, and the floor is the Program's eligibility under it (δ ≥ 0), so no declaration
  lets a frame out before the bucket that releases its request's last frame.
- Bucket 0's cap R_in holds the certifier's own frame: no certified count passes it.
- The digest distinguishes every input, and the egress counts are the advice's releases.
"""

from dataclasses import fields
from fractions import Fraction

import pytest

from verity.core.protocols.certifier import advice as A, advised as AD
from verity.core.protocols.certifier.certificate import Direction, RecordHead
from verity.core.protocols.certifier.grid import COMPLETE, Certifier, GridParams, fill, occupancy
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet

from advised_traffic import held

T, R, RIN, OWN = 8, 2, 4, 0
ONE = ClockSync(1, Fraction(1, 1))
ONLY_ONE = SyncSet((1,), (1,))
#: a request of n prompt tokens takes ⌈n/2⌉ ingress frames, for prompts of 1 … 8 tokens
FRAMING = tuple(-(-n // 2) for n in range(1, 9))
CAUSAL = [0, 2, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0], (4,)


def _advice(release, n=1, prompts=None):
    """`n` requests, one kept output each, each a whole prefill in its own step, of `prompts[k]` tokens (4, two
    frames, by default); output k is request k's, released in bucket `release[k]`."""
    prompts = prompts or (4,) * n
    reqs = tuple(A.RequestAdvice(chr(ord("a") + i), prompts[i], 1, "eos") for i in range(n))
    steps = tuple(A.StepAdvice((A.Row(i, "prefill", 0, prompts[i]),)) for i in range(n))
    return A.Advice(reqs, steps, tuple(release))


def _ends(adv, received, certified=None):
    """`advised.ends` of `adv` under IoTieRequest's verdict (its stand-in, `advised_traffic.held`) on a record of the
    prompts `certified`, by default `adv`'s own: what passes with no tie to the record."""
    link = AD.LinkAdvice(adv, received)
    return AD.ends(link, FRAMING, **held(link, certified))


def _view(adv, received, sync=ONE, sync_set=ONLY_ONE, T_=T, certified=None):
    """`advised.view` of `adv`'s window 0 under the verdict `_ends` reads, on a window the certifier ran under
    `sync`."""
    link = AD.LinkAdvice(adv, received)
    return AD.view(link, 0, T_, R, [sync], sync_set, FRAMING, clocks=[sync], **held(link, certified))


def _accepts(adv, received, ingress, egress, sync=ONE, sync_set=ONLY_ONE, own=OWN, certified=None):
    try:
        v = _view(adv, received, sync, sync_set, certified=certified)
    except ValueError:
        return False
    return AD.reference(T, RIN, own, ingress, egress, v.released, v.floor, v.received)


def _two_frame_splits(n=2, total=4):
    """Every pair of prompt lengths the framing admits whose frames sum to `total`."""
    return [(p, q) for p in range(1, len(FRAMING) + 1) for q in range(1, len(FRAMING) + 1)
            if FRAMING[p - 1] + FRAMING[q - 1] == total]


def test_the_honest_advised_window_is_accepted():
    """One request, its 2 frames released in bucket 2 and its output in bucket 3, the bucket after."""
    assert _accepts(_advice((3,)), (2,), [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0])


def test_an_output_released_before_its_requests_frames_is_refused_whatever_prompt_lengths_the_advice_states():
    """Two requests of 2 frames, released in buckets 1 and 3; one output released in bucket 3 and one in bucket 5. The
    link's first request's output at 3 and its second's at 5 are causal and accepted. The second's at 3, before its
    frames were in, is refused: the advice has no ingress end to state (only v0's fields, `received` and the tie), and
    the last request's end is the link's certified total whatever its prompt lengths say, so no split of the 4 frames
    between the two requests passes, even under a verdict on a record of the split's lengths. Which body an output answers is
    the I/O tie's to fix: advice request j is the link's j-th."""
    ingress, egress, received = CAUSAL
    assert [f.name for f in fields(AD.LinkAdvice)] == ["advice", "received"]
    assert _accepts(_advice((3, 5), 2), received, ingress, egress)
    assert not _accepts(_advice((5, 3), 2), received, ingress, egress)
    splits = _two_frame_splits()
    assert {_ends(_advice((5, 3), 2, s), received) for s in splits} == {(1, 4), (2, 4), (3, 4)}
    for prompts in splits:
        assert not _accepts(_advice((5, 3), 2, prompts), received, ingress, egress), prompts


def test_on_one_record_only_the_certified_split_passes():
    """On one certified record of two 4-token prompts, the advices whose stated lengths would pass (each a distinct v0
    string and digest) differ only in v0's prompt lengths; under the record's tie only the certified split does, and
    every other is refused by the tie (IoTieRequest's verdict) before any floor is computed. A view is a function of
    v0's encoding, `received` and the tie, so one v0 string has one floor."""
    ingress, egress, received = CAUSAL
    certified = (4, 4)
    candidates = [_advice((3, 5), 2, s) for s in _two_frame_splits()]
    stated = [a for a in candidates if _accepts(a, received, ingress, egress)]
    assert len(stated) >= 2
    assert len({A.encode(a) for a in stated}) == len(stated)
    assert len({AD.digest(AD.LinkAdvice(a, received)) for a in stated}) == len(stated)
    assert [a for a in candidates if _accepts(a, received, ingress, egress, certified=certified)] == [_advice((3, 5), 2)]
    for a in candidates:
        if a != _advice((3, 5), 2):
            with pytest.raises(ValueError, match="doesn't accept"):
                _view(a, received, certified=certified)
    for a in stated:
        assert _view(A.decode(A.encode(a)), received) == _view(a, received)


def test_no_declared_sync_releases_a_frame_before_its_requests_last_frame():
    """One request, its 2 frames certified in bucket 3. Released in bucket 3 under the honest latency (δ = 2) it
    needed the frames by bucket 1, and is refused: a declared δ binds. Under an instant sync (δ = 0) the floor is the
    Program's eligibility, the anchor bucket itself, so the release in bucket 3 is accepted (a served first token in
    its request's bucket), and one in bucket 2, before the request's last frame, is refused, under δ = 0 as under
    every sync, since δ ≥ 0 never moves the floor past the release bucket. A sync outside Σ_sync is refused outright,
    in the window that declares it and only there: an earlier window's floor reads the sync the certifier ran."""
    ingress, egress, received = [0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], (2,)
    loose = SyncSet((0, 2), (1,))
    assert not _accepts(_advice((3,)), received, ingress, egress, ClockSync(2, 1), loose)
    assert _accepts(_advice((3,)), received, ingress, egress, ClockSync(0, 1), loose)
    for sync in (ClockSync(0, 1), ClockSync(2, 1)):
        assert not _accepts(_advice((2,)), received, ingress, [0, 0, 1, 0, 0, 0, 0, 0], sync, loose)
    assert _view(_advice((3,)), received, ClockSync(0, 1), loose).floor[2:4] == (0, 2)
    with pytest.raises(ValueError, match="Σ_sync"):
        _view(_advice((4,)), received, ClockSync(0, 1), SyncSet((2,), (1,)))
    later = AD.LinkAdvice(_advice((T + 3,)), (2, 0))
    off = [ONE, ClockSync(0, 1)]
    with pytest.raises(ValueError, match="Σ_sync"):
        AD.view(later, 1, T, R, off, ONLY_ONE, FRAMING, clocks=[ONE, ONE], **held(later))
    assert AD.view(later, 0, T, R, off, ONLY_ONE, FRAMING, clocks=[ONE, ONE], **held(later)) == \
        AD.view(later, 0, T, R, [ONE, ONE], ONLY_ONE, FRAMING, clocks=[ONE, ONE], **held(later))


def test_bucket_zero_is_capped_at_rin_with_the_certifiers_own_frame():
    """R_in already holds the own frame: the ingress certifier queues it ahead of bucket 0's arrivals and releases
    min(queue, R_in) a bucket, so R_in request frames arriving in bucket 0 are certified as R_in there (the own frame
    and R_in − 1 of them) and 1 in bucket 1, and the honest window is accepted. A count of R_in + 1 isn't a certified
    count: no grid or head holds it."""
    T_, params = 4, GridParams(buckets=4, allocation=RIN, frame_bytes=1)
    cert = Certifier("requests", params)
    cert.enqueue(b"s")
    for _ in range(RIN):
        cert.enqueue(b"q")
    for _ in range(T_):
        cert.tick()
    ingress = list(occupancy(cert.close_window().grid))
    assert ingress == [RIN, 1, 0, 0]
    a = _advice((2,), 1, (8,))
    v = _view(a, (RIN,), T_=T_)
    assert AD.reference(T_, RIN, 1, ingress, [0, 0, 1, 0], v.released, v.floor, v.received)
    assert not AD.reference(T_, RIN, 1, [RIN + 1, 0, 0, 0], [0, 0, 1, 0], v.released, v.floor, v.received)
    with pytest.raises(ValueError, match="a bucket holds"):
        fill([b"q"] * (RIN + 1), [RIN + 1, 0, 0, 0], RIN)
    with pytest.raises(ValueError, match="counts of 0"):
        RecordHead("requests", Direction.INGRESS, 0, T_, RIN, 1, 1, COMPLETE, True, bytes(64), (RIN + 1, 0, 0, 0),
                   bytes(64))


def test_the_advice_digest_distinguishes_every_input_so_cross_window_swaps_are_refused():
    """Every window reports `advised.digest` of the whole advice, and `network_verdict` holds every window to the
    first's. The digest moves with `release`, with the prompt lengths the ends are derived from, and with `received`,
    so a window can't carry a different advice under the first window's digest. The tie isn't in it, or in the
    advice: it is IoTieRequest's verdict on the advice of that digest, from the verifier's own tie source."""
    base = AD.digest(AD.LinkAdvice(_advice((3, 5), 2), (4,)))
    assert held(AD.LinkAdvice(_advice((3, 5), 2), (4,)))["tie"].advice == base
    assert AD.digest(AD.LinkAdvice(_advice((4, 5), 2), (4,))) != base
    assert AD.digest(AD.LinkAdvice(_advice((3, 5), 2, (2, 6)), (4,))) != base
    assert AD.digest(AD.LinkAdvice(_advice((3, 5), 2), (3, 1))) != base
    assert AD.digest(AD.LinkAdvice(_advice((3, 5), 2), (4,))) == base


def test_the_egress_counts_are_pinned_to_the_certified_histogram():
    """`reference` requires `egress == view.released`, so the egress counts are the advice's releases; one frame off
    in any bucket is refused. Which output each release is, within the histogram, is `release`, which `advice.bits`
    charges."""
    ingress, _, received = CAUSAL
    v = _view(_advice((3, 5), 2), received)
    assert AD.reference(T, RIN, OWN, ingress, list(v.released), v.released, v.floor, v.received)
    for t in range(T):
        for d in (-1, 1):
            bad = list(v.released)
            if bad[t] + d >= 0:
                bad[t] += d
                assert not AD.reference(T, RIN, OWN, ingress, bad, v.released, v.floor, v.received), (t, d)
