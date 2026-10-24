"""Phase-3 adversarial cases against N2's advised counts (`advised`, #3502 @032c2729), from the network-accounting red
team. Each states the bits it bounds or the soundness gap it shows. They hold the verifier's reference (`advised.view`
and `advised.reference`, which `circuit-check` holds to the circuit) and the advice's validity (`advised.LinkAdvice`,
`advised.digest`), the CPU-only surfaces a malicious prover reaches.

What the verifier pins and what it leaves to the prover:
- the egress counts are pinned to the certified egress histogram (`reference` requires `egress == view.released`), and
  the per-output release assignment within that histogram is charged by `advice.bits` (A_c): CLOSED up to A_c;
- the window's `received` is pinned to the certified ingress sum, and the advice `digest` distinguishes every input, so
  the one-advice AND of `service.LinkWindows.network_verdict` refuses a cross-window swap: CLOSED;
- `ingress_end` (N2's ingress part) is the prover's: nothing ties it to the record until IO-TIE lands, and
  `advice.bits` does not charge it. It drives the causality floor (`view.floor`), so the prover forges it to accept an
  acausal run (an output released before its request's frames arrived). OPEN: up to bitlen(Σ received) bits a request,
  and the floor's causality guarantee holds only against the declared `ingress_end`, not reality;
- the declared clock sync is not validated against the clock: a looser `delta` relaxes the floor. The sync charge
  counts log2 #Σ_sync a window, but the floor guarantee again holds only against the declared sync. OPEN (compounds);
- bucket 0's ingress count, which the certifier's own frame inflates, is capped at RIN not RIN + OWN: an honest window
  that fills bucket 0 refuses. Candidate liveness bug.
"""

from fractions import Fraction
from itertools import permutations

from verity.core.protocols.certifier import advice as A, advised as AD
from verity.core.protocols.certifier.schedule import ClockSync

T, R, RIN, OWN = 8, 2, 4, 0
ONE = ClockSync(1, Fraction(1, 1))


def _advice(release, n=1):
    """`n` requests, one kept output each, each a whole prefill in its own step; output k is request k's, released in
    bucket `release[k]`. `outputs` order is step then row, so output k is request k's."""
    reqs = tuple(A.RequestAdvice(chr(ord("a") + i), 4, 1, "eos") for i in range(n))
    steps = tuple(A.StepAdvice((A.Row(i, "prefill", 0, 4),)) for i in range(n))
    return A.Advice(reqs, steps, tuple(release))


def _accepts(adv, ends, received, ingress, egress, sync=ONE, own=OWN):
    v = AD.view(AD.LinkAdvice(adv, ends, received), 0, T, R, [sync])
    return AD.reference(T, RIN, own, ingress, egress, v.released, v.floor, v.received)


def test_the_honest_advised_window_is_accepted():
    """Sanity: one request, output released in its anchor's window, the certified ingress carrying it in time."""
    adv = _advice((3,))
    assert _accepts(adv, (2,), (2,), [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0])


def test_the_prover_forges_ingress_end_to_pass_an_acausal_run():
    """SOUNDNESS GAP. Two requests, each one output: request 0's out in bucket 3, request 1's in bucket 5. The certified
    ingress carries 2 frames in bucket 1 and 2 in bucket 3 (prefix 2 by bucket 2, 4 by bucket 3); the certified egress
    is one frame in bucket 3 and one in bucket 5.

    Request 0 arrived LAST on the ingress link, so IO-TIE would pin `ingress_end = (4, 2)`: request 0's output, released
    at bucket 3, is eligible at bucket 2, by which only 2 of its 4 frames were in — acausal — and the honest run is
    refused. But `ingress_end` is the prover's and `advice.bits` does not charge it: the prover claims request 0 ended
    early (`(1, 4)` or `(2, 4)`), the floor drops, and the same certified records are accepted. The causality floor is
    only as strong as the unbound `ingress_end`."""
    adv = _advice((3, 5), n=2)
    ingress, egress, received = [0, 2, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0], (4,)
    assert not _accepts(adv, (4, 2), received, ingress, egress)  # honest: request 0 arrived last -> acausal -> refused
    assert _accepts(adv, (1, 4), received, ingress, egress)      # forged: request 0 "ends early" -> accepted
    assert _accepts(adv, (2, 4), received, ingress, egress)
    # advice.bits charges `release` but never `ingress_end`: the three runs cost the prover the same.
    assert A.total_bits(adv) == 118


def test_ingress_end_is_an_uncharged_channel_many_assignments_one_record():
    """The channel's bits. On ONE fixed certified (ingress, egress) record more than one `ingress_end` is accepted, each
    a distinct advice (distinct `digest`) at the same `advice.bits`. The prover picks freely among them; the choice is
    the ingress-timing channel, at most bitlen(Σ received) bits a request, which `advice.bits` doesn't charge."""
    adv = _advice((3, 5), n=2)
    ingress, egress, received = [0, 2, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0], (4,)
    accepted = {ends for ends in permutations(range(1, 5), 2)
                if max(ends) == received[0] and _accepts(adv, ends, received, ingress, egress)}
    assert len(accepted) >= 2
    digests = {AD.digest(AD.LinkAdvice(adv, ends, received)) for ends in accepted}
    assert len(digests) == len(accepted)                 # each accepted choice is a distinct advice
    assert len({A.total_bits(adv) for _ in accepted}) == 1  # all at the same charge


def test_a_looser_declared_sync_relaxes_the_causality_floor():
    """OPEN (compounds). The verifier validates the declared clock sync only by `a ≥ 0`, never against the clock. One
    request, output in bucket 3, its 2 frames certified in bucket 3. Under the real latency (δ = 2) the output at bucket
    3 is eligible at bucket 1, by which no frame was in — refused. The prover declares δ = 0 (instant), the eligibility
    moves to bucket 3, and the same records are accepted. The sync charge counts log2 #Σ_sync a window, but the floor's
    guarantee holds only against the declared sync."""
    adv = _advice((3,))
    ingress, egress, received = [0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], (2,)
    assert not _accepts(adv, (2,), received, ingress, egress, sync=ClockSync(2, Fraction(1, 1)))  # honest latency
    assert _accepts(adv, (2,), received, ingress, egress, sync=ClockSync(0, Fraction(1, 1)))      # forged instant


def test_bucket_zero_saturated_with_the_own_frame_refuses_an_honest_window():
    """Candidate LIVENESS bug. `reference` caps every ingress bucket at RIN, but the certifier's own frame inflates
    bucket 0's certified count. A window whose ingress link carries RIN request frames in bucket 0 then certifies
    RIN + 1 there, which the cap refuses, though the window is honest. The cap on bucket 0 should be RIN + OWN."""
    assert not AD.reference(4, 4, 1, [5, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4, 4)  # 4 request frames + 1 own
    assert AD.reference(4, 4, 1, [4, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4, 3)      # 3 request frames + 1 own: fits


def test_the_advice_digest_distinguishes_every_input_so_cross_window_swaps_are_refused():
    """CLOSED. Every window reports `advised.digest` of the whole advice, and `network_verdict` ANDs that all windows
    name the first's. The digest moves with `release`, `ingress_end` and `received`, so a window can't carry a different
    advice under the first window's digest."""
    base = AD.digest(AD.LinkAdvice(_advice((3, 5), n=2), (1, 4), (4,)))
    assert AD.digest(AD.LinkAdvice(_advice((4, 5), n=2), (1, 4), (4,))) != base  # a different release
    assert AD.digest(AD.LinkAdvice(_advice((3, 5), n=2), (2, 4), (4,))) != base  # a different ingress_end
    assert AD.digest(AD.LinkAdvice(_advice((3, 5), n=2), (1, 4), (4,))) == base  # same advice, same digest


def test_the_egress_counts_are_pinned_to_the_certified_histogram():
    """CLOSED up to A_c. `reference` requires `egress == view.released`, so the egress counts are the advice's releases;
    one frame off in any bucket is refused. The per-output release assignment within the histogram is `release`, which
    `advice.bits` charges, so the egress channel of accepted runs is at most A_c."""
    adv = _advice((3, 5), n=2)
    v = AD.view(AD.LinkAdvice(adv, (1, 4), (4,)), 0, T, R, [ONE])
    ingress, received = [0, 2, 0, 2, 0, 0, 0, 0], 4
    assert AD.reference(T, RIN, OWN, ingress, list(v.released), v.released, v.floor, received)
    for t in range(T):
        for d in (-1, 1):
            bad = list(v.released)
            if bad[t] + d >= 0:
                bad[t] += d
                assert not AD.reference(T, RIN, OWN, ingress, bad, v.released, v.floor, received), (t, d)
