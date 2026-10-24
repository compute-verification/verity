"""Adversarial cases against N2's advised counts (`advised`) from the advice red team (PROTOCOL.md, "Advice for
arbitrary serving: what it can carry"), each refused:

- Attack 11, two readings of one advice: N2's frames (`advised.frames`) are advice v0's outputs (`advice.outputs`),
  step, request and index, on every advice v0 accepts, so the counts N2 holds and the frames the tie and the compiler
  read are one map.
- A response handed before its request's anchor plus the window's δ and ⌊λ/ρ⌋ is refused (δ = 0 admits the anchor's
  own bucket).
- Egress counts one frame off in any bucket, a queue released early, ingress over RIN, short of OWN or not summing to
  the window's `received`, each refused.
"""

from fractions import Fraction
from types import SimpleNamespace as NS

import pytest

from verity.core.protocols.certifier import advice as A, advised as N
from verity.core.protocols.certifier.schedule import ClockSync

from test_network_certifier_advice_adversarial import schedule

T, R, RIN, OWN = 8, 2, 4, 0


def _trailing(a) -> bool:
    """A row past a stop: it reaches the request's last known token after its last output (async scheduling's)."""
    return any(r.start + r.n == a.requests[r.request].prompt_len + a.requests[r.request].output_len
               for st in a.steps for r in st.rows)


def _same_frames(a) -> bool:
    return [(f.step, f.request, f.index) for f in N.frames(a)] == [(o.step, o.request, o.index) for o in A.outputs(a)]


def test_n2s_frames_are_advice_v0s_outputs():
    plain = [a for a in (schedule(seed) for seed in range(400)) if not _trailing(a)]
    assert plain
    assert all(_same_frames(a) for a in plain)


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="advised.frames refuses the one row async scheduling runs past a stop, which advice v0 "
                          "accepts and discards: N2 would refuse honest sessions and read the advice otherwise")
def test_n2_discards_the_row_past_a_stop_as_advice_v0_does():
    a = A.Advice((A.RequestAdvice("a", 4, 2, "eos"),),
                 (A.StepAdvice((A.Row(0, "prefill", 0, 4),)), A.StepAdvice((A.Row(0, "decode", 4, 1),)),
                  A.StepAdvice((A.Row(0, "decode", 5, 1),))), (0, 1))
    assert len(A.outputs(a)) == 2
    assert _same_frames(a)


def _advice(hands, prompt=4, out=2, ingress_frames=2, received=(2,)):
    """One request: a whole prefill then decodes, step s handed at `hands[s]`, its frames received in window 0."""
    steps = [NS(rows=[NS(request=0, start=0, n=prompt)])] + [NS(rows=[NS(request=0, start=prompt + i, n=1)])
                                                             for i in range(out - 1)]
    return NS(requests=[NS(prompt_len=prompt, output_len=out, ingress_frames=ingress_frames)], steps=steps,
              release=list(hands), received=list(received))


def _accepts(adv, ingress, sync, egress=None):
    v = N.view(adv, 0, T, R, [sync])
    egress = N.derive(v.handed, v.queued, R) if egress is None else egress
    return N.reference(T, R, RIN, OWN, ingress, egress, v.handed, v.floor, v.queued, v.received)


INGRESS = [0, 0, 2, 0, 0, 0, 0, 0]  # the request's two frames released in bucket 2, its anchor


@pytest.mark.parametrize("delta, rho, first, accepted", [
    (1, 1, 3, True),     # anchor 2 + δ 1
    (1, 1, 2, False),    # one bucket early
    (0, 1, 2, True),     # δ = 0: the anchor's own bucket
    (0, 1, 1, False),    # before the anchor
    (2, Fraction(1, 2), 4, True),  # frame 0 at 2 + 2; frame 1 at least 2 + 2 + ⌊1/ρ⌋ = 6
    (2, Fraction(1, 2), 3, False),
])
def test_a_response_before_its_requests_anchor_plus_delta_and_rate_is_refused(delta, rho, first, accepted):
    second = first + 1 + (rho < 1)
    assert _accepts(_advice([first, second]), INGRESS, ClockSync(delta, rho)) is accepted


def test_a_frame_handed_before_any_request_frame_could_be_released_has_no_view():
    with pytest.raises(ValueError, match="before its request"):
        N.view(_advice([0, 1]), 0, T, R, [ClockSync(1, 1)])


def test_egress_one_frame_off_in_any_bucket_is_refused():
    adv = _advice([3, 3, 3], out=3)
    sync = ClockSync(0, 4)
    v = N.view(adv, 0, T, R, [sync])
    honest = N.derive(v.handed, v.queued, R)
    assert honest[3] == R and honest[4] == 1  # three frames handed in bucket 3 at R = 2: one waits
    assert _accepts(adv, INGRESS, sync, honest)
    for t in range(T):
        for d in (-1, 1):
            if honest[t] + d >= 0:
                bad = list(honest)
                bad[t] += d
                assert not _accepts(adv, INGRESS, sync, bad), (t, d)
    early = list(honest)
    early[3], early[4] = 3, 0  # the queued frame released with the others, past R
    assert not _accepts(adv, INGRESS, sync, early)


@pytest.mark.parametrize("ingress", [
    [0, 0, 5, 0, 0, 0, 0, 0],   # a bucket over RIN
    [0, 0, 1, 0, 0, 0, 0, 0],   # short of the window's received
    [0, 0, 2, 1, 0, 0, 0, 0],   # past it
    [0, 0, 0, 0, 2, 0, 0, 0],   # released after the response was handed
])
def test_ingress_that_doesnt_carry_the_advices_requests_is_refused(ingress):
    assert _accepts(_advice([3, 4]), INGRESS, ClockSync(1, 1))
    assert not _accepts(_advice([3, 4]), ingress, ClockSync(1, 1))
