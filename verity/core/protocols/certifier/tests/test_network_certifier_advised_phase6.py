"""REDTEAM phase 6, target 1, against N2's advised counts (`advised`), held as regressions of the tie
(`advised.PromptTie`, PROTOCOL.md's named host-side premise): on one certified ingress and egress record, a prover who
splits the request frames among the requests otherwise (a short prompt and a long one swapped), or who numbers the
requests in other than arrival order, lowers an early request's end and so the floor, and the tie refuses both. Without
the tie the link is refused, never accepted.
"""

from fractions import Fraction

import pytest

from verity.core.protocols.certifier import advice as I1, advised as AD
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet

T, R, RIN = 4, 32, 9
#: CERT-VARLEN's id-request framing at B = 4096 (`active.id_request_frames`): a prompt of up to 337 tokens is one
#: frame, of 338 to 399 two
RF = (1,) * 337 + (2,) * 62
SHORT, LONG = 16, 338
SYNC = ClockSync(0, Fraction(1))
SS = SyncSet((0,), (Fraction(1),))


def _one_prefill(prompt_lens, release):
    """Requests of `prompt_lens` tokens in first-row order, each prefilled whole in its own step, one kept output
    each, released in buckets `release`."""
    reqs = tuple(I1.RequestAdvice(f"r{i}", p, 1, "length") for i, p in enumerate(prompt_lens))
    steps = tuple(I1.StepAdvice((I1.Row(i, "prefill", 0, p),)) for i, p in enumerate(prompt_lens))
    return I1.Advice(reqs, steps, tuple(release))


def _tie(prompt_len, order=None):
    return AD.PromptTie(tuple(prompt_len), tuple(range(len(prompt_len))) if order is None else tuple(order))


def _accepts(link, ingress, egress, own=0):
    try:
        v = AD.view(link, 0, T, R, [SYNC], SS, RF)
    except ValueError:
        return False
    return AD.reference(T, RIN, own, ingress, egress, v.released, v.floor, v.received)


def test_a_split_other_than_the_certified_prompts_is_refused_by_the_tie():
    """The certified ingress has a 338-token request (2 frames) and then a 16-token one (1 frame), its frames in
    buckets 0, 1 and 2; the egress releases an output in bucket 1 and one in bucket 3. The first request's output in
    bucket 1, before its second frame was in, is acausal, and the honest split is refused. The forged split (16 then
    338) lowers the first request's end to 1 frame and would pass under a tie that stated it; under the certified tie
    it is refused before any floor is computed. Both are valid v0s of the same charge."""
    ingress, egress = [1, 1, 1, 0], [0, 1, 0, 1]
    certified = _tie((LONG, SHORT))
    honest = AD.LinkAdvice(_one_prefill((LONG, SHORT), (1, 3)), (3,), certified)
    forged = AD.LinkAdvice(_one_prefill((SHORT, LONG), (1, 3)), (3,), certified)
    assert AD.ends(honest, RF) == (2, 3)
    assert not _accepts(honest, ingress, egress)
    with pytest.raises(ValueError, match="request 0's prompt_len 16 isn't its certified prompt's 338 ids"):
        AD.ends(forged, RF)
    assert not _accepts(forged, ingress, egress)
    untied = AD.LinkAdvice(forged.advice, (3,), _tie((SHORT, LONG)))
    assert AD.ends(untied, RF) == (1, 3) and _accepts(untied, ingress, egress)
    assert I1.total_bits(honest.advice) == I1.total_bits(forged.advice)


def test_requests_numbered_in_other_than_arrival_order_are_refused():
    """v0's requests are in first-row order, which isn't arrival order when a runner schedules a later short request
    ahead of an earlier long one; ENC-DERIVED's `walk.order` maps each back to its certified request, and the tie
    refuses any order but arrival. With equal prompt lengths the lengths alone can't tell: two 16-token requests
    certified in buckets 0 and 2, the advice's first (the certified second) released in bucket 1, before it arrived,
    passes under the identity tie and is refused under the order (1, 0)."""
    long_first = _one_prefill((LONG, SHORT), (1, 3))
    short_first = _one_prefill((SHORT, LONG), (1, 3))
    with pytest.raises(ValueError, match="aren't the tie's 2 certified requests in arrival order"):
        AD.ends(AD.LinkAdvice(short_first, (3,), _tie((LONG, SHORT), (1, 0))), RF)
    assert AD.ends(AD.LinkAdvice(long_first, (3,), _tie((LONG, SHORT))), RF) == (2, 3)

    ingress, egress = [1, 0, 1, 0], [0, 1, 0, 1]
    swapped = _one_prefill((SHORT, SHORT), (1, 3))
    assert _accepts(AD.LinkAdvice(swapped, (2,), _tie((SHORT, SHORT))), ingress, egress)
    with pytest.raises(ValueError, match="arrival order"):
        AD.view(AD.LinkAdvice(swapped, (2,), _tie((SHORT, SHORT), (1, 0))), 0, T, R, [SYNC], SS, RF)
    assert not _accepts(AD.LinkAdvice(swapped, (2,), _tie((SHORT, SHORT), (1, 0))), ingress, egress)


def test_a_link_without_the_tie_is_refused():
    """The case the forged split passes with a tie stating it is refused with none: the tie is never optional."""
    link = AD.LinkAdvice(_one_prefill((SHORT, LONG), (1, 3)), (3,))
    with pytest.raises(ValueError, match="without the prompt-length tie"):
        AD.view(link, 0, T, R, [SYNC], SS, RF)
    assert not _accepts(link, [1, 1, 1, 0], [0, 1, 0, 1])


def test_a_late_salt_is_conservative_never_a_false_accept():
    """`reference` takes the own frame from bucket 0; a salt that leaves later understates the request frames before
    it, a stricter floor: the honest release is refused (liveness), never a forged one accepted."""
    link = AD.LinkAdvice(_one_prefill((SHORT,), (1,)), (1,), _tie((SHORT,)))
    assert _accepts(link, [2, 0, 0, 0], [0, 1, 0, 0], own=1)
    assert not _accepts(link, [1, 1, 0, 0], [0, 1, 0, 0], own=1)
