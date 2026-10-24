"""REDTEAM phases 6 and 7, target 1 and target 3, against N2's advised counts (`advised`), held as regressions of the
tie (IoTieRequest's verdict, `advised.IoTieVerdict`; here its stand-in, `advised_traffic.tied`): on one certified
ingress and egress record, a prover who splits the request frames among the requests otherwise (a short prompt and a
long one swapped) lowers an early request's end and so the floor, and the verdict on the certified record refuses it.
A tie the prover states (phase 7: one stating the swapped split reopened phase 6) isn't one, the honest advice's
verdict doesn't carry to another advice, and without a verdict the link is refused, never accepted.
"""

from dataclasses import fields, replace
from fractions import Fraction

import pytest

from verity.core.protocols.certifier import advice as I1, advised as AD
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet

from advised_traffic import tied

T, R, RIN = 4, 32, 9
#: CERT-VARLEN's id-request framing at B = 4096 (`active.id_request_frames`): a prompt of up to 337 tokens is one
#: frame, of 338 to 399 two
RF = (1,) * 337 + (2,) * 62
SHORT, LONG = 16, 338
SYNC = ClockSync(0, Fraction(1))
SS = SyncSet((0,), (Fraction(1),))
#: the certified record: a 338-token request (2 frames) and then a 16-token one (1 frame), its frames in buckets 0, 1
#: and 2; the egress releases an output in bucket 0 and one in bucket 3
CERTIFIED, INGRESS, EGRESS = (LONG, SHORT), [1, 1, 1, 0], [1, 0, 0, 1]


def _one_prefill(prompt_lens, release):
    """Requests of `prompt_lens` tokens in first-row order, each prefilled whole in its own step, one kept output
    each, released in buckets `release`."""
    reqs = tuple(I1.RequestAdvice(f"r{i}", p, 1, "length") for i, p in enumerate(prompt_lens))
    steps = tuple(I1.StepAdvice((I1.Row(i, "prefill", 0, p),)) for i, p in enumerate(prompt_lens))
    return I1.Advice(reqs, steps, tuple(release))


def _accepts(link, ingress, egress, own=0):
    try:
        v = AD.view(link, 0, T, R, [SYNC], SS, RF)
    except ValueError:
        return False
    return AD.reference(T, RIN, own, ingress, egress, v.released, v.floor, v.received)


def test_a_split_other_than_the_certified_prompts_is_refused_by_the_tie():
    """The first request's output in bucket 0, before its second frame was in (bucket 1), is acausal, and the honest
    split is refused. The forged split (16 then 338) lowers the first request's end to 1 frame and would pass on a
    record of that split; IoTieRequest's verdict on the certified record refuses it before any floor is computed.
    Both are valid v0s of the same charge."""
    honest = tied(AD.LinkAdvice(_one_prefill((LONG, SHORT), (0, 3)), (3,)), CERTIFIED)
    forged = tied(AD.LinkAdvice(_one_prefill((SHORT, LONG), (0, 3)), (3,)), CERTIFIED)
    assert AD.ends(honest, RF) == (2, 3)
    assert not _accepts(honest, INGRESS, EGRESS)
    with pytest.raises(ValueError, match="doesn't accept"):
        AD.ends(forged, RF)
    assert not _accepts(forged, INGRESS, EGRESS)
    elsewhere = tied(AD.LinkAdvice(forged.advice, (3,)), (SHORT, LONG))
    assert AD.ends(elsewhere, RF) == (1, 3) and _accepts(elsewhere, INGRESS, EGRESS)
    assert I1.total_bits(honest.advice) == I1.total_bits(forged.advice)


def test_a_tie_the_prover_states_is_refused():
    """REDTEAM phase 7, target 3's residual gap: a tie stating the forged split (16 then 338) accepted it, which
    reopened phase 6's acausal release. The tie is only IoTieRequest's verdict, on the advice digest the verifier
    computed its view from: prompt lengths a prover states aren't a tie, the honest advice's verdict doesn't carry to
    the forged advice, and the verdict on the certified record refuses it."""
    honest = tied(AD.LinkAdvice(_one_prefill((LONG, SHORT), (0, 3)), (3,)), CERTIFIED)
    forged = AD.LinkAdvice(_one_prefill((SHORT, LONG), (0, 3)), (3,))
    for stated in ((SHORT, LONG), {"prompt_len": (SHORT, LONG), "order": (0, 1)}):
        with pytest.raises(I1.MalformedAdvice, match="IoTieVerdict"):
            AD.LinkAdvice(forged.advice, (3,), stated)
    for link, why in ((replace(forged, tie=honest.tie), "another advice"), (tied(forged, CERTIFIED), "doesn't accept")):
        with pytest.raises(ValueError, match=why):
            AD.view(link, 0, T, R, [SYNC], SS, RF)
        assert not _accepts(link, INGRESS, EGRESS)


def test_the_order_is_arrival_order_with_no_field_to_forge():
    """v0's requests are in first-row order, and advice request j is the link's j-th: IoTieRequest's ordinals are the
    verifier's, so the verdict has no order a prover could state. A runner that scheduled a later short request ahead
    of an earlier long one numbers v0 otherwise, which the verdict on the certified record refuses; with equal prompt
    lengths there is one reading, the arrival order, and which body each output answers is the egress tie's."""
    assert [f.name for f in fields(AD.IoTieVerdict)] == ["advice", "accepted"]
    long_first = tied(AD.LinkAdvice(_one_prefill((LONG, SHORT), (0, 3)), (3,)), CERTIFIED)
    short_first = tied(AD.LinkAdvice(_one_prefill((SHORT, LONG), (0, 3)), (3,)), CERTIFIED)
    assert AD.ends(long_first, RF) == (2, 3)
    with pytest.raises(ValueError, match="doesn't accept"):
        AD.ends(short_first, RF)
    same = tied(AD.LinkAdvice(_one_prefill((SHORT, SHORT), (1, 3)), (2,)), (SHORT, SHORT))
    assert AD.ends(same, RF) == (1, 2) and _accepts(same, [1, 0, 1, 0], [0, 1, 0, 1])


def test_a_link_without_the_tie_is_refused():
    """The case the forged split passes on a record of it is refused with no verdict: the tie is never optional."""
    link = AD.LinkAdvice(_one_prefill((SHORT, LONG), (0, 3)), (3,))
    with pytest.raises(ValueError, match="without IoTieRequest's verdict"):
        AD.view(link, 0, T, R, [SYNC], SS, RF)
    assert not _accepts(link, INGRESS, EGRESS)


def test_a_late_salt_is_conservative_never_a_false_accept():
    """`reference` takes the own frame from bucket 0; a salt that leaves later understates the request frames before
    it, a stricter floor: the honest release in the request's own bucket is refused (liveness), never a forged one
    accepted."""
    link = tied(AD.LinkAdvice(_one_prefill((SHORT,), (0,)), (1,)))
    assert _accepts(link, [2, 0, 0, 0], [1, 0, 0, 0], own=1)
    assert not _accepts(link, [1, 1, 0, 0], [1, 0, 0, 0], own=1)
