"""N2's advised counts (`advised`), held as regressions of the tie (IoTieRequest's verdict, `advised.IoTieVerdict`;
here its stand-in, `advised_traffic.StandInTie`): on one certified ingress and egress record, a prover who splits the
request frames among the requests otherwise (a short prompt and a long one swapped) lowers an early request's end and
so the floor, and the verdict on the certified record refuses it.
A tie the prover states (one stating the swapped split reopens its acausal release) isn't one, nor is a verdict it
builds and delivers (one for the forged advice's digest, every window accepted, would get the swapped split's ends
(1, 3) accepted through the verifier of record), the honest advice's verdict doesn't carry to another advice, and
without a verdict from the verifier's own source the link is refused, never accepted.
"""

import inspect
from dataclasses import fields
from fractions import Fraction

import pytest

from verity.core.protocols.certifier import advice as I1, advised as AD
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.certifier.service import LinkWindows

import signers
from advised_traffic import LINK, StandInClocks, StandInTie, held

T, R, RIN = 4, 32, 9
#: The id-request framing at B = 4096 (`active.id_request_frames`): a prompt of up to 337 tokens is one
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


def _view(link, tie):
    return AD.view(link, 0, T, R, [SYNC], SS, RF, clocks=[SYNC], **tie)


def _accepts(link, ingress, egress, own=0, certified=None, tie=None):
    """`advised.reference` on window 0 of `link`, under the stand-in verdict on a record of the prompts `certified`
    (by default `link`'s own), or under `tie`, `advised.view`'s tie arguments."""
    try:
        v = _view(link, tie or held(link, certified))
    except ValueError:
        return False
    return AD.reference(T, RIN, own, ingress, egress, v.released, v.floor, v.received)


HONEST = AD.LinkAdvice(_one_prefill((LONG, SHORT), (0, 3)), (3,))
FORGED = AD.LinkAdvice(_one_prefill((SHORT, LONG), (0, 3)), (3,))


def _self_built(link):
    """The verdict a prover builds from its own advice alone, on the link and the registrations the verifier holds,
    which are public: the advice's digest and every ingress window accepted. No proof goes in."""
    tie = held(link)
    return tie | {"tie": AD.IoTieVerdict(LINK, tie["registrations"], AD.digest(link), (True,))}


def test_a_split_other_than_the_certified_prompts_is_refused_by_the_tie():
    """The first request's output in bucket 0, before its second frame was in (bucket 1), is acausal, and the honest
    split is refused. The forged split (16 then 338) lowers the first request's end to 1 frame and would pass on a
    record of that split; IoTieRequest's verdict on the certified record refuses it before any floor is computed.
    Both are valid v0s of the same charge."""
    assert AD.ends(HONEST, RF, **held(HONEST, CERTIFIED)) == (2, 3)
    assert not _accepts(HONEST, INGRESS, EGRESS, certified=CERTIFIED)
    with pytest.raises(ValueError, match="doesn't accept"):
        AD.ends(FORGED, RF, **held(FORGED, CERTIFIED))
    assert not _accepts(FORGED, INGRESS, EGRESS, certified=CERTIFIED)
    assert AD.ends(FORGED, RF, **held(FORGED, (SHORT, LONG))) == (1, 3)
    assert _accepts(FORGED, INGRESS, EGRESS, certified=(SHORT, LONG))
    assert I1.total_bits(HONEST.advice) == I1.total_bits(FORGED.advice)


def test_a_tie_the_prover_states_is_refused():
    """A tie stating the forged split (16 then 338) would accept it, reopening the swapped split's acausal release.
    The tie is only IoTieRequest's verdict, on the advice digest the verifier
    computed its view from: the advice carries no tie, so prompt lengths a prover states have nowhere to go, the
    honest advice's verdict doesn't carry to the forged advice, and the verdict on the certified record refuses it."""
    for stated in ((SHORT, LONG), {"prompt_len": (SHORT, LONG), "order": (0, 1)}):
        with pytest.raises(TypeError):
            AD.LinkAdvice(FORGED.advice, (3,), stated)
    honest = held(HONEST, CERTIFIED)
    for tie, why in ((honest, "another advice"), (held(FORGED, CERTIFIED), "doesn't accept")):
        with pytest.raises(ValueError, match=why):
            _view(FORGED, tie)
        assert not _accepts(FORGED, INGRESS, EGRESS, tie=tie)


def _phase6_link(key, ties):
    """A link of this case's shape (T = 4, R = 32, R_in = 9, one window, the B = 4096 framing) whose verifier's tie
    source is `ties`."""
    return LinkWindows(LINK, GridParams(T, R, 1), SS, key, "r-phase6", signers.SESSION, ingress=GridParams(T, RIN, 1),
                       proof_deadline=60.0, advised=AD.AdvisedProfile(1, RF, 0), ties=ties,
                       clocks=StandInClocks((SYNC,)))


def test_a_verdict_the_prover_builds_never_reaches_the_verifier_of_record(tmp_path):
    """A verdict the prover built for the forged split's digest, every window accepted, would get its ends (1, 3)
    accepted through a `LinkWindows.verify` that read it off the delivered advice. The
    reference functions trust the verdict they're given (under the self-built one the split's ends are (1, 3)), so
    the verifier gives them only its own source's: the advice carries no tie, no entry point of the verifier of record
    takes one, and its view (`_view`, which `public_arguments`, `arguments` and `advised_check` read) of the forged
    split is refused under the source on the certified record and without a source, as every advised link is in the
    package."""
    assert AD.ends(FORGED, RF, **_self_built(FORGED)) == (1, 3) and _accepts(FORGED, INGRESS, EGRESS,
                                                                            tie=_self_built(FORGED))
    assert [f.name for f in fields(AD.LinkAdvice)] == ["advice", "received"]
    for entry in (LinkWindows.verify, LinkWindows.clear, LinkWindows.advised_check, LinkWindows.outcome,
                  LinkWindows.arguments, LinkWindows.public_arguments, LinkWindows.unproved_outcome):
        assert not {"tie", "ties", "verdict"} & set(inspect.signature(entry).parameters), entry.__name__
    key = signers.signer(tmp_path).public_key()
    certified = StandInTie(CERTIFIED, windows=1)
    assert _phase6_link(key, None)._view(0, [SYNC], FORGED) is None
    assert _phase6_link(key, None)._view(0, [SYNC], HONEST) is None
    assert _phase6_link(key, certified)._view(0, [SYNC], FORGED) is None
    assert _phase6_link(key, certified)._view(0, [SYNC], HONEST) == _view(HONEST, held(HONEST, CERTIFIED))
    assert _phase6_link(key, StandInTie((SHORT, LONG), windows=1))._view(0, [SYNC], FORGED) is not None


def test_the_order_is_arrival_order_with_no_field_to_forge():
    """v0's requests are in first-row order, and advice request j is the link's j-th: IoTieRequest's ordinals are the
    verifier's, so the verdict has no order a prover could state. A runner that scheduled a later short request ahead
    of an earlier long one numbers v0 otherwise, which the verdict on the certified record refuses; with equal prompt
    lengths there is one reading, the arrival order, and which body each output answers is the egress tie's."""
    assert [f.name for f in fields(AD.IoTieVerdict)] == ["link", "registrations", "advice", "accepted"]
    assert AD.ends(HONEST, RF, **held(HONEST, CERTIFIED)) == (2, 3)
    with pytest.raises(ValueError, match="doesn't accept"):
        AD.ends(FORGED, RF, **held(FORGED, CERTIFIED))
    same = AD.LinkAdvice(_one_prefill((SHORT, SHORT), (1, 3)), (2,))
    assert AD.ends(same, RF, **held(same, (SHORT, SHORT))) == (1, 2)
    assert _accepts(same, [1, 0, 1, 0], [0, 1, 0, 1], certified=(SHORT, SHORT))


def test_a_link_without_the_tie_is_refused():
    """The case the forged split passes on a record of it is refused with no verdict: the tie is never optional."""
    with pytest.raises(ValueError, match="without IoTieRequest's verdict"):
        _view(FORGED, held(FORGED) | {"tie": None})
    assert not _accepts(FORGED, INGRESS, EGRESS, tie=held(FORGED) | {"tie": None})


def test_a_late_salt_is_conservative_never_a_false_accept():
    """`reference` takes the own frame from bucket 0; a salt that leaves later understates the request frames before
    it, a stricter floor: the honest release in the request's own bucket is refused (liveness), never a forged one
    accepted."""
    link = AD.LinkAdvice(_one_prefill((SHORT,), (0,)), (1,))
    assert _accepts(link, [2, 0, 0, 0], [1, 0, 0, 0], own=1)
    assert not _accepts(link, [1, 1, 0, 0], [1, 0, 0, 0], own=1)
