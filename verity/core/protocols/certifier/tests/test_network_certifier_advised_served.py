"""N2's advised counts (`advised`) on real served advice: window 5 of a run, LLAMA32_1B served by
vLLM, its advice written by `verity-vllm advice` (`fixtures/served_window.json`). The advice's clock is the engine's:
each kept output's release is the bucket of τ = 0.1 s from the window's first step, with no certifier on the path, so
every request's first output is released in the bucket it arrived in (bucket 0: the four requests arrived 16 to 41 ms
before the first step). The floor is the Program's eligibility under the declared sync, so that advice is accepted
under δ = 0 as served, and no release passes before the bucket that releases its request's last ingress frame.

The link's framing here is an id request of 4 bytes a prompt token in 256-byte frames (`active.request_frame_count`),
which puts the four prompts (47, 148, 75 and 135 tokens) in 1, 3, 2 and 3 frames; the ingress counts are the ingress
certifier's (`grid.Certifier`), its own frame first, each request's frames queued in the bucket it arrived in.
"""

import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from verity.core.protocols.certifier import advice as A, advised as AD
from verity.core.protocols.certifier.active import request_frame_count
from verity.core.protocols.certifier.grid import Certifier, GridParams, occupancy
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet

from advised_traffic import held

DOC = json.loads((Path(__file__).parent / "fixtures" / "served_window.json").read_text())
ADVICE = A.from_json(DOC["advice"])
ARRIVE = [max(0, math.floor(s / DOC["tau_s"])) for s in DOC["arrive_s"]]
FRAMING = tuple(request_frame_count(4 * n, 256) for n in range(1, 2049))
FRAMES = [FRAMING[q.prompt_len - 1] for q in ADVICE.requests]
T, R = 8, 32
#: δ of 0 or 1 bucket, and a stream of up to 8 frames a bucket, 80 tokens a second at τ = 0.1 s
SERVED = SyncSet((0, 1), (8,))
INSTANT, ONE = ClockSync(0, 8), ClockSync(1, 8)


def _ingress(rin, own=0, late=0):
    """The ingress certifier's counts: its own frame first if `own`, each request's frames queued `late` buckets after
    the bucket it arrived in, and min(queue, `rin`) released a bucket."""
    cert = Certifier("requests", GridParams(buckets=T, allocation=rin, frame_bytes=1))
    if own:
        cert.enqueue(b"s")
    for t in range(T):
        for j, a in enumerate(ARRIVE):
            if a + late == t:
                for _ in range(FRAMES[j]):
                    cert.enqueue(b"q")
        cert.tick()
    return list(occupancy(cert.close_window().grid))


def _view(adv, sync):
    link = AD.LinkAdvice(adv, (sum(FRAMES),))
    return AD.view(link, 0, T, R, [sync], SERVED, FRAMING, clocks=[sync], **held(link))


def _egress(adv):
    per = Counter(adv.release)
    return [per[t] for t in range(T)]


def _accepts(adv, ingress, sync, rin, own=0):
    try:
        v = _view(adv, sync)
    except ValueError:
        return False
    return AD.reference(T, rin, own, ingress, _egress(adv), v.released, v.floor, v.received)


def _shifted(adv, by):
    """`adv` with request j's releases `by[j]` buckets later."""
    return replace(adv, release=tuple(b + by[o.request] for o, b in zip(A.outputs(adv), adv.release)))


def _first(adv):
    first = {}
    for o, b in zip(A.outputs(adv), adv.release):
        first.setdefault(o.request, b)
    return [first[j] for j in range(len(adv.requests))]


def test_the_fixture_is_the_served_window():
    """Four requests in bucket 0, each first output released there too, 64 outputs over buckets 0 to 3."""
    assert A.to_json(ADVICE) == DOC["advice"]
    assert [q.prompt_len for q in ADVICE.requests] == [47, 148, 75, 135] and FRAMES == [1, 3, 2, 3]
    assert ARRIVE == [0, 0, 0, 0] and _first(ADVICE) == [0, 0, 0, 0]
    assert _egress(ADVICE) == [26, 18, 11, 9, 0, 0, 0, 0]


@pytest.mark.parametrize("own", [0, 1])
def test_served_first_outputs_in_their_requests_bucket_are_accepted_under_an_instant_sync(own):
    """With the four requests' 9 frames certified in bucket 0 (an allocation of 16), the served advice is accepted
    under δ = 0, with the salt off and on. Under δ = 1, the floor the certifier's one-bucket readiness had put under
    every declaration, it is refused before any count is read, and the same advice one bucket later is accepted
    there: a certified path's honest advice, whose proxy hands an output to the egress no earlier than the bucket
    after its request's last frame."""
    ingress = _ingress(16, own)
    assert ingress == [9 + own, 0, 0, 0, 0, 0, 0, 0]
    assert _view(ADVICE, INSTANT).floor[0] == 9
    assert _accepts(ADVICE, ingress, INSTANT, 16, own)
    with pytest.raises(ValueError, match="before its request"):
        _view(ADVICE, ONE)
    assert _accepts(_shifted(ADVICE, [1] * 4), ingress, ONE, 16, own)


def test_no_served_release_passes_before_its_requests_last_ingress_frame():
    """The served advice on ingress certified one bucket late is refused under every sync Σ_sync holds. Under an
    allocation of 4 the 9 frames take buckets 0 to 2, so requests 2 and 3 end in buckets 1 and 2, and the served
    releases of their outputs in bucket 0 are refused; the same advice with those two requests' releases moved to
    their ends' buckets is accepted. The circuit agrees on each case."""
    for sync in SERVED:
        assert not _accepts(ADVICE, _ingress(16, late=1), sync, 16)
    spread = _ingress(4)
    assert spread == [4, 4, 1, 0, 0, 0, 0, 0]
    link = AD.LinkAdvice(ADVICE, (9,))
    assert AD.ends(link, FRAMING, **held(link)) == (1, 4, 6, 9)
    assert not _accepts(ADVICE, spread, INSTANT, 4)
    moved = _shifted(ADVICE, [0, 0, 1, 2])
    assert _accepts(moved, spread, INSTANT, 4)
    assert not _accepts(_shifted(ADVICE, [0, 0, 1, 1]), spread, INSTANT, 4)
    cases = [(spread, _egress(a), _view(a, INSTANT)) for a in (ADVICE, moved)]
    assert AD.evaluate((T, 4, 0), cases) == [0, 1]
    late = [(_ingress(16, late=1), _egress(ADVICE), _view(ADVICE, INSTANT)),
            (_ingress(16), _egress(ADVICE), _view(ADVICE, INSTANT))]
    assert AD.evaluate((T, 16, 0), late) == [0, 1]
