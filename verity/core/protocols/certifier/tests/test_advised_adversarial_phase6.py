"""REDTEAM phase 6, target 1: the ingress-end rule of N2's advised unit, as landed (N2-ADVICE #3502 at
d74de53fd9, with CERT-VARLEN's framing fd7a9ca919).

`advised.ends` derives each request's end on the ingress link from v0's prompt lengths alone
(request j ends at Σ_{i≤j} request_frames[prompt_len_i − 1]); the only binding to the certified record is that the
last end equals Σ received. `advised.view` turns those ends into the per-bucket `floor` the ingress prefix sums must
clear, and `advised.reference` enforces it. Nothing in the unit — and nothing in v0's own validity — holds the
per-request prompt lengths, or the request order, to the certified ingress head. The unit's docstring is explicit that
this is deferred to IO-TIE's map; the cases here show, executably, what that deferral costs if the map doesn't bind
the lengths and the order under VARLEN traffic.

Run against a worktree that carries both advised.py and active.id_request_frames (phase6 branch has both merged).
This is red-team evidence, not a suite in the merge gate; it remaps `verity` to the worktree so it runs regardless of
the editable install pointing at main.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

# <root>/verity/core/protocols/certifier/tests/<this file>  ->  parents[5] is the repo root.
_ROOT = Path(__file__).resolve().parents[5]
try:  # make `verity` resolve to this worktree, not the editable install's main checkout
    from editables.redirector import RedirectingFinder as _F

    _F.map_module("verity", str(_ROOT / "verity" / "__init__.py"))
except Exception:  # pragma: no cover - plain checkout, nothing to remap
    pass

from verity.core.protocols.certifier import active as A  # noqa: E402
from verity.core.protocols.certifier import advice as I1  # noqa: E402
from verity.core.protocols.certifier import advised as AD  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet  # noqa: E402

B, T, R, RIN = 4096, 4, 32, 9
# the public framing table: entry n − 1 is the ingress frames an id request of an n-token prompt takes.
RF = tuple(A.id_request_frames(n, B) for n in range(1, 400))
# at B = 4096 a prompt of ≤ 337 tokens is 1 frame, 338 is 2 (see CERT-VARLEN's note); SHORT/LONG straddle that step.
SHORT, LONG = 16, 338
SYNC = ClockSync(0, Fraction(1))
SS = SyncSet((0,), (Fraction(1),))


def _one_prefill(prompt_lens: tuple[int, ...], release: tuple[int, ...]) -> I1.Advice:
    """An advice of len(prompt_lens) requests, each prefilled whole in its own step (one kept output each), in the
    given first-row order and with the given egress release buckets."""
    reqs = tuple(I1.RequestAdvice(id=f"r{i}", prompt_len=p, output_len=1, finish="length")
                 for i, p in enumerate(prompt_lens))
    steps = tuple(I1.StepAdvice((I1.Row(request=i, kind="prefill", start=0, n=p),))
                  for i, p in enumerate(prompt_lens))
    return I1.Advice(requests=reqs, steps=steps, release=release)


def _accepts(link: AD.LinkAdvice, ingress: list[int], egress: list[int], own: int = 0) -> bool:
    v = AD.view(link, 0, T, R, [SYNC], SS, RF)
    return AD.reference(T, RIN, own, ingress, egress, v.released, v.floor, v.received)


def test_split_prompt_lengths_defeat_the_floor():
    """CLAIM (advised.py docstring): "no frame is released before the ingress certifier had released its request's
    ends[j] request frames" — the floor is the causality guard on the egress release schedule.
    REALITY: `ends` reads the split of the total into per-request prompt lengths straight from v0, and holds only the
    *total* to the record (Σ ends == Σ received). A prover who swaps a short and a long prompt between an early and a
    later request keeps the total fixed but lowers the early request's end, lowering the floor, so an egress record in
    which request 0's output left before request 0's prompt had finished arriving is accepted. The honest split of the
    SAME certified ingress and egress is refused. v0 validates both advices; `advice.bits` charges both the same.
    ONE-LINE FIX: bind each request's prompt_len (and so its end) to the ingress record's content — a VARLEN IO-TIE
    that reads the id_request fields, or certifier-signed per-request frame counts in the head — not to v0 alone.
    """
    # req0's second frame only lands in bucket 1, yet its output is released in bucket 1 (anchor bucket 0): acausal.
    ingress, egress = [1, 1, 1, 0], [0, 1, 0, 1]
    honest = AD.LinkAdvice(advice=_one_prefill((LONG, SHORT), (1, 3)), received=(3,))
    forged = AD.LinkAdvice(advice=_one_prefill((SHORT, LONG), (1, 3)), received=(3,))

    assert AD.ends(honest, RF) == (2, 3)
    assert AD.ends(forged, RF) == (1, 3)
    assert _accepts(honest, ingress, egress) is False, "the true split must be refused: the release is acausal"
    assert _accepts(forged, ingress, egress) is True, "the forged split is accepted on the same certified record"
    # two distinct charged v0s, same cost: the mis-split is not caught, only charged like any other advice.
    assert AD.digest(honest) != AD.digest(forged)
    assert I1.total_bits(honest.advice) == I1.total_bits(forged.advice)


def test_request_order_is_not_tied_to_arrival():
    """CLAIM: advice request j is the link's j-th request (FCFS admission: v0's first-row order is arrival order), so
    `ends` may map request j to ingress slot j.
    REALITY: v0 only enforces that `requests` is in *first-row* order (the order requests first appear in the steps);
    it has no arrival timestamps and cannot check FCFS. An engine that schedules a later-arriving short request before
    an earlier long one produces an advice whose first-row order is not arrival order, yet validates; `ends` then uses
    the wrong prompt length for each ingress slot — the same floor-lowering as a mis-split. N2 catches neither; the
    whole binding rests on IO-TIE's map plus the unproven FCFS premise.
    ONE-LINE FIX: the floor's request-to-slot map must come from the ingress record (IO-TIE), and FCFS must be a named,
    checked assumption of the advised unit, not folded silently into v0's first-row order.
    """
    short_first = _one_prefill((SHORT, LONG), (1, 3))  # engine ran the short request first
    long_first = _one_prefill((LONG, SHORT), (1, 3))
    assert isinstance(short_first, I1.Advice) and isinstance(long_first, I1.Advice)  # both valid v0
    # the ends N2 derives depend only on the advice's order; nothing ties that order to arrival.
    assert AD.ends(AD.LinkAdvice(advice=short_first, received=(3,)), RF) == (1, 3)
    assert AD.ends(AD.LinkAdvice(advice=long_first, received=(3,)), RF) == (2, 3)


def test_late_salt_is_conservative_never_a_false_accept():
    """CLAIM (CERT-VARLEN): the ends are exact only while the certifier's own (salt) frame leaves in the window's first
    bucket, and CERT-VARLEN is moving the salt to the queue head so that always holds.
    REALITY: `reference` subtracts OWN at bucket 0. If the salt actually leaves later, cum is understated before the
    salt's real bucket, so the floor is *stricter* and an otherwise-honest run is refused — a liveness hit, never a
    false accept. This confirms the salt-at-head change is a liveness fix, not a soundness patch (ranked low).
    ONE-LINE FIX: none needed for soundness; CERT-VARLEN's salt-at-head keeps honest runs from being refused.
    """
    adv = _one_prefill((SHORT,), (1,))  # one request, 1 frame, output released at bucket 1 (floor[0] = 1)
    link = AD.LinkAdvice(advice=adv, received=(1,))
    # salt at the head of bucket 0 (honest): request frame + own frame both in bucket 0.
    assert _accepts(link, [2, 0, 0, 0], [0, 1, 0, 0], own=1) is True
    # the salt slips to bucket 1 (late): reference subtracts it from bucket 0 and refuses the same honest release.
    assert _accepts(link, [1, 1, 0, 0], [0, 1, 0, 0], own=1) is False


def test_request_frames_are_a_function_of_prompt_length_alone():
    """CLAIM: `request_frames[n − 1]` is the ingress frames a request of an n-token prompt takes, a function of the
    prompt length alone (so the floor derivation can trust it).
    REALITY: closed — `active.request_frame_count` is ceil over the request's byte length, which `id_request` fixes
    from the prompt length and frame size; how the bytes were chunked onto the socket cannot change it, because the
    certifier's `Requests` cutter reassembles at frame boundaries. The frame count is identical however the same
    request bytes are split before framing.
    """
    raw = A.id_request(model="", user="", prompt=[0] * LONG, max_tokens=1, frame_bytes=B)
    framed = A.request_frames(raw, B)
    count = A.request_frame_count(len(raw), B)
    assert count == A.id_request_frames(LONG, B) == 2
    # the number of B-byte frames is a pure function of the byte length, independent of any socket-level chunking.
    assert len(framed) == count * B
    for chunk in (1, 7, 100, 4087, len(raw)):  # feed the same bytes in different socket-sized pieces
        pieces = [raw[i:i + chunk] for i in range(0, len(raw), chunk)]
        assert A.request_frame_count(sum(len(p) for p in pieces), B) == count
