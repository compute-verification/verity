"""N2's advised counts (`advised`): the circuit held to its reference, and advised link-windows through the proof
system on simulated vLLM serving written as I1's advice v0 (varied prompt, request and output lengths, chunked
prefill, prefix-cache hits and their sources, a preemption and its resume, the async row past a finish, a step's rows
out of request order, frames released after they're handed and responses spanning windows): honest traffic accepted
in every window, salted or not, and an advice whose output length is one off, which misses a request or moves a
release bucket, a response released before its request or faster than its declared rate, a window's ingress total
off, a sync outside Σ_sync, and extra egress each refused; and the ingress ends, derived from v0's prompt lengths
through the link's framing."""

import itertools
import json
import random
from collections import Counter
from dataclasses import fields, replace

import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.core.protocols.certifier import advice as ADV, advised as AD, head_check as HC, served as SV, \
    service as SVC
from verity.core.protocols.certifier.certificate import Direction, Issuer, parameters_digest
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync
from verity.core.protocols.certifier.service import ADVICE, OUTPUTS_ADVISED, PUBLIC_ADVISED, LinkWindows
from verity.core.protocols.sampled_proofs import service as S

import signers
import standin
from advised_traffic import (FRAMING, LINK, P, PIN, R, RIN, SYNC_SET, SYNCS, T, WINDOWS, Session, StandInClocks,
                             StandInTie, counts, egress_records, feeds_a_hit, held, ingress_records, moved, prompts,
                             serve, with_output_len, without_request)

SALTS = [tuple(bytes([4 * w + k + 1]) * 192 for k in range(4)) for w in range(WINDOWS)]
SEEDS = (1, 2, 3, 5, 7, 8)


def _view(link, w, syncs=SYNCS, certified=None, clocks=None):
    """`advised.view` under the stand-in tie on an ingress that certified the prompts `certified` (by default the
    advice's own), on windows the certifier ran under `clocks` (by default the declared `syncs`)."""
    return AD.view(link, w, T, R, syncs, SYNC_SET, FRAMING, clocks=syncs if clocks is None else clocks,
                   **held(link, certified))


def check(session: Session, link=None, own=0, egress=None, ingress=None, certified=None) -> list[bool]:
    """`advised.reference` on every window's heads' counts and its view of the advice, False where `view` refuses,
    under the tie on the session's certified ingress (or one that certified the prompts `certified`)."""
    link = session.link if link is None else link
    certified = session.certified if certified is None else certified
    out = []
    for w, (e, i) in enumerate(zip(counts(egress or session.egress), counts(ingress or session.ingress))):
        try:
            v = _view(link, w, certified=certified)
        except ValueError:
            out.append(False)
            continue
        out.append(AD.reference(T, RIN, own, i, e, v.released, v.floor, v.received))
    return out


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


class Run:
    """An advised link with its certifier's signer and the challenger's stream, on injected coins and clock, each
    window decided by the verifier of record on a stand-in proof (`standin`), the in-the-clear check beside it, and the
    tie from the verifier's own source, `ties` (by default the stand-in on an ingress that certified any advice's
    prompts; None, as in the package, refuses every window), and the certified syncs from its own clock source,
    `clocks` (by default the declared `SYNCS`)."""

    def __init__(self, tmp_path, own=0, ties=StandInTie(), clocks=StandInClocks()) -> None:
        self.tmp = tmp_path
        self.signer = signers.signer(tmp_path)
        self.issuer = Issuer(self.signer, signers.SESSION)
        self.lw = LinkWindows(LINK, P, SYNC_SET, self.signer.public_key(), "r-advised-test", signers.SESSION,
                              ingress=PIN, proof_deadline=60.0, advised=AD.AdvisedProfile(WINDOWS, FRAMING, own),
                              ties=ties, clocks=clocks)
        self.stream = self.lw.stream(budget=8, delta=2.0 ** -40, challenger="challenger-test",
                                     coins=lambda n: b"\x07" * n, clock=lambda: 0.0)
        self.previous = None

    def draw(self, w, egress, ingress):
        """Window w's opening on certificates of `egress` and `ingress`, registered, and its sample."""
        opening = self.lw.commit(w, self.issuer.certify(egress, Direction.EGRESS, P, parameters_digest(P, SYNC_SET)),
                                 self.issuer.certify(ingress, Direction.INGRESS, PIN, parameters_digest(PIN)),
                                 SALTS[w])
        reg = self.lw.register(self.stream, self.lw.registration(opening, self.previous))
        self.previous = reg.digest
        return opening, self.stream.select(reg)

    def round(self, w, egress, ingress, link) -> S.Outcome:
        opening, draw = self.draw(w, egress, ingress)
        return standin.decide(self.lw, self.stream, draw, opening, SYNCS,
                              self.tmp / f"proof-{draw.registration.order}", advice=link)

    def link(self, session: Session, link=None, egress=None, ingress=None) -> list[bool]:
        """Every window's outcome of record, in order, and the network verdict's agreement with their AND."""
        link = session.link if link is None else link
        egress, ingress = egress or session.egress, ingress or session.ingress
        accepted = [self.round(w, egress[w], ingress[w], link).accepted for w in range(WINDOWS)]
        verdict = self.lw.network_verdict(self.stream)
        assert verdict["accepted"] is all(accepted) and verdict["windows"] == WINDOWS
        assert verdict["refused"] == [w for w, a in enumerate(accepted) if not a]
        assert verdict["advice"] == (AD.digest(link) if any(accepted) else None)
        return accepted


def _past(advice) -> int:
    """The rows past a request's last kept output: the async row after a finish."""
    known, kept, out = [q.prompt_len for q in advice.requests], [0] * len(advice.requests), 0
    for st in advice.steps:
        for r in st.rows:
            if r.start + r.n == known[r.request]:
                if kept[r.request] < advice.requests[r.request].output_len:
                    kept[r.request] += 1
                    known[r.request] += 1
                else:
                    out += 1
    return out


def test_the_simulated_sessions_are_varied():
    """The sessions the tests run hold what makes traffic arbitrary: requests of different sizes and output lengths,
    chunked prefill, prefix-cache hits on other requests' KV, a preemption resumed from a hit, the async row past a
    finish, a step's rows out of request order (requests first scheduled in arrival order, the canonical map), frames
    released after they're handed (across a window's start too) and a response released in two windows."""
    sessions = [serve(seed) for seed in SEEDS]
    for s in sessions:
        assert len(ADV.outputs(s.link.advice)) == len(s.link.advice.release) == len(s.hand) == \
            sum(q.output_len for q in s.link.advice.requests)
    requests = [q for s in sessions for q in s.link.advice.requests]
    assert len({q.output_len for q in requests}) > 3
    assert {FRAMING[q.prompt_len - 1] for q in requests} == {1, 2, 3}
    rows = [(s, r) for s in sessions for st in s.link.advice.steps for r in st.rows]
    assert any(r.kind == "prefill" and r.start + r.n < s.link.advice.requests[r.request].prompt_len for s, r in rows)
    assert any(r.cache_hit and not r.preempted for _, r in rows)
    assert any(r.preempted and r.cache_hit for _, r in rows)
    assert any(_past(s.link.advice) for s in sessions)
    assert all(s.ordinal == sorted(s.ordinal) for s in sessions)
    assert any([r.request for r in st.rows] != sorted(r.request for r in st.rows)
               for s in sessions for st in s.link.advice.steps)
    pairs = [(h, b) for s in sessions for h, b in zip(s.hand, s.link.advice.release)]
    assert all(b >= h for h, b in pairs) and any(b // T > h // T for h, b in pairs)
    spans = [len({b // T for o, b in zip(ADV.outputs(s.link.advice), s.link.advice.release) if o.request == j})
             for s in sessions for j in range(len(s.link.advice.requests))]
    assert max(spans) > 1


@pytest.mark.parametrize("own", [0, 1])
def test_honest_advised_traffic_is_accepted_in_every_window(tmp_path, own):
    """No egress profile and no FIXLEN: each window's egress counts are the advice's releases in its buckets, its
    ingress counts release every output's request in time, and every window of every session is accepted through the
    proof system, salted (the certifier's own frame in each window) or not."""
    for seed in SEEDS:
        session = serve(seed, own)
        assert check(session, own=own) == [True] * WINDOWS
        (tmp_path / str(seed)).mkdir()
        assert Run(tmp_path / str(seed), own, StandInTie(session.certified)).link(session) == [True] * WINDOWS


def test_the_row_past_a_stop_is_accepted_and_hands_no_frame():
    """Async scheduling runs one decode past the stopping EOS (the third step samples past a
    request of 4 prompt and 2 kept tokens, 2 ingress frames released in bucket 0). Advice v0 accepts it and discards
    that sample, and N2 reads v0's outputs, so the window counts the 2 kept frames, released in buckets 1 and 2, and is
    accepted, and a frame for the discarded sample is extra egress."""
    a = ADV.Advice((ADV.RequestAdvice("a", 4, 2, "eos"),),
                   (ADV.StepAdvice((ADV.Row(0, "prefill", 0, 4),)), ADV.StepAdvice((ADV.Row(0, "decode", 4, 1),)),
                    ADV.StepAdvice((ADV.Row(0, "decode", 5, 1),))), (1, 2))
    assert _past(a) == 1 and len(ADV.outputs(a)) == 2
    v = _view(AD.LinkAdvice(a, (2,)), 0, [ClockSync(1, 2)])
    ingress = [2] + [0] * (T - 1)
    assert v.released == (0, 1, 1) + (0,) * (T - 3)
    assert AD.reference(T, RIN, 0, ingress, v.released, v.released, v.floor, v.received)
    for t in range(T):
        extra = [n + (u == t) for u, n in enumerate(v.released)]
        assert not AD.reference(T, RIN, 0, ingress, extra, v.released, v.floor, v.received)


def test_the_ingress_ends_are_derived_from_v0s_prompt_lengths():
    """No field of the advice but v0's says where a request ends: `advised.ends` reads each request's prompt length
    through the link's framing, request j the link's j-th, and in every session they are the ingress link's own ends
    (each request's last request frame, in arrival order), the last the windows' total. Ends off the windows' total
    (one frame more received) and a prompt past the framing are refused, in every window."""
    assert [f.name for f in fields(AD.LinkAdvice)] == ["advice", "received"]
    for seed in SEEDS:
        session = serve(seed)
        assert AD.ends(session.link, FRAMING, **held(session.link, session.certified)) == session.ends
        assert session.ends[-1] == sum(session.link.received)
    link = session.link
    more = AD.LinkAdvice(link.advice, link.received[:-1] + (link.received[-1] + 1,))
    for w in range(WINDOWS):
        with pytest.raises(ValueError, match="windows receive"):
            _view(more, w, certified=session.certified)
        with pytest.raises(ValueError, match="past the link's framing"):
            AD.view(link, w, T, R, SYNCS, SYNC_SET, FRAMING[:1], clocks=SYNCS, **held(link))


def _signed_end(counts, own, end):
    """The first bucket where the ingress head's signed counts, less one own frame a window, reach
    `end` request frames."""
    total = 0
    for t, n in enumerate(counts):
        total += n - (own if t % T == 0 else 0)
        if total >= end:
            return t
    return None


@pytest.mark.parametrize("own", [0, 1])
def test_the_ingress_ends_are_where_the_signed_counts_put_them(own):
    """Request k ends in the first bucket where the ingress head's signed counts, less one own frame
    a window, reach `advised.ends`'s k-th, Σ_{i ≤ k} request_frames[prompt_len_i − 1]. With each window's own frame at
    the head of its queue, ahead of carried request frames (`active.Bucketer`), that is the bucket that releases
    request k's last frame in every session, salted sessions whose windows start with frames carried in among them, and
    every window is accepted."""
    carried = 0
    for seed in SEEDS:
        s = serve(seed, own)
        sizes = [e - p for p, e in zip((0,) + s.ends, s.ends)]
        records, anchor = ingress_records(s.arrive, sizes, own)
        assert counts(records) == counts(s.ingress)
        got = [_signed_end([n for row in counts(records) for n in row], own, e)
               for e in AD.ends(s.link, FRAMING, **held(s.link, s.certified))]
        assert got == anchor
        assert check(s, own=own) == [True] * WINDOWS
        released = list(itertools.accumulate(sum(row.count(b"q") for row in r.grid) for r in records))
        arrived = [sum(n for a, n in zip(s.arrive, sizes) if a < w * T) for w in range(1, WINDOWS)]
        carried += any(a > r for a, r in zip(arrived, released))
    assert carried


def _one_more(link):
    """The link's advice with one more request, released and received in the last window: a view of every other
    window the same."""
    a = link.advice
    b = next(t for t in range(WINDOWS * T - 1, (WINDOWS - 1) * T + 1, -1) if Counter(a.release)[t] < R)
    more = ADV.Advice(a.requests + (ADV.RequestAdvice("u-extra", 2, 1, "eos"),),
                      a.steps + (ADV.StepAdvice((ADV.Row(len(a.requests), "prefill", 0, 2),)),), a.release + (b,))
    return AD.LinkAdvice(more, link.received[:-1] + (link.received[-1] + 1,))


def test_every_window_is_held_to_one_advice(tmp_path):
    """Each window reports the advice it read (`advised.digest`, its `advice` output, of the whole advice's encoding):
    a window checked against another advice that agrees with the link's on that window's view (one more request,
    released and received in the last window) is accepted alone, but the network verdict refuses it, so the link's
    counts are one advice's."""
    session = serve(SEEDS[0])
    link, a = session.link, session.link.advice
    other = _one_more(link)
    assert AD.digest(other) != AD.digest(link)
    assert [_view(other, w) == _view(link, w) for w in range(WINDOWS)] == \
        [True] * (WINDOWS - 1) + [False]
    finished = replace(a.requests[0], finish="length" if a.requests[0].finish == "eos" else "eos")
    renamed = AD.LinkAdvice(replace(a, requests=(finished,) + a.requests[1:]), link.received)
    assert AD.digest(renamed) != AD.digest(link)
    run = Run(tmp_path)
    outs = [run.round(w, session.egress[w], session.ingress[w], other if w == 2 else link) for w in range(WINDOWS)]
    assert [o.accepted for o in outs] == [True] * WINDOWS
    verdict = run.lw.network_verdict(run.stream)
    assert not verdict["accepted"] and verdict["refused"] == [2] and verdict["advice"] == AD.digest(link)


def test_the_proof_opens_only_to_the_verifiers_view_of_the_advice(tmp_path):
    """The verifier of record opens the proof's public row to its own view of the advice it is given: a proof made on
    the link's advice, whose unit accepts every window, is accepted against another advice where that advice's view of
    the window is the same, and refused in the last window, whose view differs; the clear check beside it, which reads
    the other advice, agrees."""
    session = serve(SEEDS[0])
    link, other = session.link, _one_more(session.link)
    run = Run(tmp_path)
    got = []
    for w in range(WINDOWS):
        opening, draw = run.draw(w, session.egress[w], session.ingress[w])
        delivered = standin.prove(run.lw, opening, draw, SYNCS, tmp_path / f"proof-{w}", advice=link)
        assert delivered.output == 0b11 and run.stream.deliver(draw, delivered.sessions()) is None
        out = run.lw.outcome(run.stream, draw, [r.certificate for r in opening.records], SYNCS, delivered,
                             circuit_sha512=standin.circuit_sha512(delivered), opening=opening, advice=other)
        verdict = run.stream.records[-1]["audit"]["verdicts"][SVC.VERIFIER]
        assert verdict["clear"]["accepted"] == out.accepted and verdict["public"]["advice"] == AD.digest(other)
        got.append(out.accepted)
    assert got == [True] * (WINDOWS - 1) + [False]


def test_a_salted_session_under_an_unsalted_profile_is_refused(tmp_path):
    """The own frame is counted only where the profile says so: a salted session's ingress under OWN = 0 (or an
    unsalted one under OWN = 1) is off by one in every window that receives."""
    salted, plain = serve(SEEDS[0], 1), serve(SEEDS[0], 0)
    assert not all(check(salted, own=0)) and not all(check(plain, own=1))
    assert Run(tmp_path, 0).link(salted) != [True] * WINDOWS


@pytest.mark.parametrize("d", [-1, 1])
def test_an_output_length_one_off_is_refused(tmp_path, d):
    """Each request's output length one short (its last frame unreleased) or one long (one frame more), each as an
    advice v0 accepts, in every session (one short: of the requests none of whose KV another's hit reads, since the row
    past its finish goes): the advice's releases aren't the certified egress, and a window is refused."""
    tried = 0
    for seed in SEEDS:
        session = serve(seed)
        a = session.link.advice
        for j, q in enumerate(a.requests):
            if q.output_len + d >= 1 and not (d < 0 and feeds_a_hit(a, j)):
                assert not all(check(session, with_output_len(session.link, j, d)))
                tried += 1
    assert tried > 20
    session = serve(SEEDS[0])
    j = next(j for j, q in enumerate(session.link.advice.requests) if q.output_len + d >= 1)
    assert not all(Run(tmp_path).link(session, with_output_len(session.link, j, d)))


@pytest.mark.parametrize("ingress", [False, True])
def test_a_missing_request_is_refused(tmp_path, ingress):
    """The advice without one request (none of whose KV another's hit reads), in every session: tied to the link's
    certified prompts, which still have the request, IoTieRequest's verdict refuses it, and so every window. Under a
    verdict on an ingress without it too, its frames are missing from the advice's releases; with its ingress frames
    kept, the requests' ends fall
    short of the windows' total, which every window refuses, and with them taken off too, the windows' `received`
    aren't the ingress's."""
    tried = 0
    for seed in SEEDS:
        session = serve(seed)
        a = session.link.advice
        for j in range(len(a.requests)):
            if feeds_a_hit(a, j):
                continue
            off = without_request(session.link, j, ingress)
            assert check(session, off) == [False] * WINDOWS
            own = check(session, off, certified=prompts(off))
            assert own == [False] * WINDOWS if not ingress else not all(own)
            tried += 1
    assert tried > 20
    session = serve(SEEDS[1])
    a = session.link.advice
    j = next(j for j in range(len(a.requests)) if not feeds_a_hit(a, j))
    run = Run(tmp_path, ties=StandInTie(session.certified))
    assert run.link(session, without_request(session.link, j, ingress)) == [False] * WINDOWS


def test_a_moved_release_bucket_is_refused(tmp_path):
    """Each kept output's release bucket moved one later or earlier, as an advice v0 accepts (its request's releases
    still non-decreasing): one bucket releases a frame fewer and its neighbour one more than the certified egress, and
    a window is refused."""
    session = serve(SEEDS[2])
    tried = []
    for k, d in itertools.product(range(len(session.link.advice.release)), (-1, 1)):
        try:
            off = moved(session.link, k, d)
        except ADV.MalformedAdvice:
            continue
        assert not all(check(session, off))
        tried.append((k, d))
    assert {d for _, d in tried} == {-1, 1}
    assert not all(Run(tmp_path).link(session, moved(session.link, *tried[0])))


def test_a_response_released_before_its_request_is_refused(tmp_path):
    """The same advice and egress, with the first request's ingress, and every request's before it, arriving when its
    response's first frame is released: the counts are the advice's, but the ingress head hasn't released the request
    by that frame's eligibility, so the floor refuses the window."""
    session = serve(SEEDS[0])
    a, j = session.link.advice, session.ordinal.index(0)
    first = min(b for o, b in zip(ADV.outputs(a), a.release) if o.request == j)
    sizes = [e - p for p, e in zip((0,) + session.ends, session.ends)]
    late, _ = ingress_records([max(t, first) for t in session.arrive], sizes, 0)
    link = AD.LinkAdvice(a, tuple(sum(row.count(b"q") for row in r.grid) for r in late))
    assert counts(late) != counts(session.ingress)
    assert [list(_view(link, w, certified=session.certified).released) for w in range(WINDOWS)] == \
        counts(session.egress)
    w = (first - SYNCS[first // T].delta) // T
    assert not check(session, link, ingress=late)[w]
    assert not Run(tmp_path).link(session, link, ingress=late)[w]


def test_a_release_faster_than_the_declared_rate_is_refused():
    """The floor holds each output to the declared rate, not only to its request: a request of one ingress frame,
    released in bucket 1, whose 3 outputs are all released in bucket 2. Under δ = 1, ρ = 4 all three are eligible there
    and the window is accepted; under ρ = 2 the third is eligible a bucket later, so the floor wants the request by
    bucket 0 and the ingress counts refuse it, while the third released in bucket 3 is accepted."""
    a = ADV.Advice((ADV.RequestAdvice("a", 1, 3, "length"),),
                   (ADV.StepAdvice((ADV.Row(0, "prefill", 0, 1),)), ADV.StepAdvice((ADV.Row(0, "decode", 1, 1),)),
                    ADV.StepAdvice((ADV.Row(0, "decode", 2, 1),))), (2, 2, 2))
    ingress = [0, 1] + [0] * (T - 2)

    def accepted(link, sync):
        v = _view(link, 0, [sync])
        return AD.reference(T, RIN, 0, ingress, v.released, v.released, v.floor, v.received)

    link = AD.LinkAdvice(a, (1,))
    assert accepted(link, ClockSync(1, 4)) and not accepted(link, ClockSync(1, 2))
    assert accepted(AD.LinkAdvice(replace(a, release=(2, 2, 3)), (1,)), ClockSync(1, 2))


def test_a_windows_ingress_total_off_is_refused(tmp_path):
    """`received` with one request frame moved to the next window: the totals still match the requests', but each of
    the two windows' ingress counts sums to one off its `received`."""
    session = serve(SEEDS[0])
    got = list(session.link.received)
    w = next(k for k in range(WINDOWS - 1) if got[k])
    got[w] -= 1
    got[w + 1] += 1
    link = AD.LinkAdvice(session.link.advice, tuple(got))
    assert check(session, link)[w:w + 2] == [False, False]
    assert Run(tmp_path).link(session, link)[w:w + 2] == [False, False]


@pytest.mark.parametrize("where", ["busy", "idle"])
def test_extra_egress_is_refused(tmp_path, where):
    """One frame more on the egress link than the advice releases, handed in a bucket with frames or in an idle one:
    the certificate verifies and the audit accepts, but the counts aren't the advice's releases."""
    session = serve(SEEDS[3])
    handed = list(session.handed)
    t = next(t for t in range(WINDOWS * T) if (handed[t] > 0) == (where == "busy"))
    handed[t] += 1
    extra = egress_records(handed)
    assert counts(extra) != counts(session.egress)
    assert not all(check(session, egress=extra))
    assert not all(Run(tmp_path).link(session, egress=extra))


class _Fixed:
    """A tie source whose record is `registrations` and whose verdict is `tie`, whatever it is asked."""

    def __init__(self, tie, registrations) -> None:
        self.tie, self.regs = tie, tuple(registrations)

    def registrations(self, link):
        return self.regs

    def verdict(self, link, advice, profile):
        return self.tie


def test_no_advice_or_a_malformed_one_is_refused(tmp_path):
    """An advised window with no advice or I1's advice alone is refused, and so is one without the tie from the
    verifier's own source, or under a verdict that refuses an ingress window or doesn't cover every one, or names
    another advice, another link or other ingress registrations than the source's record, in the clear and by the
    verifier of record on the proof; a `LinkAdvice` takes no tie at all, and a `received` that isn't u32s and a
    verdict that doesn't name a link, registration digests and an advice digest with one bool a registration can't be
    built; and every window refuses an advice that releases more than R frames in a bucket (all in the last), a frame
    before its request could be (the first in bucket 0), or one past the link's windows; and a window that declares no
    sync, or one outside Σ_sync, is refused, and only that window: every other window's view is the honest one, since
    every floor reads the sync the certifier ran."""
    session = serve(SEEDS[0])
    run = Run(tmp_path, ties=StandInTie(session.certified))
    opening = run.lw.commit(0, run.issuer.certify(session.egress[0], Direction.EGRESS, P,
                                                  parameters_digest(P, SYNC_SET)),
                            run.issuer.certify(session.ingress[0], Direction.INGRESS, PIN, parameters_digest(PIN)),
                            SALTS[0])
    link, a = session.link, session.link.advice
    assert run.lw.advised_check(opening, SYNCS, link)
    assert not run.lw.advised_check(opening, SYNCS, None) and not run.lw.advised_check(opening, SYNCS, a)
    got = link.received
    for advice, received in ((ADV.to_json(a), got), (a, (-1,) + got[1:]), (a, got + (1 << 32,)), (a, 4)):
        with pytest.raises(ADV.MalformedAdvice):
            AD.LinkAdvice(advice, received)
    tie = held(link, session.certified)
    good = tie["tie"]
    with pytest.raises(TypeError):
        AD.LinkAdvice(a, got, good)
    d, regs, every = good.advice, good.registrations, good.accepted
    for name, registrations, advice, accepted in (
            ("", regs, d, every), (b"x", regs, d, every), (LINK, regs[0], d, every),
            (LINK, (d[:-1],) * WINDOWS, d, every), (LINK, regs, d.upper(), every),
            (LINK, regs, bytes.fromhex(d), every), (LINK, regs, d, (1,) * WINDOWS),
            (LINK, regs, d, "yes"), (LINK, regs, d, every[:-1]), (LINK, regs[:-1], d, every)):
        with pytest.raises(ADV.MalformedAdvice):
            AD.IoTieVerdict(name, registrations, advice, accepted)
    other_regs = StandInTie(session.certified, "requests-elsewhere").registrations(LINK)
    bad_ties = [(None, "without IoTieRequest's verdict"),
                (replace(good, accepted=(True,) * (WINDOWS - 1) + (False,)), "doesn't accept"),
                (AD.IoTieVerdict(LINK, regs[:-1], d, every[:-1]), "other ingress registrations"),
                (StandInTie().verdict(LINK, _one_more(link)), "another advice"),
                (replace(good, link="requests-elsewhere"), "another link"),
                (replace(good, registrations=other_regs), "other ingress registrations")]
    for bad, why in bad_ties:
        for w in range(WINDOWS):
            with pytest.raises(ValueError, match=why):
                AD.view(link, w, T, R, SYNCS, SYNC_SET, FRAMING, clocks=SYNCS, **(tie | {"tie": bad}))
        assert not replace(run.lw, ties=_Fixed(bad, regs)).advised_check(opening, SYNCS, link)
    assert replace(run.lw, ties=_Fixed(good, regs)).advised_check(opening, SYNCS, link)
    assert not replace(run.lw, ties=None).advised_check(opening, SYNCS, link)
    rounds = [(StandInTie(session.certified), None), (StandInTie(session.certified), a), (None, link),
              (StandInTie(session.certified, "requests-elsewhere"), link),
              (StandInTie(session.certified, windows=WINDOWS - 1), link)]
    for k, (ties, bad) in enumerate(rounds):
        (tmp_path / f"run-{k}").mkdir()
        out = Run(tmp_path / f"run-{k}", ties=ties).round(0, session.egress[0], session.ingress[0], bad)
        assert not out.accepted
    over, early, late = (WINDOWS * T - 1,) * len(a.release), (0,) + a.release[1:], a.release[:-1] + (WINDOWS * T,)
    for release, why in ((over, "past the allocation"), (early, "before its request"), (late, "past the link's")):
        bad = AD.LinkAdvice(replace(a, release=release), got)
        for w in range(WINDOWS):
            with pytest.raises(ValueError, match=why):
                _view(bad, w, certified=session.certified)
        assert not run.lw.advised_check(opening, SYNCS, bad)
    assert sorted({b // T for b in a.release})[-1] > 0
    honest = [_view(link, w) for w in range(WINDOWS)]
    for v in range(WINDOWS):
        for sync in (ClockSync(0, 2), ClockSync(1, 1), ClockSync(3, 4)):
            off = SYNCS[:v] + [sync] + SYNCS[v + 1:]
            for w in range(WINDOWS):
                if w == v:
                    with pytest.raises(ValueError, match="Σ_sync"):
                        _view(link, w, off, clocks=SYNCS)
                else:
                    assert _view(link, w, off, clocks=SYNCS) == honest[w], (v, sync, w)
            assert run.lw.advised_check(opening, off, link) is (v != 0)
        with pytest.raises(ValueError, match="declares no sync"):
            _view(link, v, SYNCS[:v], clocks=SYNCS)
    with pytest.raises(ValueError, match="receives in"):
        _view(link, WINDOWS)


KEYS = [(8, 2, 0), (8, 2, 1), (5, 1, 1), (1, 3, 0), (1, 3, 1)]


def _cases(key, rng, n):
    T_, RIN_, OWN = key
    cases = []
    for _ in range(n):
        released = [rng.choice((0, 0, 1, 2, 5, 1 << 20, AD.U32_MAX)) for _ in range(T_)]
        egress = list(released)
        ingress = [rng.randint(0, RIN_) for _ in range(T_)]
        if rng.random() < 0.8:
            ingress[0] = max(ingress[0], OWN)
        cum = list(itertools.accumulate(ingress))
        floor = [max(0, c - OWN - rng.choice((0, 0, 1, 2))) for c in cum]
        floor = [max(floor[:t + 1]) for t in range(T_)]
        received = cum[-1] - OWN
        roll = rng.random()
        if roll < 0.15:
            k = rng.randrange(T_)
            egress[k] ^= 1 << rng.randrange(32)
        elif roll < 0.25:
            floor[rng.randrange(T_)] += rng.choice((1, 1 << 31))
        elif roll < 0.35:
            received += rng.choice((-1, 1))
        elif roll < 0.45:
            ingress[rng.randrange(T_)] = RIN_ + rng.choice((1, 1 << 24))
        cases.append((ingress, egress, AD.WindowView(tuple(released), tuple(floor), max(0, received))))
    return cases


@pytest.mark.parametrize("key", KEYS)
def test_the_circuit_is_its_reference(key):
    """Random windows of each size class, salted and not, with egress counts equal to the releases or one bit off
    anywhere in a u32, floors met and missed, ingress past R_in and bucket 0 without its own frame, and every honest
    session's windows: the circuit's bit is `advised.reference`'s."""
    rng = random.Random(sum(key))
    cases = _cases(key, rng, 300)
    if key[:2] == (T, RIN):
        for seed in SEEDS:
            session = serve(seed, key[2])
            for w in range(WINDOWS):
                cases.append((counts(session.ingress)[w], counts(session.egress)[w], _view(session.link, w)))
    want = [int(AD.reference(*key, i, e, v.released, v.floor, v.received)) for i, e, v in cases]
    assert AD.evaluate(key, cases) == want and 0 < sum(want) < len(want)


def test_the_advised_unit_is_n2s_beside_the_advised_counts(tmp_path):
    """The advised unit holds N2's checks and the advised counts as one gate, the spec declares the advice and the
    advised sizes public and binds the framing, an advised link has no egress occupancy profile or served profile, and
    one T and τ_b on both ends, and a framing is a positive frame count for each prompt length from 1 on."""
    lw = Run(tmp_path).lw
    assert lw.unit.id == "NetworkAdvisedLinkWindow_v2{EGRESS=(9,8,0),INGRESS=(9,8,0),ADVISED=(8,2,0)}"
    assert lw.proof_unit.id == lw.unit.id and lw.proof_unit.signature() == lw.unit.signature()
    assert set(ADVICE) <= set(lw.layout["pub"]) and lw.bits("pub") % 16 == 0
    assert lw.spec.public == PUBLIC_ADVISED and {"advice", "advised", "sync"} <= {p.item for p in lw.spec.public}
    assert lw.spec.check.public_outputs == OUTPUTS_ADVISED == ("verdict", "deadline", "advice")
    assert "served" not in {p.item for p in lw.spec.public} and lw.windows == WINDOWS
    assert lw.spec.digest() != replace(lw, advised=AD.AdvisedProfile(WINDOWS + 1, FRAMING)).spec.digest()
    assert lw.spec.digest() != replace(lw, advised=AD.AdvisedProfile(WINDOWS, FRAMING + (4,))).spec.digest()
    assert replace(lw, advised=AD.AdvisedProfile(WINDOWS, FRAMING, 1)).unit.id.endswith("ADVISED=(8,2,1)}")
    assert lw.gates()[-1] is AD.gate(T, RIN, 0)
    with pytest.raises(ValueError, match="served .* or advised"):
        replace(lw, served=SV.ServedProfile(3, 1, WINDOWS))
    with pytest.raises(ValueError, match="no egress occupancy"):
        replace(lw, occupancy=(((0,) * T,), None))
    with pytest.raises(ValueError, match="share T"):
        replace(lw, ingress=GridParams(4, 2, 1))
    with pytest.raises(ValueError, match="0 or 1 own"):
        AD.AdvisedProfile(WINDOWS, FRAMING, 2)
    for framing in ((), (1, 0), (1, 1 << 32), "123", (1.0,)):
        with pytest.raises(ValueError, match="framing"):
            AD.AdvisedProfile(WINDOWS, framing)


def _accepts_any_counts(B, S_, *args):
    return B.call(BL.constant(1))


@pytest.mark.parametrize("key", [(8, 2, 0), (8, 2, 1)])
def test_the_advised_gate_names_its_pinned_circuit_digest(key):
    """The advised counts' gate names its Definition's circuit digest as `advised.circuit-check.json` pins it, which is
    the Definition's own (`head_check.expand`), and advised counts with the same id and signature that accept any
    counts expand no gate of it."""
    d, gate = AD.unit(*key), AD.gate(*key)
    pinned = json.loads(AD.PINS.read_text())["digests"][d.id]["sha512"]
    assert gate.id == f"Gate[{d.id}@{pinned}]_v2" and HC.pinned(gate) and HC.expand(gate, d) is d
    forged = bind(replace(AD.NetworkAdvisedCounts, body=_accepts_any_counts, register=False),
                  **dict(zip(AD.STATICS, key)))
    assert forged.id == d.id and forged.signature() == d.signature()
    with pytest.raises(ValueError, match="another circuit"):
        HC.expand(gate, forged)
