"""N2's advised counts (`advised`): the circuit held to its reference, and advised link-windows through the proof
service on simulated vLLM serving written as I1's advice v0 (varied prompt, request and output lengths, chunked
prefill, prefix-cache hits and their sources, a preemption and its resume, the async row past a finish, requests first
scheduled out of their arrival order, frames released after they're handed and responses spanning windows): honest
traffic accepted in every window, salted or not, and an advice whose output length is one off, which misses a request
or moves a release bucket, a response released before its request, a window's ingress total off, and extra egress
each refused; and the advice's charge, v0's bits and the ingress ends'."""

import itertools
import json
import random
from collections import Counter, deque
from dataclasses import dataclass, replace

import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.core.protocols.certifier import advice as ADV, advised as AD, head_check as HC, served as SV
from verity.core.protocols.certifier.certificate import Direction, Issuer, parameters_digest
from verity.core.protocols.certifier.grid import Certifier, GridParams, occupancy
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.certifier.service import OUTPUTS_ADVISED, PUBLIC_ADVISED, LinkWindows
from verity.core.protocols.sampled_proofs import service as S

import signers

LINK = "responses"
T, WINDOWS = 8, 4
P = GridParams(buckets=T, allocation=3, frame_bytes=1)
PIN = GridParams(buckets=T, allocation=2, frame_bytes=1)
R, RIN = P.allocation, PIN.allocation
SYNC_SET = SyncSet((1, 2), (2, 4))
SYNCS = [ClockSync(1, 2), ClockSync(2, 4), ClockSync(2, 2), ClockSync(1, 4)]
SALTS = [tuple(bytes([4 * w + k + 1]) * 192 for k in range(4)) for w in range(WINDOWS)]
BUDGET, SEQS = 6, 4
SEEDS = (1, 2, 3, 5, 7, 8)


@dataclass(frozen=True)
class Session:
    """A simulated session on the link: its advice, each end's records, each request's ingress arrival bucket (in
    arrival order), each advice request's place in that order, each kept output's hand bucket (in `advice.outputs`
    order) and the frames handed in each bucket."""

    link: AD.LinkAdvice
    egress: list
    ingress: list
    arrive: list[int]
    ordinal: list[int]
    hand: list[int]
    handed: list[int]


def ingress_records(arrive, frames_in, own):
    """The ingress certifier's records, with its own frame queued first in each window (behind any carried request
    frames, as `active.Bucketer` queues it), and each request's anchor, the bucket that releases its last frame."""
    cert, records, anchor, got = Certifier(LINK, PIN), [], [None] * len(arrive), 0
    need = list(itertools.accumulate(frames_in))
    for t in range(WINDOWS * T):
        if own and t % T == 0:
            cert.enqueue(b"s")
        for j, a in enumerate(arrive):
            if a == t:
                for _ in range(frames_in[j]):
                    cert.enqueue(b"q")
        got += cert.tick().count(b"q")
        anchor = [t if a is None and need[j] <= got else a for j, a in enumerate(anchor)]
        if (t + 1) % T == 0:
            records.append(cert.close_window())
    return records, anchor


def egress_records(handed):
    cert, records = Certifier(LINK, P), []
    for t in range(WINDOWS * T):
        for _ in range(handed[t]):
            cert.enqueue(b"e")
        cert.tick()
        if (t + 1) % T == 0:
            records.append(cert.close_window())
    assert cert.queued == 0, "the egress queue drains by the link's last window"
    return records


def counts(records):
    return [list(occupancy(r.grid)) for r in records]


def _hit(made, j, want, resume):
    """The longest prefix of `want` positions whose KV some request computed (request j itself only on a resume), and
    its sources, each run's request the first that holds it."""
    runs = []
    for p in range(want):
        have = [i for i in range(len(made)) if p in made[i] and (i != j or resume)]
        if not have:
            break
        if runs and runs[-1][0] in have:
            runs[-1][1] += 1
        else:
            runs.append([have[0], 1])
    return sum(n for _, n in runs), [tuple(r) for r in runs]


def serve(seed: int, own: int = 0) -> Session:
    """A vLLM-like session: 5 … 8 requests of 1 … 3 ingress frames arriving over the first two windows, and a scheduler
    of two steps a bucket, a token budget of 6 and 4 sequences, which admits a request the bucket after its last
    ingress frame is released, prefills it in chunks (from a prefix-cache hit on another request's KV, where one is
    wanted and computed), decodes it, preempts one running request and recomputes it on resume (from a hit on its own
    and others' KV), and runs the async row past some finishes. Rows are in Model Runner V2's order (n = 1 first), so
    two requests first scheduled in one step can be listed out of their arrival order. Each step's kept outputs are
    handed to the egress certifier the bucket after the step runs, one frame each, and released FIFO at R, so frames
    released after they're handed carry across windows; `release` is the certified egress record's."""
    rng = random.Random(seed)
    n = rng.randint(5, 8)
    arrive = sorted(rng.randrange(2 * T) for _ in range(n))
    frames_in = [rng.randint(1, 3) for _ in range(n)]
    prompt = [rng.randint(2, 9) for _ in range(n)]
    out = [rng.randint(1, 10) for _ in range(n)]
    want = [rng.randrange(p) if rng.random() < 0.5 else 0 for p in prompt]
    trailing = [rng.random() < 0.4 for _ in range(n)]
    finish = [rng.choice(ADV.FINISHES) for _ in range(n)]
    ingress, anchor = ingress_records(arrive, frames_in, own)

    c, sampled, made, admit = [0] * n, [0] * n, [set() for _ in range(n)], [None] * n
    admitted, waiting, running, done = set(), deque(), [], set()
    preempt_at, preempted_once = rng.randrange(6, 20), False
    steps, hand = [], []
    for s in itertools.count():
        if len(done) == n:
            break
        b = s // 2
        assert b + 1 < WINDOWS * T, "the session outlasts the link's windows"
        for j in range(n):
            if j not in admitted and anchor[j] < b:
                admitted.add(j)
                waiting.append(j)
                c[j], sources = _hit(made, j, want[j], False)
                admit[j] = (c[j], False, sources)
        while waiting and len(running) < SEQS:
            running.append(waiting.popleft())
        victims = [j for j in running if 0 < sampled[j] < out[j]]
        if s >= preempt_at and not preempted_once and victims:
            j = victims[0]
            running.remove(j)
            waiting.appendleft(j)
            c[j], sources = _hit(made, j, rng.randrange(prompt[j] + sampled[j]), True)
            admit[j], preempted_once = (c[j], True, sources), True
        budget, rows = BUDGET, []
        for j in running:
            k = min(prompt[j] + sampled[j] - c[j], budget)
            if k == 0:
                break
            hit, resumed, sources = admit[j] or (0, False, [])
            admit[j] = None
            rows.append((j, "decode" if k == 1 and c[j] >= prompt[j] else "prefill", c[j], k, hit, resumed, sources))
            made[j].update(range(c[j], c[j] + k))
            c[j] += k
            budget -= k
        if not rows:
            continue
        rows.sort(key=lambda r: (r[3] != 1, r[3]))
        for j, _, start, k, *_ in rows:
            if start + k == prompt[j] + sampled[j]:
                if sampled[j] < out[j]:
                    hand.append(b + 1)
                sampled[j] += 1
                if sampled[j] > out[j] or (sampled[j] == out[j] and not trailing[j]):
                    running.remove(j)
                    done.add(j)
        steps.append(rows)
    first = list(dict.fromkeys(r[0] for rows in steps for r in rows))
    new = {j: i for i, j in enumerate(first)}
    handed = [0] * (WINDOWS * T)
    for h in hand:
        handed[h] += 1
    egress = egress_records(handed)
    release = tuple(t for t, k in enumerate(c for row in counts(egress) for c in row) for _ in range(k))
    advice = ADV.Advice(
        tuple(ADV.RequestAdvice(f"u{j}", prompt[j], out[j], finish[j]) for j in first),
        tuple(ADV.StepAdvice(tuple(ADV.Row(new[j], kind, start, k, hit, resumed,
                                           tuple(ADV.Source(new[i], m) for i, m in sources))
                                   for j, kind, start, k, hit, resumed, sources in rows)) for rows in steps),
        release)
    need = list(itertools.accumulate(frames_in))
    received = tuple(sum(row.count(b"q") for row in r.grid) for r in ingress)
    link = AD.LinkAdvice(advice, tuple(need[j] for j in first), received)
    return Session(link, egress, ingress, arrive, first, hand, handed)


def check(session: Session, link=None, own=0, egress=None, ingress=None) -> list[bool]:
    """`advised.reference` on every window's heads' counts and its view of the advice, False where `view` refuses."""
    link = session.link if link is None else link
    out = []
    for w, (e, i) in enumerate(zip(counts(egress or session.egress), counts(ingress or session.ingress))):
        try:
            v = AD.view(link, w, T, R, SYNCS)
        except ValueError:
            out.append(False)
            continue
        out.append(AD.reference(T, RIN, own, i, e, v.released, v.floor, v.received))
    return out


class Run:
    """An advised link with its certifier's signer and the challenger's stream, on injected coins and clock."""

    def __init__(self, tmp_path, own=0) -> None:
        self.signer = signers.signer(tmp_path)
        self.issuer = Issuer(self.signer, signers.SESSION)
        self.lw = LinkWindows(LINK, P, SYNC_SET, self.signer.public_key(), "r-advised-test", signers.SESSION,
                              ingress=PIN, proof_deadline=60.0, advised=AD.AdvisedProfile(WINDOWS, own))
        self.stream = self.lw.stream(budget=8, delta=2.0 ** -40, challenger="challenger-test",
                                     coins=lambda n: b"\x07" * n, clock=lambda: 0.0)
        self.previous = None

    def round(self, w, egress, ingress, link) -> S.Outcome:
        opening = self.lw.commit(w, self.issuer.certify(egress, Direction.EGRESS, P, parameters_digest(P, SYNC_SET)),
                                 self.issuer.certify(ingress, Direction.INGRESS, PIN, parameters_digest(PIN)),
                                 SALTS[w])
        reg = self.lw.register(self.stream, self.lw.registration(opening, self.previous))
        self.previous = reg.digest
        draw = self.stream.select(reg)
        assert self.stream.deliver(draw, [self.lw.session(opening, draw)]) is None
        return self.lw.outcome(self.stream, draw, opening, SYNCS, advice=link)

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


def _rebuilt(link, requests, steps, extra=(), ingress_end=None, received=None):
    """`link` with the advice's requests and steps replaced, each kept output released where `link` releases the same
    request's (by id) same output, or where `extra`'s (id, index, bucket) says."""
    a = link.advice
    at = {(a.requests[o.request].id, o.index): b for o, b in zip(ADV.outputs(a), a.release)}
    at |= {(i, k): b for i, k, b in extra}
    probe = ADV.Advice(requests, steps, (0,) * sum(q.output_len for q in requests))
    release = tuple(at[(requests[o.request].id, o.index)] for o in ADV.outputs(probe))
    return AD.LinkAdvice(ADV.Advice(requests, steps, release),
                         link.ingress_end if ingress_end is None else ingress_end,
                         link.received if received is None else received)


def _room(link, j, frm=None):
    """The first bucket from request j's last release on (or from `frm`) that releases fewer than R frames."""
    a, per = link.advice, Counter(link.advice.release)
    last = max((b for o, b in zip(ADV.outputs(a), a.release) if o.request == j), default=0) if frm is None else frm
    return next(t for t in range(last, WINDOWS * T) if per[t] < R)


def _with_output_len(link, j, d):
    """The advice with request j's output length d off, as an advice v0 accepts: one short, its last kept output's row
    becomes the row past its finish (and the one there was is dropped); one long, the row past its finish keeps its
    sample (or a decode row is added at the end), released at the first bucket from its last with room."""
    a = link.advice
    q = a.requests[j]
    requests = a.requests[:j] + (replace(q, output_len=q.output_len + d),) + a.requests[j + 1:]
    past = q.prompt_len + q.output_len
    steps = [st.rows for st in a.steps]
    extra = ()
    if d < 0:
        steps = [tuple(r for r in rows if not (r.request == j and r.start + r.n == past)) for rows in steps]
    else:
        if not any(r.request == j and r.start + r.n == past for rows in steps for r in rows):
            steps.append((ADV.Row(j, "decode", past - 1, 1),))
        extra = ((q.id, q.output_len, _room(link, j)),)
    return _rebuilt(link, requests, tuple(ADV.StepAdvice(rows) for rows in steps if rows), extra)


def _without_request(link, j, ingress):
    """The advice with request j dropped, its rows with it, and, with `ingress`, its ingress frames off the link: the
    later requests' ends lowered by them, and `received` lowered in the last windows that released any."""
    a = link.advice
    index = {k: k - (k > j) for k in range(len(a.requests)) if k != j}
    steps = []
    for st in a.steps:
        rows = tuple(replace(r, request=index[r.request],
                             sources=tuple(replace(s, request=index[s.request]) for s in r.sources))
                     for r in st.rows if r.request != j)
        if rows:
            steps.append(ADV.StepAdvice(rows))
    requests = a.requests[:j] + a.requests[j + 1:]
    ends, received = list(link.ingress_end), list(link.received)
    end = ends.pop(j)
    if ingress:
        size = end - max([e for e in ends if e < end], default=0)
        ends = [e - size if e > end else e for e in ends]
        for w in reversed(range(len(received))):
            take = min(size, received[w])
            received[w] -= take
            size -= take
    return _rebuilt(link, requests, tuple(steps), ingress_end=tuple(ends), received=tuple(received))


def _moved(link, k, d):
    release = list(link.advice.release)
    release[k] += d
    return AD.LinkAdvice(replace(link.advice, release=tuple(release)), link.ingress_end, link.received)


def test_the_simulated_sessions_are_varied():
    """The sessions the tests run hold what makes traffic arbitrary: requests of different sizes and output lengths,
    chunked prefill, prefix-cache hits on other requests' KV, a preemption resumed from a hit, the async row past a
    finish, requests first scheduled out of their arrival order, frames released after they're handed (across a
    window's start too) and a response released in two windows."""
    sessions = [serve(seed) for seed in SEEDS]
    for s in sessions:
        assert len(ADV.outputs(s.link.advice)) == len(s.link.advice.release) == len(s.hand) == \
            sum(q.output_len for q in s.link.advice.requests)
    requests = [q for s in sessions for q in s.link.advice.requests]
    assert len({q.output_len for q in requests}) > 3
    sizes = {e - p for s in sessions for p, e in zip([0] + sorted(s.link.ingress_end), sorted(s.link.ingress_end))}
    assert sizes == {1, 2, 3}
    rows = [(s, r) for s in sessions for st in s.link.advice.steps for r in st.rows]
    assert any(r.kind == "prefill" and r.start + r.n < s.link.advice.requests[r.request].prompt_len for s, r in rows)
    assert any(r.cache_hit and not r.preempted for _, r in rows)
    assert any(r.preempted and r.cache_hit for _, r in rows)
    assert any(_past(s.link.advice) for s in sessions)
    assert any(s.ordinal != sorted(s.ordinal) for s in sessions)
    pairs = [(h, b) for s in sessions for h, b in zip(s.hand, s.link.advice.release)]
    assert all(b >= h for h, b in pairs) and any(b // T > h // T for h, b in pairs)
    spans = [len({b // T for o, b in zip(ADV.outputs(s.link.advice), s.link.advice.release) if o.request == j})
             for s in sessions for j in range(len(s.link.advice.requests))]
    assert max(spans) > 1


@pytest.mark.parametrize("own", [0, 1])
def test_honest_advised_traffic_is_accepted_in_every_window(tmp_path, own):
    """No egress profile and no FIXLEN: each window's egress counts are the advice's releases in its buckets, its
    ingress counts release every output's request in time, and every window of every session is accepted through the
    proof service, salted (the certifier's own frame in each window) or not."""
    for seed in SEEDS:
        session = serve(seed, own)
        assert check(session, own=own) == [True] * WINDOWS
        (tmp_path / str(seed)).mkdir()
        assert Run(tmp_path / str(seed), own).link(session) == [True] * WINDOWS


def test_the_row_past_a_stop_is_accepted_and_hands_no_frame():
    """The red team's case: async scheduling runs one decode past the stopping EOS (the third step samples past a
    request of 4 prompt and 2 kept tokens). Advice v0 accepts it and discards that sample, and N2 reads v0's outputs, so
    the window counts the 2 kept frames and is accepted, and a frame for the discarded sample is extra egress."""
    a = ADV.Advice((ADV.RequestAdvice("a", 4, 2, "eos"),),
                   (ADV.StepAdvice((ADV.Row(0, "prefill", 0, 4),)), ADV.StepAdvice((ADV.Row(0, "decode", 4, 1),)),
                    ADV.StepAdvice((ADV.Row(0, "decode", 5, 1),))), (0, 1))
    assert _past(a) == 1 and len(ADV.outputs(a)) == 2
    v = AD.view(AD.LinkAdvice(a, (2,), (2,)), 0, T, R, [ClockSync(0, 1)])
    ingress = [2] + [0] * (T - 1)
    assert v.released == (1, 1) + (0,) * (T - 2)
    assert AD.reference(T, RIN, 0, ingress, v.released, v.released, v.floor, v.received)
    for t in range(T):
        extra = [n + (u == t) for u, n in enumerate(v.released)]
        assert not AD.reference(T, RIN, 0, ingress, extra, v.released, v.floor, v.received)


def test_the_ingress_ends_are_charged_beside_advice_v0s_bits():
    """`advised.bits` charges v0's bits and bitlen(Σ received) bits for each request's ingress end, and nothing for
    `received`, which the unit holds to the ingress head; the windows' charges sum to `total_bits`. An end goes to the
    earliest window whose floor reads it, which may precede its request's first release: here a release at bucket T
    with δ = 1 needs the request's frames by bucket T − 1, so window 0 reads its end, while v0 charges window 1."""
    for seed in SEEDS:
        link = serve(seed).link
        width = sum(link.received).bit_length()
        assert AD.ingress_bits(link) == len(link.advice.requests) * width > 0
        assert AD.total_bits(link) == ADV.total_bits(link.advice) + AD.ingress_bits(link)
        by, v0 = AD.bits_by_window(link, T, SYNCS), ADV.bits_by_window(link.advice, T)
        assert sum(by.values()) == AD.total_bits(link)
        assert all(by.get(w, 0) >= n for w, n in v0.items())
        assert sum(by.values()) - sum(v0.values()) == AD.ingress_bits(link)
        assert [AD.bits(link, w, T, SYNCS) for w in range(WINDOWS)] == [by.get(w, 0) for w in range(WINDOWS)]
    a = ADV.Advice((ADV.RequestAdvice("a", 4, 1, "eos"),), (ADV.StepAdvice((ADV.Row(0, "prefill", 0, 4),)),), (T,))
    link = AD.LinkAdvice(a, (2,), (2, 0))
    assert ADV.bits_by_window(a, T) == {1: ADV.total_bits(a)}
    assert AD.bits_by_window(link, T, [ClockSync(1, 1)] * 2) == {0: 2, 1: ADV.total_bits(a)}
    assert AD.view(link, 0, T, R, [ClockSync(1, 1)] * 2).floor[T - 1] == 2


def test_every_window_is_held_to_one_advice(tmp_path):
    """Each window reports the advice it read (`advised.digest`, its `advice` output, of the whole advice's encoding):
    a window checked against another advice that agrees with the link's on that window's view (one more request,
    released and received in the last window) is accepted alone, but the network verdict refuses it, so the link's
    counts are one advice's."""
    session = serve(SEEDS[0])
    link, a = session.link, session.link.advice
    b = next(t for t in range(WINDOWS * T - 1, (WINDOWS - 1) * T + 1, -1) if Counter(a.release)[t] < R)
    other = AD.LinkAdvice(ADV.Advice(a.requests + (ADV.RequestAdvice("u-extra", 2, 1, "eos"),),
                                     a.steps + (ADV.StepAdvice((ADV.Row(len(a.requests), "prefill", 0, 2),)),),
                                     a.release + (b,)),
                          link.ingress_end + (sum(link.received) + 1,),
                          link.received[:-1] + (link.received[-1] + 1,))
    assert AD.digest(other) != AD.digest(link)
    assert [AD.view(other, w, T, R, SYNCS) == AD.view(link, w, T, R, SYNCS) for w in range(WINDOWS)] == \
        [True] * (WINDOWS - 1) + [False]
    finished = replace(a.requests[0], finish="length" if a.requests[0].finish == "eos" else "eos")
    renamed = AD.LinkAdvice(replace(a, requests=(finished,) + a.requests[1:]), link.ingress_end, link.received)
    assert AD.digest(renamed) != AD.digest(link)
    run = Run(tmp_path)
    outs = [run.round(w, session.egress[w], session.ingress[w], other if w == 2 else link) for w in range(WINDOWS)]
    assert [o.accepted for o in outs] == [True] * WINDOWS
    verdict = run.lw.network_verdict(run.stream)
    assert not verdict["accepted"] and verdict["refused"] == [2] and verdict["advice"] == AD.digest(link)


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
            if q.output_len + d >= 1 and not (d < 0 and any(s.request == j for st in a.steps for r in st.rows
                                                              for s in r.sources)):
                assert not all(check(session, _with_output_len(session.link, j, d)))
                tried += 1
    assert tried > 20
    session = serve(SEEDS[0])
    j = next(j for j, q in enumerate(session.link.advice.requests) if q.output_len + d >= 1)
    assert not all(Run(tmp_path).link(session, _with_output_len(session.link, j, d)))


@pytest.mark.parametrize("ingress", [False, True])
def test_a_missing_request_is_refused(tmp_path, ingress):
    """The advice without one request (none of whose KV another's hit reads), in every session: its frames are missing
    from the advice's releases, and, with its ingress frames taken off too, the windows' `received` aren't the
    ingress's; the last request on the ingress dropped alone leaves the ends short of the windows' total, which the
    advice refuses."""
    tried = 0
    for seed in SEEDS:
        session = serve(seed)
        a = session.link.advice
        for j in range(len(a.requests)):
            if any(s.request == j for st in a.steps for r in st.rows for s in r.sources):
                continue
            try:
                off = _without_request(session.link, j, ingress)
            except ADV.MalformedAdvice:
                assert not ingress and session.link.ingress_end[j] == max(session.link.ingress_end)
                continue
            assert not all(check(session, off))
            tried += 1
    assert tried > 20
    session = serve(SEEDS[1])
    a = session.link.advice
    j = next(j for j in range(len(a.requests)) if session.link.ingress_end[j] < max(session.link.ingress_end)
             and not any(s.request == j for st in a.steps for r in st.rows for s in r.sources))
    assert not all(Run(tmp_path).link(session, _without_request(session.link, j, ingress)))


def test_a_moved_release_bucket_is_refused(tmp_path):
    """Each kept output's release bucket moved one later or earlier, as an advice v0 accepts (its request's releases
    still non-decreasing): one bucket releases a frame fewer and its neighbour one more than the certified egress, and
    a window is refused."""
    session = serve(SEEDS[2])
    moved = []
    for k, d in itertools.product(range(len(session.link.advice.release)), (-1, 1)):
        try:
            off = _moved(session.link, k, d)
        except ADV.MalformedAdvice:
            continue
        assert not all(check(session, off))
        moved.append((k, d))
    assert {d for _, d in moved} == {-1, 1}
    assert not all(Run(tmp_path).link(session, _moved(session.link, *moved[0])))


def test_a_response_released_before_its_request_is_refused(tmp_path):
    """The same advice and egress, with the first request's ingress, and every request's before it, arriving when its
    response's first frame is released: the counts are the advice's, but the ingress head hasn't released the request
    by that frame's eligibility, so the floor refuses the window."""
    session = serve(SEEDS[0])
    a, j = session.link.advice, session.ordinal.index(0)
    first = min(b for o, b in zip(ADV.outputs(a), a.release) if o.request == j)
    sizes = [e - p for p, e in zip([0] + sorted(session.link.ingress_end), sorted(session.link.ingress_end))]
    late, _ = ingress_records([max(t, first) for t in session.arrive], sizes, 0)
    link = AD.LinkAdvice(a, session.link.ingress_end, tuple(sum(row.count(b"q") for row in r.grid) for r in late))
    assert counts(late) != counts(session.ingress)
    assert [list(AD.view(link, w, T, R, SYNCS).released) for w in range(WINDOWS)] == counts(session.egress)
    w = (first - SYNCS[first // T].delta) // T
    assert not check(session, link, ingress=late)[w]
    assert not Run(tmp_path).link(session, link, ingress=late)[w]


def test_a_windows_ingress_total_off_is_refused(tmp_path):
    """`received` with one request frame moved to the next window: the totals still match the requests', but each of
    the two windows' ingress counts sums to one off its `received`."""
    session = serve(SEEDS[0])
    got = list(session.link.received)
    w = next(k for k in range(WINDOWS - 1) if got[k])
    got[w] -= 1
    got[w + 1] += 1
    link = AD.LinkAdvice(session.link.advice, session.link.ingress_end, tuple(got))
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


def test_no_advice_or_a_malformed_one_is_refused(tmp_path):
    """An advised window with no advice, or I1's advice without N2's ingress part, is refused; an ingress part that
    isn't one can't be built; and every window refuses an advice that releases more than R frames in a bucket (all in
    the last), a frame before its request could be (the first in bucket 0), or one in a window with no declared sync
    (the last past the link's)."""
    session = serve(SEEDS[0])
    run = Run(tmp_path)
    opening = run.lw.commit(0, run.issuer.certify(session.egress[0], Direction.EGRESS, P,
                                                  parameters_digest(P, SYNC_SET)),
                            run.issuer.certify(session.ingress[0], Direction.INGRESS, PIN, parameters_digest(PIN)),
                            SALTS[0])
    link, a = session.link, session.link.advice
    assert run.lw.advised_check(opening, SYNCS, link)
    assert not run.lw.advised_check(opening, SYNCS, None) and not run.lw.advised_check(opening, SYNCS, a)
    ends, got = link.ingress_end, link.received
    for advice, end, received in ((ADV.to_json(a), ends, got), (a, ends[:-1], got), (a, (ends[1],) + ends[1:], got),
                                  (a, (0,) + ends[1:], got), (a, ends, got[:-1] + (got[-1] + 1,)),
                                  (a, ends, (-1,) + got[1:])):
        with pytest.raises(ADV.MalformedAdvice):
            AD.LinkAdvice(advice, end, received)
    over, early, late = (WINDOWS * T - 1,) * len(a.release), (0,) + a.release[1:], a.release[:-1] + (WINDOWS * T,)
    for release, why in ((over, "past the allocation"), (early, "before its request"), (late, "declares no sync")):
        bad = AD.LinkAdvice(replace(a, release=release), ends, got)
        for w in range(WINDOWS):
            with pytest.raises(ValueError, match=why):
                AD.view(bad, w, T, R, SYNCS)
        assert not run.lw.advised_check(opening, SYNCS, bad)
    with pytest.raises(ValueError, match="receives in"):
        AD.view(link, WINDOWS, T, R, SYNCS)


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
                cases.append((counts(session.ingress)[w], counts(session.egress)[w],
                              AD.view(session.link, w, T, R, SYNCS)))
    want = [int(AD.reference(*key, i, e, v.released, v.floor, v.received)) for i, e, v in cases]
    assert AD.evaluate(key, cases) == want and 0 < sum(want) < len(want)


def test_the_advised_unit_is_n2s_beside_the_advised_counts(tmp_path):
    """The advised unit holds N2's checks and the advised counts as one gate, the spec declares the advice and the
    advised sizes public, and an advised link has no egress occupancy profile or served profile, and one T and τ_b on
    both ends."""
    lw = Run(tmp_path).lw
    assert lw.unit.id == "NetworkAdvisedLinkWindow_v1{EGRESS=(9,8,0),INGRESS=(9,8,0),ADVISED=(8,2,0)}"
    assert lw.spec.public == PUBLIC_ADVISED and {"advice", "advised", "sync"} <= {p.item for p in lw.spec.public}
    assert lw.spec.check.public_outputs == OUTPUTS_ADVISED == ("verdict", "deadline", "advice")
    assert "served" not in {p.item for p in lw.spec.public} and lw.windows == WINDOWS
    assert lw.spec.digest() != replace(lw, advised=AD.AdvisedProfile(WINDOWS + 1)).spec.digest()
    assert replace(lw, advised=AD.AdvisedProfile(WINDOWS, 1)).unit.id.endswith("ADVISED=(8,2,1)}")
    assert lw.gates()[-1] is AD.gate(T, RIN, 0)
    with pytest.raises(ValueError, match="served .* or advised"):
        replace(lw, served=SV.ServedProfile(3, 1, WINDOWS))
    with pytest.raises(ValueError, match="no egress occupancy"):
        replace(lw, occupancy=(((0,) * T,), None))
    with pytest.raises(ValueError, match="share T"):
        replace(lw, ingress=GridParams(4, 2, 1))
    with pytest.raises(ValueError, match="0 or 1 own"):
        AD.AdvisedProfile(WINDOWS, 2)


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
