"""N2's advised counts (`advised`): the circuit held to its reference, and advised link-windows through the proof
service on simulated vLLM serving (varied prompt, request and output lengths, chunked prefill, a prefix-cache hit, a
preemption and its recompute, trailing rows after a finish, responses spanning windows): honest traffic accepted in
every window, salted or not, and an advice whose output length is one off, which misses a request or moves a release
bucket, a response handed before its request, a window's ingress total off, and extra egress each refused."""

import itertools
import json
import random
from collections import deque
from dataclasses import dataclass, replace

import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.core.protocols.certifier import advised as AD, head_check as HC, served as SV
from verity.core.protocols.certifier.certificate import Direction, Issuer, parameters_digest
from verity.core.protocols.certifier.grid import Certifier, GridParams, occupancy
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.certifier.service import PUBLIC_ADVISED, LinkWindows
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
SEEDS = (1, 2, 3, 5, 8)


# I1's dataclasses until ADV-SPEC's `advice.py` lands, with N2's `ingress_frames` and `received`

@dataclass(frozen=True)
class RequestAdvice:
    id: str
    prompt_len: int
    output_len: int
    finish: str
    ingress_frames: int


@dataclass(frozen=True)
class Row:
    request: int
    kind: str
    start: int
    n: int
    cache_hit: int = 0


@dataclass(frozen=True)
class StepAdvice:
    rows: tuple[Row, ...]
    preempted: tuple[int, ...] = ()


@dataclass(frozen=True)
class Advice:
    requests: tuple[RequestAdvice, ...]
    steps: tuple[StepAdvice, ...]
    release: tuple[int, ...]
    received: tuple[int, ...]


@dataclass(frozen=True)
class Session:
    """A simulated session on the link: the advice, each end's records, each request's ingress arrival bucket and each
    step's released frames."""

    advice: Advice
    egress: list
    ingress: list
    arrive: list[int]
    step_frames: list[int]


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


def serve(seed: int, own: int = 0) -> Session:
    """A vLLM-like session: 5 … 8 requests of 1 … 3 ingress frames arriving over the first two windows, and a scheduler
    of two steps a bucket, a token budget of 6 and 4 sequences, which admits a request the bucket after its last
    ingress frame is released, prefills it in chunks (some from a prefix-cache hit), decodes it, preempts one running
    request and recomputes it on resume (from a fresh cache hit), and schedules a trailing row after some finishes.
    Rows are in Model Runner V2's order (n = 1 first). Each step's released tokens are handed to the egress certifier
    the bucket after the step runs, one frame each, so the egress queue carries across windows."""
    rng = random.Random(seed)
    n = rng.randint(5, 8)
    arrive = sorted(rng.randrange(2 * T) for _ in range(n))
    frames_in = [rng.randint(1, 3) for _ in range(n)]
    prompt = [rng.randint(2, 9) for _ in range(n)]
    out = [rng.randint(1, 10) for _ in range(n)]
    hit = [rng.randrange(p) if rng.random() < 0.3 else 0 for p in prompt]
    trailing = [rng.random() < 0.4 for _ in range(n)]
    ingress, anchor = ingress_records(arrive, frames_in, own)

    c, sampled = [0] * n, [0] * n
    admitted, waiting, running, done = set(), deque(), [], set()
    preempt_at, preempted_once = rng.randrange(6, 20), False
    steps, release, step_frames = [], [], []
    handed = [0] * (WINDOWS * T)
    for s in itertools.count():
        if len(done) == n:
            break
        b = s // 2
        assert b + 1 < WINDOWS * T, "the session outlasts the link's windows"
        for j in range(n):
            if j not in admitted and anchor[j] < b:
                admitted.add(j)
                waiting.append(j)
                c[j] = hit[j]
        while waiting and len(running) < SEQS:
            running.append(waiting.popleft())
        preempted = ()
        victims = [j for j in running if 0 < sampled[j] < out[j]]
        if s >= preempt_at and not preempted_once and victims:
            j = victims[0]
            running.remove(j)
            waiting.appendleft(j)
            c[j] = rng.randrange(prompt[j] + sampled[j])
            preempted, preempted_once = (j,), True
        budget, rows = BUDGET, []
        for j in running:
            k = min(prompt[j] + sampled[j] - c[j], budget)
            if k == 0:
                break
            rows.append(Row(j, "decode" if k == 1 and sampled[j] else "prefill", c[j], k, hit[j]))
            c[j] += k
            budget -= k
        if not rows:
            continue
        rows.sort(key=lambda r: (r.n != 1, r.n))
        frames = 0
        for row in rows:
            j = row.request
            if row.start + row.n == prompt[j] + sampled[j]:
                frames += sampled[j] < out[j]
                sampled[j] += 1
                if sampled[j] > out[j] or (sampled[j] == out[j] and not trailing[j]):
                    running.remove(j)
                    done.add(j)
        steps.append(StepAdvice(tuple(rows), preempted))
        release.append(b + 1)
        step_frames.append(frames)
        handed[b + 1] += frames
    received = tuple(sum(row.count(b"q") for row in r.grid) for r in ingress)
    requests = tuple(RequestAdvice(f"u{j}", prompt[j], out[j], "length", frames_in[j]) for j in range(n))
    return Session(Advice(requests, tuple(steps), tuple(release), received), egress_records(handed), ingress, arrive,
                   step_frames)


def counts(records):
    return [list(occupancy(r.grid)) for r in records]


def check(session: Session, advice=None, own=0, egress=None, ingress=None) -> list[bool]:
    """`advised.reference` on every window's heads' counts and its view of the advice, False where `view` refuses."""
    advice = session.advice if advice is None else advice
    out = []
    for w, (e, i) in enumerate(zip(counts(egress or session.egress), counts(ingress or session.ingress))):
        try:
            v = AD.view(advice, w, T, R, SYNCS)
        except ValueError:
            out.append(False)
            continue
        out.append(AD.reference(T, R, RIN, own, i, e, v.handed, v.floor, v.queued, v.received))
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

    def round(self, w, egress, ingress, advice) -> S.Outcome:
        opening = self.lw.commit(w, self.issuer.certify(egress, Direction.EGRESS, P, parameters_digest(P, SYNC_SET)),
                                 self.issuer.certify(ingress, Direction.INGRESS, PIN, parameters_digest(PIN)),
                                 SALTS[w])
        reg = self.lw.register(self.stream, self.lw.registration(opening, self.previous))
        self.previous = reg.digest
        draw = self.stream.select(reg)
        assert self.stream.deliver(draw, [self.lw.session(opening, draw)]) is None
        return self.lw.outcome(self.stream, draw, opening, SYNCS, advice=advice)

    def link(self, session: Session, advice=None, egress=None, ingress=None) -> list[bool]:
        """Every window's outcome of record, in order, and the network verdict's agreement with their AND."""
        advice = session.advice if advice is None else advice
        egress, ingress = egress or session.egress, ingress or session.ingress
        accepted = [self.round(w, egress[w], ingress[w], advice).accepted for w in range(WINDOWS)]
        verdict = self.lw.network_verdict(self.stream)
        assert verdict["accepted"] is all(accepted) and verdict["windows"] == WINDOWS
        assert verdict["refused"] == [w for w, a in enumerate(accepted) if not a]
        return accepted


def _trailing(advice) -> int:
    """The rows that sample past their request's `output_len`."""
    sampled, out = [0] * len(advice.requests), 0
    for st in advice.steps:
        for r in st.rows:
            if r.start + r.n == advice.requests[r.request].prompt_len + sampled[r.request]:
                out += sampled[r.request] >= advice.requests[r.request].output_len
                sampled[r.request] += 1
    return out


def test_the_simulated_sessions_are_varied():
    """The sessions the tests run hold what makes traffic arbitrary: requests of different sizes and output lengths,
    chunked prefill, a prefix-cache hit, a preemption, trailing rows, egress queued across a window's start and a
    response whose frames are handed in two windows."""
    sessions = [serve(seed) for seed in SEEDS]
    for s in sessions:
        assert sum(s.step_frames) == sum(r.output_len for r in s.advice.requests) == len(AD.frames(s.advice))
    requests = [r for s in sessions for r in s.advice.requests]
    assert len({r.output_len for r in requests}) > 3 and len({r.ingress_frames for r in requests}) == 3
    rows = [(s, r) for s in sessions for st in s.advice.steps for r in st.rows]
    assert any(r.kind == "prefill" and r.start + r.n < s.advice.requests[r.request].prompt_len for s, r in rows)
    assert any(r.cache_hit for _, r in rows)
    assert any(st.preempted for s in sessions for st in s.advice.steps)
    assert any(_trailing(s.advice) for s in sessions)
    assert any(AD.view(s.advice, w, T, R, SYNCS).queued for s in sessions for w in range(1, WINDOWS))
    spans = [len({h // T for f, h in zip(AD.frames(s.advice), AD.hands(s.advice)) if f.request == j})
             for s in sessions for j in range(len(s.advice.requests))]
    assert max(spans) > 1


@pytest.mark.parametrize("own", [0, 1])
def test_honest_advised_traffic_is_accepted_in_every_window(tmp_path, own):
    """No egress profile and no FIXLEN: each window's egress counts are the release of the advice's frames from the
    queue the advice carries in, its ingress counts release every hand's request in time, and every window of every
    session is accepted through the proof service, salted (the certifier's own frame in each window) or not."""
    for seed in SEEDS:
        session = serve(seed, own)
        assert check(session, own=own) == [True] * WINDOWS
        (tmp_path / str(seed)).mkdir()
        assert Run(tmp_path / str(seed), own).link(session) == [True] * WINDOWS


def test_a_salted_session_under_an_unsalted_profile_is_refused(tmp_path):
    """The own frame is counted only where the profile says so: a salted session's ingress under OWN = 0 (or an
    unsalted one under OWN = 1) is off by one in every window that receives."""
    salted, plain = serve(SEEDS[0], 1), serve(SEEDS[0], 0)
    assert not all(check(salted, own=0)) and not all(check(plain, own=1))
    assert Run(tmp_path, 0).link(salted) != [True] * WINDOWS


def _with_output_len(advice, j, d):
    reqs = list(advice.requests)
    reqs[j] = replace(reqs[j], output_len=reqs[j].output_len + d)
    return replace(advice, requests=tuple(reqs))


def _without_request(advice, j, adjust):
    """The advice with request j dropped, its rows with it, and, with `adjust`, its ingress frames off `received` in
    the windows that released them, so its frames are simply missing."""
    keep = [k for k in range(len(advice.requests)) if k != j]
    index = {k: i for i, k in enumerate(keep)}
    steps, release = [], []
    for st, rel in zip(advice.steps, advice.release):
        rows = tuple(replace(r, request=index[r.request]) for r in st.rows if r.request != j)
        if rows:
            steps.append(StepAdvice(rows, tuple(index[k] for k in st.preempted if k != j)))
            release.append(rel)
    received = list(advice.received)
    if adjust:
        left = advice.requests[j].ingress_frames
        for w in reversed(range(len(received))):
            take = min(left, received[w])
            received[w] -= take
            left -= take
    return replace(advice, requests=tuple(advice.requests[k] for k in keep), steps=tuple(steps),
                   release=tuple(release), received=tuple(received))


def _moved(advice, s, d):
    release = list(advice.release)
    release[s] += d
    return replace(advice, release=tuple(release))


@pytest.mark.parametrize("d", [-1, 1])
def test_an_output_length_one_off_is_refused(tmp_path, d):
    """Each request's output length one short (its last frame becomes a trailing row's, handed nowhere) or one long
    (the advice samples too few, or its trailing row's sample becomes a frame): the advice's frames aren't the
    certified egress, and a window is refused."""
    session = serve(SEEDS[0])
    for j in range(len(session.advice.requests)):
        if session.advice.requests[j].output_len + d >= 0:
            assert not all(check(session, _with_output_len(session.advice, j, d)))
    assert not all(Run(tmp_path).link(session, _with_output_len(session.advice, 0, d)))


@pytest.mark.parametrize("adjust", [False, True])
def test_a_missing_request_is_refused(tmp_path, adjust):
    """The advice without one request: its ingress frames are then more than the advice's requests, which is malformed,
    or, with the windows' `received` lowered to match, the request's frames are missing from the advice's egress and
    the floors of the requests behind it are lower than their ingress, and a window is refused."""
    session = serve(SEEDS[1])
    for j in range(len(session.advice.requests)):
        assert not all(check(session, _without_request(session.advice, j, adjust)))
    assert not all(Run(tmp_path).link(session, _without_request(session.advice, 0, adjust)))


def test_a_moved_release_bucket_is_refused(tmp_path):
    """A step's release bucket moved one later or earlier in the advice (non-decreasing still): wherever that changes
    the egress certifier's release, the advice's counts aren't the certified ones and a window is refused."""
    session = serve(SEEDS[2])
    a, moved = session.advice, []
    certified = [c for row in counts(session.egress) for c in row]
    for s, d in itertools.product(range(len(a.steps)), (-1, 1)):
        b = a.release[s] + d
        if b < 0 or (s and b < a.release[s - 1]) or (s + 1 < len(a.steps) and b > a.release[s + 1]) or \
                not session.step_frames[s]:
            continue
        hands = AD.hands(_moved(a, s, d))
        if AD.derive([sum(h == t for h in hands) for t in range(WINDOWS * T)], 0, R) != certified:
            assert not all(check(session, _moved(a, s, d)))
            moved.append((s, d))
    assert {d for _, d in moved} == {-1, 1}
    s, d = moved[0]
    assert not all(Run(tmp_path).link(session, _moved(a, s, d)))


def test_a_response_handed_before_its_request_is_refused(tmp_path):
    """The same advice and egress, with a request's ingress arriving later than its response's first frames are handed:
    the counts are the advice's, but the ingress head hasn't released the request by its frames' eligibility, so the
    floor refuses the window."""
    session = serve(SEEDS[0])
    arrive = list(session.arrive)
    first = min(h for f, h in zip(AD.frames(session.advice), AD.hands(session.advice)) if f.request == 0)
    arrive[0] = first
    late, _ = ingress_records(arrive, [r.ingress_frames for r in session.advice.requests], 0)
    advice = replace(session.advice, received=tuple(sum(row.count(b"q") for row in r.grid) for r in late))
    assert counts(late) != counts(session.ingress)
    w = first // T
    v = AD.view(advice, w, T, R, SYNCS)
    assert AD.derive(v.handed, v.queued, R) == counts(session.egress)[w]
    assert not check(session, advice, ingress=late)[w]
    assert not Run(tmp_path).link(session, advice, ingress=late)[w]


def test_a_windows_ingress_total_off_is_refused(tmp_path):
    """`received` with one request frame moved to the next window: the totals still match the requests', but each of
    the two windows' ingress counts sums to one off its `received`."""
    session = serve(SEEDS[0])
    got = list(session.advice.received)
    w = next(k for k in range(WINDOWS - 1) if got[k])
    got[w] -= 1
    got[w + 1] += 1
    advice = replace(session.advice, received=tuple(got))
    assert check(session, advice)[w:w + 2] == [False, False]
    assert Run(tmp_path).link(session, advice)[w:w + 2] == [False, False]


@pytest.mark.parametrize("where", ["busy", "idle"])
def test_extra_egress_is_refused(tmp_path, where):
    """One frame more on the egress link than the advice hands, in a bucket with frames or in an idle one: the
    certificate verifies and the audit accepts, but the counts aren't the release of the advice's frames."""
    session = serve(SEEDS[3])
    hands = AD.hands(session.advice)
    handed = [sum(h == t for h in hands) for t in range(WINDOWS * T)]
    t = next(t for t in range(WINDOWS * T) if (handed[t] > 0) == (where == "busy"))
    handed[t] += 1
    extra = egress_records(handed)
    assert counts(extra) != counts(session.egress)
    assert not all(check(session, egress=extra))
    assert not all(Run(tmp_path).link(session, egress=extra))


def test_no_advice_or_a_malformed_one_is_refused(tmp_path):
    """An advised window with no advice is refused, and so is every window of an advice that samples past a row's known
    tokens, names no request, or releases steps out of order."""
    session = serve(SEEDS[0])
    run = Run(tmp_path)
    opening = run.lw.commit(0, run.issuer.certify(session.egress[0], Direction.EGRESS, P,
                                                  parameters_digest(P, SYNC_SET)),
                            run.issuer.certify(session.ingress[0], Direction.INGRESS, PIN, parameters_digest(PIN)),
                            SALTS[0])
    assert run.lw.advised_check(opening, SYNCS, session.advice)
    assert not run.lw.advised_check(opening, SYNCS, None)
    a = session.advice
    row = a.steps[0].rows[0]
    far = replace(a, steps=(StepAdvice((replace(row, start=a.requests[row.request].prompt_len),)),) + a.steps[1:])
    stray = replace(a, steps=(StepAdvice((replace(row, request=len(a.requests)),)),) + a.steps[1:])
    order = replace(a, release=(a.release[1] + 1,) + a.release[1:])
    for bad in (far, stray, order):
        with pytest.raises(ValueError):
            AD.view(bad, 0, T, R, SYNCS)
        assert not run.lw.advised_check(opening, SYNCS, bad)
    assert check(session, order) == [False] * WINDOWS


def test_releases_reproduce_the_certified_counts():
    """The advice can take each step's release from the certified egress record (`releases`, the bucket that releases
    the step's first frame): when each step's frames were handed together, the advice's counts are the record's."""
    for seed in SEEDS:
        session = serve(seed)
        flat = [c for row in counts(session.egress) for c in row]
        rel = AD.releases(session.step_frames, flat)
        assert all(r >= h for r, h, n in zip(rel, session.advice.release, session.step_frames) if n)
        assert check(session, replace(session.advice, release=tuple(rel))) == [True] * WINDOWS


KEYS = [(8, 3, 2, 0), (8, 3, 2, 1), (5, 1, 1, 1), (1, 2, 3, 0), (1, 2, 3, 1)]


def _cases(key, rng, n):
    T_, R_, RIN_, OWN = key
    cases = []
    for _ in range(n):
        handed = [rng.choice((0, 0, 1, 2, 5, R_)) for _ in range(T_)]
        queued = rng.choice((0, 0, 1, R_, 7, AD.U32_MAX))
        egress = AD.derive(handed, queued, R_)
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
            egress[k] = max(0, egress[k] + rng.choice((-1, 1, 1 << 20)))
        elif roll < 0.25:
            floor[rng.randrange(T_)] += rng.choice((1, 1 << 31))
        elif roll < 0.35:
            received += rng.choice((-1, 1))
        elif roll < 0.4:
            ingress[rng.randrange(T_)] = RIN_ + rng.choice((1, 1 << 24))
        elif roll < 0.45:
            handed[rng.randrange(T_)] = AD.U32_MAX
        cases.append((ingress, egress, AD.WindowView(tuple(handed), tuple(floor), queued, max(0, received))))
    return cases


@pytest.mark.parametrize("key", KEYS)
def test_the_circuit_is_its_reference(key):
    """Random windows of each size class, salted and not, with derived counts and counts one off or past a u32's
    range, floors met and missed, queues from empty to u32's top, ingress past R_in and bucket 0 without its own frame,
    and every honest session's windows: the circuit's bit is `advised.reference`'s."""
    rng = random.Random(sum(key))
    cases = _cases(key, rng, 300)
    if key[:3] == (T, R, RIN):
        for seed in SEEDS:
            session = serve(seed, key[3])
            for w in range(WINDOWS):
                cases.append((counts(session.ingress)[w], counts(session.egress)[w],
                              AD.view(session.advice, w, T, R, SYNCS)))
    want = [int(AD.reference(*key, i, e, *(v.handed, v.floor, v.queued, v.received))) for i, e, v in cases]
    assert AD.evaluate(key, cases) == want and 0 < sum(want) < len(want)


def test_the_advised_unit_is_n2s_beside_the_advised_counts(tmp_path):
    """The advised unit holds N2's checks and the advised counts as one gate, the spec declares the advice and the
    advised sizes public, and an advised link has no egress occupancy profile or served profile, and one T and τ_b on
    both ends."""
    lw = Run(tmp_path).lw
    assert lw.unit.id == "NetworkAdvisedLinkWindow_v1{EGRESS=(9,8,0),INGRESS=(9,8,0),ADVISED=(8,3,2,0)}"
    assert lw.spec.public == PUBLIC_ADVISED and {"advice", "advised", "sync"} <= {p.item for p in lw.spec.public}
    assert "served" not in {p.item for p in lw.spec.public} and lw.windows == WINDOWS
    assert lw.spec.digest() != replace(lw, advised=AD.AdvisedProfile(WINDOWS + 1)).spec.digest()
    assert replace(lw, advised=AD.AdvisedProfile(WINDOWS, 1)).unit.id.endswith("ADVISED=(8,3,2,1)}")
    assert lw.gates()[-1] is AD.gate(T, R, RIN, 0)
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


@pytest.mark.parametrize("key", [(8, 3, 2, 0), (8, 3, 2, 1)])
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
