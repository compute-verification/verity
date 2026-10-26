"""Simulated vLLM serving on an advised link, written as I1's advice v0, and the advice edits the tests refuse: the
advised tests' traffic (`test_network_certifier_advised.py`), and `benchmarks/network_traces/n2_advised_zk.py`'s at
T = 512. It imports no test framework, so a node's bare venv runs it."""

from __future__ import annotations

import hashlib
import itertools
import random
from collections import Counter, deque
from dataclasses import dataclass, replace
from typing import Sequence

from verity.core.protocols.certifier import advice as ADV, advised as AD
from verity.core.protocols.certifier.grid import Certifier, GridParams, occupancy
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet

LINK = "responses"
T, WINDOWS = 8, 4
P = GridParams(buckets=T, allocation=3, frame_bytes=1)
PIN = GridParams(buckets=T, allocation=2, frame_bytes=1)
R, RIN = P.allocation, PIN.allocation
SYNC_SET = SyncSet((1, 2), (2, 4))
SYNCS = [ClockSync(1, 2), ClockSync(2, 4), ClockSync(2, 2), ClockSync(1, 4)]
#: the link's framing: a request of n prompt tokens takes ⌈n/3⌉ ingress frames, for prompts of 1 … 9 tokens
FRAMING = tuple(-(-n // 3) for n in range(1, 10))
BUDGET, SEQS = 6, 4


@dataclass(frozen=True)
class Shape:
    """A simulated link's sizes: T buckets a window, its windows, each end's grid, and the buckets requests arrive in."""

    T: int
    windows: int
    egress: GridParams
    ingress: GridParams
    arrive: Sequence[int]


SHAPE = Shape(T, WINDOWS, P, PIN, range(2 * T))


@dataclass(frozen=True)
class Session:
    """A simulated session on the link: its advice, each end's records, each request's ingress arrival bucket (in
    arrival order), each advice request's place in that order, each kept output's hand bucket (in `advice.outputs`
    order), the frames handed in each bucket, each request's end on the ingress link (in arrival order), and the
    certified ingress's prompt lengths (in arrival order), which the tie (`StandInTie`) holds the advice to."""

    link: AD.LinkAdvice
    egress: list
    ingress: list
    arrive: list[int]
    ordinal: list[int]
    hand: list[int]
    handed: list[int]
    ends: tuple[int, ...]
    certified: tuple[int, ...] = ()


def ingress_records(arrive, frames_in, own, shape=SHAPE):
    """The ingress certifier's records, with its own frame queued at the head of each window's queue, ahead of any
    carried request frames (as `active.Bucketer` queues it), and each request's anchor, the bucket that releases its
    last frame."""
    cert, records, anchor, got = Certifier(LINK, shape.ingress), [], [None] * len(arrive), 0
    need = list(itertools.accumulate(frames_in))
    for t in range(shape.windows * shape.T):
        if own and t % shape.T == 0:
            cert.enqueue(b"s", first=True)
        for j, a in enumerate(arrive):
            if a == t:
                for _ in range(frames_in[j]):
                    cert.enqueue(b"q")
        got += cert.tick().count(b"q")
        anchor = [t if a is None and need[j] <= got else a for j, a in enumerate(anchor)]
        if (t + 1) % shape.T == 0:
            records.append(cert.close_window())
    return records, anchor


def egress_records(handed, shape=SHAPE):
    cert, records = Certifier(LINK, shape.egress), []
    for t in range(shape.windows * shape.T):
        for _ in range(handed[t]):
            cert.enqueue(b"e")
        cert.tick()
        if (t + 1) % shape.T == 0:
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


def serve(seed: int, own: int = 0, shape: Shape = SHAPE) -> Session:
    """A vLLM-like session: 5 … 8 requests of 2 … 9 prompt tokens, each in the framing's 1 … 3 ingress frames, arriving
    in `shape.arrive` (by default over the first two windows), and a scheduler of two steps a bucket, a token budget of
    6 and 4 sequences, which admits a request (FCFS) the bucket after its last ingress frame is released, prefills it in
    chunks (from a prefix-cache hit on another request's KV, where one is wanted and computed), decodes it, preempts
    one running request and recomputes it on resume (from a hit on its own and others' KV), and runs the async row past
    some finishes. A step's rows are the running requests' in Model Runner V2's order (n = 1 first), then those it
    first schedules, in arrival order, so v0's first-row order is the arrival order (the canonical map). Each step's
    kept outputs are handed to the egress certifier the bucket after the step runs, one frame each, and released FIFO
    at R, so frames released after they're handed carry across windows; `release` is the certified egress record's."""
    rng = random.Random(seed)
    n = rng.randint(5, 8)
    arrive = sorted(rng.choice(shape.arrive) for _ in range(n))
    prompt = [rng.randint(2, 9) for _ in range(n)]
    frames_in = [FRAMING[p - 1] for p in prompt]
    out = [rng.randint(1, 10) for _ in range(n)]
    want = [rng.randrange(p) if rng.random() < 0.5 else 0 for p in prompt]
    trailing = [rng.random() < 0.4 for _ in range(n)]
    finish = [rng.choice(ADV.FINISHES) for _ in range(n)]
    ingress, anchor = ingress_records(arrive, frames_in, own, shape)
    last = shape.windows * shape.T

    c, sampled, made, admit = [0] * n, [0] * n, [set() for _ in range(n)], [None] * n
    admitted, waiting, running, done = set(), deque(), [], set()
    preempt_at, preempted_once = rng.randrange(6, 20), False
    steps, hand, scheduled = [], [], set()
    for s in itertools.count():
        if len(done) == n:
            break
        b = s // 2
        assert b + 1 < last, "the session outlasts the link's windows"
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
        fresh = {r[0] for r in rows} - scheduled
        rows.sort(key=lambda r: (True, r[0]) if r[0] in fresh else (False, r[3] != 1, r[3]))
        scheduled |= fresh
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
    handed = [0] * last
    for h in hand:
        handed[h] += 1
    egress = egress_records(handed, shape)
    release = tuple(t for t, k in enumerate(c for row in counts(egress) for c in row) for _ in range(k))
    advice = ADV.Advice(
        tuple(ADV.RequestAdvice(f"u{j}", prompt[j], out[j], finish[j]) for j in first),
        tuple(ADV.StepAdvice(tuple(ADV.Row(new[j], kind, start, k, hit, resumed,
                                           tuple(ADV.Source(new[i], m) for i, m in sources))
                                   for j, kind, start, k, hit, resumed, sources in rows)) for rows in steps),
        release)
    need = list(itertools.accumulate(frames_in))
    received = tuple(sum(row.count(b"q") for row in r.grid) for r in ingress)
    link = AD.LinkAdvice(advice, received)
    return Session(link, egress, ingress, arrive, first, hand, handed, tuple(need), tuple(prompt))


def prompts(link) -> tuple[int, ...]:
    """The advice's prompt lengths, in its requests' order."""
    return tuple(q.prompt_len for q in link.advice.requests)


@dataclass(frozen=True)
class StandInTie:
    """A stand-in of the verifier's own tie source (`advised.TieSource`, `service.LinkWindows(ties=…)`), as `standin`
    stands in for N2's proofs: IoTieRequest's verdict by its reference in the clear on the prompt lengths. Its record
    of the link's ingress is `windows` registrations (stand-in digests of `link` and the window), and its verdict names
    `link` and them, and accepts every ingress window when the advice receives in `windows` windows and its prompt
    lengths are `certified`, the certified ingress's requests' in arrival order (None: an ingress that certified the
    advice's own), else refuses every one. Nothing the prover delivers reaches it, and the package never does."""

    certified: tuple[int, ...] | None = None
    link: str = LINK
    windows: int = WINDOWS

    def registrations(self, link: str) -> tuple[str, ...]:
        return tuple(hashlib.sha512(f"stand-in ingress registration {self.link}/{w}".encode()).hexdigest()
                     for w in range(self.windows))

    def verdict(self, link: str, advice, profile=None) -> AD.IoTieVerdict:
        ok = (self.certified is None or prompts(advice) == tuple(self.certified)) and \
            len(advice.received) == self.windows
        return AD.IoTieVerdict(self.link, self.registrations(link), AD.digest(advice), (ok,) * self.windows)


@dataclass(frozen=True)
class StandInClocks:
    """A stand-in of the verifier's own clock source (`advised.ClockSource`, `service.LinkWindows(clocks=…)`): the
    syncs the certifier ran the link's windows under, `ran`, by default the declared `SYNCS`, as an honest certifier
    runs them. Nothing the prover delivers reaches it."""

    ran: tuple[ClockSync, ...] = tuple(SYNCS)

    def syncs(self, link: str) -> tuple[ClockSync, ...]:
        return self.ran


def held(link, certified=None, windows=None) -> dict:
    """`advised.view`'s and `advised.ends`'s tie arguments for `link` on the link `LINK`: the stand-in source's verdict
    (`StandInTie(certified)`, by default over `link`'s windows) and the link and registrations the verifier holds."""
    source = StandInTie(certified, LINK, len(link.received) if windows is None else windows)
    return {"tie": source.verdict(LINK, link), "link_name": LINK, "registrations": source.registrations(LINK)}


# --- the advice edits, which leave the certified ingress (`Session.certified`) as it was ------------------------------

def rebuilt(link, requests, steps, extra=(), received=None):
    """`link` with the advice's requests and steps replaced, each kept output released where `link` releases the same
    request's (by id) same output, or where `extra`'s (id, index, bucket) says."""
    a = link.advice
    at = {(a.requests[o.request].id, o.index): b for o, b in zip(ADV.outputs(a), a.release)}
    at |= {(i, k): b for i, k, b in extra}
    probe = ADV.Advice(requests, steps, (0,) * sum(q.output_len for q in requests))
    release = tuple(at[(requests[o.request].id, o.index)] for o in ADV.outputs(probe))
    return AD.LinkAdvice(ADV.Advice(requests, steps, release), link.received if received is None else received)


def room(link, j, shape=SHAPE):
    """The first bucket from request j's last release on that releases fewer than the egress allocation's frames."""
    a, per = link.advice, Counter(link.advice.release)
    last = max((b for o, b in zip(ADV.outputs(a), a.release) if o.request == j), default=0)
    return next(t for t in range(last, shape.windows * shape.T) if per[t] < shape.egress.allocation)


def with_output_len(link, j, d, shape=SHAPE):
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
        extra = ((q.id, q.output_len, room(link, j, shape)),)
    return rebuilt(link, requests, tuple(ADV.StepAdvice(rows) for rows in steps if rows), extra)


def without_request(link, j, ingress):
    """The advice with request j dropped, its rows with it, and, with `ingress`, its ingress frames off the link:
    `received` lowered by them in the last windows that released any. The link's certified prompts still have request
    j's."""
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
    received = list(link.received)
    if ingress:
        size = FRAMING[a.requests[j].prompt_len - 1]
        for w in reversed(range(len(received))):
            take = min(size, received[w])
            received[w] -= take
            size -= take
    return rebuilt(link, requests, tuple(steps), received=tuple(received))


def moved(link, k, d):
    """The advice with kept output k's release bucket d later."""
    release = list(link.advice.release)
    release[k] += d
    return AD.LinkAdvice(replace(link.advice, release=tuple(release)), link.received)


def feeds_a_hit(advice, j) -> bool:
    """Whether some request's prefix-cache hit reads request j's KV."""
    return any(s.request == j for st in advice.steps for r in st.rows for s in r.sources)
