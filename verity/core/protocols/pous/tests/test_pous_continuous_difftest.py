"""Differential tests: `verity.core.protocols.pous`'s continuous audit against the Lean model of it
(`Security/Definitions/Pous/Model/Continuous.lean`), restated as the oracle in `pous_continuous_model.py`.

Each action of a trace goes to the memory challenger and to `ClockOracle` at one reading of an injected clock, and after
every action the two must agree exactly: whether the action was refused and why, the challenge revealed, every closed
round's challenge, index, verdict and delay, the misses, the late count, `pending`, `exceeded` and `accepted`. The
traces are a state-aware fuzzer's (answers at and one nanosecond past each limit, wrong labels, pipelined, stale and
repeated answers, challenges and `expire()` before the hard deadline, dropped answers closed by `expire()` or by the
next challenge, actions after the last round) and a few scripted ones. `test_lean_machine_bridge` then drives the memory
challenger from Lean `ContMachine` scripts and checks its verdicts against `cont_records`, the transcription of Lean's
`contRecords`, and its acceptance against `contAccepts`.

The memory challenger is `ContinuousMemoryChallenger`, each challenge a one-round `memory_challenger.MemoryChallenger`
session signed by a test signer.
"""

from __future__ import annotations

import inspect
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

import pous_continuous_model as model
import pous_test_signer as ts
from pous_continuous_model import (ALL_SENT, CLOSED, HIT, LATE, NONE_OPEN, NOT_REVEALED, REJECT, UNANSWERED, WITHIN,
                                   WRONG, ClockOracle, Record, Refused)
from verity.ml.randomness import derive
from verity.core.protocols.pous import band
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous.codec import encode_segments
from verity.core.protocols.pous.verifier import VerifierKey
from verity.core.protocols.pous.continuous import ContinuousMemoryChallenger, Reply, run_continuous

ROOT = Path(__file__).resolve().parents[5]
MODEL = "Security/Definitions/Pous/Model/Continuous.lean"

PARAMS = band.BandParams(16, 64, 12, 2)
SALT = bytes(range(24))
CALL = PARAMS.call_ns
"""The fastest whole call, 241.5 µs: one of Lean's units (the ideal-permutation model's, not an adversary's floor at the
reference `Π₂`)."""
QUERY = 50_000
"""The bridge's time per query of the adversary's throughput, so that `Q = limit // QUERY`."""
LIMITS = [(500_000, 2_000_000, 0), (500_000, 2_000_000, 40_000), (2 * CALL, 2 * CALL, 0), (CALL - 1, 13 * CALL - 1, 0)]
"""`(Δ, Δ_late, RTT)`: the default, with a round-trip allowance, `Δ_late = Δ` on an exact multiple of a call (no answer
can be late without being rejected), and `D = 0` with the largest `D_late`, 12."""

MESSAGES = {
    UNANSWERED: "the previous challenge is unanswered",
    ALL_SENT: r"all \d+ challenges are answered",
    NONE_OPEN: "no challenge is open",
    CLOSED: r"challenge \d+ is closed",
    NOT_REVEALED: r"challenge \d+ is not revealed",
    WITHIN: "the open challenge is within its hard deadline",
}
"""The memory challenger's error for each reason the oracle refuses."""


@pytest.fixture(autouse=True)
def _one_coin_key(monkeypatch):
    ts.one_coin_key(monkeypatch)


@pytest.fixture(scope="module")
def world():
    prims = band.reference(PARAMS)
    w = random.Random(3).randbytes(PARAMS.w_bytes)
    blocks = [b for seg in encode_segments(PARAMS.codec, PARAMS, prims, w, SALT) for b in seg]
    assert len(set(blocks)) == len(blocks)
    return blocks, VerifierKey.from_segments(PARAMS, SALT, [blocks[:16], blocks[16:]])


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


def _session(j: int) -> bytes:
    return j.to_bytes(32, "big")


def _index(source: bytes, j: int) -> int:
    """Challenge `j`'s index: its session's one coin, drawn from `source` (`memory_challenger.CoinTree.draw`)."""
    return derive(source, mc.COIN_DOMAIN, {"session": _session(j), "blocks": PARAMS.blocks, "k": 1}).uniform(PARAMS.blocks, 0)


def verdict(a) -> str:
    """An `audit.Answer` as a Lean verdict."""
    if a.rejected:
        assert not a.correct and not a.on_time, a
        return REJECT
    if not a.on_time:
        return LATE
    return HIT if a.correct else WRONG


class Subject:
    """The continuous audit's memory challenger, behind the oracle's actions."""

    def __init__(self, world, source, n, deadline, late, rtt, clock, max_misses):
        blocks, vk = world
        signer = ts.TestSigner(source, height=max(1, (n - 1).bit_length()))
        self.v = ContinuousMemoryChallenger(PARAMS, source, [_session(j) for j in range(n)], signer, log=ts.session_log(),
                                            digests=[mc.block_digest(b) for b in blocks], binding=vk.root,
                                            late_deadline_ns=late, max_misses=max_misses, deadline_ns=deadline,
                                            rtt_allowance_ns=rtt, clock=clock, schemes=ts.SCHEMES)
        self.max_misses = max_misses

    def challenge(self):
        return self.v.challenge()

    def respond(self, j, block):
        return self.v.respond(j, block)

    def expire(self):
        return self.v.expire()

    def records(self) -> list[Record]:
        return [Record(a.j, a.index, verdict(a), a.elapsed_ns) for a in self.v.answers]

    def counts(self) -> dict:
        """What this memory challenger reports of the run."""
        r = self.v.result()
        return {"done": self.v.done, "accepted": self.v.accepted, "misses": self.v.misses, "late": r.late,
                "pending": self.v.pending, "exceeded": self.v.exceeded}

    def expected(self, o: ClockOracle) -> dict:
        """The same, from the oracle."""
        return {"done": o.done, "accepted": o.accepts(self.max_misses), "misses": o.misses, "late": o.late,
                "pending": o.pending, "exceeded": o.misses > self.max_misses}


class Harness:
    """A memory challenger and the oracle on one clock, compared after every action; `seen` counts each
    `(label, outcome)`."""

    def __init__(self, world, limits, source, n, max_misses=0):
        self.blocks = world[0]
        deadline, late, rtt = limits
        self.source, self.clock = source, Clock()
        self.subject = Subject(world, source, n, deadline, late, rtt, self.clock, max_misses)
        self.oracle = ClockOracle(n, self.index, self.blocks, deadline + rtt, late + rtt)
        self.trace: list[tuple] = []
        self.seen: Counter = Counter()

    @property
    def now(self) -> int:
        return self.clock.now

    def index(self, j: int) -> int:
        return _index(self.source, j)

    def block(self, j: int, right: bool = True) -> bytes:
        i = self.index(j)
        return self.blocks[i if right else (i + 1) % len(self.blocks)]

    def act(self, label: str, action: str, t: int, *args):
        assert t >= self.now, "the clock runs forward"
        self.clock.now = t
        self.trace.append((label, action, t, *args[:1]))
        expected = getattr(self.oracle, action)(t, *args)
        where = f"{label}: {action}{(t, *args[:1])} after {self.trace[-8:-1]}"
        try:
            got = getattr(self.subject, action)(*args)
        except RuntimeError as e:
            assert isinstance(expected, Refused), f"{where}: the memory challenger refused ({e}), the model did {expected}"
            assert re.search(MESSAGES[expected.reason], str(e)), f"{where}: refused with {e!r}, not {expected.reason}"
            outcome = expected.reason
        else:
            assert not isinstance(expected, Refused), f"{where}: the memory challenger did it, the model refuses: {expected}"
            if action == "challenge":
                assert got == expected, where
            outcome = "revealed" if action == "challenge" else expected.verdict
        assert self.subject.records() == self.oracle.records, where
        assert self.subject.counts() == self.subject.expected(self.oracle), where
        self.seen[label, outcome] += 1
        return expected


# The oracle is the Lean model's.

def _transcriptions():
    return [obj for _, obj in inspect.getmembers(model) if (inspect.getdoc(obj) or "").startswith("Lean `")]


def _words(text: str) -> str:
    return " ".join(text.split())


def test_the_oracle_transcribes_the_lean_model():
    """Each Lean definition the oracle transcribes is, word for word, the one in the Lean file its docstring names. A
    failure means the model changed: re-derive the oracle from it before trusting these tests."""
    assert (ROOT / MODEL).exists(), f"{MODEL} is gone: re-point the oracle's docstrings at the model's new home"
    found = {}
    for obj in _transcriptions():
        head, lean = inspect.getdoc(obj).split("\n", 1)
        path = re.search(r"\(`([^`]+\.lean)`\)", head).group(1)
        found[re.match(r"Lean `([^`]+)`", head).group(1)] = path
        assert _words(lean) in _words((ROOT / path).read_text()), f"{obj.__name__}: {path} no longer says this"
    assert set(found) == {"Prog.run", "Prog.depth", "Prog.work", "ContMachine.round", "contVerdict", "ContState",
                          "ContState.start", "contSend", "contMem", "contProg", "contNext", "contAfter", "ContRecord",
                          "contRecord", "contRecords", "contAccepts"}


# Fuzzed traces.

def _fuzz(h: Harness, rng: random.Random) -> None:
    o = h.oracle
    lim, hard = o.limit, o.late_limit
    marks = {0: "0", lim: "limit", lim + 1: "limit+1", hard: "late_limit", hard + 1: "late_limit+1"}

    def name(e):
        return marks.get(e, "between")

    def other(j):
        return rng.choice([j - 1, j + 1, j + 2] if j else [j + 1, j + 2])

    while not o.done:
        r = rng.random()
        if o.open is None:
            if r < 0.75:
                h.act("challenge", "challenge", h.now + rng.choice([0, 1, rng.randrange(10**6), rng.randrange(10**9)]))
            elif r < 0.9:
                k = other(len(o.records))
                h.act("out-of-turn", "respond", h.now + rng.randrange(10**6), k, h.block(k))
            else:
                h.act("expire-none-open", "expire", h.now + rng.randrange(10**6))
            continue
        j, _, send = o.open
        if r < 0.5:
            t = max(h.now, send + rng.choice([*marks, rng.randrange(2 * hard + 2)]))
            right = rng.random() < 0.85
            h.act(f"{'answer' if right else 'wrong'}@{name(t - send)}", "respond", t, j, h.block(j, right))
        elif r < 0.6:
            k = rng.choice([j + 1, j + 2])
            h.act("pipelined", "respond", max(h.now, send + rng.choice([0, lim])), k, h.block(k))
        elif r < 0.65 and j:
            h.act("stale", "respond", max(h.now, send + rng.choice([0, lim])), j - 1, h.block(j - 1))
        elif r < 0.75:
            e = rng.choice([0, lim, hard])
            h.act(f"early-challenge@{name(e)}", "challenge", max(h.now, send + e))
        elif r < 0.87:
            last = "-last" if j == o.n - 1 else ""
            h.act(f"dropped{last}->challenge", "challenge", max(h.now, send + hard + 1 + rng.choice([0, 10**7])))
        else:
            e = rng.choice([0, lim, hard, hard + 1, hard + 1 + rng.randrange(10**7)])
            h.act(f"expire@{name(e)}" if e <= hard else "dropped->expire", "expire", max(h.now, send + e))
    for _ in range(3):
        action = rng.choice(["challenge", "respond", "expire"])
        k = rng.randrange(o.n + 2)
        args = [k, h.block(k)] if action == "respond" else []
        h.act(f"after-done-{action}", action, h.now + rng.randrange(10**6), *args)


REQUIRED = {
    ("answer@0", HIT), ("answer@limit", HIT), ("answer@limit+1", LATE), ("wrong@0", WRONG), ("wrong@limit+1", LATE),
    ("early-challenge@0", UNANSWERED), ("early-challenge@limit", UNANSWERED), ("challenge", "revealed"),
    ("after-done-challenge", ALL_SENT), ("answer@late_limit", LATE), ("answer@late_limit+1", REJECT),
    ("wrong@late_limit+1", REJECT), ("early-challenge@late_limit", UNANSWERED), ("dropped->challenge", "revealed"),
    ("dropped-last->challenge", ALL_SENT), ("dropped->expire", REJECT), ("expire@0", WITHIN),
    ("expire@late_limit", WITHIN), ("expire-none-open", NONE_OPEN), ("after-done-expire", NONE_OPEN),
    ("pipelined", NOT_REVEALED), ("stale", CLOSED), ("out-of-turn", CLOSED), ("out-of-turn", NOT_REVEALED),
    ("after-done-respond", CLOSED), ("after-done-respond", NOT_REVEALED),
}
"""The `(label, outcome)` pairs a fuzz run over `LIMITS` must reach."""


def test_fuzzed_traces_match_the_model(world):
    """60 seeded traces of 1 to 40 rounds for each of `LIMITS`, every action checked against the oracle, and every
    adversarial case of `REQUIRED` reached at least once."""
    seen = Counter()
    for limits in LIMITS:
        for seed in range(60):
            rng = random.Random(f"continuous/{limits}/{seed}")
            n = rng.randrange(1, 41)
            f0 = rng.choice([0, 1, 2, rng.randrange(n + 1)])
            h = Harness(world, limits, b"fuzz-%d" % seed, n, max_misses=f0)
            _fuzz(h, rng)
            seen += h.seen
    assert REQUIRED <= set(seen), sorted(REQUIRED - set(seen))
    assert ("answer@limit+1", REJECT) not in seen and ("answer@late_limit", REJECT) not in seen


# Scripted adversarial traces.

def _scripted(h: Harness) -> None:
    lim, hard, o = h.oracle.limit, h.oracle.late_limit, h.oracle

    def opened():
        if o.open is None:
            h.act("challenge", "challenge", h.now)
        return o.open[2]

    h.act("challenge", "challenge", 0)
    h.act("early-challenge", "challenge", lim)
    h.act("pipelined", "respond", lim, 1, h.block(1))
    h.act("on-deadline", "respond", lim, 0, h.block(0))
    h.act("repeat", "respond", h.now, 0, h.block(0))
    h.act("challenge", "challenge", h.now + 10**6)
    h.act("past-deadline", "respond", opened() + lim + 1, 1, h.block(1))
    h.act("challenge", "challenge", h.now)
    h.act("on-hard-deadline", "respond", opened() + hard, 2, h.block(2, right=False))
    h.act("challenge", "challenge", h.now)
    h.act("expire-early", "expire", opened() + hard)
    h.act("dropped->expire", "expire", opened() + hard + 1)
    h.act("stale", "respond", h.now, 3, h.block(3))
    h.act("challenge", "challenge", h.now)
    h.act("dropped->challenge", "challenge", opened() + hard + 1)
    h.act("stale", "respond", h.now, 4, h.block(4))
    h.act("past-hard-deadline", "respond", opened() + hard + 1, 5, h.block(5))
    h.act("challenge", "challenge", h.now)
    h.act("dropped-last->challenge", "challenge", opened() + hard + 1)
    h.act("after-done", "expire", h.now)


@pytest.mark.parametrize("limits", LIMITS)
def test_scripted_adversarial_trace(world, limits):
    """Pipelining, answers exactly at and one nanosecond past each limit, a repeated answer, a dropped answer closed by
    `expire()` and by the next challenge, stale answers after each, and the last round dropped: action for action as
    the model says."""
    h = Harness(world, limits, b"scripted", 7, max_misses=4)
    _scripted(h)
    assert h.oracle.done


def _same_index_twice():
    for n in range(1000):
        source = b"same-%d" % n
        if _index(source, 0) == _index(source, 1):
            return source
    raise AssertionError("no source draws the same index twice in a row")


def test_a_stale_answer_for_the_same_block(world):
    """Challenge 0 is dropped and challenge 1 asks for the same label; answer 0 then arrives. Lean's round 1 is
    answered by whatever the responder sends after its send, and the memory challenger refuses an answer naming another
    challenge, as the model does."""
    h = Harness(world, LIMITS[0], _same_index_twice(), 2)
    h.act("challenge", "challenge", 0)
    h.act("dropped->challenge", "challenge", h.oracle.late_limit + 1)
    assert h.act("stale-same-block", "respond", h.now + 1, 0, h.block(0)) == Refused(CLOSED)


# Lean machines, run by the memory challenger.

@dataclass
class Scripted:
    """A Lean `ContMachine` that answers round `j` after `script[j] = (d, w, right)`: `w` queries at its first unit,
    none after, and its answer at unit `d` (never, for `d = None`), the challenged label or a wrong one."""

    script: list
    C: list

    def rest(self, h, t, m):
        return (len(h), t)

    def step(self, h, m, msg):
        idx, units = (msg, 0) if msg is not None else m
        d, w, right = self.script[len(h)]
        if d == units:
            return "inl", self.C[idx if right else (idx + 1) % len(self.C)]
        return "inr", ((idx,) * (w if units == 0 else 0), lambda answers: (idx, units + 1))


def _script(rng, n, D, Q, D_late, Q_late):
    out = []
    for _ in range(n):
        d = rng.choice([0, 1, D, D + 1, rng.randrange(D + 3), D_late, D_late + 1, D_late + 3, None])
        w = 0 if d == 0 else rng.choice([0, Q, Q + 1, rng.randrange(Q + 3), Q_late, Q_late + 1])
        out.append((d, w, rng.random() < 0.8))
    return out


def _lean_bounds(lim, hard):
    """`(D, Q, D_late, Q_late)` for the bridge."""
    return (lim // CALL, lim // QUERY, hard // CALL, hard // QUERY)


def _elapsed(d, w):
    return max(d * CALL, w * QUERY)


@pytest.mark.parametrize("jitter", [False, True], ids=["exact", "jitter"])
def test_lean_machine_bridge(world, jitter):
    """A scripted `ContMachine` per seed, its answers arriving `max(d·call_ns, w·QUERY)` after the memory challenger's
    send (plus a network delay with `jitter`), its arrivals `arr j` units apart. Exact: the memory challenger's verdicts
    are Lean's `contRecords`, its misses and late count Lean's non-hits and late or rejected rounds, and it accepts iff
    `contAccepts f₀`. With jitter the memory challenger is never more lenient: a hit is Lean's hit, a Lean late round is
    late or rejected, a Lean rejected round is rejected, so it accepts only runs Lean accepts. Each send is no earlier
    than Lean's (the model's "only sends the next index later") and no earlier than the last round's close."""
    blocks, _ = world
    rank = {HIT: 0, WRONG: 0, LATE: 1, REJECT: 2}
    for limits in LIMITS:
        deadline, late, rtt = limits
        lim, hard = deadline + rtt, late + rtt
        D, Q, D_late, Q_late = _lean_bounds(lim, hard)
        for seed in range(40):
            rng = random.Random(f"bridge/continuous/{limits}/{jitter}/{seed}")
            n = rng.randrange(1, 30)
            script = _script(rng, n, D, Q, D_late, Q_late)
            gaps = [rng.choice([0, rng.randrange(D_late + 2), rng.randrange(100)]) for _ in range(n)]
            arr = [sum(gaps[:j + 1]) for j in range(n)]
            mach, f = Scripted(script, blocks), (lambda q: q)
            source = b"bridge-%d" % seed
            indices = [_index(source, j) for j in range(n)]
            lean = model.cont_records(mach, D, Q, D_late, Q_late, f, blocks, arr.__getitem__, model.cont_start(),
                                      indices)
            misses = sum(r.verdict != HIT for r in lean)
            f0 = max(0, misses + rng.choice([-1, 0, 1]))
            h = Harness(world, limits, source, n, max_misses=f0)
            earliest = 0
            for j, (d, w, right) in enumerate(script):
                h.act("challenge", "challenge", max(h.now, arr[j] * CALL, earliest))
                send, earliest = h.now, 0
                delay = rng.choice([0, 1, rng.randrange(lim + 1)]) if jitter else 0
                if d is not None and (_elapsed(d, w) <= hard or rng.random() < 0.5):
                    h.act("answer", "respond", send + _elapsed(d, w) + delay, j, h.block(j, right))
                elif j == n - 1 or rng.random() < 0.5:
                    h.act("dropped->expire", "expire", send + hard + 1 + delay)
                else:
                    earliest = send + hard + 1 + delay
            got = [r.verdict for r in h.subject.records()]
            want = [r.verdict for r in lean]
            lean_accepts = model.cont_accepts(h.subject.max_misses, lean)
            lean_late = sum(r.verdict in (LATE, REJECT) for r in lean)
            counts = h.subject.counts()
            where = f"seed {seed}, {limits}, script {script}, arrivals {arr}"
            if not jitter:
                assert got == want, where
                assert counts["accepted"] == lean_accepts, where
                assert (counts["misses"], counts["late"]) == (misses, lean_late), where
            else:
                assert all(g == w or rank[g] > rank[w] for g, w in zip(got, want, strict=True)), where
                assert lean_accepts or not counts["accepted"], where
                assert counts["misses"] >= misses and counts["late"] >= lean_late, where
            closes = [send + r.elapsed for send, r in zip(h.oracle.sends, h.oracle.records)]
            assert all(nxt >= close for nxt, close in zip(h.oracle.sends[1:], closes)), where
            for j in range(n):
                s = model.cont_after(mach, D_late, f, arr.__getitem__, indices[:j])
                lean_send = model.cont_send(arr.__getitem__, s)
                assert h.oracle.sends[j] >= lean_send * CALL, where
                after = model.cont_next(mach, D_late, f, arr.__getitem__, s, indices[j])
                assert s.closed <= lean_send and after.closed <= lean_send + D_late, where


def test_the_driver_matches_lean(world):
    """`run_continuous` over a wire, with arrivals by `wait`, driving scripted machines whose answers come within, past
    and far past the hard deadline: the verdicts are Lean's `contRecords` with the driver's arrivals. A silent machine
    sends nothing, its transport giving up one nanosecond past `closes_at`, and the driver closes that round by the
    clock then, as Lean's at `D_late`."""
    blocks, _ = world
    silent = 0
    for seed in range(40):
        rng = random.Random(f"driver/continuous/{seed}")
        limits = rng.choice(LIMITS)
        n = rng.randrange(1, 30)
        h = Harness(world, limits, b"driver-%d" % seed, n, max_misses=rng.randrange(n + 1))
        lim, hard = h.oracle.limit, h.oracle.late_limit
        D, Q, D_late, Q_late = _lean_bounds(lim, hard)
        script = _script(rng, n, D, Q, D_late, Q_late)
        mach, f = Scripted(script, blocks), (lambda q: q)
        v, clock = h.subject.v, h.clock

        def respond(i):
            d, w, right = script[len(v.answers)]
            clock.now += _elapsed(d, w)
            return blocks[i if right else (i + 1) % len(blocks)]

        arr = [rng.randrange(10**3) for _ in range(n)]
        arr = [sum(arr[:j + 1]) for j in range(n)]
        wire = []

        def send(j, i):
            if script[j][0] is not None:
                wire.append(Reply(j, respond(i)))

        def receive(until):
            if wire:
                return wire.pop()
            clock.now = max(clock.now, until + 1)
            return None

        result = run_continuous(v, send, receive, wait=lambda j: setattr(clock, "now", max(clock.now, arr[j] * CALL)))
        assert all(a.elapsed_ns == hard + 1 for a, (d, _, _) in zip(result.answers, script) if d is None)
        silent += sum(d is None for d, _, _ in script)
        lean = model.cont_records(mach, D, Q, D_late, Q_late, f, blocks, arr.__getitem__, model.cont_start(),
                                  [h.index(j) for j in range(n)])
        assert [verdict(a) for a in result.answers] == [r.verdict for r in lean], (seed, limits, script)
        assert v.accepted == model.cont_accepts(h.subject.max_misses, lean)
    assert silent
