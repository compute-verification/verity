"""The verifier's loop when a coin's exchange with the challenger costs time on the verifier's clock.

`test_session.PassClock` moves only as the prover's passes end, so it cannot hold the verifier in an exchange while the
prover waits on the coin that exchange brings. `ExchangeClock` is a clock both sides spend time on: a pass lasts
`PASS_NS`, a round trip to the challenger (one `open` or `open_many`) lasts the session's `exchange_ns`, and time
moves, to the earliest instant a blocked side wakes at, only once both sides' session threads are blocked and every
frame either sent has been stamped and handed on, so a session runs one schedule. Its `wait` is
`verifier.RealClock.wait`'s: past `until_ns` it returns None without reading the inbox.

Through the certifier's coin link a request sent right behind the last answer waits for the next bucket's tick and its
answer for the tick after, two buckets (`coin_link.CoinLinkParams.bucket_ms`); one that reaches the link just before a
tick costs one. At an allowance below two buckets, a session of one pass a coin whose every exchange costs two is
rejected `few` at group 1 with every root in time: each boundary is due before the exchange before it returns, so the
loop reads no root until the last coin has left. At one bucket an exchange the same session is accepted, and so is a
session of 64 passes a coin at two, each of whose round trips asks for one coin. The clock charges every round trip
the same: the link takes a cell a bucket each way, so a round trip for two coins costs it a bucket more."""
from __future__ import annotations

import os
import queue
import random
import secrets
import socket
import sys
import threading
from collections import deque
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

from verity.core.protocols.certifier.coin_link import CoinLinkParams  # noqa: E402
from verity.core.protocols.remote_bandwidth import coins  # noqa: E402
from verity.core.protocols.remote_bandwidth.session import challengers, verifier  # noqa: E402
from verity.core.protocols.spot import challenger as C  # noqa: E402

import prover  # noqa: E402
from test_session import P512, PROVER, VERIFIER, Counted  # noqa: E402

BUCKET_NS = CoinLinkParams().bucket_ms * 10 ** 6
#: `Verifier.allowance` of a 77.3 ms pass, a 3.4 ms gap, a 1.7 ms round trip and 50 ms of jitter each way
ALLOWANCE_NS = 182_334_386
PASS_NS = 46_262_457
#: `PACE_PASS_MS` of `verity bench memory`, which sets a word's groups a stretch
PACE_PASS_MS = 85.0
#: one pass a coin, 12 groups a stretch, and 64 passes a coin, 11 groups a stretch: 7 stretches and the closing one,
#: 6 openings a group
ONE, SIXTY_FOUR = "1:1000:72:7", "64:60000:66:7"


class Inbox:
    """A side's inbox on the clock's lock, so the clock sees whether the side's session thread waits on it empty."""

    def __init__(self, clock: ExchangeClock, side: int) -> None:
        self.clock, self.side, self.items = clock, side, deque()

    def put(self, item) -> None:
        with self.clock.lock:
            self.items.append(item)
            self.clock.lock.notify_all()

    def get(self):
        return self.clock.wait(self, None)

    def get_nowait(self):
        with self.clock.lock:
            if not self.items:
                raise queue.Empty
            return self.items.popleft()


class Roots(Counted):
    """The verifier's end, keeping each root's arrival as its reader stamps it, whether or not the loop reads it."""

    def __init__(self, sock: socket.socket, clock: ExchangeClock, side: int) -> None:
        super().__init__(sock, clock, side)
        self.roots: dict[tuple[int, int], int] = {}

    def recv(self):
        head, body, stamp = super().recv()
        if head["type"] == "root":
            self.roots[head["t"], head["i"]] = stamp
        return head, body, stamp


class ExchangeClock:
    """Time in nanoseconds that a side's session thread spends in `sleep` (a pass, an exchange) or waiting on its inbox
    (`wait`). It moves only when both threads are blocked, neither has a message to read, and each side's reader has
    handed on every frame the other sent (`Counted`'s counts), and then to the earliest instant either wakes at: what
    a side sends is stamped at the instant it was sent, and nothing a side does takes time but its passes and its
    exchanges. Openings and the Lean's answers come when their threads make them, which no boundary or root reads."""

    def __init__(self) -> None:
        self.t = 0
        self.lock = threading.Condition()
        self.sent, self.asked = [0, 0], [0, 0]
        self.inboxes = (Inbox(self, VERIFIER), Inbox(self, PROVER))
        #: each side's session thread while blocked: (the inbox a message wakes it from, the instant it wakes at)
        self.blocked: list[tuple[Inbox | None, int | None] | None] = [None, None]

    def conns(self) -> tuple[Roots, Counted]:
        a, b = socket.socketpair()
        return Roots(a, self, VERIFIER), Counted(b, self, PROVER)

    def now(self) -> int:
        return self.t

    def sending(self, side: int, kind: str) -> None:
        with self.lock:
            self.sent[side] += 1

    def reading(self, side: int, n: int) -> None:
        with self.lock:
            self.asked[side] = n
            self._move()

    def _settled(self, side: int) -> bool:
        if self.blocked[side] is None or self.asked[side] != self.sent[1 - side]:
            return False
        inbox, _ = self.blocked[side]
        return inbox is None or not inbox.items

    def _move(self) -> bool:
        if not (self._settled(VERIFIER) and self._settled(PROVER)):
            return False
        due = [at for _, at in self.blocked if at is not None]
        if not due or min(due) <= self.t:
            return False
        self.t = min(due)
        self.lock.notify_all()
        return True

    def _block(self, side: int, inbox: Inbox | None, at: int | None, ready) -> None:
        self.blocked[side] = (inbox, at)
        while not ready():
            if not self._move():
                self.lock.wait()
        self.blocked[side] = None

    def sleep(self, side: int, ns: int) -> None:
        with self.lock:
            at = self.t + ns
            self._block(side, None, at, lambda: self.t >= at)

    def wait(self, inbox: Inbox, until_ns: int | None):
        with self.lock:
            past = lambda: until_ns is not None and self.t >= until_ns
            self._block(inbox.side, inbox, until_ns, lambda: past() or bool(inbox.items))
            return None if past() else inbox.items.popleft()

    def idle(self) -> None:
        pass

    def busy(self) -> None:
        pass


def on_clock(side: int) -> property:
    """A side's `inbox` as the clock's, whatever its constructor assigns: its reader thread starts there."""
    return property(lambda self: self.conn.pass_clock.inboxes[side], lambda self, _: None)


class Verifier(verifier.Verifier):
    inbox = on_clock(VERIFIER)


class ProverSide(prover.ProverSide):
    inbox = on_clock(PROVER)


class Timed(prover.CpuBackend):
    def __init__(self, clock: ExchangeClock, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.clock = clock

    def run_pass(self, seed: bytes, plan=None):
        out = super().run_pass(seed, plan)
        self.clock.sleep(PROVER, PASS_NS)
        return out


class Exchanged(C.LiveChallenger):
    """A challenger each of whose round trips, an `open` or an `open_many`, costs the verifier `exchange_ns`; `trips`
    has the coins each asked for."""

    def __init__(self, clock: ExchangeClock, exchange_ns: int, trips: list[int], *args) -> None:
        super().__init__(*args)
        self.clock, self.exchange_ns, self.trips = clock, exchange_ns, trips

    def open(self, label, index, after):
        return self.open_many([(label, index, after)])[0]

    def open_many(self, requests):
        self.trips.append(len(requests))
        self.clock.sleep(VERIFIER, self.exchange_ns)
        return [C.LiveChallenger.open(self, *r) for r in requests]


class Paid(challengers.Local):
    def __init__(self, clock: ExchangeClock, exchange_ns: int) -> None:
        self.clock, self.exchange_ns, self.trips = clock, exchange_ns, []

    def open(self, registration: bytes, k: int, windows: int, appends: int, per_coin: int = 1) -> Exchanged:
        return Exchanged(self.clock, self.exchange_ns, self.trips, coins.SCHEME, k,
                         coins.families(windows, appends, per_coin), secrets.token_bytes)


def session(spec: str, exchange_ns: int, blocks: int = 16) -> tuple[dict, dict, dict, list[int]]:
    """One honest session paced `spec` at `ALLOWANCE_NS` over a store of `blocks` blocks, each round trip to its
    challenger `exchange_ns`: the verifier's result, the prover's, each root's arrival, {(group, pass): ns}, and the
    coins each round trip asked for."""
    clock = ExchangeClock()
    vconn, pconn = clock.conns()
    p = ProverSide(pconn, Timed(clock, blocks, random.Random(7), read_bytes_per_s=1e20), b"test-node",
                   prover.parse_plan("honest"), seed=11, log=lambda m: None)
    paid = Paid(clock, exchange_ns)
    v = Verifier(vconn, source=os.urandom(32), windows=3, window_ns=20, samples=6, delta=2.0 ** -40, jitter_ns=10 ** 9,
                 clock=clock, log=lambda m: None, challenger=paid)
    results = []

    def prove() -> None:
        p.register(codec=P512)
        (mode, fraction), = p.plan
        head, _, _ = p.expect("session")
        results.append(p.session(head, mode, fraction))

    th = threading.Thread(target=prove, daemon=True)
    th.start()
    try:
        v.register()
        r = v.session(0, ALLOWANCE_NS, None, verifier.pace_of(spec, PACE_PASS_MS))
        th.join()
    finally:
        v.stop()
    return r, results[0], vconn.roots, paid.trips


def test_one_pass_a_coin_is_rejected_few_at_group_1_when_each_exchange_outlasts_the_allowance():
    """Each exchange returns after the next boundary's latest time, the group before's deadline, so `wait` returns None
    without reading the inbox and the coins leave back to back, one exchange apart. The prover runs its pass under
    each as it comes and every root arrives a pass after its group's boundary, inside the allowance, but the loop
    reads the first only once the last boundary has left, and group 0's draws, one more exchange later, leave after
    the last deadline: group 1 is closed with no root read."""
    E, W = 2 * BUCKET_NS, 12 * 8
    assert E > ALLOWANCE_NS > BUCKET_NS + PASS_NS
    r, pr, roots, trips = session(ONE, E)
    assert r["pace"]["groups"] == 12 and r["windows"] == W
    assert not r["accepted"] and r["failure"] == {"t": 1, "failure": "few", "at": None}
    assert r["verdict"] == "lean" and r["disagreement"] is None
    seen = r["windows_seen"]
    t0 = seen[0]["released_ns"]
    assert [(w["released_ns"], w["boundary_ns"]) for w in seen] == [(t0 + g * E, t0 + (g + 1) * E) for g in range(W)]
    assert [w["deadline_ns"] for w in seen] == [w["boundary_ns"] + ALLOWANCE_NS for w in seen]
    assert pr["passes"] == [1] * W and pr["discarded"] == 0
    assert roots == {(g, 0): seen[g]["boundary_ns"] + PASS_NS for g in range(W)}
    assert all(roots[g, 0] < seen[g]["deadline_ns"] for g in range(W))
    assert [w["arrivals_ns"] for w in seen] == [[roots[0, 0]]] + [[]] * (W - 1)
    assert seen[0]["draws_sent_ns"] == seen[-1]["boundary_ns"] + E > seen[-1]["deadline_ns"]
    assert trips == [1] * (W + 1)


def test_one_pass_a_coin_is_accepted_when_each_exchange_fits_the_allowance():
    """At one bucket a round trip, coin 1's returns before group 0's deadline, so the loop reads each root and closes
    its group before the next boundary, every group with its pass. A group's openings coin and the challenge two
    groups on go in one round trip (`coins.SessionCoins.ahead`) when that boundary is due as the root is read, which
    on this clock it is not when its group's coin left at that same instant."""
    W = 12 * 8
    r, pr, roots, trips = session(ONE, BUCKET_NS)
    assert r["accepted"] and pr["accepted"] and r["failure"] is None and r["windows"] == W
    assert r["verdict"] == "lean" and r["disagreement"] is None
    seen = r["windows_seen"]
    assert [w["arrivals_ns"] for w in seen] == [[roots[g, 0]] for g in range(W)] and len(roots) == W
    assert pr["passes"] == [1] * W and pr["discarded"] == 0
    assert all(w["slack_ns"] > 0 for w in seen)
    assert set(trips) == {1, 2} and sum(trips) == 2 * W


@pytest.mark.slow
def test_64_passes_a_coin_are_accepted_when_each_exchange_outlasts_the_allowance():
    """A group's boundary leaves at its 63rd root and is due only 63 allowances after its coin, so an exchange of two
    buckets delays the next group's coin and starves nothing: the group's last root arrives a pass after its
    boundary, during that exchange, and is read once it returns. Every group holds its 64 passes, none discarded,
    and every round trip asks for one coin, a group's challenge or its openings coin, so two buckets is the most
    the link takes for one."""
    E, K, W = 2 * BUCKET_NS, 64, 11 * 8
    r, pr, roots, trips = session(SIXTY_FOUR, E)
    assert r["pace"]["groups"] == 11 and r["windows"] == W
    assert r["accepted"] and pr["accepted"] and r["failure"] is None
    assert r["verdict"] == "lean" and r["disagreement"] is None
    seen = r["windows_seen"]
    assert [w["arrivals_ns"] for w in seen] == [[roots[g, i] for i in range(K)] for g in range(W)]
    assert len(roots) == K * W and pr["passes"] == [K] * W and pr["discarded"] == 0
    assert all(w["boundary_ns"] == w["arrivals_ns"][K - 2] for w in seen)
    assert all(w["arrivals_ns"][K - 1] == w["boundary_ns"] + PASS_NS < w["deadline_ns"] for w in seen)
    assert all(b["released_ns"] == a["boundary_ns"] and b["arrivals_ns"][0] == b["released_ns"] + E + PASS_NS
               for a, b in zip(seen, seen[1:]))
    assert trips == [1] * (2 * W)
