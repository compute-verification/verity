"""The network certifier's coin link (`coin_link`), driven a quarter bucket at a time on a `VirtualClock`: its prover side
an in-memory connection that speaks `remote`'s frames, its challenger an in-memory one over a real `LiveChallenger`'s
coins, which a test can make misbehave. An honest session passes, its coins exempt and nothing else, every other frame
counted by name, and the client's own reader (`remote._Wire`) reads the re-cut stream as the challenger wrote it; a
forged, replayed, out-of-order, unrequested or non-canonical opening, and a frame of another type, are each counted and
never exempt; the challenger's and the prover's timing within a bucket is re-clocked away, so the other side sees the same
cells in the same buckets; and a prover that stops reading or hangs up, a queue past its bound, a frame `remote` refuses
whole, cells queued when their connection is cut and a late tick each fail the link-window closed."""

from __future__ import annotations

import asyncio
import json
import math
import random
import socket
import types
from collections import defaultdict

import pytest

from verity.core.protocols.certifier.active import VirtualClock
from verity.core.protocols.certifier.coin_link import (COIN, COINS, COUNTED, REQUESTS, CoinLink, CoinLinkParams,
                                                       canonical, recut)
from verity.core.protocols.certifier.grid import Reason
from verity.core.protocols.sampled_proofs import remote as R
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveChallenger, LiveOpening, live_reason

SCHEME = "verity/test/coins/v1"
FAMILIES = [("positions", 3), ("replay", 2)]
PARAMS = CoinLinkParams(cell_bytes=512, coin_cells=2, request_cells=1, bucket_ms=100, window=40, queue_cells=64)
B, TAU, STEPS = PARAMS.cell_bytes, PARAMS.bucket_ms / 1000, 4
EOF = None
HELLO, CLOSE = {"type": "hello"}, {"type": "close"}


def live(seed: str = "coins") -> LiveChallenger:
    return LiveChallenger(SCHEME, 0, FAMILIES, random.Random(seed).randbytes)


def root(k: int) -> bytes:
    return bytes([k]) * 64


def ask(label: str, index: int, k: int) -> dict:
    return {"type": "open", "label": label, "index": index, "after": root(k).hex()}


def frame(head: dict | bytes) -> bytes:
    h = head if isinstance(head, bytes) else canonical(head)
    return R.HEADER.pack(len(h), 0) + h


def opening_frame(opening: LiveOpening) -> bytes:
    return frame({"type": "opening", "opening": opening.to_json()})


def wire_heads(data: bytes, n: int | None = None) -> list[dict]:
    """The first `n` heads `remote`'s own reader takes from `data`, or every one, when `data` ends between frames."""
    a, b = socket.socketpair()
    with a, b:
        a.sendall(data)
        a.shutdown(socket.SHUT_WR)
        wire, out = R._Wire(b), []
        while (n is None or len(out) < n) and b.recv(1, socket.MSG_PEEK):
            out.append(wire.recv())
        return out


@types.coroutine
def _turn():
    yield


async def settle(turns: int = 200) -> None:
    """Let every task run until none has anything left to do: everything here is in memory."""
    for _ in range(turns):
        await _turn()


class Prover:
    """The prover's side of one connection: `frames` fed to the link, each at its clock time in `at` (default: at once),
    then its end of stream unless `eof` is False, or a connection reset at `reset_at`; every write the link makes, with
    the tick it was made at. `backlog` is what it leaves untaken, and `gone` a prover that hung up."""

    def __init__(self, link: CoinLink, clock: VirtualClock, frames: list[bytes], *, at: list[float] | None = None,
                 eof: bool = True, reset_at: float | None = None, backlog: int = 0, gone: bool = False) -> None:
        self.link, self.clock, self.backlog, self.gone = link, clock, backlog, gone
        self.reader = asyncio.StreamReader()
        self.writes: list[tuple[int, bytes]] = []
        self.eof_at: int | None = None
        self.closed_at: int | None = None
        self.transport = self
        self.feeder = asyncio.create_task(self._feed(frames, at, eof, reset_at))
        self.task = asyncio.create_task(link.handle(self.reader, self))

    async def _feed(self, frames: list[bytes], at: list[float] | None, eof: bool, reset_at: float | None) -> None:
        for i, f in enumerate(frames):
            if at is not None:
                await self.clock.sleep(at[i] - self.clock())
            self.reader.feed_data(f)
        if reset_at is not None:
            await self.clock.sleep(reset_at - self.clock())
            self.reader.set_exception(ConnectionResetError("reset by peer"))
        elif eof:
            self.reader.feed_eof()

    def write(self, data: bytes) -> None:
        assert self.closed_at is None and self.eof_at is None
        self.writes.append((self.link.ticks, bytes(data)))

    def write_eof(self) -> None:
        self.eof_at = self.link.ticks

    def can_write_eof(self) -> bool:
        return True

    def is_closing(self) -> bool:
        return self.gone or self.closed_at is not None

    def close(self) -> None:
        if self.closed_at is None:
            self.closed_at = self.link.ticks

    def get_write_buffer_size(self) -> int:
        return self.backlog

    def received(self) -> bytes:
        return b"".join(d for _, d in self.writes)


class Challenger:
    """The challenger's side in memory: each connection answers each frame the link writes as `remote.serve` would, with
    `coins`' coins, except that `answers[(n, k)]`, when given, is the raw frames connection n sends for its k-th frame
    instead (EOF ends its stream), and each answer is fed `delay(k)` seconds after the link wrote its frame. `coins.open`
    keeps no order, so this challenger answers a coin out of order where `remote.serve` would refuse it."""

    def __init__(self, coins: LiveChallenger, clock: VirtualClock, *, answers: dict | None = None, delay=None,
                 refuse: bool = False) -> None:
        self.coins, self.clock, self.answers, self.delay, self.refuse = coins, clock, answers or {}, delay, refuse
        self.conns: list[ChallengerConn] = []
        self.link: CoinLink | None = None

    async def connect(self):
        if self.refuse:
            raise ConnectionRefusedError("nothing listens")
        conn = ChallengerConn(self, len(self.conns))
        self.conns.append(conn)
        return conn.reader, conn


class ChallengerConn:
    def __init__(self, ch: Challenger, n: int) -> None:
        self.ch, self.n = ch, n
        self.reader = asyncio.StreamReader()
        self.buffer = bytearray()
        self.heads: list[dict] = []
        self.writes: list[tuple[int, bytes]] = []
        self.ended = self.fed_eof = self.got_eof = self.closed = False
        self.last: asyncio.Task | None = None
        self.transport = self

    def write(self, data: bytes) -> None:
        assert not self.closed and not self.got_eof
        self.writes.append((self.ch.link.ticks, bytes(data)))
        self.buffer += data
        while len(self.buffer) >= R.HEADER.size:
            nh, nb = R.HEADER.unpack_from(self.buffer)
            if len(self.buffer) < R.HEADER.size + nh + nb:
                break
            raw = bytes(self.buffer[R.HEADER.size:R.HEADER.size + nh])
            del self.buffer[:R.HEADER.size + nh + nb]
            k = len(self.heads)
            self.heads.append(json.loads(raw))
            out = self.ch.answers[(self.n, k)] if (self.n, k) in self.ch.answers else self.honest(k, self.heads[-1])
            if self.ch.delay is None:
                self.feed(out)
            else:
                self.last = asyncio.create_task(self.later(self.last, self.ch.clock() + self.ch.delay(k), out))

    def honest(self, k: int, head: dict) -> list:
        if self.ended:
            return []
        if k == 0:
            if head != HELLO:
                return self.refuse("a client's first frame is hello")
            return [frame({"type": "commitment", "commitment": self.ch.coins.commitment.to_json()})]
        if head == CLOSE:
            self.ended = True
            return [frame({"type": "closed"}), EOF]
        try:
            return [opening_frame(self.ch.coins.open(head["label"], head["index"], bytes.fromhex(head["after"])))]
        except (CoinError, KeyError, ValueError) as e:
            return self.refuse(str(e))

    def refuse(self, why: str) -> list:
        self.ended = True
        return [frame({"type": "refused", "reason": why}), EOF]

    async def later(self, before: asyncio.Task | None, at: float, out: list) -> None:
        if before is not None:
            await before
        await self.ch.clock.sleep(at - self.ch.clock())
        self.feed(out)

    def feed(self, out: list) -> None:
        for f in out:
            if self.fed_eof:
                return
            if f is EOF:
                self.fed_eof = True
                self.reader.feed_eof()
            else:
                self.reader.feed_data(f)

    def write_eof(self) -> None:
        self.got_eof = True

    def can_write_eof(self) -> bool:
        return True

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def get_write_buffer_size(self) -> int:
        return 0


def session(provers: list[dict], *, params: CoinLinkParams = PARAMS, seed: str = "coins", answers=None, delay=None,
            refuse: bool = False, late: dict[int, float] | None = None, stop_at: int | None = None):
    """A link over `seed`'s coins, its challenger as `Challenger` makes it, one prover connection a dict of `Prover`'s
    keywords (each opened once the one before it has), run until every connection has ended (or tick `stop_at`), then
    stopped and run to its window's end; `late[t]` puts tick t that many seconds late. (link, provers, challenger)."""
    coins = live(seed)

    async def main():
        clock = VirtualClock()
        ch = Challenger(coins, clock, answers=answers, delay=delay, refuse=refuse)
        link = CoinLink(params, coins.commitment, ch.connect, origin=0.0, clock=clock, sleep=clock.sleep)
        ch.link = link
        running = asyncio.create_task(link.run())
        opened: list[Prover] = []
        step = 0
        while not running.done():
            await settle()
            if len(opened) < len(provers) and all(p.task.done() for p in opened):
                opened.append(Prover(link, clock, **provers[len(opened)]))
                continue
            if (len(opened) == len(provers) and all(p.task.done() for p in opened)) or \
                    (stop_at is not None and link.ticks >= stop_at) or link.ticks >= 4 * params.window:
                link.stop()
            step += 1
            t, q = divmod(step, STEPS)
            clock.advance(t * params.bucket_ms / 1000 + q * params.bucket_ms / 1000 / STEPS + (late or {}).get(t, 0))
        await running
        await settle()
        return link, opened, ch

    return asyncio.run(main())


def honest_frames(asks: list[tuple[str, int, int]]) -> list[bytes]:
    return [frame(HELLO), *(frame(ask(*a)) for a in asks), frame(CLOSE)]


ASKS = [("positions", 0, 1), ("replay", 0, 2), ("positions", 1, 3)]


def by_tick(writes: list[tuple[int, bytes]]) -> dict[int, int]:
    """Bytes written at each tick."""
    out: dict[int, int] = defaultdict(int)
    for t, data in writes:
        out[t] += len(data)
    return dict(out)


def statuses(link: CoinLink) -> list[tuple[str | None, str | None]]:
    return [(w[COINS], w[REQUESTS]) for w in link.windows]


# --- an honest session -----------------------------------------------------------------------------------------------


def test_an_honest_session_passes_with_its_coins_exempt_and_every_other_frame_counted():
    """hello, three opens of two families and close: the prover's own reader takes the commitment, three openings that
    open against the committed root for what it asked, and closed; the challenger's reader takes the prover's five
    frames as it sent them; the coins' bytes are exactly the three opening frames', the commitment and closed frames
    are counted by name, every write is whole cells, at most r a bucket, and every link-window is complete."""
    link, (prover,), ch = session([{"frames": honest_frames(ASKS)}])
    heads = wire_heads(prover.received())
    assert [h["type"] for h in heads] == ["commitment", "opening", "opening", "opening", "closed"]
    coins = live()
    for h, (label, index, k) in zip(heads[1:4], ASKS):
        assert live_reason(coins.commitment, LiveOpening.from_json(h["opening"]), label, index, root(k)) is None
    (conn,) = ch.conns
    assert wire_heads(b"".join(d for _, d in conn.writes)) == [HELLO, *(ask(*a) for a in ASKS), CLOSE]
    assert conn.got_eof and conn.closed and prover.eof_at is not None and prover.closed_at is not None

    s = link.summary()
    assert s["complete"] and statuses(link) == [(None, None)]
    assert s["coins"] == {"frames": 3, "bytes": sum(len(frame(h)) for h in heads[1:4])}
    assert s["counted"] == {"commitment": {"frames": 1, "bytes": len(frame(heads[0]))},
                            "closed": {"frames": 1, "bytes": len(frame(heads[4]))}}
    assert s["charge_bits"][COINS]["content"] == 8 * (len(frame(heads[0])) + len(frame(heads[4])))
    assert s["requests"] == {"frames": 5, "bytes": sum(len(f) for f in honest_frames(ASKS))}
    assert s["buckets"] == PARAMS.window and s["charge_bits"][COINS]["timing"] == PARAMS.window * math.log2(3)
    assert [(e["cut"], e["errors"]) for e in s["connections"]] == [(False, {})] and s["late"]["ticks"] == 0
    for writes, r, grid in ((prover.writes, PARAMS.coin_cells, COINS), (conn.writes, PARAMS.request_cells, REQUESTS)):
        sizes = by_tick(writes)
        assert all(n % B == 0 and n <= r * B for n in sizes.values()) and len(sizes) == len(writes)
        assert sum(sizes.values()) == s["cells"][grid] * B


def test_recut_is_whole_cells_that_the_reader_takes_as_the_same_head():
    """A head of any length is padded with spaces only, to the fewest whole cells, and `remote`'s reader takes the same
    object from it; a head that would pass `remote.MAX_HEAD` once padded is refused."""
    rng = random.Random(7)
    for n in [0, 1, B - 9, B - 8, B - 7, 3 * B + 5]:
        head = canonical({"type": "x", "pad": "".join(rng.choice("ab") for _ in range(n))})
        cells = recut(head, B)
        data = b"".join(cells)
        assert all(len(c) == B for c in cells) and len(cells) == -(-(len(head) + 8) // B)
        assert data[8 + len(head):] == b" " * (len(data) - 8 - len(head))
        assert wire_heads(data) == [json.loads(head)]
    with pytest.raises(ValueError):
        recut(b"{" + b" " * (R.MAX_HEAD - 1), B)


# --- what is counted (RP5) -------------------------------------------------------------------------------------------


def test_a_forged_opening_is_counted_unverified_and_never_exempt():
    """The challenger answers the last open with that coin's opening, one byte of its coin flipped, in canonical form:
    the prover gets it (it is counted, not dropped), and it is `unverified`, not a coin."""
    coins = live()
    forged = coins.open("positions", 1, root(3)).to_json()
    forged["coin"] = ("0" if forged["coin"][0] != "0" else "1") + forged["coin"][1:]
    link, (prover,), _ = session([{"frames": honest_frames(ASKS)}],
                                 answers={(0, 3): [frame({"type": "opening", "opening": forged})]})
    s = link.summary()
    assert s["coins"]["frames"] == 2 and s["counted"]["unverified"]["frames"] == 1
    assert s["complete"] and wire_heads(prover.received())[3]["opening"] == forged


def test_a_replayed_or_out_of_order_coin_is_counted():
    """A coin asked for twice and answered twice is exempt once (`replayed` the second time); a family's coin 1
    answered before its coin 0 is `out-of-order`, and coin 0 after it is still a coin; a valid opening resent for
    another coin's ask is `unverified`."""
    coins = live()
    first = coins.open("positions", 0, root(1))
    asks = [("positions", 0, 1), ("positions", 0, 1), ("replay", 1, 2), ("replay", 0, 3), ("positions", 1, 4)]
    link, _, _ = session([{"frames": honest_frames(asks)}],
                         answers={(0, 1): [opening_frame(first)], (0, 2): [opening_frame(first)],
                                  (0, 5): [opening_frame(first)]})
    s = link.summary()
    assert s["coins"]["frames"] == 2  # positions 0 once, replay 0 after replay 1
    assert {k: v["frames"] for k, v in s["counted"].items()} == \
        {"commitment": 1, "closed": 1, "replayed": 1, "out-of-order": 1, "unverified": 1}
    assert link.classifier.next == {"positions": 1, "replay": 1}


def test_a_frame_of_another_kind_or_form_is_counted_by_name():
    """A verified opening in another encoding is `non-canonical`; an extra frame (of another type) is `other` and takes
    the place of the next answer, so the opening after it answers the wrong ask and is `unverified`, never in the
    prover's favour; a head that isn't a JSON object with a type is `malformed`; and an opening when nothing is asked is
    `unrequested`."""
    coins = live()
    o = coins.open("positions", 0, root(1))
    head = json.dumps({"type": "opening", "opening": o.to_json()}, sort_keys=True).encode()
    spaced = R.HEADER.pack(len(head), 0) + head
    note = frame({"type": "note", "text": "x" * 300})
    link, (prover,), _ = session(
        [{"frames": honest_frames(ASKS)}],
        answers={(0, 1): [spaced, note], (0, 3): [frame(b"[1, 2]")]})
    counts = {k: v["frames"] for k, v in link.summary()["counted"].items()}
    assert counts == {"commitment": 1, "non-canonical": 1, "other": 1, "unverified": 1, "malformed": 1, "closed": 1}
    assert link.summary()["coins"]["frames"] == 0 and set(counts) <= set(COUNTED)
    assert [h.get("type") for h in wire_heads(prover.received(), 4)] == ["commitment", "opening", "note", "opening"]

    link, _, _ = session([{"frames": [frame(HELLO)], "eof": False}], stop_at=6,
                         answers={(0, 0): [frame({"type": "commitment", "commitment": coins.commitment.to_json()}),
                                           opening_frame(o)]})
    assert {k: v["frames"] for k, v in link.summary()["counted"].items()} == {"commitment": 1, "unrequested": 1}


def test_a_coin_is_exempt_once_across_connections():
    """A refused hello's connection, then an honest one, then one that asks for the same coins again and is answered
    them: served one at a time, in order; the refusal is counted, the second connection's coins are exempt, and the
    third's are `replayed`."""
    coins = live()
    again = [opening_frame(coins.open(*a[:2], root(a[2]))) for a in ASKS]
    link, provers, ch = session(
        [{"frames": [frame({"type": "hello", "secret": "00"})]}, {"frames": honest_frames(ASKS)},
         {"frames": honest_frames(ASKS)}],
        answers={(2, 1): [again[0]], (2, 2): [again[1]], (2, 3): [again[2]]})
    s = link.summary()
    assert [e["ordinal"] for e in s["connections"]] == [1, 2, 3] and len(ch.conns) == 3
    assert all(ch.conns[i].closed for i in range(3))
    assert s["coins"]["frames"] == 3 and s["counted"]["replayed"]["frames"] == 3
    assert s["counted"]["refused"]["frames"] == 1 and s["complete"]
    assert wire_heads(provers[0].received())[0]["type"] == "refused"


# --- re-clocking (RP6, RP8) ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", ["a", "b", "c"])
def test_timing_within_a_bucket_is_re_clocked_away(seed):
    """The same session with the challenger answering up to 0.7 τ after the tick that sent it the ask, and the prover
    sending each frame anywhere in its bucket: the prover's writes (tick and bytes) are the same as with an instant
    challenger and a prover that sends at mid-bucket, and so are the challenger's. (The clock moves a quarter bucket a
    step, so 0.7 τ keeps each event strictly inside its bucket.)"""
    rng = random.Random(seed)
    frames = honest_frames(ASKS)
    plain_link, (plain,), plain_ch = session([{"frames": frames, "at": [(i + 0.5) * TAU for i in range(len(frames))]}])
    jitter = [rng.uniform(0, 0.7 * TAU) for _ in range(len(frames))]
    link, (prover,), ch = session([{"frames": frames, "at": [(i + rng.uniform(0.05, 0.7)) * TAU
                                                             for i in range(len(frames))]}],
                                  delay=lambda k: jitter[k])
    assert prover.writes == plain.writes and ch.conns[0].writes == plain_ch.conns[0].writes
    assert link.summary() == plain_link.summary()


def test_the_link_stops_at_its_window_boundary():
    """Stopped at tick 7 with no connection, the link ticks on to the end of its link-window and no further."""
    link, _, _ = session([], stop_at=7)
    assert link.ticks == PARAMS.window and statuses(link) == [(None, None)] and link.summary()["complete"]


# --- fail closed -----------------------------------------------------------------------------------------------------


def test_a_prover_that_stops_reading_or_hangs_up_fails_the_link_closed():
    """A prover that leaves more than the link's backlog untaken gets nothing more and is cut: `delivery-refused` on the
    coins, its requests still queued `dropped`, and the window is not complete; so is one that hung up."""
    for prover_ in ({"backlog": 1 << 30}, {"gone": True}):
        link, (prover,), _ = session([{"frames": honest_frames(ASKS), **prover_}])
        assert statuses(link) == [(Reason.DELIVERY_REFUSED.value, Reason.DROPPED.value)]
        assert prover.writes == [] and link.log[0]["cut"] and not link.summary()["complete"]


def test_a_frame_remote_refuses_whole_or_a_flood_fails_closed():
    """A prover frame with a body is `malformed` on the requests grid, which cuts the connection, and the coins still
    queued for it are dropped; a challenger that sends more cells than the queue holds overflows."""
    body = R.HEADER.pack(2, 1) + b"{}x"
    link, _, _ = session([{"frames": [frame(HELLO), body], "eof": False}])
    assert statuses(link) == [(None, Reason.MALFORMED.value)] and link.log[0]["cut"]

    coins = live()
    flood = [frame({"type": "commitment", "commitment": coins.commitment.to_json()})] * (PARAMS.queue_cells + 1)
    link, (prover,), _ = session([{"frames": [frame(HELLO)], "eof": False}], answers={(0, 0): flood})
    assert statuses(link) == [(Reason.OVERFLOW.value, None)] and link.log[0]["cut"]

    slow = CoinLinkParams(cell_bytes=512, coin_cells=1, request_cells=1, bucket_ms=100, window=40, queue_cells=64)
    link, _, _ = session([{"frames": [frame(HELLO), frame(ask("positions", 0, 1)), body], "eof": False,
                           "at": [0.05, 0.05, 0.35]}], params=slow,
                         answers={(0, 0): [frame({"type": "note", "text": "y" * 3000})]})
    assert statuses(link) == [(Reason.DROPPED.value, Reason.MALFORMED.value)]
    assert link.log[0]["dropped"][COINS] > 0 and not link.summary()["complete"]


def test_a_late_tick_faults_the_window_and_an_unreachable_challenger_sends_nothing():
    """A tick more than the slack late is `monitor-unavailable` on both grids. A connection whose challenger doesn't
    answer the connect is closed at the next tick with nothing sent either way."""
    link, _, _ = session([{"frames": honest_frames(ASKS)}], late={3: 0.05})
    assert statuses(link) == [(Reason.MONITOR_UNAVAILABLE.value,) * 2] and link.late == 1

    link, (prover,), ch = session([{"frames": honest_frames(ASKS)}], refuse=True)
    assert prover.writes == [] and prover.closed_at is not None and ch.conns == []
    assert link.log[0]["unreachable"].startswith("ConnectionRefusedError") and link.summary()["complete"]


def test_a_connection_reset_ends_its_side_and_its_entry_names_it():
    """A prover whose connection resets once its frames are in ends its side as an end of stream would: the challenger
    still gets every frame and the end, the prover every answer, and the connection's entry names the error."""
    link, (prover,), ch = session([{"frames": honest_frames(ASKS), "reset_at": 0.5}])
    assert [h["type"] for h in wire_heads(prover.received())] == ["commitment", "opening", "opening", "opening", "closed"]
    assert ch.conns[0].got_eof and link.summary()["complete"]
    assert link.log[0]["errors"] == {"prover": "ConnectionResetError"} and not link.log[0]["cut"]


def test_the_parameters_are_d7s_by_default_and_bound_by_their_digest():
    p = CoinLinkParams()
    assert (p.cell_bytes, p.coin_cells, p.bucket_ms, p.window) == (4096, 1, 100, 600)
    assert p.digest != CoinLinkParams(coin_cells=2).digest and len(p.digest) == 64
    with pytest.raises(ValueError):
        CoinLinkParams(cell_bytes=8)
    assert COIN not in COUNTED
