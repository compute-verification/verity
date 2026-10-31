"""The byte-stream adapter (`stream.run`): a party over its protocol's own bytes, its sources answered once the step that
asked is done and in the order it asked, its frames unstamped, what it recorded kept, and what the stream doesn't carry
refused; and the side (`stream.Side`): where `period.json` puts it, and one session with a side in a process of its
own."""

from __future__ import annotations

import sys

import pytest

from verity.core.deploy.party import Finish, Identity, Record, Send, Timer
from verity.core.deploy.stream import Side, run
from verity.core.wire import Message


class Duplex:
    """A stream whose reads come from `incoming` and whose writes are kept in `out`."""

    def __init__(self, incoming: bytes) -> None:
        self.incoming, self.out = incoming, bytearray()

    def read(self, n: int) -> bytes:
        got, self.incoming = self.incoming[:n], self.incoming[n:]
        return got

    def write(self, b: bytes) -> None:
        self.out += b

    def flush(self) -> None:
        pass


class U64:
    """Each message one `u64be n`: a `Ping` written, a `Pong` read."""

    def write(self, f, m: Message) -> None:
        f.write(m.fields["n"].to_bytes(8, "big"))
        f.flush()

    def read(self, f) -> Message:
        b = f.read(8)
        if len(b) < 8:
            raise ConnectionError("the stream ended")
        return Message("Pong", {"n": int.from_bytes(b, "big")})


class Pinger:
    """Pings, reads its clock and draws three coins in one step; records the reading, pings the coins' length, waits
    until the time the pong names, and finishes with its clock's reading then."""

    def __init__(self) -> None:
        self.seen: list = []

    def start(self, params, identity):
        return [Send("peer", Message("Ping", {"n": params["n"]})), Send("clock", Message("ClockRead")),
                Send("coins", Message("CoinsDraw", {"n": 3}))]

    def on(self, r):
        self.seen.append((r.channel, r.message.type, r.at))
        m = r.message
        if m.type == "ClockReading":
            return [Record({"ns": m.fields["ns"]}, "private")]
        if m.type == "Coins":
            return [Send("peer", Message("Ping", {"n": len(m.body)}))]
        if m.type == "Pong":
            return [Timer(m.fields["n"])]
        return [Finish(m.fields["ns"])]


class Clock:
    """Each reading 100 ns after the last; `wait` moves it to the time waited for; each reading notes how many bytes
    the stream had been written by then."""

    def __init__(self, f: Duplex) -> None:
        self.f, self.now, self.written, self.waited = f, 0, [], []

    def __call__(self) -> int:
        self.now += 100
        self.written.append(len(self.f.out))
        return self.now

    def wait(self, clock, at: int) -> None:
        self.waited.append(at)
        self.now = max(self.now, at)


def test_a_party_over_its_own_bytes() -> None:
    f = Duplex((5_000).to_bytes(8, "big"))
    clock, party = Clock(f), Pinger()
    done, records = run(party, {"n": 7}, Identity("pinger"), f, U64(), peer="peer", clock=clock,
                        coins=lambda n: bytes(range(n)), wait=clock.wait)
    assert bytes(f.out) == (7).to_bytes(8, "big") + (3).to_bytes(8, "big")
    assert party.seen == [("clock", "ClockReading", None), ("coins", "Coins", None), ("peer", "Pong", None),
                          ("clock", "ClockTimer", None)]
    assert clock.written[0] == 8, "the clock is read once the step that asked has sent its ping"
    assert records == [Record({"ns": 100}, "private")] and clock.waited == [5_000] and done == Finish(5_100)


class Asks:
    def __init__(self, *actions) -> None:
        self.actions = list(actions)

    def start(self, params, identity):
        return self.actions

    def on(self, r):
        return [Finish()]


@pytest.mark.parametrize("action, why", [
    (Send("elsewhere", Message("Ping", {"n": 1})), "which a stream to peer doesn't carry"),
    (Send("coins", Message("CoinsDraw", {"n": 1})), "draws coins, and its stream has none"),
    (Send("clock", Message("ClockRead")), "reads its clock, and its stream has none")])
def test_what_the_stream_does_not_carry_is_refused(action, why) -> None:
    with pytest.raises(ValueError, match=why):
        run(Asks(action), {}, Identity("asker"), Duplex(b""), U64(), peer="peer")


def test_a_stream_that_ends_is_the_callers() -> None:
    with pytest.raises(ConnectionError, match="the stream ended"):
        run(Asks(), {}, Identity("asker"), Duplex(b"\0\0"), U64(), peer="peer")


def test_the_side_follows_period_json() -> None:
    assert Side.of({}, "n0", "pous", ["run", "it"]) == Side("n0", ("run", "it"))
    assert Side.of({"challenger_host": "n0"}, "n0", "pous", ["run"]) == Side("n0", ("run",))
    wired = {"challenger_host": "a1", "challenger": {"pous": {"command": ["ssh", "a1", "x"], "address": "192.0.2.7:71"}}}
    assert Side.of(wired, "n0", "pous", ["run"]) == Side("a1", ("ssh", "a1", "x"), ("192.0.2.7", 71))
    with pytest.raises(RuntimeError, match="a1 isn't wired for the wipe: period.json's challenger.wipe.command"):
        Side.of(wired, "n0", "wipe", ["run"], "the wipe")


ECHO = """
import json, socket, sys
request = json.loads(sys.stdin.readline())
with socket.create_server(("127.0.0.1", 0)) as srv:
    print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
    conn, _ = srv.accept()
    with conn, conn.makefile("rwb") as f:
        line = f.readline()
        f.write(line.upper())
        f.flush()
print(json.dumps({"got": line.decode(), "request": request}), flush=True)
sys.exit(request["rc"])
"""


def _shout(s) -> bytes:
    with s.makefile("rwb") as f:
        f.write(b"hello\n")
        f.flush()
        return f.readline()


def test_one_session_with_a_side_of_its_own() -> None:
    side = Side("auditor", (sys.executable, "-c", ECHO))
    assert side.talk({"rc": 0}, _shout) == ({"got": "hello\n", "request": {"rc": 0}}, b"HELLO\n")
    with pytest.raises(RuntimeError, match="the side on auditor exited 4$"):
        side.talk({"rc": 4}, _shout)
    with pytest.raises(RuntimeError, match="the side on auditor exited 3 before it listened"):
        Side("auditor", (sys.executable, "-c", "import sys; sys.stdin.readline(); sys.exit(3)")).talk({}, _shout)
    refuse = "import sys; sys.stdin.readline(); print('{\"refused\": \"a second session\"}'); sys.exit(1)"
    with pytest.raises(RuntimeError, match="the side on auditor refused: a second session$"):
        Side("auditor", (sys.executable, "-c", refuse)).talk({}, _shout)
