"""The stepping loop of the timed rounds (`challenger_session.run`): a party over its protocol's own bytes, its sources
answered once the step that asked is done and in the order it asked, its frames unstamped, what it recorded kept, and
what the stream doesn't carry refused."""

from __future__ import annotations

import pytest

from verity.party import Finish, Identity, LogEntry, Send, Timer
from verity.channel import Message
from verity.protocols.accounting.pous.device.challenger_session import run


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
    """Pings, reads its clock and flips three coins in one step; records the reading, pings the coins' length, waits
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
            return [LogEntry({"ns": m.fields["ns"]}, "private")]
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
    assert records == [LogEntry({"ns": 100}, "private")] and clock.waited == [5_000] and done == Finish(5_100)


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
