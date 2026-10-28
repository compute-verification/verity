"""Sources: the parties that give a party its clock readings and coins, on the same machine, each on a channel nothing
else touches. A party asks and its source answers:

~~~text
clock   ClockRead -> ClockReading {ns};  ClockAlarm {at} (a party's `Timer`) -> ClockTimer {at, ns} once the clock reads at
coins   CoinsDraw {n} -> Coins, n bytes in its body
~~~

Every source message is `private`: a party records the coins it publishes itself, once they are public. `VirtualClock`
reads a `VirtualTime` that the runtime advances, when nothing else can happen, to the next alarm or frame in flight;
each reading may step it (`step`, the cost of a reading). `RealClock` reads the machine's monotonic clock, and the runtime
waits for its alarms. `SeededCoins` is a SHA-512 counter stream of its seed; `OsCoins` the OS's randomness. A source's
end of its channel is `owner`.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable

from verity.core.wire import Body, Int, Message, Registry, Schema

from .party import Action, Identity, Params, Received, Send

CLOCK = Registry([Schema("ClockRead", 1, "private"), Schema("ClockReading", 1, "private", {"ns": Int()}),
                  Schema("ClockAlarm", 1, "private", {"at": Int()}),
                  Schema("ClockTimer", 1, "private", {"at": Int(), "ns": Int()})])
COINS = Registry([Schema("CoinsDraw", 1, "private", {"n": Int(1, 1 << 16)}), Schema("Coins", 1, "private",
                                                                                    body=Body(max=1 << 16))])
ASKS = {"clock": CLOCK.only(["ClockRead", "ClockAlarm"]), "coins": COINS.only(["CoinsDraw"])}
"""What a party may send each source."""
ANSWERS = {"clock": CLOCK.only(["ClockReading", "ClockTimer"]), "coins": COINS.only(["Coins"])}
"""What each source sends its party."""
OWNER = "owner"
SEEDED_TAG = b"verity/deploy/seeded-coins/v1\0"


class VirtualTime:
    """A run's simulated time, in ns: it moves only when the runtime advances it, or a virtual clock's reading steps it."""

    def __init__(self, now: int = 0) -> None:
        self.now = now

    def advance(self, to: int) -> None:
        self.now = max(self.now, to)


class _Clock:
    virtual = False

    def __init__(self) -> None:
        self.alarms: list[int] = []

    def read(self) -> int:
        raise NotImplementedError

    def start(self, params: Params, identity: Identity) -> list[Action]:
        return []

    def on(self, msg: Received) -> list[Action]:
        m = msg.message
        if m.type == "ClockRead":
            return [Send(OWNER, Message("ClockReading", {"ns": self.read()}))]
        self.alarms.append(m.fields["at"])
        return []

    def alarm(self) -> int | None:
        """The earliest alarm set, if any."""
        return min(self.alarms, default=None)

    def fire(self, now: int) -> list[Action]:
        """`ClockTimer` for each alarm at or before `now`, in order."""
        due = sorted(a for a in self.alarms if a <= now)
        self.alarms = [a for a in self.alarms if a > now]
        return [Send(OWNER, Message("ClockTimer", {"at": a, "ns": now})) for a in due]


class VirtualClock(_Clock):
    """A clock on `time`: each reading returns it, then steps it by `step` ns."""

    virtual = True

    def __init__(self, time: VirtualTime | None = None, step: int = 0) -> None:
        super().__init__()
        self.time, self.step = time or VirtualTime(), step

    def read(self) -> int:
        now = self.time.now
        self.time.now += self.step
        return now

    def now(self) -> int:
        return self.time.now


class RealClock(_Clock):
    """The machine's monotonic clock, in ns (`read`, `time.monotonic_ns` unless given)."""

    def __init__(self, read: Callable[[], int] = time.monotonic_ns) -> None:
        super().__init__()
        self._read = read

    def read(self) -> int:
        return self._read()

    def now(self) -> int:
        return self._read()


class _Coins:
    def start(self, params: Params, identity: Identity) -> list[Action]:
        return []

    def on(self, msg: Received) -> list[Action]:
        return [Send(OWNER, Message("Coins", body=self.draw(msg.message.fields["n"])))]

    def draw(self, n: int) -> bytes:
        raise NotImplementedError


class SeededCoins(_Coins):
    """SHA-512(SEEDED_TAG ‖ seed ‖ u64 i) for i = 0, 1, …, read in order: the same seed draws the same coins."""

    def __init__(self, seed: bytes) -> None:
        self.seed, self._i, self._buf = bytes(seed), 0, b""

    def draw(self, n: int) -> bytes:
        while len(self._buf) < n:
            self._buf += hashlib.sha512(SEEDED_TAG + self.seed + self._i.to_bytes(8, "big")).digest()
            self._i += 1
        out, self._buf = self._buf[:n], self._buf[n:]
        return out


class OsCoins(_Coins):
    def draw(self, n: int) -> bytes:
        return os.urandom(n)


def seeded(seed: bytes, *labels: str) -> SeededCoins:
    """A seeded source of its own for each `labels` under one run's `seed`."""
    return SeededCoins(hashlib.sha512(SEEDED_TAG + seed + "/".join(labels).encode()).digest())


__all__ = ["ANSWERS", "ASKS", "CLOCK", "COINS", "OWNER", "SEEDED_TAG", "OsCoins", "RealClock", "SeededCoins",
           "VirtualClock", "VirtualTime", "seeded"]
