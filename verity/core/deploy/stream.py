"""The byte-stream adapter: a party over a stream whose bytes are its protocol's own rather than the wire's frames, and
the side that carries such a stream to a process or machine of its own.

`run(party, params, identity, f, codec, peer=...)` drives one party (`party.Party`) against its one peer over the stream
`f`. Each message it sends `peer` is written by `codec.write`, and each that reaches it is read by `codec.read`. Its
sources are answered here once the step that asked is done, in the order it asked: `ClockRead` with a reading of
`clock`, `CoinsDraw` with `coins(n)`, and a `Timer` with `ClockTimer` once `wait(clock, at)` returns. Nothing is framed,
signed or kept on the way, so the stream carries the codec's bytes alone and no handshake. A frame reaches the party
unstamped (`Received.at` is None), so a party that times a frame asks its clock, and reads nothing it didn't ask for. An
exception out of the party, the codec or the stream is the caller's: nothing is retried, and nothing becomes a refusal.
`run` returns the party's `Finish` and what it recorded, in order.

`Side` is a party's side as another process runs it, on this machine or another: `host`, the machine's name; `command`,
the argv that starts it there, which reads its request as one JSON line on stdin, prints `{"listening": PORT}`, serves
one stream and prints its report as its last line; `address`, where the stream connects, None for this machine at the
port the side prints. `Side.of` reads which machine from a node's `period.json`: its `challenger_host` (the node itself
by default), and `challenger.<key>`'s `command` and `address` (`"HOST:PORT"`), which infra writes once that host serves
the side. `talk(request, drive)` starts the side, hands `drive` the connected socket, and returns the side's report
with what `drive` returned.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO, Protocol, TypeVar

from verity.core.wire import Message

from .party import Finish, Identity, Params, Party, Received, Record, Send, Timer

T = TypeVar("T")


class Codec(Protocol):
    def write(self, f: BinaryIO, message: Message) -> None:
        """`message` on the stream, as its protocol's bytes, flushed."""

    def read(self, f: BinaryIO) -> Message:
        """The next message off the stream; ConnectionError when it ends first."""


def sleep_until(clock: Callable[[], int], at: int) -> None:
    """Return once `clock` reads `at` or later, sleeping on the machine's clock meanwhile."""
    while (now := clock()) < at:
        time.sleep((at - now) / 1e9)


def run(party: Party, params: Params, identity: Identity, f: BinaryIO, codec: Codec, *, peer: str,
        clock: Callable[[], int] | None = None, coins: Callable[[int], bytes] | None = None,
        wait: Callable[[Callable[[], int], int], None] = sleep_until) -> tuple[Finish, list[Record]]:
    """`party` against `peer` over `f` until it finishes (module docstring): its finish, and what it recorded."""
    records: list[Record] = []
    asked: deque[tuple[str, int]] = deque()

    def act(actions: list[Any]) -> Finish | None:
        for a in actions:
            if isinstance(a, Finish):
                return a
            if isinstance(a, Record):
                records.append(a)
            elif isinstance(a, Timer):
                asked.append(("ClockAlarm", a.at))
            elif isinstance(a, Send) and a.channel == peer:
                codec.write(f, a.message)
            elif isinstance(a, Send) and a.channel == "clock" and a.message.type in ("ClockRead", "ClockAlarm"):
                asked.append((a.message.type, a.message.fields.get("at", 0)))
            elif isinstance(a, Send) and a.channel == "coins" and a.message.type == "CoinsDraw":
                asked.append(("CoinsDraw", a.message.fields["n"]))
            else:
                raise ValueError(f"{identity.party} took {a!r}, which a stream to {peer} doesn't carry")
        return None

    def answer(ask: str, arg: int) -> Received:
        if ask == "CoinsDraw":
            if coins is None:
                raise ValueError(f"{identity.party} draws coins, and its stream has none")
            return Received("coins", Message("Coins", body=coins(arg)))
        if clock is None:
            raise ValueError(f"{identity.party} reads its clock, and its stream has none")
        if ask == "ClockRead":
            return Received("clock", Message("ClockReading", {"ns": clock()}))
        wait(clock, arg)
        return Received("clock", Message("ClockTimer", {"at": arg, "ns": clock()}))

    done = act(party.start(params, identity))
    while done is None:
        r = answer(*asked.popleft()) if asked else Received(peer, codec.read(f))
        done = act(party.on(r))
    return done, records


@dataclass(frozen=True)
class Side:
    """A party's side as another process runs it (module docstring)."""

    host: str
    command: tuple[str, ...]
    address: tuple[str, int] | None = None

    @classmethod
    def of(cls, period: Mapping[str, Any], node: str, key: str, local: Sequence[str], what: str | None = None) -> Side:
        """`period.json`'s side `key` for `node`: `local`, the argv that starts it on this machine, when the challenger
        host is the node, else `challenger.<key>`'s `command` and `address`; refused unless infra has wired it (`what`
        names the side, `key` by default)."""
        host = period.get("challenger_host") or node
        if host == node:
            return cls(node, tuple(local))
        entry = (period.get("challenger") or {}).get(key) or {}
        if not entry.get("command") or not entry.get("address"):
            raise RuntimeError(f"the challenger host {host} isn't wired for {what or key}: period.json's "
                               f"challenger.{key}.command and .address start its side there and reach it from {node} "
                               "(infra's)")
        h, _, p = entry["address"].rpartition(":")
        return cls(host, tuple(entry["command"]), (h, int(p)))

    def talk(self, request: Mapping[str, Any], drive: Callable[[socket.socket], T]) -> tuple[dict[str, Any], T]:
        """One session with this side: `request` sent, `drive` handed the stream once it connects, and the side's
        report; refused (RuntimeError) when the side exits before it listens or exits nonzero."""
        p = subprocess.Popen(list(self.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            p.stdin.write(json.dumps(dict(request), sort_keys=True) + "\n")
            p.stdin.close()
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"the side on {self.host} exited {p.wait()} before it listened")
            port = json.loads(line)["listening"]
            with socket.create_connection(self.address or ("127.0.0.1", port)) as s:
                got = drive(s)
            report = json.loads(p.stdout.readline())
        finally:
            rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"the side on {self.host} exited {rc}")
        return report, got


__all__ = ["Codec", "Side", "run", "sleep_until"]
