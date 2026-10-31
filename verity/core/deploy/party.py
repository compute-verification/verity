"""A party: a next-message machine, sans-IO (`note:memory-accounting/20261026T2305Z-draft-deployment-api-v3` §1).

`Security/Definitions/Claims.lean` states parties this way: `Prover.next` gives the next message from the messages so far
and the prover's tape, `Verifier.next` the next from the prover's messages and its coins. A `Party` is that function as an
object the runtime drives: `start(params, identity)` and then `on(received)` for each message that reaches it, each
returning the actions it takes:

- `Send(channel, message)`: a message to the party at the other end of `channel` (a peer's name, or a source's:
  `clock`, `coins`);
- `Timer(at)`: wake me when my clock reads `at` (its clock source answers with `ClockTimer`);
- `Record(entry, kind)`: an entry of one of the public kinds for the record, or `private`, kept in the transcript only;
- `Finish(outcome, refused, kind)`: the party's conclusion, or its refusal by name; it receives nothing after. A
  refusal's name is public; the outcome is of `kind`.

A peer's frame reaches a party that has a clock stamped with its arrival on that clock (`Received.at`), a reading the
runtime takes for it and its transcript keeps.

Besides its peers' messages and its sources', the runtime tells a party that a channel closed (`Closed`) or that a frame
on it was refused (`Refused {name, detail}`, after which the party is finished refused by that name unless it finishes
first). A party does no I/O and reads no ambient clock or randomness: its coins and clock readings are messages from its sources,
so its transcript (`transcript.PartyLog`) replays it exactly, and the same code runs live, simulated or replayed. What
the runtime gives it in `Identity` is its own device: its name, the configuration's hash, its key and its state directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from verity.core.wire import KINDS, Message
from verity.core.wire.handshake import Key

type Params = Mapping[str, Any]
"""A protocol's public structural parameters, canonical JSON."""


@dataclass(frozen=True)
class Identity:
    """Who a party is and what its machine gives it: its name in the manifest, its instance (`memory-challenger@s1`), the
    deployment configuration's hash, its key, its state directory (persisted across restarts) and the local capabilities
    the manifest says it needs (a signer, a store), by name."""

    party: str
    instance: str = ""
    configuration: bytes = b""
    key: Key | None = None
    state: Path | None = None
    capabilities: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Received:
    """A message that reached a party on `channel`: a peer's name, or a source's. `at` is when the runtime took a peer's
    frame off its channel, in ns on the party's own clock source (virtual time in a simulated run); None for a source's
    message, what the runtime tells it, and every message to a party that has no clock."""

    channel: str
    message: Message
    at: int | None = None


@dataclass(frozen=True)
class Send:
    channel: str
    message: Message


@dataclass(frozen=True)
class Timer:
    """Wake the party when its clock reads `at` (ns): its clock source sends `ClockTimer` then."""

    at: int


@dataclass(frozen=True)
class Record:
    entry: Mapping[str, Any]
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"a record entry's kind is one of {KINDS}, not {self.kind!r}")


@dataclass(frozen=True)
class Finish:
    """The party's end: its `outcome` (of `kind`, `output` unless it is witness), or its refusal by name."""

    outcome: Any = None
    refused: str | None = None
    kind: str = "output"

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"an outcome's kind is one of {KINDS}, not {self.kind!r}")


type Action = Send | Timer | Record | Finish


class Party(Protocol):
    def start(self, params: Params, identity: Identity) -> list[Action]: ...

    def on(self, msg: Received) -> list[Action]: ...


__all__ = ["Action", "Finish", "Identity", "Params", "Party", "Received", "Record", "Send", "Timer"]
