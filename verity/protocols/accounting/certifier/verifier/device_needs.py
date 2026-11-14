"""What the network certifier's device needs of the party runtime (`verity.runtime`, `verity.party`) that the runtime
does not give: seven types, one for each question to the runtime's owner, each the call the device would make.

None is implemented. The runtime's owner gives each, or refuses it and the device keeps doing that part in its own
processes outside the runtime, as it does today: the relay (`warden.Warden`), its pacer (`pacer`), the publisher
process that holds the key, and the auditor host's relay session, which starts the relay on a listener it holds.

1. `HeldSession`: a listener that outlives one call. A caller starts the party's run, leaves it, asks about it and
   collects it. The run listens on a socket it was handed, and says where it listens in a local file.
2. `LinkEnds`: the link's two ends. Many connections from clients outside the manifest come in, in the link's own
   bytes, and one connection a client goes out to the serving endpoint, plus the watermark's.
3. `DeviceClock`: a clock that is the device's own. An alarm that fires within a bound its party states, a write that
   leaves at a time fixed in advance, and a connection's backlog read before a write that would block.
4. `LaterCapability`: a process of the device. A capability that answers later, as a message on a channel of its name,
   the way a source does. The signer is one: it holds the key, so no commit or signature holds a tick.
5. `LinkTranscript`: what a transcript keeps of a link. A digest a connection and a bucket, with the bytes kept as
   witness, so a replay covers the link and the transcript isn't the size of the traffic.
6. `DeviceStart`: who starts the device, and admission. The runtime starts the device's parties on the auditor's
   machine, and admits a caller: one session at a time, its length capped.
7. `Hop`: the challenger's channel through the device. A channel between two other parties whose frames pass the
   device, which re-cuts each into its cells on its grids and counts every one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from verity.channel import Message
from verity.party import Finish, Params


class HeldSession(Protocol):
    """A party's run that outlives the call that opened it: open returns once it listens, and a later call collects it."""

    def open(self, params: Params, *, listen_fd: int | None = None) -> str:
        """Start the run on the listening socket `listen_fd` its caller holds (or one it binds) and return, once it
        listens, the address it listens at. The run goes on after the call returns."""
        ...

    def endpoints(self) -> Mapping[str, str]:
        """Where the run listens (`listen`) and the endpoint it relays to (`upstream`): a local file a caller on the same
        machine reads, never a message and never in a transcript."""
        ...

    def status(self) -> Mapping[str, Any]:
        """Whether the run is listening, finished or refused, without waiting."""
        ...

    def collect(self) -> Finish:
        """The run's end, after its last execution phase, and the listener let go."""
        ...


class LinkEnds(Protocol):
    """The link an instance holds (`link`, its name as a certificate signs it): connections in from outside the
    manifest and out to the serving endpoint, each by a number the runtime gives."""

    link: str

    def accept(self) -> int | None:
        """The next client connection in, or None when none waits. Its far end is no party (principal `anyone`)."""
        ...

    def connect(self, upstream: str) -> int:
        """A connection out to the serving endpoint at `upstream`, in the endpoint's own protocol: one a client, and one
        for the watermark."""
        ...

    def read(self, conn: int, limit: int) -> bytes:
        """At most `limit` bytes `conn` holds now, b"" when it holds none."""
        ...

    def close(self, conn: int) -> None: ...


class DeviceClock(Protocol):
    """The device's own clock, in ns: its alarms are bounded, and its writes leave at times fixed before they are made."""

    def alarm(self, at: int, within: int) -> None:
        """Wake the party when the clock reads `at`, at most `within` later while frames arrive, or fault the run."""
        ...

    def write_at(self, conn: int, at: int, data: bytes) -> None:
        """Send `data` on `conn` when the clock reads `at`, off the party's turn. When it left, and whether the link took
        it all, comes back as a message (`written {conn, at, left, taken}`)."""
        ...

    def backlog(self, conn: int) -> int:
        """The bytes `conn` holds unwritten now, read before a write that would block."""
        ...


class LaterCapability(Protocol):
    """A capability that answers later: an answer comes as a message on the channel `name`, as a source's does."""

    name: str

    def ask(self, request: Message) -> int:
        """Hand `request` to the device's process and return its ticket at once. The answer arrives as a `Received` on
        `name` whose head carries the ticket (`ticket`)."""
        ...


class LinkTranscript(Protocol):
    """What a party's transcript keeps of its link."""

    def keep(self, conn: int, bucket: int, cells: bytes) -> bytes:
        """Keep `cells`, `conn`'s bytes in `bucket`, as witness, and their digest in the transcript; returns the digest."""
        ...


class DeviceStart(Protocol):
    """Who starts the device's parties on the auditor's machine, and who may ask it to."""

    def admit(self, caller: bytes, params: Params) -> str | None:
        """None when the caller's key `caller` may start a session of `params` now (one at a time, its length capped),
        else the refusal's name."""
        ...

    def start(self, parties: Sequence[str], params: Params, *, listen_fd: int | None = None) -> HeldSession:
        """Start `parties` of the manifest on this machine, under `params`, on the listener `listen_fd` when given."""
        ...


class Hop(Protocol):
    """A channel between two other parties whose pipe passes through the device (the challenger's to the workers)."""

    def carry(self, sender: str, receiver: str, frame: bytes) -> Sequence[bytes]:
        """`frame`, the runtime's own on `sender -> receiver` (its handshake's included), as the device's cells on its
        grids, each counted in the device's records."""
        ...


__all__ = ["DeviceClock", "DeviceStart", "HeldSession", "Hop", "LaterCapability", "LinkEnds", "LinkTranscript"]
