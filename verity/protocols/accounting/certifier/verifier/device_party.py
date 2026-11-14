"""The network certifier's device as a party of the deployment layer (`verity.runtime`): `NetworkCertifier`, the role
`parties.NETWORK_CERTIFIER` of the certifier's manifest (`parties.MANIFEST`), and `PROTOCOL`, that manifest with it.

Its needs are the manifest's: its clock and coins are the runtime's sources, and its signing key the capability
`signing-key` (`certificate.Signer`). Its parameters are the schedule's (`warden.WardenParams`, under ``params``), the
number of execution phases (``windows``) and the first one's opening on its clock in ns (``origin``).

Its life is the methods its turns call:

- `start` opens the session (`open_session`), hands out the run's deployment and load (`hand_out_run`) and sets a
  timer at each execution phase's close.
- `on` takes each timer: the execution phase it closes is certified (`certify`), signed (`sign`) and handed out
  (`hand_out`). After the last, the session's end is collected (`collect`) and the party finishes (`close`), handing
  out the run's groups when it has them.

`hand_out`, `hand_out_run` and `close` are written: they make the manifest's messages out of the documents the device
keeps. `open_session`, `certify`, `sign` and `collect` raise `NotImplementedError` naming the goal
`network/certifier-device-is-a-party`. Each is done today by the device's own processes outside the runtime, whose I/O
the runtime has no form for yet (`device_needs`): the relay (`warden.Warden`), its pacer and its publisher process,
started by the auditor host's relay session.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.channel import Message
from verity.party import Action, Finish, Identity, Params, Received, Send, Timer
from verity.runtime.runner import ProtocolDef

from . import parties as NC
from .warden import WardenParams

GOAL = "network/certifier-device-is-a-party"
SENDS = {NC.WORKERS: (NC.OUT_OPENING, NC.RECORDS, NC.CELLS_SALTS),
         NC.VERIFIER: (NC.OUT_RECORD, NC.CERTIFICATE, NC.REGISTRATION, NC.LOAD, NC.DEPLOYMENT, NC.GROUPS)}
"""What the device sends each peer: its channels' messages in the manifest."""


@dataclass(frozen=True)
class Opened:
    """A session as the device opened it: the run's `deployment.json` and, when a load was signed, its `load.json`."""

    deployment: bytes
    load: bytes | None = None


@dataclass(frozen=True)
class Kept:
    """One of a closed execution phase's three records (`name`, of `parties.RECORD_NAMES`) under the link's name as its
    certificate signs it, packed as the device keeps it."""

    name: str
    link: str
    packed: bytes


@dataclass(frozen=True)
class SignedRecord:
    """A kept record with its line of `certified.jsonl` and, for a record of the wire, its line of `cells.jsonl` and its
    rows' salts and the wiring's (`cells.salts`)."""

    kept: Kept
    certificate: bytes
    registration: bytes | None = None
    cells_salts: bytes | None = None


@dataclass(frozen=True)
class Out:
    """An egress execution phase's output value (`out_commit.CertifiedOut`): its witness, the rows' salts, and its
    public record as canonical JSON."""

    witness: bytes
    public: bytes


@dataclass(frozen=True)
class Phase:
    """A closed execution phase (`window`) on the link `link` as the device signed it."""

    window: int
    link: str
    records: tuple[SignedRecord, ...]
    out: Out | None = None


class NetworkCertifier:
    """The network certifier's device on one link: `start` and `on` are the runtime's calls, and the rest its life."""

    def __init__(self) -> None:
        self.identity: Identity | None = None
        self.schedule: WardenParams | None = None
        self.windows = 0
        self.closes: tuple[int, ...] = ()
        self.next = 0

    def start(self, params: Params, identity: Identity) -> list[Action]:
        self.identity = identity
        self.schedule = WardenParams(**params["params"])
        self.windows = params["windows"]
        span = self.schedule.window * self.schedule.bucket_ms * 1_000_000
        self.closes = tuple(params["origin"] + (w + 1) * span for w in range(self.windows))
        actions: list[Action] = self.hand_out_run(self.open_session(identity))
        return actions + [Timer(at) for at in self.closes]

    def on(self, msg: Received) -> list[Action]:
        if msg.channel != "clock" or msg.message.type != "ClockTimer" or self.next >= self.windows:
            return [Finish(refused="out-of-phase")]
        window, self.next = self.next, self.next + 1
        actions = self.hand_out(self.sign(self.certify(window)))
        if self.next == self.windows:
            actions += self.close(self.collect())
        return actions

    def open_session(self, identity: Identity) -> Opened:
        """Start the relay on its listener and the publisher on the key (`identity.capabilities["signing-key"]`), and
        return the run's deployment and load. Seam: the relay listens on a socket the auditor host's relay session
        holds and binds its own connections, I/O done outside the runtime (`device_needs.HeldSession`)."""
        raise NotImplementedError(f"{GOAL}: open_session, the relay's session on its held listener")

    def certify(self, window: int) -> tuple[Kept, ...]:
        """The three records of execution phase `window` as the relay closed them. Seam: the relay's ticks on its own
        clock and its connections in and out are its own network I/O outside the runtime (`device_needs.LinkEnds`,
        `device_needs.DeviceClock`)."""
        raise NotImplementedError(f"{GOAL}: certify, the relay's records of execution phase {window}")

    def sign(self, kept: Sequence[Kept]) -> Phase:
        """`kept` certified and its wire's rows committed and signed, by the publisher process that alone holds the key.
        Seam: the publisher is a process of the device that answers later (`device_needs.LaterCapability`)."""
        raise NotImplementedError(f"{GOAL}: sign, the publisher's certificates and registrations")

    def collect(self) -> bytes | None:
        """The session's end, collected once its last execution phase is handed out: the run's `groups.json` when it is
        grouped. Seam: the relay session's end is collected by a later call (`device_needs.HeldSession.collect`)."""
        raise NotImplementedError(f"{GOAL}: collect, the relay session's end")

    def hand_out_run(self, opened: Opened) -> list[Action]:
        """The run's deployment, and its load when one was signed, to the verifier."""
        out: list[Action] = [Send(NC.VERIFIER, Message(NC.DEPLOYMENT, body=opened.deployment))]
        if opened.load is not None:
            out.append(Send(NC.VERIFIER, Message(NC.LOAD, body=opened.load)))
        return out

    def hand_out(self, phase: Phase) -> list[Action]:
        """A signed execution phase's messages: each record packed and its rows' salts to the workers, its certificate
        and registration to the verifier, and an egress phase's output value split between them."""
        out: list[Action] = []
        for r in phase.records:
            head = {"link": r.kept.link, "window": phase.window}
            out.append(Send(NC.WORKERS, Message(NC.RECORDS, head | {"record": r.kept.name}, r.kept.packed)))
            out.append(Send(NC.VERIFIER, Message(NC.CERTIFICATE, head, r.certificate)))
            if r.registration is not None:
                out.append(Send(NC.VERIFIER, Message(NC.REGISTRATION, head, r.registration)))
            if r.cells_salts is not None:
                out.append(Send(NC.WORKERS, Message(NC.CELLS_SALTS, head, r.cells_salts)))
        if phase.out is not None:
            head = {"link": phase.link, "window": phase.window}
            out.append(Send(NC.WORKERS, Message(NC.OUT_OPENING, head, phase.out.witness)))
            out.append(Send(NC.VERIFIER, Message(NC.OUT_RECORD, head, phase.out.public)))
        return out

    def close(self, groups: bytes | None) -> list[Action]:
        """The run's groups, when it is grouped, to the verifier, and the party's end: the execution phases handed out."""
        out: list[Action] = []
        if groups is not None:
            out.append(Send(NC.VERIFIER, Message(NC.GROUPS, body=groups)))
        return out + [Finish({"windows": self.next})]


def _network_certifier() -> NetworkCertifier:
    return NetworkCertifier()


PROTOCOL = ProtocolDef(NC.MANIFEST, {NC.NETWORK_CERTIFIER: _network_certifier})

__all__ = ["GOAL", "Kept", "NetworkCertifier", "Opened", "Out", "PROTOCOL", "Phase", "SENDS", "SignedRecord"]
