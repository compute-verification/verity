"""The memory challenger's session as PoUS runs it: the wire both sides speak, the request and the id a session runs
under, and the challenger's side as PoUS is handed it. The challenger's side is the
auditor-trusted device (`challenger_party`, on a host the developer doesn't control: the challenger host, `trusted-device`),
which imports this module and which this module never imports; how it runs a session is its docstring's.

The two sides are parties of the deployment layer (`verity.runtime`), the challenger (`challenger_party.Challenger`)
and the prover (`verity.protocols.accounting.pous.prover.challenger_prover.Prover`), whose messages are one frame each
way a round (`MANIFEST`):

~~~text
memory-challenger -> prover   Warmup {index} ... then per round Index {index} (coins); End
prover -> memory-challenger   per frame Block, the whole block (private)
~~~

`Frames` is their codec over the session's stream, `u64be index` for `Warmup` and `Index`, `u64be 2^64 − 1` for `End`
and the whole block for a `Block`, which `run` steps them on (`challenger_prover.answer` runs the prover there). Each side reads
its connection through `Polled`, which polls for up to `SPIN_NS` before it blocks, so neither host's CPU sleeps through
a round.

The loop. `run(party, params, identity, f, codec, peer=...)` drives one party (`party.Party`) against its one peer over
the stream `f`. Each message it sends `peer` is written by `codec.write`, and each that reaches it is read by
`codec.read`. Its sources are answered here once the step that asked is done, in the order it asked: a read of its clock
with a reading of `clock`, a request for `n` bytes of coins with `coins(n)`, and a `Timer` with `ClockTimer` once
`wait(clock, at)` returns. Nothing is framed, signed or kept on the way, so the stream carries the codec's bytes alone
and no handshake. A frame reaches the party unstamped (`Received.at` is None), so a party that times a frame asks its
clock, and reads nothing it didn't ask for. An exception out of the party, the codec or the stream is the caller's:
nothing is retried, and nothing becomes a refusal. `run` returns the party's `Finish` and what it recorded, in order.

A request that names no audit runs under the id the auditor designates from its coins, the node and the setup's root
(`coin_session_id`, under `period.SESSION_DOMAIN`). One that names its audit (`request_of` with `audit`) runs under
`session_id(audit, commitment, salt)`, of the commitment and salt the challenger logged for the audit, which the
verifier computes from the audit's binding and what setup published, never from the report; it carries setup's published
outputs and P2's parameters (`PARAMS`), which `setup_of` reads back. The challenger keeps each session it completes on
its own host under `SETUPS`, from which PoUB's verifier on the same host builds its registration
(`verity.protocols.accounting.poub.verifier.p2_party --pous-record`).

`Side` is the challenger's side as what starts a run hands it over, wherever `period.json` puts it, and
`command(device)` the argv that starts it on this machine. `salt(side, audit, commitment)` asks it for a salt, and
`challenger_prover.session(side, request, respond)` is one session with the side, each index answered with
`respond(index)`. `placement` says which host ran each party and whether `trusted-device`'s premise holds.
"""

from __future__ import annotations

import hashlib
import socket
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO, Protocol, TypeVar

from verity.commitments.randomness import derive
from verity.protocol import Channel, Manifest, Role
from verity.party import Finish, Identity, LogEntry, Params, Party, Received, Send, Timer
from verity.channel import Body, Int, Message, Schema
from verity.protocols.accounting.pous.verifier import run as R
from verity.protocols.accounting.pous.verifier.schemes.p2 import P2Params

from . import memory_challenger as mc

END = 2 ** 64 - 1
#: the untimed exchanges before round 0 (`challenger_party`'s docstring); the index is public, so they reveal no coin
WARMUPS, WARMUP_INDEX = 8, 0
MODULE = "verity.protocols.accounting.pous.device.challenger_party"
PREMISE = "trusted-device"
ID = "verity/pous/memory-challenger"
VERSION = 1
CHALLENGER, PROVER = "memory-challenger", "prover"
SESSION_ID_TAG = b"verity-pous/memory-challenger/session-id/v1\0"
AUDIT_BYTES = 64
MAX_BLOCK = 1 << 24
MANIFEST = Manifest(
    ID, VERSION, "verity.protocols.accounting.pous.device.challenger_session:request_of",
    parties=(Role(CHALLENGER, "auditor", needs=("clock", "coins", "signing-key")),
             Role(PROVER, "developer", per="server", needs=("store",))),
    channels=(Channel(CHALLENGER, PROVER, ("Warmup", "Index", "End"), carries_coins=True),
              Channel(PROVER, CHALLENGER, ("Block",))),
    messages=(Schema("Warmup", 1, "structure", {"index": Int()}), Schema("Index", 1, "coins", {"index": Int()}),
              Schema("End", 1, "structure"), Schema("Block", 1, "private", body=Body(max=MAX_BLOCK))))
SETUPS = "setups"
PARAMS = ("blocks_per_segment", "c0", "expander", "segments", "w")
SPIN_NS = 2_000_000
"""How long a side's read polls its connection before it blocks (`Polled`). A side that blocks between rounds pays its
host's wake-up, which between the KVM guests of the cross-host path took rounds from about 90 µs to 550–650 µs, most
of the RTT allowance. A round's frame comes well within this budget, and a longer wait (the session's setup) blocks."""


class Polled:
    """A connected socket as a session's stream: `read(n)` polls it without blocking for up to `spin_ns` on `clock` while
    nothing arrives, then blocks until something does, and returns short only when the peer closes. Polling changes no
    reading: the challenger still reads its clock after the whole block is in."""

    def __init__(self, sock: socket.socket, spin_ns: int = SPIN_NS,
                 clock: Callable[[], int] = time.perf_counter_ns) -> None:
        self.sock, self.spin_ns, self.clock = sock, spin_ns, clock

    def read(self, n: int) -> bytes:
        buf = bytearray(n)
        view, got, until = memoryview(buf), 0, None
        while got < n:
            try:
                k = self.sock.recv_into(view[got:], n - got, socket.MSG_DONTWAIT)
            except BlockingIOError:
                now = self.clock()
                until = now + self.spin_ns if until is None else until
                if now < until:
                    continue
                k = self.sock.recv_into(view[got:], n - got)
            if k == 0:
                break
            got, until = got + k, None
        return bytes(view[:got])

    def write(self, b: bytes) -> None:
        self.sock.sendall(b)

    def flush(self) -> None:
        pass


T = TypeVar("T")


class Side(Protocol):
    """The challenger's side as what starts a run hands it to PoUS (the runtime's `Side` is one): the machine it runs
    on; `ask(request)`, one request it answers with no session: the line it printed as JSON, the line and its exit
    code; and `talk(request, drive)`, one session with it: `request` sent, `drive` handed the connected socket, the
    side's report back with what `drive` returned."""

    host: str

    def ask(self, request: Mapping[str, Any]) -> tuple[dict[str, Any], str, int]: ...

    def talk(self, request: Mapping[str, Any], drive: Callable[[socket.socket], T]) -> tuple[dict[str, Any], T]: ...


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
        wait: Callable[[Callable[[], int], int], None] = sleep_until) -> tuple[Finish, list[LogEntry]]:
    """`party` against `peer` over `f` until it finishes (module docstring): its finish, and what it recorded."""
    records: list[LogEntry] = []
    asked: deque[tuple[str, int]] = deque()

    def act(actions: list[Any]) -> Finish | None:
        for a in actions:
            if isinstance(a, Finish):
                return a
            if isinstance(a, LogEntry):
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


def _read(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ConnectionError(f"the peer closed after {len(b)} of {n} bytes")
    return b


class Frames:
    """The parties' messages as the session's bytes (module docstring): `Warmup` and `Index` as `u64be index`, `End` as
    `u64be END`, a `Block` as its bytes. The challenger's end reads a `Block` of `block_bytes`; the prover's end
    (`block_bytes` None) reads an `Index` or the `End`."""

    def __init__(self, block_bytes: int | None = None) -> None:
        self.block_bytes = block_bytes

    def write(self, f: BinaryIO, m: Message) -> None:
        f.write(m.body if m.type == "Block" else (END if m.type == "End" else m.fields["index"]).to_bytes(8, "big"))
        f.flush()

    def read(self, f: BinaryIO) -> Message:
        if self.block_bytes is not None:
            return Message("Block", body=_read(f, self.block_bytes))
        i = int.from_bytes(_read(f, 8), "big")
        return Message("End") if i == END else Message("Index", {"index": i})


def session_id(audit: bytes, commitment: bytes, salt: bytes) -> bytes:
    """The id an audit's session runs under: the first 32 bytes of `SHA-512(SESSION_ID_TAG ‖ audit ‖ commitment ‖ salt)`,
    of the audit's binding, the commitment to `W` and the salt the challenger drew after it for that audit. The salt and
    the commitment are in it because the signed header binds only the root of `C*`: an id of the audit alone would let a
    setup under a salt of the prover's pass as one under the challenger's."""
    if (len(audit), len(commitment), len(salt)) != (AUDIT_BYTES, 64, mc.SALT_BYTES):
        raise ValueError(f"an audit's binding and a commitment to W are 64 bytes, a salt {mc.SALT_BYTES}")
    return hashlib.sha512(SESSION_ID_TAG + audit + commitment + salt).digest()[:32]


def coin_session_id(coins: bytes, domain: str, node: str, root: bytes) -> bytes:
    """The id a session that names no audit runs under: the one designated from the auditor's `coins` under `domain`
    (`period.SESSION_DOMAIN`), the node and the root of the `C*` it audits."""
    return derive(coins, domain, {"node": node, "setup": root}).stream("session", 32)


def command(device: str | Path) -> tuple[str, ...]:
    """The argv that starts the challenger's side as a process on this machine, its key and logs in `device`."""
    return (sys.executable, "-m", MODULE, "--device", str(device), "--listen", "127.0.0.1:0")


def placement(node: str, host: str, *, verifier: str | None = None) -> dict[str, Any]:
    """Which host ran each party, and whether `trusted-device`'s premise holds: that the memory challenger ran on a
    machine the developer doesn't control, which is never the developer's node."""
    held = host != node
    return {"parties": {"memory_challenger": host, "prover": node, "verifier": verifier or node},
            PREMISE: (f"holds: the memory challenger ran on {host}, not the prover's {node}" if held else
                      f"unmet: the memory challenger ran on the prover's node {node}")}


def salt(side: Side, audit: bytes, commitment: bytes) -> bytes:
    """The salt's source for the audit `audit`, which the challenger's side `side` samples after `commitment` and logs
    (`give_salt`); RuntimeError with the side's refusal, or when it answers anything else or exits nonzero."""
    doc, line, rc = side.ask({"salt": {"audit": audit.hex(), "commitment": commitment.hex()}})
    if rc == 0 and isinstance(doc.get("source"), str):
        return bytes.fromhex(doc["source"])
    why = doc.get("refused") or (f"it answered {line.strip()[:200]!r}" if "listening" in doc else f"it exited {rc}")
    raise RuntimeError(f"the memory challenger on {side.host} drew no salt: {why}")


def request_of(node: str, point: R.Point, binding: bytes, *, audit: bytes | None = None,
               published: R.Setup | None = None) -> dict[str, Any]:
    """The session's request, `binding` the root of the `C*` it audits; with `audit`, the request names its audit and
    setup's `published` outputs, whose root is `binding`."""
    q = point.params
    out = {"binding": binding.hex(), "block_bytes": q.block_bytes, "blocks": q.blocks, "k": point.k,
           "limit_ns": point.limit_ns, "node": node}
    if audit is not None:
        out |= {"audit": audit.hex(), "published": published.to_json(), "params": {k: getattr(q, k) for k in PARAMS}}
    return out


def setup_of(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """Setup's published outputs with P2's parameters, as a request naming its audit carries them (`request_of`), None
    for a request without them; ValueError unless their root is the session's binding, their blocks the request's,
    their commitment and root 64 bytes, their salt PoUS's, and the parameters P2's at those blocks and block size. Each
    ValueError names the field and the rule, and quotes none of the request (`challenger_party.main` returns it)."""
    if "published" not in request and "params" not in request:
        return None
    s, p = request.get("published"), request.get("params")
    if not isinstance(s, dict) or sorted(s) != sorted(("blocks", "commitment", "root", "salt")):
        raise ValueError("a request's published outputs are {blocks, commitment, root, salt}")
    if s["root"] != request.get("binding"):
        raise ValueError("the setup's root is not the session's binding")
    if s["blocks"] != request.get("blocks"):
        raise ValueError("the setup's blocks (`published.blocks`) are not the session's `blocks`")
    if not isinstance(p, dict) or sorted(p) != sorted(PARAMS):
        raise ValueError(f"a setup's params are {list(PARAMS)}")
    q = P2Params(**p)
    if (q.blocks, q.block_bytes) != (request.get("blocks"), request.get("block_bytes")):
        raise ValueError("the setup's parameters (`params`) are not P2's at the session's blocks and block_bytes")
    try:
        published = R.Setup.from_json(s)
    except (TypeError, ValueError):
        raise ValueError("a setup's published blocks are an integer and its commitment, root and salt hex") from None
    if len(published.commitment) != 64 or len(published.root) != 64 or len(published.salt) != R.SALT_BYTES:
        raise ValueError("a setup's commitment and root are 64 bytes and its salt PoUS's")
    return s | {"params": p}


__all__ = ["CHALLENGER", "END", "ID", "MANIFEST", "MODULE", "PARAMS", "PREMISE", "PROVER",
           "SESSION_ID_TAG", "SETUPS", "WARMUPS", "WARMUP_INDEX", "Codec", "Frames", "Polled", "coin_session_id", "command",
           "placement", "request_of", "run", "salt", "session_id", "setup_of", "sleep_until"]
