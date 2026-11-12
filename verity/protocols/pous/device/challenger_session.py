"""The memory challenger's session as PoUS runs it: the wire both sides speak, the prover's side, the request and the id
a session runs under, and how the leg starts the challenger's side and talks to it. The challenger's side is the
auditor-trusted device (`challenger_party`, on a host the developer doesn't control: the challenger host, `trusted-host`),
which imports this module and which this module never imports; how it runs a session is its docstring's.

The two sides are parties of the deployment layer (`verity.configuration.deploy`), the challenger (`challenger_party.Challenger`)
and the prover (`Prover`), whose messages are one frame each way a round (`MANIFEST`):

~~~text
memory-challenger -> prover   Warmup {index} ... then per round Index {index} (coins); End
prover -> memory-challenger   per frame Block, the whole block (private)
~~~

`Frames` is their codec over the session's stream, `u64be index` for `Warmup` and `Index`, `u64be 2^64 − 1` for `End`
and the whole block for a `Block`, which the deployment layer's byte-stream adapter (`verity.configuration.deploy.stream`) drives
them on: `answer` is the prover's party run there. Each side reads its connection through `Polled`, which polls for up
to `SPIN_NS` before it blocks, so neither host's CPU sleeps through a round.

A request that names no audit runs under the id the auditor designates from its coins, the node and the setup's root
(`coin_session_id`, under `period.SESSION_DOMAIN`). One that names its audit (`request_of` with `audit`) runs under
`session_id(audit, commitment, salt)`, of the commitment and salt the challenger logged for the audit, which the
verifier computes from the audit's binding and what setup published, never from the report; it carries setup's published
outputs and P2's parameters (`PARAMS`), which `setup_of` reads back. The challenger keeps each session it completes on
its own host under `SETUPS`, from which the remote bandwidth audit's verifier on the same host builds its registration
(`remote_bandwidth.p2_party --pous-record`).

`side_of(period, node, device)` is where `period.json` puts the challenger's side (`stream.Side`, started by `command`),
`salt(side, audit, commitment)` asks it for a salt, and `session(side, request, respond)` is the leg's: it starts the
side, connects to it and answers each index with `respond(index)` (`answer`). `placement` says which host ran each
party and whether `trusted-host`'s premise holds.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

from verity.configuration.deploy import stream
from verity.configuration.deploy.manifest import Channel, Manifest, Role
from verity.configuration.deploy.party import Finish, Identity, Received, Send
from verity.configuration.deploy.stream import Side
from verity.configuration.wire import Body, Int, Message, Schema
from verity.commitments.randomness import derive

from . import memory_challenger as mc
from verity.protocols.pous.verifier import run as R
from verity.protocols.pous.verifier.schemes.p2 import P2Params

END = 2 ** 64 - 1
#: the untimed exchanges before round 0 (`challenger_party`'s docstring); the index is public, so they reveal no coin
WARMUPS, WARMUP_INDEX = 8, 0
MODULE = "verity.protocols.pous.device.challenger_party"
PREMISE = "trusted-host"
ID = "verity/pous/memory-challenger"
VERSION = 1
CHALLENGER, PROVER = "memory-challenger", "prover"
SESSION_ID_TAG = b"verity-pous/memory-challenger/session-id/v1\0"
AUDIT_BYTES = 64
MAX_BLOCK = 1 << 24
MANIFEST = Manifest(
    ID, VERSION, "verity.protocols.pous.device.challenger_session:request_of",
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

    def __enter__(self) -> Polled:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


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


class Prover:
    """The prover: each index answered with its whole block (`respond`, else its machine's `capabilities["store"]`)
    until the challenger ends the session; it finishes with the frames it answered, the warm-ups among them."""

    def __init__(self, respond: Callable[[int], bytes] | None = None) -> None:
        self.respond = respond

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.block_bytes, self.answered = params["block_bytes"], 0
        if self.respond is None:
            self.respond = identity.capabilities["store"]
        return []

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Refused", "Closed"):
            return [Finish(self.answered, m.fields["name"] if m.type == "Refused" else "closed")]
        if m.type == "End":
            return [Finish(self.answered)]
        block = self.respond(m.fields["index"])
        if len(block) != self.block_bytes:
            raise ValueError(f"an answer is the whole {self.block_bytes}-byte block, not {len(block)} bytes")
        self.answered += 1
        return [Send(CHALLENGER, Message("Block", body=block))]


def answer(f: BinaryIO, block_bytes: int, respond: Callable[[int], bytes]) -> int:
    """The prover's side over the connection `f`: `Prover`, each index answered with `respond(index)` until the
    challenger ends the session, the collector paused as on the challenger's side; the frames answered, the warm-ups
    among them."""
    with mc.collector_paused():
        done, _ = stream.run(Prover(respond), {"block_bytes": block_bytes}, Identity(PROVER), f, Frames(),
                             peer=CHALLENGER)
    return done.outcome


def command(device: str | Path) -> tuple[str, ...]:
    """The argv that starts the challenger's side as a process on this machine, its key and logs in `device`."""
    return (sys.executable, "-m", MODULE, "--device", str(device), "--listen", "127.0.0.1:0")


def side_of(period: Mapping[str, Any], node: str, device: str | Path) -> Side:
    """`period.json`'s challenger host (`challenger_host`, by default `node`): this machine's process when it is the
    node, else `challenger.pous`'s `command` and `address`, which infra writes once the host serves the challenger's
    side (the bandwidth gate's path, `verity/protocols/poub/device/live_verifier`)."""
    return Side.of(period, node, "pous", command(device))


def placement(node: str, host: str, *, verifier: str | None = None) -> dict[str, Any]:
    """Which host ran each party, and whether `trusted-host`'s premise holds: that the memory challenger ran on a
    machine the developer doesn't control, which is never the developer's node."""
    held = host != node
    return {"parties": {"memory_challenger": host, "prover": node, "verifier": verifier or node},
            PREMISE: (f"holds: the memory challenger ran on {host}, not the prover's {node}" if held else
                      f"unmet: the memory challenger ran on the prover's node {node}")}


def session(side: Side, request: Mapping[str, Any], respond: Callable[[int], bytes]) -> dict[str, Any]:
    """One session with the challenger's side `side` starts: `request` sent, each index answered with `respond`, the
    side's report back, with `rounds`, the timed rounds the prover answered (the frames less the side's warm-ups)."""

    def drive(s: socket.socket) -> int:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return answer(Polled(s), request["block_bytes"], respond)

    out, rounds = side.talk(request, drive)
    return out | {"rounds": rounds - out.get("warmups", 0)}


def salt(side: Side, audit: bytes, commitment: bytes) -> bytes:
    """The salt's source for the audit `audit`, which the challenger's side `side` samples after `commitment` and logs
    (`give_salt`); RuntimeError with the side's refusal, or when it answers anything else or exits nonzero."""
    p = subprocess.Popen(list(side.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        p.stdin.write(json.dumps({"salt": {"audit": audit.hex(), "commitment": commitment.hex()}}, sort_keys=True) + "\n")
        p.stdin.close()
        line = p.stdout.readline()
        doc = json.loads(line) if line.strip() else {}
        if "listening" in doc:
            # a side older than the salt request waits for a session's prover, holding the host's gate: end it now
            with socket.create_connection(side.address or ("127.0.0.1", doc["listening"])):
                pass
    finally:
        rc = p.wait()
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
    their commitment and root 64 bytes, their salt PoUS's, and the parameters P2's at those blocks and block size."""
    if "published" not in request and "params" not in request:
        return None
    s, p = request.get("published"), request.get("params")
    if not isinstance(s, dict) or sorted(s) != sorted(("blocks", "commitment", "root", "salt")):
        raise ValueError("a request's published outputs are {blocks, commitment, root, salt}")
    if s["root"] != request["binding"]:
        raise ValueError("the setup's root is not the session's binding")
    if s["blocks"] != request["blocks"]:
        raise ValueError(f"the setup is of {s['blocks']} blocks, the session of {request['blocks']}")
    if not isinstance(p, dict) or sorted(p) != sorted(PARAMS):
        raise ValueError(f"a setup's params are {list(PARAMS)}")
    q = P2Params(**p)
    if (q.blocks, q.block_bytes) != (request["blocks"], request["block_bytes"]):
        raise ValueError("the setup's parameters are not the session's blocks")
    published = R.Setup.from_json(s)
    if len(published.commitment) != 64 or len(published.root) != 64 or len(published.salt) != R.SALT_BYTES:
        raise ValueError("a setup's commitment and root are 64 bytes and its salt PoUS's")
    return s | {"params": p}


__all__ = ["AUDIT_BYTES", "CHALLENGER", "END", "ID", "MANIFEST", "MAX_BLOCK", "MODULE", "PARAMS", "PREMISE", "PROVER",
           "SESSION_ID_TAG", "SETUPS", "SPIN_NS", "VERSION", "WARMUPS", "WARMUP_INDEX", "Frames", "Polled", "Prover", "answer",
           "coin_session_id", "command", "placement", "request_of", "salt", "session", "session_id", "setup_of", "side_of"]
