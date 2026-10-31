"""PoUS's memory challenger as a party of its own, so it runs on a host the prover doesn't control (the challenger host,
`trusted-host`) while the prover stays on its node.

The challenger listens and the prover connects. Each round is one frame each way: the challenger sends `u64be index`
(reading the send time) and the prover answers with the whole block, `block_bytes` of it (the challenger reads the receipt
time); `u64be 2^64 − 1` ends the session, after its last round or a late one. Before round 0 the challenger sends
`WARMUPS` frames on `WARMUP_INDEX`, which no coin chose, and discards their answers: nothing times, records or signs them,
so the first timed round doesn't pay for waking both processes and the prover's memory path after the session's setup.
The challenger draws its coins on its own host, designates the session id from them, the node and the setup's root
(`period.SESSION_DOMAIN`), signs under its own key and claims the id in its own session log. Nothing but the indices, one
a round, goes on the wire: the coin tree's root, the session id and the openings leave only in its report, `{"host",
"public_key", "session", "session_id", "warmups"}`, after the session's last answer and its signature.

The two sides are parties of the deployment layer (`verity.core.deploy`), the challenger (`Challenger`) and the prover
(`Prover`), whose messages are those frames. The challenger takes its coin tree's hiding key and each round's salt from
its host's coins too:

~~~text
memory-challenger -> prover   Warmup {index} ... then per round Index {index} (coins); End
prover -> memory-challenger   per frame Block, the whole block (private)
~~~

`Frames` is their codec over the session's stream, which the deployment layer's byte-stream adapter
(`verity.core.deploy.stream`) drives them on: `serve` and `answer` are the two parties run there, on the clock and coins
they are given. The challenger reads its clock once before each index it sends and once after each block it takes,
those readings alone, so on the same coins and readings it signs the same bytes wherever it runs.

`python -m verity.core.protocols.pous.challenger_party --device DIR [--listen HOST:PORT]` is the challenger's side
(`main`): it reads the request `{"binding", "block_bytes", "blocks", "k", "limit_ns", "node"}` as one JSON line on stdin,
prints `{"listening": PORT}`, serves one prover (`serve`) and prints its report. `side_of(period, node, device)` is where
`period.json` puts that side (`stream.Side`), and `session(side, request, respond)` is the leg's: it starts the side,
connects to it and answers each index with `respond(index)` (`answer`).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from verity.core.deploy import stream
from verity.core.deploy.manifest import Channel, Manifest, Role
from verity.core.deploy.party import Finish, Identity, Received, Send
from verity.core.deploy.runtime import ProtocolDef
from verity.core.deploy.stream import Side
from verity.core.wire import Body, Int, Message, Phase, Refused, Schema
from verity.ml.randomness import derive

from . import memory_challenger as mc
from . import run as R

END = 2 ** 64 - 1
#: the untimed exchanges before round 0 (module docstring); the index is public, so they reveal no coin
WARMUPS, WARMUP_INDEX = 8, 0
MODULE = "verity.core.protocols.pous.challenger_party"
PREMISE = "trusted-host"
ID = "verity/pous/memory-challenger"
VERSION = 1
CHALLENGER, PROVER = "memory-challenger", "prover"
COIN_BYTES = 64
"""The session's coins, which the challenger draws first (`main`'s `secrets.token_bytes(64)`), before its coin tree's
hiding key and salts."""
MAX_BLOCK = 1 << 24
MANIFEST = Manifest(
    ID, VERSION, "verity.core.protocols.pous.challenger_party:request_of",
    parties=(Role(CHALLENGER, "auditor", needs=("clock", "coins", "signing-key")),
             Role(PROVER, "developer", per="server", needs=("store",))),
    channels=(Channel(CHALLENGER, PROVER, ("Warmup", "Index", "End"), carries_coins=True),
              Channel(PROVER, CHALLENGER, ("Block",))),
    messages=(Schema("Warmup", 1, "structure", {"index": Int()}), Schema("Index", 1, "coins", {"index": Int()}),
              Schema("End", 1, "structure"), Schema("Block", 1, "private", body=Body(max=MAX_BLOCK))))


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


def _read_clock() -> Send:
    return Send("clock", Message("ClockRead"))


class Challenger:
    """The memory challenger: one session of `request_of`'s parameters on its clock and coins (module docstring), the id
    designated under `domain` (`period.SESSION_DOMAIN`). Its machine gives it `signing-key` (`capabilities["signer"]`,
    a `memory_challenger.Signer`, with `capabilities["log"]`, its session log); `resolution_ns` is how finely its clock
    ticks (`memory_challenger.resolution_of`). It finishes with the session and its id (private: what it signs is
    witness), or refused by name when the prover leaves (`closed`), sends a frame the wire refuses, or one out of
    phase."""

    def __init__(self, domain: str, warmups: int = WARMUPS, resolution_ns: float = 1) -> None:
        self.domain, self.warmups, self.resolution_ns = domain, warmups, resolution_ns
        self.c: mc.MemoryChallenger | None = None
        self.sid: bytes | None = None
        self._drawn: list[bytes] = []
        self.session: R.Session | None = None
        self._reading: int | None = None
        self._block: bytes | None = None
        self._warm = 0

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.request, caps = dict(params), identity.capabilities
        self.signer, self.log = caps["signer"], caps["log"]
        self.phase = Phase("Coins")
        return [Send("coins", Message("CoinsDraw", {"n": COIN_BYTES}))]

    def _clock(self) -> int:
        """The reading the challenger asked its clock for, used once: `memory_challenger.MemoryChallenger` reads it here."""
        if self._reading is None:
            raise RuntimeError("the memory challenger read its clock without a fresh reading")
        out, self._reading = self._reading, None
        return out

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Refused", "Closed"):
            return [Finish(None, m.fields["name"] if m.type == "Refused" else "closed", kind="private")]
        if m.type == "ClockReading":
            self._reading = m.fields["ns"]
            return self._challenge() if self._block is None else self._receive()
        try:
            self.phase.check(m)
        except Refused as e:
            return [Finish(None, e.name, kind="private")]
        self.phase.expect()
        if m.type == "Coins":
            return self._coins(m.body)
        if self._warm <= self.warmups:
            return self._warmup()
        self._block = m.body
        return [_read_clock()]

    def _coins(self, body: bytes) -> list[Any]:
        """The session's coins, then the coin tree's hiding key, then each round's salt, each drawn in turn; once all
        are in, the session's challenger."""
        q, self._drawn = self.request, [*self._drawn, body]
        if len(self._drawn) < q["k"] + 2:
            self.phase.expect("Coins")
            n = mc.HM96.key_bytes if len(self._drawn) == 1 else mc.HM96.salt_bytes
            return [Send("coins", Message("CoinsDraw", {"n": n}))]
        coins, key, *salts = self._drawn
        binding = bytes.fromhex(q["binding"])
        self.sid = derive(coins, self.domain, {"node": q["node"], "setup": binding}).stream("session", 32)
        tree = mc.CoinTree.draw(coins, self.sid, q["blocks"], q["k"], key=mc.coin_key(key), salts=salts)
        self.c = mc.MemoryChallenger(tree, self.signer, log=self.log, limit_ns=q["limit_ns"],
                                     block_bytes=q["block_bytes"], binding=binding, clock=self._clock,
                                     resolution_ns=self.resolution_ns)
        return self._warmup()

    def _warmup(self) -> list[Any]:
        """The next warm-up, or, after the last one's answer, the reading round 0 is sent at."""
        self._warm += 1
        if self._warm > self.warmups:
            return [_read_clock()]
        self.phase.expect("Block")
        return [Send(PROVER, Message("Warmup", {"index": WARMUP_INDEX}))]

    def _challenge(self) -> list[Any]:
        i = self.c.challenge()
        self.phase.expect("Block")
        return [Send(PROVER, Message("Index", {"index": i}))]

    def _receive(self) -> list[Any]:
        block, self._block = self._block, None
        self.c.receive(block)
        if not self.c.done:
            return [_read_clock()]
        self.session = R.Session(self.c.close(), self.c.commitment, self.c.openings())
        return [Send(PROVER, Message("End")),
                Finish({"session": self.session.to_json(), "session_id": self.sid.hex()}, kind="private")]


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


def _challenger() -> Challenger:
    from . import period  # period imports this module

    return Challenger(period.SESSION_DOMAIN)


PROTOCOL = ProtocolDef(MANIFEST, {CHALLENGER: _challenger, PROVER: Prover})


def serve(f: BinaryIO, request: Mapping[str, Any], *, signer: mc.Signer, log: mc.SessionLog, coins: bytes,
          domain: str, clock: Callable[[], int] = time.perf_counter_ns, warmups: int = WARMUPS,
          draw: Callable[[int], bytes] = os.urandom) -> tuple[bytes, R.Session]:
    """The challenger's side of one session over the connection `f`: `Challenger` on `coins` and `clock`, its id
    designated under `domain`, its coin tree's hiding key and salts from `draw`, `warmups` untimed exchanges first, the
    session signed by `signer`, with the collector paused (`memory_challenger.collector_paused`); its id and record."""
    party, given = Challenger(domain, warmups, mc.resolution_of(clock)), [coins]
    with mc.collector_paused():
        stream.run(party, request, Identity(CHALLENGER, capabilities={"signer": signer, "log": log}), f,
                   Frames(request["block_bytes"]), peer=PROVER, clock=clock,
                   coins=lambda n: given.pop() if given else draw(n))
    return party.sid, party.session


def answer(f: BinaryIO, block_bytes: int, respond: Callable[[int], bytes]) -> int:
    """The prover's side over the connection `f`: `Prover`, each index answered with `respond(index)` until the
    challenger ends the session, the collector paused as on the challenger's side; the frames answered, the warm-ups
    among them."""
    with mc.collector_paused():
        done, _ = stream.run(Prover(respond), {"block_bytes": block_bytes}, Identity(PROVER), f, Frames(),
                             peer=CHALLENGER)
    return done.outcome


def report(sid: bytes, session: R.Session, public_key: bytes, warmups: int = WARMUPS) -> dict[str, Any]:
    return {"host": socket.gethostname(), "public_key": public_key.hex(), "session": session.to_json(),
            "session_id": sid.hex(), "warmups": warmups}


def command(device: str | Path) -> tuple[str, ...]:
    """The argv that starts the challenger's side as a process on this machine, its key and session log in `device`."""
    return (sys.executable, "-m", MODULE, "--device", str(device), "--listen", "127.0.0.1:0")


def side_of(period: Mapping[str, Any], node: str, device: str | Path) -> Side:
    """`period.json`'s challenger host (`challenger_host`, by default `node`): this machine's process when it is the
    node, else `challenger.pous`'s `command` and `address`, which infra writes once the host serves the challenger's
    side (the bandwidth gate's path, `benchmarks/live_verifier`)."""
    return Side.of(period, node, "pous", command(device))


def placement(node: str, host: str, *, verifier: str | None = None) -> dict[str, Any]:
    """Which host ran each party, and whether `trusted-host`'s premise holds: that the memory challenger ran on a
    machine the prover doesn't control, which is never the prover's node."""
    held = host != node
    return {"parties": {"memory_challenger": host, "prover": node, "verifier": verifier or node},
            PREMISE: (f"holds: the memory challenger ran on {host}, not the prover's {node}" if held else
                      f"unmet: the memory challenger ran on the prover's node {node}")}


def session(side: Side, request: Mapping[str, Any], respond: Callable[[int], bytes]) -> dict[str, Any]:
    """One session with the challenger's side `side` starts: `request` sent, each index answered with `respond`, the
    side's report back, with `rounds`, the timed rounds the prover answered (the frames less the side's warm-ups)."""

    def drive(s: socket.socket) -> int:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with s.makefile("rwb") as f:
            return answer(f, request["block_bytes"], respond)

    out, rounds = side.talk(request, drive)
    return out | {"rounds": rounds - out.get("warmups", 0)}


def request_of(node: str, point: R.Point, binding: bytes) -> dict[str, Any]:
    q = point.params
    return {"binding": binding.hex(), "block_bytes": q.block_bytes, "blocks": q.blocks, "k": point.k,
            "limit_ns": point.limit_ns, "node": node}


def main(argv: Sequence[str] | None = None) -> int:
    from . import period  # noqa: PLC0415  (period imports this module)

    ap = argparse.ArgumentParser(prog=MODULE, description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", required=True, help="the challenger's key and session log (`period.challenger`)")
    ap.add_argument("--listen", default="127.0.0.1:0", help="HOST:PORT to listen on (port 0: any)")
    a = ap.parse_args(argv)
    request = json.loads(sys.stdin.readline())
    device = Path(a.device)
    signer = period.challenger(device, workers=min(period.SETUP_WORKERS, len(os.sched_getaffinity(0))))
    if not (device / "sessions").exists():
        mc.SessionLog.create(device / "sessions")
    host, _, port = a.listen.rpartition(":")
    with socket.create_server((host, int(port))) as srv:
        print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
        conn, _ = srv.accept()
        with conn, conn.makefile("rwb") as f:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sid, s = serve(f, request, signer=signer, log=mc.SessionLog(device / "sessions"),
                           coins=secrets.token_bytes(COIN_BYTES), domain=period.SESSION_DOMAIN)
    print(json.dumps(report(sid, s, signer.public_key), sort_keys=True), flush=True)
    return 0


__all__ = ["CHALLENGER", "COIN_BYTES", "END", "ID", "MANIFEST", "PREMISE", "PROTOCOL", "PROVER", "VERSION", "WARMUPS",
           "WARMUP_INDEX", "Challenger", "Frames", "Prover", "answer", "command", "main", "placement", "report",
           "request_of", "serve", "session", "side_of"]

if __name__ == "__main__":
    sys.exit(main())
