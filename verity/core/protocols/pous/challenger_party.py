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

`python -m verity.core.protocols.pous.challenger_party --device DIR [--listen HOST:PORT]` is the challenger's side
(`main`): it reads the request `{"binding", "block_bytes", "blocks", "k", "limit_ns", "node"}` as one JSON line on stdin,
prints `{"listening": PORT}`, serves one prover (`serve`) and prints its report. `session(party, request, respond)` is
the leg's side: it starts the challenger's side with `party.command`, connects to `party.address` (by default the port
it printed, on this machine) and answers each index with `respond(index)` (`answer`).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from verity.ml.randomness import derive

from . import memory_challenger as mc
from . import run as R

END = 2 ** 64 - 1
#: the untimed exchanges before round 0 (module docstring); the index is public, so they reveal no coin
WARMUPS, WARMUP_INDEX = 8, 0
MODULE = "verity.core.protocols.pous.challenger_party"
PREMISE = "trusted-host"


def _read(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ConnectionError(f"the peer closed after {len(b)} of {n} bytes")
    return b


def serve(f: BinaryIO, request: Mapping[str, Any], *, signer: mc.Signer, log: mc.SessionLog, coins: bytes,
          domain: str, clock: Callable[[], int] = time.perf_counter_ns,
          warmups: int = WARMUPS) -> tuple[bytes, R.ChallengerRecord]:
    """The challenger's side of one session over the connection `f`: the coin tree drawn from `coins` under the id it
    designates from them, `request`'s node and binding (`domain`'s derivation), `warmups` untimed exchanges on
    `WARMUP_INDEX`, then each index sent and its block read on `clock`, the session signed by `signer`; its id and
    record."""
    binding = bytes.fromhex(request["binding"])
    sid = derive(coins, domain, {"node": request["node"], "setup": binding}).stream("session", 32)
    tree = mc.CoinTree.draw(coins, sid, request["blocks"], request["k"])
    c = mc.MemoryChallenger(tree, signer, log=log, limit_ns=request["limit_ns"], block_bytes=request["block_bytes"],
                            binding=binding, clock=clock)

    def respond(i: int) -> bytes:
        f.write(i.to_bytes(8, "big"))
        f.flush()
        return _read(f, request["block_bytes"])

    with mc.collector_paused():
        for _ in range(warmups):
            respond(WARMUP_INDEX)
        signed = c.run(respond)
    f.write(END.to_bytes(8, "big"))
    f.flush()
    return sid, R.ChallengerRecord(signed, c.commitment, c.openings())


def answer(f: BinaryIO, block_bytes: int, respond: Callable[[int], bytes]) -> int:
    """The prover's side over the connection `f`: each index answered with `respond(index)` until the challenger ends the
    session, the collector paused as on the challenger's side (`memory_challenger.collector_paused`); the frames
    answered, the warm-ups among them."""
    n = 0
    with mc.collector_paused():
        while (i := int.from_bytes(_read(f, 8), "big")) != END:
            block = respond(i)
            if len(block) != block_bytes:
                raise ValueError(f"an answer is the whole {block_bytes}-byte block, not {len(block)} bytes")
            f.write(block)
            f.flush()
            n += 1
    return n


def report(sid: bytes, session: R.ChallengerRecord, public_key: bytes, warmups: int = WARMUPS) -> dict[str, Any]:
    return {"host": socket.gethostname(), "public_key": public_key.hex(), "session": session.to_json(),
            "session_id": sid.hex(), "warmups": warmups}


@dataclass(frozen=True)
class Party:
    """Where the memory challenger runs: `host`, the machine's name; `command`, the argv that starts its side there and
    speaks `main`'s stdin and stdout; `address`, where the prover connects, `None` for this machine at the port the side
    prints."""

    host: str
    command: tuple[str, ...]
    address: tuple[str, int] | None = None

    @classmethod
    def local(cls, host: str, device: str | Path) -> Party:
        """The challenger's side as a process on this machine (`host`), its key and session log in `device`."""
        return cls(host, (sys.executable, "-m", MODULE, "--device", str(device), "--listen", "127.0.0.1:0"))

    @classmethod
    def of(cls, period: Mapping[str, Any], node: str, device: str | Path) -> Party:
        """`period.json`'s challenger host (`challenger_host`, by default `node`): this machine's process when it is the
        node, else `challenger.pous`'s `command` and `address` (`"HOST:PORT"`), which infra writes once the host serves
        the challenger's side (the bandwidth gate's path, `benchmarks/live_verifier`)."""
        host = period.get("challenger_host") or node
        if host == node:
            return cls.local(node, device)
        entry = (period.get("challenger") or {}).get("pous") or {}
        if not entry.get("command") or not entry.get("address"):
            raise RuntimeError(f"the challenger host {host} isn't wired for pous: period.json's challenger.pous.command "
                               f"and .address start its side there and reach it from {node} (infra's)")
        h, _, p = entry["address"].rpartition(":")
        return cls(host, tuple(entry["command"]), (h, int(p)))


def placement(node: str, host: str, *, verifier: str | None = None) -> dict[str, Any]:
    """Which host ran each party, and whether `trusted-host`'s premise holds: that the memory challenger ran on a
    machine the prover doesn't control, which is never the prover's node."""
    held = host != node
    return {"parties": {"memory_challenger": host, "prover": node, "verifier": verifier or node},
            PREMISE: (f"holds: the memory challenger ran on {host}, not the prover's {node}" if held else
                      f"unmet: the memory challenger ran on the prover's node {node}")}


def session(party: Party, request: Mapping[str, Any], respond: Callable[[int], bytes]) -> dict[str, Any]:
    """One session with the challenger's side `party` starts: `request` sent, each index answered with `respond`, the
    side's report back, with `rounds`, the timed rounds the prover answered (the frames less the side's warm-ups)."""
    p = subprocess.Popen(list(party.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        p.stdin.write(json.dumps(dict(request), sort_keys=True) + "\n")
        p.stdin.close()
        port = json.loads(p.stdout.readline())["listening"]
        with socket.create_connection(party.address or ("127.0.0.1", port)) as s, s.makefile("rwb") as f:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            rounds = answer(f, request["block_bytes"], respond)
        out = json.loads(p.stdout.readline())
    finally:
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"the challenger's side on {party.host} exited {rc}")
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
                           coins=secrets.token_bytes(64), domain=period.SESSION_DOMAIN)
    print(json.dumps(report(sid, s, signer.public_key), sort_keys=True), flush=True)
    return 0


__all__ = ["END", "PREMISE", "WARMUPS", "WARMUP_INDEX", "Party", "answer", "main", "placement", "report", "request_of",
           "serve", "session"]

if __name__ == "__main__":
    sys.exit(main())
