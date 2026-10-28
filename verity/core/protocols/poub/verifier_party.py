"""The remote bandwidth session's verifier and challenger as a party of their own, so they run on a host the prover
doesn't control (the challenger host, `trusted-host`) while the prover stays on its node.

The verifier's side listens and the prover connects; each message is one line of JSON. The prover opens with its
store's block count and payload root (`{"register": {"blocks", "payload_root"}}`). The verifier registers it from
its own coins and answers with the registration and the session's coin commitment. It then sends each pass's challenge
(`{"run": {"challenge", "i", "previous", "t"}}`, answered `{"root"}`) and each window's draws (`{"open": {"draws",
"t"}}`, answered `{"openings": hex}` in the openings frame, `wire.encode_openings`, which it decodes strictly), and
`{"end": true}` last. Its clock, log and coins are the verifier's own (`period.session`), so the case is the one
`period.run` makes in one process.

`python -m verity.core.protocols.poub.verifier_party [--listen HOST:PORT]` is the verifier's side (`main`):
it reads the request `{"deadline_ns", "node", "passes", "samples", "window_ns", "windows"}` as one JSON line on stdin,
prints `{"listening": PORT}`, serves one prover and prints `{"answer", "case", "host"}`. `period_with(party, request,
store)` is the leg's side: it starts the verifier's side with `party.command`, connects to `party.address` (by default
the port it printed, on this machine) and plays the prover (`prove`).
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
from typing import Any, BinaryIO

from . import audit, wire
from . import coins as C
from . import period as P

MODULE = "verity.core.protocols.poub.verifier_party"
PREMISE = "trusted-host"


def _send(f: BinaryIO, doc: Mapping[str, Any]) -> None:
    f.write(json.dumps(dict(doc), sort_keys=True).encode() + b"\n")
    f.flush()


def _recv(f: BinaryIO) -> dict[str, Any]:
    line = f.readline()
    if not line.endswith(b"\n"):
        raise ConnectionError("the peer closed mid-session")
    doc = json.loads(line)
    if not isinstance(doc, dict) or len(doc) != 1:
        raise ValueError("a message is a JSON object of one key")
    return doc


class Remote:
    """The prover across the connection `f`, as the verifier's side sees it (`period.Prover`)."""

    def __init__(self, f: BinaryIO) -> None:
        self.f = f

    def register(self) -> tuple[int, bytes]:
        m = _recv(self.f).get("register")
        if not isinstance(m, dict) or type(m.get("blocks")) is not int or not isinstance(m.get("payload_root"), str):
            raise ValueError("the prover opens with {\"register\": {\"blocks\", \"payload_root\"}}")
        return m["blocks"], bytes.fromhex(m["payload_root"])

    def hold(self, registration: audit.Registration, commitment: C.LiveCommitment) -> None:
        _send(self.f, {"hold": {"commitment": commitment.to_json(), "registration": P.registration_json(registration)}})

    def run(self, challenge: C.LiveOpening, t: int, i: int, previous: bytes) -> bytes:
        _send(self.f, {"run": {"challenge": challenge.to_json(), "i": i, "previous": previous.hex(), "t": t}})
        root = _recv(self.f).get("root")
        if not isinstance(root, str):
            raise ValueError("a pass is answered {\"root\": hex}")
        return bytes.fromhex(root)

    def open(self, t: int, draws: Sequence[tuple[int, int]]) -> list[audit.Opening]:
        _send(self.f, {"open": {"draws": [list(d) for d in draws], "t": t}})
        body = _recv(self.f).get("openings")
        got = wire.decode_openings(bytes.fromhex(body)) if isinstance(body, str) else None
        return [] if got is None else got

    def end(self) -> None:
        _send(self.f, {"end": True})


def prove(f: BinaryIO, prover: P.Local) -> int:
    """The prover's side over the connection `f`, `prover` holding the store: each message answered until the verifier
    ends the session; the passes run."""
    blocks, root = prover.register()
    _send(f, {"register": {"blocks": blocks, "payload_root": root.hex()}})
    passes = 0
    while True:
        (kind, m), = _recv(f).items()
        if kind == "hold":
            prover.hold(P.registration_of(m["registration"]), C.LiveCommitment.from_json(m["commitment"]))
        elif kind == "run":
            root = prover.run(C.LiveOpening.from_json(m["challenge"]), m["t"], m["i"], bytes.fromhex(m["previous"]))
            _send(f, {"root": root.hex()})
            passes += 1
        elif kind == "open":
            _send(f, {"openings": wire.encode_openings(prover.open(m["t"], [tuple(d) for d in m["draws"]])).hex()})
        elif kind == "end":
            return passes
        else:
            raise ValueError(f"no message {kind!r}")


def serve(f: BinaryIO, request: Mapping[str, Any], *, coins: bytes, clock: Callable[[], int] = time.monotonic_ns,
          sleep_until: Callable[[Callable[[], int], int], None] = P._sleep_until) -> P.Period:
    """The verifier's side of one session over the connection `f` (`period.session` against `Remote`)."""
    remote = Remote(f)
    out = P.session(request["node"], remote, coins, windows=request["windows"], passes=request["passes"],
                    samples=request["samples"], window_ns=request["window_ns"], deadline_ns=request["deadline_ns"],
                    clock=clock, sleep_until=sleep_until)
    remote.end()
    return out


@dataclass(frozen=True)
class Party:
    """Where the verifier and its challenger run: `host`, the machine's name; `command`, the argv that starts their side
    there and speaks `main`'s stdin and stdout; `address`, where the prover connects, `None` for this machine at the
    port the side prints."""

    host: str
    command: tuple[str, ...]
    address: tuple[str, int] | None = None

    @classmethod
    def of(cls, period: Mapping[str, Any], node: str) -> Party:
        """`period.json`'s challenger host (`challenger_host`, by default `node`): this machine's process when it is the
        node, else `challenger.bandwidth`'s `command` and `address` (`"HOST:PORT"`), which infra writes once the host
        serves the verifier's side (`benchmarks/live_verifier`)."""
        host = period.get("challenger_host") or node
        if host == node:
            return cls(node, (sys.executable, "-m", MODULE, "--listen", "127.0.0.1:0"))
        entry = (period.get("challenger") or {}).get("bandwidth") or {}
        if not entry.get("command") or not entry.get("address"):
            raise RuntimeError(f"the challenger host {host} isn't wired for bandwidth: period.json's "
                               f"challenger.bandwidth.command and .address start the verifier there and reach it from "
                               f"{node} (infra's)")
        h, _, p = entry["address"].rpartition(":")
        return cls(host, tuple(entry["command"]), (h, int(p)))


def placement(node: str, host: str) -> dict[str, Any]:
    """Which host ran each party, and whether `trusted-host`'s premise holds: that the verifier and its challenger
    ran on a machine the prover doesn't control, which is never the prover's node."""
    return {"parties": {"challenger": host, "prover": node, "verifier": host},
            PREMISE: (f"holds: the verifier and its challenger ran on {host}, not the prover's {node}" if host != node else
                      f"unmet: the verifier and its challenger ran on the prover's node {node}")}


def request_of(node: str, *, windows: int = 2, passes: int = 1, samples: int = audit.SAMPLES,
               window_ns: int = 100_000_000, deadline_ns: int = 500_000_000) -> dict[str, Any]:
    """The session's request, `period.run`'s defaults unless given."""
    return {"deadline_ns": deadline_ns, "node": node, "passes": passes, "samples": samples, "window_ns": window_ns,
            "windows": windows}


def period_with(party: Party, request: Mapping[str, Any], store: bytes, *,
                entropy: Callable[[int], bytes] = os.urandom) -> tuple[P.Period, dict[str, Any]]:
    """One session with the verifier's side `party` starts and this process the prover of `store`: the period, and the
    side's report (`host`, `passes`)."""
    p = subprocess.Popen(list(party.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        p.stdin.write(json.dumps(dict(request), sort_keys=True) + "\n")
        p.stdin.close()
        port = json.loads(p.stdout.readline())["listening"]
        with socket.create_connection(party.address or ("127.0.0.1", port)) as s, s.makefile("rwb") as f:
            passes = prove(f, P.Local(request["node"], store, entropy=entropy))
        got = json.loads(p.stdout.readline())
    finally:
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"the verifier's side on {party.host} exited {rc}")
    return P.Period(got["answer"], got["case"]), {"host": got["host"], "passes": passes}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=MODULE, description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", default="127.0.0.1:0", help="HOST:PORT to listen on (port 0: any)")
    a = ap.parse_args(argv)
    request = json.loads(sys.stdin.readline())
    host, _, port = a.listen.rpartition(":")
    with socket.create_server((host, int(port))) as srv:
        print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
        conn, _ = srv.accept()
        with conn, conn.makefile("rwb") as f:
            period = serve(f, request, coins=secrets.token_bytes(64))
    print(json.dumps({"answer": period.answer, "case": period.case, "host": socket.gethostname()}, sort_keys=True),
          flush=True)
    return 0


__all__ = ["PREMISE", "Party", "Remote", "main", "period_with", "placement", "prove", "request_of", "serve"]

if __name__ == "__main__":
    sys.exit(main())
