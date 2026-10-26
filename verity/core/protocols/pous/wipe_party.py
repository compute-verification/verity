"""The streamed wipe's memory challenger as a party of its own, so it runs on a host the prover doesn't control (the
challenger host, `trusted-nebius-host`) while the prover stays on its node: `challenger_party`'s shape, for the wipe.

The challenger listens and the prover connects. The challenger streams the fill as frames `F ‖ u64 first ‖ u32 count ‖
labels` and ends it with `D`, which the prover acknowledges with `A` once it holds the fill, so that no read's time
includes the fill still in flight. Then it sends each batch as `C ‖ u32 count ‖ u64 index …` (reading the send time),
and the prover answers with the batch's labels, 64 bytes an index (the challenger reads the receipt time); `E` ends the
session, after its last batch or a late or wrong one. The challenger draws the labels' secret, the session id and its coins on its own host, claims the
id in its own session log, and signs under its own key with the egress its host was given (`--egress-bits`, the network
certifier's count over the session; `UNCOUNTED`, which bounds nothing, when it has none): never a count the prover
sends. A prover that leaves before the session ends leaves it unsigned. The side reports `{"commitment", "ended",
"host", "public_key", "reads", "signed"}`, `signed` null when unsigned.

`python -m verity.core.protocols.pous.wipe_party --device DIR [--listen HOST:PORT] [--egress-bits N]` is the
challenger's side (`main`): it reads the request `{"k", "labels", "limit_ns", "node", "tiers"}` as one JSON line on
stdin, prints `{"listening": PORT}`, serves one prover (`serve`) and prints its report. `session(party, request,
prover)` is the leg's side: it starts the challenger's side with `party.command`, connects to `party.address` (by
default the port it printed, on this machine) and answers from `prover` (`answer`).
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

from . import memory_challenger as mc
from . import streamed_wipe as sw

MODULE = "verity.core.protocols.pous.wipe_party"
PREMISE = "trusted-nebius-host"
FILL, DONE, ACK, CHALLENGE, END = b"F", b"D", b"A", b"C", b"E"
CHUNK = 1 << 14
"""Labels a fill frame: 1 MiB."""
UNCOUNTED = sw.EGRESS_MAX_UNITS * sw.EGRESS_UNIT_BITS
"""The egress a session signs when no network certifier counted it: saturated, so it bounds nothing."""


def _read(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ConnectionError(f"the peer closed after {len(b)} of {n} bytes")
    return b


def params_of(request: Mapping[str, Any]) -> sw.WipeParams:
    return sw.WipeParams(request["labels"], request["k"], 1, request["limit_ns"],
                         tuple((t, b) for t, b in request["tiers"]))


def serve(f: BinaryIO, request: Mapping[str, Any], *, signer: mc.Signer, log: mc.SessionLog, egress_bits: int,
          clock: Callable[[], int] = time.perf_counter_ns, chunk: int = CHUNK) -> sw.Challenger:
    """The challenger's side of one wipe over the connection `f`: the labels' secret and the session id drawn here, the
    fill streamed, each batch revealed and its answers read on `clock`, the session signed by `signer` with
    `egress_bits` once it ends. A prover that closes first leaves it unsigned."""
    c = sw.Challenger(secrets.token_bytes(32), request["node"].encode(), secrets.token_bytes(32), params_of(request),
                      signer, log=log, clock=clock)
    try:
        for first, data in c.fill(chunk):
            f.write(FILL + first.to_bytes(8, "big") + (len(data) // sw.LABEL_BYTES).to_bytes(4, "big") + data)
        f.write(DONE)
        f.flush()
        if _read(f, 1) != ACK:
            raise ConnectionError("the prover didn't acknowledge the fill")
        while c.ended is None:
            idx = c.challenge()
            f.write(CHALLENGE + len(idx).to_bytes(4, "big") + b"".join(i.to_bytes(8, "big") for i in idx))
            f.flush()
            c.receive(_read(f, sw.LABEL_BYTES * len(idx)))
        f.write(END)
        f.flush()
    except (ConnectionError, BrokenPipeError):
        return c
    c.close(egress_bits)
    return c


def answer(f: BinaryIO, prover: sw.Prover, stop_after: int | None = None) -> int:
    """The prover's side over the connection `f`: each fill frame kept by `prover`, each batch answered from it until
    the challenger ends the session (or, with `stop_after`, the prover leaves after that many batches); the batches
    answered."""
    n = 0
    while (tag := _read(f, 1)) != END:
        if tag == FILL:
            first, count = int.from_bytes(_read(f, 8), "big"), int.from_bytes(_read(f, 4), "big")
            prover.receive(first, _read(f, sw.LABEL_BYTES * count))
        elif tag == DONE:
            f.write(ACK)
            f.flush()
        elif tag == CHALLENGE:
            if stop_after is not None and n >= stop_after:
                return n
            count = int.from_bytes(_read(f, 4), "big")
            f.write(prover.answer([int.from_bytes(_read(f, 8), "big") for _ in range(count)]))
            f.flush()
            n += 1
        else:
            raise ValueError(f"the challenger sent the frame tag {tag!r}")
    return n


def report(c: sw.Challenger, public_key: bytes) -> dict[str, Any]:
    return {"commitment": c.commitment.to_json(), "ended": c.ended, "host": socket.gethostname(),
            "public_key": public_key.hex(), "reads": len(c.batches),
            "signed": None if c.signed is None else c.signed.to_json()}


def commitment_of(j: Mapping[str, Any]) -> sw.WipeCommitment:
    """`WipeCommitment.to_json()`'s inverse, on the challenger's own report."""
    p = sw.WipeParams(j["labels"], j["k"], j["batch"], j["limit_ns"], tuple((t, b) for t, b in j["tiers"]))
    return sw.WipeCommitment(bytes.fromhex(j["server"]), bytes.fromhex(j["session"]), p)


def signed_of(j: Mapping[str, Any]) -> sw.SignedWipe:
    """`SignedWipe.to_json()`'s inverse, on the challenger's own report."""
    batches = tuple(sw.BatchRecord(r["b"], r["first"], r["count"], r["send_ns"], r["receipt_ns"], r["wrong"],
                                   bytes.fromhex(r["digest"])) for r in j["batches"])
    return sw.SignedWipe(commitment_of(j["commitment"]), j["ended"], j["fill_start_ns"], j["fill_end_ns"], batches,
                         j["egress_bits"], j["scheme"], bytes.fromhex(j["public_key"]), bytes.fromhex(j["signature"]))


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
        node, else `challenger.wipe`'s `command` and `address` (`"HOST:PORT"`), which infra writes once the host serves
        the challenger's side."""
        host = period.get("challenger_host") or node
        if host == node:
            return cls.local(node, device)
        entry = (period.get("challenger") or {}).get("wipe") or {}
        if not entry.get("command") or not entry.get("address"):
            raise RuntimeError(f"the challenger host {host} isn't wired for the wipe: period.json's challenger.wipe.command "
                               f"and .address start its side there and reach it from {node} (infra's)")
        h, _, p = entry["address"].rpartition(":")
        return cls(host, tuple(entry["command"]), (h, int(p)))


def placement(node: str, host: str, *, verifier: str | None = None) -> dict[str, Any]:
    """Which host ran each party, and whether `trusted-nebius-host`'s premise holds: that the memory challenger ran on a
    machine the prover doesn't control, which is never the prover's node."""
    held = host != node
    return {"parties": {"memory_challenger": host, "prover": node, "verifier": verifier or node},
            PREMISE: (f"holds: the memory challenger ran on {host}, not the prover's {node}" if held else
                      f"unmet: the memory challenger ran on the prover's node {node}")}


def request_of(node: str, params: sw.WipeParams) -> dict[str, Any]:
    if not params.covered:
        raise ValueError("the wipe's party reveals one index a batch, the reveal the guarantee covers")
    return {"k": params.k, "labels": params.labels, "limit_ns": params.limit_ns, "node": node,
            "tiers": [[t, b] for t, b in params.tiers]}


def session(party: Party, request: Mapping[str, Any], prover: sw.Prover, *,
            stop_after: int | None = None) -> dict[str, Any]:
    """One wipe with the challenger's side `party` starts: `request` sent, the fill kept and each batch answered by
    `prover`, the side's report back, with `answered`, the batches the prover answered."""
    p = subprocess.Popen(list(party.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        p.stdin.write(json.dumps(dict(request), sort_keys=True) + "\n")
        p.stdin.close()
        port = json.loads(p.stdout.readline())["listening"]
        with socket.create_connection(party.address or ("127.0.0.1", port)) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with s.makefile("rwb") as f:
                answered = answer(f, prover, stop_after)
        out = json.loads(p.stdout.readline())
    finally:
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"the challenger's side on {party.host} exited {rc}")
    return out | {"answered": answered}


def main(argv: Sequence[str] | None = None) -> int:
    from . import period  # noqa: PLC0415  (its challenger key, made once in the device directory)

    ap = argparse.ArgumentParser(prog=MODULE, description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", required=True, help="the challenger's key and session log (`period.challenger`)")
    ap.add_argument("--listen", default="127.0.0.1:0", help="HOST:PORT to listen on (port 0: any)")
    ap.add_argument("--egress-bits", type=int, default=UNCOUNTED,
                    help="the network certifier's egress count over the session, in bits (default: uncounted)")
    a = ap.parse_args(argv)
    request = json.loads(sys.stdin.readline())
    device = Path(a.device)
    signer = period.challenger(device, workers=len(os.sched_getaffinity(0)))
    if not (device / "sessions").exists():
        mc.SessionLog.create(device / "sessions")
    host, _, port = a.listen.rpartition(":")
    with socket.create_server((host, int(port))) as srv:
        print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
        conn, _ = srv.accept()
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with conn.makefile("rwb") as f:
                c = serve(f, request, signer=signer, log=mc.SessionLog(device / "sessions"), egress_bits=a.egress_bits)
    print(json.dumps(report(c, signer.public_key), sort_keys=True), flush=True)
    return 0


__all__ = ["CHUNK", "PREMISE", "UNCOUNTED", "Party", "answer", "commitment_of", "main", "params_of", "placement", "report",
           "request_of", "serve", "session", "signed_of"]

if __name__ == "__main__":
    sys.exit(main())
