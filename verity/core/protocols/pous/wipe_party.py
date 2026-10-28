"""The streamed wipe's memory challenger as a party of its own, so it runs on a host the prover doesn't control (the
challenger host, `trusted-host`) while the prover stays on its node: `challenger_party`'s shape, for the wipe.

The challenger listens and the prover connects. The challenger streams the fill as frames `F ‖ u64 first ‖ u32 count ‖
labels` and ends it with `D`, which the prover acknowledges with `A` once it holds the fill, so that no read's time
includes the fill still in flight. Then it sends each batch as `C ‖ u32 count ‖ u64 index …` (reading the send time),
and the prover answers with the batch's labels, 64 bytes an index (the challenger reads the receipt time); `E` ends the
session, after its last batch or a late or wrong one. The challenger draws the labels' secret, the session id and its coins on its own host, claims the
id in its own session log, and signs under its own key with the egress its host was given (`--egress-bits`, the network
certifier's count over the session; `UNCOUNTED`, which bounds nothing, when it has none): never a count the prover
sends. It times the number of reads its own command names (`--reads`, `READS` by default), which the commitment
publishes and the certificate is computed at: a request that names `k` is refused, so the prover never chooses it. A
prover that leaves before the session ends leaves it unsigned. The side reports `{"commitment", "ended",
"host", "public_key", "reads", "signed", "window"}`, `signed` null when unsigned, and `window` the session's `t0` (the
prover connected) and `t1` (the session signed, or ended unsigned) on the challenger host's real-time clock: the
interval a network certifier counts the session's egress over.

`python -m verity.core.protocols.pous.wipe_party --device DIR [--listen HOST:PORT] [--reads K] [--egress-bits N]
[--streamer build|PATH [--threads T]]` is the challenger's side (`main`): it reads the request `{"labels", "limit_ns",
"node", "tiers"}` as one JSON line on stdin, prints `{"listening": PORT}`, serves one prover (`serve`) and prints its report.
With `--streamer` the fill is `wipe_stream`'s, native on this host's CPUs and gated before the side listens, and the
report adds the streamer's gate (`streamer`) and its figures for the fill (`fill`). `session(party, request, prover)` is
the leg's side: it starts the challenger's side with `party.command`, connects to `party.address` (by default the port
it printed, on this machine), answers from `prover` (`answer`) and adds the connection's two addresses as it saw them.
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
from . import wipe_stream

MODULE = "verity.core.protocols.pous.wipe_party"
PREMISE = "trusted-host"
FILL, DONE, ACK, CHALLENGE, END = b"F", b"D", b"A", b"C", b"E"
CHUNK = 1 << 14
"""Labels a fill frame: 1 MiB."""
UNCOUNTED = sw.EGRESS_MAX_UNITS * sw.EGRESS_UNIT_BITS
"""The egress a session signs when no network certifier counted it: saturated, so it bounds nothing."""
READS = 2152
"""The timed spot reads a challenger runs when its command doesn't name them (`--reads`)."""
REQUEST_KEYS = {"labels", "limit_ns", "node", "tiers"}


def _read(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ConnectionError(f"the peer closed after {len(b)} of {n} bytes")
    return b


def params_of(request: Mapping[str, Any], reads: int = READS) -> sw.WipeParams:
    """The session's parameters: the prover's request (`REQUEST_KEYS`: its node, fill, deadline and declared tiers) and
    the challenger's own `reads`."""
    if set(request) != REQUEST_KEYS:
        raise ValueError(f"a wipe request is {sorted(REQUEST_KEYS)}, not {sorted(request)}: the challenger chooses the "
                         "reads (--reads), never the prover")
    return sw.WipeParams(request["labels"], reads, 1, request["limit_ns"], tuple((t, b) for t, b in request["tiers"]))


def serve(f: BinaryIO, request: Mapping[str, Any], *, signer: mc.Signer, log: mc.SessionLog, egress_bits: int,
          reads: int = READS, clock: Callable[[], int] = time.perf_counter_ns, chunk: int = CHUNK,
          fill: Callable[[bytes, int], Any] | None = None) -> sw.Challenger:
    """The challenger's side of one wipe over the connection `f`: the labels' secret and the session id drawn here, the
    fill streamed (by `fill(K, N)` when given, which writes the frames to `f`'s connection itself: the native streamer,
    `wipe_stream`), `reads` batches revealed and their answers read on `clock`, the session signed by `signer` with
    `egress_bits` once it ends. A prover that closes first leaves it unsigned."""
    c = sw.Challenger(secrets.token_bytes(32), request["node"].encode(), secrets.token_bytes(32),
                      params_of(request, reads), signer, log=log, clock=clock)
    try:
        if fill is not None:
            f.flush()
            c.fill_by(fill)
        else:
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
    """The prover's side over the connection `f`: each fill frame kept by `prover` (read straight into its store when
    it has `receive_into`, as `wipe_store.StoreProver`), the fill acknowledged once it is in the store (after the
    prover's `sync`, when it has one: a disk's writes on the media), each batch answered from it until the challenger
    ends the session (or, with `stop_after`, the prover leaves after that many batches); the batches answered."""
    n = 0
    into, sync = getattr(prover, "receive_into", None), getattr(prover, "sync", None)
    while (tag := _read(f, 1)) != END:
        if tag == FILL:
            first, count = int.from_bytes(_read(f, 8), "big"), int.from_bytes(_read(f, 4), "big")
            if into is not None:
                into(first, count, f)
            else:
                prover.receive(first, _read(f, sw.LABEL_BYTES * count))
        elif tag == DONE:
            if sync is not None:
                sync()
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


def utc(unix_ns: int) -> str:
    """A real-time clock reading as ISO 8601 UTC, to the microsecond."""
    s, ns = divmod(unix_ns, 10 ** 9)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(s)) + f".{ns // 1000:06d}Z"


def report(c: sw.Challenger, public_key: bytes, window: tuple[int, int] | None = None) -> dict[str, Any]:
    """The side's report; `window`, the session's `(t0, t1)` in Unix nanoseconds on this host's real-time clock."""
    out = {"commitment": c.commitment.to_json(), "ended": c.ended, "host": socket.gethostname(),
           "public_key": public_key.hex(), "reads": len(c.batches),
           "signed": None if c.signed is None else c.signed.to_json()}
    if window is not None:
        out["window"] = {"t0": utc(window[0]), "t1": utc(window[1]), "t0_unix_ns": window[0], "t1_unix_ns": window[1],
                         "clock": f"{out['host']}'s real-time clock"}
    return out


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
    def local(cls, host: str, device: str | Path, extra: Sequence[str] = ()) -> Party:
        """The challenger's side as a process on this machine (`host`), its key and session log in `device`, `extra`
        more of `main`'s arguments (`--streamer build`)."""
        return cls(host, (sys.executable, "-m", MODULE, "--device", str(device), "--listen", "127.0.0.1:0", *extra))

    @classmethod
    def of(cls, period: Mapping[str, Any], node: str, device: str | Path, extra: Sequence[str] = ()) -> Party:
        """`period.json`'s challenger host (`challenger_host`, by default `node`): this machine's process when it is the
        node (with `extra`), else `challenger.wipe`'s `command` and `address` (`"HOST:PORT"`), which infra writes once
        the host serves the challenger's side."""
        host = period.get("challenger_host") or node
        if host == node:
            return cls.local(node, device, extra)
        entry = (period.get("challenger") or {}).get("wipe") or {}
        if not entry.get("command") or not entry.get("address"):
            raise RuntimeError(f"the challenger host {host} isn't wired for the wipe: period.json's challenger.wipe.command "
                               f"and .address start its side there and reach it from {node} (infra's)")
        h, _, p = entry["address"].rpartition(":")
        return cls(host, tuple(entry["command"]), (h, int(p)))


def link(name: str, node: str, to: str, via: str, *, mbps: int | None = None,
         counted_by: str | None = None) -> dict[str, Any]:
    """A link leaving the prover's node, as `Accounting.Placed.Links` declares one: its `ends` (the node, and `to` past
    its boundary), what carries it (`via`), its capacity in Mb/s when known, and the network certifier that counts what
    leaves on it (`counted_by`), `None` when none does, so its egress is outside the session's count."""
    return {"name": name, "ends": [node, to], "via": via, "mbps": mbps, "counted_by": counted_by}


def placement(node: str, host: str, *, verifier: str | None = None, store: Mapping[str, Any] | None = None,
              simulated: bool = False, links: Sequence[Mapping[str, Any]] = (),
              undeclared: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Which host ran each party, and whether `trusted-host`'s premise holds: that the memory challenger ran on a
    machine the prover doesn't control, which is never the prover's node. `kind` names the placement: `own-host`,
    `simulated` (the challenger a process of its own on the prover's node, standing in for its own host, the premise
    assumed) or `prover-node`. `store` is the memory the prover kept the fill in (`wipe_store`'s `info`), when it says.
    `links` are the links leaving the prover's node (`link`), and `undeclared` the memory the node has outside the fill's
    declared tiers (`{"tier", "bytes", "what"}`), which the bound doesn't count."""
    held = host != node
    if simulated and held:
        raise ValueError(f"a simulated placement runs the memory challenger on the prover's node {node}, not on {host}")
    kind = "own-host" if held else "simulated" if simulated else "prover-node"
    out = {"kind": kind, "parties": {"memory_challenger": host, "prover": node, "verifier": verifier or node},
           PREMISE: {"own-host": f"holds: the memory challenger ran on {host}, not the prover's {node}",
                     "simulated": f"simulated: the memory challenger ran as a process of its own on the prover's node "
                                  f"{node}, its key in its own scratch, standing in for a host the prover doesn't control",
                     "prover-node": f"unmet: the memory challenger ran on the prover's node {node}"}[kind],
           "links": [dict(e) for e in links], "undeclared_tiers": [dict(u) for u in undeclared]}
    return out if store is None else out | {"prover_store": dict(store)}


def request_of(node: str, params: sw.WipeParams) -> dict[str, Any]:
    if not params.covered:
        raise ValueError("the wipe's party reveals one index a batch, the reveal the guarantee covers")
    return {"labels": params.labels, "limit_ns": params.limit_ns, "node": node,
            "tiers": [[t, b] for t, b in params.tiers]}


def session(party: Party, request: Mapping[str, Any], prover: sw.Prover, *,
            stop_after: int | None = None) -> dict[str, Any]:
    """One wipe with the challenger's side `party` starts: `request` sent, the fill kept and each batch answered by
    `prover`, the side's report back, with `answered`, the batches the prover answered, and `connection`, the
    socket's `local` and `peer` addresses (`HOST:PORT`) on the prover's side."""
    p = subprocess.Popen(list(party.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        p.stdin.write(json.dumps(dict(request), sort_keys=True) + "\n")
        p.stdin.close()
        line = p.stdout.readline()
        if not line:
            raise RuntimeError(f"the challenger's side on {party.host} exited {p.wait()} before it listened")
        port = json.loads(line)["listening"]
        with socket.create_connection(party.address or ("127.0.0.1", port)) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection = {k: "%s:%d" % a[:2] for k, a in (("local", s.getsockname()), ("peer", s.getpeername()))}
            with s.makefile("rwb") as f:
                answered = answer(f, prover, stop_after)
        out = json.loads(p.stdout.readline())
    finally:
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"the challenger's side on {party.host} exited {rc}")
    return out | {"answered": answered, "connection": connection}


def main(argv: Sequence[str] | None = None) -> int:
    from . import period  # noqa: PLC0415  (its challenger key, made once in the device directory)

    ap = argparse.ArgumentParser(prog=MODULE, description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", required=True, help="the challenger's key and session log (`period.challenger`)")
    ap.add_argument("--listen", default="127.0.0.1:0", help="HOST:PORT to listen on (port 0: any)")
    ap.add_argument("--reads", type=int, default=READS,
                    help=f"the timed spot reads, which the certificate is computed at (default {READS})")
    ap.add_argument("--egress-bits", type=int, default=UNCOUNTED,
                    help="the network certifier's egress count over the session, in bits (default: uncounted)")
    ap.add_argument("--streamer", help="stream the fill natively (`wipe_stream`): a built wipe_stream, or `build` to "
                    "compile it into --device; gated before the side listens")
    ap.add_argument("--threads", type=int, help="the streamer's threads (default: every CPU of this host)")
    a = ap.parse_args(argv)
    request = json.loads(sys.stdin.readline())
    params_of(request, a.reads)
    device = Path(a.device)
    streamer, extra = None, {}
    if a.streamer:
        streamer = (wipe_stream.Streamer.build(device, threads=a.threads) if a.streamer == "build" else
                    wipe_stream.Streamer(Path(a.streamer), a.threads))
        extra["streamer"] = streamer.gate()
    signer = period.challenger(device, workers=len(os.sched_getaffinity(0)))
    if not (device / "sessions").exists():
        mc.SessionLog.create(device / "sessions")
    host, _, port = a.listen.rpartition(":")
    with socket.create_server((host, int(port))) as srv:
        print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
        conn, _ = srv.accept()
        t0 = time.time_ns()
        with conn:
            conn.setblocking(True)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            def native(key: bytes, n: int) -> None:
                extra["fill"] = streamer.stream(key, n, conn.fileno())

            with conn.makefile("rwb") as f:
                c = serve(f, request, signer=signer, log=mc.SessionLog(device / "sessions"), egress_bits=a.egress_bits,
                          reads=a.reads, fill=None if streamer is None else native)
            t1 = time.time_ns()
    print(json.dumps(report(c, signer.public_key, (t0, t1)) | extra, sort_keys=True), flush=True)
    return 0


__all__ = ["CHUNK", "PREMISE", "READS", "REQUEST_KEYS", "UNCOUNTED", "Party", "answer", "commitment_of", "link", "main",
           "params_of", "placement", "report", "request_of", "serve", "session", "signed_of", "utc"]

if __name__ == "__main__":
    sys.exit(main())
