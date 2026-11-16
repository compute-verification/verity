"""The streamed wipe's memory challenger, on the challenger host: its side of one wipe (`WipeMemoryChallenger`, the
steps `streamed_wipe`'s docstring states), that side as a process of its own over `wipe_party`'s wire (`serve`, `report`,
`main`), and as a party of the deployment layer (`verity.runtime`), `MemoryChallengerParty`, over the messages
`verity.protocols.resource_accounting.memory.pous.verifier.wipe_parties` declares.

`python -m verity.protocols.resource_accounting.memory.pous.attester.wipe_challenger --device DIR [--listen HOST:PORT] [--reads K]
[--streamer build|PATH [--threads T]]` is the challenger's side (`main`, which `wipe_party.command` starts): it reads the
request `{"labels", "limit_ns", "node", "tiers"}` as one JSON line on stdin, prints `{"listening": PORT}`, serves one
prover (`serve`) and prints its report (`report`). With `--streamer` the fill is `wipe_stream`'s, native on this host's
CPUs and gated before the side listens, and the report adds the streamer's gate (`streamer`) and its figures for the
fill (`fill`).

The party drives the same `WipeMemoryChallenger` as `serve`, in the same order: the
labels' secret and the session id from its coins (64 bytes, the secret first), the fill's start and end, each batch's
send and receipt on its clock, the coin secret sampled when the first batch is revealed, the session signed once it
ends. It is given no egress count and signs none. On the same coins, clock readings and key it signs the same bytes as
`serve`. Its header names the raw monotonic clock (`streamed_wipe.CLOCK`), so it refuses by name (`clock`), before any
coin and unsigned, a clock source that reads another real clock (`Identity.clock`): the runtime's own real clock reads
`CLOCK_MONOTONIC`, which time synchronisation slews, and the run that starts it hands it
`RealClock(streamed_wipe.raw_clock_ns, streamed_wipe.CLOCK)` instead. Virtual time (a simulated or seeded run) and a
replay name no clock, and it signs their readings under the header's name, as `WipeMemoryChallenger` signs
a clock a test injects. Its machine gives it `signing-key` (`capabilities["signer"]`, a
`memory_challenger.Signer`, with `capabilities["log"]`, its session log) and the reads it times
(`capabilities["reads"]`, `wipe_party.READS` when none, as `wipe_party`'s `--reads`). The run's parameters are
`wipe_party.request_of`'s request, which names no reads: the challenger chooses them, never the prover. The challenger
finishes with `report` (private: what it signs is witness), refused by name, and unsigned, when the prover
leaves (`closed`), sends a frame the wire refuses, or answers a batch with another count of labels (`schema`);
parameters that name the reads are refused `schema` before any fill. It streams the fill `WINDOW` frames at a time,
letting the prover take each window before the next, so neither side holds more of it than that. `PROTOCOL` declares
its one role, and `start` how a deployment starts it on the auditor's device: first, in its state directory, its key,
refused unless the configuration pins it for the wipe (`onsite.signer`), then its session log and the wipe's request
(`wipe_parties.deployed_request`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import socket
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from verity.party import Finish, Identity, Received, LogEntry, Send, Timer
from verity.protocol import ProtocolDef, Start
from verity.channel import Message, Phase, Refused
from verity.protocols.resource_accounting.memory.pous.attester import memory_challenger as mc
from verity.protocols.resource_accounting.memory.pous.attester import onsite
from verity.protocols.resource_accounting.memory.pous.attester import wipe_stream
from verity.protocols.resource_accounting.memory.pous.attester.memory_challenger import CLOCK_RESOLUTION_NS, SessionLog, Signer
from verity.protocols.resource_accounting.memory.pous.verifier import streamed_wipe as sw
from verity.protocols.resource_accounting.memory.pous.verifier.streamed_wipe import (
    COMPLETE,
    LABEL_BYTES,
    LATE,
    WRONG,
    BatchRecord,
    SignedWipe,
    WipeCommitment,
    WipeParams,
    _sha512,
    batch_digest,
    sample_indices,
    label,
    labels,
    message,
    raw_clock_ns,
    start_message,
    wipe_key,
)
from verity.protocols.resource_accounting.memory.pous.verifier import wipe_party as WP
from verity.protocols.resource_accounting.memory.pous.verifier.wipe_parties import (
    CHALLENGER,
    COMPONENT,
    MANIFEST,
    PROVER,
    deployed_request,
)

WINDOW = 8
"""Fill frames sent before the challenger lets the prover take them."""
logger = logging.getLogger(__name__)


class WipeMemoryChallenger:
    """The memory challenger's side of one streamed-label wipe of `server` under `session`, signed by `signer` (module
    docstring).  The session id goes into `log` before anything else.  `coin_secret` is sampled from the OS when the fill
    completes unless a test gives it.  `mark_source` takes the start message and returns the network certifier's start certificate for it, as bytes;
    with none the session's start mark is empty.  `clock` is `raw_clock_ns` unless given.  The header names `CLOCK`
    whatever `clock` reads, so a `clock` given outside a test reads the raw clock too."""

    def __init__(self, secret: bytes, server: bytes, session: bytes, params: WipeParams, signer: Signer, *,
                 log: SessionLog, clock: Callable[[], int] = raw_clock_ns, resolution_ns: int | None = None,
                 coin_secret: bytes | None = None, mark_source: Callable[[bytes], bytes] | None = None) -> None:
        if resolution_ns is None:
            resolution_ns = time.clock_getres(time.CLOCK_MONOTONIC_RAW) * 1e9 if clock is raw_clock_ns else 1
        if resolution_ns > CLOCK_RESOLUTION_NS:
            raise ValueError(f"the clock ticks every {resolution_ns} ns, coarser than {CLOCK_RESOLUTION_NS} ns")
        self.key = wipe_key(secret, server, session)
        log.claim(session)
        self.commitment = WipeCommitment(server, session, params)
        self.signer, self.clock, self._coin_secret = signer, clock, coin_secret
        self._mark_source = mark_source
        self.start_certificate = b""
        self.start_mark = b""
        self.filled = 0
        self.fill_start_ns: int | None = None
        self.fill_end_ns: int | None = None
        self.indices: tuple[int, ...] | None = None
        self.batches: list[BatchRecord] = []
        self.ended: str | None = None
        self.signed: SignedWipe | None = None
        self._open: tuple[int, int, tuple[int, ...], int] | None = None

    @property
    def params(self) -> WipeParams:
        return self.commitment.params

    def _hold_start_mark(self) -> None:
        """With a mark source: hand it the start message, keep the start certificate's bytes it returns, and set the
        start mark to their SHA-512.  The bytes are not verified.  A source that returns none raises."""
        if self._mark_source is None:
            return
        certificate = self._mark_source(start_message(self.commitment))
        if not isinstance(certificate, (bytes, bytearray)) or not certificate:
            raise ValueError("the mark source returned no start certificate: the fill waits for its bytes")
        self.start_certificate = bytes(certificate)
        self.start_mark = _sha512(self.start_certificate)

    def fill(self, chunk: int) -> Iterator[tuple[int, bytes]]:
        """The labels in order, `chunk` at a time, as `(first index, bytes)`; once only.  The start mark is held
        before the clock is read, so before any label."""
        if self.fill_start_ns is not None:
            raise RuntimeError("the fill streams once")
        if chunk < 1:
            raise ValueError("a chunk holds at least one label")
        self._hold_start_mark()
        self.fill_start_ns = self.clock()
        n = self.params.labels
        for start in range(0, n, chunk):
            count = min(chunk, n - start)
            yield start, labels(self.key, start, count)
            self.filled = start + count
        self.fill_end_ns = self.clock()

    def fill_by(self, send: Callable[[bytes, int], Any]) -> Any:
        """`fill`'s labels streamed by `send(K, N)`, which returns once it has sent the last; once only. Returns what
        `send` returns. The answers are checked against this module's `label` whatever `send` streamed.  The start mark
        is held before the clock is read, so before `send` is called."""
        if self.fill_start_ns is not None:
            raise RuntimeError("the fill streams once")
        self._hold_start_mark()
        self.fill_start_ns = self.clock()
        out = send(self.key, self.params.labels)
        self.filled = self.params.labels
        self.fill_end_ns = self.clock()
        return out

    def challenge(self) -> tuple[int, ...]:
        """Reveal the next batch's indices and read its send time; the coins are flipped at the first batch."""
        if self.fill_end_ns is None:
            raise RuntimeError("the challenges start once the fill is complete")
        if self.ended is not None:
            raise RuntimeError(f"the session has ended ({self.ended}): no batch is retried and no index redrawn")
        if self._open is not None:
            raise RuntimeError("the previous batch is unanswered")
        if self.indices is None:
            secret = self._coin_secret if self._coin_secret is not None else os.urandom(32)
            c = self.commitment
            self.indices = sample_indices(secret, c.server, c.session, self.params.labels, self.params.k)
        b = len(self.batches)
        first = b * self.params.batch
        idx = self.indices[first:first + self.params.batch]
        self._open = (b, first, idx, self.clock())
        return idx

    def receive(self, answers: bytes) -> BatchRecord:
        """The open batch's answers, `LABEL_BYTES` an index in order: the receipt time first, then the check."""
        receipt = self.clock()
        if self._open is None:
            raise RuntimeError("no batch is open")
        b, first, idx, send = self._open
        if len(answers) != LABEL_BYTES * len(idx):
            raise ValueError(f"a batch of {len(idx)} is answered with {LABEL_BYTES * len(idx)} bytes, not {len(answers)}")
        self._open = None
        wrong = sum(answers[LABEL_BYTES * t:LABEL_BYTES * (t + 1)] != label(self.key, i) for t, i in enumerate(idx))
        r = BatchRecord(b, first, len(idx), send, receipt, wrong, batch_digest(idx, answers))
        self.batches.append(r)
        if not 0 <= r.elapsed_ns <= self.params.limit_ns:
            self.ended = LATE
        elif wrong:
            self.ended = WRONG
        elif first + len(idx) == self.params.k:
            self.ended = COMPLETE
        return r

    def run(self, respond: Callable[[tuple[int, ...]], bytes]) -> None:
        """Drive the challenges against `respond(indices)` until the session ends."""
        while self.ended is None:
            self.receive(respond(self.challenge()))

    def close(self) -> SignedWipe:
        """Sign the ended session, once."""
        if self.ended is None:
            raise RuntimeError("the session has not ended")
        if self.signed is None:
            assert self.fill_start_ns is not None and self.fill_end_ns is not None
            batches = tuple(self.batches)
            m = message(self.commitment, self.ended, self.start_mark, self.fill_start_ns, self.fill_end_ns, batches)
            self.signed = SignedWipe(self.commitment, self.ended, self.start_mark, self.fill_start_ns, self.fill_end_ns,
                                     batches, self.signer.scheme, self.signer.public_key,
                                     self.signer.sign(self.commitment.session, m))
        return self.signed

    def coins(self) -> tuple[int, ...]:
        """The sampled indices, public once the session is signed (the verifier's samples)."""
        if self.signed is None or self.indices is None:
            raise RuntimeError("the coins are published after the session, once it is signed")
        return self.indices


def serve(f: BinaryIO, request: Mapping[str, Any], *, signer: mc.Signer, log: mc.SessionLog,
          reads: int = WP.READS, clock: Callable[[], int] = sw.raw_clock_ns, chunk: int = WP.CHUNK,
          fill: Callable[[bytes, int], Any] | None = None) -> WipeMemoryChallenger:
    """The challenger's side of one wipe over the connection `f`: the labels' secret and the session id sampled here, the
    fill streamed (by `fill(K, N)` when given, which writes the frames to `f`'s connection itself: the native streamer,
    `wipe_stream`), `reads` batches revealed and their answers read on `clock` (the raw monotonic clock,
    `sw.raw_clock_ns`, which the header names), the session signed by `signer` once it ends. A prover that closes first
    leaves it unsigned. This side has no mark source, so the session's start mark is empty."""
    c = WipeMemoryChallenger(secrets.token_bytes(32), request["node"].encode(), secrets.token_bytes(32),
                                WP.params_of(request, reads), signer, log=log, clock=clock)
    try:
        if fill is not None:
            f.flush()
            c.fill_by(fill)
        else:
            for first, data in c.fill(chunk):
                f.write(WP.FILL + first.to_bytes(8, "big") + (len(data) // sw.LABEL_BYTES).to_bytes(4, "big") + data)
        f.write(WP.DONE)
        f.flush()
        if WP._read(f, 1) != WP.ACK:
            raise ConnectionError("the prover didn't acknowledge the fill")
        while c.ended is None:
            idx = c.challenge()
            f.write(WP.CHALLENGE + len(idx).to_bytes(4, "big") + b"".join(i.to_bytes(8, "big") for i in idx))
            f.flush()
            c.receive(WP._read(f, sw.LABEL_BYTES * len(idx)))
        f.write(WP.END)
        f.flush()
    except (ConnectionError, BrokenPipeError) as e:
        logger.warning("the prover left before the wipe ended, so its session stays unsigned: %s", e)
        return c
    c.close()
    return c


def report(c: WipeMemoryChallenger, public_key: bytes, window: tuple[int, int] | None = None) -> dict[str, Any]:
    """The side's report; `window`, the session's `(t0, t1)` in Unix nanoseconds on this host's real-time clock."""
    out = {"commitment": c.commitment.to_json(), "ended": c.ended, "host": socket.gethostname(),
           "public_key": public_key.hex(), "reads": len(c.batches),
           "signed": None if c.signed is None else c.signed.to_json()}
    if window is not None:
        out["window"] = {"t0": WP.utc(window[0]), "t1": WP.utc(window[1]), "t0_unix_ns": window[0], "t1_unix_ns": window[1],
                         "clock": f"{out['host']}'s real-time clock"}
    return out


def _read() -> Send:
    return Send("clock", Message("ClockRead"))


class MemoryChallengerParty:
    """The memory challenger: one wipe, on its clock and coins (module docstring). `chunk` labels a fill frame."""

    def __init__(self, chunk: int = WP.CHUNK) -> None:
        self.chunk = chunk
        self.c: WipeMemoryChallenger | None = None
        self._reading: int | None = None
        self._then: Callable[[], list[Any]] | None = None

    def start(self, params: dict[str, Any], identity: Identity) -> list[Any]:
        given = identity.capabilities
        self.signer, self.log = given["signer"], given["log"]
        if identity.clock not in (None, sw.CLOCK):
            return [Finish({"ended": None, "signed": None, "detail": (
                f"the wipe's header names {sw.CLOCK}, and this challenger's clock source reads {identity.clock}: the run "
                f"that starts it gives it sources={{{CHALLENGER!r}: {{'clock': RealClock(streamed_wipe.raw_clock_ns, "
                "streamed_wipe.CLOCK)}}")}, "clock", kind="private")]
        try:
            self.request, self.wp = dict(params), WP.params_of(params, given.get("reads", WP.READS))
        except ValueError:
            return self._finish("schema")
        self.phase = Phase("Coins")
        return [Send("coins", Message("CoinsDraw", {"n": 64}))]

    def _clock(self) -> int:
        """The reading the challenger asked its clock for, used once: `WipeMemoryChallenger` reads it
        here."""
        if self._reading is None:
            raise RuntimeError("the wipe read its clock without a fresh reading")
        out, self._reading = self._reading, None
        return out

    def _after_reading(self, then: Callable[[], list[Any]]) -> list[Any]:
        self._then = then
        return [_read()]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type == "ClockReading":
            self._reading, then, self._then = m.fields["ns"], self._then, None
            return then()
        if m.type == "ClockTimer":
            return self._stream()
        if m.type in ("Refused", "Closed"):
            return self._finish(m.fields["name"] if m.type == "Refused" else "closed")
        try:
            self.phase.check(m)
        except Refused as e:
            return self._finish(e.name)
        if m.type == "Coins" and self.c is None:
            secret, session = m.body[:32], m.body[32:]
            self.c = WipeMemoryChallenger(secret, self.request["node"].encode(), session, self.wp, self.signer,
                                             log=self.log, clock=self._clock)
            self.phase.expect()
            return [LogEntry({"commitment": self.c.commitment.to_json()}, "structure"),
                    *self._after_reading(self._fill_start)]
        if m.type == "FillAck":
            self.phase.expect("Coins")
            return [Send("coins", Message("CoinsDraw", {"n": 32}))]
        if m.type == "Coins":
            # The coin secret `serve` samples from the OS on revealing the first batch, sampled from this party's coins.
            self.c._coin_secret = m.body
            self.phase.expect()
            return self._after_reading(self._challenge)
        if len(m.body) != sw.LABEL_BYTES * len(self._open):
            return self._finish("schema")
        self._answers = m.body
        self.phase.expect()
        return self._after_reading(self._receive)

    def _fill_start(self) -> list[Any]:
        self._fill = self.c.fill(self.chunk)
        self._next = next(self._fill)
        return self._stream()

    def _stream(self) -> list[Any]:
        out: list[Any] = []
        while self._next is not None and len(out) < WINDOW:
            first, data = self._next
            count = len(data) // sw.LABEL_BYTES
            out.append(Send(PROVER, Message("Fill", {"first": first, "count": count}, data)))
            if first + count == self.wp.labels:
                self._next = None
                return out + self._after_reading(self._fill_end)
            self._next = next(self._fill)
        return out + [Timer(0)]

    def _fill_end(self) -> list[Any]:
        for _ in self._fill:
            raise RuntimeError("the fill streamed past its last label")
        self.phase.expect("FillAck")
        return [Send(PROVER, Message("FillDone"))]

    def _challenge(self) -> list[Any]:
        self._open = self.c.challenge()
        self.phase.expect("Answer")
        return [Send(PROVER, Message("Read", {"indices": list(self._open)}))]

    def _receive(self) -> list[Any]:
        self.c.receive(self._answers)
        if self.c.ended is None:
            return self._after_reading(self._challenge)
        self.c.close()
        return [Send(PROVER, Message("End")), *self._finish(None)]

    def _finish(self, refused: str | None) -> list[Any]:
        if self.c is None:
            return [Finish({"ended": None, "signed": None}, refused or "closed", kind="private")]
        return [Finish(report(self.c, self.signer.public_key), refused, kind="private")]


def start(configuration: Mapping[str, Any], bound: Mapping[str, Any], state: Path | None) -> Start:
    """The challenger's start on the auditor's device (`ProtocolDef.starts`, module docstring)."""
    signer = onsite.signer(state, onsite.pin(configuration, CHALLENGER, COMPONENT))
    if not (state / "sessions").exists():
        mc.SessionLog.create(state / "sessions")
    log = mc.SessionLog(state / "sessions")
    return Start(deployed_request(configuration, bound), {"signer": signer, "log": log, "reads": WP.READS})


PROTOCOL = ProtocolDef(MANIFEST, {CHALLENGER: MemoryChallengerParty}, {CHALLENGER: start})


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=WP.MODULE, description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", required=True, help="the challenger's key and session log (`onsite.device_key`)")
    ap.add_argument("--listen", default="127.0.0.1:0", help="HOST:PORT to listen on (port 0: any)")
    ap.add_argument("--reads", type=int, default=WP.READS,
                    help=f"the timed spot reads, which the certificate is computed at (default {WP.READS})")
    ap.add_argument("--streamer", help="stream the fill natively (`wipe_stream`): a built wipe_stream, or `build` to "
                    "compile it into --device; gated before the side listens")
    ap.add_argument("--threads", type=int, help="the streamer's threads (default: every CPU of this host)")
    a = ap.parse_args(argv)
    request = json.loads(sys.stdin.readline())
    WP.params_of(request, a.reads)
    device = Path(a.device)
    streamer, extra = None, {}
    if a.streamer:
        streamer = (wipe_stream.Streamer.build(device, threads=a.threads) if a.streamer == "build" else
                    wipe_stream.Streamer(Path(a.streamer), a.threads))
        extra["streamer"] = streamer.gate()
    signer = onsite.device_key(device, workers=len(os.sched_getaffinity(0)))
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
                c = serve(f, request, signer=signer, log=mc.SessionLog(device / "sessions"), reads=a.reads,
                          fill=None if streamer is None else native)
            t1 = time.time_ns()
    print(json.dumps(report(c, signer.public_key, (t0, t1)) | extra, sort_keys=True), flush=True)
    return 0


__all__ = ["PROTOCOL", "WINDOW", "MemoryChallengerParty", "WipeMemoryChallenger", "main", "report", "serve", "start"]

if __name__ == "__main__":
    sys.exit(main())
