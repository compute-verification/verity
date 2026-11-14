"""The streamed wipe's memory challenger as a party of the deployment layer (`verity.runtime`), on the challenger host:
`MemoryChallengerParty`, over the messages `verity.protocols.accounting.pous.verifier.wipe_parties` declares.

The challenger drives the same `streamed_wipe.WipeMemoryChallenger` as `wipe_party.serve`, in the same order: the
labels' secret and the session id from its coins (64 bytes, the secret first), the fill's start and end, each batch's
send and receipt on its clock, the coin secret sampled when the first batch is revealed, the session signed once it
ends. It is given no egress count and signs none. On the same coins, clock readings and key it signs the same bytes as
`serve`. Its header names the raw monotonic clock (`streamed_wipe.CLOCK`), so the clock its machine gives it is to read
that one, `sources.RealClock(streamed_wipe.raw_clock_ns)`: the runtime's own real clock reads `time.monotonic_ns`,
which time synchronisation slews. Its machine gives it `signing-key` (`capabilities["signer"]`, a
`memory_challenger.Signer`, with `capabilities["log"]`, its session log) and the reads it times
(`capabilities["reads"]`, `wipe_party.READS` when none, as `wipe_party`'s `--reads`). The run's parameters are
`wipe_party.request_of`'s request, which names no reads: the challenger chooses them, never the prover. The challenger
finishes with `wipe_party.report` (private: what it signs is witness), refused by name, and unsigned, when the prover
leaves (`closed`), sends a frame the wire refuses, or answers a batch with another count of labels (`schema`);
parameters that name the reads are refused `schema` before any fill. It streams the fill `WINDOW` frames at a time,
letting the prover take each window before the next, so neither side holds more of it than that. `PROTOCOL` declares
its one role.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verity.party import Finish, Identity, Received, LogEntry, Send, Timer
from verity.protocol import ProtocolDef
from verity.channel import Message, Phase, Refused
from verity.protocols.accounting.pous.verifier import streamed_wipe as sw
from verity.protocols.accounting.pous.verifier import wipe_party as WP
from verity.protocols.accounting.pous.verifier.wipe_parties import CHALLENGER, MANIFEST, PROVER

WINDOW = 8
"""Fill frames sent before the challenger lets the prover take them."""


def _read() -> Send:
    return Send("clock", Message("ClockRead"))


class MemoryChallengerParty:
    """The memory challenger: one wipe, on its clock and coins (module docstring). `chunk` labels a fill frame."""

    def __init__(self, chunk: int = WP.CHUNK) -> None:
        self.chunk = chunk
        self.c: sw.WipeMemoryChallenger | None = None
        self._reading: int | None = None
        self._then: Callable[[], list[Any]] | None = None

    def start(self, params: dict[str, Any], identity: Identity) -> list[Any]:
        caps = identity.capabilities
        self.signer, self.log = caps["signer"], caps["log"]
        try:
            self.request, self.wp = dict(params), WP.params_of(params, caps.get("reads", WP.READS))
        except ValueError:
            return self._finish("schema")
        self.phase = Phase("Coins")
        return [Send("coins", Message("CoinsDraw", {"n": 64}))]

    def _clock(self) -> int:
        """The reading the challenger asked its clock for, used once: `streamed_wipe.WipeMemoryChallenger` reads it
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
            self.c = sw.WipeMemoryChallenger(secret, self.request["node"].encode(), session, self.wp, self.signer,
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
        return [Finish(WP.report(self.c, self.signer.public_key), refused, kind="private")]


PROTOCOL = ProtocolDef(MANIFEST, {CHALLENGER: MemoryChallengerParty})


__all__ = ["PROTOCOL", "WINDOW", "MemoryChallengerParty"]
