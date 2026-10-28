"""The streamed wipe as two parties of the deployment layer (`verity.core.deploy`): the memory challenger
(`Challenger`) and the prover (`Prover`), the protocol `wipe_party` speaks as a byte stream, in messages.

~~~text
memory-challenger -> prover   Fill {first, count} + labels (private) ... FillDone; then per batch Read {indices} (coins); End
prover -> memory-challenger   FillAck once it holds the fill; per batch Answer, 64 bytes an index (private)
~~~

The challenger drives the same `streamed_wipe.WipeMemoryChallenger` as `wipe_party.serve`, in the same order: the
labels' secret and the session id from its coins (64 bytes, the secret first), the fill's start and end, each batch's
send and receipt on its clock, the coin secret drawn when the first batch is revealed, the session signed once it ends
with the egress its machine was given. On the same coins, clock readings and key it signs the same bytes as `serve`. Its
machine gives it `signing-key` (`capabilities["signer"]`, a `memory_challenger.Signer`, with `capabilities["log"]`, its
session log), its egress count (`capabilities["egress_bits"]`, `wipe_party.UNCOUNTED` when none) and the reads it times
(`capabilities["reads"]`, `wipe_party.READS` when none, as `wipe_party`'s `--reads`); the prover's gives it `store`
(`capabilities["store"]`, the prover a `WipeParams` makes: `streamed_wipe.Prover` unless given, `wipe_store.StoreProver`
on a node). The run's parameters are `wipe_party.request_of`'s request, which names no reads: the challenger chooses
them, never the prover. The challenger finishes with `wipe_party.report` (private: what it signs is witness), refused by
name, and unsigned, when the prover leaves (`closed`), sends a frame the wire refuses, or answers a batch with another
count of labels (`schema`); parameters that name the reads are refused `schema` before any fill. It streams the fill
`WINDOW` frames at a time, letting the prover take each window before the next, so neither side holds more of it than
that.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verity.core.deploy.manifest import Channel, Manifest, Role
from verity.core.deploy.party import Finish, Identity, Received, Record, Send, Timer
from verity.core.deploy.runtime import ProtocolDef
from verity.core.wire import Body, Int, List, Message, Phase, Refused, Schema

from . import streamed_wipe as sw
from . import wipe_party as WP

ID = "verity/pous/streamed-wipe"
VERSION = 1
CHALLENGER, PROVER = "memory-challenger", "prover"
WINDOW = 8
"""Fill frames sent before the challenger lets the prover take them."""
MAX_BATCH = 1 << 10
MANIFEST = Manifest(
    ID, VERSION, "verity.core.protocols.pous.wipe_party:request_of",
    parties=(Role(CHALLENGER, "auditor", needs=("clock", "coins", "signing-key")),
             Role(PROVER, "developer", per="server", needs=("store",))),
    channels=(Channel(CHALLENGER, PROVER, ("Fill", "FillDone", "Read", "End"), carries_coins=True),
              Channel(PROVER, CHALLENGER, ("FillAck", "Answer"))),
    messages=(Schema("Fill", 1, "private", {"first": Int(), "count": Int(1, WP.CHUNK)},
                     Body(max=WP.CHUNK * sw.LABEL_BYTES, unit=sw.LABEL_BYTES)),
              Schema("FillDone", 1, "structure"), Schema("FillAck", 1, "structure"),
              Schema("Read", 1, "coins", {"indices": List(Int(), MAX_BATCH)}),
              Schema("Answer", 1, "private", body=Body(max=MAX_BATCH * sw.LABEL_BYTES, unit=sw.LABEL_BYTES)),
              Schema("End", 1, "structure")))


def _read() -> Send:
    return Send("clock", Message("ClockRead"))


class Challenger:
    """The memory challenger: one wipe, on its clock and coins (module docstring). `chunk` labels a fill frame."""

    def __init__(self, chunk: int = WP.CHUNK) -> None:
        self.chunk = chunk
        self.c: sw.WipeMemoryChallenger | None = None
        self._reading: int | None = None
        self._then: Callable[[], list[Any]] | None = None

    def start(self, params: dict[str, Any], identity: Identity) -> list[Any]:
        caps = identity.capabilities
        self.signer, self.log = caps["signer"], caps["log"]
        self.egress_bits = caps.get("egress_bits", WP.UNCOUNTED)
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
            return [Record({"commitment": self.c.commitment.to_json()}, "structure"),
                    *self._after_reading(self._fill_start)]
        if m.type == "FillAck":
            self.phase.expect("Coins")
            return [Send("coins", Message("CoinsDraw", {"n": 32}))]
        if m.type == "Coins":
            # The coin secret `serve` draws from the OS on revealing the first batch, drawn from this party's coins.
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
        self.c.close(self.egress_bits)
        return [Send(PROVER, Message("End")), *self._finish(None)]

    def _finish(self, refused: str | None) -> list[Any]:
        if self.c is None:
            return [Finish({"ended": None, "signed": None}, refused or "closed", kind="private")]
        return [Finish(WP.report(self.c, self.signer.public_key), refused, kind="private")]


class Prover:
    """The prover: it keeps the fill in its store (`capabilities["store"]`) and answers each batch from it."""

    def start(self, params: dict[str, Any], identity: Identity) -> list[Any]:
        self.answered = 0
        try:
            self.store = identity.capabilities.get("store", sw.Prover)(WP.params_of(params))
        except ValueError:
            return [Finish({"answered": 0}, "schema")]
        self.phase = Phase("Fill", "FillDone")
        return []

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Refused", "Closed"):
            return [Finish({"answered": self.answered}, m.fields["name"] if m.type == "Refused" else "closed")]
        try:
            self.phase.check(m)
        except Refused as e:
            return [Finish({"answered": self.answered}, e.name)]
        if m.type == "Fill":
            self.store.receive(m.fields["first"], m.body)
            return []
        if m.type == "FillDone":
            self.phase.expect("Read")
            return [Send(CHALLENGER, Message("FillAck"))]
        if m.type == "Read":
            self.phase.expect("Read", "End")
            self.answered += 1
            return [Send(CHALLENGER, Message("Answer", body=self.store.answer(m.fields["indices"])))]
        return [Finish({"answered": self.answered})]


PROTOCOL = ProtocolDef(MANIFEST, {CHALLENGER: Challenger, PROVER: Prover})


__all__ = ["CHALLENGER", "ID", "MANIFEST", "PROTOCOL", "PROVER", "VERSION", "WINDOW", "Challenger", "Prover"]
