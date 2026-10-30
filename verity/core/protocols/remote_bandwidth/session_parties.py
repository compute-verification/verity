"""The remote bandwidth session (`period.session`) as two parties of the deployment layer (`verity.core.deploy`): the
verifier and its challenger (`Verifier`), on a machine the prover doesn't control, and the prover (`Prover`) on its node.

~~~text
prover   -> verifier   Register {blocks} + payload root; per pass Root {t, i} + root; per window Openings {t} + openings
verifier -> prover     Hold {registration}; per pass Run {t, i, challenge} + the root before (the challenge a
                       `coins.Released`); per window Open {t} + draws (u32be pass ‖ u64be position each); End + the
                       session's coins disclosed once it has its outcome ({commitment, openings}, `audit.AuditRun.disclose`)
~~~

The verifier's session is `period.session`'s, so its case is one `RemoteAudit.Protocol` replays. Its coins source gives
the period's live secret and the registration's nonce, drawn once. Its clock times each event of the log:

- a root's receive at the root's arrival (`Received.at`), which `audit.AuditRun.receive` holds to its window's deadline,
  and the registration's `not_before_ns` at the payload root's;
- the start at a reading of its clock;
- each boundary at the reading its timer fires at, `next_boundary_ns`, whether or not the window's roots are in: the
  pass in flight then is its window's last, and the next window's first is asked once its root is in (`audit` §4);
- each close at its timer too, at the window's deadline `D_t + 1`, or at a reading for a window the next window's
  first root made final.

The verifier finishes with `{"answer", "case"}` (private: the case holds the openings) once it has recorded the answer
(`period.answer`, as `RemoteAudit.Protocol.check` gives it): after the last window's verdict, or at the first rejection,
with the case of the events so far. A prover that leaves, a frame the wire refuses or one out of phase ends it refused
by name, with the answer as it stands. The prover's machine gives it its store (`capabilities["store"]`, a
`period.Prover`: `period.Local` is the reference); it finishes on `End`, refused `coins` when a challenge it was sent
doesn't open against the coins disclosed (`period.Prover.end`).
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from typing import Any

from verity.core.deploy.manifest import Channel, Manifest, Role
from verity.core.deploy.party import Finish, Identity, Received, Record, Send, Timer
from verity.core.deploy.runtime import ProtocolDef
from verity.core.wire import Body, Int, Json, Message, Phase, Refused, Schema
from verity.ml.commitments.identity import canonical_json_bytes

from . import audit, receipts
from . import coins as C
from . import period as P
from . import wire as W

ID = "verity/remote-bandwidth/session"
VERSION = 1
VERIFIER, PROVER = "verifier", "prover"
SECRET_BYTES = 64
"""The period's live secret, which the registration's codec seed and the challenger's coins are drawn from."""
DRAW = struct.Struct(">IQ")
MAX_DRAWS = 1 << 20
MAX_DISCLOSED = 1 << 24
ROOT = Body(size=receipts.DIGEST_BYTES, max=receipts.DIGEST_BYTES)
MANIFEST = Manifest(
    ID, VERSION, "verity.core.protocols.remote_bandwidth.session_parties:request_of",
    parties=(Role(VERIFIER, "auditor", needs=("clock", "coins")), Role(PROVER, "developer", per="server", needs=("store",))),
    channels=(Channel(VERIFIER, PROVER, ("Hold", "Run", "Open", "End"), carries_coins=True),
              Channel(PROVER, VERIFIER, ("Register", "Root", "Openings"))),
    messages=(Schema("Register", 1, "commitment", {"blocks": Int(1)}, ROOT),
              Schema("Hold", 1, "coins", {"registration": Json()}),
              Schema("Run", 1, "coins", {"t": Int(), "i": Int(), "challenge": Json(1 << 14)}, ROOT),
              Schema("Root", 1, "commitment", {"t": Int(), "i": Int()}, ROOT),
              Schema("Open", 1, "coins", {"t": Int()}, Body(max=MAX_DRAWS * DRAW.size, unit=DRAW.size)),
              Schema("Openings", 1, "private", {"t": Int()}, Body(max=(1 << 32) - 1)),
              Schema("End", 1, "coins", body=Body(max=MAX_DISCLOSED))))


def request_of(node: str, *, windows: int = 2, passes: int = 1, samples: int = audit.SAMPLES,
               window_ns: int = 100_000_000, deadline_ns: int = 500_000_000) -> dict[str, Any]:
    """The session's request, its parameters: `period.run`'s defaults unless given."""
    return {"deadline_ns": deadline_ns, "node": node, "passes": passes, "samples": samples, "window_ns": window_ns,
            "windows": windows}


def _arrival(r: Received) -> int:
    if r.at is None:
        raise RuntimeError(f"a frame on {r.channel} reached the verifier without its arrival on its clock")
    return r.at


class Verifier:
    """The verifier and its challenger: one session of the period, on its clock and coins (module docstring)."""

    def start(self, params: dict[str, Any], identity: Identity) -> list[Any]:
        self.request = dict(params)
        self.s: audit.AuditRun | None = None
        self.held, self.events = P._Held(), []
        self._reading: Callable[[int], list[Any]] | None = None
        self._timer: tuple[int, Callable[[int], list[Any]]] | None = None
        self.window = self.asked = self.closing = 0
        self.in_flight: tuple[int, int] | None = None
        self.opening: int | None = None
        self.step = -1
        self.coins: bytes | None = None
        self._early: tuple[Message, int] | None = None
        return [Send("coins", Message("CoinsDraw", {"n": SECRET_BYTES + audit.NONCE_BYTES}))]

    def on(self, r: Received) -> list[Any]:
        """A source's answer, or the prover's next frame: the one the session waits for, or it ends `out-of-phase`
        (a registration that beats the coins waits for them)."""
        m = r.message
        if m.type == "Coins":
            self.coins, self.nonce = m.body[:SECRET_BYTES], m.body[SECRET_BYTES:]
            early, self._early = self._early, None
            return [] if early is None else self._register(*early)
        if m.type == "ClockReading":
            then, self._reading = self._reading, None
            return then(m.fields["ns"])
        if m.type == "ClockTimer":
            if self._timer is None or self._timer[0] != m.fields["at"]:
                return []
            (_, then), self._timer = self._timer, None
            return then(m.fields["ns"])
        if m.type in ("Refused", "Closed"):
            return self._finish(m.fields["name"] if m.type == "Refused" else "closed")
        if m.type == "Register" and self.s is None and self._early is None:
            if self.coins is None:
                self._early = (m, _arrival(r))
                return []
            return self._register(m, _arrival(r))
        if m.type == "Root" and self.in_flight == (m.fields["t"], m.fields["i"]):
            return self._receive(m.fields["t"], m.fields["i"], m.body, _arrival(r))
        if m.type == "Openings" and self.opening is not None and m.fields["t"] == self.opening:
            return self._verify(m.body, _arrival(r))
        return self._finish("out-of-phase")

    def _read(self, then: Callable[[int], list[Any]]) -> list[Any]:
        self._reading = then
        return [Send("clock", Message("ClockRead"))]

    def _alarm(self, at: int, then: Callable[[int], list[Any]]) -> list[Any]:
        self._timer = (at, then)
        return [Timer(at)]

    def _event(self, event: dict[str, Any], now: int) -> None:
        self.held.now = now
        self.events.append(event | ({} if event["kind"] == "verify" else {"now": now}))

    def _register(self, m: Message, at: int) -> list[Any]:
        q = self.request
        try:
            reg = audit.register(self.coins, audit.site_log(q["node"].encode(), m.fields["blocks"], m.body),
                                 audit.STORE, nonce=self.nonce)
        except ValueError:
            return self._finish("schema")
        self.s = P.opened(reg, self.coins, self.held, not_before=at, windows=q["windows"], passes=q["passes"],
                          samples=q["samples"], window_ns=q["window_ns"], deadline_ns=q["deadline_ns"])
        self.previous = reg.digest
        return [Send(PROVER, Message("Hold", {"registration": P.registration_json(reg)})), *self._read(self._start)]

    def _start(self, now: int) -> list[Any]:
        self._event({"kind": "start"}, now)
        self.challenge = self.s.start()
        return self._alarm(self.s.next_boundary_ns(), self._boundary) + self._ask()

    def _ask(self) -> list[Any]:
        """The next pass, once the root before it is in: none past the window's passes, or while its boundary is
        pending after them."""
        if self.in_flight is not None or self.window >= self.s.params.windows or self.asked >= self.request["passes"]:
            return []
        t, i = self.in_flight = (self.window, self.asked)
        self.asked += 1
        challenge = C.Released.of(self.challenge).to_json()
        return [Send(PROVER, Message("Run", {"t": t, "i": i, "challenge": challenge}, self.previous))]

    def _receive(self, t: int, i: int, root: bytes, now: int) -> list[Any]:
        self.in_flight, self.previous, self.step = None, root, self.step + 1
        self._event({"kind": "receive", "t": t, "i": i, "root": root.hex(), "runs": None, "change": None,
                     "step": self.step}, now)
        if self.s.receive(t, i, root, None, change=None, step=self.step) is not None:
            return self._finish()
        return self._ask()

    def _boundary(self, now: int) -> list[Any]:
        self._event({"kind": "boundary"}, now)
        challenge = self.s.boundary()
        self.window, self.asked = self.window + 1, 0
        if challenge is None:
            return self._close()
        self.challenge = challenge
        return self._alarm(self.s.next_boundary_ns(), self._boundary) + self._ask()

    def _close(self) -> list[Any]:
        t = self.closing
        if t == self.s.params.windows:
            return self._finish()
        if self.s.windows[t].final:
            return self._read(self._closed)
        return self._alarm(self.s.deadline(t) + 1, self._closed)

    def _closed(self, now: int) -> list[Any]:
        t = self.closing
        self._event({"kind": "close", "t": t}, now)
        if isinstance(self.s.close(t), audit.Verdict):
            return self._finish()
        self.opening = t
        draws = b"".join(DRAW.pack(i, p) for i, p in self.s.windows[t].draws)
        return [Send(PROVER, Message("Open", {"t": t}, draws))]

    def _verify(self, body: bytes, now: int) -> list[Any]:
        t, self.opening = self.opening, None
        got = W.decode_openings(body) or []
        self._event({"kind": "verify", "t": t, "openings": [P.opening_json(o) for o in got]}, now)
        if not self.s.verify(t, got).accepted:
            return self._finish()
        self.closing += 1
        return self._close()

    def _finish(self, refused: str | None = None) -> list[Any]:
        if self.s is None:
            return [Finish({"answer": None, "case": None}, refused or "closed", kind="private")]
        answer = P.answer(self.s)
        out = [Record(answer, "output"), Finish({"answer": answer, "case": P.case_of(self.s, self.events)}, refused,
                                                kind="private")]
        return out if refused is not None else [Send(PROVER, Message("End", body=self.disclosed())), *out]

    def disclosed(self) -> bytes:
        """`End`'s body: the session's coin commitment and every released coin's opening once it has its outcome
        (`audit.AuditRun.disclose`), as canonical JSON; empty for a session left undecided."""
        if self.s.outcome is None:
            return b""
        commitment, openings = self.s.disclose()
        return canonical_json_bytes({"commitment": commitment.to_json(), "openings": [o.to_json() for o in openings]})


class Prover:
    """The prover: its store (`capabilities["store"]`, a `period.Prover`) registered, held to the registration and the
    session's coin commitment it is sent, and read for each pass and each window's openings."""

    def start(self, params: dict[str, Any], identity: Identity) -> list[Any]:
        self.store, self.passes, self.phase = identity.capabilities["store"], 0, Phase("Hold")
        blocks, root = self.store.register()
        return [Send(VERIFIER, Message("Register", {"blocks": blocks}, root))]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Refused", "Closed"):
            return [Finish({"passes": self.passes}, m.fields["name"] if m.type == "Refused" else "closed")]
        try:
            self.phase.check(m)
        except Refused as e:
            return [Finish({"passes": self.passes}, e.name)]
        self.phase.expect("Run", "Open", "End")
        if m.type == "Hold":
            self.store.hold(P.registration_of(m.fields["registration"]))
            return []
        t = m.fields.get("t")
        if m.type == "Run":
            try:
                challenge = C.Released.from_json(m.fields["challenge"])
            except ValueError:
                return [Finish({"passes": self.passes}, "coins")]
            root = self.store.run(challenge, t, m.fields["i"], m.body)
            self.passes += 1
            return [Send(VERIFIER, Message("Root", {"t": t, "i": m.fields["i"]}, root))]
        if m.type == "Open":
            draws = [DRAW.unpack_from(m.body, k) for k in range(0, len(m.body), DRAW.size)]
            return [Send(VERIFIER, Message("Openings", {"t": t}, W.encode_openings(self.store.open(t, draws))))]
        if m.body:
            try:
                doc = json.loads(m.body)
                self.store.end(C.LiveCommitment.from_json(doc["commitment"]),
                               [C.LiveOpening.from_json(o) for o in doc["openings"]])
            except (ValueError, KeyError, TypeError):
                return [Finish({"passes": self.passes}, "coins")]
        return [Finish({"passes": self.passes})]


PROTOCOL = ProtocolDef(MANIFEST, {VERIFIER: Verifier, PROVER: Prover})
PROTOCOLS = {ID: PROTOCOL}


__all__ = ["ID", "MANIFEST", "PROTOCOL", "PROTOCOLS", "PROVER", "VERIFIER", "VERSION", "Prover", "Verifier",
           "request_of"]
