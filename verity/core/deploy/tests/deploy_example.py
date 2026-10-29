"""The worked example: a two-party protocol the deployment layer's tests run every way it runs a protocol.

The challenger samples a 32-byte nonce from its coins, reads its clock and sends it (`Challenge`, coins); the responder
must answer SHA-512 of it (`Answer`) before the challenger's deadline (`deadline_ns`, a `Timer`). The challenger reads
its clock again on the answer, sends its verdict (`Verdict`, output) and finishes: refused `late` when its timer fires
first, `wrong` for another answer, and by the wire's name for a frame it refuses. The responder finishes on the verdict.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

from verity.core.deploy.manifest import Channel, Manifest, Role
from verity.core.deploy.party import Finish, Received, Record, Send, Timer
from verity.core.deploy.runtime import ProtocolDef
from verity.core.wire import Body, Bool, Int, Message, Schema
from verity.ml.crypto import lms

ID = "verity/example/deadline"
HERE = Path(__file__).parent
MANIFEST = Manifest(
    ID, 1, "deploy_example:PARAMS",
    parties=(Role("challenger", "auditor", needs=("clock", "coins")), Role("responder", "developer")),
    channels=(Channel("challenger", "responder", ("Challenge", "Verdict"), latency_ms=1, carries_coins=True),
              Channel("responder", "challenger", ("Answer",), latency_ms=1)),
    messages=(Schema("Challenge", 1, "coins", {"deadline_ns": Int()}, Body(size=32, max=32)),
              Schema("Answer", 1, "private", body=Body(size=64, max=64)),
              Schema("Verdict", 1, "output", {"ok": Bool(), "elapsed_ns": Int()})))
PARAMS = {"deadline_ns": 5_000_000}


class Challenger:
    def start(self, params, identity):
        self.deadline, self.timer = params["deadline_ns"], params.get("timer", True)
        self.nonce = self.sent = self.answer = None
        return [Send("coins", Message("CoinsDraw", {"n": 32}))]

    def on(self, r: Received):
        m = r.message
        if m.type == "Refused":
            return [Finish({"channel": r.channel, "detail": m.fields["detail"]}, m.fields["name"])]
        if m.type == "Closed":
            return [Finish({"channel": r.channel}, "closed")]
        if m.type == "Coins":
            self.nonce = m.body
            return [Send("clock", Message("ClockRead"))]
        if m.type == "ClockReading" and self.sent is None:
            self.sent = m.fields["ns"]
            out = [Record({"nonce": self.nonce.hex()}, "coins"),
                   Send("responder", Message("Challenge", {"deadline_ns": self.deadline}, self.nonce))]
            return out + ([Timer(self.sent + self.deadline)] if self.timer else [])
        if m.type == "Answer":
            self.answer = m.body
            return [Send("clock", Message("ClockRead"))]
        if m.type == "ClockReading":
            elapsed = m.fields["ns"] - self.sent
            ok = self.answer == hashlib.sha512(self.nonce).digest() and elapsed <= self.deadline
            verdict = Send("responder", Message("Verdict", {"ok": ok, "elapsed_ns": elapsed}))
            if elapsed > self.deadline:
                return [verdict, Finish({"elapsed_ns": elapsed}, "late")]
            if not ok:
                return [verdict, Finish({"elapsed_ns": elapsed}, "wrong")]
            return [verdict, Finish({"elapsed_ns": elapsed})]
        if m.type == "ClockTimer" and self.answer is None:
            return [Finish({"elapsed_ns": m.fields["ns"] - self.sent}, "late")]
        return []


class Responder:
    def start(self, params, identity):
        return []

    def answer(self, nonce: bytes) -> bytes:
        return hashlib.sha512(nonce).digest()

    def on(self, r: Received):
        m = r.message
        if m.type == "Challenge":
            return [Send("challenger", Message("Answer", body=self.answer(m.body)))]
        if m.type == "Verdict":
            return [Finish(dict(m.fields))]
        return [Finish({"told": m.type})]


class NewResponder(Responder):
    """Another implementation of the responder, which must answer as the first did."""

    def answer(self, nonce: bytes) -> bytes:
        h = hashlib.sha512()
        for i in range(0, len(nonce), 8):
            h.update(nonce[i:i + 8])
        return h.digest()


class WrongResponder(Responder):
    def answer(self, nonce: bytes) -> bytes:
        return hashlib.sha512(nonce + b"!").digest()


class SilentResponder(Responder):
    def on(self, r: Received):
        return [] if r.message.type == "Challenge" else super().on(r)


class RudeResponder(Responder):
    """Sends the challenger a verdict, which the manifest lets only the challenger send."""

    def on(self, r: Received):
        if r.message.type == "Challenge":
            return [Send("challenger", Message("Verdict", {"ok": True, "elapsed_ns": 0}))]
        return super().on(r)


class DyingResponder(Responder):
    def answer(self, nonce: bytes) -> bytes:
        raise RuntimeError("the responder's process fell over")


class GreedyChallenger(Challenger):
    """A challenger that samples 64 coins where the recorded one drew 32."""

    def start(self, params, identity):
        super().start(params, identity)
        return [Send("coins", Message("CoinsDraw", {"n": 64}))]


def protocol(**parties) -> ProtocolDef:
    return ProtocolDef(MANIFEST, {"challenger": Challenger, "responder": Responder, **parties})


PROTOCOL = protocol()
PROTOCOLS = {ID: PROTOCOL}

KEY_PARAMS = lms.Params(64, 1, 4)


@functools.cache
def _private(party: str) -> lms.PrivateKey:
    return lms.keygen(KEY_PARAMS, hashlib.sha512(b"verity/deploy/tests/key/" + party.encode()).digest())


class PartyKey:
    """A party's test key: an LMS key of 16 leaves from its name, each signature at the next leaf."""

    def __init__(self, party: str) -> None:
        self.key, self.q = _private(party), 0
        self.public_key = lms.public_key(self.key)

    def sign(self, message: bytes) -> bytes:
        self.q += 1
        return lms.sign_at(self.key, self.q - 1, message).encode()


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    return lms.verify(KEY_PARAMS, public, message, lms.Signature.decode(KEY_PARAMS, signature))


def keys(*parties: str) -> dict[str, PartyKey]:
    return {p: PartyKey(p) for p in parties}


def config(**extra) -> dict:
    """The deployment configuration: the protocol, its parameters, and each party's machine and public key."""
    return {"protocol": ID, "params": dict(PARAMS),
            "party": [{"name": p, "machine": m, "key": PartyKey(p).public_key.hex()}
                      for p, m in (("challenger", "auditor-box"), ("responder", "gpu-node"))],
            **extra}


PROCESS = {"protocols": {ID: "deploy_example:PROTOCOL"}, "keys": "deploy_example:PartyKey",
           "verify": "deploy_example:verify", "path": [str(HERE)]}
"""What a local process needs to run a party of the example (`verity.core.deploy.process`)."""
