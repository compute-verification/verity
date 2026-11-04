"""A live-coin session: the prover's messages in order, and the challenger's coins between them.

Every coin is the challenger's, flipped after the messages before it are recorded: the challenger takes `after`, the
transcript's digest at that point, logs it beside the coin it flips, and never derives a coin from it. The verifier reads
coins only from that log and refuses a log whose entry doesn't follow the transcript it rebuilds (`coin-order`). Nothing
here is Fiat–Shamir: the digest orders the log, it is no source of randomness.

`ProverSession` and `VerifierSession` have the same calls, `message(label, data)` and `coins(nbytes)`, so the prover and
the verifier of each step are one function each, read side by side.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from .field import P, Ext

_START = b"verity/lin/transcript/v0\0"
_STEP = b"verity/lin/message/v0\0"


class Challenger(Protocol):
    def flip(self, after: bytes, nbytes: int) -> bytes:
        """`nbytes` fresh uniform bytes, logged beside `after`."""
        ...


@dataclass(frozen=True)
class Coin:
    after: bytes
    data: bytes


@dataclass(frozen=True)
class Message:
    label: str
    data: bytes


class Refused(Exception):
    """A check failed; `code` names it."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class _Transcript:
    def __init__(self, statement: bytes) -> None:
        self.digest = hashlib.sha512(_START + statement).digest()

    def absorb(self, label: str, data: bytes) -> None:
        tag = label.encode()
        self.digest = hashlib.sha512(_STEP + self.digest + len(tag).to_bytes(8, "big") + tag
                                     + len(data).to_bytes(8, "big") + data).digest()


class ProverSession:
    def __init__(self, statement: bytes, challenger: Challenger) -> None:
        self._t = _Transcript(statement)
        self._challenger = challenger
        self.messages: list[Message] = []

    def message(self, label: str, data: bytes) -> bytes:
        self._t.absorb(label, data)
        self.messages.append(Message(label, data))
        return data

    def coins(self, nbytes: int) -> bytes:
        return self._challenger.flip(self._t.digest, nbytes)


class VerifierSession:
    def __init__(self, statement: bytes, messages: list[Message], log: list[Coin]) -> None:
        self._t = _Transcript(statement)
        self._messages = list(messages)
        self._log = list(log)

    def message(self, label: str, data: bytes | None = None) -> bytes:
        if not self._messages:
            raise Refused("missing", label)
        m = self._messages.pop(0)
        if m.label != label:
            raise Refused("label", f"expected {label}, got {m.label}")
        self._t.absorb(m.label, m.data)
        return m.data

    def coins(self, nbytes: int) -> bytes:
        if not self._log:
            raise Refused("coin-order", "the log has no coin here")
        c = self._log.pop(0)
        if c.after != self._t.digest or len(c.data) != nbytes:
            raise Refused("coin-order", "a coin that doesn't follow the transcript")
        return c.data

    def finish(self) -> None:
        if self._messages or self._log:
            raise Refused("trailing", f"{len(self._messages)} messages and {len(self._log)} coins left over")


Session = ProverSession | VerifierSession


class Reader:
    """Exact samplers over one coin's bytes, drawing more coins from the session when a rejection runs past them."""

    def __init__(self, session: Session, nbytes: int) -> None:
        self._s = session
        self._chunk = nbytes
        self._buf = session.coins(nbytes)

    def _take(self, k: int) -> bytes:
        while len(self._buf) < k:
            self._buf += self._s.coins(self._chunk)
        out, self._buf = self._buf[:k], self._buf[k:]
        return out

    def field(self) -> int:
        while True:
            x = int.from_bytes(self._take(8), "little")
            if x < P:
                return x

    def ext(self) -> Ext:
        return (self.field(), self.field(), self.field())

    def index(self, n: int) -> int:
        """Uniform in [0, n) for n a power of two at most 2^64."""
        if n < 1 or n & (n - 1) or n > 1 << 64:
            raise ValueError(f"{n} is not a power of two at most 2^64")
        return int.from_bytes(self._take(8), "little") % n


def ext_coin(session: Session) -> Ext:
    return Reader(session, 32).ext()
