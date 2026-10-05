"""Coins protocol for Ligerito (lane ligerito-proto owns this file; every Ligerito lane draws randomness ONLY here).

Verifier randomness is consumed through one interface so the same prover/verifier code runs against

* ``FiatShamirCoins``  -- BLAKE2b transcript + SHAKE-256 squeeze (the prototype / laptop diagnostic; non-interactive),
* ``LocalCoins``       -- deterministic from a seed, IGNORES the absorbed data (diagnostic only: lets two processes
                          agree on the coins without a transcript; never a soundness argument),
* ``LiveCoins``        -- verifier coin slots (rule 8c: every challenge is a function of verifier coins alone); the
                          slot stream comes from ``archive/direct/ligero/live.py`` (lane live-2 posts the round-latency
                          numbers and the stream binding).  Same ``challenge``/``indices`` surface; ``absorb`` is a no-op
                          on the coin side (the verifier binds the prover's messages itself) but still counts messages.

Interface (frozen 17:45Z, see the lane note ``## Interface``):

    coins.absorb(label: bytes, data: bytes) -> None      prover message (root, sumcheck message, opened rows, ...)
    coins.challenge(label: bytes, n: int) -> np.ndarray  (n, 6) int64 F_{p^6} elements (coords in [0, p)); ONE coin/round trip
    coins.indices(label: bytes, domain: int, count: int) -> list[int]   ``count`` DISTINCT positions in [0, domain); one coin
    coins.rounds -> int                                  sequential verifier messages consumed so far

Extension field: F_{p^6} = F_p[x] / (x^6 - 31) (coordinate i = coefficient of x^i); see ``proto/ext.py``.
The order of calls IS the protocol: round i absorbs the root, then ``challenge`` k'_i times (one per sumcheck variable,
each after absorbing the (3, 6) message), then ``indices`` for S_{i-1}, then ``challenge`` for the (1 + |S|) batching
coefficients alpha.  ``rounds`` is what an interactive run would round-trip: Sigma_i (1 + k'_i + 1) + 1.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

P = 2013265921
D = 6


@runtime_checkable
class Coins(Protocol):
    rounds: int

    def absorb(self, label: bytes, data: bytes) -> None: ...

    def challenge(self, label: bytes, n: int) -> np.ndarray: ...

    def indices(self, label: bytes, domain: int, count: int) -> list[int]: ...


def _ext_from_stream(raw: bytes, n: int) -> np.ndarray:
    u = np.frombuffer(raw, dtype="<u8").astype(np.int64, copy=False)
    return (u % P).reshape(n, D)                        # 64-bit reduction: bias < 2^-33


def _distinct_indices(stream, domain: int, count: int) -> list[int]:
    assert count <= domain
    out: list[int] = []
    seen: set[int] = set()
    while len(out) < count:
        raw = np.frombuffer(stream(8 * 2 * count), dtype="<u8")
        for u in raw:
            i = int(u % domain)
            if i not in seen:
                seen.add(i)
                out.append(i)
                if len(out) == count:
                    break
    return out


class FiatShamirCoins:
    """BLAKE2b-256 chained transcript; challenges squeezed with SHAKE-256 keyed by (state, counter)."""

    def __init__(self, label: bytes = b"ligerito-proto/v1"):
        self.state = hashlib.blake2b(label, digest_size=32).digest()
        self.counter = 0
        self.rounds = 0

    def absorb(self, label: bytes, data: bytes) -> None:
        h = hashlib.blake2b(digest_size=32)
        h.update(self.state)
        h.update(label)
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
        self.state = h.digest()

    def _stream(self, nbytes: int) -> bytes:
        x = hashlib.shake_256(self.state + self.counter.to_bytes(8, "little")).digest(nbytes)
        self.counter += 1
        return x

    def challenge(self, label: bytes, n: int) -> np.ndarray:
        self.rounds += 1
        self.absorb(b"challenge", label)
        return _ext_from_stream(self._stream(8 * n * D), n)

    def indices(self, label: bytes, domain: int, count: int) -> list[int]:
        self.rounds += 1
        self.absorb(b"indices", label + domain.to_bytes(8, "little"))
        return _distinct_indices(self._stream, domain, count)


class LocalCoins:
    """Deterministic coins from a seed; ``absorb`` only counts.  Diagnostic: NOT bound to the prover's messages."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.counter = 0
        self.rounds = 0
        self.absorbed = 0

    def absorb(self, label: bytes, data: bytes) -> None:
        self.absorbed += 1

    def _stream(self, nbytes: int) -> bytes:
        x = hashlib.shake_256(b"local-coins" + self.seed.to_bytes(8, "little") + self.counter.to_bytes(8, "little")).digest(nbytes)
        self.counter += 1
        return x

    def challenge(self, label: bytes, n: int) -> np.ndarray:
        self.rounds += 1
        return _ext_from_stream(self._stream(8 * n * D), n)

    def indices(self, label: bytes, domain: int, count: int) -> list[int]:
        self.rounds += 1
        return _distinct_indices(self._stream, domain, count)


class LiveCoins:
    """Verifier coin slots.  ``slots``: an iterator of 32-byte verifier coins (one per round trip) from the live session
    (``archive/direct/ligero/live.py``); each coin seeds SHAKE-256 for that round's challenge/indices.  The verifier
    side replays the same slots.  Not exercised by the prototype benchmark (Fiat-Shamir there); the surface is fixed so
    ligerito-relation / live-2 can plug the stream in without touching pcs.py."""

    def __init__(self, slots):
        self.slots = iter(slots)
        self.rounds = 0
        self.absorbed = 0
        self._coin = b""
        self._sub = 0

    def absorb(self, label: bytes, data: bytes) -> None:
        self.absorbed += 1

    def _next(self) -> None:
        self._coin = bytes(next(self.slots))
        if len(self._coin) < 16:
            raise ValueError("a verifier coin must be at least 16 bytes")
        self._sub = 0
        self.rounds += 1

    def _stream(self, nbytes: int) -> bytes:
        x = hashlib.shake_256(self._coin + self._sub.to_bytes(8, "little")).digest(nbytes)
        self._sub += 1
        return x

    def challenge(self, label: bytes, n: int) -> np.ndarray:
        self._next()
        return _ext_from_stream(self._stream(8 * n * D), n)

    def indices(self, label: bytes, domain: int, count: int) -> list[int]:
        self._next()
        return _distinct_indices(self._stream, domain, count)


# ------------------------------------------------------------------------------------------ prototype compatibility

class Transcript(FiatShamirCoins):
    """The prototype's names: ``squeeze_ext(n)`` / ``squeeze_indices(domain, count)`` / ``coins``."""

    @property
    def coins(self) -> int:
        return self.rounds

    def squeeze_ext(self, count: int) -> np.ndarray:
        return self.challenge(b"ext", count)

    def squeeze_indices(self, n: int, count: int) -> list[int]:
        return self.indices(b"S", n, count)
