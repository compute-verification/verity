"""SHA-256 Fiat-Shamir transcript, the structure of ``archive/gkr/src/transcript.rs`` over BabyBear^6.

``state <- H(state || message)``; ``challenge_i <- H(state || "chal" || ctr)`` read as eight little-endian
32-bit words, each accepted iff below ``p`` (rejection sampling), six accepted words per extension challenge.
An extension element is absorbed as 24 bytes (six LE u32 coordinates).  ``slots`` counts every challenge
drawn (``rounds.protocol``).  Fiat-Shamir is what the CPU prover uses too (the interactive object and step 0 of
PROTOCOL.md are not implemented in either); the diagnostic label stays ``NON_ZK_PROOF_DIAGNOSTIC``.

:class:`LiveTranscript` is the interactive object for route (a) (PROTOCOL §17.6): the same absorb chain, but every
challenge is the live verifier's.  A round sends the state (it chains every message absorbed so far and every earlier
round's coins) and a count ``n``; the verifier answers with ``6 n`` OS-drawn words below ``p``, which are the round's
``n`` challenges and are then chained into the state as ``H(state || LIVE_TAG || words LE)``.  A challenge drawn with
no absorb since the last round takes the round's next unused coin; any absorb discards the rest.  :class:`ReplayTranscript`
is the verifier's side over the session record: at each round its own state must equal the recorded one (the proof's
messages are the ones committed before the coin), and every recorded round must be consumed.
"""
from __future__ import annotations

import hashlib
import struct

from .field import DEG, Ext, P, exts_to_bytes, to_bytes

DOMAIN = b"verity-gkr/candidate-a/v1/babybear6"
LIVE_TAG = b"verity-gkr/live-coins/v1"


class CoinError(Exception):
    """A live round was refused, or a replayed round disagrees with the session record."""


class Transcript:
    live = False

    def __init__(self, domain: bytes = DOMAIN) -> None:
        self.state = hashlib.sha256(domain).digest()
        self.ctr = 0
        self.slots = 0

    def absorb(self, data: bytes) -> None:
        self.state = hashlib.sha256(self.state + data).digest()

    def absorb_ext(self, x: Ext) -> None:
        self.absorb(to_bytes(x))

    def absorb_exts(self, xs) -> None:
        self.absorb(exts_to_bytes(xs))

    def reserve(self, n: int) -> None:
        """The next ``n`` challenges follow with no absorb between them (one live round; nothing under Fiat-Shamir)."""

    def _words(self):
        while True:
            d = hashlib.sha256(self.state + b"chal" + self.ctr.to_bytes(8, "little")).digest()
            self.ctr += 1
            for i in range(8):
                v = int.from_bytes(d[4 * i:4 * i + 4], "little")
                if v < P:
                    yield v

    def challenge(self) -> Ext:
        self.slots += 1
        g = self._words()
        return tuple(next(g) for _ in range(DEG))

    def challenges(self, n: int) -> list[Ext]:
        self.reserve(n)
        return [self.challenge() for _ in range(n)]

    def challenge_bytes(self) -> bytes:
        """One whole digest (32 uniform bytes, a GF(2^256) coin of the bit link); the next challenge starts afresh."""
        self.slots += 1
        d = hashlib.sha256(self.state + b"chal" + self.ctr.to_bytes(8, "little")).digest()
        self.ctr += 1
        return d


class LiveTranscript(Transcript):
    """``call(state, n) -> 6 n words``: one round with the live verifier (see the module docstring)."""
    live = True

    def __init__(self, call, domain: bytes = DOMAIN) -> None:
        super().__init__(domain)
        self.call = call
        self.buf: list[Ext] = []
        self.dirty = True
        self.want = 0
        self.rounds: list[tuple[bytes, int, list[int]]] = []
        self.wait_s = 0.0

    def absorb(self, data: bytes) -> None:
        super().absorb(data)
        self.dirty = True

    def reserve(self, n: int) -> None:
        if self.dirty or not self.buf:
            self.want = max(self.want, n)

    def _round(self) -> None:
        import time

        n = max(1, self.want)
        self.want = 0
        t = time.perf_counter()
        words = list(self.call(self.state, n))
        self.wait_s += time.perf_counter() - t
        if len(words) != DEG * n or any(not 0 <= w < P for w in words):
            raise CoinError(f"live round {len(self.rounds)}: {len(words)} words for {n} challenges, or a word not below p")
        self.rounds.append((self.state, n, words))
        self.state = hashlib.sha256(self.state + LIVE_TAG + struct.pack(f"<{len(words)}I", *words)).digest()
        self.buf = [tuple(words[DEG * i: DEG * i + DEG]) for i in range(n)]
        self.dirty = False

    def challenge(self) -> Ext:
        self.slots += 1
        if self.dirty or not self.buf:
            self._round()
        return self.buf.pop(0)

    def challenge_bytes(self) -> bytes:
        raise CoinError("live prime coins are BabyBear^6 challenges only")


class ReplayTranscript(LiveTranscript):
    """The verifier's transcript over the recorded rounds ``[(state, n, words)]``; ``before_commit`` rounds precede root_F."""

    def __init__(self, rounds, before_commit: int, domain: bytes = DOMAIN) -> None:
        self.rec = [(bytes(s), int(n), [int(w) for w in ws]) for s, n, ws in rounds]
        self.before_commit = before_commit
        self.k = 0
        super().__init__(self._next, domain)

    def _next(self, state: bytes, n: int) -> list[int]:
        k = self.k
        if k >= len(self.rec):
            raise CoinError(f"R2-prime: round {k}: the verifier issued no such coin")
        st, rn, words = self.rec[k]
        self.k += 1
        if rn != n:
            raise CoinError(f"R2-prime: round {k}: {n} challenges drawn, {rn} issued")
        if st != state:
            raise CoinError(f"R2-prime: round {k}: the proof's messages differ from those committed before the coin")
        return words

    def at_root_f(self) -> None:
        """root_F is computed here: exactly the rounds issued before the session's Commit precede it."""
        if self.k != self.before_commit:
            raise CoinError(f"R2-prime: root_F follows {self.k} prime rounds, the session issued {self.before_commit} before Commit")

    def finish(self) -> None:
        if self.k != len(self.rec):
            raise CoinError(f"R2-prime: {len(self.rec)} prime rounds recorded, the proof consumed {self.k}")


def parse_coins(text: str) -> ReplayTranscript:
    """``prime_coins.txt`` (written from the session record by ``tools/cell_gate.py``): ``prime-coins live-coins/v1``,
    ``before_commit K``, then one ``round STATE_HEX N W0 W1 ...`` line per round."""
    lines = [ln.split() for ln in text.splitlines() if ln.strip()]
    if not lines or lines[0] != ["prime-coins", "live-coins/v1"]:
        raise CoinError("prime_coins.txt: not a live-coins/v1 record")
    kv = {t[0]: t[1:] for t in lines[1:] if t[0] != "round"}
    rounds = [(bytes.fromhex(t[1]), int(t[2]), [int(w) for w in t[3:]]) for t in lines[1:] if t[0] == "round"]
    return ReplayTranscript(rounds, int(kv["before_commit"][0]))


def coins_text(rounds, before_commit: int) -> str:
    return "prime-coins live-coins/v1\nbefore_commit %d\n" % before_commit + "".join(
        f"round {bytes(s).hex()} {n} {' '.join(str(w) for w in ws)}\n" for s, n, ws in rounds)
