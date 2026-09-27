"""Live prime coins (PROTOCOL 17.6): the prover's LiveTranscript against a stand-in verifier, and the verifier's
ReplayTranscript over the recorded rounds.  Negatives: a message other than the one committed before its coin (a coin
before its commitment), prime coins replayed across sessions, and tampered records."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")                                  # gpu.field is torch arithmetic
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu.transcript import CoinError, LiveTranscript, ReplayTranscript, Transcript, coins_text, parse_coins  # noqa: E402

P, DEG = 2013265921, 6


class Verifier:
    """The session server's prime side: OS-like coins after each committed state."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.rounds: list[tuple[bytes, int, list[int]]] = []
        self.before_commit = None

    def __call__(self, state: bytes, n: int) -> list[int]:
        words = [self.rng.randrange(P) for _ in range(DEG * n)]
        self.rounds.append((state, n, words))
        return words

    def commit(self) -> None:
        self.before_commit = len(self.rounds)


def protocol(tr: Transcript, msg, commit=lambda: None, alter: int | None = None) -> list[bytes]:
    """A toy prime proof: a root, the reserved start coins, sumcheck-like rounds (each message a function of the
    previous coin), root_F (the verifier's Commit), a reserved batching round and a query set.  ``msg(i, default)``
    lets a verifier absorb the proof's messages instead; returns the messages absorbed."""
    out = []

    def send(i: int, b: bytes) -> None:
        b = msg(i, b)
        if alter == i:
            b = bytes([b[0] ^ 1]) + b[1:]
        out.append(b)
        tr.absorb(b)

    send(0, b"ligero root")
    tr.reserve(3)
    c = tr.challenges(3)
    for i in range(1, 6):
        send(i, repr(c[-1]).encode())
        c.append(tr.challenge())
    if isinstance(tr, ReplayTranscript):
        tr.at_root_f()
    commit()
    send(6, b"sigma messages")
    tr.reserve(4)
    rhos = [tr.challenge() for _ in range(4)]
    send(7, repr(rhos).encode())
    tr.reserve(5)
    [tr.challenge() for _ in range(5)]
    return out


def prove(seed: int, **kw) -> tuple[Verifier, list[bytes]]:
    v = Verifier(seed)
    tr = LiveTranscript(v)
    msgs = protocol(tr, lambda i, b: b, v.commit, **kw)
    return v, msgs


def verify(rounds, before_commit: int, msgs: list[bytes]) -> None:
    tr = ReplayTranscript(rounds, before_commit)
    protocol(tr, lambda i, b: msgs[i])
    tr.finish()


def test_honest_replay_and_rounds_group():
    v, msgs = prove(1)
    assert [n for _, n, _ in v.rounds] == [3, 1, 1, 1, 1, 1, 4, 5] and v.before_commit == 6
    verify(v.rounds, v.before_commit, msgs)
    verify(parse_coins(coins_text(v.rounds, v.before_commit)).rec, v.before_commit, msgs)


def test_fiat_shamir_unchanged_by_reserve():
    a, b = Transcript(), Transcript()
    a.absorb(b"x")
    b.absorb(b"x")
    b.reserve(7)
    assert a.challenges(3) == b.challenges(3) and a.slots == b.slots == 3


def test_message_committed_after_its_coin_is_rejected():
    # the proof's message 3 is not the one whose state the prover sent before the coin that follows it
    v, msgs = prove(1)
    bad = list(msgs)
    bad[3] = b"chosen after the coin"
    with pytest.raises(CoinError, match="differ from those committed"):
        verify(v.rounds, v.before_commit, bad)


def test_prime_coins_replayed_across_sessions_are_rejected():
    va, msgs_a = prove(1)
    vb, msgs_b = prove(2)
    verify(vb.rounds, vb.before_commit, msgs_b)
    with pytest.raises(CoinError, match="round 1"):
        verify(vb.rounds, vb.before_commit, msgs_a)
    # a prover that ignores session B's coins and answers with A's: its sent states are A's, B's record then holds
    # them with B's coins, and A's proof still disagrees at the first round whose state chains a coin
    mixed = [(sa, n, wb) for (sa, n, _), (_, _, wb) in zip(va.rounds, vb.rounds)]
    with pytest.raises(CoinError):
        verify(mixed, vb.before_commit, msgs_a)


@pytest.mark.parametrize("edit,match", [
    (lambda r, b: (r[0][2].__setitem__(4, (r[0][2][4] + 1) % P), b)[1], "round 1"),
    (lambda r, b: (r.__setitem__(2, (bytes(32), r[2][1], r[2][2])), b)[1], "round 2"),
    (lambda r, b: (r.append(r[-1]), b)[1], "consumed"),
    (lambda r, b: (r.pop(), b)[1], "no such coin"),
    (lambda r, b: (r.__setitem__(0, (r[0][0], 2, r[0][2][:12])), b)[1], "3 challenges drawn, 2 issued"),
    (lambda r, b: b - 1, "root_F follows"),
])
def test_tampered_record_is_rejected(edit, match):
    v, msgs = prove(1)
    rounds = [(s, n, list(w)) for s, n, w in v.rounds]
    before = edit(rounds, v.before_commit)
    with pytest.raises(CoinError, match=match):
        verify(rounds, before, msgs)


def test_live_refuses_bad_words_and_bytes_coins():
    tr = LiveTranscript(lambda s, n: [P] * (DEG * n))
    tr.absorb(b"x")
    with pytest.raises(CoinError):
        tr.challenge()
    with pytest.raises(CoinError):
        LiveTranscript(lambda s, n: []).challenge_bytes()
