"""PoUS's session verdict (`pous.verdict_circuit`): its gates are `gates`' list, the list `PousVerdict.verdictProg`
restates and proves of; it agrees with the reference `verdict_value` on rounds that hit and that miss each way, and at
M = 0 with `memory_challenger.verdict` on a signed session; its AND count at the audit's 132 challenges."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

import pous_test_signer as ts
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as EB
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.commitments import rowleaf
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import verdict_circuit as V

RNG = random.Random(27)
KINDS = ("hit", "zero", "edge", "wrong", "late", "early", "round")
"""How a drawn round turns out: a hit, a hit with no time elapsed or exactly LIMIT, a digest one bit off, past LIMIT,
received before it was sent, or a record of another round."""


def _bits(data: bytes) -> list[int]:
    return [(b >> k) & 1 for b in data for k in range(8)]


def _evaluate(K: int, M: int, LIMIT: int, sessions) -> list[int]:
    recs = np.array([[_bits(r) for r in rs] for rs, _ in sessions], dtype=np.uint8)
    exps = np.array([[_bits(e) for e in es] for _, es in sessions], dtype=np.uint8)
    return [int(v) for v in EB.evaluate_bits(bind(V.PousVerdict, K=K, M=M, LIMIT=LIMIT), recs, exps).reshape(-1)]


def _round(j: int, LIMIT: int, kind: str) -> tuple[bytes, bytes]:
    e = RNG.randbytes(64)
    send = RNG.randrange(1 << 63)
    elapsed = {"zero": 0, "edge": LIMIT, "late": LIMIT + 1 + RNG.randrange(1 << 20)}.get(kind, RNG.randrange(LIMIT + 1))
    receipt = send - 1 - RNG.randrange(1 << 20) if kind == "early" else send + elapsed
    receipt = min(max(receipt, 0), (1 << 64) - 1)
    h = bytes([e[0] ^ 1 << RNG.randrange(8)]) + e[1:] if kind == "wrong" else e
    rj = j ^ 1 << RNG.randrange(32) if kind == "round" else j
    return mc.Round(rj, RNG.randrange(1 << 64), send, receipt, h).value(), e


def _session(K: int, LIMIT: int, kinds) -> tuple[list[bytes], list[bytes]]:
    rounds = [_round(j, LIMIT, k) for j, k in enumerate(kinds)]
    return [r for r, _ in rounds], [e for _, e in rounds]


def _flat(d) -> tuple[list[tuple], list[int]]:
    """The one-call Program's gates, in `gates`' terms."""
    prog, _ = standalone_call(d)
    C = prog.circuit
    out: list[tuple] = []
    for i in range(C.size):
        g = C[i]
        if g.prim.id == "Input1_v1":
            out.append(("input",))
        elif BL.is_constant_bit(g.prim):
            out.append(("const", BL.CONSTANTS.index(g.prim)))
        else:
            out.append((BL.gate_kind(g.prim)[0], tuple(g.operands)))
    return out, list(C.root_outputs())


@pytest.mark.parametrize("K,M,LIMIT", [(1, 0, 1), (3, 1, 1000), (2, 5, (1 << 64) - 1)])
def test_gates_are_the_definitions(K, M, LIMIT):
    """The Definition's canonical layout is `gates`' list, gate for gate and operand for operand: what the Lean proof is
    about is the Program."""
    gs, outs = V.gates(K, M, LIMIT)
    want = [(g[0], tuple(g[1])) if g[0] in ("not", "and", "xor") else g for g in gs]
    assert _flat(bind(V.PousVerdict, K=K, M=M, LIMIT=LIMIT)) == (want, outs)


@pytest.mark.parametrize("K,M,LIMIT", [(1, 0, 7), (4, 1, 500_000), (6, 2, 1), (5, 0, (1 << 64) - 1)])
def test_against_the_reference(K, M, LIMIT):
    sessions = [_session(K, LIMIT, ["hit"] * K)]
    sessions += [_session(K, LIMIT, [RNG.choice(KINDS) for _ in range(K)]) for _ in range(12)]
    sessions += [_session(K, LIMIT, [k] * K) for k in KINDS]
    want = [V.verdict_value(K, M, LIMIT, *s) for s in sessions]
    assert _evaluate(K, M, LIMIT, sessions) == want
    assert 0 in want and 1 in want


def test_round_verdicts():
    LIMIT = 1000
    for kind, verdict in (("hit", "hit"), ("zero", "hit"), ("edge", "hit"), ("wrong", "wrong"), ("late", "late"),
                          ("early", "reject"), ("round", "reject")):
        assert V.round_verdict(LIMIT, 3, *_round(3, LIMIT, kind)) == verdict


def test_at_most_m_misses():
    K, LIMIT = 5, 100
    for M in range(K + 2):
        for misses in range(K + 1):
            s = _session(K, LIMIT, ["late"] * misses + ["hit"] * (K - misses))
            assert _evaluate(K, M, LIMIT, [s]) == [int(misses <= M)]


BLOCKS, ROUNDS, BLOCK_BYTES, LIMIT_NS = 1 << 10, 6, 64, 854_000


def _block(i: int) -> bytes:
    return hashlib.shake_256(b"C%d" % i).digest(BLOCK_BYTES)


def _expected(i: int) -> bytes:
    return hashlib.sha512(rowleaf.sha512_row_v2_prefix(8 * BLOCK_BYTES) + _block(i)).digest()


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000

    def __call__(self) -> int:
        return self.now


@pytest.mark.parametrize("wrong", [(), (2,), (0, 5)])
def test_m0_is_the_memory_challengers_verdict(wrong):
    """On a signed session, M = 0 accepts exactly when `memory_challenger.verdict` does: every answer its label, on time."""
    clock = _Clock()
    signer = ts.TestSigner(b"challenger key", height=2)
    session = hashlib.sha512(b"verdict session %r" % (wrong,)).digest()[:32]
    coins = mc.CoinTree.draw(b"secret", session, BLOCKS, ROUNDS, key=mc.HM96.key_from_label(b"coin key"),
                             salts=[hashlib.sha512(b"salt%d" % j).digest() * 3 for j in range(ROUNDS)])
    c = mc.MemoryChallenger(coins, signer, log=ts.session_log(), limit_ns=LIMIT_NS, block_bytes=BLOCK_BYTES,
                            binding=hashlib.sha512(b"root").digest(), clock=clock)
    j = 0

    def answer(i: int) -> bytes:
        nonlocal j
        clock.now += 20_000
        j += 1
        return _block(i + 1) if j - 1 in wrong else _block(i)

    signed = c.run(answer)
    clear = mc.verdict(signed, c.commitment, c.openings(), _expected, trusted={signer.public_key}, schemes=ts.SCHEMES)
    records = [r.value() for r in signed.rounds]
    expected = [_expected(r.index) for r in signed.rounds]
    assert _evaluate(ROUNDS, 0, LIMIT_NS, [(records, expected)]) == [int(clear.accepted)]
    assert clear.accepted == (not wrong)


def test_and_count():
    """At the audit's 132 challenges with no miss allowed: 673 ANDs a round (545 in the hit, 64 in each chain) and W = 8 a
    round in the counter, and W in the verdict."""
    K, M, LIMIT = 132, 0, 500_000
    gs, _ = V.gates(K, M, LIMIT)
    ands = sum(len(g[1]) - 1 for g in gs if g[0] == "and")
    W = V.counter_width(K, M)
    assert (W, ands) == (8, 673 * K + W * (K + 1)) == (8, 89_900)
    assert bind(V.PousVerdict, K=K, M=M, LIMIT=LIMIT).gates == len(gs) - (V.RECORD_BITS + V.DIGEST_BITS) * K


@pytest.mark.parametrize("statics", [dict(K=0, M=0, LIMIT=1), dict(K=1, M=-1, LIMIT=1), dict(K=1, M=0, LIMIT=0),
                                     dict(K=1, M=0, LIMIT=1 << 64), dict(K=True, M=0, LIMIT=1)])
def test_statics_refused(statics):
    with pytest.raises(ValueError):
        V.gates(**statics)
    if not any(isinstance(v, bool) for v in statics.values()):  # `bind` keys True as 1
        with pytest.raises(ValueError):
            bind(V.PousVerdict, **statics).gates
