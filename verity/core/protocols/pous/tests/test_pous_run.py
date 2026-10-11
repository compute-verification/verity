"""`run.check`, the composed check of a recorded PoUS run: an honest run at a reduced point is accepted, each refusal is the
first that applies, its reason word for word, and `check` is `check_at` at the deployed point under S1's LMS.  Every clock is
injected."""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

import pous_test_signer as ts
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import run
from verity.core.protocols.pous.schemes import p2 as P2
from verity.core.protocols.pous.verifier import VerifierKey

POINT = run.Point(P2.P2Params(64, 189, 4, 2), k=3, limit_ns=1_000)
Q = POINT.params
SALT = hashlib.sha512(b"the setup's salt").digest()[:run.SALT_BYTES]
KEY = mc.HM96.key_from_label(b"run coin key")


def block(i: int, n: int = Q.block_bytes) -> bytes:
    return hashlib.shake_256(b"C*%d" % i).digest(n)


ROWS = [mc.block_digest(block(i)) for i in range(Q.blocks)]
DIGESTS = b"".join(ROWS)
ROOT = VerifierKey.from_row_digests(Q, SALT, ROWS).root
SETUP = run.Setup(Q.blocks, hashlib.sha512(b"the commitment to W").digest(), ROOT, SALT)
OTHER_ROOT = VerifierKey.from_row_digests(Q, bytes(run.SALT_BYTES), ROWS).root


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000

    def __call__(self) -> int:
        return self.now


def session(*, blocks: int = Q.blocks, k: int = POINT.k, limit_ns: int = POINT.limit_ns, block_bytes: int = Q.block_bytes,
            binding: bytes = ROOT, spend: int = 10, respond=None, signer=None) -> run.Session:
    signer = signer or ts.TestSigner(b"run challenger", height=1)
    tree = mc.CoinTree.draw(b"source", hashlib.sha512(b"session").digest()[:32], blocks, k, key=KEY,
                            salts=[hashlib.sha512(b"salt%d" % j).digest() * 3 for j in range(k)])
    clock = Clock()
    c = mc.MemoryChallenger(tree, signer, limit_ns=limit_ns, block_bytes=block_bytes, binding=binding, clock=clock)

    def answer(i: int) -> bytes:
        clock.now += spend
        return (respond or (lambda i: block(i, block_bytes)))(i)

    signed = c.run(answer)
    return run.Session(signed, c.commitment, c.openings(), (signer.public_key,))


def outcome(setup: run.Setup = SETUP, digests: bytes = DIGESTS, s: run.Session | None = None, point: run.Point = POINT,
            schemes=ts.SCHEMES) -> str | None:
    try:
        run.check_at(point, schemes, setup, digests, s or session())
    except run.Refused as e:
        return str(e)
    return None


def verdict_reasons(s: run.Session) -> str:
    v = mc.verdict(s.signed, s.commitment, s.openings, lambda i: ROWS[i] if i < Q.blocks else None,
                   trusted=frozenset(s.trusted), schemes=ts.SCHEMES)
    assert not v.accepted
    return "; ".join(v.reasons)


def test_an_honest_run_is_accepted():
    assert outcome() is None
    assert outcome(s=session(limit_ns=POINT.limit_ns - 1)) is None, "a limit tighter than the point's is a stricter audit"


def test_each_refusal_is_the_first_that_applies():
    flipped = DIGESTS[:64] + bytes([DIGESTS[64] ^ 1]) + DIGESTS[65:]
    cases = {
        "blocks": (dict(setup=dataclasses.replace(SETUP, blocks=Q.blocks + 1)),
                   f"the setup published {Q.blocks + 1} blocks, not the point's {Q.blocks}"),
        "salt": (dict(setup=dataclasses.replace(SETUP, salt=SALT + bytes(8))), "the setup's salt is 32 bytes, not 24"),
        "short table": (dict(digests=DIGESTS[:-1]),
                        f"the table holds {len(DIGESTS) - 1} bytes, not a 64-byte digest for each of {Q.blocks} blocks"),
        "another table": (dict(digests=flipped), "the table's leaves do not have the root the setup published"),
        "another salt's root": (dict(setup=dataclasses.replace(SETUP, root=OTHER_ROOT)),
                                "the table's leaves do not have the root the setup published"),
        "session blocks": (dict(s=session(blocks=2 * Q.blocks)),
                           f"the session audits {2 * Q.blocks} blocks, not the setup's {Q.blocks}"),
        "k": (dict(s=session(k=POINT.k + 1)), f"the session runs {POINT.k + 1} rounds, not the point's {POINT.k}"),
        "limit": (dict(s=session(limit_ns=POINT.limit_ns + 1)),
                  f"the session's limit of {POINT.limit_ns + 1} ns is past the point's Δ + RTT = {POINT.limit_ns} ns"),
        "block bytes": (dict(s=session(block_bytes=2 * Q.block_bytes)),
                        f"the session's answers are {2 * Q.block_bytes} bytes, not the point's {Q.block_bytes}"),
        "binding": (dict(s=session(binding=OTHER_ROOT)), "the session is not bound to the root the setup published"),
    }
    for name, (kw, reason) in cases.items():
        assert outcome(**kw) == reason, name


def test_the_verdict_refuses_last():
    wrong = session(respond=lambda i: block(i + 1))
    late = session(spend=POINT.limit_ns + 1)
    untrusted = dataclasses.replace(session(), trusted=())
    for s in (wrong, late, untrusted):
        assert outcome(s=s) == verdict_reasons(s)
    assert outcome(s=wrong) == f"{POINT.k} answers are not their blocks of C"
    assert outcome(schemes=mc.SCHEMES) == f"signature scheme {ts.scheme(1)!r} is not accepted"


def test_the_formats_round_trip():
    s = session()
    assert run.Setup.from_json(SETUP.to_json()) == SETUP
    assert run.Session.from_json(s.to_json()) == s
    assert sorted(SETUP.to_json()) == ["blocks", "commitment", "root", "salt"]
    assert sorted(s.to_json()) == ["commitment", "openings", "signed", "trusted"]


@pytest.mark.slow
def test_check_is_the_deployed_point_under_s1s_lms():
    """`check` is `check_at(DEPLOYED, memory_challenger.SCHEMES)`: the point is `2^19` blocks of 2,056 bytes in 8 segments,
    `k = 105` and `Δ + RTT = 854,000 ns`; a run at it that a test signer signed is accepted under the test signers' schemes
    and refused by `check` for its scheme alone, and a reduced run is refused by `check` for its block count."""
    d = run.DEPLOYED
    assert (d.params.blocks, d.params.block_bytes, d.params.segments, d.params.blocks_per_segment) == (2 ** 19, 2056, 8, 2 ** 16)
    assert (d.params.w, d.params.c0, d.k, d.limit_ns) == (16448, 21065, 105, 854_000)
    assert run.SALT_BYTES * 8 == 192
    q = d.params
    tree_session = hashlib.sha512(b"deployed session").digest()[:32]
    tree = mc.CoinTree.draw(b"source", tree_session, q.blocks, d.k, key=KEY,
                            salts=[hashlib.sha512(b"salt%d" % j).digest() * 3 for j in range(d.k)])
    rows = [hashlib.sha512(b"row %d" % i).digest() for i in range(q.blocks)]
    for i in tree.indices:
        rows[i] = mc.block_digest(block(i, q.block_bytes))
    root = VerifierKey.from_row_digests(q, SALT, rows).root
    setup = run.Setup(q.blocks, SETUP.commitment, root, SALT)
    signer, clock = ts.TestSigner(b"deployed challenger", height=1), Clock()
    c = mc.MemoryChallenger(tree, signer, limit_ns=d.limit_ns, block_bytes=q.block_bytes, binding=root, clock=clock)

    def answer(i: int) -> bytes:
        clock.now += 20_000
        return block(i, q.block_bytes)

    s = run.Session(c.run(answer), c.commitment, c.openings(), (signer.public_key,))
    table = b"".join(rows)
    run.check_at(d, ts.SCHEMES, setup, table, s)
    with pytest.raises(run.Refused, match=f"^signature scheme {ts.scheme(1)!r} is not accepted$"):
        run.check(setup, table, s)
    with pytest.raises(run.Refused, match=f"^the setup published {Q.blocks} blocks, not the point's {2 ** 19}$"):
        run.check(SETUP, DIGESTS, session())
