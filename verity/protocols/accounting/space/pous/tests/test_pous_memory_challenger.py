"""The memory challenger (`memory_challenger`): the coin tree's commit and openings, the signed session record, the
verdict that fails closed, the whole block hashed, and a late round that ends the session with no retry.  Every clock is
injected."""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

import pous_test_signer as ts
from verity.protocols.accounting.space.pous import memory_challenger as mc
from verity.protocols.accounting.space.pous import schemes

BLOCKS, K, BLOCK_BYTES, LIMIT = 1 << 10, 12, 64, 854_000
SOURCE, SESSION = b"challenger secret", hashlib.sha512(b"session").digest()[:32]
KEY = mc.HM96.key_from_label(b"test coin key")
SALTS = [hashlib.sha512(b"salt%d" % j).digest() * 3 for j in range(K)]


def block(i: int) -> bytes:
    return hashlib.shake_256(b"C%d" % i).digest(BLOCK_BYTES)


def expected(i: int) -> bytes:
    return hashlib.sha512(block(i)).digest()


class Clock:
    """Time advances only when a responder spends it."""

    def __init__(self) -> None:
        self.now = 1_000_000

    def __call__(self) -> int:
        return self.now


def coins(session: bytes = SESSION) -> mc.CoinTree:
    return mc.CoinTree.draw(SOURCE, session, BLOCKS, K, key=KEY, salts=SALTS)


def session(respond=None, *, signer=None, clock=None, spend=lambda j: 20_000):
    clock = clock or Clock()
    signer = signer or ts.TestSigner(b"challenger key", height=2)
    c = mc.MemoryChallenger(coins(), signer, limit_ns=LIMIT, block_bytes=BLOCK_BYTES, binding=b"vk root", clock=clock)
    j = 0

    def answer(i: int) -> bytes:
        nonlocal j
        clock.now += spend(j)
        j += 1
        return (respond or block)(i)

    signed = c.run(answer)
    return c, signed, signer


def judge(c, signed, signer, *, schemes_=ts.SCHEMES, openings=None):
    return mc.verdict(signed, c.commitment, c.openings() if openings is None else openings, expected,
                      trusted={signer.public_key}, schemes=schemes_)


def test_coin_tree_commits_before_and_opens_after():
    tree = coins()
    want = mc.derive(SOURCE, mc.COIN_DOMAIN, {"session": SESSION, "blocks": BLOCKS, "k": K})
    assert tree.indices == tuple(want.uniform(BLOCKS, j) for j in range(K))
    c = mc.MemoryChallenger(tree, ts.TestSigner(b"k", height=1), limit_ns=LIMIT, block_bytes=BLOCK_BYTES, clock=Clock())
    assert c.commitment.coin_root == tree.root and len(tree.root) == 64
    with pytest.raises(RuntimeError, match="opened after the session"):
        c.openings()
    for j in range(K):
        assert mc.verify_coin(c.commitment, tree.opening(j))
    o = tree.opening(3)
    bad = [dataclasses.replace(o, index=(o.index + 1) % BLOCKS), dataclasses.replace(o, salt=bytes(len(o.salt))),
           dataclasses.replace(o, j=4), dataclasses.replace(o, path=o.path[:-1]),
           dataclasses.replace(o, path=(bytes(64),) + o.path[1:])]
    assert not any(mc.verify_coin(c.commitment, x) for x in bad)
    other = dataclasses.replace(c.commitment, session=hashlib.sha512(b"other").digest()[:32])
    assert not mc.verify_coin(other, o)
    assert coins(other.session).root != tree.root
    assert mc.CoinTree(SESSION, BLOCKS, tree.indices, KEY, SALTS[::-1]).root != tree.root


def test_honest_session_is_accepted_and_publishes_only_the_verdict():
    c, signed, signer = session()
    assert signed.ended == mc.COMPLETE and len(signed.rounds) == K
    assert [r.index for r in signed.rounds] == list(coins().indices)
    assert all(r.digest == expected(r.index) and r.elapsed_ns == 20_000 for r in signed.rounds)
    v = judge(c, signed, signer)
    assert v.accepted and v.reasons == () and v.public() == {"accepted": True}
    assert mc.SignedSession.from_json(signed.to_json()) == signed
    assert c.close() is signed


def test_fail_closed_outside_tests():
    c, signed, signer = session()
    v = judge(c, signed, signer, schemes_=mc.SCHEMES)
    assert not v.accepted and v.reasons == (f"signature scheme {ts.SCHEME!r} is not accepted",)
    with pytest.raises(ValueError, match="test signer"):
        mc.register(ts.SCHEME, ts.verify)
    assert ts.SCHEME not in mc.SCHEMES
    unsigned = dataclasses.replace(signed, signature=b"")
    assert judge(c, unsigned, signer).reasons == ("the signature does not verify",)
    assert judge(c, dataclasses.replace(signed, scheme="none"), signer).reasons == ("signature scheme 'none' is not accepted",)


def test_bad_signature_and_edited_records_are_rejected():
    c, signed, signer = session()
    flipped = bytearray(signed.signature)
    flipped[100] ^= 1
    assert judge(c, dataclasses.replace(signed, signature=bytes(flipped)), signer).reasons == ("the signature does not verify",)
    r = signed.rounds[5]
    edited = signed.rounds[:5] + (dataclasses.replace(r, receipt_ns=r.receipt_ns - 1),) + signed.rounds[6:]
    assert judge(c, dataclasses.replace(signed, rounds=edited), signer).reasons == ("the signature does not verify",)
    stranger = ts.TestSigner(b"someone else", height=1)
    forged = dataclasses.replace(signed, public_key=stranger.public_key, signature=stranger.sign(signed.message()))
    assert judge(c, forged, signer).reasons == ("the session is signed by a key that is not a trusted memory challenger's",)
    moved = dataclasses.replace(signed.commitment, limit_ns=LIMIT + 1)
    assert "the signed header is not the session's commitment" in mc.verdict(
        signed, moved, c.openings(), expected, trusted={signer.public_key}, schemes=ts.SCHEMES).reasons


def test_the_whole_block_is_hashed():
    tampered = lambda i: block(i)[:-1] + bytes([block(i)[-1] ^ 0x80])  # noqa: E731
    c, signed, signer = session(lambda i: tampered(i) if i == coins().indices[7] else block(i))
    v = judge(c, signed, signer)
    assert not v.accepted and v.wrong == 1 and v.reasons == ("1 answers are not their blocks of C",)
    with pytest.raises(ValueError, match="whole 64-byte block"):
        session(lambda i: block(i)[:-1])


def test_receipt_time_is_read_before_hashing(monkeypatch):
    events: list[str] = []

    class Seen(Clock):
        def __call__(self) -> int:
            events.append("clock")
            return self.now

    real = hashlib.sha512

    def sha512(data: bytes = b""):
        events.append("hash")
        return real(data)

    c = mc.MemoryChallenger(coins(), ts.TestSigner(b"k", height=1), limit_ns=LIMIT, block_bytes=BLOCK_BYTES, clock=Seen())
    i = c.challenge()
    monkeypatch.setattr(mc.hashlib, "sha512", sha512)
    events.clear()
    c.receive(block(i))
    assert events == ["clock", "hash"]


def test_a_late_round_ends_the_session_with_no_retry():
    c, signed, signer = session(spend=lambda j: LIMIT + 1 if j == 4 else 20_000)
    assert signed.ended == mc.LATE and len(signed.rounds) == 5 and signed.rounds[-1].elapsed_ns == LIMIT + 1
    with pytest.raises(RuntimeError, match="no round is retried"):
        c.challenge()
    v = judge(c, signed, signer)
    assert not v.accepted and v.late == 1
    assert v.reasons == (f"the session ended late after 5 of {K} rounds", f"1 answers are past Δ + RTT = {LIMIT} ns")
    on_time = session(spend=lambda j: LIMIT)
    assert judge(*on_time).accepted


def test_openings_must_match_the_revealed_indices():
    c, signed, signer = session()
    opened = list(c.openings())
    opened[2] = coins().opening(3)
    assert judge(c, signed, signer, openings=opened).reasons == ("round 2's index is not its committed coin",)


def test_refusals():
    with pytest.raises(ValueError, match="coarser than 1000 ns"):
        mc.MemoryChallenger(coins(), ts.TestSigner(b"k", height=1), limit_ns=LIMIT, block_bytes=BLOCK_BYTES, clock=Clock(),
                            resolution_ns=10_000)
    c = mc.MemoryChallenger(coins(), ts.TestSigner(b"k", height=1), limit_ns=LIMIT, block_bytes=BLOCK_BYTES, clock=Clock())
    with pytest.raises(RuntimeError, match="not ended"):
        c.close()
    c.challenge()
    with pytest.raises(RuntimeError, match="unanswered"):
        c.challenge()
    signer = ts.TestSigner(b"k", height=1)
    signer.sign(b"a"), signer.sign(b"b")
    with pytest.raises(RuntimeError, match="every one-time key"):
        signer.sign(b"c")


def test_the_certificate_covers_p2_v3_at_the_cap():
    from verity.protocols.accounting.space.pous.schemes import p2

    scheme = schemes.get("p2-16448/v3")
    limit = 500_000 + p2.RTT_ALLOWANCE_CAP_NS
    assert limit == 854_000 < p2.ROOT_CALL_NS
    cert = mc.covering_certificate(scheme, p2.CERT_K, limit, experimental=True)
    assert cert.k == p2.CERT_K == 105
    with pytest.raises(ValueError, match="experimental"):
        mc.covering_certificate(scheme, p2.CERT_K, limit)
    with pytest.raises(ValueError, match="no certificate"):
        mc.covering_certificate(scheme, p2.CERT_K, limit + 1_000, experimental=True)
