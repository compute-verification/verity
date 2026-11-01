"""The memory challenger (`memory_challenger`): the coin tree's commit and openings, the signed session record, the
verdict that fails closed, the whole block hashed under `sha512/row/v2`, a late round that ends the session with no retry,
no garbage collection inside the rounds, and the session log that refuses a used id.  Every clock is injected."""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import os
import tempfile

import pytest

import pous_test_signer as ts
from verity.ml.commitments import rowleaf
from verity.ml.crypto import lms
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import schemes

BLOCKS, K, BLOCK_BYTES, LIMIT = 1 << 10, 12, 64, 854_000
SOURCE, SESSION = b"challenger secret", hashlib.sha512(b"session").digest()[:32]
KEY = mc.HM96.key_from_label(b"test coin key")
ROOT = hashlib.sha512(b"the setup's root of C*").digest()
SALTS = [hashlib.sha512(b"salt%d" % j).digest() * 3 for j in range(K)]


def block(i: int) -> bytes:
    return hashlib.shake_256(b"C%d" % i).digest(BLOCK_BYTES)


def expected(i: int) -> bytes:
    return hashlib.sha512(rowleaf.sha512_row_v2_prefix(8 * BLOCK_BYTES) + block(i)).digest()


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
    c = mc.MemoryChallenger(coins(), signer, log=ts.session_log(), limit_ns=LIMIT, block_bytes=BLOCK_BYTES, binding=ROOT,
                            clock=clock)
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
    c = mc.MemoryChallenger(tree, ts.TestSigner(b"k", height=1), log=ts.session_log(), limit_ns=LIMIT,
                            block_bytes=BLOCK_BYTES, binding=ROOT, clock=Clock())
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
    assert signed.scheme == ts.scheme(2) and signed.scheme.startswith(mc.TEST_PREFIX)
    v = judge(c, signed, signer, schemes_=mc.SCHEMES)
    assert not v.accepted and v.reasons == (f"signature scheme {signed.scheme!r} is not accepted",)
    with pytest.raises(ValueError, match="test signer"):
        mc.register(signed.scheme, ts.SCHEMES[signed.scheme])
    assert set(mc.SCHEMES) == {mc.LMS} and not set(ts.SCHEMES) & set(mc.SCHEMES)
    as_lms = dataclasses.replace(signed, scheme=mc.LMS)
    assert judge(c, as_lms, signer, schemes_=mc.SCHEMES).reasons == ("the signature does not verify",)
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
    stranger = ts.TestSigner(b"someone else", height=2)
    forged = dataclasses.replace(signed, public_key=stranger.public_key,
                                 signature=stranger.sign(signed.commitment.session, signed.message()))
    assert judge(c, forged, signer).reasons == ("the session is signed by a key that is not a trusted memory challenger's",)
    assert signed.commitment.binding == ROOT and signed.commitment.header()["binding"] == ROOT.hex()
    for moved in (dataclasses.replace(signed.commitment, limit_ns=LIMIT + 1),
                  dataclasses.replace(signed.commitment, binding=hashlib.sha512(b"another setup's root").digest())):
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

    real = mc.block_digest

    def digest(data: bytes) -> bytes:
        events.append("hash")
        return real(data)

    c = mc.MemoryChallenger(coins(), ts.TestSigner(b"k", height=1), log=ts.session_log(), limit_ns=LIMIT,
                            block_bytes=BLOCK_BYTES, binding=ROOT, clock=Seen())
    i = c.challenge()
    monkeypatch.setattr(mc, "block_digest", digest)
    events.clear()
    c.receive(block(i))
    assert events == ["clock", "hash"]


def test_a_rounds_digest_is_the_blocks_sha512_row_v2_digest_in_a_92_byte_record():
    """`H(block)` is the inner digest of C's `hm96-sha512/row/v2` leaf, which `answer_circuit.PousAnswerOpens_v1` opens; a
    plain SHA-512 of the block is not, and the verdict counts a round carrying one as wrong."""
    b = block(5)
    assert mc.block_digest(b) == rowleaf.sha512_row_v2_digest(b, 8 * BLOCK_BYTES) == expected(5)
    assert mc.block_digest(b) != hashlib.sha512(b).digest()
    assert mc.block_digest(bytes(2056)) == hashlib.sha512(rowleaf.sha512_row_v2_prefix(16448) + bytes(2056)).digest()
    _, signed, signer = session()
    assert mc.RECORD_SCHEMA == "pous/round-record/v2"
    assert all(len(r.value()) == 92 and r.value()[-64:] == expected(r.index) for r in signed.rounds)
    plain = mc.MemoryChallenger(coins(), signer, log=ts.session_log(), limit_ns=LIMIT, block_bytes=BLOCK_BYTES,
                                binding=ROOT, clock=Clock())
    while not plain.done:
        i = plain.challenge()
        plain.receive(block(i))
        r = plain.rounds[-1]
        plain.rounds[-1] = dataclasses.replace(r, digest=hashlib.sha512(block(i)).digest())
    v = judge(plain, plain.close(), signer)
    assert not v.accepted and v.wrong == K


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


def test_no_collection_runs_inside_the_rounds() -> None:
    seen: list[bool] = []
    c, signed, signer = session(respond=lambda i: (seen.append(gc.isenabled()), block(i))[1])
    assert len(seen) == K and not any(seen) and gc.isenabled() and judge(c, signed, signer).accepted
    gc.disable()
    try:
        session()
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_a_clock_that_runs_backwards_ends_the_session_late_as_the_verdict_judges_it():
    c, signed, signer = session(spend=lambda j: -1 if j == 2 else 20_000)
    assert signed.ended == mc.LATE and len(signed.rounds) == 3 and signed.rounds[-1].elapsed_ns == -1
    v = judge(c, signed, signer)
    assert not v.accepted and v.late == 1


def test_openings_must_match_the_revealed_indices():
    c, signed, signer = session()
    opened = list(c.openings())
    opened[2] = coins().opening(3)
    assert judge(c, signed, signer, openings=opened).reasons == ("round 2's index is not its committed coin",)


def test_refusals():
    def open_(**kw):
        return mc.MemoryChallenger(coins(), ts.TestSigner(b"k", height=1), log=ts.session_log(), limit_ns=LIMIT,
                                   block_bytes=BLOCK_BYTES, clock=Clock(), **kw)

    with pytest.raises(ValueError, match="coarser than 1000 ns"):
        open_(binding=ROOT, resolution_ns=10_000)
    with pytest.raises(TypeError, match="binding"):
        open_()
    for binding in (b"", b"vk root", ROOT + b"\0"):
        with pytest.raises(ValueError, match="the setup's 64-byte root of C\\*"):
            open_(binding=binding)
    c = open_(binding=ROOT)
    with pytest.raises(RuntimeError, match="not ended"):
        c.close()
    c.challenge()
    with pytest.raises(RuntimeError, match="unanswered"):
        c.challenge()
    signer = ts.TestSigner(b"k", height=1)
    signer.sign(SESSION, b"a"), signer.sign(SESSION, b"b")
    with pytest.raises(lms.StateError, match="all 2\\^1 leaves are used"):
        signer.sign(SESSION, b"c")
    with pytest.raises(TypeError, match="log"):
        mc.MemoryChallenger(coins(), signer, limit_ns=LIMIT, block_bytes=BLOCK_BYTES, binding=ROOT, clock=Clock())


def test_a_session_id_is_used_once_and_on_disk_before_the_first_challenge(tmp_path, monkeypatch):
    """The session log: the id is appended and fsynced when the session commits, before its first challenge reads the
    clock, so a session that never runs has used it too; a second session under it is refused, by the same log or by
    the file opened again, and another id is not."""
    events: list[str] = []
    fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append("fsync"), fsync(fd))[1])

    class Seen(Clock):
        def __call__(self) -> int:
            events.append("clock")
            return self.now

    path = tmp_path / "sessions"
    mc.SessionLog.create(path)
    log, signer = mc.SessionLog(path), ts.TestSigner(b"k", height=2)

    def open_(log: mc.SessionLog, session: bytes = SESSION) -> mc.MemoryChallenger:
        return mc.MemoryChallenger(coins(session), signer, log=log, limit_ns=LIMIT, block_bytes=BLOCK_BYTES,
                                   binding=ROOT, clock=Seen())

    events.clear()
    c = open_(log)
    assert events == ["fsync"] and SESSION in log and path.read_text() == f"{mc.SESSION_LOG}\n{SESSION.hex()}\n"
    c.challenge()
    assert events == ["fsync", "clock"]
    for again in (log, mc.SessionLog(path)):
        with pytest.raises(mc.SessionLogError, match="runs one session under an id"):
            open_(again)
    other = hashlib.sha512(b"the next audit's id").digest()[:32]
    open_(mc.SessionLog(path), other)
    assert path.read_text() == f"{mc.SESSION_LOG}\n{SESSION.hex()}\n{other.hex()}\n"
    with pytest.raises(ValueError, match="32 bytes"):
        log.claim(SESSION[:31])


def test_a_session_log_that_is_missing_unreadable_or_made_twice_is_refused(tmp_path):
    """A missing log is never made in its place, a log is made once, and a log whose lines are not distinct whole ids
    (cut short, edited, an id twice) refuses to open, and refuses a claim when it changes so after opening."""
    with pytest.raises(mc.SessionLogError, match="missing"):
        mc.SessionLog(tmp_path / "none")
    path = tmp_path / "sessions"
    mc.SessionLog.create(path)
    with pytest.raises(mc.SessionLogError, match="never made again"):
        mc.SessionLog.create(path)
    head, a = mc.SESSION_LOG, SESSION.hex()
    for text in ("", f"{head}", "another/log\n", f"{head}\n{a}", f"{head}\n{a}\n{a}\n", f"{head}\n{a[:-1]}\n",
                 f"{head}\n{a.upper()}\n", f"{head}\n\n", f"{head}\n{a}\n\n"):
        bad = tmp_path / "bad"
        bad.write_text(text)
        with pytest.raises(mc.SessionLogError, match="unreadable"):
            mc.SessionLog(bad)
    log = mc.SessionLog(path)
    with open(path, "a") as f:
        f.write(a[:10])
    with pytest.raises(mc.SessionLogError, match="unreadable"):
        log.claim(SESSION)
    path.unlink()
    with pytest.raises(mc.SessionLogError, match="missing"):
        log.claim(SESSION)
    assert not path.exists()


def test_the_test_signer_leaves_nothing_outside_basetemp(tmp_path_factory, monkeypatch):
    """`pous_test_signer`'s leaf states and session logs are made under the suite's basetemp (`DIR`, the conftest's), which
    the check removes when the suite ends, or in a directory the caller passes, and never in `$TMPDIR`, where nothing
    removes them (`pous-session-log-*` and `pous-test-signer-*` held 82% of a GPU node's /workspace inodes);
    with neither, they refuse."""
    tmp = tmp_path_factory.mktemp("tmpdir")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    basetemp = tmp_path_factory.getbasetemp().resolve()
    signer, log = ts.TestSigner(b"k", height=1), ts.session_log()
    signer.sign(SESSION, b"a")
    log.claim(SESSION)
    assert list(tmp.iterdir()) == []
    assert all(p.resolve().is_relative_to(basetemp) for p in (signer.state, log.path))
    own = tmp_path_factory.mktemp("own")
    assert ts.session_log(own).path.parent.parent == ts.TestSigner(b"k", 1, own).state.parent.parent == own
    monkeypatch.setattr(ts, "DIR", None)
    for make in (lambda: ts.TestSigner(b"k", height=1), ts.session_log):
        with pytest.raises(RuntimeError, match="in a directory the caller owns"):
            make()
    assert list(tmp.iterdir()) == []


def test_the_certificate_covers_p2_v4_at_the_cap():
    from verity.core.protocols.pous.schemes import p2

    scheme = schemes.get("p2-16448/v4")
    limit = 500_000 + p2.RTT_ALLOWANCE_CAP_NS
    assert limit == 854_000 < p2.ROOT_CALL_NS
    cert = mc.covering_certificate(scheme, p2.CERT_K, limit, experimental=True)
    assert cert.k == p2.CERT_K == 105
    with pytest.raises(ValueError, match="experimental"):
        mc.covering_certificate(scheme, p2.CERT_K, limit)
    with pytest.raises(ValueError, match="no certificate"):
        mc.covering_certificate(scheme, p2.CERT_K, limit + 1_000, experimental=True)


class _SparseTree:
    """A deployed key's tree for leaf 0 alone: its path's siblings arbitrary, its root the one they give, so a test signs
    once at `lms.DEPLOYED` without its 2^20-leaf keygen."""

    def __init__(self, I: bytes, seed: bytes) -> None:
        self.root = b""
        probe = lms.sign_at(lms.PrivateKey(lms.DEPLOYED, I, seed, self), 0, b"probe")  # type: ignore[arg-type]
        self.root = lms.root_of(lms.DEPLOYED, I, b"probe", probe)

    def __getitem__(self, r: int) -> bytes:
        return self.root if r == 1 else hashlib.sha512(b"sibling%d" % r).digest()


def deployed_signer(tmp_path) -> mc.LmsSigner:
    I, seed = hashlib.sha512(b"I").digest()[:16], hashlib.sha512(b"seed").digest()[:lms.DEPLOYED.N]
    key = lms.PrivateKey(lms.DEPLOYED, I, seed, _SparseTree(I, seed))  # type: ignore[arg-type]
    lms.LeafState.create(tmp_path / "state", lms.public_key(key), lms.DEPLOYED.H)
    return mc.LmsSigner(key, tmp_path / "state")


def test_an_lms_signed_session_is_accepted_under_the_deployed_scheme(tmp_path):
    signer = deployed_signer(tmp_path)
    assert signer.scheme == mc.LMS and len(signer.public_key) == 16 + lms.DEPLOYED.N
    c, signed, _ = session(signer=signer)
    assert len(signed.signature) == lms.DEPLOYED.signature_bytes()
    assert judge(c, signed, signer, schemes_=mc.SCHEMES).public() == {"accepted": True}
    verify = mc.SCHEMES[mc.LMS]
    session_id, record = signed.commitment.session, signed.message()
    assert verify(signed.public_key, session_id, record, signed.signature)
    assert not verify(signed.public_key, hashlib.sha512(b"other").digest()[:32], record, signed.signature)
    assert not verify(signed.public_key[:-1], session_id, record, signed.signature)
    with pytest.raises(ValueError, match="an LMS signature"):
        verify(signed.public_key, session_id, record, signed.signature[:-1])
    plain = lms.sign_at(signer.key, 0, record).encode()
    assert not verify(signed.public_key, session_id, record, plain)


def test_the_deployed_scheme_id_is_lms_at_the_deployed_parameters():
    small = lms.keygen(lms.Params(64, 1, 1), bytes(64))
    with pytest.raises(ValueError, match="lms-sha512/n64-w1-h20/v0 is LMS at"):
        mc.LmsSigner(small, None)  # type: ignore[arg-type]
    big = lms.PrivateKey(lms.DEPLOYED, bytes(16), bytes(64), (b"", bytes(64)))
    with pytest.raises(ValueError, match="is LMS at"):
        mc.LmsSigner(big, None, scheme=ts.scheme(20))  # type: ignore[arg-type]
