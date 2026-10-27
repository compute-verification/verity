"""`verity.ml.crypto.lms`: RFC 8554's Test Case 2 with SHA-256 swapped in (keys from (I, SEED), C, every chain value
and path node), the parameter table, pinned SHA-512 vectors at the deployed shape (N = 64, W = 1) and a small tree, the roles'
digests, refusal of tampered signatures, and a leaf state that fails closed."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import queue
import shutil
import threading
from pathlib import Path

import pytest

from verity.ml.crypto import lms as R

FIXTURE = Path(__file__).with_name("lms_rfc8554_tc2.json")
SEED = bytes(range(64))
SMALL = R.Params(64, 1, 4)
#: SMALL's key from SEED: its public key, and the first 16 bytes of SHA-512 over the encoded signatures sign_at gives at
#: (q, message) for q = 0 (a certificate) and q = 15 (a memory challenger record)
PUBLIC = "326497cc0d59b0a4b1cebb5b8e7786372ccf46cb21f1a8b37ca0d33c24e43004743eacee9765553fc4097cd7b3e1af1d71c372488ea4bdb6b63bd8d48de09b676f1760513adca36d48ea256bac3edeb6"
CERTIFICATE = "2938e044ad925de2abad3ecd70b4af8cba031498ffb0e79fee74138a19f130bee81651e4aba07b823d9addde6c039be951ea4172ba38f46275fc9232c2273c37"
RECORD = "7e50764d622125f1e625bf8af2a01c86545171c2cf8f6d9e20adc9ec41bbb1701ee99c3fc97f57de3c6c7faef614f6b63b94a4acdd5150c5b608948fcd594562"
SIGNATURES = {0: "35983e360efbad1fabb2b44b0b67b10e", 15: "5c2f9cf77f3a00a187acb34605474f5a"}
SESSION = bytes(range(32))


class _Sha256(R.Params):
    """RFC 8554's LMS_SHA256_M32 hash: N = 32, which `R.Params` admits only here, for the RFC's vectors and table."""

    def __post_init__(self) -> None:
        if self.N != 32:
            raise ValueError(f"SHA-256's N is 32, not {self.N}")

    def hash(self, data: bytes) -> bytes:
        return hashlib.sha256(data).digest()


@pytest.fixture(scope="module")
def small() -> R.PrivateKey:
    return R.keygen(SMALL, SEED)


def _sig(s: dict) -> R.Signature:
    return R.Signature(s["q"], bytes.fromhex(s["C"]), tuple(map(bytes.fromhex, s["y"])), tuple(map(bytes.fromhex, s["path"])))


@pytest.mark.parametrize("level", ["bottom", "top"])
def test_rfc8554_test_case_2(level) -> None:
    """The RFC's two trees (W = 8, H = 5 and W = 4, H = 10) from their (I, SEED): the root, and the signature bit for bit."""
    t = json.loads(FIXTURE.read_text())[level]
    p = _Sha256(32, t["W"], t["H"])
    k = R.key(p, bytes.fromhex(t["I"]), bytes.fromhex(t["seed"]))
    assert k.root.hex() == t["root"]
    want, m = _sig(t["signature"]), bytes.fromhex(t["message"])
    assert R.sign_at(k, want.q, m) == want
    assert R.verify(p, R.public_key(k), m, want)


def test_the_parameter_table_is_rfc8554s() -> None:
    """P and LS for every (N, W) in RFC 8554 Appendix B's formulas (its Table 1 at N = 32, SHA-256's), the deployed
    parameters and their signature size, and no N but 64 (or the toy, by name) over SHA-512."""
    got = {(n, w): (p.P, p.LS) for n in (32, 64) for w in (1, 2, 4, 8) for p in [(_Sha256 if n == 32 else R.Params)(n, w, 1)]}
    assert got == {(32, 1): (265, 7), (32, 2): (133, 6), (32, 4): (67, 4), (32, 8): (34, 0),
                   (64, 1): (522, 6), (64, 2): (261, 6), (64, 4): (131, 4), (64, 8): (66, 0)}
    assert R.DEPLOYED == R.Params(64, 1, 20) and R.DEPLOYED.signature_bytes() == 34756
    assert R.DEPLOYED.hash(b"x") == hashlib.sha512(b"x").digest()
    for n in (32, R.TOY_N, 48):
        with pytest.raises(ValueError):
            R.Params(n, 1, 1)
    assert R.Params(R.TOY_N, 1, 1, toy=True).N == R.TOY_N


def test_pinned_vectors(small) -> None:
    assert R.public_key(small).hex() == PUBLIC
    msgs = {0: R.certificate_message(small.I, SESSION, "responses", "egress", 7, b"record"),
            15: R.memory_challenger_message(small.I, "session-1", b"record")}
    assert (msgs[0].hex(), msgs[15].hex()) == (CERTIFICATE, RECORD)
    for q, m in msgs.items():
        s = R.sign_at(small, q, m)
        assert hashlib.sha512(s.encode()).hexdigest()[:32] == SIGNATURES[q]
        assert R.Signature.decode(SMALL, s.encode()) == s and R.verify(SMALL, R.public_key(small), m, s)


def test_keygen_in_workers_is_the_same_key(small) -> None:
    assert R.keygen(SMALL, SEED, workers=2) == small


def test_role_digests_separate_roles_and_contexts(small) -> None:
    I = small.I
    S = SESSION
    base = R.certificate_message(I, S, "l", "egress", 1, b"r")
    others = [R.certificate_message(I, S, "l", "ingress", 1, b"r"), R.certificate_message(I, S, "l", "egress", 2, b"r"),
              R.certificate_message(I, S, "m", "egress", 1, b"r"), R.certificate_message(bytes(16), S, "l", "egress", 1, b"r"),
              R.certificate_message(I, S, "l", "egress", 1, b"s"), R.memory_challenger_message(I, "l", b"r"),
              R.certificate_message(I, S, "l", "egress", 1, b"r\x00"),
              R.certificate_message(I, bytes(32), "l", "egress", 1, b"r")]
    assert len(base) == 64 and len({base, *others}) == 1 + len(others)
    assert R.ROLES["network-certificate"] == "verity/network-certificate-message/v2"
    assert R.memory_challenger_message(I, "s", b"r") == R.memory_challenger_message(I, b"s", b"r")
    with pytest.raises(ValueError, match="direction"):
        R.certificate_message(I, S, "l", "up", 1, b"r")
    with pytest.raises(ValueError, match="u64"):
        R.certificate_message(I, S, "l", "egress", -1, b"r")
    for n in (0, 31, 33):
        with pytest.raises(ValueError, match="session id is 32 bytes"):
            R.certificate_message(I, bytes(n), "l", "egress", 1, b"r")


def test_tampered_signatures_are_refused(small) -> None:
    pk, m = R.public_key(small), R.certificate_message(small.I, SESSION, "l", "egress", 3, b"r")
    s = R.sign_at(small, 5, m)
    flip = lambda b: bytes([b[0] ^ 1]) + b[1:]  # noqa: E731
    bad = [R.Signature(6, s.C, s.y, s.path), R.Signature(s.q, flip(s.C), s.y, s.path),
           R.Signature(s.q, s.C, (flip(s.y[0]),) + s.y[1:], s.path), R.Signature(s.q, s.C, s.y, s.path[:2] + (flip(s.path[2]),) + s.path[3:]),
           R.Signature(s.q, s.C, s.y[:-1], s.path), R.Signature(2 ** SMALL.H, s.C, s.y, s.path)]
    assert R.verify(SMALL, pk, m, s)
    assert not any(R.verify(SMALL, pk, m, b) for b in bad)
    assert not R.verify(SMALL, pk, flip(m), s) and not R.verify(SMALL, flip(pk), m, s)
    with pytest.raises(ValueError, match="bytes"):
        R.Signature.decode(SMALL, s.encode()[:-1])


# --- the leaf state ----------------------------------------------------------------------------------------------------------

@pytest.fixture
def state(tmp_path, small) -> Path:
    p = tmp_path / "leaf.state"
    R.LeafState.create(p, R.public_key(small), SMALL.H)
    return p


def _open(p: Path, k: R.PrivateKey, **kw) -> R.LeafState:
    return R.LeafState(p, R.public_key(k), k.params.H, **kw)


def test_signing_advances_the_state_on_disk_before_it_returns(state, small) -> None:
    with _open(state, small) as st:
        s0, s1 = R.sign(small, st, b"a"), R.sign(small, st, b"b")
        assert (s0.q, s1.q) == (0, 1) and json.loads(state.read_text())["next"] == 2
    assert R.verify(SMALL, R.public_key(small), b"a", s0)


def test_a_restart_continues_after_the_last_leaf(state, small) -> None:
    with _open(state, small) as st:
        R.sign(small, st, b"a")
    with _open(state, small) as st:
        assert R.sign(small, st, b"b").q == 1


def test_a_missing_or_unreadable_state_is_refused(state, small, tmp_path) -> None:
    with pytest.raises(R.StateError, match="missing"):
        _open(tmp_path / "nothing", small)
    journal = state.with_name(state.name + ".journal")
    saved = journal.read_bytes()
    journal.unlink()
    with pytest.raises(R.StateError, match="missing"):
        _open(state, small)
    journal.write_bytes(saved)
    for text in ("", "{", '{"next": 0}', state.read_text().replace('"next": 0', '"next": 9')):
        good = state.read_text()
        state.write_text(text)
        with pytest.raises(R.StateError, match="unreadable"):
            _open(state, small)
        state.write_text(good)
    journal.write_text("x\n")
    with pytest.raises(R.StateError, match="unreadable"):
        _open(state, small)


def test_a_rolled_back_state_is_refused(state, small, tmp_path) -> None:
    old = tmp_path / "old"
    shutil.copy(state, old)
    with _open(state, small) as st:
        R.sign(small, st, b"a")
        R.sign(small, st, b"b")
    shutil.copy(old, state)
    with pytest.raises(R.StateError, match="already used"):
        _open(state, small)


def test_a_state_restored_with_its_journal_is_refused_below_the_floor(state, small, tmp_path) -> None:
    """Both files from one snapshot look like a fresh state; the caller's floor (1 + the last q it published) refuses it."""
    snap = tmp_path / "snap"
    snap.mkdir()
    for f in (state, state.with_name(state.name + ".journal")):
        shutil.copy(f, snap / f.name)
    with _open(state, small) as st:
        last = R.sign(small, st, b"a").q
    for f in snap.iterdir():
        shutil.copy(f, state.parent / f.name)
    with pytest.raises(R.StateError, match="floor"):
        _open(state, small, floor=last + 1)


def test_a_crash_between_the_state_and_its_journal_skips_one_leaf(state, small) -> None:
    with _open(state, small) as st:
        R.sign(small, st, b"a")
        R.sign(small, st, b"b")
    journal = state.with_name(state.name + ".journal")
    journal.write_text("0\n")
    with _open(state, small) as st:
        assert R.sign(small, st, b"c").q == 2
    journal.write_text("")
    with pytest.raises(R.StateError, match="behind its state"):
        _open(state, small)


def test_a_second_signer_on_one_state_is_refused(state, small) -> None:
    with _open(state, small) as st:
        with pytest.raises(R.StateError, match="held by another signer"):
            _open(state, small)
        R.sign(small, st, b"a")
    with _open(state, small) as st:
        assert R.sign(small, st, b"b").q == 1


def test_a_waiting_signer_takes_its_turn_at_the_next_leaf(state, small, monkeypatch) -> None:
    """With ``wait`` a second signer on a held state waits on its lock rather than refusing, and once the holder closes
    the state it signs at the leaf after the holder's. `flock` is patched to say "waiting" when a blocking lock has to
    wait, so the holder signs and lets go only once the second signer waits, with no clock."""
    real, said, got = fcntl.flock, queue.SimpleQueue(), []

    def flock(f, op):
        if op == fcntl.LOCK_EX:
            try:
                return real(f, op | fcntl.LOCK_NB)
            except BlockingIOError:
                said.put("waiting")
        return real(f, op)

    def second() -> None:
        try:
            with _open(state, small, wait=True) as st:
                got.append(R.sign(small, st, b"b").q)
        except Exception as e:  # noqa: BLE001: the test reads it below
            got.append(e)
        said.put("done")

    monkeypatch.setattr(fcntl, "flock", flock)
    holder = _open(state, small)
    t = threading.Thread(target=second)
    t.start()
    try:
        first = said.get()
        signed = R.sign(small, holder, b"a").q
    finally:
        holder.close()
        t.join()
    rest = [said.get() for _ in range(said.qsize())]
    assert (first, signed, got, rest) == ("waiting", 0, [1], ["done"])
    assert state.with_name(state.name + ".journal").read_text().split() == ["0", "1"]


def test_a_state_changed_under_its_signer_is_refused(state, small, tmp_path) -> None:
    old = tmp_path / "old"
    shutil.copy(state, old)
    with _open(state, small) as st:
        R.sign(small, st, b"a")
        shutil.copy(old, state)
        with pytest.raises(R.StateError):
            R.sign(small, st, b"b")


def test_another_keys_state_and_an_existing_state_are_refused(state, small, tmp_path) -> None:
    other = R.keygen(SMALL, bytes(64))
    with pytest.raises(R.StateError, match="another key"):
        _open(state, other)
    with _open(state, small) as st:
        with pytest.raises(R.StateError, match="another key"):
            R.sign(other, st, b"a")
    with pytest.raises(R.StateError, match="already has"):
        R.LeafState.create(state, R.public_key(small), SMALL.H)


def test_every_leaf_is_used_once_then_signing_is_refused(tmp_path) -> None:
    p, k = tmp_path / "s", R.keygen(R.Params(64, 1, 1), SEED)
    R.LeafState.create(p, R.public_key(k), 1)
    with _open(p, k) as st:
        assert [R.sign(k, st, b"m").q for _ in range(2)] == [0, 1]
        with pytest.raises(R.StateError, match="used"):
            R.sign(k, st, b"m")
    assert os.path.getsize(p.with_name("s.journal")) == 4


def test_a_session_is_started_once_on_disk(state, small) -> None:
    """`start` records the session id before it returns; a signer starts one session, and no later signer of the state
    starts it again, after a restart or once other sessions have run."""
    a, b = bytes(32), b"\x01" * 32
    with _open(state, small) as st:
        st.start(a)
        st.start(a)
        assert json.loads(state.read_text())["sessions"] == [a.hex()]
        with pytest.raises(R.StateError, match="already started a session"):
            st.start(b)
        assert R.sign(small, st, b"m").q == 0
    with _open(state, small) as st:
        with pytest.raises(R.StateError, match="already started"):
            st.start(a)
        st.start(b)
        assert R.sign(small, st, b"m").q == 1
    with _open(state, small) as st:
        for s in (a, b):
            with pytest.raises(R.StateError, match="already started"):
                st.start(s)
    assert json.loads(state.read_text())["sessions"] == [a.hex(), b.hex()]


def test_a_state_without_its_sessions_is_refused(state, small) -> None:
    body = json.loads(state.read_text())
    del body["sessions"]
    state.write_text(json.dumps(body))
    with pytest.raises(R.StateError, match="unreadable"):
        _open(state, small)
