"""`verity.primitives.crypto.lms`: RFC 8554's Test Case 2 with SHA-256 swapped in (keys from (I, SEED), C, every chain value
and path node), the parameter table, pinned SHA-512 vectors at the deployed shape (N = 32, W = 1) and a small tree, the roles'
digests, refusal of tampered signatures, and a leaf state that fails closed."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from verity.primitives.crypto import lms as R

FIXTURE = Path(__file__).with_name("lms_rfc8554_tc2.json")
SEED = bytes(range(32))
SMALL = R.Params(32, 1, 4)
#: SMALL's key from SEED: its public key, and the first 16 bytes of SHA-512 over the encoded signatures sign_at gives at
#: (q, message) for q = 0 (a certificate) and q = 15 (a memory challenger record)
PUBLIC = "e436778fb8460c2bbc30ccf0fdd50f38f80696ae37d3cbdb8ea9aab24f36a42271b73f281ae62b28a53b26ec927399e9"
CERTIFICATE = "d6548a0c2ed08a7279ad82ffe4647e8ef25171749d94b504cb0f4560c6a31fd2bce4e9ad113867f526a386ed1209ae7e48fa6a8bf960db0233be1001ef31deaf"
RECORD = "635304bea4a7f2a90ad9ee0ca6c49ff8a9bfb7a05c84a5b73c91ed7a480cb7f16acdaec3c008dd552334819d13698cbc01ac039e8fd98abfef1a155377539b81"
SIGNATURES = {0: "b2c0906514c9f7d9608decea0b9ea981", 15: "ec943d027d27c42ffa32ae24c3112c64"}


class _Sha256(R.Params):
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
    """P and LS for every (N, W) in RFC 8554 Appendix B's formulas (its Table 1 at N = 32)."""
    got = {(n, w): (R.Params(n, w, 1).P, R.Params(n, w, 1).LS) for n in (32, 64) for w in (1, 2, 4, 8)}
    assert got == {(32, 1): (265, 7), (32, 2): (133, 6), (32, 4): (67, 4), (32, 8): (34, 0),
                   (64, 1): (522, 6), (64, 2): (261, 6), (64, 4): (131, 4), (64, 8): (66, 0)}
    assert R.DEPLOYED == R.Params(32, 1, 20) and R.DEPLOYED.signature_bytes() == 9156


def test_pinned_vectors(small) -> None:
    assert R.public_key(small).hex() == PUBLIC
    msgs = {0: R.certificate_message(small.I, "responses", "egress", 7, b"record"),
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
    base = R.certificate_message(I, "l", "egress", 1, b"r")
    others = [R.certificate_message(I, "l", "ingress", 1, b"r"), R.certificate_message(I, "l", "egress", 2, b"r"),
              R.certificate_message(I, "m", "egress", 1, b"r"), R.certificate_message(bytes(16), "l", "egress", 1, b"r"),
              R.certificate_message(I, "l", "egress", 1, b"s"), R.memory_challenger_message(I, "l", b"r"),
              R.certificate_message(I, "l", "egress", 1, b"r\x00")]
    assert len(base) == 64 and len({base, *others}) == 1 + len(others)
    assert R.memory_challenger_message(I, "s", b"r") == R.memory_challenger_message(I, b"s", b"r")
    with pytest.raises(ValueError, match="direction"):
        R.certificate_message(I, "l", "up", 1, b"r")
    with pytest.raises(ValueError, match="u64"):
        R.certificate_message(I, "l", "egress", -1, b"r")


def test_tampered_signatures_are_refused(small) -> None:
    pk, m = R.public_key(small), R.certificate_message(small.I, "l", "egress", 3, b"r")
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


def test_a_state_changed_under_its_signer_is_refused(state, small, tmp_path) -> None:
    old = tmp_path / "old"
    shutil.copy(state, old)
    with _open(state, small) as st:
        R.sign(small, st, b"a")
        shutil.copy(old, state)
        with pytest.raises(R.StateError):
            R.sign(small, st, b"b")


def test_another_keys_state_and_an_existing_state_are_refused(state, small, tmp_path) -> None:
    other = R.keygen(SMALL, bytes(32))
    with pytest.raises(R.StateError, match="another key"):
        _open(state, other)
    with _open(state, small) as st:
        with pytest.raises(R.StateError, match="another key"):
            R.sign(other, st, b"a")
    with pytest.raises(R.StateError, match="already has"):
        R.LeafState.create(state, R.public_key(small), SMALL.H)


def test_every_leaf_is_used_once_then_signing_is_refused(tmp_path) -> None:
    p, k = tmp_path / "s", R.keygen(R.Params(32, 1, 1), SEED)
    R.LeafState.create(p, R.public_key(k), 1)
    with _open(p, k) as st:
        assert [R.sign(k, st, b"m").q for _ in range(2)] == [0, 1]
        with pytest.raises(R.StateError, match="used"):
            R.sign(k, st, b"m")
    assert os.path.getsize(p.with_name("s.journal")) == 4
