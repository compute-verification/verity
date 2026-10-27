"""A sealed pair link's AEAD (`verity.ml.crypto.aead`) and its two halves (`active.Sealer`, `active.Opener`): the
construction its docstring states, recomputed from `hashlib` and `hmac` alone; a unit that opens only
whole, at its own nonce, under its own direction's keys of its own session; a sender that never seals two frames at
one nonce; and a receiver that fails closed for good on the first unit that doesn't open."""

import hashlib
import hmac
import random

import pytest
from verity.ml.crypto import aead
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.grid import GridParams

SECRET, SESSION = bytes(range(aead.SECRET_BYTES)), b"session 1"
P = GridParams(buckets=4, allocation=2, frame_bytes=200)


def test_the_keys_and_the_seal_are_the_stated_construction():
    """The keys are HKDF-Expand's first block (RFC 5869) under the secret, over the label, the length-prefixed session
    and direction, and 0x01; the nonce is u64be window ‖ u32be bucket ‖ u32be slot; the keystream is
    SHA-512(K_enc ‖ N ‖ u64be c) for c = 0, 1, …, cut to the frame, each input 88 bytes, so one compression; the unit
    is C ‖ HMAC-SHA512(K_mac, N ‖ C). Sealing is deterministic, and a pinned digest holds the bytes."""
    info = len(SESSION).to_bytes(2, "big") + SESSION + len(b"a>b").to_bytes(2, "big") + b"a>b"
    enc = hmac.new(SECRET, b"verity-network-certifier/pair/v1/enc" + info + b"\x01", hashlib.sha512).digest()
    mac = hmac.new(SECRET, b"verity-network-certifier/pair/v1/mac" + info + b"\x01", hashlib.sha512).digest()
    k = aead.keys(SECRET, SESSION, "a>b")
    assert (k.enc, k.mac) == (enc, mac) and len(enc) == len(mac) == 64 and enc != mac
    assert "enc" not in repr(k) and enc.hex() not in repr(k)
    n = (5).to_bytes(8, "big") + (3).to_bytes(4, "big") + (1).to_bytes(4, "big")
    assert aead.nonce(5, 3, 1) == n and aead.NONCE_BYTES == len(n) == 16
    assert len(enc + n) + 8 == 88 <= 111  # one SHA-512 block holds 111 bytes of message
    frame = bytes(range(200))
    stream = b"".join(hashlib.sha512(enc + n + c.to_bytes(8, "big")).digest() for c in range(4))[:200]
    assert aead.keystream(enc, n, 200) == stream
    c = bytes(x ^ y for x, y in zip(frame, stream, strict=True))
    unit = aead.seal(k, n, frame)
    assert unit == c + hmac.new(mac, n + c, hashlib.sha512).digest() == aead.seal(k, n, frame)
    assert len(unit) == len(frame) + aead.TAG_BYTES == 264
    assert hashlib.sha512(unit).hexdigest()[:32] == PINNED


PINNED = "2cf78fe1184a9a3830731b7ea7911dd1"


@pytest.mark.parametrize("length", [0, 1, 63, 64, 65, 512])
def test_a_unit_opens_to_its_frame_and_never_with_a_bit_flipped_or_cut(length):
    """A sealed unit opens to its frame; with any one of its bits flipped (ciphertext or tag), a byte cut off or added,
    or shorter than a tag, it doesn't open."""
    k, n = aead.keys(SECRET, SESSION, "a>b"), aead.nonce(0, 1, 1)
    frame = random.Random(length).randbytes(length)
    unit = aead.seal(k, n, frame)
    assert aead.unseal(k, n, unit) == frame
    for i in range(8 * len(unit)):
        bad = bytearray(unit)
        bad[i // 8] ^= 1 << (i % 8)
        assert aead.unseal(k, n, bytes(bad)) is None, i
    for bad in (unit[:-1], unit[1:], unit + b"\0", b"\0" + unit, unit[: aead.TAG_BYTES - 1], b""):
        assert aead.unseal(k, n, bad) is None


def test_a_unit_opens_only_at_its_nonce_under_its_directions_keys_of_its_session():
    """A unit sealed at (window 1, bucket 2, slot 0) under direction a>b's keys of session 1 doesn't open at another
    slot, bucket or window, nor under the other direction's keys, another session's, another secret's, or keys whose
    session and direction run together into the same bytes. Two slots' keystreams differ, so one padding frame sealed
    at each gives units with nothing in common."""
    k, n = aead.keys(SECRET, SESSION, "a>b"), aead.nonce(1, 2, 0)
    unit = aead.seal(k, n, b"x" * 100)
    for other in (aead.nonce(1, 2, 1), aead.nonce(1, 3, 0), aead.nonce(2, 2, 0), aead.nonce(0, 0, 0)):
        assert aead.unseal(k, other, unit) is None
    others = [aead.keys(SECRET, SESSION, "b>a"), aead.keys(SECRET, b"session 2", "a>b"),
              aead.keys(bytes(aead.SECRET_BYTES), SESSION, "a>b"), aead.keys(SECRET, SESSION + b"a", ">b"),
              aead.keys(SECRET, SESSION[:-1], "1a>b")]
    assert len({(o.enc, o.mac) for o in [k, *others]}) == 1 + len(others)
    for other in others:
        assert aead.unseal(other, n, unit) is None
    pad = bytes(200)
    units = [aead.seal(k, aead.nonce(0, t, j), pad) for t in range(4) for j in range(2)]
    assert len({u[:64] for u in units}) == len(units)
    with pytest.raises(ValueError, match="secret is 64 bytes"):
        aead.keys(SECRET[:-1], SESSION, "a>b")
    with pytest.raises(ValueError, match="secret is 64 bytes"):
        aead.keys(SECRET.hex(), SESSION, "a>b")
    with pytest.raises(ValueError, match="session id is bytes"):
        aead.keys(SECRET, "session 1", "a>b")
    with pytest.raises(ValueError, match="nonce is 16 bytes"):
        aead.seal(k, n[:-1], b"x")
    with pytest.raises(ValueError, match="nonce is 16 bytes"):
        aead.unseal(k, n + b"\0", unit)


def test_the_sealer_seals_each_slot_at_its_nonce_once_and_in_order():
    """Slot j of bucket t (counted from window 0) is sealed at nonce (t // T, t mod T, j). A slot at or before the last
    sealed is refused, so no nonce is used twice under the sealer's keys; a frame of another size, a slot outside the
    bucket's r, a negative bucket and a bucket of other than r frames are refused."""
    k, f = aead.keys(SECRET, SESSION, "a>b"), bytes(range(200))
    s = active.Sealer(k, P)
    assert s.seal(5, 1, f) == aead.seal(k, aead.nonce(1, 1, 1), f)
    for t, j in ((5, 1), (5, 0), (4, 1), (0, 0)):
        with pytest.raises(RuntimeError, match="used once"):
            s.seal(t, j, f)
    assert s.seal_bucket(6, [f, bytes(200)]) == [aead.seal(k, aead.nonce(1, 2, j), g) for j, g in
                                                  enumerate((f, bytes(200)))]
    for t, j, frame in ((7, 0, f[:-1]), (7, 2, f), (7, -1, f), (-1, 0, f)):
        with pytest.raises(ValueError, match="sealed slot"):
            s.seal(t, j, frame)
    with pytest.raises(ValueError, match="2 slots"):
        s.seal_bucket(7, [f])
    secret = active.fresh_pair_secret()
    assert len(secret) == aead.SECRET_BYTES and secret != active.fresh_pair_secret()


def stream(k: aead.Keys, frames: list[bytes]) -> list[bytes]:
    """The units of `frames` as the sending certifier writes them: the i-th is slot i of its grid P from window 0."""
    s = active.Sealer(k, P)
    return [s.seal(*divmod(i, P.allocation), f) for i, f in enumerate(frames)]


def test_the_opener_opens_the_kth_unit_as_slot_k_and_fails_closed_for_good():
    """Three windows of units open in order to their frames, and from window 1 on to an opener back from maintenance
    there. A unit dropped, replayed, swapped with the next, cut or flipped makes that unit and every later one refused,
    the whole units at their own slots after it too; so do the sender's units opened under the other direction's
    keys."""
    k = aead.keys(SECRET, SESSION, "b>a")
    rng = random.Random(4)
    frames = [rng.randbytes(200) for _ in range(3 * P.buckets * P.allocation)]
    units = stream(k, frames)
    o = active.Opener(k, P)
    assert o.unit_bytes == 264 and [o.open(u) for u in units] == frames and not o.refused
    back = active.Opener(k, P, start=1)
    assert [back.open(u) for u in units[8:]] == frames[8:]
    flipped = bytearray(units[3])
    flipped[100] ^= 0x80
    cases = [(units[:3] + units[4:], 3), (units[:4] + [units[3]] + units[4:], 4),
             (units[:3] + [units[4], units[3]] + units[5:], 3), (units[:3] + [units[3][:-1]] + units[4:], 3),
             (units[:3] + [bytes(flipped)] + units[4:], 3), (units[:3] + [units[3] + b"\0"] + units[4:], 3)]
    for case, first_bad in cases:
        o = active.Opener(k, P)
        got = [o.open(u) for u in case]
        assert got[:first_bad] == frames[:first_bad] and got[first_bad:] == [None] * (len(case) - first_bad)
        assert o.refused and o.position == len(case)
    reflected = active.Opener(aead.keys(SECRET, SESSION, "a>b"), P)
    assert reflected.open(units[0]) is None and reflected.refused
