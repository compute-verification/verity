"""`verity.ml.crypto.transport`: the key schedule recomputed from `hashlib` and `hmac` alone (HKDF over SHA-512, the
imprint, the ratchet, the session in each mode, the traffic and AEAD keys, a record and an update), with a pinned digest
of a whole stream; a stream that comes back record for record across updates, each unit as long as its declared size
makes it; every flipped, cut, extended, replayed, skipped, reflected, foreign, injected or malformed unit refused by
name, and every unit after it; truncation, a unit after the close, an oversized body and a full epoch refused by name;
two sides' ratchets agreeing within the window and refusing past it; and the imprint's confirmation."""
from __future__ import annotations

import hashlib
import hmac
import random
import struct

import pytest

from verity.ml.crypto import aead
from verity.ml.crypto import transport as T

R_D, R_S = bytes(range(64)), bytes(range(64, 128))
ID_D, ID_S = b"device public key (dummy)", b"server public key (dummy)"
SS = bytes(range(100, 132))


def _hmac(k: bytes, x: bytes) -> bytes:
    return hmac.new(k, x, hashlib.sha512).digest()


def _label(s: bytes) -> bytes:
    full = b"verity/transport/v1/" + s
    return struct.pack(">H", len(full)) + full


def _expand(k: bytes, s: bytes, ctx: bytes = b"") -> bytes:
    return _hmac(k, _label(s) + struct.pack(">I", len(ctx)) + ctx + b"\x01")


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869's HKDF-Expand over SHA-512, every block."""
    out, t, i = b"", b"", 1
    while len(out) < length:
        t = _hmac(prk, t + info + bytes([i]))
        out, i = out + t, i + 1
    return out[:length]


def _record(kind: int, body: bytes, size: int) -> bytes:
    return bytes([kind]) + struct.pack(">I", len(body)) + body + bytes(size - 5 - len(body))


ROOT = T.imprint(R_D, R_S, ID_D, ID_S)
TRANSCRIPT = T.transcript(b"hello i", b"hello r and its signature", b"signature i")
SIGMA = T.session(TRANSCRIPT, key=ROOT)


def _refused(name: str, f, *args):
    with pytest.raises(T.Refused) as e:
        f(*args)
    assert e.value.name == name, e.value
    return e.value


def test_the_key_schedule_is_the_stated_construction():
    """Extract is HMAC(salt, ikm) and expand is HKDF-Expand's first block under the label and the length-prefixed
    context, whose info starts with 0x00 where `aead.keys`'s starts with ``ve``; the imprint, ratchet, transcript,
    session (the chain key's PRF, alone or XORed with the KEM secret's), traffic and AEAD keys are the stated HMACs; a
    record is its padded plaintext sealed at nonce u64be epoch ‖ u64be seq, and an update moves the secret to
    expand(s, "update") at epoch 1. A pinned digest holds a whole stream's bytes."""
    assert T.extract(b"salt", b"ikm") == _hmac(b"salt", b"ikm")
    info = _label(b"x") + struct.pack(">I", 3) + b"ctx"
    assert T.expand(R_D, "x", b"ctx") == _hmac(R_D, info + b"\x01") == _hkdf_expand(R_D, info, 64)
    assert info[0] == 0 and aead.ENC_LABEL[:2] == aead.MAC_LABEL[:2] == b"ve"

    ids = struct.pack(">H", len(ID_D)) + ID_D + struct.pack(">H", len(ID_S)) + ID_S
    assert ROOT == _expand(_hmac(R_D, R_S), b"root", ids)
    k1 = _expand(ROOT, b"ratchet")
    km, stored = T.advance(T.Chain(0, ROOT), 1)
    assert km == k1 and stored == T.Chain(2, _expand(k1, b"ratchet"))

    t = hashlib.sha512(_label(b"transcript") + b"".join(struct.pack(">I", len(m)) + m for m in (b"a", b"bb"))).digest()
    assert T.transcript(b"a", b"bb") == t
    by_key, by_kem = _expand(k1, b"session", t), _expand(SS, b"session kem", t)
    assert T.session(t, key=k1) == by_key
    assert T.session(t, key=k1, kem_secret=SS) == bytes(a ^ b for a, b in zip(by_key, by_kem, strict=True))

    s = _expand(by_key, b"traffic i>r")
    assert T.traffic(by_key, "i>r") == s and T.traffic(by_key, "r>i") == _expand(by_key, b"traffic r>i") != s
    k = T.keys(s)
    assert (k.enc, k.mac) == (_expand(s, b"enc"), _expand(s, b"mac"))
    assert T.nonce(1, 2) == struct.pack(">QQ", 1, 2) and len(T.nonce(0, 0)) == aead.NONCE_BYTES
    sender = T.Sender(s)
    assert sender.data(b"hello", 32) == aead.seal(k, bytes(16), _record(T.DATA, b"hello", 32))
    assert sender.update() == aead.seal(k, struct.pack(">QQ", 0, 1), _record(T.UPDATE, b"", 5))
    s1 = _expand(s, b"update")
    assert sender.data(b"", 5) == aead.seal(T.keys(s1), struct.pack(">QQ", 1, 0), _record(T.DATA, b"", 5))

    stream = T.Sender(T.traffic(SIGMA, "i>r"))
    units = [stream.data(bytes(range(n)), 64) for n in (0, 1, 59)] + [stream.update(), stream.data(b"z", 9),
                                                                       stream.close()]
    assert hashlib.sha512(b"".join(units)).hexdigest()[:32] == PINNED


PINNED = "bbe7cd03ab3c95e13ebe93beeee46957"


def test_a_stream_comes_back_record_for_record_across_updates_each_unit_its_declared_size():
    """Bodies of every length up to the size class come back in order across three updates, every unit is the declared
    size plus the tag whatever its body's length, and a stream ended by its close finishes."""
    s = T.traffic(SIGMA, "i>r")
    sender, receiver = T.Sender(s), T.Receiver(s)
    rng = random.Random(7)
    for i in range(64):
        if i % 16 == 15:
            assert receiver.open(sender.update()) is None
            continue
        body = rng.randbytes(rng.randrange(0, 300))
        unit = sender.data(body, 305)
        assert len(unit) == 305 + aead.TAG_BYTES
        assert receiver.open(unit) == body
    assert receiver.open(sender.close()) is None
    receiver.finish()
    assert (sender.epoch, receiver.epoch, sender.closed, receiver.closed) == (4, 4, True, True)


def _foreign(where: str, units: list[bytes], s: bytes) -> bytes:
    """A unit that isn't the one a receiver expects at position 1."""
    if where == "cut":
        return units[1][:-1]
    if where == "extended":
        return units[1] + b"\0"
    if where == "replayed":
        return units[0]
    if where == "skipped":
        return units[2]
    if where == "reflected":
        other = T.Sender(T.traffic(SIGMA, "r>i"))
        return [other.data(bytes([i]) * 10, 32) for i in range(2)][1]
    if where == "another session":
        other = T.Sender(T.traffic(T.session(T.transcript(b"another"), key=ROOT), "i>r"))
        return [other.data(bytes([i]) * 10, 32) for i in range(2)][1]
    if where == "injected":
        return random.Random(3).randbytes(len(units[1]))
    if where == "past a dropped update":
        other = T.Sender(s)
        other.data(bytes([0]) * 10, 32)
        other.update()
        return other.data(bytes([1]) * 10, 32)
    kind, body, pad = {"unknown type": (7, b"x", b""), "long length": (T.DATA, b"x", b""),
                       "nonzero padding": (T.DATA, b"x", b"\1"), "update with a body": (T.UPDATE, b"x", b"")}[where]
    plain = bytearray(_record(kind, body, 32))
    if where == "long length":
        plain[1:5] = struct.pack(">I", 28)
    if pad:
        plain[-1] = pad[0]
    return aead.seal(T.keys(s), T.nonce(0, 1), bytes(plain))


@pytest.mark.parametrize("where", ["cut", "extended", "replayed", "skipped", "reflected", "another session", "injected",
                                   "past a dropped update", "unknown type", "long length", "nonzero padding",
                                   "update with a body"])
def test_a_unit_out_of_place_is_refused_and_so_is_every_unit_after_it(where):
    """At position 1, a unit cut or extended, a replay of unit 0, unit 2 (a drop or a reordering), the same position of
    the other direction or of another session, random bytes, the next epoch's first unit (its update dropped), and
    well-sealed plaintexts that aren't records (an unknown type, a length past the record, nonzero padding, an update
    with a body) are refused (`record-refused`), and then so is the true unit 1."""
    s = T.traffic(SIGMA, "i>r")
    sender = T.Sender(s)
    units = [sender.data(bytes([i]) * 10, 32) for i in range(3)]
    receiver = T.Receiver(s)
    assert receiver.open(units[0]) == bytes([0]) * 10
    _refused("record-refused", receiver.open, _foreign(where, units, s))
    assert receiver.refused
    _refused("record-refused", receiver.open, units[1])


def test_a_unit_with_any_bit_flipped_is_refused():
    """Every one-bit change of a unit, in its ciphertext or its tag, is refused (`record-refused`)."""
    s = T.traffic(SIGMA, "i>r")
    unit = T.Sender(s).data(b"abc", 8)
    for i in range(8 * len(unit)):
        bad = bytearray(unit)
        bad[i // 8] ^= 1 << (i % 8)
        _refused("record-refused", T.Receiver(s).open, bytes(bad))
    assert T.Receiver(s).open(unit) == b"abc"


def test_truncation_a_unit_after_the_close_an_oversized_body_and_a_full_epoch_are_refused_by_name():
    """A stream that stops before its close is `truncated`; a unit after the close (sealed at the next position under
    the right keys) and a send after the close are `after-close`; a body past its size is `oversized`; data in an epoch's
    last slot is `epoch-full`, while the update that ends the epoch goes through and opens."""
    s = T.traffic(SIGMA, "i>r")
    sender, receiver = T.Sender(s), T.Receiver(s)
    receiver.open(sender.data(b"x", 8))
    _refused("truncated", receiver.finish)

    sender, receiver = T.Sender(s), T.Receiver(s)
    receiver.open(sender.close())
    receiver.finish()
    _refused("after-close", sender.data, b"x", 8)
    _refused("after-close", receiver.open, aead.seal(T.keys(s), T.nonce(0, 1), _record(T.DATA, b"x", 8)))
    _refused("record-refused", receiver.open, b"")

    sender = T.Sender(s)
    assert len(sender.data(b"abc", 8)) == 8 + aead.TAG_BYTES
    _refused("oversized", sender.data, b"abcd", 8)

    sender, receiver = T.Sender(s), T.Receiver(s)
    sender.seq = receiver.seq = T.EPOCH_RECORDS - 1
    _refused("epoch-full", sender.data, b"", 5)
    assert receiver.open(sender.update()) is None
    assert receiver.open(sender.data(b"next", 9)) == b"next" and (receiver.epoch, receiver.seq) == (1, 1)


def test_two_sides_agree_on_the_ratchet_within_the_window_and_refuse_past_it():
    """In step, both key the session from K_n and store (n + 1, K_{n+1}), which isn't K_n. When the responder never got
    the last handshake message, the initiator is one ahead and the next handshake agrees on the initiator's key, the
    responder stepping forward. Indices `WINDOW` apart still agree; one more, or a spent ratchet, is
    `ratchet-mismatch`."""
    i, r = T.Chain(0, ROOT), T.Chain(0, ROOT)
    ki, i1 = T.advance(i, r.index)
    kr, r1 = T.advance(r, i.index)
    assert ki == kr == ROOT and i1 == r1 == T.Chain(1, _expand(ROOT, b"ratchet")) and i1.key != ki

    ki, i2 = T.advance(i1, r.index)
    kr, r2 = T.advance(r, i1.index)
    assert ki == kr == i1.key and i2 == r2 == T.Chain(2, _expand(i1.key, b"ratchet"))

    k = ROOT
    for _ in range(T.WINDOW):
        k = _expand(k, b"ratchet")
    assert T.advance(T.Chain(0, ROOT), T.WINDOW)[0] == T.advance(T.Chain(T.WINDOW, k), 0)[0] == k
    _refused("ratchet-mismatch", T.advance, T.Chain(0, ROOT), T.WINDOW + 1)
    _refused("ratchet-mismatch", T.advance, T.Chain(T.WINDOW + 1, ROOT), 0)
    _refused("ratchet-mismatch", T.advance, T.Chain(2**64 - 1, ROOT), 2**64 - 1)
    with pytest.raises(ValueError):
        T.Chain(2**64, ROOT)
    with pytest.raises(ValueError):
        T.Chain(0, ROOT[:32])


def test_the_imprint_binds_both_ids_and_both_randomnesses_and_refuses_a_mismatch():
    """The root changes with either side's randomness or the ids' roles; each end's confirmation is expand(root,
    "imprint"), and a confirmation from a root derived from other bytes is `imprint-mismatch`."""
    assert len({ROOT, T.imprint(R_S, R_D, ID_D, ID_S), T.imprint(R_D, R_S, ID_S, ID_D),
                T.imprint(R_D, bytes(64), ID_D, ID_S)}) == 4
    assert T.check_imprint(ROOT, _expand(ROOT, b"imprint")) == _expand(ROOT, b"imprint")
    other = T.imprint(R_D, bytes(64), ID_D, ID_S)
    _refused("imprint-mismatch", T.check_imprint, ROOT, T.check_imprint(other, _expand(other, b"imprint")))
    with pytest.raises(ValueError):
        T.imprint(R_D[:32], R_S, ID_D, ID_S)


def test_the_session_modes_and_the_two_ends():
    """The chain key alone (onsite) and the chain key with an ML-KEM-1024 secret (remote) give different sessions, each
    bound to the transcript and to the KEM secret; there is no session on a KEM secret alone, and a KEM
    secret is exactly 32 bytes. The initiator's and the responder's ends talk in both directions, and no end opens what
    it sent. No repr shows a key."""
    sessions = {T.session(TRANSCRIPT, key=ROOT), T.session(TRANSCRIPT, key=ROOT, kem_secret=SS),
                T.session(TRANSCRIPT, key=ROOT, kem_secret=SS[::-1]), T.session(T.transcript(b"other"), key=ROOT)}
    assert len(sessions) == 4
    with pytest.raises(ValueError, match="chain key"):
        T.session(TRANSCRIPT, None, kem_secret=SS)
    with pytest.raises(TypeError):
        T.session(TRANSCRIPT, kem_secret=SS)  # type: ignore[call-arg]
    for bad in (SS[:31], SS + b"\x00"):
        with pytest.raises(ValueError):
            T.session(TRANSCRIPT, key=ROOT, kem_secret=bad)
    with pytest.raises(ValueError):
        T.traffic(SIGMA, "a>b")

    i_out, i_in = T.ends(SIGMA, initiator=True)
    r_out, r_in = T.ends(SIGMA, initiator=False)
    assert r_in.open(i_out.data(b"to the responder", 32)) == b"to the responder"
    assert i_in.open(r_out.data(b"to the initiator", 32)) == b"to the initiator"
    _refused("record-refused", i_in.open, i_out.data(b"back to myself", 32))

    assert repr(T.Chain(0, ROOT)) == "Chain(index=0, key=…)" and repr(T.keys(SIGMA)) == "Keys(…)"
    assert repr(i_out) == "Sender(epoch=0, seq=2, closed=False)"
    assert repr(r_in) == "Receiver(epoch=0, seq=1, closed=False)"
