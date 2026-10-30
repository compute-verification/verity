"""Confidential channels between an auditor party and the prover: the key schedule and the record layer over `aead`.
The handshake's messages, its LMS signatures, its ML-KEM-1024 encapsulation and the channel's framing are the deployment
layer's; this module takes the handshake's transcript and gives each direction a sender and a receiver.

Every channel has a SHA-512 chain key. An onsite device's is its imprint key, a key class of its own:
installed over the cable once per auditor device, then ratcheted. A remote party with no cable holds a chain key
installed at provisioning, and each of its sessions also XORs in an ML-KEM-1024 shared secret (``ind-cca/ml-kem-1024``,
the one-hash rule's transport exception), so its security never drops below the chain key's alone. No session runs on a
KEM secret alone, and the certifiers' pair links never use this module.

H is SHA-512, HMAC is HMAC-SHA512 (RFC 2104), [x] is u16be len(x) ‖ x and ⟨x⟩ is u32be len(x) ‖ x.

- **Derivation** (RFC 5869 over SHA-512): ``extract(salt, ikm) = HMAC(salt, ikm)``, and
  ``expand(k, s, ctx) = HMAC(k, [LABEL ‖ s] ‖ ⟨ctx⟩ ‖ 0x01)``, HKDF-Expand's first 64-byte block. Every info here starts
  with a 0x00 byte and every info of `aead.keys` with ``ve``, so no input is shared with the certifier's pair links.
- **Imprint** (`imprint`, `check_imprint`): over the cable between a device D and its server S, before D forwards any
  traffic, each sends 64 random bytes; ``K_0 = expand(extract(R_D, R_S), "root", [id_D] ‖ [id_S])``, and the two compare
  ``expand(K_0, "imprint")``. K_0 is secret while nothing else reads the cable during that exchange.
- **Ratchet** (`Chain`, `advance`): ``K_{n+1} = expand(K_n, "ratchet")``. A handshake carries both sides' indices;
  they agree on m, the larger, when the two differ by at most `WINDOW` (a dropped last handshake message leaves the
  sides one apart), the side behind steps forward, the session is keyed from K_m, and each side stores (m + 1, K_{m+1})
  in place of its state before its first record. A key once replaced opens no earlier session.
- **Session** (`transcript`, `session`, `traffic`, `ends`): σ from the transcript hash T and the channel's secrets:
  ``expand(K_m, "session", T)`` onsite, and for a remote party its XOR with ``expand(ss, "session kem", T)``, ss the
  session's ML-KEM-1024 shared secret. Each PRF reads T, which holds the KEM's ciphertext, so σ is pseudorandom while
  either secret is (PRF-then-XOR; Giacon, Heuer, Poettering 2018). Direction d (``i>r`` or ``r>i``) starts at
  ``s_d = expand(σ, "traffic " ‖ d)``, and its keys at secret s are ``aead.Keys(expand(s, "enc"), expand(s, "mac"))``.
- **Records** (`Sender`, `Receiver`): the plaintext ``u8 type ‖ u32be len(body) ‖ body``, zero-padded to the frame's
  declared size, sealed by `aead.seal` at the implicit nonce ``u64be epoch ‖ u64be seq``. A record is data, an update
  (then ``s ← expand(s, "update")``, epoch + 1 and seq 0, as TLS 1.3's KeyUpdate) or the close. A receiver opens each
  unit only at its own position, so a replayed, reordered, dropped, reflected or injected unit is refused, and so is
  every unit after it; a stream that ends without its close is truncated.

`aead`'s tag covers N ‖ C and N has one length, so units of different sizes under one key are sound. A size is public
structure (the frame's declared size class), never a private value's length.

Every key is an HMAC-SHA512 output under a uniform key, so a PRF under ``prf/sha-512`` (Bellare 2006), and a KEM's
shared secret enters only as an HMAC key. Standard library only. The premises are named Props in
``Security/Definitions/Core/Assumptions/Transport.lean``: ``WitnessPathPrivate`` for a channel without keys,
``ImprintPathPrivate`` for the cable during the imprint, and ``ChannelHides`` and ``ChannelAuthentic`` for what this
module gives; ``Verity.Guarantees.TransportPassiveZk`` takes the first or the third.
"""
from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass

from verity.ml.crypto import aead

__all__ = ["CLOSE", "DATA", "DIRECTIONS", "EPOCH_RECORDS", "HEADER", "KEM_SECRET_BYTES", "KEY_BYTES", "LABEL",
           "REFUSALS", "UPDATE", "WINDOW", "Chain", "Receiver", "Refused", "Sender", "advance", "check_imprint", "ends", "expand", "extract",
           "imprint", "keys", "nonce", "session", "traffic", "transcript"]

LABEL = b"verity/transport/v1/"
KEY_BYTES = 64
DIRECTIONS = ("i>r", "r>i")
DATA, UPDATE, CLOSE = 0, 1, 2
HEADER = struct.Struct(">BI")
NONCE = struct.Struct(">QQ")
#: records per epoch, the last kept for an update or the close
EPOCH_RECORDS = 1 << 32
#: how far apart two sides' ratchet indices may be and still agree
WINDOW = 1 << 16
#: an ML-KEM-1024 shared secret's length (FIPS 203)
KEM_SECRET_BYTES = 32
REFUSALS = ("imprint-mismatch", "ratchet-mismatch", "oversized", "epoch-full", "record-refused", "after-close",
            "truncated")
_U64 = 1 << 64


class Refused(ValueError):
    """A refusal by name: `name` is one of `REFUSALS`."""

    def __init__(self, name: str, detail: str) -> None:
        assert name in REFUSALS, name
        super().__init__(f"{name}: {detail}")
        self.name = name


def _field(x: bytes) -> bytes:
    if len(x) >= 1 << 16:
        raise ValueError("a label or an id is under 64 KiB")
    return struct.pack(">H", len(x)) + x


def _long(x: bytes) -> bytes:
    if len(x) >= 1 << 32:
        raise ValueError("a context is under 4 GiB")
    return struct.pack(">I", len(x)) + x


def _key(x: bytes, what: str) -> bytes:
    if not (isinstance(x, bytes) and len(x) == KEY_BYTES):
        raise ValueError(f"{what} is {KEY_BYTES} bytes")
    return x


def extract(salt: bytes, ikm: bytes) -> bytes:
    """HKDF-Extract over SHA-512: HMAC(salt, ikm)."""
    return hmac.digest(salt, ikm, "sha512")


def expand(key: bytes, label: str, ctx: bytes = b"") -> bytes:
    """HKDF-Expand's first block over SHA-512, info [LABEL ‖ label] ‖ ⟨ctx⟩: HMAC(key, info ‖ 0x01)."""
    return hmac.digest(key, _field(LABEL + label.encode()) + _long(ctx) + b"\x01", "sha512")


def imprint(r_device: bytes, r_server: bytes, device: bytes, server: bytes) -> bytes:
    """K_0 of the channel between `device` and `server` (their registered public keys) from the 64 random bytes each
    sent over the cable."""
    ikm = extract(_key(r_device, "a device's imprint randomness"), _key(r_server, "a server's imprint randomness"))
    return expand(ikm, "root", _field(device) + _field(server))


def check_imprint(root: bytes, peer: bytes) -> bytes:
    """This side's confirmation of `root`; refuses (`imprint-mismatch`) the peer's confirmation `peer` unless equal."""
    mine = expand(_key(root, "an imprint root"), "imprint")
    if not hmac.compare_digest(mine, peer):
        raise Refused("imprint-mismatch", "the two ends of the cable derived different roots")
    return mine


@dataclass(frozen=True, slots=True)
class Chain:
    """One side's stored ratchet state for a channel: index n and key K_n."""

    index: int
    key: bytes

    def __post_init__(self) -> None:
        if not (isinstance(self.index, int) and 0 <= self.index < _U64):
            raise ValueError(f"a ratchet index is a u64, not {self.index!r}")
        _key(self.key, "a ratchet key")

    def __repr__(self) -> str:
        return f"Chain(index={self.index}, key=…)"


def advance(stored: Chain, peer: int) -> tuple[bytes, Chain]:
    """The handshake's K_m, m the larger of `stored.index` and the peer's index `peer`, and the state (m + 1, K_{m+1})
    to store in place of `stored` before the first record. Refuses (`ratchet-mismatch`) indices over `WINDOW` apart."""
    if not (isinstance(peer, int) and 0 <= peer < _U64):
        raise ValueError(f"a ratchet index is a u64, not {peer!r}")
    if abs(stored.index - peer) > WINDOW:
        raise Refused("ratchet-mismatch", f"indices {stored.index} and {peer} are more than {WINDOW} apart")
    m, k = max(stored.index, peer), stored.key
    if m + 1 >= _U64:
        raise Refused("ratchet-mismatch", "the ratchet is spent")
    for _ in range(m - stored.index):
        k = expand(k, "ratchet")
    return k, Chain(m + 1, expand(k, "ratchet"))


def transcript(*messages: bytes) -> bytes:
    """T: H([LABEL ‖ "transcript"] ‖ ⟨M1⟩ ‖ ⟨M2⟩ ‖ …) over the handshake's messages."""
    return hashlib.sha512(_field(LABEL + b"transcript") + b"".join(map(_long, messages))).digest()


def session(t: bytes, key: bytes, kem_secret: bytes | None = None) -> bytes:
    """σ of the handshake whose transcript hash is `t`, from the ratchet's K_m (`key`), XORed for a remote party with
    the PRF of its ML-KEM-1024 shared secret `kem_secret`. There is no session without `key`."""
    if not (isinstance(t, bytes) and len(t) == 64):
        raise ValueError("a transcript hash is 64 bytes")
    if key is None:
        raise ValueError("every session has its chain key; a KEM's shared secret is only ever XORed in beside it")
    sigma = expand(_key(key, "a session's ratchet key"), "session", t)
    if kem_secret is None:
        return sigma
    if not (isinstance(kem_secret, bytes) and len(kem_secret) == KEM_SECRET_BYTES):
        raise ValueError(f"an ML-KEM-1024 shared secret is {KEM_SECRET_BYTES} bytes")
    return bytes(a ^ b for a, b in zip(sigma, expand(kem_secret, "session kem", t), strict=True))


def traffic(sigma: bytes, direction: str) -> bytes:
    """Direction `direction`'s first traffic secret in session σ."""
    if direction not in DIRECTIONS:
        raise ValueError(f"a direction is one of {DIRECTIONS}, not {direction!r}")
    return expand(_key(sigma, "a session secret"), "traffic " + direction)


def keys(secret: bytes) -> aead.Keys:
    """The AEAD keys of a direction at traffic secret `secret`."""
    return aead.Keys(expand(secret, "enc"), expand(secret, "mac"))


def nonce(epoch: int, seq: int) -> bytes:
    """Record `seq` of epoch `epoch`'s nonce: u64be ‖ u64be, never sent."""
    return NONCE.pack(epoch, seq)


class _Direction:
    def __init__(self, secret: bytes) -> None:
        self._secret, self._keys = secret, keys(_key(secret, "a traffic secret"))
        self.epoch = self.seq = 0
        self.closed = False

    def _update(self) -> None:
        self._secret = expand(self._secret, "update")
        self._keys = keys(self._secret)
        self.epoch, self.seq = self.epoch + 1, 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(epoch={self.epoch}, seq={self.seq}, closed={self.closed})"


class Sender(_Direction):
    """One direction's sending half (no I/O): each call returns the next unit, one channel frame."""

    def _seal(self, kind: int, body: bytes, size: int) -> bytes:
        if self.closed:
            raise Refused("after-close", "this direction is closed")
        if HEADER.size + len(body) > size:
            raise Refused("oversized", f"a {len(body)}-byte body doesn't fit a {size}-byte record")
        if kind == DATA and self.seq >= EPOCH_RECORDS - 1:
            raise Refused("epoch-full", f"epoch {self.epoch} holds {EPOCH_RECORDS} records; update first")
        plain = HEADER.pack(kind, len(body)) + body + bytes(size - HEADER.size - len(body))
        unit = aead.seal(self._keys, nonce(self.epoch, self.seq), plain)
        self.seq += 1
        return unit

    def data(self, body: bytes, size: int) -> bytes:
        """`body` as a record of the declared plaintext size `size`, so a unit of size + `aead.TAG_BYTES` bytes."""
        return self._seal(DATA, body, size)

    def update(self) -> bytes:
        """An update, after which this direction seals under the next epoch's keys."""
        unit = self._seal(UPDATE, b"", HEADER.size)
        self._update()
        return unit

    def close(self) -> bytes:
        """The close, after which this direction sends nothing."""
        unit = self._seal(CLOSE, b"", HEADER.size)
        self.closed = True
        return unit


class Receiver(_Direction):
    """One direction's receiving half (no I/O): `open` takes the units in the order they were read off the channel."""

    def __init__(self, secret: bytes) -> None:
        super().__init__(secret)
        self.refused = False

    def _refuse(self, name: str, detail: str) -> Refused:
        self.refused = True
        return Refused(name, detail)

    def open(self, unit: bytes) -> bytes | None:
        """The next unit's body if it is data, None if it is an update or the close. Refuses (`record-refused`) a unit
        that doesn't open at its position and every unit after one refused, and (`after-close`) a unit after the
        close."""
        if self.refused:
            raise Refused("record-refused", "a unit before this one was refused")
        if self.closed:
            raise self._refuse("after-close", "a unit after the close")
        plain = aead.unseal(self._keys, nonce(self.epoch, self.seq), unit)
        if plain is None or len(plain) < HEADER.size:
            raise self._refuse("record-refused", f"unit {self.seq} of epoch {self.epoch} doesn't open")
        kind, n = HEADER.unpack_from(plain)
        end = HEADER.size + n
        if kind not in (DATA, UPDATE, CLOSE) or end > len(plain) or any(plain[end:]) or (kind != DATA and n):
            raise self._refuse("record-refused", f"unit {self.seq} of epoch {self.epoch} isn't a record")
        self.seq += 1
        if kind == UPDATE:
            self._update()
            return None
        if kind == CLOSE:
            self.closed = True
            return None
        return plain[HEADER.size:end]

    def finish(self) -> None:
        """Refuses (`truncated`) a stream that ended without its close."""
        if not self.closed:
            raise self._refuse("truncated", f"the stream ended before its close, at unit {self.seq} of epoch "
                                            f"{self.epoch}")


def ends(sigma: bytes, initiator: bool) -> tuple[Sender, Receiver]:
    """This side's sender and receiver in session σ: the initiator sends on ``i>r`` and opens ``r>i``, the responder the
    other way round."""
    out, into = DIRECTIONS if initiator else DIRECTIONS[::-1]
    return Sender(traffic(sigma, out)), Receiver(traffic(sigma, into))
