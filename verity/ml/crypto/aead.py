"""An AEAD built from SHA-512 alone, with its whole 64-byte output: counter mode over a keyed SHA-512, then HMAC-SHA512
over the nonce and the ciphertext (encrypt-then-MAC). It seals the frames between two of the auditor's network
certifiers (`verity.core.protocols.certifier.active.Pair`).

H is SHA-512, HMAC is HMAC-SHA512 (RFC 2104) and [x] is u16be len(x) ‖ x.

- **Keys** (`keys`): from a 64-byte secret S, a public session id σ and the name ℓ of one direction of a link, by
  HKDF-Expand (RFC 5869) with one output block, S already uniform (so no extract step):
  ``K_enc = HMAC(S, ENC_LABEL ‖ [σ] ‖ [ℓ] ‖ 0x01)`` and ``K_mac = HMAC(S, MAC_LABEL ‖ [σ] ‖ [ℓ] ‖ 0x01)``.
- **Nonce** (`nonce`): ``u64be(window) ‖ u32be(bucket) ‖ u32be(slot)``, 16 bytes, public structure.
- **Seal** (`seal`) of a frame f at nonce N: ``KS = H(K_enc ‖ N ‖ u64be 0) ‖ H(K_enc ‖ N ‖ u64be 1) ‖ …`` cut to
  len(f) bytes (`keystream`), ``C = f ⊕ KS``, ``tag = HMAC(K_mac, N ‖ C)``; the sealed unit is ``C ‖ tag``,
  len(f) + `TAG_BYTES` bytes.
- **Open** (`unseal`): the tag recomputed over N ‖ C and compared in constant time; the frame, or None.

Each keystream input is 88 bytes, inside SHA-512's first block, so a keystream block is one compression with the key in
its message input: a PRF under ``prf/sha-512`` (`verity.core.claims`), the claim `verity.ml.randomness`'s keyed streams
rest on, and every input has one length, so length extension doesn't arise. Counter mode over a PRF is IND-CPA while no
nonce repeats under a key (Bellare, Desai, Jokipii, Rogaway 1997); HMAC-SHA512 is a PRF under the same assumption
(Bellare 2006), so a strongly unforgeable MAC; encrypt-then-MAC with the nonce under the tag is IND-CCA and INT-CTXT
(Bellare, Namprempre 2000; scheme N2 of Namprempre, Rogaway, Shrimpton 2014). The caller never seals two frames at one
nonce under one key, and gives every frame of one key the same length, so N ‖ C needs no length field.

Standard library only.
"""
from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass

__all__ = ["ENC_LABEL", "MAC_LABEL", "NONCE_BYTES", "SECRET_BYTES", "TAG_BYTES", "Keys", "keys", "keystream", "nonce",
           "seal", "unseal"]

SECRET_BYTES = 64
TAG_BYTES = 64
NONCE = struct.Struct(">QII")
NONCE_BYTES = NONCE.size
ENC_LABEL = b"verity-network-certifier/pair/v1/enc"
MAC_LABEL = b"verity-network-certifier/pair/v1/mac"
_BLOCK = hashlib.sha512().digest_size


def _field(x: bytes) -> bytes:
    if len(x) >= 1 << 16:
        raise ValueError("a key's context field is under 64 KiB")
    return struct.pack(">H", len(x)) + x


@dataclass(frozen=True, slots=True)
class Keys:
    """One direction's keys: `enc` for the keystream, `mac` for the tag, 64 bytes each."""

    enc: bytes
    mac: bytes

    def __repr__(self) -> str:
        return "Keys(…)"


def keys(secret: bytes, session: bytes, link: str) -> Keys:
    """The keys of direction `link` in session `session` under the 64-byte `secret` (HKDF-Expand, one block each)."""
    if not (isinstance(secret, bytes) and len(secret) == SECRET_BYTES):
        raise ValueError(f"a pair's secret is {SECRET_BYTES} bytes")
    if not isinstance(session, bytes):
        raise ValueError("a session id is bytes")
    info = _field(session) + _field(link.encode())
    return Keys(hmac.digest(secret, ENC_LABEL + info + b"\x01", "sha512"),
                hmac.digest(secret, MAC_LABEL + info + b"\x01", "sha512"))


def nonce(window: int, bucket: int, slot: int) -> bytes:
    """The nonce of slot `slot` of bucket `bucket` of window `window`: u64be ‖ u32be ‖ u32be."""
    return NONCE.pack(window, bucket, slot)


def keystream(enc: bytes, n_: bytes, length: int) -> bytes:
    """The first `length` bytes of H(enc ‖ n_ ‖ u64be 0) ‖ H(enc ‖ n_ ‖ u64be 1) ‖ …, each block from one copy of the
    hash state after enc ‖ n_ (a quarter faster than hashing the whole input each time, the same bytes)."""
    head, out = hashlib.sha512(enc + n_), []
    for c in range(-(-length // _BLOCK)):
        h = head.copy()
        h.update(c.to_bytes(8, "big"))
        out.append(h.digest())
    return b"".join(out)[:length]


def _xor(a: bytes, b: bytes) -> bytes:
    return (int.from_bytes(a) ^ int.from_bytes(b)).to_bytes(len(a))


def seal(k: Keys, n_: bytes, frame: bytes) -> bytes:
    """`frame` sealed at nonce `n_`: C ‖ HMAC(K_mac, n_ ‖ C), C = frame ⊕ keystream."""
    if len(n_) != NONCE_BYTES:
        raise ValueError(f"a nonce is {NONCE_BYTES} bytes")
    c = _xor(frame, keystream(k.enc, n_, len(frame)))
    return c + hmac.digest(k.mac, n_ + c, "sha512")


def unseal(k: Keys, n_: bytes, unit: bytes) -> bytes | None:
    """The frame `unit` seals at nonce `n_`, or None when its tag doesn't verify (or it is shorter than a tag)."""
    if len(n_) != NONCE_BYTES:
        raise ValueError(f"a nonce is {NONCE_BYTES} bytes")
    if len(unit) < TAG_BYTES:
        return None
    c, tag = unit[:-TAG_BYTES], unit[-TAG_BYTES:]
    if not hmac.compare_digest(tag, hmac.digest(k.mac, n_ + c, "sha512")):
        return None
    return _xor(c, keystream(k.enc, n_, len(c)))
