"""The store's codec: Sloth over p = 2^512 − 569, k ≥ 2 rounds, a 4 KiB block as 64 words.  Byte-exact with
`verity/word-codec/kv/v0` at `p512` (`benchmarks/word_codec/kv.py`, the kernel `kv.cu`), with block `j`'s words at
store indices `64 j … 64 j + 63`, which is kv/v0's token-layer layout for `j = slot · LAYERS + layer`.

~~~text
p          2^512 − 569 (p ≡ 3 mod 4)
word       64 bytes, little-endian, canonical (< p); a block is 64 words, word w of block j at index i = 64 j + w
ρ(a)       s = a^((p+1)/4); if s² = a the even one of {s, p − s}, else (s² = −a) the odd one
ρ⁻¹(y)     y² if y is even, p − y² if y is odd
XOF(m)     SHA512(m ‖ u32be 0) ‖ SHA512(m ‖ u32be 1) ‖ …
c_r        LE(XOF(DOMAIN ‖ u8 4 ‖ "p512" ‖ seed ‖ u32be r)[0:72]) mod p, r = 0 … k − 1, DOMAIN = "verity/word-codec/kv/v0\0"
τ          the same at r = 2^32 − 1, and 1 in place of 0;  t(i) = τ · i mod p
encode     x ↦ ρ(x + c_r + t(i)) for r = 0 … k − 1, from the payload word
decode     x ↦ ρ⁻¹(x) − c_r − t(i) for r = k − 1 … 0
~~~

Encoding costs k square roots a word (about 510 squarings each), decoding k squarings.  `rounds` below 2 is refused:
at k = 1 a word can be partly kept and completed for less than an encode (red team F27).  ρ is a permutation of the
canonical words, so `decode` is a bijection from canonical blocks to payload blocks, and a payload commitment with the
codec seed fixes the encoded store.  A payload word at or above p is refused: under bf16 or E4M3 its top 16 bits are
all ones, a NaN, so every block of finite values encodes.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

DOMAIN = b"verity/word-codec/kv/v0\0"
FIELD = b"p512"
P = (1 << 512) - 569
ROOT = (P + 1) // 4
WORD_BYTES = 64
BLOCK_WORDS = 64
BLOCK_BYTES = WORD_BYTES * BLOCK_WORDS
SEED_BYTES = 32
TWEAK_INDEX = (1 << 32) - 1
MIN_ROUNDS = 2


def _xof(message: bytes, length: int) -> bytes:
    out = b""
    ctr = 0
    while len(out) < length:
        out += hashlib.sha512(message + struct.pack(">I", ctr)).digest()
        ctr += 1
    return out[:length]


def _element(seed: bytes, r: int) -> int:
    msg = DOMAIN + bytes([len(FIELD)]) + FIELD + seed + struct.pack(">I", r)
    return int.from_bytes(_xof(msg, WORD_BYTES + 8), "little") % P


@dataclass(frozen=True, slots=True)
class Key:
    """The round constants `c_0 … c_(k−1)` and the tweak multiplier `τ` of one codec seed."""

    c: tuple[int, ...]
    tau: int

    @property
    def rounds(self) -> int:
        return len(self.c)


def key(seed: bytes, rounds: int = MIN_ROUNDS) -> Key:
    if type(seed) is not bytes or len(seed) != SEED_BYTES:
        raise ValueError(f"a codec seed is {SEED_BYTES} bytes")
    if type(rounds) is not int or not MIN_ROUNDS <= rounds < TWEAK_INDEX:
        raise ValueError(f"the codec runs at least {MIN_ROUNDS} rounds: at one a word is completed for less than an encode")
    return Key(tuple(_element(seed, r) for r in range(rounds)), _element(seed, TWEAK_INDEX) or 1)


def rho(a: int) -> int:
    s = pow(a, ROOT, P)
    square = s * s % P == a
    if (s & 1) == (0 if square else 1):
        return s
    return P - s if s else 0


def rho_inv(y: int) -> int:
    q = y * y % P
    return q if y & 1 == 0 else (P - q) % P


def encode_word(m: int, i: int, k: Key) -> int:
    if not 0 <= m < P:
        raise ValueError("a payload word is below p")
    t = k.tau * i % P
    x = m
    for c in k.c:
        x = rho((x + c + t) % P)
    return x


def decode_word(y: int, i: int, k: Key) -> int:
    if not 0 <= y < P:
        raise ValueError("an encoded word is canonical, below p")
    t = k.tau * i % P
    x = y
    for c in reversed(k.c):
        x = (rho_inv(x) - c - t) % P
    return x


def _words(block: bytes) -> list[int]:
    if type(block) is not bytes or len(block) != BLOCK_BYTES:
        raise ValueError(f"a block is {BLOCK_BYTES} bytes")
    return [int.from_bytes(block[WORD_BYTES * w:WORD_BYTES * (w + 1)], "little") for w in range(BLOCK_WORDS)]


def _block(words: list[int]) -> bytes:
    return b"".join(x.to_bytes(WORD_BYTES, "little") for x in words)


def encode(payload: bytes, j: int, k: Key) -> bytes:
    """`E_j`, block `j` of the store, from its payload."""
    i0 = BLOCK_WORDS * _index(j)
    return _block([encode_word(m, i0 + w, k) for w, m in enumerate(_words(payload))])


def decode(block: bytes, j: int, k: Key) -> bytes | None:
    """Block `j`'s payload, or None if a word of it is not canonical (no payload encodes to it)."""
    i0 = BLOCK_WORDS * _index(j)
    words = _words(block)
    if any(y >= P for y in words):
        return None
    return _block([decode_word(y, i0 + w, k) for w, y in enumerate(words)])


def _index(j: int) -> int:
    if type(j) is not int or not 0 <= j < 1 << 58:
        raise ValueError("a block index is below 2^58, so its words' indices fit a u64")
    return j
