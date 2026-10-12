"""The KV cache's codec, byte-exact: Sloth over one wide word, the word as its payload (`kv.cu` is the kernel).

~~~text
field      p = 2^n − c, n a multiple of 32, p ≡ 3 (mod 4), c the smallest such: 2^512 − 569, 2^768 − 825,
           2^1024 − 105, 2^1280 − 1665 and 2^1536 − 3453
word       S = n/8 bytes, little-endian, canonical (< p); the payload is the whole word (64 to 192 bytes)
ρ(a)       Sloth's permutation: s = a^((p+1)/4); if s² = a the even one of {s, p − s}, else (s² = −a) the odd one
ρ⁻¹(y)     y² if y is even, p − y² if y is odd: one squaring and the sign rule
XOF(m)     SHA512(m ‖ u32be 0) ‖ SHA512(m ‖ u32be 1) ‖ …
c_j        LE(XOF(DOMAIN ‖ frame ‖ seed ‖ u32be j)[0 : S + 8]) mod p, j = 0 … k − 1, frame = u8 len(name) ‖ name
τ          the same at j = 2^32 − 1, and 1 in place of 0
t(i)       τ · i mod p, i the word's index in the store
round j    x ↦ ρ(x + c_j + t(i)); encode is rounds 0 … k − 1 from the payload, decode rounds k − 1 … 0 inverted,
           x ↦ ρ⁻¹(x) − c_j − t(i)
~~~

The derivation is `benchmarks/codec_tensor/wide.py`'s (ra-codec-tensor) under a domain of its own, at the fields
ra-codec-theory chose (2^1536 − 3453 its operating point), whose words have no slack. A payload is any word below p. For p = 2^n − c
with c < 2^16 a payload at or above p has its top 16 bits all ones, a NaN in bf16 (and its top byte 0xFF, a NaN in
E4M3), so every word of finite values is one; `encode_word` refuses the rest.

Encoding a word costs k roots, each (p+1)/4 = (2^M − 1)·2^b + LOW: x_M = a^(2^M − 1) by M's bits from the top
(x_2j = x_j^(2^j)·x_j, x_(j+1) = x_j²·a), then b squarings with a multiply by a at LOW's set bits, b = 8 or the
fewest bits that hold (c − 1)/4, so n − 3 squarings and 16 to 20 multiplies (`root_multiplies`), then the square
test (one more). Decoding costs k squarings.

The KV cache's layout (`token_layer_words`): a token's keys and values in one layer, K then V, each `KV_HEADS` ×
`HEAD_DIM` bf16 (Llama-3.1-70B: 8 × 128, 4,096 bytes together), are the payloads of W = ⌈4096 / S⌉ consecutive words,
zero-padded (64 words at 2^512 − 569, 43 at 2^768 − 825, 32, 26 and 22 at 1,024, 1,280 and 1,536 bits), the words of
slot s's layer l at index (s · LAYERS + l) · W.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

DOMAIN = b"verity/word-codec/kv/v0\0"
SEED_BYTES = 32
TWEAK_INDEX = (1 << 32) - 1
LOW_MIN = 8

# Llama-3.1-70B's attention: 80 layers, 8 KV heads of 128, bf16.
LAYERS = 80
KV_HEADS = 8
HEAD_DIM = 128
KV_ELEMENT_BYTES = 2
TOKEN_LAYER_BYTES = 2 * KV_HEADS * HEAD_DIM * KV_ELEMENT_BYTES
TOKEN_BYTES = LAYERS * TOKEN_LAYER_BYTES


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    n: int
    c: int

    @property
    def p(self) -> int:
        return (1 << self.n) - self.c

    @property
    def limbs(self) -> int:
        return self.n // 32

    @property
    def word_bytes(self) -> int:
        return self.n // 8

    @property
    def root_exp(self) -> int:
        return (self.p + 1) // 4

    @property
    def low_bits(self) -> int:
        return max(LOW_MIN, ((self.c - 1) // 4).bit_length())

    @property
    def chain(self) -> tuple[int, int]:
        """(M, LOW) with (p+1)/4 = (2^M − 1)·2^b + LOW, b = `low_bits`."""
        b = self.low_bits
        return self.n - 2 - b, (1 << b) - (self.c - 1) // 4

    @property
    def root_squarings(self) -> int:
        """The chain's squarings: M − 1 for x_M, then b."""
        return self.chain[0] - 1 + self.low_bits

    @property
    def root_multiplies(self) -> int:
        m, low = self.chain
        return m.bit_length() - 1 + m.bit_count() - 1 + low.bit_count()


FIELDS = {f.name: f for f in (Field("p512", 512, 569), Field("p768", 768, 825), Field("p1024", 1024, 105),
                               Field("p1280", 1280, 1665), Field("p1536", 1536, 3453))}


def field(name: str) -> Field:
    try:
        return FIELDS[name]
    except KeyError:
        raise ValueError(f"unknown field {name!r}; known: {sorted(FIELDS)}") from None


def check_fields() -> None:
    for f in FIELDS.values():
        m, low = f.chain
        assert f.n % 32 == 0 and f.p % 4 == 3 and f.p.bit_length() == f.n and f.c < 1 << 16
        assert 0 < low < 1 << f.low_bits and ((1 << m) - 1) * (1 << f.low_bits) + low == f.root_exp


@dataclass(frozen=True, slots=True)
class Key:
    field: Field
    c: tuple[int, ...]
    tau: int

    @property
    def rounds(self) -> int:
        return len(self.c)

    def tweak(self, i: int) -> int:
        return self.tau * i % self.field.p


def xof(message: bytes, length: int) -> bytes:
    out = b""
    ctr = 0
    while len(out) < length:
        out += hashlib.sha512(message + struct.pack(">I", ctr)).digest()
        ctr += 1
    return out[:length]


def _element(f: Field, seed: bytes, j: int) -> int:
    name = f.name.encode()
    msg = DOMAIN + bytes([len(name)]) + name + seed + struct.pack(">I", j)
    return int.from_bytes(xof(msg, f.word_bytes + 8), "little") % f.p


def key(name: str, seed: bytes, rounds: int) -> Key:
    f = field(name)
    if type(seed) is not bytes or len(seed) != SEED_BYTES:
        raise ValueError(f"a seed is {SEED_BYTES} bytes")
    if not 1 <= rounds < TWEAK_INDEX:
        raise ValueError("a word has at least one round")
    return Key(f, tuple(_element(f, seed, j) for j in range(rounds)), _element(f, seed, TWEAK_INDEX) or 1)


def rho(a: int, f: Field) -> int:
    p = f.p
    if not 0 <= a < p:
        raise ValueError("ρ takes a canonical element")
    s = pow(a, f.root_exp, p)
    square = s * s % p == a
    if (s & 1) == (0 if square else 1):
        return s
    return p - s if s else 0


def rho_inv(y: int, f: Field) -> int:
    p = f.p
    if not 0 <= y < p:
        raise ValueError("ρ⁻¹ takes a canonical element")
    q = y * y % p
    return q if y & 1 == 0 else (p - q) % p


def encode_word(m: int, i: int, k: Key) -> int:
    f = k.field
    if not 0 <= m < f.p:
        raise ValueError("a payload is a word below p")
    t = k.tweak(i)
    x = m
    for c in k.c:
        x = rho((x + c + t) % f.p, f)
    return x


def decode_word(y: int, i: int, k: Key) -> int:
    f = k.field
    t = k.tweak(i)
    x = y
    for c in reversed(k.c):
        x = (rho_inv(x, f) - c - t) % f.p
    return x


def token_layer_words(f: Field) -> int:
    return -(-TOKEN_LAYER_BYTES // f.word_bytes)


def token_words(f: Field) -> int:
    return LAYERS * token_layer_words(f)


def payloads(data: bytes, f: Field) -> list[int]:
    """One token-layer's bytes (K then V) as its words' payloads, zero-padded to whole words."""
    if len(data) != TOKEN_LAYER_BYTES:
        raise ValueError(f"a token-layer is {TOKEN_LAYER_BYTES} bytes")
    s = f.word_bytes
    padded = data + bytes(token_layer_words(f) * s - len(data))
    return [int.from_bytes(padded[s * w : s * (w + 1)], "little") for w in range(token_layer_words(f))]


def encode_token_layer(data: bytes, slot: int, layer: int, k: Key) -> list[int]:
    f = k.field
    w0 = (slot * LAYERS + layer) * token_layer_words(f)
    return [encode_word(m, w0 + w, k) for w, m in enumerate(payloads(data, f))]


def decode_token_layer(words: list[int], slot: int, layer: int, k: Key) -> bytes:
    f = k.field
    w0 = (slot * LAYERS + layer) * token_layer_words(f)
    s = f.word_bytes
    out = b"".join(decode_word(y, w0 + w, k).to_bytes(s, "little") for w, y in enumerate(words))
    return out[:TOKEN_LAYER_BYTES]


def to_limbs(x: int, f: Field) -> list[int]:
    return [(x >> (32 * j)) & 0xFFFFFFFF for j in range(f.limbs)]


def from_limbs(limbs: list[int]) -> int:
    return sum(v << (32 * j) for j, v in enumerate(limbs))
