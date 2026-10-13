"""The store's codec: Sloth over one wide word, byte-exact with `verity/word-codec/kv/v0` (ra-codec-mix's
`benchmarks/word_codec/kv.py`, its kernel `kv.cu`), a 4 KiB block as W words.  A codec is its field, its k ≥ 2 rounds
and whether σ_K masks between them, and a registration names it (`Codec.name`).  The default is Daniel's ruling (#1884,
12 Oct): p = 2^1024 − 105, masked, k = 3, with k = 2 at the same width the fallback.  p = 2^512 − 569, unmasked, is kept
as the upgrade path.

~~~text
field      p = 2^n − c, p ≡ 3 (mod 4): p1024 = 2^1024 − 105, p512 = 2^512 − 569
word       S = n/8 bytes, little-endian, canonical (< p); a block is W = 4096/S words (32 of 128 bytes at p1024, 64 of 64
           at p512), word w of block j at store index i = W j + w
ρ(a)       s = a^((p+1)/4); if s² = a the even one of {s, p − s}, else (s² = −a) the odd one
ρ⁻¹(y)     y² if y is even, p − y² if y is odd
XOF(m)     SHA512(m ‖ u32be 0) ‖ SHA512(m ‖ u32be 1) ‖ …
c_r        LE(XOF(DOMAIN ‖ u8 len(name) ‖ name ‖ seed ‖ u32be r)[0 : S + 8]) mod p, r = 0 … k − 1, name the field's
           ("p1024", "p512"), DOMAIN = "verity/word-codec/kv/v0\0"
τ          the same at r = 2^32 − 1, and 1 in place of 0;  t(i) = τ · i mod p
K          a masked codec's: LE(XOF(DOMAIN ‖ u8 len(name) ‖ name ‖ seed ‖ u32be 2^32 − 2)[0 : S]), n bits, not reduced
σ_K(x)     x ⊕ K if that is below p, else x: an involution on [0, p)
encode     x ↦ ρ(x + c_r + t(i)) for r = 0 … k − 1, from the payload word; masked, x ↦ σ_K(x) before each round r ≥ 1
decode     x ↦ ρ⁻¹(x) − c_r − t(i) for r = k − 1 … 0; masked, x ↦ σ_K(x) after each round r ≥ 1
~~~

Encoding costs k square roots a word (n − 3 squarings each), decoding k squarings and, masked, k − 1 masks: at the
default a block is 96 roots of 1,021 squarings to encode and 96 squarings of 1,024 bits to decode.  `rounds` below 2
is refused: at k = 1 a word can be partly kept and completed for less than an encode (red team F27).  ρ and σ_K permute
the canonical words, so `decode` is a bijection from canonical blocks to payload blocks, and a payload commitment with
the codec seed fixes the encoded store.  A payload word at or above p is refused: c is below 2^16, so its top 16 bits
are all ones, a NaN under bf16 (and its top byte 0xFF, one under E4M3), and every block of finite values encodes.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

DOMAIN = b"verity/word-codec/kv/v0\0"
BLOCK_BYTES = 4096
SEED_BYTES = 32
TWEAK_INDEX = (1 << 32) - 1
MASK_INDEX = (1 << 32) - 2
MIN_ROUNDS = 2
WORD_INDICES = 1 << 64


@dataclass(frozen=True, slots=True)
class Field:
    """p = 2^n − c, named as kv/v0 names it."""

    name: str
    n: int
    c: int

    @property
    def p(self) -> int:
        return (1 << self.n) - self.c

    @property
    def word_bytes(self) -> int:
        return self.n // 8

    @property
    def block_words(self) -> int:
        return BLOCK_BYTES // self.word_bytes


P1024 = Field("p1024", 1024, 105)
P512 = Field("p512", 512, 569)
FIELDS = {f.name: f for f in (P1024, P512)}


@dataclass(frozen=True, slots=True)
class Codec:
    """A store's codec: Sloth over `field` at k = `rounds`, with σ_K between the rounds if `masked`."""

    field: Field
    rounds: int
    masked: bool

    def __post_init__(self) -> None:
        if type(self.field) is not Field or FIELDS.get(self.field.name) != self.field:
            raise ValueError(f"a codec's field is one of {sorted(FIELDS)}")
        if type(self.rounds) is not int or not MIN_ROUNDS <= self.rounds < MASK_INDEX:
            raise ValueError(f"the codec runs at least {MIN_ROUNDS} rounds: at one a word is completed for less than an encode")
        if type(self.masked) is not bool:
            raise ValueError("a codec is masked or not")

    @property
    def name(self) -> str:
        """What a registration binds: `p1024/k3/masked`, `p512/k2`."""
        return f"{self.field.name}/k{self.rounds}" + ("/masked" if self.masked else "")

    @property
    def max_blocks(self) -> int:
        """The most blocks a store under it holds: its words' indices fit a u64 (2^59 at p1024, 2^58 at p512)."""
        return WORD_INDICES // self.field.block_words

    def to_json(self) -> dict:
        """Its parameters: the field `p = 2^n − c`, k, and the mask."""
        return {"c": self.field.c, "field": self.field.name, "masked": self.masked, "n": self.field.n,
                "rounds": self.rounds}


DEFAULT = Codec(P1024, 3, True)
FALLBACK = Codec(P1024, 2, True)


def _xof(message: bytes, length: int) -> bytes:
    out = b""
    ctr = 0
    while len(out) < length:
        out += hashlib.sha512(message + struct.pack(">I", ctr)).digest()
        ctr += 1
    return out[:length]


def _message(f: Field, seed: bytes, r: int) -> bytes:
    name = f.name.encode()
    return DOMAIN + bytes([len(name)]) + name + seed + struct.pack(">I", r)


def _element(f: Field, seed: bytes, r: int) -> int:
    return int.from_bytes(_xof(_message(f, seed, r), f.word_bytes + 8), "little") % f.p


@dataclass(frozen=True, slots=True)
class Key:
    """A codec seed's round constants `c_0 … c_(k−1)`, its tweak multiplier `τ` and, masked, its mask `K`."""

    codec: Codec
    c: tuple[int, ...]
    tau: int
    mask: int | None = None

    @property
    def rounds(self) -> int:
        return len(self.c)

    def sigma(self, x: int) -> int:
        v = x ^ self.mask
        return v if v < self.codec.field.p else x


def key(seed: bytes, codec: Codec = DEFAULT) -> Key:
    if type(seed) is not bytes or len(seed) != SEED_BYTES:
        raise ValueError(f"a codec seed is {SEED_BYTES} bytes")
    if type(codec) is not Codec:
        raise ValueError("a key is of a Codec")
    f = codec.field
    mask = int.from_bytes(_xof(_message(f, seed, MASK_INDEX), f.word_bytes), "little") if codec.masked else None
    return Key(codec, tuple(_element(f, seed, r) for r in range(codec.rounds)), _element(f, seed, TWEAK_INDEX) or 1, mask)


def rho(a: int, f: Field) -> int:
    p = f.p
    s = pow(a, (p + 1) // 4, p)
    square = s * s % p == a
    if (s & 1) == (0 if square else 1):
        return s
    return p - s if s else 0


def rho_inv(y: int, f: Field) -> int:
    p = f.p
    q = y * y % p
    return q if y & 1 == 0 else (p - q) % p


def encode_word(m: int, i: int, k: Key) -> int:
    f = k.codec.field
    p = f.p
    if not 0 <= m < p:
        raise ValueError("a payload word is below p")
    t = k.tau * i % p
    x = m
    for r, c in enumerate(k.c):
        if r and k.mask is not None:
            x = k.sigma(x)
        x = rho((x + c + t) % p, f)
    return x


def decode_word(y: int, i: int, k: Key) -> int:
    f = k.codec.field
    p = f.p
    if not 0 <= y < p:
        raise ValueError("an encoded word is canonical, below p")
    t = k.tau * i % p
    x = y
    for r in reversed(range(k.rounds)):
        x = (rho_inv(x, f) - k.c[r] - t) % p
        if r and k.mask is not None:
            x = k.sigma(x)
    return x


def _words(block: bytes, f: Field) -> list[int]:
    if type(block) is not bytes or len(block) != BLOCK_BYTES:
        raise ValueError(f"a block is {BLOCK_BYTES} bytes")
    s = f.word_bytes
    return [int.from_bytes(block[s * w:s * (w + 1)], "little") for w in range(f.block_words)]


def _block(words: list[int], f: Field) -> bytes:
    return b"".join(x.to_bytes(f.word_bytes, "little") for x in words)


def encode(payload: bytes, j: int, k: Key) -> bytes:
    """`E_j`, block `j` of the store, from its payload."""
    f = k.codec.field
    i0 = f.block_words * _index(j, k.codec)
    return _block([encode_word(m, i0 + w, k) for w, m in enumerate(_words(payload, f))], f)


def decode(block: bytes, j: int, k: Key) -> bytes | None:
    """Block `j`'s payload, or None if a word of it is not canonical (no payload encodes to it)."""
    f = k.codec.field
    i0 = f.block_words * _index(j, k.codec)
    words = _words(block, f)
    if any(y >= f.p for y in words):
        return None
    return _block([decode_word(y, i0 + w, k) for w, y in enumerate(words)], f)


def _index(j: int, codec: Codec) -> int:
    if type(j) is not int or not 0 <= j < codec.max_blocks:
        raise ValueError(f"a block index is below {codec.max_blocks}, so its words' indices fit a u64")
    return j
