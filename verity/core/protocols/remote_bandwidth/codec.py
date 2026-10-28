"""The store's codec: Sloth over one wide word, byte-exact with `verity/word-codec/kv/v0` (ra-codec-mix's
`benchmarks/word_codec/kv.py`, its kernel `kv.cu`), a 4 KiB chunk as W words.  A codec is its field, its k ≥ 2 rounds
and whether σ_K masks between them, and a registration names it (`Codec.name`).  The default is p = 2^1024 − 105,
masked, k = 3, with k = 2 at the same width the fallback.  p = 2^512 − 569, unmasked, is kept as the upgrade path.

~~~text
field      p = 2^n − c, p ≡ 3 (mod 4): p1024 = 2^1024 − 105, p512 = 2^512 − 569
word       S = n/8 bytes, little-endian, canonical (< p); a chunk is W = 4096/S words (32 of 128 bytes at p1024, 64 of 64
           at p512), word w of chunk j at store index i = W j + w
ρ(a)       s = a^((p+1)/4); if s² = a the even one of {s, p − s}, else (s² = −a) the odd one
ρ⁻¹(y)     y² if y is even, p − y² if y is odd
XOF(m)     SHA512(m ‖ u32be 0) ‖ SHA512(m ‖ u32be 1) ‖ …
c_r        LE(XOF(DOMAIN ‖ u8 len(name) ‖ name ‖ seed ‖ u32be r)[0 : S + 8]) mod p, r = 0 … k − 1, name the field's
           ("p1024", "p512"), DOMAIN = "verity/word-codec/kv/v0\0"
τ          the same at r = 2^32 − 1, and 1 in place of 0;  t(i) = τ · i mod p
K          a masked codec's: LE(XOF(DOMAIN ‖ u8 len(name) ‖ name ‖ seed ‖ u32be 2^32 − 2)[0 : S]), n bits, not reduced
σ_K(x)     x ⊕ K if that is below p, else x: an involution on [0, p)
d_j        a bound chunk's digest: sha512/row/v2 of its payload, SHA512(P(32768) ‖ payload_j), its payload leaf's x_j
b_j        a bound chunk's offset: LE(XOF(DOMAIN ‖ u8 len(name) ‖ name ‖ seed ‖ u32be 2^32 − 3 ‖ u64be j ‖ d_j)[0 : 64])
           mod p; under the seed's key b_j = 0
c_a        an append's coin: 32 bytes the verifier draws once the append's payload digests are committed
b_j(c_a)   a bound chunk's offset under its append's coin: LE(XOF(DOMAIN ‖ u8 len(name) ‖ name ‖ seed ‖ u32be 2^32 − 4 ‖
           u64be j ‖ d_j ‖ c_a)[0 : 64]) mod p
encode     x ↦ ρ(x + c_r + t(i) + b_j) for r = 0 … k − 1, from the payload word; masked, x ↦ σ_K(x) before each round
           r ≥ 1
decode     x ↦ ρ⁻¹(x) − c_r − t(i) − b_j for r = k − 1 … 0; masked, x ↦ σ_K(x) after each round r ≥ 1
~~~

A chunk's key is the seed's (b_j = 0) or its own, bound (`encode(..., bound=True)`, `decode(..., d)`).  The seed's key
is for chunks whose payloads were committed before the codec seed was drawn (a registration's): such a chunk's
encoding costs k roots a word whatever the payload.  A chunk written after the seed (a KV page) takes its bound key,
with d_j kept beside it, since under the seed's key its writer could pick its encoding E (all zeros, or another
chunk's) and write the payload `decode(E)`, holding nothing for it; under the bound key a chosen E decodes to a
payload of that digest only by a search over SHA-512.  The coin form (`encode(..., bound=True, coin=c_a)`,
`decode(..., d, c_a)`) also binds the coin its append drew after committing d_j, so no choice made before that commit
can depend on the chunk's encoding; its derivation index, 2^32 − 4, keeps it apart from the form without a coin,
which `kv.py` and its pinned vectors keep.  Either offset enters the words the same way, so a device that takes b_j as
an input encodes both.

Encoding costs k square roots a word (n − 3 squarings each), decoding k squarings and, masked, k − 1 masks: at the
default a chunk is 96 roots of 1,021 squarings to encode and 96 squarings of 1,024 bits to decode.  A bound chunk
adds d_j (34 SHA-512 compressions over the payload, which its payload leaf hashes anyway) and b_j (2) to its encode,
and b_j to its decode unless b_j is kept in d_j's place.  `rounds` below 2 is refused: at k = 1 a word can be partly
kept and completed for less than an encode.  ρ and σ_K permute the canonical words, so `decode` under
the seed's key is a bijection from canonical chunks to payload chunks, and a payload commitment with the codec seed
fixes the encoded store; under a bound key, a chunk and a digest decode to the payload only if it has that digest.  A
payload word at or above p is refused: c is below 2^16, so its top 16 bits are all ones, a NaN under bf16 (and its top
byte 0xFF, one under E4M3), and every chunk of finite values encodes.

A registration over PoUS's P2 store names `P2Codec` instead: PoUS's codec, which this protocol only names.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from verity.ml.commitments.rowleaf import sha512_row_v2_digest

DOMAIN = b"verity/word-codec/kv/v0\0"
BLOCK_BYTES = 4096
SEED_BYTES = 32
DIGEST_BYTES = 64
TWEAK_INDEX = (1 << 32) - 1
MASK_INDEX = (1 << 32) - 2
BIND_INDEX = (1 << 32) - 3
COIN_INDEX = (1 << 32) - 4
COIN_BYTES = 32
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
        if type(self.rounds) is not int or not MIN_ROUNDS <= self.rounds < COIN_INDEX:
            raise ValueError(f"the codec runs at least {MIN_ROUNDS} rounds: at one a word is completed for less than an encode")
        if type(self.masked) is not bool:
            raise ValueError("a codec is masked or not")

    @property
    def name(self) -> str:
        """What a registration binds: `p1024/k3/masked`, `p512/k2`."""
        return f"{self.field.name}/k{self.rounds}" + ("/masked" if self.masked else "")

    @property
    def max_blocks(self) -> int:
        """The most chunks a store under it holds: its words' indices fit a u64 (2^59 at p1024, 2^58 at p512)."""
        return WORD_INDICES // self.field.block_words

    @property
    def encode_squarings(self) -> int:
        """A chunk's encode in squarings: k roots a word, each the bit length of (p + 1)/4 less one (98,016 at
        p1024/k3/masked, 65,344 at p1024/k2/masked, 65,152 at p512/k2)."""
        return self.field.block_words * self.rounds * (((self.field.p + 1) // 4).bit_length() - 1)

    def to_json(self) -> dict:
        """Its parameters: the field `p = 2^n − c`, k, and the mask."""
        return {"c": self.field.c, "field": self.field.name, "masked": self.masked, "n": self.field.n,
                "rounds": self.rounds}


DEFAULT = Codec(P1024, 3, True)
FALLBACK = Codec(P1024, 2, True)

P2_POINT = (16448, 21065)
"""`w` and `c0` of PoUS's registered P2 point (`pous.schemes.p2.POINT`, which the suite holds this to)."""
P2_EXPANDER = "sha512-ctr-v4"
P2_LAYOUT = "p2-unit4/v0"
"""A P2 store's label layout: a receipt's vector is a unit of four consecutive labels in store order (`p2.py`)."""


@dataclass(frozen=True, slots=True)
class P2Codec:
    """PoUS's P2 (`verity.core.protocols.pous.schemes.p2`, `p2-16448/v4`) as a P2 registration names it
    (`p2.Registration`): the registered point's `w` and `c0`, keys by counter-mode SHA-512, and the store's
    segments.  A P2 store's labels are PoUS's 2,056-byte codewords, opened on PoUS's `vk` and never decoded by this
    protocol, and its receipts are of units under `P2_LAYOUT`.  Nothing of the Sloth codec above takes it, and
    `audit.Rebuild` refuses it: μ_rebuild over a P2 store waits on the squaring ceiling at P2's encode
    (`Accounting.Remote.Held.Assumptions.SquaringCeilingP2`, drafted)."""

    blocks_per_segment: int
    segments: int
    w: int = P2_POINT[0]
    c0: int = P2_POINT[1]
    expander: str = P2_EXPANDER

    def __post_init__(self) -> None:
        if (self.w, self.c0, self.expander) != (*P2_POINT, P2_EXPANDER):
            raise ValueError(f"a P2 store is at PoUS's registered point, w = {P2_POINT[0]} and c0 = {P2_POINT[1]}, "
                             f"keys by {P2_EXPANDER}")
        for name in ("blocks_per_segment", "segments"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"a P2 store's {name} is a positive integer")
        if self.blocks_per_segment > 1 << 16 or self.segments >= 1 << 32:
            raise ValueError("a P2 store's segments are at most 2^16 blocks (vk's u16 index), and fewer than 2^32")

    @property
    def name(self) -> str:
        return f"p2-{self.w}/v4"

    @property
    def blocks(self) -> int:
        """N, the store's 2,056-byte labels."""
        return self.blocks_per_segment * self.segments

    @property
    def block_encode_squarings(self) -> int:
        """A label's encode in squarings, two roots each the bit length of (p + 1)/4 less one: 32,890."""
        return 2 * ((((1 << self.w) - self.c0 + 1) // 4).bit_length() - 1)

    def to_json(self) -> dict:
        return {"blocks_per_segment": self.blocks_per_segment, "c0": self.c0, "expander": self.expander,
                "segments": self.segments, "w": self.w}


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
    """A codec seed's round constants `c_0 … c_(k−1)`, its tweak multiplier `τ`, masked, its mask `K`, and the seed,
    which a bound chunk's offset `b_j` is derived from."""

    codec: Codec
    c: tuple[int, ...]
    tau: int
    mask: int | None = None
    seed: bytes | None = None

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
    c = tuple(_element(f, seed, r) for r in range(codec.rounds))
    return Key(codec, c, _element(f, seed, TWEAK_INDEX) or 1, mask, seed)


def digest(payload: bytes) -> bytes:
    """`d_j`, a payload's digest: sha512/row/v2 of its 4,096 bytes, the `x_j` its payload leaf hides (`receipts.hidden`)."""
    if type(payload) is not bytes or len(payload) != BLOCK_BYTES:
        raise ValueError(f"a block is {BLOCK_BYTES} bytes")
    return sha512_row_v2_digest(payload, 8 * BLOCK_BYTES)


def binding(d: bytes, j: int, k: Key, coin: bytes | None = None) -> int:
    """`b_j`, the offset of chunk `j`'s bound key from the digest `d` of its payload: one SHA-512 output, `K_j =
    H(seed ‖ j ‖ H(m_j))` in the codec's framing, added to every round of the block's words; with `coin`, its append's
    coin `c_a`, `K_j = H(seed ‖ j ‖ H(m_j) ‖ c_a)` under a derivation index of its own."""
    if type(d) is not bytes or len(d) != DIGEST_BYTES:
        raise ValueError(f"a block's digest is {DIGEST_BYTES} bytes")
    if k.seed is None:
        raise ValueError("a bound key needs its codec seed")
    f, at = k.codec.field, _index(j, k.codec).to_bytes(8, "big")
    if coin is None:
        message = _message(f, k.seed, BIND_INDEX) + at + d
    elif type(coin) is not bytes or len(coin) != COIN_BYTES:
        raise ValueError(f"an append's coin is {COIN_BYTES} bytes")
    else:
        message = _message(f, k.seed, COIN_INDEX) + at + d + coin
    return int.from_bytes(_xof(message, DIGEST_BYTES), "little") % f.p


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


def encode_word(m: int, i: int, k: Key, b: int = 0) -> int:
    f = k.codec.field
    p = f.p
    if not 0 <= m < p:
        raise ValueError("a payload word is below p")
    t = (k.tau * i + b) % p
    x = m
    for r, c in enumerate(k.c):
        if r and k.mask is not None:
            x = k.sigma(x)
        x = rho((x + c + t) % p, f)
    return x


def decode_word(y: int, i: int, k: Key, b: int = 0) -> int:
    f = k.codec.field
    p = f.p
    if not 0 <= y < p:
        raise ValueError("an encoded word is canonical, below p")
    t = (k.tau * i + b) % p
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


def encode(payload: bytes, j: int, k: Key, *, bound: bool = False, coin: bytes | None = None) -> bytes:
    """`E_j`, chunk `j` of the store, from its payload: under the seed's key, or, `bound`, under the chunk's own key,
    for a chunk written after the codec seed, which keeps `digest(payload)` beside it, and with `coin` under its
    append's coin too."""
    if coin is not None and not bound:
        raise ValueError("an append's coin keys a bound block")
    f = k.codec.field
    i0 = f.block_words * _index(j, k.codec)
    words = _words(payload, f)
    b = binding(digest(payload), j, k, coin) if bound else 0
    return _block([encode_word(m, i0 + w, k, b) for w, m in enumerate(words)], f)


def decode(block: bytes, j: int, k: Key, d: bytes | None = None, coin: bytes | None = None) -> bytes | None:
    """Chunk `j`'s payload, under the seed's key, or under the bound key of the digest `d` kept beside it (and of its
    append's `coin`, if given).  None if a word of it is not canonical, or if `d` is given and the payload decoded
    under it does not have that digest: no payload encodes to it."""
    if coin is not None and d is None:
        raise ValueError("an append's coin keys a bound block, decoded under its digest")
    f = k.codec.field
    i0 = f.block_words * _index(j, k.codec)
    words = _words(block, f)
    b = 0 if d is None else binding(d, j, k, coin)
    if any(y >= f.p for y in words):
        return None
    payload = _block([decode_word(y, i0 + w, k, b) for w, y in enumerate(words)], f)
    return None if d is not None and digest(payload) != d else payload


def _index(j: int, codec: Codec) -> int:
    if type(j) is not int or not 0 <= j < codec.max_blocks:
        raise ValueError(f"a block index is below {codec.max_blocks}, so its words' indices fit a u64")
    return j
