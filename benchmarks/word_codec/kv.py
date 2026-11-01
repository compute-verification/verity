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
K          a masked key's (`key(..., mask=True)`): LE(XOF(DOMAIN ‖ frame ‖ seed ‖ u32be 2^32 − 2)[0 : S]), n bits, not
           reduced mod p
σ_K(x)     x ⊕ K if that is below p, else x: an involution on [0, p), applied to x before each round j ≥ 1 of a masked
           word's encode and after each inverted round j ≥ 1 of its decode
d          a bound token-layer's digest: sha512/row/v2 of its 4,096 bytes, SHA512(P(32768) ‖ data)
b          its offset: LE(XOF(DOMAIN ‖ frame ‖ seed ‖ u32be 2^32 − 3 ‖ u64be block ‖ d)[0 : 64]) mod p, block = slot ·
           LAYERS + layer, added to t(i) in every round of its words (0 under the seed's key)
~~~

The table is the store's codec (`verity.core.protocols.poub.codec`), whose field, key, ρ, rounds and bound
offset these are: `key` is its `derive` over a `Sloth` at the fields above and any k ≥ 1, which at p512 and p1024,
k ≥ 2, is `codec.key` of the codec a registration names, and `encode_word`, `decode_word` and `binding` are its own.
This module keeps the KV cache's layout and the kernel's root chain.

A KV token-layer is written after the codec seed, so it is encoded under its bound key (`encode_token_layer(...,
bound=True)`), with d kept beside it and given to `decode_token_layer`.  Under the seed's key alone its writer could
pick its encoding (all zeros, or another block's) and write the token-layer that decodes from it, holding nothing for
it; under the bound key a chosen encoding decodes to a token-layer with the digest it was decoded under only by a
search over SHA-512.  The bound key is the store codec's for block slot · LAYERS + layer.  `kv.cu` takes it as each
token-layer's offset b (`kv_encode_bound`, `kv_decode_bound`, and `benchmarks/vllm_encoded/paged.cu`'s paths), gated
against these vectors (`kv_bench.py`'s `gate_bound`); `kv_twin.py` takes the seed's key only.

The derivation is `benchmarks/codec_tensor/wide.py`'s under a domain of its own, at the fields
the codec's cost model chose (2^1536 − 3453 its operating point), whose words have no slack. A payload is any word below p. For p = 2^n − c
with c < 2^16 a payload at or above p has its top 16 bits all ones, a NaN in bf16 (and its top byte 0xFF, a NaN in
E4M3), so every word of finite values is one; `encode_word` refuses the rest.

Encoding a word costs k roots, each (p+1)/4 = (2^M − 1)·2^b + LOW: x_M = a^(2^M − 1) by M's bits from the top
(x_2j = x_j^(2^j)·x_j, x_(j+1) = x_j²·a), then b squarings with a multiply by a at LOW's set bits, b = 8 or the
fewest bits that hold (c − 1)/4 (`low_bits`), so n − 3 squarings and 16 to 20 multiplies (`root_multiplies`), then
the square test (one more). Decoding costs k squarings.

The KV cache's layout (`token_layer_words`): a token's keys and values in one layer, K then V, each `KV_HEADS` ×
`HEAD_DIM` bf16 (Llama-3.1-70B: 8 × 128, 4,096 bytes together), are the payloads of W = ⌈4096 / S⌉ consecutive words,
zero-padded (64 words at 2^512 − 569, 43 at 2^768 − 825, 32, 26 and 22 at 1,024, 1,280 and 1,536 bits), the words of
slot s's layer l at index (s · LAYERS + l) · W.
"""

from __future__ import annotations

import hashlib

from verity.core.protocols.poub.codec import (  # noqa: F401  the store codec's, which this module's users take
    BIND_INDEX, DOMAIN, P512, P1024, SEED_BYTES, Field, Key, Sloth, binding, decode_word, derive, encode_word, rho,
    rho_inv)

LOW_MIN = 8

# Llama-3.1-70B's attention: 80 layers, 8 KV heads of 128, bf16.
LAYERS = 80
KV_HEADS = 8
HEAD_DIM = 128
KV_ELEMENT_BYTES = 2
TOKEN_LAYER_BYTES = 2 * KV_HEADS * HEAD_DIM * KV_ELEMENT_BYTES
TOKEN_BYTES = LAYERS * TOKEN_LAYER_BYTES
# sha512/row/v2's first block for a row of TOKEN_LAYER_BYTES (`verity.ml.commitments.rowleaf`, held to it by the tests).
ROW_PREFIX = (b"verity/sha512-row/v2\0" + (8 * TOKEN_LAYER_BYTES).to_bytes(8, "big")).ljust(128, b"\0")

FIELDS = {f.name: f for f in (P512, Field("p768", 768, 825), P1024, Field("p1280", 1280, 1665),
                               Field("p1536", 1536, 3453))}


def field(name: str) -> Field:
    try:
        return FIELDS[name]
    except KeyError:
        raise ValueError(f"unknown field {name!r}; known: {sorted(FIELDS)}") from None


def low_bits(f: Field) -> int:
    return max(LOW_MIN, ((f.c - 1) // 4).bit_length())


def chain(f: Field) -> tuple[int, int]:
    """(M, LOW) with (p+1)/4 = (2^M − 1)·2^b + LOW, b = `low_bits`."""
    b = low_bits(f)
    return f.n - 2 - b, (1 << b) - (f.c - 1) // 4


def root_multiplies(f: Field) -> int:
    m, low = chain(f)
    return m.bit_length() - 1 + m.bit_count() - 1 + low.bit_count()


def check_fields() -> None:
    for f in FIELDS.values():
        m, low = chain(f)
        assert f.full_slot and f.p % 4 == 3 and f.p.bit_length() == f.n and f.c < 1 << 16
        assert 0 < low < 1 << low_bits(f) and ((1 << m) - 1) * (1 << low_bits(f)) + low == f.root_exp
        assert f.root_squarings == m - 1 + low_bits(f)


def key(name: str, seed: bytes, rounds: int, mask: bool = False) -> Key:
    return derive(seed, Sloth(field(name), rounds, bool(mask)))


def digest(data: bytes) -> bytes:
    """`d`, a token-layer's digest: sha512/row/v2 of its bytes."""
    if len(data) != TOKEN_LAYER_BYTES:
        raise ValueError(f"a token-layer is {TOKEN_LAYER_BYTES} bytes")
    return hashlib.sha512(ROW_PREFIX + bytes(data)).digest()


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


def encode_token_layer(data: bytes, slot: int, layer: int, k: Key, bound: bool = False) -> list[int]:
    """Slot `slot`'s layer `layer` encoded under the seed's key, or, `bound`, under its own, which keeps
    `digest(data)` beside it."""
    f = k.field
    block = slot * LAYERS + layer
    w0 = block * token_layer_words(f)
    words = payloads(data, f)
    b = binding(digest(data), block, k) if bound else 0
    return [encode_word(m, w0 + w, k, b) for w, m in enumerate(words)]


def decode_token_layer(words: list[int], slot: int, layer: int, k: Key, d: bytes | None = None) -> bytes | None:
    """The token-layer under the seed's key, or under the bound key of the digest `d` kept beside it: None if it does
    not have that digest."""
    f = k.field
    block = slot * LAYERS + layer
    w0 = block * token_layer_words(f)
    b = 0 if d is None else binding(d, block, k)
    s = f.word_bytes
    out = b"".join(decode_word(y, w0 + w, k, b).to_bytes(s, "little") for w, y in enumerate(words))[:TOKEN_LAYER_BYTES]
    return None if d is not None and digest(out) != d else out


def to_limbs(x: int, f: Field) -> list[int]:
    return [(x >> (32 * j)) & 0xFFFFFFFF for j in range(f.limbs)]


def from_limbs(limbs: list[int]) -> int:
    return sum(v << (32 * j) for j, v in enumerate(limbs))
